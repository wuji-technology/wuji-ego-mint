#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用学生推理引擎：ckpt → build_model → 分窗 forward → 拆相机/手部/置信度输出。

模型无关：走训练框架 model_train.build_model，对 model_train 注册的任意学生（lingbotmap/vggt/pi3…）
通用，具体模型由 config 选定。唯一外部依赖：复用 model_train（build_model + 内联 backbone _vendor/
lingbot_map）及其图像预处理 preprocess_frames —— 推理须与训练同构，故复用模型本体与预处理而非复刻。
产出与 GT 同 schema 的预测（pose_enc、hand 与可选 hand_confidence，见 inference.base）。

由各模型适配器（predictors/<model>）经 inference.registry 构造；自身不含 argparse 入口。
类名 StudentEngine；HandReprojPredictor 为兼容别名（旧调用点）。
"""
from __future__ import annotations

import glob
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from .acceleration import (
    COMPILE_MODES,
    FP8_MODES,
    apply_dynamic_fp8,
    compile_hotspots,
    normalize_optional_mode,
    prepare_allocator_for_compile,
)
from .base import FullSequenceTooLong, InferenceCancelled  # noqa: F401

# Viewer 通常在本模块之后才首次初始化 CUDA；可扩展 segment 能减少长序列不同尺寸临时张量造成的碎片。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_DIR = Path(__file__).resolve().parents[3]   # inference/ -> model_effect -> eval -> <repo>
MODEL_TRAIN = REPO_DIR / "model_train"


def _ensure_model_train_on_path() -> None:
    """挂 model_train 根 + 内联 _vendor，使 build_model / lingbot_map 可 import（与 train.py 一致）。"""
    for p in (str(MODEL_TRAIN), str(MODEL_TRAIN / "_vendor")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _abs(p):
    """相对路径按仓库根解析（与 train.py 的 _abs 同约定）。"""
    if p is None or os.path.isabs(str(p)):
        return p
    return str(REPO_DIR / p)


def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for exported confidence logits."""
    clipped = np.clip(logits, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


_MAX_HAND_OVERLAP = 8
HAND_WINDOW_MODES = ("hard", "blend", "smooth")
DEFAULT_HAND_WINDOW_MODE = "hard"
CAMERA_INFERENCE_MODES = ("chunked", "max_chunked", "streaming", "full")


def _postprocess_hands(result: dict, hand_mode: str,
                       ukf_params: dict | None = None) -> bool:
    """Apply the production camera-frame UKF+RTS smoother after window blending."""
    if hand_mode != "smooth" or result.get("hand") is None:
        return False
    logits = result.get("hand_presence_logits")
    confidence = result.get("hand_confidence")
    valid = ((np.asarray(logits) >= 0.0) if logits is not None else
             ((np.asarray(confidence) >= 0.5) if confidence is not None else None))
    from .hand_smoothing import smooth_hand_output
    result["hand"] = smooth_hand_output(result["hand"], valid, **(ukf_params or {}))
    return True


def _auto_full_max_frames(window: int, total_memory_bytes: int | None) -> int:
    """Choose a conservative practical full limit in whole training-window units."""
    gib = 0.0 if total_memory_bytes is None else total_memory_bytes / (1024 ** 3)
    multiplier = 3 if gib >= 80 else (2 if gib >= 40 else 1)
    return int(window) * multiplier


def discover_visible_cuda_devices() -> list[str]:
    """Return every CUDA device visible to this process as logical ``cuda:N`` names."""
    import torch

    cuda_count = torch.cuda.device_count()
    if not torch.cuda.is_available() or cuda_count <= 0:
        return []
    return [f"cuda:{index}" for index in range(cuda_count)]


def _resolve_devices(device=None, devices=None) -> list[str]:
    """Resolve legacy single-device and new explicit/auto multi-device settings."""
    if device is not None:
        return [str(device)]
    if devices == "auto":
        selected = discover_visible_cuda_devices()
        print(f"[predictor] 自动使用全部可见 GPU: {selected or '无（回退 CPU）'}", flush=True)
        return selected or ["cpu"]
    if isinstance(devices, str):
        selected = [item.strip() for item in devices.split(",") if item.strip()]
    elif devices is None:
        selected = []
    else:
        selected = [str(item) for item in devices]
    if selected:
        normalized = [
            item if item == "cpu" or item.startswith("cuda") else f"cuda:{item}"
            for item in selected
        ]
        return list(dict.fromkeys(normalized))
    import torch
    return ["cuda" if torch.cuda.is_available() else "cpu"]


def _hand_window_overlap(window: int) -> int:
    """Use 25% overlap up to eight frames; the trained 32-frame window gets overlap=8."""
    window = int(window)
    if window < 2:
        return 0
    return min(_MAX_HAND_OVERLAP, max(1, window // 4), window - 1)


def _window_bounds(num_frames: int, window: int, overlap: int,
                   full_tail: bool = True) -> list[tuple[int, int]]:
    """Build covering windows, optionally shifting the tail left to keep it full-sized."""
    num_frames, window = int(num_frames), int(window)
    if num_frames <= 0:
        return []
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if num_frames <= window:
        return [(0, num_frames)]

    overlap = max(0, min(int(overlap), window - 1))
    if not full_tail:
        bounds, start = [], 0
        while True:
            end = min(start + window, num_frames)
            bounds.append((start, end))
            if end >= num_frames:
                return bounds
            start = end - overlap

    stride = window - overlap
    last_start = num_frames - window
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + window) for start in starts]


def _window_blend_weights(bounds: list[tuple[int, int]], index: int) -> np.ndarray:
    """Return linear fade-in/out weights matching the actual neighboring overlaps."""
    start, end = bounds[index]
    weights = np.ones(end - start, dtype=np.float32)
    if index > 0:
        left_overlap = max(0, bounds[index - 1][1] - start)
        left_overlap = min(left_overlap, len(weights))
        if left_overlap:
            ramp = np.arange(1, left_overlap + 1, dtype=np.float32) / (left_overlap + 1)
            weights[:left_overlap] = np.minimum(weights[:left_overlap], ramp)
    if index + 1 < len(bounds):
        right_overlap = max(0, end - bounds[index + 1][0])
        right_overlap = min(right_overlap, len(weights))
        if right_overlap:
            ramp = np.arange(right_overlap, 0, -1, dtype=np.float32) / (right_overlap + 1)
            weights[-right_overlap:] = np.minimum(weights[-right_overlap:], ramp)
    return weights


def _find_weight_file(ckpt: str) -> str:
    """ckpt 给目录（accelerate save_state 的 step_* 目录）时定位权重文件；给文件则原样返回。"""
    if os.path.isfile(ckpt):
        return ckpt
    if os.path.isdir(ckpt):
        for pat in ("model.safetensors", "pytorch_model.bin",
                    "*.safetensors", "pytorch_model*.bin", "*.pt", "*.pth"):
            hits = sorted(glob.glob(os.path.join(ckpt, pat)))
            if hits:
                return hits[0]
        raise FileNotFoundError(
            f"{ckpt} 下未找到权重文件（model.safetensors / pytorch_model.bin / *.pt 等）")
    raise FileNotFoundError(f"ckpt 不存在: {ckpt}")


def _clip_len_from_ckpt(ckpt: str) -> int | None:
    """从 ckpt 路径反查其训练 config 的 data.clip_len，使推理片段长度 = 训练 clip 长度。

    ckpt → 上溯到 run 根（含 logs/record/config.yaml、config/config.yaml 或 wandb/ 的层），按优先级取 clip_len：
      1) <run_root>/logs/record/config.yaml —— 当前训练脚本写出的真实生效配置；
      2) <run_root>/config/config.yaml —— 旧 run 的解析后配置快照；
      3) wandb-metadata.json 的 args `--config <path>` 指向的 yaml 的当前内容（更旧 run 无快照时的回退）。
    任一步失败返回 None（上层再回退命令行 config / 16）。
    """
    import json

    def _clip_len_of(yaml_path: str) -> int | None:
        with open(yaml_path) as f:
            tcfg = yaml.safe_load(f) or {}
        cl = (tcfg.get("data") or {}).get("clip_len")
        return int(cl) if cl is not None else None

    try:
        run_root = Path(ckpt).resolve()
        for _ in range(4):                     # step_*/ 或权重文件起步，最多上溯几层找 run 根
            if ((run_root / "logs" / "record" / "config.yaml").is_file()
                    or (run_root / "config" / "config.yaml").is_file()
                    or (run_root / "wandb").is_dir()):
                break
            run_root = run_root.parent
        else:
            return None
        # 1/2) 优先：训练时落盘的解析后配置快照
        for snap in (run_root / "logs" / "record" / "config.yaml",
                     run_root / "config" / "config.yaml"):
            if snap.is_file():
                cl = _clip_len_of(str(snap))
                if cl is not None:
                    return cl
        # 3) 回退：wandb-metadata 记录的 --config 路径（读其当前内容）
        metas = sorted(glob.glob(str(run_root / "wandb" / "*" / "files" / "wandb-metadata.json")))
        if metas:
            meta = json.load(open(metas[-1]))
            args = meta.get("args", [])
            if "--config" in args:
                cfg_path = args[args.index("--config") + 1]
                cfg_path = cfg_path if os.path.isabs(cfg_path) else str(REPO_DIR / cfg_path)
                return _clip_len_of(cfg_path)
        return None
    except Exception:
        return None


class StudentEngine:
    """加载训练 ckpt，对帧序列分窗前向，产出与 GT 同 schema 的双手 + 相机预测。"""

    def __init__(self, config_path: str, ckpt: str | None = None,
                 device: str | None = None, window: int | None = None,
                 devices=None, full_max_frames: int | None = None,
                 compile_mode: str | None = None, fp8_mode: str | None = None):
        self.compile_mode = normalize_optional_mode(
            compile_mode, choices=COMPILE_MODES, name="compile_mode"
        )
        self.fp8_mode = normalize_optional_mode(
            fp8_mode, choices=FP8_MODES, name="fp8_mode"
        )
        allocator_changed = prepare_allocator_for_compile(self.compile_mode)
        _ensure_model_train_on_path()
        import torch
        from models import build_model                       # noqa: E402  (model_train)
        from utils.weight_loader import load_state_dict_file  # noqa: E402

        if allocator_changed:
            print(
                "[predictor] torch.compile CUDA Graph 已关闭 expandable_segments allocator",
                flush=True,
            )

        with open(_abs(config_path)) as f:
            cfg = yaml.safe_load(f)
        mcfg = dict(cfg["model"])
        self._raw_mcfg = dict(cfg["model"])   # 未改动的 model 段，供换 ckpt 时判架构是否一致（可就地重载）
        # size_hw 取训练 data 段，保证预处理 resize 与训练一致（默认 [378,518]）。
        self.size_hw = tuple(cfg.get("data", {}).get("size_hw", (378, 518)))

        # window=None（默认）→ 自动令推理片段长度 = 训练 clip_len，保证与训练前向一致：
        # 优先该 ckpt 自带训练 config 的 clip_len；拿不到回退命令行 config 的 clip_len，最后 16。
        # 显式传 window 则原样尊重（覆盖自动推断）。
        if window is None:
            cl = _clip_len_from_ckpt(ckpt) if ckpt else None
            src = "ckpt 训练 config" if cl is not None else None
            if cl is None:
                cl = cfg.get("data", {}).get("clip_len")
                src = "命令行 config" if cl is not None else None
            if cl is None:
                cl, src = 16, "默认回退"
            window = int(cl)
            print(f"[predictor] window={window}（自动=训练 clip_len，来源：{src}）")
        else:
            print(f"[predictor] window={int(window)}（显式指定）")
        self.window = int(window)

        # 给了 ckpt：跳过 config 的 pretrained（避免重复加载 4.6G 骨干，ckpt 已含 backbone.*）。
        # 没给 ckpt（smoke）：用 config 的 pretrained 骨干 + 随机 hand_head，仅验证能跑通。
        if ckpt:
            mcfg["pretrained"] = None
            mcfg["_ckpt_provided"] = True   # 跳过 config 预训练；权重随后由 load_state_dict 整体载入
        elif mcfg.get("pretrained"):
            mcfg["pretrained"] = _abs(mcfg["pretrained"])

        requested_devices = _resolve_devices(device=device, devices=devices)
        resolved_devices = [torch.device(item) for item in requested_devices]
        self.device = resolved_devices[0]
        cuda_memories = [
            torch.cuda.get_device_properties(item).total_memory
            for item in resolved_devices
            if item.type == "cuda"
        ]
        limiting_memory = min(cuda_memories) if cuda_memories else None
        if full_max_frames is None:
            self.full_max_frames = _auto_full_max_frames(self.window, limiting_memory)
            full_limit_source = "按所选 GPU 最小显存自动"
        else:
            self.full_max_frames = int(full_max_frames)
            if self.full_max_frames <= 0 or self.full_max_frames % self.window:
                raise ValueError(
                    f"full_max_frames 必须是训练窗 {self.window} 的正整数倍，"
                    f"得到 {self.full_max_frames}"
                )
            full_limit_source = "显式指定"
        print(
            f"[predictor] exact full 安全上限={self.full_max_frames} 帧，"
            f"max_chunked={self.full_max_frames} 帧/窗（{full_limit_source}）"
        )
        model = build_model(mcfg)
        sd = None
        presence_missing = False
        if ckpt:
            wf = _find_weight_file(ckpt)
            sd = load_state_dict_file(wf)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            presence_missing = any(key.startswith("hand_presence_head.") for key in missing)
            if presence_missing:
                model.enable_hand_presence = False
                print("[predictor] ckpt 不含 hand_presence_head 权重，已禁用随机置信度输出")
            print(f"[predictor] 载入 ckpt {os.path.basename(wf)}: "
                  f"{len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
        self.model = model.to(self.device).eval()
        self.models = [self.model]
        self.devices = [self.device]

        # 每卡常驻一个副本；只复制模型 forward，窗口输出仍回 CPU 后按原顺序缝合。
        # 自动模式下若某张副卡在选卡后被抢占而 OOM，只跳过该副卡，不影响主卡可用性。
        for replica_device in resolved_devices[1:]:
            replica = None
            try:
                replica = build_model(mcfg)
                if sd is not None:
                    replica.load_state_dict(sd, strict=False)
                    if presence_missing:
                        replica.enable_hand_presence = False
                else:
                    # smoke 模式没有 checkpoint，也必须复制相同随机头，避免各卡输出不一致。
                    replica.load_state_dict(self.model.state_dict(), strict=True)
                replica = replica.to(replica_device).eval()
            except torch.cuda.OutOfMemoryError:
                del replica
                if replica_device.type == "cuda":
                    with torch.cuda.device(replica_device):
                        torch.cuda.empty_cache()
                print(f"[predictor] {replica_device} 加载模型副本时显存不足，已跳过该卡", flush=True)
                continue
            self.models.append(replica)
            self.devices.append(replica_device)

        del sd
        self._fp8_conversions = []
        if self.fp8_mode == "dynamic":
            for replica, replica_device in zip(self.models, self.devices):
                conversion = apply_dynamic_fp8(replica, replica_device)
                self._fp8_conversions.append(conversion)
            print(
                f"[predictor] FP8 dynamic 已启用: 每副本量化 "
                f"{len(self._fp8_conversions[0].module_names)} 个 aggregator Linear, "
                f"torchao={self._fp8_conversions[0].torchao_version}",
                flush=True,
            )

        self._compiled_module_names = []
        if self.compile_mode is not None:
            # RoPE modules cache cos/sin tensors. They must be created eagerly before
            # compiled blocks capture graphs, otherwise the cache owns graph outputs
            # that are overwritten on the next replay.
            for replica, replica_device in zip(self.models, self.devices):
                if replica_device.type != "cuda":
                    continue
                backbone = getattr(replica, "backbone", None)
                aggregator = getattr(backbone, "aggregator", None)
                kv_cache = getattr(aggregator, "kv_cache", None)
                cache_state = None
                if getattr(aggregator, "use_sdpa", False) and isinstance(kv_cache, dict):
                    backbone.clean_kv_cache()
                    cache_state = kv_cache
                    aggregator.kv_cache = None
                try:
                    warm_images = torch.zeros(
                        (1, self.window, 3, *self.size_hw),
                        dtype=torch.float32,
                        device=replica_device,
                    )
                    with torch.inference_mode(), torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16):
                        replica({"images": warm_images})
                    torch.cuda.synchronize(replica_device)
                    del warm_images
                finally:
                    if cache_state is not None:
                        aggregator.kv_cache = cache_state
                        backbone.clean_kv_cache()
            self._compiled_module_names = [
                compile_hotspots(replica, self.compile_mode) for replica in self.models
            ]
            print(
                f"[predictor] torch.compile 已启用: mode={self.compile_mode}, "
                f"每副本 {len(self._compiled_module_names[0])} 个固定形状热点模块，"
                "dynamic=False（首次对应 shape 前向会编译/CUDA Graph 捕获）",
                flush=True,
            )
        self._model_locks = [threading.Lock() for _ in self.models]
        self._forward_pool = (
            ThreadPoolExecutor(max_workers=len(self.models), thread_name_prefix="gpu-forward")
            if len(self.models) > 1 else None
        )
        print(f"[predictor] 推理设备({len(self.devices)}): "
              f"{', '.join(map(str, self.devices))}", flush=True)
        self.has_hand = bool(getattr(self.model, "enable_hand", False))
        self.has_hand_presence = bool(
            getattr(self.model, "enable_hand_presence", False)
        )

    @property
    def parallel_device_count(self) -> int:
        return len(getattr(self, "models", [self.model]))

    @property
    def parallel_device_names(self) -> list[str]:
        return [str(item) for item in getattr(self, "devices", [self.device])]

    @property
    def supports_weight_reload(self) -> bool:
        return self.fp8_mode is None and self.compile_mode is None

    @property
    def acceleration_metadata(self) -> dict:
        return {
            "compile_mode": self.compile_mode or "off",
            "fp8_mode": self.fp8_mode or "off",
        }

    def _effective_window_batch_size(self, requested_batch: int) -> int:
        """Keep compiled replicas on their captured local batch-one shape."""
        requested_batch = max(1, int(requested_batch))
        if self.compile_mode is None:
            return requested_batch
        return min(requested_batch, self.parallel_device_count)

    def warmup_acceleration(self, window_batch_size: int = 1, passes: int = 2) -> dict:
        """Compile/capture the fixed full-window shape before measured inference."""
        import time
        import torch

        requested_batch = max(1, int(window_batch_size))
        batch = self._effective_window_batch_size(requested_batch)
        passes = max(1, int(passes))
        frame_count = self.window + (batch - 1) * max(1, self.window - 1)
        frames = torch.zeros(
            (frame_count, 3, *self.size_hw), dtype=torch.float32, device="cpu"
        )
        started = time.perf_counter()
        last_timings = {}
        for _ in range(passes):
            output = self.predict(
                frames,
                cam_mode="chunked",
                window_batch_size=batch,
                hand_mode="hard",
                preprocessed=True,
            )
            last_timings = dict(output.get("_timings") or {})
        for replica_device in self.devices:
            if replica_device.type == "cuda":
                torch.cuda.synchronize(replica_device)
        return {
            "passes": passes,
            "frames": frame_count,
            "requested_window_batch_size": requested_batch,
            "window_batch_size": batch,
            "seconds": time.perf_counter() - started,
            "last_timings": last_timings,
        }

    def reload(self, ckpt: str):
        """就地把新 ckpt 权重灌进所有常驻 GPU 副本（架构不变时用）。

        返回 (missing, unexpected)：missing 非空表示新 ckpt 未覆盖模型全部权重，就地重载会残留旧 run
        的那部分（不安全），调用方应据此改走整模型重建。"""
        if not self.supports_weight_reload:
            raise RuntimeError(
                "FP8/torch.compile 引擎切换 checkpoint 必须重建模型，"
                "以重新量化权重并重建编译图"
            )
        from utils.weight_loader import load_state_dict_file   # noqa: E402
        wf = _find_weight_file(ckpt)
        models = getattr(self, "models", [self.model])
        # 单卡保留直接读到 GPU 的快速路径；多卡只读一次 CPU 权重，再依次分发。
        sd = load_state_dict_file(wf, device=str(self.device) if len(models) == 1 else "cpu")
        missing = unexpected = None
        for replica_index, model in enumerate(models):
            replica_missing, replica_unexpected = model.load_state_dict(sd, strict=False)
            if replica_index == 0:
                missing, unexpected = replica_missing, replica_unexpected
            model.enable_hand_presence = bool(
                hasattr(model, "hand_presence_head")
                and not any(key.startswith("hand_presence_head.") for key in replica_missing)
            )
        presence_missing = any(
            key.startswith("hand_presence_head.") for key in missing
        )
        if presence_missing:
            print("[predictor] ckpt 不含 hand_presence_head 权重，已禁用随机置信度输出")
        self.has_hand = bool(getattr(self.model, "enable_hand", False))
        self.has_hand_presence = bool(
            getattr(self.model, "enable_hand_presence", False)
        )
        print(f"[predictor] 就地重载 {os.path.basename(wf)} 到 {len(models)} 个设备: "
              f"{len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        return missing, unexpected

    def _forward_window_batch(
        self,
        x,
        bounds: list[tuple[int, int]],
        disable_persistent_kv_cache: bool = False,
    ) -> tuple[dict, float]:
        """Run equal-length windows across persistent replicas and restore batch order."""
        import time
        import torch

        models = getattr(self, "models", [self.model])
        devices = getattr(self, "devices", [self.device])
        locks = getattr(self, "_model_locks", None)
        if locks is None:
            locks = [threading.Lock() for _ in models]
        worker_count = min(len(models), len(bounds))
        shard_sizes = [len(bounds) // worker_count] * worker_count
        for index in range(len(bounds) % worker_count):
            shard_sizes[index] += 1

        shards = []
        offset = 0
        for model_index, shard_size in enumerate(shard_sizes):
            indices = list(range(offset, offset + shard_size))
            clips_cpu = torch.stack([
                x[bounds[index][0]:bounds[index][1]] for index in indices
            ])
            shards.append((model_index, indices, clips_cpu))
            offset += shard_size

        def _run(shard):
            model_index, indices, clips_cpu = shard
            model, replica_device = models[model_index], devices[model_index]
            cuda_replica = replica_device.type == "cuda"
            with locks[model_index], torch.inference_mode():
                cache_state = None
                backbone = getattr(model, "backbone", None)
                aggregator = getattr(backbone, "aggregator", None)
                kv_cache = getattr(aggregator, "kv_cache", None)
                if (disable_persistent_kv_cache
                        and getattr(aggregator, "use_sdpa", False)
                        and isinstance(kv_cache, dict)):
                    # Batch SDPA with a fresh cache is numerically identical to cache=None, but the
                    # former retains full K/V for every global block until forward returns.
                    backbone.clean_kv_cache()
                    cache_state = (aggregator, kv_cache)
                    aggregator.kv_cache = None
                try:
                    clips = clips_cpu.to(replica_device, non_blocking=True)
                    if cuda_replica:
                        torch.cuda.synchronize(replica_device)
                    with torch.autocast(
                            device_type="cuda", dtype=torch.bfloat16, enabled=cuda_replica):
                        if self.compile_mode is not None:
                            torch.compiler.cudagraph_mark_step_begin()
                        output = model({"images": clips})
                    if cuda_replica:
                        torch.cuda.synchronize(replica_device)
                    cpu_output = {
                        key: value.float().cpu().numpy()
                        for key, value in output.items()
                        if key in {
                            "pose_enc", "hand", "hand_presence_logits",
                            "_hand_refine_initial",
                        }
                    }
                finally:
                    if cache_state is not None:
                        cache_aggregator, original_cache = cache_state
                        cache_aggregator.kv_cache = original_cache
                        backbone.clean_kv_cache()
            return indices, cpu_output

        started = time.perf_counter()
        if len(shards) == 1:
            results = [_run(shards[0])]
        else:
            pool = getattr(self, "_forward_pool", None)
            owns_pool = pool is None
            pool = pool or ThreadPoolExecutor(
                max_workers=len(shards), thread_name_prefix="gpu-forward"
            )
            futures = [pool.submit(_run, shard) for shard in shards]
            results, errors = [], []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:  # wait for every GPU before retrying an OOM batch
                    errors.append(exc)
            if owns_pool:
                pool.shutdown(wait=True)
            if errors:
                raise errors[0]
        elapsed = time.perf_counter() - started

        keys = set().union(*(output.keys() for _indices, output in results))
        ordered = {}
        for key in keys:
            values = [None] * len(bounds)
            for indices, output in results:
                if key not in output:
                    continue
                for local_index, batch_index in enumerate(indices):
                    values[batch_index] = output[key][local_index]
            if all(value is not None for value in values):
                ordered[key] = np.stack(values)
        return ordered, elapsed

    def predict(self, frames_uint8: np.ndarray, on_step=None, cancel_check=None,
                cam_mode: str = "chunked", window_batch_size: int | None = None,
                hand_mode: str = DEFAULT_HAND_WINDOW_MODE,
                preprocessed: bool = False) -> dict:
        """frames_uint8: [N,H0,W0,3] uint8 RGB → 相机、手部和可选置信度 numpy 输出。

        分窗（self.window 帧/窗）前向；手部拼窗由 hand_mode 控制：
          · "hard"（默认）：保留原始 1 帧重叠硬切，末窗允许短于训练窗。
          · "blend"：最多 8 帧重叠并线性渐入/渐出；N>=window 时末窗保持完整。
          · "smooth"：先按 blend 拼窗，再对相机系 MANO 做速度自适应 UKF+RTS 双向平滑。
        on_step(done, total)：可选回调，每窗前向后上报进度（网页端进度条用）。
        cancel_check()->bool：可选，每窗**开头**查一次，返回 True 则抛 InferenceCancelled 中断前向
        （网页端「停止」按钮用；中断粒度=窗边界，未算的窗不再前向）。

        cam_mode：相机推理策略。
          · "chunked"（默认）：分窗前向 + 相邻窗 SE(3) 链式拼接（现状；窗间误差累积）。
          · "streaming"：相机走 backbone 原生流式（整段因果 + KV cache 一次跑完、不拼接），消除窗间
            累积漂移；手部参数/存在性仍分窗（原生流式不含新头）。相机阶段一次跑完，无逐帧进度、
            不可中途取消，进度/取消由随后的输出头分窗体现。
          · "full"：安全上限内整段只做一次普通模型前向，相机和手部均不分窗、不拼接；超过上限直接
            抛 FullSequenceTooLong，且单次 GPU 前向运行期间只能在前后响应取消。
          · "max_chunked"：使用 full 安全上限作为窗长做分窗推理和拼接。
        window_batch_size：chunked 模式一次前向多少个互相独立的窗口；None 时取已加载
            GPU 副本数，使所有卡各处理一窗。不会让窗口间产生 attention；CUDA OOM 时自动减半重试。
            启用 torch.compile 时会自动限制为 GPU 副本数，保持每卡 local batch=1 的固定捕获形状。
        preprocessed：输入已经是 CPU [N,3,H,W] float[0,1] 时跳过预处理；Benchmark
            用它分块解码超长序列，避免构造整段原分辨率 float32 临时张量。
        """
        import time
        import torch
        from data.transforms import preprocess_frames   # noqa: E402  (model_train，与训练同预处理)

        from .hand_smoothing import parse_hand_smoothing_mode
        requested_hand_mode = str(hand_mode)
        try:
            hand_mode, ukf_params = parse_hand_smoothing_mode(requested_hand_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"未知 hand_mode: {requested_hand_mode!r}，可选值: "
                f"{', '.join(HAND_WINDOW_MODES)}；smooth 可携带 q/r/beta"
            ) from exc
        if cam_mode not in CAMERA_INFERENCE_MODES:
            raise ValueError(
                f"未知 cam_mode: {cam_mode!r}，可选值: {', '.join(CAMERA_INFERENCE_MODES)}"
            )
        input_frames = int(frames_uint8.shape[0])
        if cam_mode == "full" and input_frames > self.full_max_frames:
            raise FullSequenceTooLong(input_frames, self.full_max_frames)

        predict_started = time.perf_counter()
        preprocess_started = time.perf_counter()
        if preprocessed:
            if (not torch.is_tensor(frames_uint8) or frames_uint8.ndim != 4
                    or frames_uint8.shape[1] != 3
                    or tuple(frames_uint8.shape[2:]) != tuple(self.size_hw)
                    or not frames_uint8.is_floating_point()):
                raise ValueError(
                    "preprocessed 输入须为 [N,3,H,W] float，且 H/W 与模型 size_hw 一致"
                )
            x = frames_uint8.contiguous()
        else:
            x = preprocess_frames(frames_uint8, self.size_hw)   # [N,3,H,W] float[0,1]
        preprocess_s = time.perf_counter() - preprocess_started
        N = x.shape[0]
        cuda = self.device.type == "cuda"
        W = self.full_max_frames if cam_mode == "max_chunked" else self.window

        # 整段模式严格执行一次普通 forward；不做分窗、拼接、融合或 OOM 降级。
        if cam_mode == "full":
            if N <= 0:
                raise ValueError("full 推理至少需要 1 帧")
            if cancel_check is not None and cancel_check():
                raise InferenceCancelled("整段推理开始前被取消")
            print(f"[predictor] 整段普通前向开始：{N} 帧（单卡、无分窗拼接）", flush=True)
            try:
                out, t_fwd = self._forward_window_batch(
                    x,
                    [(0, N)],
                    disable_persistent_kv_cache=True,
                )
            except torch.cuda.OutOfMemoryError:
                print(
                    f"[predictor] 整段普通前向 {N} 帧显存不足；请减少帧数或改用 chunked",
                    flush=True,
                )
                raise
            if cancel_check is not None and cancel_check():
                raise InferenceCancelled("整段推理在单次前向完成后被取消")

            res = {"pose_enc": out["pose_enc"][0].astype(np.float32)}
            if "hand" in out:
                res["hand"] = out["hand"][0].astype(np.float32)
            if "_hand_refine_initial" in out:
                res["_hand_refine_initial"] = out["_hand_refine_initial"][0].astype(
                    np.float32)
            if "hand_presence_logits" in out:
                logits = out["hand_presence_logits"][0].astype(np.float32)
                res["hand_presence_logits"] = logits
                res["hand_confidence"] = _sigmoid_np(logits)
            hand_smoothed = _postprocess_hands(res, hand_mode, ukf_params)
            if on_step is not None:
                try:
                    on_step(1, 1)
                except Exception:  # noqa: BLE001
                    pass
            print(
                f"[predictor] 整段普通前向完成：{N} 帧，{t_fwd:.2f}s，"
                f"{t_fwd / N * 1000:.1f} ms/帧",
                flush=True,
            )
            res["_timings"] = {
                "preprocess_s": preprocess_s,
                "forward_s": t_fwd,
                "total_s": time.perf_counter() - predict_started,
                "inference_mode": "full",
                "window_batch_size": 1,
                "forward_batches": 1,
                "windows": 1,
                "devices": [str(self.device)],
                "window_overlap": 0,
                "hand_mode": hand_mode,
                "ukf_params": ukf_params if hand_smoothed else None,
                "hand_postprocess": "ukf_cam_rts" if hand_smoothed else "none",
                "persistent_kv_cache": False,
                "full_max_frames": self.full_max_frames,
                **self.acceleration_metadata,
            }
            return res

        # ── cam_mode="streaming":相机走 backbone 原生流式(整段因果+KV cache,不分窗拼接),
        # 消除窗间 SE(3) 链式累积漂移;手部参数/存在性输出头仍分窗。 ──
        if cam_mode == "streaming":
            _ensure_model_train_on_path()
            from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
            from lingbot_map.utils.rotation import mat_to_quat
            if cancel_check is not None and cancel_check():
                raise InferenceCancelled("流式推理开始前被取消")
            # 相机:整段一次流式。num_scale_frames=训练窗长,使首段双向上下文≈训练所见;
            # output_device=cpu 逐帧 offload,长视频不 OOM(depth 已关,轻量)。
            print(f"[predictor] 流式相机推理开始:整段 {N} 帧逐帧因果前向(无逐帧进度回调,"
                  f"通常比分窗慢数倍,请耐心等)...", flush=True)
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cuda):
                sout = self.model.backbone.inference_streaming(
                    x.unsqueeze(0).to(self.device),
                    num_scale_frames=min(W, N),
                    output_device=torch.device("cpu"))
            if cuda:
                torch.cuda.synchronize()
            pe = sout["pose_enc"][0].float().numpy()            # [N,9] 流式单一全局系
            print(f"[predictor] 流式相机 forward {N} 帧,{time.perf_counter() - t0:.2f}s "
                  f"({(time.perf_counter() - t0) / max(N, 1) * 1000:.1f} ms/帧)", flush=True)
            # 重锚帧 0=identity(E'_i = E_i·E_0⁻¹),与训练 dataloader 及 chunked 输出同参考系。
            extr = pose_encoding_to_extri_intri(
                torch.from_numpy(pe)[None], build_intrinsics=False)[0][0].numpy()   # [N,3,4]
            E = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
            E[:, :3, :4] = extr
            E_rb = E @ np.linalg.inv(E[0])                      # 帧0 → identity
            R_rb, T_rb = E_rb[:, :3, :3], E_rb[:, :3, 3]
            q_rb = mat_to_quat(torch.from_numpy(R_rb).float()).numpy()   # xyzw(与 pose_enc 同约定)
            pose = np.concatenate(
                [T_rb.astype(np.float32), q_rb.astype(np.float32), pe[:, 7:]], axis=-1)  # FoV 不变
            res = {"pose_enc": pose.astype(np.float32)}
            # 两个手部输出头仍分窗前向；hard 保留原始硬切，blend 才做线性融合。
            if self.has_hand or self.has_hand_presence:
                blend_hands = hand_mode in {"blend", "smooth"}
                overlap = _hand_window_overlap(W) if blend_hands else (1 if W >= 2 else 0)
                bounds = _window_bounds(N, W, overlap, full_tail=blend_hands)
                hand_chunks, initial_chunks, presence_logit_chunks = [], [], []
                hand_sum = hand_weight = None
                initial_sum = initial_weight = None
                presence_sum = presence_weight = None
                with torch.inference_mode():
                    for wi, (s, e) in enumerate(bounds):
                        if cancel_check is not None and cancel_check():
                            raise InferenceCancelled(f"流式(手部)在第 {wi + 1}/{len(bounds)} 窗被取消")
                        clip = x[s:e].unsqueeze(0).to(self.device)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cuda):
                            out = self.model({"images": clip})
                        j0 = 0 if (wi == 0 or not overlap) else 1
                        blend = _window_blend_weights(bounds, wi) if blend_hands else None
                        if "hand" in out:
                            values = out["hand"][0].float().cpu().numpy()
                            if blend_hands:
                                if hand_sum is None:
                                    hand_sum = np.zeros((N, values.shape[-1]), dtype=np.float32)
                                    hand_weight = np.zeros((N, 1), dtype=np.float32)
                                hand_sum[s:e] += values * blend[:, None]
                                hand_weight[s:e] += blend[:, None]
                            else:
                                hand_chunks.append(values[j0:])
                        if "_hand_refine_initial" in out:
                            values = out["_hand_refine_initial"][0].float().cpu().numpy()
                            if blend_hands:
                                if initial_sum is None:
                                    initial_sum = np.zeros((N, values.shape[-1]), dtype=np.float32)
                                    initial_weight = np.zeros((N, 1), dtype=np.float32)
                                initial_sum[s:e] += values * blend[:, None]
                                initial_weight[s:e] += blend[:, None]
                            else:
                                initial_chunks.append(values[j0:])
                        if "hand_presence_logits" in out:
                            values = out["hand_presence_logits"][0].float().cpu().numpy()
                            if blend_hands:
                                if presence_sum is None:
                                    presence_sum = np.zeros((N, values.shape[-1]), dtype=np.float32)
                                    presence_weight = np.zeros((N, 1), dtype=np.float32)
                                presence_sum[s:e] += values * blend[:, None]
                                presence_weight[s:e] += blend[:, None]
                            else:
                                presence_logit_chunks.append(values[j0:])
                        if on_step is not None:
                            try:
                                on_step(wi + 1, len(bounds))
                            except Exception:  # noqa: BLE001
                                pass
                if hand_chunks:
                    res["hand"] = np.concatenate(hand_chunks, axis=0).astype(np.float32)
                elif hand_sum is not None:
                    res["hand"] = (hand_sum / np.maximum(hand_weight, 1e-8)).astype(np.float32)
                if initial_chunks:
                    res["_hand_refine_initial"] = np.concatenate(
                        initial_chunks, axis=0).astype(np.float32)
                elif initial_sum is not None:
                    res["_hand_refine_initial"] = (
                        initial_sum / np.maximum(initial_weight, 1e-8)).astype(np.float32)
                if presence_logit_chunks:
                    logits = np.concatenate(presence_logit_chunks, axis=0).astype(np.float32)
                    res["hand_presence_logits"] = logits
                    res["hand_confidence"] = _sigmoid_np(logits)
                elif presence_sum is not None:
                    logits = (presence_sum / np.maximum(presence_weight, 1e-8)).astype(np.float32)
                    res["hand_presence_logits"] = logits
                    res["hand_confidence"] = _sigmoid_np(logits)
            hand_smoothed = _postprocess_hands(res, hand_mode, ukf_params)
            res["_timings"] = {
                "preprocess_s": preprocess_s,
                "forward_s": time.perf_counter() - t0,
                "total_s": time.perf_counter() - predict_started,
                "window_batch_size": 1,
                "window_overlap": overlap if (self.has_hand or self.has_hand_presence) else 0,
                "hand_mode": hand_mode,
                "ukf_params": ukf_params if hand_smoothed else None,
                "hand_postprocess": "ukf_cam_rts" if hand_smoothed else "none",
                **self.acceleration_metadata,
            }
            return res

        # blend/smooth 使用多帧重叠和完整末窗；hard 保留原始硬切。
        blend_hands = hand_mode in {"blend", "smooth"} and (self.has_hand or self.has_hand_presence)
        overlap = (_hand_window_overlap(W) if blend_hands else (1 if W >= 2 else 0))
        win_bounds = _window_bounds(N, W, overlap, full_tail=blend_hands)
        nwin = len(win_bounds)

        _ensure_model_train_on_path()
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
        from lingbot_map.utils.rotation import mat_to_quat

        pose_out = np.empty((N, 9), dtype=np.float32)
        global_extr = np.zeros((N, 4, 4), dtype=np.float64)
        next_pose_frame = 0
        hand_sum = hand_weight = None
        initial_sum = initial_weight = None
        presence_sum = presence_weight = None
        hand_chunks, initial_chunks, presence_logit_chunks = [], [], []
        t_fwd = 0.0   # 纯前向累计耗时（不含 preprocess）
        requested_batch = max(
            1,
            self.parallel_device_count if window_batch_size is None else int(window_batch_size),
        )
        effective_batch = self._effective_window_batch_size(requested_batch)
        if effective_batch != requested_batch:
            print(
                f"[predictor] torch.compile 为保持每卡 local batch=1，窗口 batch "
                f"从 {requested_batch} 自动限制为 {effective_batch} "
                f"（GPU 副本数={self.parallel_device_count}）",
                flush=True,
            )
        forward_batches = 0
        with torch.inference_mode():
            wi = 0
            while wi < nwin:
                if cancel_check is not None and cancel_check():
                    raise InferenceCancelled(f"推理在第 {wi + 1}/{nwin} 窗被取消")

                # 最后一个窗口可能短于 W；不同长度不能 stack，先把同长度窗口组成 batch。
                batch_end = min(wi + effective_batch, nwin)
                first_len = win_bounds[wi][1] - win_bounds[wi][0]
                while batch_end > wi + 1:
                    last_s, last_e = win_bounds[batch_end - 1]
                    if last_e - last_s == first_len:
                        break
                    batch_end -= 1
                batch_bounds = win_bounds[wi:batch_end]
                try:
                    out, dt = self._forward_window_batch(
                        x,
                        batch_bounds,
                        # A fresh batch cache is numerically identical to no cache and only
                        # retains/clones K/V tensors that cannot be reused by the next window.
                        disable_persistent_kv_cache=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    if len(batch_bounds) <= 1:
                        raise
                    effective_batch = max(1, (len(batch_bounds) + 1) // 2)
                    for replica_device in getattr(self, "devices", [self.device]):
                        if replica_device.type == "cuda":
                            with torch.cuda.device(replica_device):
                                torch.cuda.empty_cache()
                    print(
                        f"\n[predictor] 窗口 batch={len(batch_bounds)} 显存不足，"
                        f"自动降到 {effective_batch} 后重试",
                        flush=True,
                    )
                    continue
                t_fwd += dt
                forward_batches += 1
                pose_batch = out["pose_enc"]
                hand_batch = out.get("hand")
                initial_batch = out.get("_hand_refine_initial")
                presence_batch = out.get("hand_presence_logits")

                for batch_index, (s, e) in enumerate(batch_bounds):
                    window_index = wi + batch_index
                    pe_l = pose_batch[batch_index]                       # [L,9] 窗首帧系
                    L = e - s
                    if overlap:
                        # 局部 w2c 外参 [L,4,4]。先按窗首帧归一，再顺序复合进全局。
                        extr_l = pose_encoding_to_extri_intri(
                            torch.from_numpy(pe_l)[None], build_intrinsics=False
                        )[0][0].numpy()
                        E_l = np.tile(np.eye(4, dtype=np.float64), (L, 1, 1))
                        E_l[:, :3, :4] = extr_l
                        E_rel = E_l @ np.linalg.inv(E_l[0])
                        G = (np.eye(4, dtype=np.float64) if window_index == 0
                             else global_extr[s])
                        E_g = E_rel @ G
                        # 相机重叠区保留先前窗的结果，本窗只写尚未覆盖的新帧。
                        j0 = max(0, next_pose_frame - s)
                        Rg, Tg = E_g[j0:, :3, :3], E_g[j0:, :3, 3]
                        qg = mat_to_quat(torch.from_numpy(Rg).float()).numpy()
                        pose_out[s + j0:e] = np.concatenate(
                            [Tg.astype(np.float32), qg.astype(np.float32), pe_l[j0:, 7:]],
                            axis=-1,
                        )
                        global_extr[s + j0:e] = E_g[j0:]
                        next_pose_frame = max(next_pose_frame, e)
                    else:
                        pose_out[s:e] = pe_l
                        next_pose_frame = max(next_pose_frame, e)
                    hand_j0 = 0 if (window_index == 0 or not overlap) else 1
                    blend = (_window_blend_weights(win_bounds, window_index)
                             if blend_hands else None)
                    if hand_batch is not None:
                        values = hand_batch[batch_index]
                        if blend_hands:
                            if hand_sum is None:
                                hand_sum = np.zeros((N, values.shape[-1]), dtype=np.float32)
                                hand_weight = np.zeros((N, 1), dtype=np.float32)
                            hand_sum[s:e] += values * blend[:, None]
                            hand_weight[s:e] += blend[:, None]
                        else:
                            hand_chunks.append(values[hand_j0:])
                    if initial_batch is not None:
                        values = initial_batch[batch_index]
                        if blend_hands:
                            if initial_sum is None:
                                initial_sum = np.zeros((N, values.shape[-1]), dtype=np.float32)
                                initial_weight = np.zeros((N, 1), dtype=np.float32)
                            initial_sum[s:e] += values * blend[:, None]
                            initial_weight[s:e] += blend[:, None]
                        else:
                            initial_chunks.append(values[hand_j0:])
                    if presence_batch is not None:
                        values = presence_batch[batch_index]
                        if blend_hands:
                            if presence_sum is None:
                                presence_sum = np.zeros((N, values.shape[-1]), dtype=np.float32)
                                presence_weight = np.zeros((N, 1), dtype=np.float32)
                            presence_sum[s:e] += values * blend[:, None]
                            presence_weight[s:e] += blend[:, None]
                        else:
                            presence_logit_chunks.append(values[hand_j0:])
                    if on_step is not None:
                        try:
                            on_step(window_index + 1, nwin)
                        except Exception:  # noqa: BLE001
                            pass

                batch_frames = sum(end - start for start, end in batch_bounds)
                print(
                    f"\r[predictor] forward 窗 {batch_end}/{nwin} "
                    f"(batch={len(batch_bounds)}, GPU={min(len(batch_bounds), self.parallel_device_count)}, "
                    f"帧至 {batch_bounds[-1][1]}/{N}) "
                    f"{dt:.2f}s, {dt / batch_frames * 1000:.1f} ms/帧\033[K",
                    end="", flush=True,
                )
                wi = batch_end
        print(f"\r[predictor] forward 完成：{N} 帧 / {nwin} 窗（重叠缝合），前向共 {t_fwd:.2f}s，"
              f"平均 {t_fwd / N * 1000:.1f} ms/帧（首窗含 CUDA 预热偏慢）\033[K", flush=True)
        res = {"pose_enc": pose_out}   # [N,9] 全局(episode 首帧系)
        if hand_chunks:
            res["hand"] = np.concatenate(hand_chunks, axis=0).astype(np.float32)
        elif hand_sum is not None:
            res["hand"] = (hand_sum / np.maximum(hand_weight, 1e-8)).astype(np.float32)
        if initial_chunks:
            res["_hand_refine_initial"] = np.concatenate(
                initial_chunks, axis=0).astype(np.float32)
        elif initial_sum is not None:
            res["_hand_refine_initial"] = (
                initial_sum / np.maximum(initial_weight, 1e-8)).astype(np.float32)
        if presence_logit_chunks:
            logits = np.concatenate(presence_logit_chunks, axis=0).astype(np.float32)
            res["hand_presence_logits"] = logits
            res["hand_confidence"] = _sigmoid_np(logits)
        elif presence_sum is not None:
            logits = (presence_sum / np.maximum(presence_weight, 1e-8)).astype(np.float32)
            res["hand_presence_logits"] = logits
            res["hand_confidence"] = _sigmoid_np(logits)
        hand_smoothed = _postprocess_hands(res, hand_mode, ukf_params)
        res["_timings"] = {
            "preprocess_s": preprocess_s,
            "forward_s": t_fwd,
            "total_s": time.perf_counter() - predict_started,
            "requested_window_batch_size": requested_batch,
            "window_batch_size": effective_batch,
            "forward_batches": forward_batches,
            "windows": nwin,
            "devices": self.parallel_device_names,
            "window_overlap": overlap,
            "hand_mode": hand_mode,
            "ukf_params": ukf_params if hand_smoothed else None,
            "hand_postprocess": "ukf_cam_rts" if hand_smoothed else "none",
            "inference_mode": cam_mode,
            "window_size": W,
            "full_max_frames": self.full_max_frames,
            **self.acceleration_metadata,
        }
        return res


HandReprojPredictor = StudentEngine   # 兼容别名：旧调用点（benchmark / 外部消费者）仍可 import 此名
