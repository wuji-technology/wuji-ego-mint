#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    resolve_compile_mode,
    resolve_fp8_mode,
)
from .base import FullSequenceTooLong, InferenceCancelled  # noqa: F401


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_DIR = Path(__file__).resolve().parents[2]
MODEL_TRAIN = REPO_DIR / "model_train"


def _ensure_model_train_on_path() -> None:
    """Internal helper."""
    for p in (str(MODEL_TRAIN), str(MODEL_TRAIN / "_vendor")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _abs(p):
    """Internal helper."""
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
        print(f"Selected inference devices: {selected or ['cpu']}.", flush=True)
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
    """Resolve a checkpoint file from either a file or training directory."""
    if os.path.isfile(ckpt):
        return ckpt
    if os.path.isdir(ckpt):
        for pat in ("model.safetensors", "pytorch_model.bin",
                    "*.safetensors", "pytorch_model*.bin", "*.pt", "*.pth"):
            hits = sorted(glob.glob(os.path.join(ckpt, pat)))
            if hits:
                return hits[0]
        raise FileNotFoundError(
            f"No supported checkpoint file was found in: {ckpt}")
    raise FileNotFoundError(f"Checkpoint does not exist: {ckpt}")


def _clip_len_from_ckpt(ckpt: str) -> int | None:
    import json

    def _clip_len_of(yaml_path: str) -> int | None:
        with open(yaml_path) as f:
            tcfg = yaml.safe_load(f) or {}
        cl = (tcfg.get("data") or {}).get("clip_len")
        return int(cl) if cl is not None else None

    try:
        run_root = Path(ckpt).resolve()
        for _ in range(4):
            if ((run_root / "logs" / "record" / "config.yaml").is_file()
                    or (run_root / "config" / "config.yaml").is_file()
                    or (run_root / "wandb").is_dir()):
                break
            run_root = run_root.parent
        else:
            return None

        for snap in (run_root / "logs" / "record" / "config.yaml",
                     run_root / "config" / "config.yaml"):
            if snap.is_file():
                cl = _clip_len_of(str(snap))
                if cl is not None:
                    return cl

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
    """Internal helper."""

    def __init__(self, config_path: str, ckpt: str | None = None,
                 device: str | None = None, window: int | None = None,
                 devices=None, full_max_frames: int | None = None,
                 compile_mode: str | None = None, fp8_mode: str | None = None):
        self.compile_mode_requested = normalize_optional_mode(
            compile_mode, choices=COMPILE_MODES, name="compile_mode"
        )
        self.fp8_mode_requested = normalize_optional_mode(
            fp8_mode, choices=FP8_MODES, name="fp8_mode"
        )
        allocator_changed = prepare_allocator_for_compile(self.compile_mode_requested)
        _ensure_model_train_on_path()
        import torch
        from models import build_model                       # noqa: E402  (model_train)
        from utils.weight_loader import load_state_dict_file  # noqa: E402

        if allocator_changed:
            print("[inference] disabled expandable_segments for CUDA Graph capture", flush=True)

        with open(_abs(config_path)) as f:
            cfg = yaml.safe_load(f)
        mcfg = dict(cfg["model"])
        self._raw_mcfg = dict(cfg["model"])

        self.size_hw = tuple(cfg.get("data", {}).get("size_hw", (378, 518)))




        if window is None:
            cl = _clip_len_from_ckpt(ckpt) if ckpt else None
            src = "checkpoint metadata" if cl is not None else None
            if cl is None:
                cl = cfg.get("data", {}).get("clip_len")
                src = "training config" if cl is not None else None
            if cl is None:
                cl, src = 16, "fallback"
            window = int(cl)
            print(f"[inference] window={window} source={src}")
        else:
            print(f"[inference] window={int(window)} source=command-line")
        self.window = int(window)



        if ckpt:
            mcfg["pretrained"] = None
            mcfg["_ckpt_provided"] = True
        elif mcfg.get("pretrained"):
            mcfg["pretrained"] = _abs(mcfg["pretrained"])

        requested_devices = _resolve_devices(device=device, devices=devices)
        resolved_devices = [torch.device(item) for item in requested_devices]
        self.device = resolved_devices[0]
        self.compile_mode, compile_reason = resolve_compile_mode(
            self.compile_mode_requested, resolved_devices
        )
        self.fp8_mode, fp8_reason = resolve_fp8_mode(
            self.fp8_mode_requested, resolved_devices, self.compile_mode
        )
        if self.compile_mode_requested == "auto":
            print(
                f"[inference] compile auto -> {self.compile_mode or 'off'} ({compile_reason})",
                flush=True,
            )
        if self.fp8_mode_requested == "auto":
            print(
                f"[inference] FP8 auto -> {self.fp8_mode or 'off'} ({fp8_reason})",
                flush=True,
            )
        cuda_memories = [
            torch.cuda.get_device_properties(item).total_memory
            for item in resolved_devices
            if item.type == "cuda"
        ]
        limiting_memory = min(cuda_memories) if cuda_memories else None
        if full_max_frames is None:
            self.full_max_frames = _auto_full_max_frames(self.window, limiting_memory)
            full_limit_source = "automatic"
        else:
            self.full_max_frames = int(full_max_frames)
            if self.full_max_frames <= 0 or self.full_max_frames % self.window:
                raise ValueError(
                    f"full_max_frames must be a positive multiple of window={self.window}; "
                    f"received {self.full_max_frames}"
                )
            full_limit_source = "configured"
        print(
            f"[inference] full_sequence_limit={self.full_max_frames} "
            f"source={full_limit_source}"
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
                print("[inference] checkpoint has no hand-presence head; feature disabled")
            print(f"[inference] checkpoint={os.path.basename(wf)} "
                  f"tensors={len(sd)} missing={len(missing)} unexpected={len(unexpected)}")
        self.model = model.to(self.device).eval()
        self.models = [self.model]
        self.devices = [self.device]



        for replica_device in resolved_devices[1:]:
            replica = None
            try:
                replica = build_model(mcfg)
                if sd is not None:
                    replica.load_state_dict(sd, strict=False)
                    if presence_missing:
                        replica.enable_hand_presence = False
                else:

                    replica.load_state_dict(self.model.state_dict(), strict=True)
                replica = replica.to(replica_device).eval()
            except torch.cuda.OutOfMemoryError:
                del replica
                if replica_device.type == "cuda":
                    with torch.cuda.device(replica_device):
                        torch.cuda.empty_cache()
                print(f"[warning] Skipping replica on {replica_device}: out of memory", flush=True)
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
                f"[inference] FP8 dynamic enabled: "
                f"{len(self._fp8_conversions[0].module_names)} Linear layers per replica, "
                f"torchao={self._fp8_conversions[0].torchao_version}",
                flush=True,
            )

        self._compiled_module_names = []
        if self.compile_mode is not None:
            # Populate RoPE caches eagerly so captured graphs do not own mutable
            # cache tensors that are overwritten on replay.
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
                f"[inference] torch.compile enabled: mode={self.compile_mode}, "
                f"hotspots={len(self._compiled_module_names[0])} per replica, "
                "dynamic=False",
                flush=True,
            )
        self._model_locks = [threading.Lock() for _ in self.models]
        self._forward_pool = (
            ThreadPoolExecutor(max_workers=len(self.models), thread_name_prefix="gpu-forward")
            if len(self.models) > 1 else None
        )
        print(f"[inference] devices={','.join(map(str, self.devices))}", flush=True)
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
            "compile_mode_requested": self.compile_mode_requested or "off",
            "compile_mode": self.compile_mode or "off",
            "fp8_mode_requested": self.fp8_mode_requested or "off",
            "fp8_mode": self.fp8_mode or "off",
        }

    def _effective_window_batch_size(self, requested_batch: int) -> int:
        """Keep every compiled model replica on its local batch-one shape."""
        requested_batch = max(1, int(requested_batch))
        if self.compile_mode is None:
            return requested_batch
        return min(requested_batch, self.parallel_device_count)

    def warmup_acceleration(self, window_batch_size: int = 1, passes: int = 2) -> dict:
        """Compile and capture the fixed full-window shape before measured inference."""
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
        if not self.supports_weight_reload:
            raise RuntimeError(
                "FP8/torch.compile engines must be rebuilt when changing checkpoints"
            )
        from utils.weight_loader import load_state_dict_file   # noqa: E402
        wf = _find_weight_file(ckpt)
        models = getattr(self, "models", [self.model])

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
            print("[inference] checkpoint has no hand-presence head; feature disabled")
        self.has_hand = bool(getattr(self.model, "enable_hand", False))
        self.has_hand_presence = bool(
            getattr(self.model, "enable_hand_presence", False)
        )
        print(f"[inference] reloaded={os.path.basename(wf)} replicas={len(models)} "
              f"tensors={len(sd)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
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
        import time
        import torch
        from data.transforms import preprocess_frames

        from .hand_smoothing import parse_hand_smoothing_mode
        requested_hand_mode = str(hand_mode)
        try:
            hand_mode, ukf_params = parse_hand_smoothing_mode(requested_hand_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unknown hand mode {requested_hand_mode!r}; choose from "
                f"{', '.join(HAND_WINDOW_MODES)}; smooth may include q/r/beta."
            ) from exc
        if cam_mode not in CAMERA_INFERENCE_MODES:
            raise ValueError(
                f"Unknown camera mode {cam_mode!r}; choose from {', '.join(CAMERA_INFERENCE_MODES)}."
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
                    "preprocessed=True requires a floating-point tensor in model input layout."
                )
            x = frames_uint8.contiguous()
        else:
            x = preprocess_frames(frames_uint8, self.size_hw)   # [N,3,H,W] float[0,1]
        preprocess_s = time.perf_counter() - preprocess_started
        N = x.shape[0]
        cuda = self.device.type == "cuda"
        W = self.full_max_frames if cam_mode == "max_chunked" else self.window


        if cam_mode == "full":
            if N <= 0:
                raise ValueError("Inference requires at least one frame.")
            if cancel_check is not None and cancel_check():
                raise InferenceCancelled("Cancelled before full-sequence inference.")
            print(f"[inference] mode=full frames={N}", flush=True)
            try:
                out, t_fwd = self._forward_window_batch(
                    x,
                    [(0, N)],
                    disable_persistent_kv_cache=True,
                )
            except torch.cuda.OutOfMemoryError:
                print(
                    f"[inference] full-sequence CUDA out of memory for {N} frames",
                    flush=True,
                )
                raise
            if cancel_check is not None and cancel_check():
                raise InferenceCancelled("Cancelled after full-sequence inference.")

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
                f"[inference] completed frames={N} forward_s={t_fwd:.2f} "
                f"ms_per_frame={t_fwd / N * 1000:.1f}",
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



        if cam_mode == "streaming":
            _ensure_model_train_on_path()
            from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
            from lingbot_map.utils.rotation import mat_to_quat
            if cancel_check is not None and cancel_check():
                raise InferenceCancelled("Cancelled before streaming inference.")


            print(f"[inference] mode=streaming frames={N}", flush=True)
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
            pe = sout["pose_enc"][0].float().numpy()
            stream_elapsed = time.perf_counter() - t0
            print(f"[inference] streaming_frames={N} forward_s={stream_elapsed:.2f} "
                  f"ms_per_frame={stream_elapsed / max(N, 1) * 1000:.1f}", flush=True)

            extr = pose_encoding_to_extri_intri(
                torch.from_numpy(pe)[None], build_intrinsics=False)[0][0].numpy()   # [N,3,4]
            E = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
            E[:, :3, :4] = extr
            E_rb = E @ np.linalg.inv(E[0])
            R_rb, T_rb = E_rb[:, :3, :3], E_rb[:, :3, 3]
            q_rb = mat_to_quat(torch.from_numpy(R_rb).float()).numpy()
            pose = np.concatenate(
                [T_rb.astype(np.float32), q_rb.astype(np.float32), pe[:, 7:]], axis=-1)
            res = {"pose_enc": pose.astype(np.float32)}

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
                            raise InferenceCancelled(
                                f"Cancelled at hand window {wi + 1}/{len(bounds)}."
                            )
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
        t_fwd = 0.0
        requested_batch = max(
            1,
            self.parallel_device_count if window_batch_size is None else int(window_batch_size),
        )
        effective_batch = self._effective_window_batch_size(requested_batch)
        if effective_batch != requested_batch:
            print(
                f"[inference] torch.compile limits window batch from {requested_batch} "
                f"to {effective_batch} so every GPU keeps local batch=1",
                flush=True,
            )
        forward_batches = 0
        with torch.inference_mode():
            wi = 0
            while wi < nwin:
                if cancel_check is not None and cancel_check():
                    raise InferenceCancelled(f"Cancelled at window {wi + 1}/{nwin}.")


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
                        f"[inference] reducing window batch from {len(batch_bounds)} "
                        f"to {effective_batch} after CUDA out of memory",
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
                    pe_l = pose_batch[batch_index]
                    L = e - s
                    if overlap:

                        extr_l = pose_encoding_to_extri_intri(
                            torch.from_numpy(pe_l)[None], build_intrinsics=False
                        )[0][0].numpy()
                        E_l = np.tile(np.eye(4, dtype=np.float64), (L, 1, 1))
                        E_l[:, :3, :4] = extr_l
                        E_rel = E_l @ np.linalg.inv(E_l[0])
                        G = (np.eye(4, dtype=np.float64) if window_index == 0
                             else global_extr[s])
                        E_g = E_rel @ G

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
                    f"[inference] windows={batch_end}/{nwin} "
                    f"batch={len(batch_bounds)} devices={min(len(batch_bounds), self.parallel_device_count)} "
                    f"frames={batch_bounds[-1][1]}/{N} forward_s={dt:.2f} "
                    f"ms_per_frame={dt / batch_frames * 1000:.1f}\r",
                    end="", flush=True,
                )
                wi = batch_end
        print(f"[inference] completed frames={N} windows={nwin} forward_s={t_fwd:.2f} "
              f"ms_per_frame={t_fwd / N * 1000:.1f}", flush=True)
        res = {"pose_enc": pose_out}
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


HandReprojPredictor = StudentEngine
