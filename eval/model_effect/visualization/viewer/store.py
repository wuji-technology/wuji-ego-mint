#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""服务端状态 + 缓存（按 episode / (episode,mode) 串行，重活只算一次）。

分层缓存：raw → pred → world → payload；2D overlay mp4 单独缓存。模型（4.6G 骨干）+ MANO 资产
后台懒加载；条目（lerobot episode / 裸视频 / Benchmark 序列）按 eid 懒登记，有/无真值按项判定。
取消/进度/2D 渲染进度均线程安全。

reproj_core / render 按方法内惰性相对 import；inference 走绝对顶级名（须 model_effect 目录在 sys.path，由入口设置）。
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

import numpy as np

from . import ckpts
from .const import (
    DEFAULT_CAM_MODE, DEFAULT_HAND_MODE, MODEL_TRAIN_ROOT, REPO_DIR,
    ROBOT_RENDER_WIDTH,
)

import re as _re
_ANSI = _re.compile(r"\x1b\[[0-9;]*[A-Za-z]")   # 去 \033[K 等 ANSI(面板 <pre> 里会显成乱码)
_EXPORT_TILE_SIZE = (960, 540)
_EXPORT_GRID_COLUMNS = 4
_EXPORT_COMPOSE_TAG = "source_sized_tiles_v7_four_columns_two_rows"
_EXPORT_BUNDLE_TAG = "individual_gt_pred_v1"


def _export_grid_layout(count: int, tile_width: int, tile_height: int):
    """Return ffmpeg xstack positions and the resulting canvas size."""
    if count < 1:
        raise ValueError("导出网格至少需要一个画面")
    columns = min(_EXPORT_GRID_COLUMNS, int(count))
    rows = (int(count) + columns - 1) // columns
    layout = "|".join(
        f"{(index % columns) * tile_width}_{(index // columns) * tile_height}"
        for index in range(int(count)))
    return layout, (columns * tile_width, rows * tile_height)


def _robot_render_size(render_size: tuple[int, int] | None) -> tuple[int, int | None]:
    if render_size is None:
        return ROBOT_RENDER_WIDTH, None
    if not isinstance(render_size, (tuple, list)) or len(render_size) != 2:
        raise ValueError("机器人渲染尺寸应为 (width, height)")
    width, height = map(int, render_size)
    if width < 4 or height < 4:
        raise ValueError("机器人渲染尺寸过小")
    return width + width % 2, height + height % 2


def _benchmark_progress_timing(progress: dict, *, running: bool,
                               started_at=None, finished_at=None) -> dict:
    """Attach wall-clock elapsed/ETA fields so a browser refresh does not reset timing."""
    value = dict(progress or {})
    if started_at is None:
        return value
    end = time.time() if running or finished_at is None else float(finished_at)
    elapsed = max(0.0, end - float(started_at))
    frac = value.get("suite_frac", value.get("frac"))
    value["elapsed_s"] = elapsed
    if not running:
        value.update(estimated_total_s=elapsed, remaining_s=0.0)
    elif isinstance(frac, (int, float)) and 0.005 <= float(frac) <= 1.0:
        estimated = elapsed / float(frac)
        value.update(estimated_total_s=estimated,
                     remaining_s=max(0.0, estimated - elapsed))
    return value


def _render_video_atomically(output: Path, render) -> Path:
    """Keep an in-progress MP4 hidden until ffmpeg has written its final metadata."""
    output = Path(output)
    temporary = output.with_name(
        f".{output.stem}.{os.getpid()}-{threading.get_ident()}.rendering.mp4"
    )
    temporary.unlink(missing_ok=True)
    try:
        render(temporary)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("视频编码失败，未生成有效 MP4")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _safe_frame_metrics(raw, pred, loss_cfg, hw, decode):
    """Keep visualization usable when optional training-loss metadata is absent."""
    try:
        from ..render import metrics as metricspl
        return metricspl.frame_metrics(raw, pred, loss_cfg, hw, decode), None
    except Exception as error:  # noqa: BLE001
        detail = f"{type(error).__name__}: {error}"
        print(
            f"[store] frame_metrics 失败(跳过逐帧 loss，继续返回可视化): {detail}",
            flush=True,
        )
        return None, detail


def _merge_live_results(shard_rows: dict, *, ckpt=None, config="", selection=None) -> dict:
    """Build the same report tree as the final aggregate from in-memory RESULT events."""
    from benchmark.dist.aggregate import merge_result_rows

    return merge_result_rows(
        shard_rows, ckpt=ckpt, config=config, selection=selection,
    )


def _aggregate_benchmark_shards(base: Path, returncodes: dict[int, int], cancelled: bool):
    """Aggregate completed shards while treating an early user cancellation as normal."""
    from benchmark.dist.aggregate import aggregate

    reports = {
        device: base / f"gpu{device}" / "report.json"
        for device in returncodes
        if (base / f"gpu{device}" / "report.json").is_file()
    }
    if not reports:
        if cancelled:
            return None, None
        logs = ", ".join(
            str(base / f"gpu{device}" / "process.log") for device in returncodes
        )
        raise RuntimeError(
            f"所有 GPU 分片均未生成 report.json；returncode={returncodes}；日志: {logs}"
        )

    merged = aggregate(base)
    missing = [device for device in returncodes if device not in reports]
    failed = {device: code for device, code in returncodes.items() if code != 0}
    if not cancelled and (missing or failed):
        detail = f"部分 GPU 分片失败：missing={missing}, returncode={failed}"
        return merged, detail
    return merged, None


class _StdoutTee:
    """benchmark 运行期临时替换 sys.stdout:既照常打到真实控制台,又把关注前缀的行喂给 sink,
    让 [predictor]/[mano] 等推理日志出现在面板「运行日志」。逐窗进度用 \\r 原地刷新(engine 里
    end='' + \\r),这里按 \\r/\\n 切行、标 is_progress,sink 端原地替换而非堆积成几百行。"""
    _PREFIXES = ("[predictor]", "[mano]", "[run]", "[report]", "[aggregate]", "[sintel]")

    def __init__(self, orig, sink):
        self._orig = orig
        self._sink = sink
        self._buf = ""

    def write(self, s):
        try:
            self._orig.write(s)                 # 照常打到服务器控制台(不改原行为)
        except Exception:
            pass
        self._buf += s
        while True:
            cands = [x for x in (self._buf.find("\n"), self._buf.find("\r")) if x >= 0]
            if not cands:
                break
            i = min(cands)
            sep = self._buf[i]
            line = _ANSI.sub("", self._buf[:i]).strip()
            self._buf = self._buf[i + 1:]
            if line and line.startswith(self._PREFIXES):
                try:
                    self._sink(line, sep == "\r")
                except Exception:
                    pass

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def __getattr__(self, name):                # isatty/encoding 等透传给原 stdout
        return getattr(self._orig, name)


class Store:
    def __init__(self, predictor, *, root: Path, default_mode: str,
                 max_frames: int | None, scene: str, cache_dir: Path,
                 loss_cfg: dict | None = None, config_path: str | None = None,
                 window: int | None = None, full_max_frames: int | None = None,
                 ckpt_path: str | None = None,
                 default_fps: float = 30.0, default_hand_frame: str = "camera",
                 init_config: str | None = None, model: str = "lingbotmap",
                 inference_devices="auto", inference_compile_mode=None,
                 inference_fp8_mode=None, batch_workers: int = 2):
        self.model = model                            # 经 inference.registry 取对应模型引擎（后台懒加载时用）
        self.inference_devices = inference_devices    # auto=加载模型时使用所有可见 GPU
        self.inference_compile_mode = inference_compile_mode
        self.inference_fp8_mode = inference_fp8_mode
        self.batch_workers = max(1, int(batch_workers))
        self.loss_cfg = loss_cfg or {}
        self.root = Path(root)                        # 浏览根（= --input 目录，前端逐级浏览）
        self.default_fps = float(default_fps)         # 视频项（无真值）默认抽帧/回放 fps
        self.default_hand_frame = default_hand_frame  # 视频项预测手坐标系（camera/world）
        # 条目按 eid 懒登记：lerobot episode、裸视频与 Benchmark 序列共用同一 eid 空间：
        #   video   : {"kind":"video",   "path":abs, "fps":float, "hand_frame":str}
        #   lerobot : {"kind":"lerobot", "ds_dir":Path, "ep":dict,  "fps":float}
        #   benchmark: {"kind":"benchmark", "dataset":str, "image_paths":[...], "fps":float}
        self._items: list[dict] = []              # eid → 条目描述
        self._item_index: dict = {}               # 归一 key → eid（首见即分配、稳定复用）
        self._items_lock = threading.Lock()
        self._ds_cache: dict[str, dict] = {}      # ds_dir(str) → {"eps":[...], "fps":float}
        self._ds_lock = threading.Lock()
        # 数据集 episode 枚举**异步化**：进入 lerobot 目录不阻塞，后台线程枚举，前端轮询进度条。
        self._ds_progress: dict[str, dict] = {}   # ds_dir(str) → {ready,stage,done,total,episodes,fps,error}
        self._dsprog_lock = threading.Lock()
        self.predictor = predictor
        self.config_path = config_path
        self.window = window
        self.full_max_frames = full_max_frames
        self.ckpt_path = ckpt_path
        self.ckpt_tag = ckpts.ckpt_tag(ckpt_path)
        self._ckpt_lock = threading.Lock()
        self._load_cv = threading.Condition(self._ckpt_lock)   # 与 _ckpt_lock 同锁：等/通知「模型就绪或失败」
        # 模型（4.6G 骨干）后台懒加载：网页秒开，首次推理前等就绪。predictor=None 时用 init_config 现建。
        self._boot_cfg = init_config or config_path
        self._reload_ckpt = None                  # 非 None=换 ckpt 但架构不变，ensure 时就地重载而非重建
        self._predictor_ready = threading.Event()
        if predictor is not None:
            self._predictor_ready.set()
        # 换模型的「代次」：每次 swap_ckpt +1。后台单加载线程每轮只提交仍是最新代的结果，
        # 途中被跨过的选择直接跳过 → 加载途中再点别的模型，最终加载的一定是最后点的那个。
        self._load_gen = 0
        self._loading = False                     # 是否已有后台加载线程在跑（保证同一时刻仅一个，不并发建模/叠显存）
        self._load_err: Exception | None = None   # 最新代加载失败时的异常，供 ensure_predictor 抛出
        self._assets_ready = threading.Event()    # MANO 权重就绪（渲染/hands_to_world 需要）
        self._assets_lock = threading.Lock()
        self.default_mode = default_mode
        self.max_frames = max_frames
        self.scene = scene
        self.cache_dir = cache_dir
        # 依赖推理的缓存 key 带上 cam_mode + hand_mode，模式切换不互相命中。
        # GT 相关（_raw/_gtw/_payload_gt）与推理无关，key 仍只用 eid。
        self._raw: dict[int, dict] = {}
        self._pred: dict[tuple, dict] = {}
        self._gtw: dict[int, dict] = {}
        self._prw: dict[tuple, dict] = {}
        self._payload: dict[tuple, dict] = {}
        self._payload_gt: dict[int, dict] = {}   # 「仅原始 GT」payload（不跑推理）
        self._mp4: dict[tuple, Path] = {}
        self._world_mp4: dict[tuple, Path] = {}
        self._world_prog: dict[tuple, dict] = {}
        self._mujoco_mp4: dict[tuple, Path] = {}
        self._mujoco_prog: dict[tuple, dict] = {}
        self._retarget_mp4: dict[tuple, Path] = {}
        self._retarget_prog: dict[tuple, dict] = {}
        # Independent EGL contexts can render concurrently. Bound the total so several browser
        # requests cannot create an unbounded number of large offscreen renderers.
        self._mujoco_render_lock = threading.BoundedSemaphore(2)
        self._reg_lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._prog: dict[int, dict] = {}          # eid → 推理进度（前端轮询用）
        self._prog2d: dict[str, dict] = {}        # 2D 渲染进度：key=(eid:mode:layout:content) → {done,total}
        self._prog_lock = threading.Lock()
        self._cancel: dict[int, threading.Event] = {}   # eid → 取消事件（「停止」按钮置位）
        self._cancel_lock = threading.Lock()
        # benchmark：进程内复用**已加载**的引擎跑评测（benchmark.run.run_benchmark），不重复加载权重。
        # 逐条结果回调更新进度/日志；一次只允许一个测评在跑；结果供前端 benchmark 面板轮询展示。
        self._bench = {"running": False, "phase": "", "count": 0, "log": [],
                       "report": None, "error": None, "out": None, "ckpt_tag": None,
                       "progress": {}, "live_report": None, "log_slots": {},
                       "models": [], "active_model": None,
                       "started_at": None, "finished_at": None}
        self._bench_lock = threading.Lock()
        self._bench_cancel = threading.Event()    # 「取消」置位，run_benchmark 在序列边界查、提前收尾
        self._bench_procs: list = []              # 多卡评测的子进程句柄（取消时逐个杀进程组）
        from benchmark.dist.aliyun import AliyunBenchmarkManager
        self._aliyun_bench = AliyunBenchmarkManager(REPO_DIR)
        self._bench_backend = "local"
        # 批量视频推理：GPU 推理串行使用全部副本，完成后交给渲染池；两阶段重叠，GPU 不再等 MP4。
        self._batch = {"running": False, "phase": "", "input_root": None,
                       "output_root": None, "name_template": None,
                       "checkpoint": None, "checkpoint_path": None, "total": 0,
                       "done": 0, "succeeded": 0, "failed": 0, "skipped": 0,
                       "current": None, "current_index": 0, "file_done": 0,
                       "file_total": 0, "devices": [], "workers": self.batch_workers,
                       "active": [], "log": [], "error": None, "manifest": None,
                       "cancelled": False}
        self._batch_lock = threading.Lock()
        # Serialize checkpoint selection with batch startup so a concurrent /api/ckpt request
        # cannot replace the model between selecting it and marking the batch as running.
        self._batch_start_lock = threading.RLock()
        self._batch_cancel = threading.Event()
        self._batch_current_eid: int | None = None
        self._batch_active_eids: set[int] = set()
        self._batch_active: dict[int, dict] = {}
        # 每数据集规模（序列条数/总帧数）：不加载模型，首访时后台线程算一次并缓存，供面板跑前展示。
        self._bench_sizes = {"computing": False, "data": None}
        # log_diff：起子进程对比两个 run 的配置和代码，读回 result.json 供前端展示。
        # 一次只允许一个对比在跑；纯读取、不动训练产物。
        self._logdiff = {"running": False, "phase": "", "result": None, "error": None,
                         "run_a": None, "run_b": None, "code_scope": "",
                         "out": None, "log": []}
        self._logdiff_lock = threading.Lock()

    # ---- 取消：前端「停止」置位 event，predict 逐窗查、payload 分段查 → 抛 InferenceCancelled ----
    def request_cancel(self, eid: int) -> None:
        with self._cancel_lock:
            self._cancel.setdefault(eid, threading.Event()).set()

    def clear_cancel(self, eid: int) -> None:
        with self._cancel_lock:
            self._cancel.setdefault(eid, threading.Event()).clear()

    def cancelled(self, eid: int) -> bool:
        with self._cancel_lock:
            ev = self._cancel.get(eid)
            return bool(ev and ev.is_set())

    def _raise_if_cancelled(self, eid: int) -> None:
        if self.cancelled(eid):
            from inference.base import InferenceCancelled
            raise InferenceCancelled(f"episode {eid} 加载/推理被取消")

    # ---- 批量视频推理：镜像目录输出预测 NPZ + overlay MP4，前端轮询状态 ----
    def batch_inference_status(self) -> dict:
        with self._batch_lock:
            status = dict(self._batch)
            status["log"] = list(self._batch.get("log", ())[-80:])
        total = int(status.get("total") or 0)
        status["progress"] = (float(status.get("done") or 0) / total) if total else 0.0
        return status

    def _batch_set(self, **values) -> None:
        with self._batch_lock:
            self._batch.update(values)

    def _batch_log(self, message: str) -> None:
        print(f"[batch] {message}", flush=True)
        with self._batch_lock:
            self._batch["log"].append(str(message))
            if len(self._batch["log"]) > 300:
                self._batch["log"] = self._batch["log"][-300:]

    def _batch_activity(self, index: int, relative: Path | str, stage: str | None,
                        *, eid: int | None = None, done: int = 0, total: int = 0) -> None:
        """Publish all in-flight videos while keeping the legacy single-current fields useful."""
        labels = {"load": "读取", "infer": "推理", "render": "渲染", "write": "写入"}
        with self._batch_lock:
            if stage is None:
                self._batch_active.pop(int(index), None)
                if eid is not None:
                    self._batch_active_eids.discard(int(eid))
            else:
                rec = {
                    "index": int(index), "path": str(relative), "stage": stage,
                    "stage_label": labels.get(stage, stage),
                    "done": int(done), "total": int(total),
                }
                self._batch_active[int(index)] = rec
                if eid is not None:
                    self._batch_active_eids.add(int(eid))

            active = [dict(self._batch_active[key]) for key in sorted(self._batch_active)]
            self._batch["active"] = active
            # Prefer showing the GPU stage as the legacy current item; render callbacks cannot hide it.
            primary = next((item for item in active if item["stage"] == "infer"),
                           active[0] if active else None)
            if primary is None:
                self._batch.update(current=None, current_index=0, file_done=0, file_total=0)
            else:
                self._batch.update(
                    current=primary["path"], current_index=primary["index"],
                    file_done=primary["done"], file_total=primary["total"],
                )

            if active and not self._batch_cancel.is_set():
                counts = {key: 0 for key in labels}
                for item in active:
                    counts[item["stage"]] = counts.get(item["stage"], 0) + 1
                parts = [f"{labels[key]} {counts[key]}" for key in labels if counts.get(key)]
                self._batch["phase"] = "并发处理中：" + " · ".join(parts)

    def start_batch_inference(self, *, input_dir: str, output_dir: str,
                              name_template: str = "{stem}_pred",
                              mode: str = "mesh_skel",
                              cam_mode: str = DEFAULT_CAM_MODE,
                              hand_mode: str = DEFAULT_HAND_MODE,
                              pred_betas_mean: bool = False,
                              pred_fov_mean: bool = False,
                              overwrite: bool = False,
                              checkpoint: str | None = None) -> dict:
        """Start a batch, switching to and loading ``checkpoint`` in the worker when requested."""
        from .batch import resolve_batch_roots, validate_name_template

        try:
            input_root, output_root = resolve_batch_roots(input_dir, output_dir)
            template = validate_name_template(name_template)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        if checkpoint and ckpts.resolve_checkpoint_path(checkpoint) is None:
            return {"ok": False, "error": "所选批量 checkpoint 文件或目录无效"}
        with self._batch_start_lock:
            with self._bench_lock:
                if self._bench["running"]:
                    return {"ok": False, "error": "Benchmark 正在运行，请结束后再启动批量推理"}
            with self._batch_lock:
                if self._batch["running"]:
                    return {"ok": False, "error": "已有批量推理任务正在运行"}

            requested_ckpt = str(Path(checkpoint).resolve()) if checkpoint else self.ckpt_path
            if checkpoint and (not self.ckpt_path
                               or Path(self.ckpt_path).resolve() != Path(requested_ckpt)):
                self.swap_ckpt(requested_ckpt)
            if not requested_ckpt and not self.predictor_ready():
                return {"ok": False, "error": "未选择批量模型，请先选择 Run 和 Step"}

            ckpt_path = str(Path(self.ckpt_path).resolve()) if self.ckpt_path else None
            ckpt_tag = self.ckpt_tag
            devices = self.predictor_devices() if self.predictor_ready() else []
            with self._batch_lock:
                self._batch_cancel.clear()
                self._batch_current_eid = None
                self._batch_active_eids.clear()
                self._batch_active.clear()
                self._batch = {
                    "running": True,
                    "phase": "扫描视频…" if self.predictor_ready() else "加载所选模型…",
                    "input_root": str(input_root), "output_root": str(output_root),
                    "name_template": template, "checkpoint": ckpt_tag,
                    "checkpoint_path": ckpt_path, "total": 0, "done": 0,
                    "succeeded": 0, "failed": 0, "skipped": 0,
                    "current": None, "current_index": 0, "file_done": 0,
                    "file_total": 0, "devices": devices, "workers": self.batch_workers,
                    "active": [], "log": [], "error": None, "manifest": None,
                    "cancelled": False,
                }
            kwargs = {
                "input_root": input_root, "output_root": output_root,
                "name_template": template, "mode": mode, "cam_mode": cam_mode,
                "hand_mode": hand_mode, "pred_betas_mean": bool(pred_betas_mean),
                "pred_fov_mean": bool(pred_fov_mean), "overwrite": bool(overwrite),
                "devices": devices, "ckpt_tag": ckpt_tag, "ckpt_path": ckpt_path,
            }
            threading.Thread(target=self._batch_inference_worker, kwargs=kwargs,
                             name="batch-inference", daemon=True).start()
        return {"ok": True, "devices": devices, "workers": self.batch_workers,
                "checkpoint": ckpt_tag,
                "checkpoint_path": ckpt_path, "model_ready": self.predictor_ready()}

    def cancel_batch_inference(self) -> dict:
        with self._batch_lock:
            if not self._batch["running"]:
                return {"ok": False, "error": "当前没有运行中的批量推理"}
            eids = set(self._batch_active_eids)
            if self._batch_current_eid is not None:
                eids.add(self._batch_current_eid)
            self._batch["phase"] = "取消中…（当前推理窗口或在途渲染结束后停止）"
        self._batch_cancel.set()
        for eid in eids:
            self.request_cancel(eid)
        return {"ok": True}

    def _batch_prepare_video(self, item: dict, *, cam_mode: str, hand_mode: str,
                             mode: str, pred_betas_mean: bool,
                             pred_fov_mean: bool, ckpt_tag: str,
                             ckpt_path: str | None) -> dict:
        """Read and infer one video; the returned CPU data can be rendered concurrently."""
        source = item["source"]
        relative = item["relative"]
        index = int(item["index"])
        eid = self.video_eid(str(source))
        self.clear_cancel(eid)
        self._batch_activity(index, relative, "load", eid=eid)
        if self._batch_cancel.is_set():
            self.request_cancel(eid)
            self._raise_if_cancelled(eid)
        self._batch_log(f"[{index}] 读取并推理 {relative}")
        raw = self.raw(eid, cancel_check=lambda: self.cancelled(eid))
        if self._batch_cancel.is_set():
            self.request_cancel(eid)
        self._raise_if_cancelled(eid)
        self._batch_activity(index, relative, "infer", eid=eid)
        with self._batch_lock:
            self._batch_current_eid = eid
        try:
            pred = self.pred(eid, cam_mode, hand_mode)
            if "hand" not in pred:
                raise RuntimeError("模型预测中没有 hand 输出，无法渲染")
            self.ensure_assets()
            pred_world = self.pred_world(eid, cam_mode, hand_mode, pred_betas_mean)
            from ..reproj_core import geometry as geom
            height, width = raw["frames"].shape[1:3]
            camera_c2w, camera_k = geom.decode_camera_pose_enc(
                pred["pose_enc"], height, width, fov_mean=pred_fov_mean)
        finally:
            with self._batch_lock:
                if self._batch_current_eid == eid:
                    self._batch_current_eid = None

        arrays = {
            "pose_enc": np.asarray(pred["pose_enc"]),
            "camera_c2w": np.asarray(camera_c2w),
            "camera_K": np.asarray(camera_k),
            "fps": np.asarray(self.item_fps(eid), dtype=np.float32),
            "frame_count": np.asarray(raw["frames"].shape[0], dtype=np.int64),
            "source_path": np.asarray(str(source)),
            "source_relpath": np.asarray(relative.as_posix()),
            "ckpt_tag": np.asarray(ckpt_tag),
            "ckpt_path": np.asarray(ckpt_path or ""),
            "cam_mode": np.asarray(cam_mode),
            "hand_mode": np.asarray(hand_mode),
            "render_mode": np.asarray(mode),
            "pred_betas": np.asarray("mean" if pred_betas_mean else "per_frame"),
            "pred_fov": np.asarray("mean" if pred_fov_mean else "per_frame"),
        }
        for key in (
            "hand", "_hand_refine_initial",
            "hand_presence_logits", "hand_confidence",
        ):
            if pred.get(key) is not None:
                arrays[key] = np.asarray(pred[key])
        if pred.get("_timings") is not None:
            import json
            arrays["timings_json"] = np.asarray(
                json.dumps(pred["_timings"], ensure_ascii=False))
        return {"eid": eid, "raw": raw, "pred": pred, "pred_world": pred_world,
                "arrays": arrays}

    def _batch_render_video(self, item: dict, prepared: dict, *, mode: str,
                            pred_betas_mean: bool, pred_fov_mean: bool) -> int:
        """Render and write one prepared video in the bounded CPU/ffmpeg pool."""
        import tempfile

        from .batch import atomic_copy_file, atomic_save_npz

        index = int(item["index"])
        relative = item["relative"]
        output_mp4 = item["output_mp4"]
        output_npz = item["output_npz"]
        eid = int(prepared["eid"])
        raw = prepared["raw"]
        pred = prepared["pred"]
        total = int(raw["frames"].shape[0])
        if self._batch_cancel.is_set():
            from inference.base import InferenceCancelled
            raise InferenceCancelled(f"{relative} 在渲染前被取消")

        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        self._batch_activity(index, relative, "render", eid=eid, total=total)
        # ffmpeg/Zip need seekable files. Build locally, then expose complete remote objects.
        with tempfile.TemporaryDirectory(prefix="wuji-viewer-batch-") as tmp_dir:
            tmp_mp4 = Path(tmp_dir) / "render.mp4"
            tmp_npz = Path(tmp_dir) / "prediction.npz"
            from ..render import compare
            compare.render_pred_overlay(
                raw["frames"], pred, tmp_mp4, mode=mode, fps=self.item_fps(eid),
                hand_frame=self.item_hand_frame(eid), betas_mean=pred_betas_mean,
                fov_mean=pred_fov_mean, pred_world_data=prepared["pred_world"],
                on_step=lambda current, count: self._batch_activity(
                    index, relative, "render", eid=eid,
                    done=int(current), total=int(count)),
            )
            atomic_save_npz(tmp_npz, **prepared["arrays"])
            self._batch_activity(index, relative, "write", eid=eid)
            atomic_copy_file(tmp_npz, output_npz)
            atomic_copy_file(tmp_mp4, output_mp4)
        return total

    def _batch_inference_worker(self, *, input_root: Path, output_root: Path,
                                name_template: str, mode: str, cam_mode: str,
                                hand_mode: str, pred_betas_mean: bool,
                                pred_fov_mean: bool, overwrite: bool,
                                devices: list[str], ckpt_tag: str,
                                ckpt_path: str | None) -> None:
        import datetime
        import traceback

        from .batch import atomic_write_json, build_output_plan

        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest_path = output_root / "batch_manifest.json"
        manifest = {
            "status": "running", "started_at": started_at, "finished_at": None,
            "input_root": str(input_root), "output_root": str(output_root),
            "name_template": name_template, "checkpoint": ckpt_tag,
            "checkpoint_path": ckpt_path,
            "devices": devices, "settings": {
                "mode": mode, "cam_mode": cam_mode, "hand_mode": hand_mode,
                "pred_betas": "mean" if pred_betas_mean else "per_frame",
                "pred_fov": "mean" if pred_fov_mean else "per_frame",
                "overwrite": overwrite, "render_workers": self.batch_workers,
            },
            "files": [],
        }
        succeeded = failed = skipped = done = 0
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(manifest_path, manifest)
            self._batch_set(manifest=str(manifest_path))
            if not self.predictor_ready():
                self._batch_log(f"加载批量任务所选模型: {ckpt_path or ckpt_tag}")
                self._batch_set(phase="加载所选模型…")
                self.ensure_predictor()
            if ckpt_path and (not self.ckpt_path
                              or Path(self.ckpt_path).resolve() != Path(ckpt_path).resolve()):
                raise RuntimeError("批量模型在任务启动期间发生变化，已停止以避免使用错误权重")
            if not bool(getattr(self.predictor, "has_hand", True)):
                raise RuntimeError("所选模型没有 hand 输出，无法渲染批量手部结果")
            devices = self.predictor_devices()
            manifest["devices"] = devices
            self._batch_set(devices=devices, phase="扫描视频…")
            plan = build_output_plan(input_root, output_root, name_template)
            atomic_write_json(manifest_path, manifest)
            self._batch_set(total=len(plan), phase=("准备推理…" if plan else "未发现支持的视频"),
                            manifest=str(manifest_path))
            self._batch_log(
                f"发现 {len(plan)} 个视频；模型 {ckpt_tag}；使用全部已加载设备: "
                + (", ".join(devices) if devices else "未知")
            )
            self._batch_log(
                f"并发流水线已启用：1 路多 GPU 推理 + {self.batch_workers} 路渲染/写盘"
            )
            records: dict[int, dict] = {}
            work_items = []
            for item in plan:
                source, relative, index = item["source"], item["relative"], int(item["index"])
                record = {
                    "index": index, "source": str(source), "relative": relative.as_posix(),
                    "mp4": str(item["output_mp4"]), "npz": str(item["output_npz"]),
                    "status": "pending",
                }
                records[index] = record
                manifest["files"].append(record)
                if not overwrite and (item["output_mp4"].exists() or item["output_npz"].exists()):
                    skipped += 1
                    done += 1
                    record.update(status="skipped", reason="输出文件已存在（未启用覆盖）")
                    self._batch_log(f"[{index}/{len(plan)}] 跳过 {relative}: 输出已存在")
                else:
                    work_items.append(item)

            self._batch_set(done=done, succeeded=succeeded, failed=failed, skipped=skipped)
            atomic_write_json(manifest_path, manifest)
            pending = {}

            def finish_future(future) -> None:
                nonlocal succeeded, failed, done
                item, prepared = pending.pop(future)
                index, relative = int(item["index"]), item["relative"]
                record, eid = records[index], int(prepared["eid"])
                try:
                    frames = int(future.result())
                    succeeded += 1
                    done += 1
                    record.update(status="succeeded", frames=frames)
                    self._batch_log(f"[{index}/{len(plan)}] 完成 {relative}")
                except Exception as exc:  # noqa: BLE001
                    from inference.base import InferenceCancelled
                    if isinstance(exc, InferenceCancelled) or self._batch_cancel.is_set():
                        record.update(status="cancelled", error=str(exc))
                        self._batch_log(f"[{index}/{len(plan)}] 已取消 {relative}")
                    else:
                        failed += 1
                        done += 1
                        record.update(status="failed", error=str(exc))
                        self._batch_log(f"[{index}/{len(plan)}] 失败 {relative}: {exc}")
                        print("[batch] " + "".join(
                            traceback.format_exception(type(exc), exc, exc.__traceback__)), flush=True)
                finally:
                    self._batch_activity(index, relative, None, eid=eid)
                    self._release_item_cache(eid)
                    self._batch_set(done=done, succeeded=succeeded, failed=failed, skipped=skipped)
                    atomic_write_json(manifest_path, manifest)

            with ThreadPoolExecutor(
                    max_workers=self.batch_workers, thread_name_prefix="batch-render") as render_pool:
                for item in work_items:
                    if self._batch_cancel.is_set():
                        break
                    # Keep one prepared item ahead of the render pool so GPU inference overlaps
                    # a full render pool, while still bounding decoded-frame memory.
                    while len(pending) >= self.batch_workers + 1:
                        completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                        for future in completed:
                            finish_future(future)
                    if self._batch_cancel.is_set():
                        break

                    index, relative = int(item["index"]), item["relative"]
                    record = records[index]
                    record["status"] = "running"
                    try:
                        prepared = self._batch_prepare_video(
                            item, cam_mode=cam_mode, hand_mode=hand_mode, mode=mode,
                            pred_betas_mean=pred_betas_mean, pred_fov_mean=pred_fov_mean,
                            ckpt_tag=ckpt_tag, ckpt_path=ckpt_path)
                    except Exception as exc:  # noqa: BLE001
                        from inference.base import InferenceCancelled
                        eid = self.video_eid(str(item["source"]))
                        self._batch_activity(index, relative, None, eid=eid)
                        self._release_item_cache(eid)
                        if isinstance(exc, InferenceCancelled) or self._batch_cancel.is_set():
                            record.update(status="cancelled", error=str(exc))
                            self._batch_log(f"[{index}/{len(plan)}] 已取消 {relative}")
                            break
                        failed += 1
                        done += 1
                        record.update(status="failed", error=str(exc))
                        self._batch_log(f"[{index}/{len(plan)}] 失败 {relative}: {exc}")
                        print("[batch] " + traceback.format_exc(), flush=True)
                        self._batch_set(done=done, succeeded=succeeded,
                                        failed=failed, skipped=skipped)
                        atomic_write_json(manifest_path, manifest)
                        continue

                    future = render_pool.submit(
                        self._batch_render_video, item, prepared, mode=mode,
                        pred_betas_mean=pred_betas_mean, pred_fov_mean=pred_fov_mean)
                    pending[future] = (item, prepared)

                for future in as_completed(tuple(pending)):
                    finish_future(future)

            if self._batch_cancel.is_set():
                for record in manifest["files"]:
                    if record["status"] == "pending":
                        record.update(status="cancelled", reason="任务在调度前取消")

            cancelled = self._batch_cancel.is_set()
            manifest["status"] = "cancelled" if cancelled else "completed"
            manifest["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            manifest["summary"] = {"total": len(plan), "done": done, "succeeded": succeeded,
                                   "failed": failed, "skipped": skipped}
            atomic_write_json(manifest_path, manifest)
            phase = "已取消" if cancelled else ("完成（未发现视频）" if not plan else "全部完成")
            self._batch_set(running=False, phase=phase, current=None, current_index=0,
                            file_done=0, file_total=0, done=done, succeeded=succeeded,
                            failed=failed, skipped=skipped, cancelled=cancelled,
                            manifest=str(manifest_path))
        except Exception as exc:  # noqa: BLE001
            self._batch_log(f"批量任务失败: {exc}")
            manifest["status"] = "failed"
            manifest["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            manifest["error"] = str(exc)
            try:
                atomic_write_json(manifest_path, manifest)
            except Exception:  # noqa: BLE001
                pass
            self._batch_set(running=False, phase="失败", error=str(exc), current=None,
                            manifest=(str(manifest_path) if manifest_path.exists() else None))
        finally:
            with self._batch_lock:
                self._batch_current_eid = None
                self._batch_active_eids.clear()
                self._batch_active.clear()
                self._batch["active"] = []

    def _release_item_cache(self, eid: int) -> None:
        """Drop per-video tensors after a batch item while keeping its stable eid registration."""
        self._raw.pop(eid, None)
        self._payload_gt.pop(eid, None)
        for cache in (self._pred, self._gtw, self._prw, self._payload, self._mp4):
            for key in list(cache):
                if key == eid or (isinstance(key, tuple) and key and key[0] == eid):
                    cache.pop(key, None)
        with self._prog_lock:
            self._prog.pop(eid, None)
            prefix = f"{eid}:"
            for key in list(self._prog2d):
                if str(key).startswith(prefix):
                    self._prog2d.pop(key, None)
        with self._cancel_lock:
            self._cancel.pop(eid, None)

    # ---- benchmark：多卡子进程跑量化评测，前端面板轮询进度/结果 ----
    def benchmark_status(self) -> dict:
        """前端轮询用快照。静态关键日志 + 固定状态槽，长跑不会持续增长刷屏。"""
        if self._bench_backend == "aliyun":
            return self._aliyun_bench.status()
        with self._bench_lock:
            b = self._bench
            log = b["log"][-16:] + list((b.get("log_slots") or {}).values())
            progress = _benchmark_progress_timing(
                b.get("progress") or {}, running=bool(b["running"]),
                started_at=b.get("started_at"), finished_at=b.get("finished_at"),
            )
            return {"running": b["running"], "phase": b["phase"], "count": b["count"],
                    "log": log, "report": b["report"],
                    "live_report": b.get("live_report"), "error": b["error"],
                    "out": b["out"], "ckpt_tag": b["ckpt_tag"],
                    "models": b.get("models") or [],
                    "active_model": b.get("active_model"),
                    "selection": b.get("selection") or {},
                    "progress": progress,
                    "backend": "local"}

    def benchmark_aliyun_defaults(self) -> dict:
        """Validated, non-secret defaults for the remote execution form."""
        return self._aliyun_bench.defaults()

    def benchmark_capabilities(self) -> dict:
        """能力清单(供前端三态选择网格):每个 head 的 required_gt + 是否实现;每个 dataset 的
        capability + 是否实现;以及当前 ckpt 的模型产出能力(按该 run 的 config 推断 hand/depth 头
        是否开启,不加载权重)。前端据此判每个「功能×数据集」格子为 可跑/待实现·待模型/不匹配。"""
        from benchmark import capabilities
        cfg = ckpts.config_for_ckpt(self.ckpt_path, self.config_path) if self.ckpt_path else self.config_path
        return capabilities(str(cfg) if cfg else None)

    def benchmark_sizes(self, force: bool = False) -> dict:
        """每数据集序列条数/总帧数快照。首访(或 force)起后台线程算一次并缓存(枚举 GT 稍慢,不阻塞
        面板);立即返回当前快照 {computing, sizes}。sizes=None 表示仍在算,前端可稍后再拉。"""
        with self._bench_lock:
            s = self._bench_sizes
            if s["data"] is not None and not force:
                return {"computing": s["computing"], "sizes": s["data"]}
            if s["computing"]:
                return {"computing": True, "sizes": s["data"]}
            s["computing"] = True
        threading.Thread(target=self._sizes_worker, daemon=True).start()
        return {"computing": True, "sizes": None}

    def _sizes_worker(self) -> None:
        try:
            from benchmark import dataset_sizes
            data = dataset_sizes()
        except Exception as e:  # noqa: BLE001  计数失败不影响面板其余功能
            data = {"_error": str(e)}
        with self._bench_lock:
            self._bench_sizes = {"computing": False, "data": data}

    def start_benchmark(self, *, datasets: str = "all", heads: str = "all",
                        max_seqs=None, max_frames=None, devices=None,
                        seq_start=0, seq_end=None, checkpoints=None,
                        dataset_selection=None, reuse_cache=True,
                        auto_ukf_best=True, backend="local", aliyun=None) -> dict:
        """Run the same benchmark for one or more checkpoints, one model at a time."""
        backend = str(backend or "local").strip().lower()
        if backend not in {"local", "aliyun"}:
            return {"ok": False, "error": "测评运行位置仅支持 local 或 aliyun"}
        with self._batch_lock:
            if backend == "local" and self._batch["running"]:
                return {"ok": False, "error": "批量推理正在运行，请结束后再启动 Benchmark"}
        if self._aliyun_bench.is_running():
            return {"ok": False, "error": "Aliyun 远程测评正在运行，请结束后再启动新测评"}
        with self._bench_lock:
            if self._bench["running"]:
                return {"ok": False, "error": "已有测评在运行，请等待或先取消"}
            if self._aliyun_bench.is_running():
                return {"ok": False, "error": "Aliyun 远程测评正在运行，请结束后再启动新测评"}
            requested = checkpoints
            if requested is None:
                requested = [{"ckpt": self.ckpt_path}] if self.ckpt_path else []
            if not isinstance(requested, (list, tuple)) or not requested:
                return {"ok": False, "error": "请至少添加一个 Benchmark 模型"}
            if len(requested) > 16:
                return {"ok": False, "error": "一次最多比较 16 个 Benchmark 模型"}
            models = []
            seen_ckpts = set()
            for item in requested:
                item = item if isinstance(item, dict) else {"ckpt": item}
                path = item.get("ckpt")
                resolved = ckpts.resolve_checkpoint_path(path)
                if resolved is None:
                    return {"ok": False, "error": f"Benchmark checkpoint 不存在: {path or '(空)'}"}
                key = str(resolved)
                if key in seen_ckpts:
                    continue
                seen_ckpts.add(key)
                run = str(item.get("run") or resolved.parent.name)
                step = str(item.get("step") or resolved.name)
                models.append({
                    "run": run, "step": step, "ckpt": key,
                    "tag": ckpts.ckpt_tag(key),
                    "label": str(item.get("label") or f"{run} / {step}"),
                    "status": "pending", "report": None, "out": None, "error": None,
                })
            if not models:
                return {"ok": False, "error": "请至少添加一个不重复的 Benchmark 模型"}
            try:
                seq_start = int(seq_start or 0)
                seq_end = None if seq_end in (None, "") else int(seq_end)
                if seq_start < 0 or (seq_end is not None and seq_end <= seq_start):
                    raise ValueError
            except (TypeError, ValueError):
                return {"ok": False, "error": "序列范围必须满足 0 <= start < end（end 可留空）"}
            from benchmark.cache import build_signature, find_cached_report, signature_key
            from benchmark.datasets.base import normalize_dataset_selection

            try:
                dataset_selection = normalize_dataset_selection(dataset_selection)
            except (TypeError, ValueError) as exc:
                return {"ok": False, "error": str(exc)}

            for model in models:
                cfg = ckpts.config_for_ckpt(model["ckpt"], self.config_path) or self.config_path
                if not cfg or not Path(cfg).is_file():
                    return {
                        "ok": False,
                        "error": f"找不到 {model['label']} 对应的模型 config: {cfg or '(空)'}",
                    }
                model["config"] = str(Path(cfg).resolve())
                model["model"] = str(self.model)
                signature = build_signature(
                    ckpt=model["ckpt"], config=model["config"], heads=heads,
                    datasets=datasets, seq_start=seq_start, seq_end=seq_end,
                    max_seqs=max_seqs, max_frames=max_frames,
                    dataset_selection=dataset_selection, hand_mode="hard",
                )
                model["benchmark_signature"] = signature
                model["benchmark_signature_key"] = signature_key(signature)
                hit = find_cached_report(signature) if bool(reuse_cache) else None
                if hit:
                    model.update(
                        status="completed", report=hit["report"], out=hit["out"],
                        report_path=hit["report_path"], cache_hit=True,
                        cache_manifest=hit["manifest_path"], error=None,
                    )
            requested_heads = {item.strip() for item in str(heads).split(",") if item.strip()}
            auto_ukf_enabled = bool(auto_ukf_best) and (
                heads == "all" or bool(requested_heads & {"hands", "hands_world", "hands_coverage"})
            )
            if auto_ukf_enabled:
                from benchmark.ranking import auto_ukf_placeholder, resolve_auto_ukf_model

                if all(model.get("status") == "completed" and model.get("report") for model in models):
                    models.append(resolve_auto_ukf_model(models, reuse_cache=bool(reuse_cache)))
                else:
                    models.append(auto_ukf_placeholder())
            cached_count = sum(bool(model.get("cache_hit")) for model in models)
            if backend == "aliyun":
                from benchmark.dist.aliyun import AliyunConfig, load_defaults
                raw_config = load_defaults().to_dict()
                if aliyun is not None:
                    if not isinstance(aliyun, dict):
                        return {"ok": False, "error": "aliyun 配置必须是对象"}
                    raw_config.update(aliyun)
                try:
                    remote_config = AliyunConfig.from_mapping(raw_config)
                except ValueError as exc:
                    return {"ok": False, "error": f"Aliyun 配置无效: {exc}"}
                result = self._aliyun_bench.start(
                    models=models, datasets=datasets, heads=heads,
                    max_seqs=max_seqs, max_frames=max_frames,
                    seq_start=seq_start, seq_end=seq_end,
                    dataset_selection=dataset_selection,
                    reuse_cache=bool(reuse_cache),
                    config=remote_config,
                )
                if result.get("ok"):
                    self._bench_backend = "aliyun"
                return result
            devs = [int(d) for d in (devices or [])] or [0]   # 去重保序,空则默认单卡 0
            seen = set(); devs = [d for d in devs if not (d in seen or seen.add(d))]
            self._bench_cancel.clear()
            self._bench_procs = []
            self._bench = {"running": True, "phase": "启动中…", "count": 0, "log": [],
                           "report": None, "error": None, "out": None,
                           "live_report": None, "log_slots": {},
                           "started_at": time.time(), "finished_at": None,
                           "ckpt_tag": models[0]["tag"] if len(models) == 1 else f"{len(models)} models",
                           "models": models, "active_model": None,
                           "selection": {"datasets": datasets, "heads": heads,
                                         "seq_start": seq_start, "seq_end": seq_end,
                                         "max_seqs": max_seqs, "max_frames": max_frames,
                                         "dataset_selection": dataset_selection,
                                         "reuse_cache": bool(reuse_cache),
                                         "auto_ukf_best": auto_ukf_enabled},
                           "progress": {"frac": 0.0, "suite_frac": 0.0,
                                        "model_index": 0, "model_total": len(models),
                                        "done": 0, "total": 0, "gpus": []}}
            self._bench_backend = "local"
        threading.Thread(target=self._bench_worker,
                         args=(models, datasets, heads, max_seqs, max_frames, devs,
                               seq_start, seq_end, dataset_selection, bool(reuse_cache),
                               auto_ukf_enabled),
                         daemon=True).start()
        return {"ok": True, "models": len(models), "cached": cached_count}

    def cancel_benchmark(self) -> dict:
        if self._bench_backend == "aliyun":
            return self._aliyun_bench.cancel()
        import os
        import signal
        with self._bench_lock:
            running = self._bench["running"]
            procs = list(self._bench_procs)
        if not running:
            return {"ok": False, "error": "当前无运行中的测评"}
        self._bench_cancel.set()
        for p in procs:                           # 杀整个进程组(子进程用 start_new_session 独立会话)
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:  # noqa: BLE001  已退出/无进程组则退回直接终止
                try:
                    p.terminate()
                except Exception:  # noqa: BLE001
                    pass
        with self._bench_lock:
            self._bench["phase"] = "取消中…（终止各卡子进程）"
        return {"ok": True}

    def benchmark_gpus(self) -> dict:
        """列出本机可用 GPU(供面板多选)。优先 nvidia-smi(带显存/利用率);失败回退 torch.cuda。"""
        import subprocess
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            gpus = []
            for ln in (out.stdout or "").strip().splitlines():
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) < 5:
                    continue
                gpus.append({"index": int(parts[0]), "name": parts[1],
                             "mem_used": int(float(parts[2])), "mem_total": int(float(parts[3])),
                             "util": int(float(parts[4]))})
            if gpus:
                return {"gpus": gpus, "source": "nvidia-smi"}
        except Exception:  # noqa: BLE001  无 nvidia-smi / 解析失败则回退 torch
            pass
        try:
            import torch
            n = torch.cuda.device_count()
            gpus = [{"index": i, "name": torch.cuda.get_device_name(i)} for i in range(n)]
            return {"gpus": gpus, "source": "torch"}
        except Exception as e:  # noqa: BLE001
            return {"gpus": [], "source": "none", "error": str(e)}

    def _bench_worker(self, models, datasets, heads, max_seqs, max_frames, devices,
                      seq_start=0, seq_end=None, dataset_selection=None,
                      reuse_cache=True, auto_ukf_best=False) -> None:
        """Run checkpoints sequentially so every model can use the full selected GPU set."""
        import json
        import time

        suite_base = (REPO_DIR / "output" / "eval" / "benchmark" /
                      time.strftime("%Y%m%d_%H%M%S"))
        try:
            suite_base.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            with self._bench_lock:
                self._bench.update(running=False, phase="异常", error=str(exc),
                                   active_model=None, finished_at=time.time())
            return
        results = [dict(model) for model in models]
        last_report = None
        errors = []
        total = len(results)

        for index, model in enumerate(results, 1):
            if self._bench_cancel.is_set():
                break
            if model.get("auto_select_ukf"):
                try:
                    from benchmark.ranking import resolve_auto_ukf_model

                    model = resolve_auto_ukf_model(
                        results[:index - 1], reuse_cache=bool(reuse_cache),
                    )
                    results[index - 1] = model
                    ranking = model.get("quality_ranking") or {}
                    with self._bench_lock:
                        self._bench["log"].append(
                            f"[auto-ukf] 选择 {model.get('source_model_label')} · "
                            f"平均归一化损失={ranking.get('score', 0):.4f} · "
                            f"有效指标={ranking.get('metrics', 0)}，追加 UKF+RTS 测评"
                        )
                        self._bench["models"] = results
                except Exception as exc:  # noqa: BLE001
                    model.update(status="failed", error=str(exc))
                    errors.append(f"UKF 自动优选: {exc}")
                    with self._bench_lock:
                        self._bench["models"] = results
                        self._bench["log"].append(f"[auto-ukf] 跳过：{exc}")
                    continue
            if model.get("cache_hit") and model.get("report"):
                last_report = model["report"]
                with self._bench_lock:
                    self._bench["log"].append(
                        f"[cache] 模型 {index}/{total} · {model['label']} 复用 {model.get('report_path')}"
                    )
                    self._bench.update(
                        models=results, active_model=None, report=last_report,
                        live_report=last_report, count=0,
                        progress={
                            "frac": 1.0, "suite_frac": index / total,
                            "model_index": index, "model_total": total,
                            "model_label": model["label"], "done": 0, "total": 0,
                            "gpus": [], "cache_hit": True,
                        },
                    )
                continue
            model["status"] = "running"
            active = {
                key: model.get(key) for key in (
                    "run", "step", "ckpt", "tag", "label", "status", "variant", "hand_mode",
                    "source_model_label",
                )
            }
            with self._bench_lock:
                self._bench.update(
                    models=results, active_model=active, report=None, live_report=None,
                    error=None, count=0, log_slots={},
                    progress={"frac": 0.0, "suite_frac": (index - 1) / total,
                              "model_index": index, "model_total": total,
                              "model_label": model["label"], "done": 0, "total": 0,
                              "gpus": []},
                )
            model_dir = suite_base / f"model_{index:02d}_{model['tag'][:80]}"
            try:
                outcome = self._bench_model_worker(
                    model["ckpt"], datasets, heads, max_seqs, max_frames, devices,
                    seq_start, seq_end, base=model_dir, model_index=index,
                    model_total=total, model_label=model["label"],
                    dataset_selection=dataset_selection,
                    hand_mode=model.get("hand_mode", "hard"),
                )
            except Exception as exc:  # noqa: BLE001
                outcome = {"status": "failed", "out": str(model_dir),
                           "report": None, "error": str(exc)}
            model.update(outcome)
            if model.get("status") == "completed" and model.get("report"):
                from benchmark.cache import register_cached_report

                report_path = Path(model.get("out") or "") / "report.json"
                model["report_path"] = str(report_path.resolve())
                try:
                    manifest = register_cached_report(
                        model["benchmark_signature"], report_path, model["report"],
                    )
                    model["cache_manifest"] = str(manifest)
                except Exception as exc:  # noqa: BLE001
                    with self._bench_lock:
                        self._bench["log"].append(f"[cache] 写入失败：{exc}")
            if model.get("report"):
                last_report = model["report"]
            if model.get("error"):
                errors.append(f"{model['label']}: {model['error']}")
            with self._bench_lock:
                self._bench["models"] = results
                self._bench["report"] = last_report
                self._bench["live_report"] = model.get("report") or self._bench.get("live_report")
            if self._bench_cancel.is_set():
                break

        if self._bench_cancel.is_set():
            for model in results:
                if model["status"] == "pending":
                    model["status"] = "cancelled"
            phase = "已取消（保留已完成模型结果）"
        elif errors:
            phase = f"完成（{len(errors)} 个模型失败）"
        else:
            phase = "完成"
        comparison = {
            "selection": {"datasets": datasets, "heads": heads,
                          "seq_start": seq_start, "seq_end": seq_end,
                          "max_seqs": max_seqs, "max_frames": max_frames,
                          "dataset_selection": dataset_selection or {},
                          "auto_ukf_best": bool(auto_ukf_best)},
            "models": results,
        }
        try:
            from benchmark.cache import publish_step_benchmark_logs

            log_paths = publish_step_benchmark_logs(results, comparison["selection"])
            with self._bench_lock:
                self._bench["log"].extend(f"[benchmark-log] {path}" for path in log_paths)
        except Exception as exc:  # noqa: BLE001
            with self._bench_lock:
                self._bench["log"].append(f"[benchmark-log] 写入失败：{exc}")
        comparison_path = suite_base / "comparison.json"
        try:
            comparison_path.write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"保存 comparison.json 失败: {exc}")
            phase = phase.replace("完成", "完成（汇总文件写入失败）", 1)
        completed = sum(model["status"] == "completed" for model in results)
        with self._bench_lock:
            progress = dict(self._bench.get("progress") or {})
            progress.update(suite_frac=(completed / total if total else 0.0),
                            model_index=min(completed + 1, total), model_total=total)
            self._bench.update(
                running=False, phase=phase, active_model=None, models=results,
                report=last_report, live_report=last_report, out=str(suite_base),
                error="；".join(errors) if errors and last_report is None else None,
                progress=progress, finished_at=time.time(),
            )

    def _bench_model_worker(self, ckpt, datasets, heads, max_seqs, max_frames, devices,
                            seq_start=0, seq_end=None, *, base=None, model_index=1,
                            model_total=1, model_label="", dataset_selection=None,
                            hand_mode="hard") -> dict:
        """多卡分片评测:每张选中卡起一个子进程(CUDA_VISIBLE_DEVICES 隔离)跑 run.py 的一个分片,
        各自加载模型、按共享输入分组并均衡帧数；逐行读子进程 stdout 聚合「已完成/总数」粗粒度进度,
        全部结束后 dist.aggregate 合并各分片 report → 面板读合并后的 report.json。"""
        import json
        import os
        import re
        import subprocess
        import sys
        import time

        def _log(line: str, slot: str | None = None):
            with self._bench_lock:
                if slot is not None:
                    self._bench.setdefault("log_slots", {})[slot] = line
                    return
                self._bench["log"].append(line)
                if len(self._bench["log"]) > 32:
                    self._bench["log"] = self._bench["log"][-32:]

        prefix = f"模型 {model_index}/{model_total} · {model_label} · "

        def _set(**kw):
            if kw.get("phase"):
                kw["phase"] = prefix + str(kw["phase"])
            with self._bench_lock:
                self._bench.update(kw)

        cfg = ckpts.config_for_ckpt(ckpt, self.config_path) or ""   # 优先该 run 自带 config，与训练一致
        run_py = str(REPO_DIR / "eval" / "model_effect" / "benchmark" / "run.py")
        base = Path(base) if base is not None else (
            REPO_DIR / "output" / "eval" / "benchmark" / time.strftime("%Y%m%d_%H%M%S"))
        n_dev = len(devices)

        # 进度聚合状态（_prog_lock 保护）：
        #   per_gpu[i]      本卡总体 {index,done,total}
        #   ds_order        数据集展示顺序（取首个分片的 [DSINIT]）
        #   per_ds[ds][i]   本卡该数据集 {done,total}（跨卡求和 = 该集综合进度）
        #   ds_done[ds]     已完成该集的分片序号集合（满 n_dev → 该集全卡完成）
        #   ds_report[ds]   该集全卡完成后合并出的局部 report {ckpt,config,heads}，供前端展开
        #   ds_t0[ds]/ds_t1[ds]  该集墙钟起止(monotonic):起=最早那卡首个进度,止=最后一卡完成。
        #                        ETA 用「已用时 / 已完成条数 × 剩余条数」估该集从开始到评完的总时长。
        per_gpu: dict[int, dict] = {}
        ds_order: list = []
        per_ds: dict[str, dict] = {}
        ds_done: dict[str, set] = {}
        ds_report: dict[str, dict] = {}
        ds_t0: dict[str, float] = {}
        ds_t1: dict[str, float] = {}
        live_rows: dict[int, dict] = {i: {} for i in range(n_dev)}
        live_revision = 0
        _prog_lock = threading.Lock()

        def _refresh_progress():
            now = time.monotonic()
            with _prog_lock:
                gpus = [dict(per_gpu[i]) for i in sorted(per_gpu)]
                order = list(ds_order)
                dsets = {}
                for ds in order:
                    shards = per_ds.get(ds, {})
                    dd = sum(v.get("done", 0) for v in shards.values())
                    dt = sum(v.get("total", 0) for v in shards.values())
                    done_all = len(ds_done.get(ds, set())) >= n_dev
                    status = "done" if done_all else ("running" if shards else "pending")
                    # 该集已用时 + 预计总时长(平均每条 × 全部条数);完成则用实际墙钟跨度。
                    t0 = ds_t0.get(ds)
                    elapsed = eta = None
                    if t0 is not None:
                        elapsed = (ds_t1.get(ds, now)) - t0
                        if done_all:
                            eta = elapsed                     # 已完成:预计=实际总耗时
                        elif dd > 0 and dt > 0:
                            eta = elapsed * dt / dd            # 平均每条 × 总条数 = 从开始到评完预计
                    dsets[ds] = {"done": dd, "total": dt, "status": status,
                                 "report": ds_report.get(ds),
                                 "elapsed_s": elapsed, "eta_s": eta}
            done = sum(g.get("done", 0) for g in gpus)
            total = sum(g.get("total", 0) for g in gpus)
            with self._bench_lock:
                self._bench["count"] = done
                frac = done / total if total else 0.0
                self._bench["progress"] = {
                    "frac": frac,
                    "suite_frac": ((model_index - 1) + frac) / model_total,
                    "model_index": model_index, "model_total": model_total,
                    "model_label": model_label,
                    "done": done, "total": total, "gpus": gpus,
                    "ds_order": order, "datasets": dsets,
                }

        def _merge_ds(ds):
            """某数据集全卡完成：合并各卡 _ds/<ds>.json（结构同 report）→ 局部 report，供前端展开。"""
            from benchmark.dist.aggregate import merge_reports
            dicts = []
            for dev in devices:
                fp = base / f"gpu{dev}" / "_ds" / f"{ds}.json"
                if fp.is_file():
                    try:
                        with open(fp, encoding="utf-8") as f:
                            dicts.append(json.load(f))
                    except Exception:  # noqa: BLE001
                        pass
            return merge_reports(dicts) if dicts else {"heads": {}}

        _SHARD_RE = re.compile(r"\[SHARD\]\s+done=(\d+)\s+total=(\d+)")
        _DS_RE = re.compile(r"\[DS\]\s+(.+?)\|(\d+)\|(\d+)")

        def _update_live_report(shard: int, row: dict):
            nonlocal live_revision
            required = {"head", "dataset", "seq_id", "status"}
            if not required.issubset(row):
                return
            key = (str(row["head"]), str(row["dataset"]), str(row["seq_id"]))
            with _prog_lock:
                live_rows[shard][key] = row
                live_revision += 1
                revision = live_revision
                snapshot = {idx: dict(rows) for idx, rows in live_rows.items()}
            report = _merge_live_results(
                snapshot, ckpt=str(ckpt), config=str(cfg),
                selection={"seq_start": seq_start, "seq_end": seq_end,
                           "max_frames": max_frames,
                           "dataset_selection": dataset_selection or {},
                           "hand_mode": str(hand_mode or "hard")},
            )
            with _prog_lock:
                if revision != live_revision:
                    return
                with self._bench_lock:
                    self._bench["live_report"] = report

        def _handle_line(i: int, dev: int, line: str):
            if line.startswith("[LIVE] "):
                try:
                    event = json.loads(line[len("[LIVE] "):])
                except json.JSONDecodeError:
                    _log(f"[gpu{dev}] 实时进度事件解析失败", slot=f"gpu{dev}:event")
                    return
                with _prog_lock:
                    gpu = per_gpu.setdefault(i, {"index": dev, "done": 0, "total": 0})
                    gpu["live"] = event
                _refresh_progress()
                return
            if line.startswith("[RESULT] "):
                try:
                    _update_live_report(i, json.loads(line[len("[RESULT] "):]))
                except json.JSONDecodeError:
                    _log(f"[gpu{dev}] 实时结果事件解析失败", slot=f"gpu{dev}:event")
                return
            if line.startswith("[DSINIT] ds="):
                names = [x for x in line[len("[DSINIT] ds="):].split(",") if x]
                with _prog_lock:
                    if not ds_order:                 # 各分片顺序一致，取首个即可
                        ds_order.extend(names)
                _refresh_progress()
                return
            if line.startswith("[DSDONE] ds="):
                ds = line[len("[DSDONE] ds="):].strip()
                with _prog_lock:
                    ds_done.setdefault(ds, set()).add(i)
                    ds_t0.setdefault(ds, time.monotonic())
                    full = len(ds_done[ds]) >= n_dev and ds not in ds_report
                    if full:
                        ds_t1[ds] = time.monotonic()
                if full:
                    rep = _merge_ds(ds)
                    with _prog_lock:
                        ds_report[ds] = rep
                _refresh_progress()
                return
            m = _SHARD_RE.search(line)
            if m:
                with _prog_lock:
                    gpu = per_gpu.setdefault(i, {"index": dev})
                    gpu.update(done=int(m.group(1)), total=int(m.group(2)))
                _refresh_progress()
                return
            md = _DS_RE.search(line)
            if md:
                with _prog_lock:
                    per_ds.setdefault(md.group(1), {})[i] = {
                        "done": int(md.group(2)), "total": int(md.group(3)),
                    }
                    ds_t0.setdefault(md.group(1), time.monotonic())
                _refresh_progress()
                return
            if line.startswith("[report]"):
                return
            prefix = re.match(r"^\[([^]]+)\]", line)
            if prefix:
                _log(f"[gpu{dev}] {line}", slot=f"gpu{dev}:{prefix.group(1)}")
            elif "error" in line.lower() or "exception" in line.lower():
                _log(f"[gpu{dev}] {line}")

        def _reader(i: int, dev: int, proc, log_path: Path):
            """Parse control events and overwrite per-GPU status slots instead of appending."""
            with log_path.open("a", encoding="utf-8") as log_file:
                for raw in iter(proc.stdout.readline, ""):
                    log_file.write(raw)
                    log_file.flush()
                    for part in re.split(r"[\r\n]+", _ANSI.sub("", raw)):
                        line = part.strip()
                        if line:
                            _handle_line(i, dev, line)
            try:
                proc.stdout.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            for i, dev in enumerate(devices):        # 每卡初始化进度占位，面板一开始就显示各卡 0/0
                per_gpu[i] = {"index": dev, "done": 0, "total": 0}
            _set(phase=f"启动 {len(devices)} 卡子进程…（各卡加载模型）")
            _log(f"[viewer] 多卡分片评测 ckpt={ckpt} 卡={devices} 输出={base}")
            procs, readers = [], []
            for i, dev in enumerate(devices):
                out_i = base / f"gpu{dev}"
                out_i.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, run_py, "--ckpt", str(ckpt), "--config", str(cfg),
                       "--model", str(self.model), "--heads", str(heads), "--datasets", str(datasets),
                       "--shard-index", str(i), "--shard-count", str(len(devices)),
                       "--out", str(out_i), "--hand-mode", str(hand_mode or "hard")]
                selected_datasets = {item.strip() for item in str(datasets).split(",")}
                selected_heads = {item.strip() for item in str(heads).split(",")}
                protocol_only = (
                    bool(selected_datasets)
                    and all(name.endswith("_hand_coverage") for name in selected_datasets)
                    and selected_heads == {"hands_coverage"}
                )
                if not protocol_only:
                    # 长序列沿用分窗；覆盖率指标的 81 帧固定片段必须整段单次前向。
                    cmd.append("--windowed")
                if max_seqs:
                    cmd += ["--max-seqs", str(int(max_seqs))]
                if seq_start:
                    cmd += ["--seq-start", str(int(seq_start))]
                if seq_end is not None:
                    cmd += ["--seq-end", str(int(seq_end))]
                if max_frames:
                    cmd += ["--max-frames", str(int(max_frames))]
                if dataset_selection:
                    cmd += ["--dataset-selection-json", json.dumps(
                        dataset_selection, ensure_ascii=True, separators=(",", ":")
                    )]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(dev)
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                _log(f"[gpu{dev}] 启动 shard {i + 1}/{len(devices)}",
                     slot=f"gpu{dev}:launch")
                p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True)
                procs.append(p)
                t = threading.Thread(
                    target=_reader,
                    args=(i, dev, p, out_i / "process.log"),
                    daemon=True,
                )
                t.start()
                readers.append(t)
            with self._bench_lock:
                self._bench_procs = list(procs)
            _set(phase="评测中…（各卡并行分片）")
            for p in procs:                          # 等全部子进程结束
                p.wait()
            for t in readers:                        # 等读线程收尾（读完剩余 stdout）
                t.join(timeout=10)
            rcs = [p.returncode for p in procs]
            _log(f"[viewer] 各卡子进程结束 returncode={rcs}，开始合并分片")
            _set(phase="合并结果中…（各卡分片）")   # 序列已全评完(进度 100%)，别再显示「评测中」
            merged, shard_error = _aggregate_benchmark_shards(
                base,
                dict(zip(devices, rcs)),
                cancelled=self._bench_cancel.is_set(),
            )
            if merged is None:
                return {"status": "cancelled", "out": str(base),
                        "report": None, "error": None}
            report = None
            jp = base / "report.json"
            if jp.is_file():
                with open(jp, encoding="utf-8") as f:
                    report = json.load(f)
            else:
                report = merged
            status = (
                "cancelled" if self._bench_cancel.is_set()
                else ("failed" if shard_error else "completed")
            )
            return {"status": status, "out": str(base), "report": report,
                    "error": shard_error or (None if report else "评测结束但未找到 report.json")}
        except Exception as e:  # noqa: BLE001
            import traceback
            tb = traceback.format_exc()
            print("[benchmark] 异常:\n" + tb, flush=True)      # 完整堆栈落服务端控制台
            for ln in tb.rstrip().splitlines():                # 完整堆栈也进面板「运行日志」，便于定位到具体文件:行
                _log("[error] " + ln)
            _set(phase="异常", error=str(e))
            return {"status": "cancelled" if self._bench_cancel.is_set() else "failed",
                    "out": str(base), "report": None, "error": str(e)}
        finally:
            with self._bench_lock:
                self._bench_procs = []

    # ---- log_diff：子进程对比两个 run，前端只接收配置和代码变更 ----
    @staticmethod
    def _resolve_logdiff_run(run: str) -> Path | None:
        if not run:
            return None
        d = (MODEL_TRAIN_ROOT / run).resolve()
        root = MODEL_TRAIN_ROOT.resolve()
        if (d == root or root not in d.parents) or not d.is_dir():
            return None
        return d

    def logdiff_scopes(self, run_a: str, run_b: str) -> dict:
        """Return top-level code snapshot modules available in either run."""
        da = self._resolve_logdiff_run(run_a)
        db = self._resolve_logdiff_run(run_b)
        if da is None or db is None:
            return {"ok": False, "error": "run_a/run_b 无效或不在 model_train 内", "scopes": []}
        scopes = set()
        for run_dir in (da, db):
            code_dir = run_dir / "logs" / "record" / "code"
            if not code_dir.is_dir():
                continue
            scopes.update(p.name for p in code_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
        return {"ok": True, "scopes": sorted(scopes, key=str.lower)}

    def logdiff_status(self) -> dict:
        """前端轮询用快照。log 只回尾部若干行。"""
        with self._logdiff_lock:
            b = self._logdiff
            return {"running": b["running"], "phase": b["phase"], "result": b["result"],
                    "error": b["error"], "run_a": b["run_a"], "run_b": b["run_b"],
                    "code_scope": b.get("code_scope", ""),
                    "out": b["out"], "log": b["log"][-40:]}

    def start_logdiff(self, run_a: str, run_b: str, code_scope: str = "") -> dict:
        """起后台线程跑 log_diff 对比。已有对比在跑、或 run 缺失/非法则拒绝。run_a/run_b 为相对 model_train 的 run 名。"""
        da = self._resolve_logdiff_run(run_a)
        db = self._resolve_logdiff_run(run_b)
        if da is None or db is None:
            return {"ok": False, "error": "run_a/run_b 无效或不在 model_train 内"}
        if da == db:
            return {"ok": False, "error": "两个 run 相同，无可对比"}
        code_scope = str(code_scope or "").strip().replace("\\", "/").strip("/")
        if code_scope:
            parts = code_scope.split("/")
            available = set(self.logdiff_scopes(run_a, run_b).get("scopes", []))
            if any(part in ("", ".", "..") for part in parts) or code_scope not in available:
                return {"ok": False, "error": f"代码模块无效或不在快照中: {code_scope}"}
        with self._logdiff_lock:
            if self._logdiff["running"]:
                return {"ok": False, "error": "已有对比在运行，请等待完成"}
            self._logdiff = {"running": True, "phase": "启动中…", "result": None, "error": None,
                             "run_a": run_a, "run_b": run_b, "code_scope": code_scope,
                             "out": None, "log": []}
        threading.Thread(target=self._logdiff_worker, args=(str(da), str(db), code_scope),
                         daemon=True).start()
        return {"ok": True}

    def _logdiff_worker(self, dir_a: str, dir_b: str, code_scope: str = "") -> None:
        """子进程跑 tools/log_diff.py --input A B --out <cache/logdiff/ts>，完成后读回 result.json。"""
        import json
        import subprocess
        import sys
        import time

        def _set(**kw):
            with self._logdiff_lock:
                self._logdiff.update(kw)

        def _log(line: str):
            with self._logdiff_lock:
                self._logdiff["log"].append(line)
                if len(self._logdiff["log"]) > 200:
                    self._logdiff["log"] = self._logdiff["log"][-200:]

        script = REPO_DIR / "tools" / "log_diff.py"
        out = self.cache_dir / "logdiff" / time.strftime("%Y%m%d_%H%M%S")
        cmd = [sys.executable, str(script), "--input", dir_a, dir_b,
               "--changes-only", "--out", str(out)]
        if code_scope:
            cmd.extend(["--code-scope", code_scope])
        try:
            _set(phase="对比中…（配置 + 代码）")
            _log("[viewer] " + " ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            for ln in (proc.stdout or "").splitlines():
                _log(ln)
            jp = out / "result.json"
            if not jp.is_file():
                err = (proc.stderr or proc.stdout or "log_diff 未产出 result.json").strip()
                _set(running=False, phase="失败", error=err[-1000:])
                return
            with open(jp, encoding="utf-8") as f:
                result = json.load(f)
            result = {k: result[k] for k in ("config", "code", "code_note", "code_scope") if k in result}
            _set(running=False, phase="完成", result=result, out=str(out), error=None)
        except subprocess.TimeoutExpired:
            _set(running=False, phase="超时", error="log_diff 运行超过 600s 被终止")
        except Exception as e:  # noqa: BLE001
            import traceback
            _log("[error] " + "".join(traceback.format_exception_only(type(e), e)).strip())
            _set(running=False, phase="异常", error=str(e))

    # ---- 模型/资产后台懒加载：网页先起，慢慢加载；首次推理前 ensure 阻塞等就绪 ----
    def ensure_assets(self) -> None:
        """确保 MANO 权重就绪（幂等、线程安全）。payload 一开头就调，供 GT/PRED 世界系解算用。"""
        if self._assets_ready.is_set():
            return
        with self._assets_lock:
            if not self._assets_ready.is_set():
                from ..reproj_core import mano
                mano.ensure_mano_weights()
                self._assets_ready.set()

    def predictor_ready(self) -> bool:
        return self._predictor_ready.is_set()

    def predictor_devices(self) -> list[str]:
        if self.predictor is None:
            return []
        return list(getattr(self.predictor, "parallel_device_names", []))

    def predictor_full_max_frames(self) -> int | None:
        if self.predictor is None:
            return None
        value = getattr(self.predictor, "full_max_frames", None)
        return None if value is None else int(value)

    def loader_active(self) -> bool:
        """是否有后台加载线程在跑（供前端区分「加载中」与「未加载/空闲」；只读 bool 不加锁）。"""
        return self._loading

    def _ensure_loader_locked(self) -> None:
        """（须持 _load_cv/锁）没有在跑的加载线程就起一个。加载线程做重活，不持锁。"""
        if not self._loading:
            self._loading = True
            threading.Thread(target=self._load_worker, name="model-load", daemon=True).start()

    def _load_worker(self) -> None:
        """后台单加载线程：循环加载「当前最新代」的模型。重活（build 4.6G / reload）在锁外做，
        只在锁内快照目标与提交结果 → swap_ckpt 期间不被阻塞。提交时若已非最新代（加载途中又切了别的）
        则丢弃本次结果、继续加载最新代。torch 构建无法中途打断，故在途这次会先跑完再被丢弃。"""
        while True:
            with self._load_cv:
                if self._predictor_ready.is_set():   # 已就绪且无新请求 → 收工
                    self._loading = False
                    return
                gen = self._load_gen
                boot_cfg, ckpt, window = self._boot_cfg, self.ckpt_path, self.window
                full_max_frames = self.full_max_frames
                reload_ckpt, base = self._reload_ckpt, self.predictor
            # ---- 重活：锁外执行，期间可自由 swap_ckpt ----
            try:
                if (reload_ckpt is not None and base is not None
                        and getattr(base, "supports_weight_reload", True)):
                    # 架构不变且权重格式可重载：只换权重，省重建。
                    missing, _ = base.reload(reload_ckpt)
                    if missing:  # 新 ckpt 未覆盖全部权重 → 就地重载会残留旧权重 → 改整重建以确保正确
                        print(f"[model] 就地重载缺 {len(missing)} 键，改整模型重建以保证权重正确", flush=True)
                        from inference.registry import get_predictor
                        new_pred = get_predictor(
                            self.model, config=boot_cfg, ckpt=ckpt, window=window,
                            full_max_frames=full_max_frames,
                            # base 已占着自动选中的卡，重建时沿用这些逻辑设备，不能重新判成“忙”。
                            devices=getattr(base, "parallel_device_names", self.inference_devices),
                            compile_mode=self.inference_compile_mode,
                            fp8_mode=self.inference_fp8_mode,
                        )
                    else:
                        new_pred = base
                else:
                    from inference.registry import get_predictor
                    print(f"[model] 后台加载模型：model={self.model} cfg={boot_cfg} ckpt={ckpt}", flush=True)
                    new_pred = get_predictor(
                        self.model, config=boot_cfg, ckpt=ckpt, window=window,
                        full_max_frames=full_max_frames,
                        devices=self.inference_devices,
                        compile_mode=self.inference_compile_mode,
                        fp8_mode=self.inference_fp8_mode,
                    )
            except Exception as e:  # noqa: BLE001
                with self._load_cv:
                    if gen == self._load_gen:        # 仍是最新代才算失败：记异常、唤醒等待者、收工
                        self._load_err = e
                        self._loading = False
                        self._load_cv.notify_all()
                        print(f"[model] 模型加载失败: {e}", flush=True)
                        return
                    continue                          # 过时的失败：最新代想要别的 → 重试最新代
            # ---- 提交：仅当仍是最新代 ----
            with self._load_cv:
                if gen == self._load_gen:
                    self.predictor = new_pred
                    self._reload_ckpt = None
                    self._load_err = None
                    self._predictor_ready.set()
                    self._loading = False
                    self._load_cv.notify_all()
                    print(f"[model] 模型加载完成，可推理：ckpt={ckpt} "
                          f"设备={new_pred.parallel_device_names}", flush=True)
                    return
                stale = new_pred if new_pred is not base else None   # reload 就地改了 base 无需丢
            if stale is not None:                     # rebuild 出的模型已过时：释放显存
                del stale
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            print(f"[model] 上一次加载(ckpt={ckpt})已被更新的选择取代，丢弃并加载最新选择", flush=True)

    def ensure_predictor(self):
        """确保「最新代」predictor 就绪（供 benchmark / --preload 阻塞等待、按需触发加载）。本函数不持锁做重活：
        无加载线程则起一个，然后等就绪；加载失败则抛出异常（与旧行为一致，pred 侧上报 500）。
        注意：viewer 前台推理**不**走此函数隐式加载（加载与推理已拆开，见 start_load / require_predictor）。"""
        if self._predictor_ready.is_set():
            return self.predictor
        with self._load_cv:
            self._ensure_loader_locked()
            while not self._predictor_ready.is_set():
                if self._load_err is not None:
                    raise self._load_err
                self._load_cv.wait()
            return self.predictor

    def start_load(self) -> dict:
        """显式启动后台加载「当前选中 ckpt」的模型（幂等、立即返回，不阻塞）。
        选 ckpt 不再自动加载 → 由前端「加载模型」按钮调此。就绪后前端据 /api/model_ready 显示。"""
        with self._load_cv:
            if self._predictor_ready.is_set():
                return {"ok": True, "ready": True}
            if self._load_err is not None:      # 清掉上次失败，允许重试
                self._load_err = None
            self._ensure_loader_locked()
            self._load_cv.notify_all()
        return {"ok": True, "ready": False}

    def require_predictor(self):
        """viewer 前台推理专用：**不触发加载、不阻塞等待**。模型未就绪即抛清晰错误（加载与推理拆开：
        未加载须先点 [加载模型]，加载中禁止推理）。就绪则返回 predictor。"""
        if self._predictor_ready.is_set():
            return self.predictor
        if self.loader_active():
            raise RuntimeError("模型加载中，请等待模型就绪后再推理")
        raise RuntimeError("模型未加载：请先选择模型并点击 [加载模型]")

    # ---- 条目懒登记：LeRobot / 裸视频 / Benchmark 序列共用 eid 空间 ----
    def n_items(self) -> int:
        """已登记（浏览过并被选取/触及）的条目数；懒增长。"""
        return len(self._items)

    def _register(self, key, item: dict) -> int:
        """归一 key → eid：首见即追加 item 并分配 eid，之后稳定复用。线程安全。"""
        with self._items_lock:
            eid = self._item_index.get(key)
            if eid is None:
                eid = len(self._items)
                self._items.append(item)
                self._item_index[key] = eid
            return eid

    def video_eid(self, path_abs: str) -> int:
        """裸视频绝对路径 → eid（无真值项）。浏览目录时给每个视频登记，避免启动全盘扫描。"""
        path_abs = str(Path(path_abs).resolve())
        return self._register(("video", path_abs),
                              {"kind": "video", "path": path_abs,
                               "fps": self.default_fps, "hand_frame": self.default_hand_frame})

    def ensure_dataset(self, ds_dir) -> dict:
        """定位/枚举某 lerobot 数据集并缓存 {"eps":[...], "fps":float}（幂等、线程安全、懒加载）。"""
        ds_dir = Path(ds_dir).resolve()
        key = str(ds_dir)
        with self._ds_lock:
            cached = self._ds_cache.get(key)
        if cached is not None:
            return cached
        from ..reproj_core import lerobot_io
        eps = lerobot_io.discover_episodes(ds_dir)
        try:
            import json
            info = json.loads((ds_dir / "meta" / "info.json").read_text(encoding="utf-8"))
            fps = float(info.get("fps") or 30.0)
        except Exception:  # noqa: BLE001
            fps = 30.0
        rec = {"eps": eps, "fps": fps}
        with self._ds_lock:
            self._ds_cache[key] = rec
        return rec

    def ensure_dataset_async(self, ds_dir) -> dict:
        """**非阻塞**：已缓存→立即 ready；否则起（或复用）后台枚举线程，进度供 dataset_progress 轮询。
        进入 lerobot 目录用此，避免同步阻塞在几十万 episode 的枚举上。返回当前快照。"""
        ds_dir = Path(ds_dir).resolve()
        key = str(ds_dir)
        with self._ds_lock:
            cached = self._ds_cache.get(key)
        if cached is not None:
            return {"ready": True, "episodes": len(cached["eps"]), "fps": cached["fps"], "stage": "done"}
        with self._dsprog_lock:
            prog = self._ds_progress.get(key)
            # 无记录、或上次出错 → 起一轮新枚举；进行中则复用（不重复起线程）。
            if prog is None or prog.get("error"):
                self._ds_progress[key] = {"ready": False, "stage": "start",
                                          "done": 0, "total": 0, "error": None}
                threading.Thread(target=self._enum_worker, args=(ds_dir, key),
                                 name="ds-enum", daemon=True).start()
            return dict(self._ds_progress[key])

    def _enum_worker(self, ds_dir: Path, key: str) -> None:
        """后台枚举：discover_episodes 带 on_step 回调更新进度；完成后落 _ds_cache 并标 ready。"""
        from ..reproj_core import lerobot_io

        def cb(stage, done, total):
            with self._dsprog_lock:
                p = self._ds_progress.get(key)
                if p is not None and not p.get("ready"):
                    p.update(stage=str(stage), done=int(done), total=int(total))
        try:
            eps = lerobot_io.discover_episodes(ds_dir, on_step=cb)
            try:
                import json
                info = json.loads((ds_dir / "meta" / "info.json").read_text(encoding="utf-8"))
                fps = float(info.get("fps") or 30.0)
            except Exception:  # noqa: BLE001
                fps = 30.0
            rec = {"eps": eps, "fps": fps}
            with self._ds_lock:
                self._ds_cache[key] = rec
            with self._dsprog_lock:
                self._ds_progress[key] = {"ready": True, "stage": "done", "episodes": len(eps),
                                          "fps": fps, "done": len(eps), "total": len(eps), "error": None}
        except Exception as e:  # noqa: BLE001
            print(f"[ds-enum] 枚举失败 {ds_dir}: {e}", flush=True)
            with self._dsprog_lock:
                self._ds_progress[key] = {"ready": False, "stage": "error",
                                          "done": 0, "total": 0, "error": str(e)}

    def dataset_progress(self, ds_dir) -> dict:
        """前端轮询：数据集枚举进度快照。已缓存→ready+episodes/fps；进行中→stage/done/total；未开始→idle。"""
        ds_dir = Path(ds_dir).resolve()
        key = str(ds_dir)
        with self._ds_lock:
            cached = self._ds_cache.get(key)
        if cached is not None:
            return {"ready": True, "episodes": len(cached["eps"]), "fps": cached["fps"], "stage": "done"}
        with self._dsprog_lock:
            p = self._ds_progress.get(key)
            return dict(p) if p else {"ready": False, "stage": "idle", "done": 0, "total": 0}

    def lerobot_eid(self, ds_dir, ep_idx: int) -> int:
        """(数据集, episode 序号) → eid（有真值项）。首访该数据集会 ensure_dataset 枚举其 episode。"""
        ds_dir = Path(ds_dir).resolve()
        rec = self.ensure_dataset(ds_dir)
        eps = rec["eps"]
        if not 0 <= ep_idx < len(eps):
            raise IndexError(f"episode 序号 {ep_idx} 越界（该数据集共 {len(eps)} 个）")
        return self._register(("lerobot", str(ds_dir), int(ep_idx)),
                              {"kind": "lerobot", "ds_dir": ds_dir,
                               "ep": eps[ep_idx], "ep_ordinal": int(ep_idx),
                               "episode_total": len(eps), "fps": rec["fps"]})

    # ---- 按 eid 取条目属性（替代旧的全局 no_truth/fps/hand_frame） ----
    def item(self, eid: int) -> dict:
        return self._items[eid]

    def is_no_truth(self, eid: int) -> bool:
        """Only LeRobot entries use the Viewer-specific GT rendering contract."""
        return self._items[eid]["kind"] != "lerobot"

    def item_fps(self, eid: int) -> float:
        return float(self._items[eid]["fps"])

    def item_hand_frame(self, eid: int) -> str:
        return self._items[eid].get("hand_frame", self.default_hand_frame)

    def item_context(self, eid: int) -> dict:
        """Return source identity for filenames and the UI without exposing cache internals."""
        item = self.item(eid)
        if item["kind"] == "video":
            source = Path(item["path"])
            return {"source_path": str(source), "source_name": source.name,
                    "episode_ordinal": None, "episode_total": None}
        if item["kind"] == "benchmark":
            return {
                "source_path": item["source_path"],
                "source_name": f"{item['dataset']} / {item['label']}",
                "episode_ordinal": None,
                "episode_total": None,
            }
        source = Path(item["ds_dir"])
        name = source.parent.name if source.name.lower() in {"lerobot", "lerobot_v3"} else source.name
        return {"source_path": str(source), "source_name": name,
                "episode_ordinal": item["ep_ordinal"], "episode_total": item["episode_total"]}

    def _item_cache_tag(self, eid: int) -> str:
        """Stable source tag so equal episode indices from different datasets never share an mp4."""
        context = self.item_context(eid)
        safe_name = _re.sub(r"[^0-9A-Za-z._-]+", "_", context["source_name"]).strip("_") or "input"
        source_hash = hashlib.sha256(context["source_path"].encode("utf-8")).hexdigest()[:10]
        return f"{safe_name[:48]}_{source_hash}"

    def set_prog(self, eid: int, **kw) -> None:
        with self._prog_lock:
            self._prog[eid] = {**self._prog.get(eid, {}), **kw}

    def get_prog(self, eid: int) -> dict:
        with self._prog_lock:
            return dict(self._prog.get(eid, {}))

    # ---- 2D 渲染进度：与 mp4/mp4_gt 的缓存键归一化一致，前端按块轮询 ----
    def prog2d_key(self, eid: int, mode: str, layout: str, content: str, raw: bool,
                   cam_mode: str = DEFAULT_CAM_MODE, hand_mode: str = DEFAULT_HAND_MODE,
                   pkey: str = "000") -> str:
        if raw:
            layout, content, cam_mode, hand_mode, pkey = "gt", "gt", "gt", "gt", "gt"
        elif self.is_no_truth(eid):
            layout, content = "overlay", "pred"
        return f"{eid}:{mode}:{layout}:{content}:{cam_mode}:{hand_mode}:{pkey}"

    def set_prog2d(self, key: str, done: int, total: int) -> None:
        with self._prog_lock:
            self._prog2d[key] = {"done": int(done), "total": int(total)}

    def get_prog2d(self, key: str) -> dict:
        with self._prog_lock:
            return dict(self._prog2d.get(key, {}))

    def set_mujoco_prog(self, key: tuple, **values) -> None:
        with self._prog_lock:
            self._mujoco_prog[key] = {**self._mujoco_prog.get(key, {}), **values}

    def get_mujoco_prog(self, key: tuple) -> dict:
        with self._prog_lock:
            return dict(self._mujoco_prog.get(key, {}))

    def set_retarget_prog(self, key: tuple, **values) -> None:
        with self._prog_lock:
            self._retarget_prog[key] = {
                **self._retarget_prog.get(key, {}), **values,
            }

    def get_retarget_prog(self, key: tuple) -> dict:
        with self._prog_lock:
            return dict(self._retarget_prog.get(key, {}))

    def set_world_prog(self, key: tuple, **values) -> None:
        with self._prog_lock:
            self._world_prog[key] = {
                **self._world_prog.get(key, {}), **values,
            }

    def get_world_prog(self, key: tuple) -> dict:
        with self._prog_lock:
            return dict(self._world_prog.get(key, {}))

    def swap_ckpt(self, ckpt: str) -> dict:
        """选到新 ckpt：**立即**作废依赖 ckpt 的缓存 + 置未就绪、记下最新目标（代次 +1），然后**立刻返回**。
        **不自动加载**——加载与推理已拆开，须由前端「加载模型」按钮调 start_load 显式触发。返回 {ckpt, tag, reload}。

        若加载线程已在跑（上一个 ckpt 正在加载）时又选别的模型：把代次再 +1、更新目标；在跑的加载线程
        本轮跑完后发现已非最新代 → 丢弃结果、直接去加载最新目标（torch 构建无法中途打断，故在途这次会先跑完）。
        GT 相关缓存（_raw/_gtw）与 ckpt 无关，保留。config/loss 面板同步切到新 run。"""
        with self._batch_start_lock:
            with self._batch_lock:
                if self._batch["running"]:
                    raise RuntimeError("批量推理正在运行，不能切换 checkpoint；请先取消或等待完成")
            cfg_used = ckpts.config_for_ckpt(ckpt, self.config_path)   # 优先该 run 自带 config
            # 架构是否与当前模型一致：一致则**就地重载权重**（快）、不重建；否则整模型重建。
            same_arch = False
            if self.predictor is not None and getattr(self.predictor, "_raw_mcfg", None) is not None:
                try:
                    import yaml
                    new_mcfg = (yaml.safe_load(open(cfg_used, encoding="utf-8")) or {}).get("model")
                    same_arch = (new_mcfg == self.predictor._raw_mcfg)
                except Exception:  # noqa: BLE001
                    same_arch = False
            can_reload = bool(
                same_arch
                and self.predictor is not None
                and getattr(self.predictor, "supports_weight_reload", True)
            )
            with self._load_cv:
                self._load_gen += 1                         # 代次 +1：作废在途加载的提交资格
                self.ckpt_path = ckpt
                self.ckpt_tag = ckpts.ckpt_tag(ckpt)
                self._boot_cfg = cfg_used
                self.loss_cfg = ckpts.load_loss_cfg(cfg_used)   # loss 面板跟随新 run
                self._pred.clear(); self._prw.clear(); self._payload.clear(); self._mp4.clear()
                self._world_mp4.clear(); self._world_prog.clear()
                self._mujoco_mp4.clear(); self._mujoco_prog.clear()
                self._retarget_mp4.clear(); self._retarget_prog.clear()
                if can_reload:
                    self._reload_ckpt = ckpt                # 架构不变：加载线程就地重载权重
                else:
                    old = self.predictor; self.predictor = None; self._reload_ckpt = None
                    if old is not None:
                        del old
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except Exception:  # noqa: BLE001
                            pass
                self._predictor_ready.clear()               # 未就绪：前端显示「未加载」，等手动点「加载模型」
                self._load_err = None
                # 不自动起加载线程（加载与推理拆开）；若已有在跑的加载线程，它会自行跟到最新代。
                if self._loading:
                    self._load_cv.notify_all()
            print(f"[ckpt] 选到 {ckpt}（{'就地重载权重' if can_reload else '重建模型'}，等待加载请求）", flush=True)
            return {"ckpt": ckpt, "tag": self.ckpt_tag, "reload": can_reload}

    def _key_lock(self, name: str) -> threading.Lock:
        with self._reg_lock:
            if name not in self._locks:
                self._locks[name] = threading.Lock()
            return self._locks[name]

    # ---- 分层缓存：raw → pred → world → payload ----
    @staticmethod
    def _read_image_sequence(paths, on_step=None, cancel_check=None) -> np.ndarray:
        """Load ordered RGB frames in parallel and normalize mixed resolutions."""
        import cv2

        paths = list(paths)
        if not paths:
            raise ValueError("Benchmark 序列没有输入帧")

        def _check_cancel():
            if cancel_check is not None and cancel_check():
                from inference.base import InferenceCancelled
                raise InferenceCancelled("图片序列加载已取消")

        def _read(path):
            _check_cancel()
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"读图失败: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            _check_cancel()
            return image

        images = []
        _check_cancel()
        with ThreadPoolExecutor(max_workers=min(8, len(paths)),
                                thread_name_prefix="benchmark-preview") as executor:
            for index, image in enumerate(executor.map(_read, paths), 1):
                _check_cancel()
                images.append(image)
                if on_step is not None:
                    on_step(index, len(paths))
        _check_cancel()
        height, width = images[0].shape[:2]
        for index, image in enumerate(images):
            _check_cancel()
            if image.shape[:2] != (height, width):
                images[index] = cv2.resize(
                    image, (width, height), interpolation=cv2.INTER_AREA)
        _check_cancel()
        return np.stack(images).astype(np.uint8)

    def raw(self, eid: int, cancel_check=None) -> dict:
        if eid not in self._raw:
            with self._key_lock(f"raw{eid}"):
                if eid not in self._raw:
                    from ..reproj_core import lerobot_io
                    # 抽帧进度回调（分批读逐批上报）：前端「加载数据 x/y 帧」+ 确定进度条。
                    cb = lambda d, t: self.set_prog(eid, stage="load", done=int(d), total=int(t))
                    it = self._items[eid]
                    if it["kind"] == "video":       # 裸视频：只抽帧，无 GT hands/相机
                        vp = it["path"]             # eid 由文件浏览时 video_eid 登记
                        frames = lerobot_io.read_video_frames(
                            vp, fps=it["fps"], max_frames=self.max_frames, on_step=cb,
                            cancel_check=cancel_check)
                        self._raw[eid] = {"episode_index": eid, "T": int(frames.shape[0]),
                                          "frames": frames, "cam_pose_enc": None,
                                          "video": vp, "hand_frame": it["hand_frame"]}
                    elif it["kind"] == "benchmark":
                        image_kwargs = {"on_step": cb}
                        if cancel_check is not None:
                            image_kwargs["cancel_check"] = cancel_check
                        frames = self._read_image_sequence(
                            it["image_paths"], **image_kwargs
                        )
                        self._raw[eid] = {
                            "episode_index": eid,
                            "T": int(frames.shape[0]),
                            "frames": frames,
                            "cam_pose_enc": None,
                            "video": None,
                            "dataset_root": str(REPO_DIR / "data" / "benchmark"),
                            "hand_frame": "camera",
                        }
                    else:
                        self._raw[eid] = lerobot_io.load_episode_raw(
                            it["ep"], max_frames=self.max_frames, on_step=cb,
                            cancel_check=cancel_check)
        return self._raw[eid]

    def pred(self, eid: int, cam_mode: str = DEFAULT_CAM_MODE,
             hand_mode: str = DEFAULT_HAND_MODE) -> dict:
        # full 没有手部拼窗，hard/blend 可共享；smooth 仍会执行后处理，必须独立缓存。
        is_smooth = str(hand_mode).startswith("smooth")
        effective_hand_mode = ("hard"
                               if cam_mode == "full" and not is_smooth
                               else hand_mode)
        k = (eid, cam_mode, effective_hand_mode)
        if k not in self._pred:
            with self._key_lock(f"pred{eid}:{cam_mode}:{effective_hand_mode}"):
                if k not in self._pred:
                    self.require_predictor()              # 加载与推理拆开：未就绪(未加载/加载中)直接报错，不隐式加载
                    self._raise_if_cancelled(eid)         # 排队期间被停止也及时退出
                    if cam_mode in {"streaming", "full"}:  # 单次整段阶段没有窗级进度，先显示明确状态
                        self.set_prog(eid, stage="stream" if cam_mode == "streaming" else "full")
                    cb = lambda done, total: self.set_prog(   # noqa: E731  逐窗上报前向进度(流式仅手部阶段回调)
                        eid, stage="infer", done=int(done), total=int(total))
                    self._pred[k] = self.predictor.predict(
                        self.raw(eid, cancel_check=lambda: self.cancelled(eid))["frames"],
                        on_step=cb,
                        cancel_check=lambda: self.cancelled(eid),   # 「停止」→ 逐窗中断
                        cam_mode=cam_mode, hand_mode=effective_hand_mode)
        return self._pred[k]

    def gt_world(self, eid: int, gt_betas_mean: bool = False) -> dict:
        k = (eid, gt_betas_mean)
        if k not in self._gtw:
            with self._key_lock(f"gtw{eid}:{int(gt_betas_mean)}"):
                if k not in self._gtw:
                    from ..render import compare
                    r = self.raw(eid)
                    # MANO GT 解网格/关节；kp21-only GT 直接从相机系点转到世界系骨架。
                    self._gtw[k] = compare.gt_to_world(r, betas_mean=gt_betas_mean)
        return self._gtw[k]

    def pred_world(self, eid: int, cam_mode: str = DEFAULT_CAM_MODE,
                   hand_mode: str = DEFAULT_HAND_MODE,
                   pred_betas_mean: bool = False) -> dict | None:
        k = (eid, cam_mode, hand_mode, pred_betas_mean)
        if k not in self._prw:
            with self._key_lock(
                    f"prw{eid}:{cam_mode}:{hand_mode}:{int(pred_betas_mean)}"):
                if k not in self._prw:
                    from ..reproj_core import geometry as geom
                    from ..render import compare
                    pred = self.pred(eid, cam_mode, hand_mode)
                    r = self.raw(eid)
                    if "hand" not in pred:
                        self._prw[k] = None
                    else:
                        # 预测手和预测相机必须共享同一个 world gauge；有无 GT 均只用 pred_c2w。
                        H, W = r["frames"].shape[1:3]
                        cam, _ = geom.decode_camera_pose_enc(pred["pose_enc"], H, W)
                        hf = self.item_hand_frame(eid) if self.is_no_truth(eid) else "camera"
                        self._prw[k] = compare.hands_to_world(
                            compare.pred_hand_to_schema(pred["hand"]), cam, hf,
                            betas_mean=pred_betas_mean)
        return self._prw[k]

    def payload(self, eid: int, cam_mode: str = DEFAULT_CAM_MODE,
                hand_mode: str = DEFAULT_HAND_MODE,
                gt_betas_mean: bool = False, pred_betas_mean: bool = False,
                pred_fov_mean: bool = False) -> dict:
        """{'gt': world_payload, 'pred': world_payload|None, 'nframes','fps','ep_idx'}。
        gt_betas_mean/pred_betas_mean/pred_fov_mean：手形/内参按每帧(False)或整段平均(True)。"""
        k = (eid, cam_mode, hand_mode, gt_betas_mean, pred_betas_mean, pred_fov_mean)
        if k not in self._payload:
            with self._key_lock(
                    f"pl{eid}:{cam_mode}:{hand_mode}:"
                    f"{int(gt_betas_mean)}{int(pred_betas_mean)}{int(pred_fov_mean)}"):
                if k not in self._payload:
                    from ..reproj_core import geometry as geom
                    from ..render import world as worldpl
                    fps = self.item_fps(eid)
                    self._raise_if_cancelled(eid)
                    self.ensure_assets()                                # MANO 权重（渲染/解算用）
                    self._raise_if_cancelled(eid)
                    self.set_prog(eid, stage="load", done=0, total=0)   # 读数据/抽帧
                    raw = self.raw(eid, cancel_check=lambda: self.cancelled(eid))
                    self._raise_if_cancelled(eid)                       # load 后取消点（前向前）
                    T, H, W = raw["frames"].shape[:3]
                    # 无真值：无 GT 世界系；只出预测（相机也用预测的）。
                    gt = (None if self.is_no_truth(eid) else worldpl.build_world_payload(
                        self.gt_world(eid, gt_betas_mean), raw["kept"], raw["cam_c2w"],
                        fps=fps))
                    self._raise_if_cancelled(eid)
                    prw = self.pred_world(eid, cam_mode, hand_mode, pred_betas_mean)
                    self._raise_if_cancelled(eid)
                    if prw is not None:
                        pred = self.pred(eid, cam_mode, hand_mode)
                        pc2w, _ = geom.decode_camera_pose_enc(
                            pred["pose_enc"], H, W, fov_mean=pred_fov_mean)
                        from ..render import compare
                        valid = compare.prediction_render_mask(pred, T)
                        pl = worldpl.build_world_payload(
                            prw, valid, pc2w, fps=fps)
                    else:
                        pl = None
                    self._raise_if_cancelled(eid)
                    self.set_prog(eid, stage="render3d")   # 解算 3D payload / 逐帧 loss
                    met = None                             # loss 逐帧定义，与手形/内参平均无关，保持每帧原值
                    metrics_error = None
                    if prw is not None and raw.get("cam_pose_enc") is not None:
                        met, metrics_error = _safe_frame_metrics(
                            raw, self.pred(eid, cam_mode, hand_mode), self.loss_cfg, (H, W),
                            geom.decode_camera_pose_enc)
                    self._raise_if_cancelled(eid)
                    try:                                    # 逐帧数值(每块视频下方面板用)，非核心，失败不阻断加载
                        from ..render import numbers as numberspl
                        # 数字表每帧+平均都列，与渲染的 betas/fov 开关无关，故不传 mode。
                        nums = numberspl.frame_numbers(
                            raw, self.pred(eid, cam_mode, hand_mode), geom.decode_camera_pose_enc)
                    except Exception as e:  # noqa: BLE001
                        print(f"[store] frame_numbers 失败(跳过数值面板): {e}", flush=True)
                        nums = None
                    self._raise_if_cancelled(eid)
                    self._payload[k] = {"gt": gt, "pred": pl, "nframes": int(T),
                                        "fps": fps, "ep_idx": raw["episode_index"],
                                        **self.item_context(eid),
                                        "metrics": met, "metrics_error": metrics_error,
                                        "nums": nums}
                    self.set_prog(eid, stage="done")
        return self._payload[k]

    def payload_gt(self, eid: int) -> dict:
        """「仅原始 GT」payload：只解 GT 世界系（手 + 相机轨迹），**不调用 predictor、不跑推理**。
        用于没 ckpt / 模型仍在后台加载 / 只想看原数据时快速查看。pred/metrics 恒为 None。"""
        if self.is_no_truth(eid):
            raise RuntimeError("裸视频项无 GT，无法「仅看原始」")
        if eid not in self._payload_gt:
            with self._key_lock(f"plgt{eid}"):
                if eid not in self._payload_gt:
                    from ..render import world as worldpl
                    fps = self.item_fps(eid)
                    self._raise_if_cancelled(eid)
                    self.ensure_assets()                                  # 只需 MANO，无需模型
                    self._raise_if_cancelled(eid)
                    self.set_prog(eid, stage="load", done=0, total=0)
                    raw = self.raw(eid, cancel_check=lambda: self.cancelled(eid))
                    self._raise_if_cancelled(eid)
                    T = len(raw["frames"])
                    gt = worldpl.build_world_payload(
                        self.gt_world(eid), raw["kept"], raw["cam_c2w"], fps=fps)
                    self._raise_if_cancelled(eid)
                    self._payload_gt[eid] = {"gt": gt, "pred": None, "nframes": T,
                                             "fps": fps, "ep_idx": raw["episode_index"],
                                             **self.item_context(eid),
                                             "metrics": None, "metrics_error": None,
                                             "raw_only": True}
                    self.set_prog(eid, stage="done")
        return self._payload_gt[eid]

    @staticmethod
    def world_video_key(eid: int, *, layout: str = "overlay",
                        views: dict | None = None,
                        coord_mode: str = "z_up",
                        show_traj: bool = True,
                        show_cam_hand: bool = True,
                        cam_mode: str = DEFAULT_CAM_MODE,
                        hand_mode: str = DEFAULT_HAND_MODE,
                        gt_betas_mean: bool = False,
                        pred_betas_mean: bool = False,
                        pred_fov_mean: bool = False,
                        raw: bool = False,
                        render_source: str = "both",
                        render_size: tuple[int, int] | None = None) -> tuple:
        from ..render.fixed_world_video import (
            CACHE_TAG, normalize_coord_mode, view_cache_tuple,
        )

        if raw:
            cam_mode, hand_mode = "gt", "gt"
            gt_betas_mean = pred_betas_mean = pred_fov_mean = False
            render_source = "gt"
        if render_source not in {"both", "gt", "pred"}:
            raise ValueError(f"未知固定世界数据源: {render_source}")
        size_key = None if render_size is None else tuple(map(int, render_size))
        return (
            int(eid), bool(raw), str(cam_mode), str(hand_mode),
            bool(gt_betas_mean), bool(pred_betas_mean), bool(pred_fov_mean),
            "side" if layout == "side" else "overlay",
            view_cache_tuple(views), normalize_coord_mode(coord_mode),
            bool(show_traj), bool(show_cam_hand), str(render_source), size_key,
            CACHE_TAG,
        )

    def world_video_progress(self, eid: int, **options) -> dict:
        key = self.world_video_key(eid, **options)
        if self._cached_world_video(key) is not None:
            return {"stage": "done", "done": 1, "total": 1}
        return self.get_world_prog(key)

    def _world_output_path(self, key: tuple) -> Path:
        (eid, raw, cam_mode, hand_mode, gt_betas_mean, pred_betas_mean,
         pred_fov_mean, layout, view_key, coord_mode,
         show_traj, show_cam_hand, render_source, render_size, tag) = key
        raw_data = self.raw(eid)
        mode_tag = "gt" if raw else (
            f"{self.ckpt_tag}_{cam_mode}_{hand_mode}_"
            f"{int(gt_betas_mean)}{int(pred_betas_mean)}{int(pred_fov_mean)}"
        )
        size_tag = ("default" if render_size is None
                    else f"{render_size[0]}x{render_size[1]}")
        view_digest = hashlib.sha256(repr(
            (layout, view_key, coord_mode, show_traj, show_cam_hand,
             render_source, size_tag, tag)).encode()
        ).hexdigest()[:16]
        return self.cache_dir / (
            f"{self.scene}_{self._item_cache_tag(eid)}_"
            f"ep{int(raw_data['episode_index']):03d}_fixed_world_"
            f"{mode_tag}_{render_source}_{size_tag}_{view_digest}.mp4"
        )

    def _cached_world_video(self, key: tuple) -> Path | None:
        cached = self._world_mp4.get(key)
        if cached is None:
            cached = self._world_output_path(key)
        if not cached.is_file() or cached.stat().st_size == 0:
            return None
        self._world_mp4[key] = cached
        return cached

    def world_video(self, eid: int, *, layout: str = "overlay",
                    views: dict | None = None,
                    coord_mode: str = "z_up",
                    show_traj: bool = True,
                    show_cam_hand: bool = True,
                    cam_mode: str = DEFAULT_CAM_MODE,
                    hand_mode: str = DEFAULT_HAND_MODE,
                    gt_betas_mean: bool = False,
                    pred_betas_mean: bool = False,
                    pred_fov_mean: bool = False,
                    raw: bool = False,
                    on_step=None,
                    render_source: str = "both",
                    render_size: tuple[int, int] | None = None) -> Path:
        """Render the fixed-world Canvas timeline only when explicitly requested."""
        options = {
            "layout": layout,
            "views": views,
            "coord_mode": coord_mode,
            "show_traj": show_traj,
            "show_cam_hand": show_cam_hand,
            "cam_mode": cam_mode,
            "hand_mode": hand_mode,
            "gt_betas_mean": gt_betas_mean,
            "pred_betas_mean": pred_betas_mean,
            "pred_fov_mean": pred_fov_mean,
            "raw": raw,
            "render_source": render_source,
            "render_size": render_size,
        }
        key = self.world_video_key(eid, **options)
        cached = self._cached_world_video(key)
        if cached is not None:
            return cached
        with self._key_lock("world-video:" + hashlib.sha256(repr(key).encode()).hexdigest()):
            cached = self._cached_world_video(key)
            if cached is not None:
                return cached
            payload = self.payload_gt(eid) if raw else self.payload(
                eid, cam_mode, hand_mode,
                gt_betas_mean, pred_betas_mean, pred_fov_mean)
            effective_source = "gt" if raw else render_source
            if effective_source == "gt":
                payload = {**payload, "pred": None}
            elif effective_source == "pred":
                payload = {**payload, "gt": None}
            frames = int(payload.get("nframes") or 0)
            output = self._world_output_path(key)
            self.set_world_prog(key, stage="queued", done=0, total=frames)
            try:
                from ..render.fixed_world_video import render_fixed_world_video
                self.set_world_prog(key, stage="render", done=0, total=frames)

                def report_step(done, total):
                    self.set_world_prog(
                        key, stage="render", done=int(done), total=int(total))
                    if on_step is not None:
                        on_step(int(done), int(total))

                render_fixed_world_video(
                    payload, output, fps=self.item_fps(eid),
                    layout=layout, views=views, coord_mode=coord_mode,
                    show_traj=show_traj, show_cam_hand=show_cam_hand,
                    size=(tuple(map(int, render_size))
                          if render_size is not None else (960, 540)),
                    on_step=report_step,
                )
                if not output.is_file() or output.stat().st_size == 0:
                    raise RuntimeError("固定世界视频编码失败，未生成有效 MP4")
            except Exception as exc:
                self.set_world_prog(key, stage="error", error=str(exc))
                raise
            self._world_mp4[key] = output
            self.set_world_prog(key, stage="done", done=frames, total=frames)
        return self._world_mp4[key]

    @staticmethod
    def mujoco_key(eid: int, source: str, cam_mode: str = DEFAULT_CAM_MODE,
                   hand_mode: str = DEFAULT_HAND_MODE,
                   betas_mean: bool = False,
                   fov_mean: bool = False,
                   render_size: tuple[int, int] | None = None) -> tuple:
        if source == "gt":
            cam_mode, hand_mode = "gt", "gt"
            fov_mean = False
        render_width, render_height = _robot_render_size(render_size)
        size_tag = (f"w{render_width}" if render_height is None
                    else f"w{render_width}h{render_height}")
        return (int(eid), str(source), str(cam_mode), str(hand_mode),
                bool(betas_mean), bool(fov_mean),
                f"shared_wuji_camera_v13_canonical_origin_{size_tag}")

    def mujoco_progress(self, eid: int, source: str,
                        cam_mode: str = DEFAULT_CAM_MODE,
                        hand_mode: str = DEFAULT_HAND_MODE,
                        betas_mean: bool = False,
                        fov_mean: bool = False) -> dict:
        key = self.mujoco_key(
            eid, source, cam_mode, hand_mode, betas_mean, fov_mean)
        if self._cached_mujoco_video(key) is not None:
            return {"stage": "done", "done": 1, "total": 1}
        return self.get_mujoco_prog(key)

    def _mujoco_output_path(self, key: tuple) -> Path:
        eid, source, cam_mode, hand_mode, betas_mean, fov_mean, tag = key
        raw = self.raw(eid)
        mode_tag = (
            f"gt_{int(betas_mean)}" if source == "gt" else
            f"{self.ckpt_tag}_{cam_mode}_{hand_mode}_"
            f"{int(betas_mean)}{int(fov_mean)}"
        )
        return self.cache_dir / (
            f"{self.scene}_{self._item_cache_tag(eid)}_"
            f"ep{int(raw['episode_index']):03d}_mujoco_{tag}_"
            f"{source}_{mode_tag}.mp4")

    def _cached_mujoco_video(self, key: tuple) -> Path | None:
        cached = self._mujoco_mp4.get(key)
        if cached is None:
            cached = self._mujoco_output_path(key)
        if not cached.is_file() or cached.stat().st_size == 0:
            return None
        self._mujoco_mp4[key] = cached
        return cached

    def mujoco_video(self, eid: int, source: str = "pred",
                     cam_mode: str = DEFAULT_CAM_MODE,
                     hand_mode: str = DEFAULT_HAND_MODE,
                     betas_mean: bool = False,
                     fov_mean: bool = False,
                     on_step=None,
                     render_size: tuple[int, int] | None = None) -> Path:
        """Render the selected GT/prediction through MuJoCo only when requested."""
        if source not in {"gt", "pred"}:
            raise ValueError(f"未知 MuJoCo 数据源: {source}")
        key = self.mujoco_key(
            eid, source, cam_mode, hand_mode, betas_mean, fov_mean, render_size)
        cached = self._cached_mujoco_video(key)
        if cached is not None:
            return cached
        with self._key_lock("mujoco:" + ":".join(map(str, key))):
            cached = self._cached_mujoco_video(key)
            if cached is not None:
                return cached
            raw = self.raw(eid)
            height, width = raw["frames"].shape[1:3]
            if source == "gt":
                if self.is_no_truth(eid):
                    raise RuntimeError("裸视频没有 GT，无法渲染 GT MuJoCo 视图")
                world = self.gt_world(eid, betas_mean)
                cameras = np.asarray(raw["cam_c2w"])
                intrinsics = np.asarray(raw["K"])
                kept = np.asarray(raw["kept"], dtype=bool)
            else:
                from ..reproj_core import geometry as geom
                from ..render import compare
                pred = self.pred(eid, cam_mode, hand_mode)
                if pred.get("hand") is None:
                    raise RuntimeError("模型没有 hand 输出，无法渲染 MuJoCo 视图")
                cameras, intrinsics = geom.decode_camera_pose_enc(
                    pred["pose_enc"], height, width, fov_mean=fov_mean)
                world = self.pred_world(eid, cam_mode, hand_mode, betas_mean)
                if world is None:
                    raise RuntimeError("模型没有可用的世界系手部输出")
                kept = compare.prediction_render_mask(pred, len(cameras))
            out_path = self._mujoco_output_path(key)
            self.set_mujoco_prog(key, stage="queued", done=0, total=int(len(cameras)))
            try:
                from ..render.mujoco_video import render_world_video
                with self._mujoco_render_lock:
                    self.set_mujoco_prog(key, stage="render", done=0,
                                         total=int(len(cameras)))

                    def report_step(done, total):
                        self.set_mujoco_prog(
                            key, stage="render", done=int(done), total=int(total))
                        if on_step is not None:
                            on_step(int(done), int(total))

                    def render_temporary(temporary):
                        render_width, render_height = _robot_render_size(render_size)
                        render_world_video(
                            world, cameras, kept, temporary,
                            fps=self.item_fps(eid),
                            intrinsics=intrinsics,
                            image_size=(width, height),
                            width=render_width,
                            height=render_height,
                            view="third",
                            label_text="GT" if source == "gt" else "PRED",
                            on_step=report_step,
                        )

                    _render_video_atomically(out_path, render_temporary)
            except Exception as exc:
                self.set_mujoco_prog(key, stage="error", error=str(exc))
                raise
            self._mujoco_mp4[key] = out_path
            self.set_mujoco_prog(key, stage="done", done=int(len(cameras)),
                                 total=int(len(cameras)))
        return self._mujoco_mp4[key]

    @staticmethod
    def retarget_key(eid: int, source: str,
                     cam_mode: str = DEFAULT_CAM_MODE,
                     hand_mode: str = DEFAULT_HAND_MODE,
                     betas_mean: bool = False,
                     fov_mean: bool = False,
                     render_size: tuple[int, int] | None = None) -> tuple:
        if source == "gt":
            cam_mode, hand_mode = "gt", "gt"
            fov_mean = False
        render_width, render_height = _robot_render_size(render_size)
        size_tag = (f"w{render_width}" if render_height is None
                    else f"w{render_width}h{render_height}")
        return (int(eid), str(source), str(cam_mode), str(hand_mode),
                bool(betas_mean), bool(fov_mean),
                f"shared_wuji_camera_v13_no_presence_hud_{size_tag}")

    def retarget_progress(self, eid: int, source: str,
                          cam_mode: str = DEFAULT_CAM_MODE,
                          hand_mode: str = DEFAULT_HAND_MODE,
                          betas_mean: bool = False,
                          fov_mean: bool = False) -> dict:
        key = self.retarget_key(
            eid, source, cam_mode, hand_mode, betas_mean, fov_mean)
        if self._cached_retarget_video(key) is not None:
            return {"stage": "done", "done": 1, "total": 1}
        return self.get_retarget_prog(key)

    def _retarget_output_path(self, key: tuple) -> Path:
        eid, source, cam_mode, hand_mode, betas_mean, fov_mean, tag = key
        raw = self.raw(eid)
        mode_tag = (
            f"gt_{int(betas_mean)}" if source == "gt" else
            f"{self.ckpt_tag}_{cam_mode}_{hand_mode}_"
            f"{int(betas_mean)}{int(fov_mean)}"
        )
        return self.cache_dir / (
            f"{self.scene}_{self._item_cache_tag(eid)}_"
            f"ep{int(raw['episode_index']):03d}_{tag}_"
            f"{source}_{mode_tag}.mp4")

    def _cached_retarget_video(self, key: tuple) -> Path | None:
        cached = self._retarget_mp4.get(key)
        if cached is None:
            cached = self._retarget_output_path(key)
        if not cached.is_file() or cached.stat().st_size == 0:
            return None
        self._retarget_mp4[key] = cached
        return cached

    def retarget_video(self, eid: int, source: str = "pred",
                       cam_mode: str = DEFAULT_CAM_MODE,
                       hand_mode: str = DEFAULT_HAND_MODE,
                       betas_mean: bool = False,
                       fov_mean: bool = False,
                       on_step=None,
                       render_size: tuple[int, int] | None = None) -> Path:
        """Render the retargeted Wuji Hand from a fitted fixed third-person camera."""
        if source not in {"gt", "pred"}:
            raise ValueError(f"未知 Wuji retargeting 数据源: {source}")
        key = self.retarget_key(
            eid, source, cam_mode, hand_mode, betas_mean, fov_mean, render_size)
        cached = self._cached_retarget_video(key)
        if cached is not None:
            return cached
        with self._key_lock("retarget:" + ":".join(map(str, key))):
            cached = self._cached_retarget_video(key)
            if cached is not None:
                return cached
            raw = self.raw(eid)
            height, width = raw["frames"].shape[1:3]
            if source == "gt":
                if self.is_no_truth(eid):
                    raise RuntimeError("裸视频没有 GT，无法生成 Wuji Hand retargeting")
                world = self.gt_world(eid, betas_mean)
                cameras = np.asarray(raw["cam_c2w"])
                intrinsics = np.asarray(raw["K"])
                kept = np.asarray(raw["kept"], dtype=bool)
            else:
                from ..reproj_core import geometry as geom
                from ..render import compare
                pred = self.pred(eid, cam_mode, hand_mode)
                if pred.get("hand") is None:
                    raise RuntimeError("模型没有 hand 输出，无法生成 Wuji Hand retargeting")
                world = self.pred_world(eid, cam_mode, hand_mode, betas_mean)
                if world is None:
                    raise RuntimeError("模型没有可用的 21 点手部输出")
                cameras, intrinsics = geom.decode_camera_pose_enc(
                    pred["pose_enc"], height, width, fov_mean=fov_mean)
                kept = compare.prediction_render_mask(pred, len(raw["frames"]))
            frames = int(len(kept))
            out_path = self._retarget_output_path(key)
            self.set_retarget_prog(key, stage="queued", done=0, total=frames)
            try:
                from ..render.wuji_retargeting_video import render_wuji_hand_video
                with self._mujoco_render_lock:
                    self.set_retarget_prog(
                        key, stage="retarget", done=0, total=frames)

                    def report_step(done, total):
                        self.set_retarget_prog(
                            key, stage="retarget", done=int(done), total=int(total))
                        if on_step is not None:
                            on_step(int(done), int(total))

                    def render_temporary(temporary):
                        render_width, render_height = _robot_render_size(render_size)
                        render_wuji_hand_video(
                            world, cameras, kept, temporary,
                            fps=self.item_fps(eid), intrinsics=intrinsics,
                            image_size=(width, height),
                            width=render_width,
                            height=render_height,
                            label_text="GT" if source == "gt" else "PRED",
                            on_step=report_step,
                        )

                    _render_video_atomically(out_path, render_temporary)
            except Exception as exc:
                self.set_retarget_prog(key, stage="error", error=str(exc))
                raise
            self._retarget_mp4[key] = out_path
            self.set_retarget_prog(
                key, stage="done", done=frames, total=frames)
        return self._retarget_mp4[key]

    def _compose_export(self, inputs: list[tuple[str, Path]], *,
                        tile_size: tuple[int, int] = _EXPORT_TILE_SIZE) -> Path:
        """Fit every source into equal source-sized tiles, up to four per row."""
        import subprocess

        if not inputs:
            raise ValueError("至少选择一路导出画面")
        paths = []
        tile_width, tile_height = map(int, tile_size)
        if tile_width < 2 or tile_height < 2:
            raise ValueError("导出单格尺寸无效")
        tile_width += tile_width % 2
        tile_height += tile_height % 2
        signature = [_EXPORT_COMPOSE_TAG, f"tile={tile_width}x{tile_height}"]
        for source_id, path in inputs:
            path = Path(path)
            if not path.is_file():
                raise RuntimeError(f"导出源视频不存在: {path}")
            stat = path.stat()
            paths.append(path)
            signature.append(
                f"{source_id}:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
        digest = hashlib.sha256("\n".join(signature).encode()).hexdigest()[:20]
        source_tag = "_".join(source_id for source_id, _path in inputs)
        output_dir = Path(self.cache_dir) / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{source_tag}_{digest}.mp4"
        if output_path.exists():
            return output_path

        with self._key_lock(f"export:{digest}"):
            if output_path.exists():
                return output_path
            command = ["ffmpeg", "-y", "-loglevel", "error"]
            for path in paths:
                command.extend(["-i", str(path)])
            filters = [
                f"[{index}:v]setpts=PTS-STARTPTS,"
                f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
                f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1,setdar={tile_width}/{tile_height}"
                f"[v{index}]"
                for index in range(len(paths))
            ]
            if len(paths) == 1:
                filters.append("[v0]null[vout]")
            else:
                streams = "".join(f"[v{index}]" for index in range(len(paths)))
                grid, _canvas_size = _export_grid_layout(
                    len(paths), tile_width, tile_height)
                filters.append(
                    f"{streams}xstack=inputs={len(paths)}:layout={grid}:"
                    "fill=black:shortest=1[vout]")
            temp_path = output_dir / f".{source_tag}_{digest}.tmp.mp4"
            command.extend([
                "-filter_complex", ";".join(filters), "-map", "[vout]", "-an",
                "-c:v", "libx264", "-preset",
                os.environ.get("VIEWER_VIDEO_PRESET", "veryfast"), "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp_path),
            ])
            proc = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0 or not temp_path.exists():
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(proc.stderr.strip() or "ffmpeg 导出失败")
            temp_path.replace(output_path)
        return output_path

    def _bundle_export(self, inputs: list[tuple[str, Path]], *,
                       episode_index: int,
                       default_source: str = "pred") -> Path:
        """Package the selected GT/PRED renders as individual MP4 files."""
        if not inputs:
            raise ValueError("至少选择一路导出画面")
        if default_source not in {"gt", "pred"}:
            raise ValueError(f"不支持的默认导出数据源: {default_source}")
        filename_parts = {
            "both_2d": "2d",
            "world_motion_3d": "world",
            "mujoco_3d": "mujoco",
            "wuji_retarget_3d": "retarget",
        }
        signature = [_EXPORT_BUNDLE_TAG, f"episode={int(episode_index)}"]
        files = []
        for source_id, path in inputs:
            path = Path(path)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"导出源视频不存在或为空: {path}")
            source = next((name for name in filename_parts if source_id == name or
                           source_id.startswith(f"{name}_")), None)
            if source is None:
                raise ValueError(f"不支持的独立导出画面: {source_id}")
            suffix = source_id[len(source):].lstrip("_") or default_source
            if suffix not in {"gt", "pred"}:
                raise ValueError(f"不支持的独立导出数据源: {source_id}")
            archive_name = (
                f"ep{int(episode_index):03d}-{filename_parts[source]}-{suffix}.mp4")
            stat = path.stat()
            signature.append(
                f"{archive_name}:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
            files.append((archive_name, path))

        digest = hashlib.sha256("\n".join(signature).encode()).hexdigest()[:20]
        output_dir = Path(self.cache_dir) / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            f"ep{int(episode_index):03d}_individual_videos_{digest}.zip")
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        with self._key_lock(f"export-bundle:{digest}"):
            if output_path.exists() and output_path.stat().st_size > 0:
                return output_path
            temp_path = output_dir / f".{output_path.name}.tmp"
            temp_path.unlink(missing_ok=True)
            try:
                with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_STORED) as archive:
                    for archive_name, path in files:
                        archive.write(path, archive_name)
                temp_path.replace(output_path)
            except Exception:
                temp_path.unlink(missing_ok=True)
                raise
        return output_path

    def export_video(self, eid: int, sources: list[str], *, mode: str,
                     layout: str = "overlay", content: str = "both",
                     cam_mode: str = DEFAULT_CAM_MODE,
                     hand_mode: str = DEFAULT_HAND_MODE,
                     gt_betas_mean: bool = False,
                     pred_betas_mean: bool = False,
                     pred_fov_mean: bool = False,
                     world_views: dict | None = None,
                     world_coord_mode: str = "z_up",
                     show_traj: bool = True,
                     show_cam_hand: bool = True,
                     raw: bool = False,
                     output_format: str = "grid",
                     on_progress=None) -> Path:
        """Render missing videos, then export a grid MP4 or individual-video ZIP."""
        allowed = {"both_2d", "world_motion_3d", "mujoco_3d", "wuji_retarget_3d"}
        ordered = list(dict.fromkeys(sources))
        unknown = [source_id for source_id in ordered if source_id not in allowed]
        if unknown:
            raise ValueError(f"不支持的导出画面: {', '.join(unknown)}")
        if not ordered:
            raise ValueError("至少选择一路导出画面")
        if output_format not in {"grid", "separate"}:
            raise ValueError(f"不支持的导出格式: {output_format}")

        labels = {
            "both_2d": "原视频渲染",
            "world_motion_3d": "固定世界·3D",
            "mujoco_3d": "MuJoCo·仿真",
            "wuji_retarget_3d": "Wuji Hand·Retargeting",
        }

        def emit(**values):
            if on_progress is not None:
                on_progress(values)

        raw_data = self.raw(eid)
        frame_height, frame_width = map(int, raw_data["frames"].shape[1:3])
        export_tile_size = (frame_width, frame_height)
        source = "gt" if raw else "pred"
        compare_sources = not raw and not self.is_no_truth(eid)
        source_count = len(ordered)
        source_fractions = [0.0] * source_count
        source_states = [
            {
                "source": source_id, "source_label": labels[source_id],
                "stage": "queued", "progress": 0.0,
                "frame_done": 0, "frame_total": 0,
            }
            for source_id in ordered
        ]
        export_progress_lock = threading.Lock()

        def update_source(source_index, fraction, *, source_id, label,
                          frame_done=None, frame_total=None, message=None):
            with export_progress_lock:
                source_fractions[source_index] = max(
                    source_fractions[source_index], max(0.0, min(1.0, float(fraction))))
                state = source_states[source_index]
                state.update(
                    stage="done" if source_fractions[source_index] >= 1.0 else "render",
                    progress=source_fractions[source_index],
                )
                if frame_done is not None:
                    state["frame_done"] = int(frame_done)
                if frame_total is not None:
                    state["frame_total"] = int(frame_total)
                states_snapshot = {
                    item["source"]: dict(item) for item in source_states
                }
                progress = 0.9 * sum(source_fractions) / source_count
                active = [
                    f"{item['source_label']} {round(item['progress'] * 100)}%"
                    for item in source_states if item["stage"] == "render"
                ]
                summary = "并行渲染：" + " · ".join(active) if active else message
            emit(stage="render", progress=progress,
                 source=source_id, source_label=label,
                 source_index=source_index + 1, source_total=source_count,
                 frame_done=state["frame_done"], frame_total=state["frame_total"],
                 sources=states_snapshot,
                 message=summary or f"正在渲染 {label}")

        def render_source(source_index, source_id):
            label = labels[source_id]

            def source_step(done, total, *, index=source_index,
                            current_id=source_id, current_label=label):
                total = max(1, int(total))
                done = max(0, min(int(done), total))
                update_source(
                    index, done / total, source_id=current_id, label=current_label,
                    frame_done=done, frame_total=total,
                    message=f"正在渲染 {current_label}：{done}/{total} 帧")

            def paired_phase_step(phase, phase_label):
                def report(done, total, *, index=source_index,
                           current_id=source_id, current_label=label):
                    total = max(1, int(total))
                    done = max(0, min(int(done), total))
                    combined_done = phase * total + done
                    combined_total = total * 2
                    update_source(
                        index, 0.48 * (phase + done / total),
                        source_id=current_id, label=current_label,
                        frame_done=combined_done, frame_total=combined_total,
                        message=(f"正在渲染 {current_label} {phase_label}："
                                 f"{done}/{total} 帧"))
                return report

            def paired_result(gt_path, pred_path):
                update_source(
                    source_index, 0.96, source_id=source_id, label=label,
                    message=f"{label} GT/PRED 已渲染")
                return [
                    (f"{source_id}_gt", gt_path),
                    (f"{source_id}_pred", pred_path),
                ]

            def robot_comparison(render):
                gt_path = render(
                    eid, "gt", cam_mode, hand_mode, gt_betas_mean, False,
                    on_step=paired_phase_step(0, "GT"),
                    render_size=export_tile_size)
                update_source(
                    source_index, 0.48, source_id=source_id, label=label,
                    message=f"{label} GT 已完成，正在准备 PRED")
                pred_path = render(
                    eid, "pred", cam_mode, hand_mode,
                    pred_betas_mean, pred_fov_mean,
                    on_step=paired_phase_step(1, "PRED"),
                    render_size=export_tile_size)
                return paired_result(gt_path, pred_path)

            def single_world_views(which):
                views = dict(world_views or {})
                source_view = views.get("vgt" if which == "gt" else "vpred")
                if not isinstance(source_view, dict):
                    source_view = views.get("vov")
                if isinstance(source_view, dict):
                    views["vov"] = source_view
                return views

            update_source(
                source_index, 0.0, source_id=source_id, label=label,
                message=f"正在准备第 {source_index + 1}/{source_count} 路：{label}")
            if source_id == "both_2d":
                if compare_sources:
                    gt_path = self.mp4_gt(
                        eid, mode, on_step=paired_phase_step(0, "GT"),
                        betas_mean=gt_betas_mean)
                    pred_path = self.mp4_pred(
                        eid, mode, cam_mode, hand_mode,
                        pred_betas_mean, pred_fov_mean,
                        on_step=paired_phase_step(1, "PRED"))
                    path = paired_result(gt_path, pred_path)
                else:
                    path = (self.mp4_gt(
                        eid, mode, on_step=source_step,
                        betas_mean=gt_betas_mean) if raw else self.mp4_pred(
                            eid, mode, cam_mode, hand_mode,
                            pred_betas_mean, pred_fov_mean,
                            on_step=source_step))
            elif source_id == "world_motion_3d":
                world_options = {
                    "layout": "overlay", "coord_mode": world_coord_mode,
                    "show_traj": show_traj, "show_cam_hand": show_cam_hand,
                    "cam_mode": cam_mode, "hand_mode": hand_mode,
                    "gt_betas_mean": gt_betas_mean,
                    "pred_betas_mean": pred_betas_mean,
                    "pred_fov_mean": pred_fov_mean,
                    "render_size": export_tile_size,
                }
                if compare_sources:
                    gt_path = self.world_video(
                        eid, views=single_world_views("gt"),
                        render_source="gt", raw=False,
                        on_step=paired_phase_step(0, "GT"), **world_options)
                    pred_path = self.world_video(
                        eid, views=single_world_views("pred"),
                        render_source="pred", raw=False,
                        on_step=paired_phase_step(1, "PRED"), **world_options)
                    path = paired_result(gt_path, pred_path)
                else:
                    path = self.world_video(
                        eid, views=world_views,
                        render_source="gt" if raw else "pred", raw=raw,
                        on_step=source_step, **world_options)
            elif source_id == "mujoco_3d":
                path = (robot_comparison(self.mujoco_video)
                        if compare_sources else self.mujoco_video(
                            eid, source, cam_mode, hand_mode,
                            gt_betas_mean if raw else pred_betas_mean,
                            False if raw else pred_fov_mean,
                            on_step=source_step,
                            render_size=export_tile_size))
            else:
                path = (robot_comparison(self.retarget_video)
                        if compare_sources else self.retarget_video(
                            eid, source, cam_mode, hand_mode,
                            gt_betas_mean if raw else pred_betas_mean,
                            False if raw else pred_fov_mean,
                            on_step=source_step,
                            render_size=export_tile_size))
            update_source(
                source_index, 1.0, source_id=source_id, label=label,
                message=f"第 {source_index + 1}/{source_count} 路已完成：{label}")
            return path if isinstance(path, list) else [(source_id, path)]

        rendered_by_index = [None] * source_count
        remaining_indices = list(range(source_count))
        if "both_2d" in ordered:
            primary_index = ordered.index("both_2d")
            rendered_by_index[primary_index] = render_source(primary_index, "both_2d")
            remaining_indices.remove(primary_index)

        if remaining_indices:
            with ThreadPoolExecutor(
                    max_workers=min(4, len(remaining_indices)),
                    thread_name_prefix="viewer-export") as pool:
                futures = {
                    pool.submit(render_source, index, ordered[index]): index
                    for index in remaining_indices
                }
                for future in as_completed(futures):
                    rendered_by_index[futures[future]] = future.result()
        rendered = [item for group in rendered_by_index if group for item in group]
        separate = output_format == "separate"
        emit(stage="compose", progress=0.92, source_index=source_count,
             source_total=source_count,
             message=("所选画面渲染完成，正在打包独立视频" if separate
                      else "所选画面渲染完成，正在合成导出视频"))
        output = (self._bundle_export(
            rendered, episode_index=int(raw_data["episode_index"]),
            default_source=source) if separate
                  else self._compose_export(rendered, tile_size=export_tile_size))
        emit(stage="done", progress=1.0, source_index=source_count,
             source_total=source_count,
             message="独立视频包已生成" if separate else "导出视频已生成")
        return output

    def mp4_gt(self, eid: int, mode: str, on_step=None,
               betas_mean: bool = False) -> Path:
        """「仅原始 GT」2D overlay（GT 手 + GT 相机，单画面），不跑推理。缓存名带 _gt，与 ckpt 无关。"""
        key = (eid, mode, "gt", "gt", bool(betas_mean))
        cached = self._mp4.get(key)
        if cached and cached.exists():
            return cached
        with self._key_lock(f"mp4gt{eid}:{mode}:{int(betas_mean)}"):
            cached = self._mp4.get(key)
            if cached and cached.exists():
                return cached
            from ..render import compare
            raw = self.raw(eid)
            ep_idx = raw["episode_index"]
            source_tag = self._item_cache_tag(eid)
            out_path = self.cache_dir / (
                f"{self.scene}_{source_tag}_ep{ep_idx:03d}_{mode}_"
                f"gt_{int(betas_mean)}_{compare.CACHE_TAG}.mp4")
            if not out_path.exists():
                pk = self.prog2d_key(eid, mode, "gt", "gt", raw=True)

                def report_step(done, total):
                    self.set_prog2d(pk, int(done), int(total))
                    if on_step is not None:
                        on_step(int(done), int(total))

                compare.render_gt_overlay(raw, out_path, mode=mode, fps=self.item_fps(eid),
                                          on_step=report_step,
                                          betas_mean=betas_mean,
                                          gt_world_data=self._gtw.get((eid, betas_mean)),
                                          presence_text="")
            self._mp4[key] = out_path
        return self._mp4[key]

    def mp4_pred(self, eid: int, mode: str,
                  cam_mode: str = DEFAULT_CAM_MODE,
                  hand_mode: str = DEFAULT_HAND_MODE,
                  pred_betas_mean: bool = False,
                  pred_fov_mean: bool = False,
                  on_step=None) -> Path:
        """Render one prediction-only 2D video at the original frame size."""
        pkey = f"{int(pred_betas_mean)}{int(pred_fov_mean)}"
        key = (eid, mode, "pred", cam_mode, hand_mode, pkey)
        cached = self._mp4.get(key)
        if cached and cached.exists():
            return cached
        with self._key_lock(
                f"mp4pred{eid}:{mode}:{cam_mode}:{hand_mode}:{pkey}"):
            cached = self._mp4.get(key)
            if cached and cached.exists():
                return cached
            from ..render import compare
            raw = self.raw(eid)
            pred = self.pred(eid, cam_mode, hand_mode)
            if pred.get("hand") is None:
                raise RuntimeError("模型没有 hand 输出，无法渲染预测 2D 视图")
            source_tag = self._item_cache_tag(eid)
            out_path = self.cache_dir / (
                f"{self.scene}_{source_tag}_{self.ckpt_tag}_"
                f"ep{int(raw['episode_index']):03d}_{mode}_pred_"
                f"{cam_mode}_{hand_mode}_{pkey}_{compare.CACHE_TAG}.mp4")
            if not out_path.exists():
                pk = self.prog2d_key(
                    eid, mode, "overlay", "pred", raw=False,
                    cam_mode=cam_mode, hand_mode=hand_mode,
                    pkey=f"0{pkey}")

                def report_step(done, total):
                    self.set_prog2d(pk, int(done), int(total))
                    if on_step is not None:
                        on_step(int(done), int(total))

                pred_world = self.pred_world(
                    eid, cam_mode, hand_mode, pred_betas_mean)
                hand_frame = (self.item_hand_frame(eid)
                              if self.is_no_truth(eid) else "camera")
                compare.render_pred_overlay(
                    raw["frames"], pred, out_path,
                    mode=mode, fps=self.item_fps(eid),
                    hand_frame=hand_frame,
                    betas_mean=pred_betas_mean, fov_mean=pred_fov_mean,
                    pred_world_data=pred_world, on_step=report_step)
            self._mp4[key] = out_path
        return self._mp4[key]

    def mp4(self, eid: int, mode: str, layout: str = "overlay", content: str = "both",
            cam_mode: str = DEFAULT_CAM_MODE, hand_mode: str = DEFAULT_HAND_MODE,
            gt_betas_mean: bool = False,
            pred_betas_mean: bool = False, pred_fov_mean: bool = False,
            on_step=None) -> Path:
        no_truth = self.is_no_truth(eid)
        # 无真值：只有「仅预测」单画面，layout/content 无意义 → 归一到固定键，避免重复渲。
        if no_truth:
            layout, content = "overlay", "pred"
        pkey = f"{int(gt_betas_mean)}{int(pred_betas_mean)}{int(pred_fov_mean)}"   # 手形/内参平均组合编码
        key = (eid, mode, layout, content, cam_mode, hand_mode, pkey)
        cached = self._mp4.get(key)
        if cached and cached.exists():
            return cached
        with self._key_lock(
                f"mp4{eid}:{mode}:{layout}:{content}:{cam_mode}:{hand_mode}:{pkey}"):
            cached = self._mp4.get(key)
            if cached and cached.exists():
                return cached
            from ..render import compare
            raw = self.raw(eid)
            ep_idx = raw["episode_index"]
            # 数据源和全部推理模式都进文件名，避免跨数据集/模式复用缓存。
            source_tag = self._item_cache_tag(eid)
            out_path = self.cache_dir / f"{self.scene}_{source_tag}_{self.ckpt_tag}_ep{ep_idx:03d}_{mode}_{layout}_{content}_{cam_mode}_{hand_mode}_{pkey}_{compare.CACHE_TAG}.mp4"
            if not out_path.exists():
                if no_truth:
                    pr = self.pred(eid, cam_mode, hand_mode)
                    if "hand" not in pr:
                        raise RuntimeError(
                            "模型无 hand 输出（enable_hand 关或该 ckpt 无手部头），裸视频无重投影内容")
                    pk = self.prog2d_key(eid, mode, layout, content, raw=False,
                                         cam_mode=cam_mode, hand_mode=hand_mode, pkey=pkey)

                    def report_step(done, total):
                        self.set_prog2d(pk, int(done), int(total))
                        if on_step is not None:
                            on_step(int(done), int(total))

                    compare.render_pred_overlay(raw["frames"], pr, out_path,
                                                mode=mode, fps=self.item_fps(eid),
                                                hand_frame=self.item_hand_frame(eid),
                                                betas_mean=pred_betas_mean, fov_mean=pred_fov_mean,
                                                pred_world_data=self._prw.get(
                                                    (eid, cam_mode, hand_mode, pred_betas_mean)),
                                                on_step=report_step)
                else:
                    pk = self.prog2d_key(eid, mode, layout, content, raw=False,
                                         cam_mode=cam_mode, hand_mode=hand_mode, pkey=pkey)

                    def report_step(done, total):
                        self.set_prog2d(pk, int(done), int(total))
                        if on_step is not None:
                            on_step(int(done), int(total))

                    compare.render_2d(raw, self.pred(eid, cam_mode, hand_mode), out_path,
                                      mode=mode, fps=self.item_fps(eid),
                                      progress=True, layout=layout, content=content,
                                      gt_betas_mean=gt_betas_mean, pred_betas_mean=pred_betas_mean,
                                      pred_fov_mean=pred_fov_mean,
                                      gt_world_data=self._gtw.get((eid, gt_betas_mean)),
                                      pred_world_data=self._prw.get(
                                          (eid, cam_mode, hand_mode, pred_betas_mean)),
                                      on_step=report_step)
            self._mp4[key] = out_path
        return self._mp4[key]

    def prerender(self, jobs: int, mode: str, layout: str = "overlay", content: str = "both") -> None:
        """后台把 --input 根数据集的全部 episode 预测 + 指定 mode/layout/content 2D 预渲好，jobs 路并发。
        仅当 --input 本身是 lerobot 数据集时生效；否则（视频根/混合树）跳过——预渲需固定 episode 集合。"""
        from ..reproj_core import lerobot_io
        ds_dir = lerobot_io.find_dataset(self.root)
        if ds_dir is None:
            print(f"[prerender] --input 非 lerobot 数据集（{self.root}），跳过批量预渲")
            return
        try:
            from ..reproj_core import mano
            mano.build_mano_faces()   # 预热 MANO，避免 worker 并发首次初始化竞态
        except Exception as e:  # noqa: BLE001
            print(f"[prerender] MANO 预热失败，跳过批量预渲: {e}")
            return
        rec = self.ensure_dataset(ds_dir)
        eids = [self.lerobot_eid(ds_dir, i) for i in range(len(rec["eps"]))]
        done = 0
        print(f"[prerender] 后台预渲 {len(eids)} 个 episode（mode={mode}, layout={layout}, content={content}, jobs={jobs}）...")
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
            futs = {ex.submit(self.mp4, eid, mode, layout, content): eid for eid in eids}
            for fut in as_completed(futs):
                eid = futs[fut]
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[prerender] ep{eid} 失败: {e}")
                done += 1
                print(f"[prerender] 进度 {done}/{len(eids)}")
        print("[prerender] 批量预渲完成")
