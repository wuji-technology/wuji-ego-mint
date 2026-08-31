"""hand_filter op：基于 YOLO 手部检测过滤无手区间并切片

来源
    本 op 等价于 scripts/batch_processing/batch_split_video.py 的多进程流水线，
    现作为 filtered/ 框架下的首个 op。原脚本保留作为旧入口，新代码请走本 op。

流水线架构（CPU/GPU 解耦三级管线，零落盘抽帧）
    extract_pool (CPU N×) → q_frames(SharedMemory) → yolo_actor (GPU 1×/卡, batch=64)
                                                              ↓
                              q_split  ←  tracker_pool (CPU M×) ←
                                    ↓
                              split_pool (CPU K×) → done
    抽帧改用 ffmpeg rawvideo pipe → SharedMemory，YOLO actor 零拷贝读取，
    省掉 JPG 编码/解码 + 小文件 IO；frame_dets 直接通过 mp.Queue 传 dict，不落盘。
    GPU 全程被 q_frames 喂满，批量推理；BYTETracker 放到独立 CPU 进程不占 GPU。
    幂等缓存：detection/model_tracks.npy（命中则跳过抽帧+检测）、<stem>.done

输出结构
    <output_root>/<rel>/<stem>/
        <stem>_part000_<start>s_<end>s.mp4
        <stem>_part001_<start>s_<end>s.mp4
        ...
        <stem>.done

依赖
    torch + ultralytics YOLO（HaWoR 提供检测权重）、ffmpeg、HaWoR.lib.pipeline.tools._DetAdaptor、tqdm
"""
from __future__ import annotations

import glob as _glob
import json
import multiprocessing as mp
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading as _th
import time
import uuid as _uuid
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

from core import ProcessResult, get_available_cpu_count, get_meta, log

# /dev/shm 是 tmpfs，写入即驻留内存；用 np.memmap 在消费端零拷贝读取
_SHM_ROOT = Path("/dev/shm")


# ── 路径与外部依赖路径解析 ────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent                  # .../filtered/ops
_FILTERED_DIR = _HERE.parent                             # .../filtered
_SCRIPTS_DIR = _FILTERED_DIR.parent                      # .../ego_pipeline/preprocessing
_REPO_ROOT = _SCRIPTS_DIR.parents[1]                     # repo root
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_HAWOR_DIR = _REPO_ROOT / "third_party" / "HaWoR"


# ── 警告抑制（必须在 import torch / ultralytics 之前生效）────────────────────

def _silence_known_warnings() -> None:
    warnings.filterwarnings("ignore", category=Warning,
                            message=r"urllib3 .* or chardet .*/charset_normalizer")
    warnings.filterwarnings("ignore", category=UserWarning,
                            message=r"Applied workaround for CuDNN")
    warnings.filterwarnings("ignore", category=FutureWarning)


_silence_known_warnings()
_WARN_EXTRA = ("ignore:urllib3:Warning,"
               "ignore:Applied workaround for CuDNN issue:UserWarning")
os.environ["PYTHONWARNINGS"] = (
    f"{os.environ['PYTHONWARNINGS']},{_WARN_EXTRA}"
    if os.environ.get("PYTHONWARNINGS") else _WARN_EXTRA
)


# ═══════════════════════════════════════════════════════════════════════════════
# op 接口
# ═══════════════════════════════════════════════════════════════════════════════

NAME = "hand_filter"
DESCRIPTION = "基于 YOLO 手部检测过滤无手区间并切片"
IS_TERMINAL = True  # 自己写多段 mp4 + .done 标记


def add_arguments(parser) -> None:
    parser.add_argument("--det_fps", type=float, default=5.0,
                        help="抽帧检测帧率（fps），默认 5.0")
    parser.add_argument("--min_absent_sec", type=float, default=0.5,
                        help="无手最短切除区间（秒），默认 0.5")
    parser.add_argument("--min_segment_sec", type=float, default=2.0,
                        help="保留段最短时长（秒），默认 2.0")
    parser.add_argument("--det_thresh", type=float, default=0.2,
                        help="YOLO 置信度阈值，默认 0.2（越低越敏感、越少切除）")
    parser.add_argument("--filter_multi_hands", action="store_true",
                        help="启用后将检测到超过两只手的区间也视为应切除；"
                             "默认仅切除无手区间")
    parser.add_argument("--require_lr_pair", action="store_true",
                        help="只保留「最多两只手且无同侧重复」的区间：每帧需 左≤1 且 右≤1 "
                             "且至少一只手；出现第三只手 / 同侧两只 → 切除。单手区间保留。")
    parser.add_argument("--watch", action="store_true",
                        help="常驻流式模式：模型只加载一次，持续轮询 --input 目录吃新段，"
                             "直到 <output_dir>/.stream_done 出现才收尾退出。供 --stream 编排器用。")
    parser.add_argument("--watch_poll", type=float, default=2.0,
                        help="watch 模式轮询间隔（秒），默认 2.0")
    parser.add_argument("--keep_work", action="store_true",
                        help="保留中间产物（抽帧 / detection cache）")
    parser.add_argument("--work_dir", default=None,
                        help="中间产物目录；不指定则 <output_dir>/.work")
    parser.add_argument("--no_copy_audio", action="store_true",
                        help="切片时不复制音轨")
    # GPU / profile
    parser.add_argument("--gpus", default="0,1,2,3",
                        help="逗号分隔 GPU id 或 auto；默认 '0,1,2,3'")
    parser.add_argument("--profile", choices=("balanced", "aggressive"),
                        default="aggressive",
                        help="运行配置：balanced 更保守，aggressive 面向多卡离线高吞吐（默认）")
    parser.add_argument("--batch", type=int, default=None,
                        help="YOLO 批大小（默认由 profile 决定：balanced=32, aggressive=64）")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="YOLO 推理 imgsz，默认 640")
    parser.add_argument("--no_half", action="store_true",
                        help="禁用 FP16")
    parser.add_argument("--micro_batch_timeout_ms", type=int, default=None,
                        help="跨视频聚合微批等待窗口（毫秒，默认由 profile 决定）")
    parser.add_argument("--max_frames_inflight", type=int, default=None,
                        help="每个 GPU actor 一次聚合的最大帧数（默认约为 batch 的 2 倍）")
    # 并发
    parser.add_argument("--extract_workers", type=int, default=None,
                        help="CPU 抽帧进程数（默认由 profile 决定）")
    parser.add_argument("--tracker_workers", type=int, default=None,
                        help="CPU ByteTrack 进程数（默认由 profile 决定）")
    parser.add_argument("--split_workers", type=int, default=None,
                        help="CPU 切片进程数（默认由 profile 决定）")
    parser.add_argument("--extract_queue_size", type=int, default=None,
                        help="extract→yolo 队列深度（默认由 profile 决定）")
    parser.add_argument("--yolo_queue_size", type=int, default=None,
                        help="yolo→tracker 队列深度（默认由 profile 决定）")
    parser.add_argument("--track_queue_size", type=int, default=None,
                        help="tracker→split 队列深度（默认由 profile 决定）")


def describe_config(args) -> str:
    return (f"det_fps={args.det_fps}, min_absent={args.min_absent_sec}s, "
            f"min_segment={args.min_segment_sec}s, det_thresh={args.det_thresh}, "
            f"filter_multi_hands={args.filter_multi_hands}, "
            f"require_lr_pair={getattr(args, 'require_lr_pair', False)}, "
            f"gpus={args.gpus}, profile={args.profile}")


def output_label(args) -> str:
    parts = ["handFlt"]
    if args.filter_multi_hands:
        parts.append("nomulti")
    return "_".join(parts)


def default_output_suffix(args) -> str:
    return "_" + output_label(args)


def plan(video: Path, args) -> str:
    meta = get_meta(video)
    if meta["duration"] <= 0:
        return "(ffprobe 失败)"
    return (f"{meta['duration']:.1f}s @ {meta['fps']:.1f}fps → "
            f"抽帧 {args.det_fps}fps → YOLO 检测 → 过滤无手区间 → 切片")


# ═══════════════════════════════════════════════════════════════════════════════
# 通用工具（模块内私有）
# ═══════════════════════════════════════════════════════════════════════════════

def _get_video_meta(video_path: str) -> tuple[float, float, int, int]:
    """返回 (fps, duration, width, height)"""
    r = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,width,height:format=duration",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    stream = data["streams"][0]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    fps = num / den if den else 0.0
    return (fps,
            float(data["format"]["duration"]),
            int(stream.get("width") or 0),
            int(stream.get("height") or 0))


# ── 共享内存帧传输（np.memmap over /dev/shm 文件，避开 resource_tracker）─────

# 每个 extract worker 进程独享：当前活跃的 ffmpeg subprocess + 它正在写的 SHM 路径。
# 被 SIGTERM/SIGINT/SIGHUP 触发时，由 _extract_signal_cleanup 负责 kill + unlink，
# 不留 orphan ffmpeg 或半成品 SHM。
_EXTRACT_ACTIVE_FFMPEG: subprocess.Popen | None = None
_EXTRACT_ACTIVE_SHM_PATH: str | None = None


def _extract_signal_cleanup(_sig=None, _frame=None) -> None:
    """收到 term 信号时：先杀 ffmpeg，再 unlink 半写的 SHM，然后退出。"""
    global _EXTRACT_ACTIVE_FFMPEG, _EXTRACT_ACTIVE_SHM_PATH
    p = _EXTRACT_ACTIVE_FFMPEG
    if p is not None:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
    path = _EXTRACT_ACTIVE_SHM_PATH
    if path:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except Exception:
            pass
    _EXTRACT_ACTIVE_FFMPEG = None
    _EXTRACT_ACTIVE_SHM_PATH = None
    os._exit(0)


def _set_parent_death_signal(sig: int = signal.SIGTERM) -> None:
    """Linux PR_SET_PDEATHSIG：父进程死亡时内核给本进程发指定信号。

    保护链：主进程被 SIGKILL → 内核发 SIGTERM 给所有子 worker → worker 的
    signal handler 触发 → 杀 ffmpeg + unlink SHM → 自己退出。即使主进程
    没机会执行 _terminate_all，也不会留 orphan ffmpeg。
    """
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0)
    except Exception:
        pass


def _extract_frames_to_shm(
    video_path: str,
    det_fps: float,
    session_prefix: str,
) -> dict:
    """ffmpeg rawvideo → /dev/shm/<name> 文件。consumer 用 np.memmap 零拷贝读。

    用普通文件 + tmpfs 而不是 multiprocessing.shared_memory，绕开 resource_tracker
    的双重 unregister/leak warning。生命周期由消费端 unlink 负责，崩溃残留
    由 _cleanup_session_shm 兜底。
    """
    try:
        real_fps, duration, width, height = _get_video_meta(video_path)
    except Exception as e:
        return {"ok": False, "error": f"ffprobe: {e}"}
    if duration <= 0 or width <= 0 or height <= 0:
        return {"ok": False, "error": "invalid ffprobe meta",
                "real_fps": real_fps, "duration": duration}

    det_fps_use = min(det_fps, real_fps) if real_fps > 0 else det_fps
    frame_bytes = width * height * 3

    cmd = [
        "ffmpeg", "-loglevel", "error", "-nostdin", "-threads", "2",
        "-i", str(video_path),
        "-vf", f"fps={det_fps_use}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]

    shm_name = f"{session_prefix}_{_uuid.uuid4().hex[:10]}"
    shm_path = str(_SHM_ROOT / shm_name)

    global _EXTRACT_ACTIVE_FFMPEG, _EXTRACT_ACTIVE_SHM_PATH
    _EXTRACT_ACTIVE_SHM_PATH = shm_path
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             bufsize=frame_bytes * 8)
    except Exception as e:
        _EXTRACT_ACTIVE_SHM_PATH = None
        return {"ok": False, "error": f"ffmpeg spawn: {e}",
                "real_fps": real_fps, "duration": duration}
    _EXTRACT_ACTIVE_FFMPEG = p

    total = 0
    try:
        with open(shm_path, "wb") as f:
            read_size = max(frame_bytes, 1 << 20)
            while True:
                chunk = p.stdout.read(read_size)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        err = p.stderr.read().decode("utf-8", "ignore")
        rc = p.wait()
    except Exception as e:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
        try:
            os.unlink(shm_path)
        except Exception:
            pass
        _EXTRACT_ACTIVE_FFMPEG = None
        _EXTRACT_ACTIVE_SHM_PATH = None
        return {"ok": False, "error": f"shm write: {e}",
                "real_fps": real_fps, "duration": duration}
    finally:
        _EXTRACT_ACTIVE_FFMPEG = None
        _EXTRACT_ACTIVE_SHM_PATH = None

    if rc != 0:
        try:
            os.unlink(shm_path)
        except Exception:
            pass
        return {"ok": False, "error": f"ffmpeg rc={rc}: {err[-400:]}",
                "real_fps": real_fps, "duration": duration}

    n_frames = total // frame_bytes
    if n_frames <= 0:
        try:
            os.unlink(shm_path)
        except Exception:
            pass
        return {"ok": False, "error": "no frames decoded",
                "real_fps": real_fps, "duration": duration}

    # 截断到帧整除（防止 ffmpeg 输出残留尾字节）
    arr_size = n_frames * frame_bytes
    if total > arr_size:
        try:
            os.truncate(shm_path, arr_size)
        except Exception:
            pass

    return {
        "ok": True,
        "frames_shm": shm_path,
        "frames_shape": (n_frames, height, width, 3),
        "frames_dtype": "uint8",
        "n_frames": n_frames,
        "real_fps": real_fps,
        "duration": duration,
        "det_fps_use": det_fps_use,
    }


def _open_shm_frames(shm_path: str, shape, dtype: str):
    """consumer 侧 mmap 文件作 ndarray view。返回 (memmap_arr, shm_path)。"""
    arr = np.memmap(shm_path, dtype=np.dtype(dtype), mode="r", shape=tuple(shape))
    return arr, shm_path


def _release_shm(shm_path: str) -> None:
    """consumer 处理完成后 unlink shm 文件（mmap 视图先 del 再调本函数）。"""
    if not shm_path:
        return
    try:
        os.unlink(shm_path)
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _cleanup_session_shm(session_prefix: str) -> int:
    """扫 /dev/shm 兜底清理本 session 的残留。返回清理数。"""
    n = 0
    for path in _glob.glob(f"/dev/shm/{session_prefix}*"):
        try:
            os.unlink(path)
            n += 1
        except Exception:
            pass
    return n


def _take_micro_batch(
    pending_states: list[dict],
    max_frames_inflight: int,
) -> tuple[list[tuple[dict, int]], list[dict], list[dict]]:
    """聚合微批；in-place 更新 cursor / batch_sizes。

    返回 (micro_batch, still_pending, completed)：
      - micro_batch: [(state, frame_idx), ...]
      - still_pending: cursor 还没到末尾的 state（已包含未被选中入批的尾部）
      - completed: 在本次调用中 cursor 推进到末尾的 state
    """
    micro_batch: list[tuple[dict, int]] = []
    still_pending: list[dict] = []
    completed: list[dict] = []
    budget = max(1, max_frames_inflight)

    full = False
    for idx, state in enumerate(pending_states):
        remaining = state["n_frames"] - state["cursor"]
        if remaining <= 0:
            # 通常不会发生：state 在 pending 里就应有剩余
            completed.append(state)
            continue
        if full:
            still_pending.append(state)
            continue
        take = min(remaining, budget - len(micro_batch))
        if take > 0:
            start = state["cursor"]
            end = start + take
            for frame_idx in range(start, end):
                micro_batch.append((state, frame_idx))
            state["cursor"] = end
            state["batch_sizes"].append(take)
        if state["cursor"] >= state["n_frames"]:
            completed.append(state)
        else:
            still_pending.append(state)
        if len(micro_batch) >= budget:
            full = True

    return micro_batch, still_pending, completed


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 1：抽帧（CPU 进程池）—— ffmpeg rawvideo → SharedMemory
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_pool_worker(
    in_q: mp.Queue, out_q: mp.Queue,
    det_fps: float, session_prefix: str,
):
    """抽帧 worker：从 in_q 取 job → 检查 detection 缓存 → 解码进 SHM → 入 out_q。

    in_q 里有 1 个 None 表示这个 worker 要退出；不往 out_q 写 None——
    下游 sentinel 由 shepherd 线程在所有 worker 完成后统一投递。

    收到 SIGTERM/SIGHUP（主进程退出 / 容器停止）时会被 _extract_signal_cleanup
    截获：先 kill 当前 ffmpeg + unlink 半写的 SHM，再 os._exit(0)。
    SIGINT 由主进程独占（worker 忽略）。
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _extract_signal_cleanup)
    signal.signal(signal.SIGHUP, _extract_signal_cleanup)
    _set_parent_death_signal()  # 主进程崩了 → 自动收到 SIGTERM → 触发清理
    while True:
        job = in_q.get()
        if job is None:
            return
        t0 = time.time()
        base = {
            "video": job["video"], "stem": job["stem"],
            "out_subdir": job["out_subdir"],
            "detection_dir": job["detection_dir"],
            "video_work": job["video_work"],
        }
        try:
            cache_path = Path(job["detection_dir"]) / "model_tracks.npy"
            if cache_path.exists():
                # detection cache 命中：不必抽帧。只取 fps/duration 给下游用。
                real_fps, duration, _w, _h = _get_video_meta(job["video"])
                det_fps_use = min(det_fps, real_fps) if real_fps > 0 else det_fps
                r = {
                    **base,
                    "ok": True,
                    "detect_cached": True,
                    "duration": duration,
                    "real_fps": real_fps,
                    "det_fps_use": det_fps_use,
                    "n_frames": max(1, int(round(duration * det_fps_use))),
                    "t_extract": time.time() - t0,
                    "ts_extract_done": time.time(),
                }
            else:
                Path(job["detection_dir"]).mkdir(parents=True, exist_ok=True)
                meta = _extract_frames_to_shm(job["video"], det_fps, session_prefix)
                r = {
                    **base,
                    **meta,
                    "t_extract": time.time() - t0,
                    "ts_extract_done": time.time(),
                }
        except Exception as e:
            import traceback
            r = {
                **base,
                "ok": False,
                "error": f"extract: {type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}",
                "t_extract": time.time() - t0,
                "ts_extract_done": time.time(),
            }
        out_q.put(r)


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 2：YOLO 检测（GPU Actor，每 GPU 1 个进程）
# ═══════════════════════════════════════════════════════════════════════════════

def _yolo_actor_main(
    gpu_id: int,
    in_q: mp.Queue,
    out_q: mp.Queue,
    det_thresh: float,
    batch: int,
    imgsz: int,
    half: bool,
    micro_batch_timeout_ms: int,
    max_frames_inflight: int,
):
    """常驻 GPU actor：跨视频聚合微批；从 SHM zero-copy 读帧。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _set_parent_death_signal()
    _silence_known_warnings()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("CUDNN_BENCHMARK", "1")

    import gc
    import torch
    from ultralytics import YOLO
    torch.backends.cudnn.benchmark = True

    device = "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = "cuda:0"

    ckpt = os.environ.get("HAWOR_DETECTOR_CKPT",
                          str(_REPO_ROOT / "model" / "hawor" / "detector.pt"))
    if not Path(ckpt).exists():
        ckpt = str(_HAWOR_DIR / "weights" / "external" / "detector.pt")

    infer_batch = max(1, batch)
    max_frames_inflight = max(infer_batch, max_frames_inflight)
    log(
        f"YOLO-GPU{gpu_id}",
        f"PID={os.getpid()} 加载模型 {ckpt} batch={infer_batch} "
        f"inflight={max_frames_inflight} imgsz={imgsz} half={half}"
    )
    model = YOLO(ckpt)

    def _release_cuda_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _release_state(state: dict) -> None:
        # 先丢 memmap 视图再 unlink 文件
        state.pop("frames", None)
        shm_path = state.pop("shm_path", None)
        if shm_path:
            _release_shm(shm_path)

    def _emit_cached(job: dict) -> None:
        cache_path = Path(job["detection_dir"]) / "model_tracks.npy"
        try:
            tracks = np.load(str(cache_path), allow_pickle=True).item()
            if isinstance(tracks, dict) and "tracks" in tracks:
                job["tracks"] = tracks["tracks"]
                job["hand_counts"] = np.array(tracks.get("hand_counts", []), dtype=np.int16)
                # 新缓存含左右计数；旧缓存缺失则为 None（_find_keep_segments 会退回按总数切）
                if "n_left" in tracks and "n_right" in tracks:
                    job["n_left"] = np.array(tracks["n_left"], dtype=np.int16)
                    job["n_right"] = np.array(tracks["n_right"], dtype=np.int16)
            else:
                job["tracks"] = tracks
                job["hand_counts"] = np.zeros(int(job.get("n_frames", 0)), dtype=np.int16)
            job["t_detect"] = 0.0
            job["ts_detect_start"] = time.time()
            job["gpu_id"] = gpu_id
            job["actual_yolo_batch"] = 0
            job["micro_batches"] = 0
        except Exception as e:
            job["ok"] = False
            job["error"] = f"cache load: {type(e).__name__}: {e}"
        out_q.put(job)

    def _enqueue(job: dict, pending_states: list[dict]) -> None:
        if not job.get("ok"):
            out_q.put(job)
            return

        if job.get("detect_cached"):
            _emit_cached(job)
            return

        shm_path = job.get("frames_shm")
        if not shm_path:
            job["ok"] = False
            job["error"] = "missing frames_shm"
            out_q.put(job)
            return

        try:
            frames, _ = _open_shm_frames(
                shm_path, job["frames_shape"], job["frames_dtype"]
            )
        except Exception as e:
            job["ok"] = False
            job["error"] = f"open SHM {shm_path}: {type(e).__name__}: {e}"
            out_q.put(job)
            return

        job["ts_detect_start"] = time.time()
        pending_states.append({
            "job": job,
            "shm_path": shm_path,
            "frames": frames,
            "n_frames": int(job["n_frames"]),
            "cursor": 0,
            "frame_dets": [],
            "batch_sizes": [],
        })

    def _fill_pending(pending_states: list[dict]) -> bool:
        should_exit = False
        deadline = time.monotonic() + max(0, micro_batch_timeout_ms) / 1000.0

        def _remaining_frames() -> int:
            return sum(s["n_frames"] - s["cursor"] for s in pending_states)

        while _remaining_frames() < max_frames_inflight:
            timeout = max(0.0, deadline - time.monotonic())
            if pending_states and timeout <= 0:
                break
            try:
                job = in_q.get(timeout=timeout if pending_states else None)
            except queue.Empty:
                break

            if job is None:
                should_exit = True
                break
            _enqueue(job, pending_states)

        return should_exit

    # warmup
    try:
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        _ = model.predict(dummy, imgsz=imgsz, conf=det_thresh, half=half,
                          verbose=False, device=device, batch=1)
        del dummy
        _release_cuda_memory()
    except Exception as e:
        log(f"YOLO-GPU{gpu_id}", f"warmup failed (忽略): {e}")

    log(f"YOLO-GPU{gpu_id}", "就绪")

    pending_states: list[dict] = []
    should_exit = False
    while True:
        if not pending_states:
            job = in_q.get()
            if job is None:
                _release_cuda_memory()
                log(f"YOLO-GPU{gpu_id}", f"退出 pid={os.getpid()}")
                return
            _enqueue(job, pending_states)
            if not pending_states:
                continue

        should_exit = _fill_pending(pending_states)
        current_batch = min(infer_batch, max_frames_inflight)
        try:
            while True:
                micro_batch, pending_states, completed = _take_micro_batch(
                    pending_states, current_batch
                )
                if not micro_batch:
                    break

                batch_imgs = [state["frames"][fi] for state, fi in micro_batch]
                with torch.inference_mode():
                    for (state, _fi), r in zip(
                        micro_batch,
                        model.predict(
                            batch_imgs, conf=det_thresh, stream=True,
                            batch=len(batch_imgs), imgsz=imgsz, half=half,
                            verbose=False, device=device,
                        ),
                    ):
                        # 单次 D2H：boxes.data = (n, 6) [x1,y1,x2,y2,conf,cls]
                        data = r.boxes.data.cpu().numpy()
                        if data.size:
                            state["frame_dets"].append(
                                (data[:, :4].copy(),
                                 data[:, 4].copy(),
                                 data[:, 5].copy())
                            )
                        else:
                            state["frame_dets"].append(
                                (np.zeros((0, 4), dtype=np.float32),
                                 np.zeros(0, dtype=np.float32),
                                 np.zeros(0, dtype=np.float32))
                            )
                del batch_imgs

                for state in completed:
                    job = state["job"]
                    job["frame_dets"] = state["frame_dets"]
                    job["n_frames"] = state["n_frames"]
                    job["t_detect"] = time.time() - job["ts_detect_start"]
                    job["ok"] = True
                    job["gpu_id"] = gpu_id
                    job["yolo_batch"] = current_batch
                    job["actual_yolo_batch"] = max(state["batch_sizes"]) if state["batch_sizes"] else 0
                    job["micro_batches"] = len(state["batch_sizes"])
                    _release_state(state)
                    out_q.put(job)

                if not pending_states:
                    break

            if should_exit and not pending_states:
                _release_cuda_memory()
                log(f"YOLO-GPU{gpu_id}", f"退出 pid={os.getpid()}")
                return
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                import traceback
                for state in pending_states:
                    state["job"]["ok"] = False
                    state["job"]["error"] = (
                        f"RuntimeError: gpu={gpu_id} batch={current_batch} imgsz={imgsz} half={half} "
                        f'stem={state["job"]["stem"]}: {e}\n{traceback.format_exc()[-400:]}'
                    )
                    _release_state(state)
                    out_q.put(state["job"])
                pending_states = []
            else:
                _release_cuda_memory()
                if current_batch == 1:
                    import traceback
                    for state in pending_states:
                        state["job"]["ok"] = False
                        state["job"]["error"] = (
                            f"OOM: gpu={gpu_id} batch=1 imgsz={imgsz} half={half} "
                            f'n_frames={state["n_frames"]} stem={state["job"]["stem"]}: {e}\n'
                            f"{traceback.format_exc()[-400:]}"
                        )
                        _release_state(state)
                        out_q.put(state["job"])
                    pending_states = []
                else:
                    next_batch = max(1, current_batch // 2)
                    log(f"YOLO-GPU{gpu_id}", f"聚合微批 OOM，batch={current_batch} -> {next_batch}")
                    infer_batch = next_batch
                    # 保持 SHM 打开，重置 cursor 重试
                    for state in pending_states:
                        state["cursor"] = 0
                        state["frame_dets"] = []
                        state["batch_sizes"] = []
            _release_cuda_memory()
        except Exception as e:
            import traceback
            for state in pending_states:
                state["job"]["ok"] = False
                state["job"]["error"] = (
                    f'{type(e).__name__}: gpu={gpu_id} stem={state["job"]["stem"]}: {e}\n'
                    f"{traceback.format_exc()[-400:]}"
                )
                _release_state(state)
                out_q.put(state["job"])
            pending_states = []
            _release_cuda_memory()


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 3：ByteTrack 关联 + 切点计算（CPU 进程池，不占 GPU）
# ═══════════════════════════════════════════════════════════════════════════════

def _bytetrack_assoc(frame_dets: list, thresh: float) -> dict:
    """将 detect 结果做 BYTETracker 关联，返回 tracks dict。"""
    from ultralytics.trackers.byte_tracker import BYTETracker
    import argparse as _ap
    cfg = _ap.Namespace(
        track_high_thresh=thresh,
        track_low_thresh=max(0.05, thresh * 0.3),
        new_track_thresh=thresh,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
    )
    tracker = BYTETracker(args=cfg, frame_rate=30)
    tracks: dict = {}

    sys.path.insert(0, str(_HAWOR_DIR))
    from lib.pipeline.tools import _DetAdaptor  # noqa: E402

    for t, (bboxes, confs, classes) in enumerate(frame_dets):
        find_right = find_left = False
        if len(bboxes) == 0:
            continue
        out = tracker.update(_DetAdaptor(bboxes, confs, classes), img=None)
        if out is None or len(out) == 0:
            continue
        for row in out:
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            tid = int(row[4])
            score = float(row[5])
            cls_val = float(row[6])
            subj = {
                "frame": t, "det": True,
                "det_box": np.array([[x1, y1, x2, y2, score]]),
                "det_handedness": np.array([cls_val]),
            }
            if (not find_right and cls_val > 0) or (not find_left and cls_val == 0):
                tracks.setdefault(tid, []).append(subj)
                if cls_val > 0:
                    find_right = True
                else:
                    find_left = True
    return tracks


def _count_detected_hands(frame_dets: list) -> np.ndarray:
    return np.array([len(bboxes) for bboxes, *_ in frame_dets], dtype=np.int16)


def _count_handedness(frame_dets: list) -> tuple[np.ndarray, np.ndarray]:
    """每帧的左手数 / 右手数（detector 类别 0=left, 1=right）。"""
    n_left, n_right = [], []
    for _bboxes, _confs, classes in frame_dets:
        arr = np.asarray(classes).reshape(-1)
        n_left.append(int(np.sum(arr == 0)))
        n_right.append(int(np.sum(arr > 0)))
    return np.array(n_left, dtype=np.int16), np.array(n_right, dtype=np.int16)


def _pad_to(arr, n_frames: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int16)
    if len(arr) < n_frames:
        out = np.zeros(n_frames, dtype=np.int16)
        out[:len(arr)] = arr
        return out
    return arr[:n_frames]


def _find_keep_segments(
    tracks: dict, hand_counts: np.ndarray, n_frames: int, det_fps: float, duration: float,
    min_absent_sec: float, min_segment_sec: float, filter_multi_hands: bool,
    require_lr_pair: bool = False,
    n_left: np.ndarray | None = None, n_right: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    any_hand = np.zeros(n_frames, dtype=bool)
    for track in tracks.values():
        for fd in track:
            f = fd["frame"]
            if f < n_frames:
                any_hand[f] = True

    padded_hand_counts = _pad_to(hand_counts, n_frames)
    cut_mask = ~any_hand
    if filter_multi_hands:
        cut_mask |= padded_hand_counts > 2
    if require_lr_pair:
        if n_left is not None and n_right is not None:
            nl = _pad_to(n_left, n_frames)
            nr = _pad_to(n_right, n_frames)
            # 保留: 左≤1 且 右≤1 且 至少一只手；否则切（0手/≥3手/同侧重复）
            valid = (nl <= 1) & (nr <= 1) & ((nl + nr) >= 1)
            cut_mask |= ~valid
        else:
            # 旧缓存无左右计数：退回按总数 >2 切（无法判同侧重复）
            cut_mask |= padded_hand_counts > 2

    min_frames = max(1, int(min_absent_sec * det_fps))
    pad = np.concatenate(([False], cut_mask, [False])).astype(np.int8)
    diff = np.diff(pad)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    cut_times = [(float(s / det_fps), float(e / det_fps))
                 for s, e in zip(starts, ends) if (e - s) >= min_frames]

    if not cut_times:
        segs = [(0.0, duration)]
    else:
        segs, cursor = [], 0.0
        for cs, ce in cut_times:
            if cursor < cs:
                segs.append((cursor, cs))
            cursor = ce
        if cursor < duration:
            segs.append((cursor, duration))

    keep = [(s, e) for s, e in segs if e - s >= min_segment_sec]
    return keep


def _tracker_pool_worker(
    in_q: mp.Queue, out_q: mp.Queue,
    det_thresh: float, min_absent_sec: float, min_segment_sec: float,
    filter_multi_hands: bool, require_lr_pair: bool = False,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _set_parent_death_signal()
    while True:
        job = in_q.get()
        if job is None:
            return
        if not job.get("ok"):
            out_q.put(job)
            continue

        frame_dets = None
        try:
            if job.get("detect_cached"):
                tracks = job["tracks"]
                hand_counts = np.array(job.get("hand_counts", []), dtype=np.int16)
                n_left = job.get("n_left")
                n_right = job.get("n_right")
            else:
                frame_dets = job.get("frame_dets")
                if not frame_dets:
                    raise FileNotFoundError(f'frame_dets missing for {job.get("stem")}')
                hand_counts = _count_detected_hands(frame_dets)
                n_left, n_right = _count_handedness(frame_dets)
                t0 = time.time()
                tracks = _bytetrack_assoc(frame_dets, det_thresh)
                job["t_track"] = time.time() - t0
                cache_path = Path(job["detection_dir"]) / "model_tracks.npy"
                try:
                    Path(job["detection_dir"]).mkdir(parents=True, exist_ok=True)
                    np.save(str(cache_path), {
                        "tracks": tracks,
                        "hand_counts": hand_counts,
                        "n_left": n_left,
                        "n_right": n_right,
                    })
                except Exception:
                    pass
                job.pop("frame_dets", None)

            keep_segs = _find_keep_segments(
                tracks, hand_counts, job["n_frames"], job["det_fps_use"], job["duration"],
                min_absent_sec, min_segment_sec, filter_multi_hands,
                require_lr_pair, n_left, n_right,
            )
            job["keep_segs"] = keep_segs
            out_q.put(job)
        except Exception as e:
            import traceback
            job["ok"] = False
            job["error"] = f"track: {type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"
            out_q.put(job)
        finally:
            del frame_dets


# ═══════════════════════════════════════════════════════════════════════════════
# 阶段 4：ffmpeg stream-copy 切片（CPU 进程池）
# ═══════════════════════════════════════════════════════════════════════════════
# 前置假设：源视频已经过 process_densegop.sh 预处理，关键帧间隔 ≤ 0.5s。
# 这样 stream-copy 切片的对齐误差被压在 ±0.5s 以内，远小于原片 ±8s 的 GOP。
# 若用未预处理的源（GOP > 1s），会有可见的"段比文件名长"现象，请先跑 densegop。

def _split_one_video(
    video_path: str, keep_segs: list, out_subdir: str, copy_audio: bool, stem: str,
) -> list[dict]:
    import concurrent.futures as _fu
    if not keep_segs:
        return []
    seg_dir = Path(out_subdir) / stem
    seg_dir.mkdir(parents=True, exist_ok=True)

    def _cut(idx, t_start, t_end):
        dur = t_end - t_start
        dst = seg_dir / f"{stem}_part{idx:03d}_{t_start:.2f}s_{t_end:.2f}s.mp4"
        if dst.exists() and dst.stat().st_size > 0:
            return True, idx, t_start, t_end, dur, str(dst), ""
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-ss", f"{t_start:.6f}", "-i", str(video_path),
               "-t", f"{dur:.6f}", "-c:v", "copy"]
        cmd += ["-c:a", "copy"] if copy_audio else ["-an"]
        cmd += ["-avoid_negative_ts", "make_zero", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0, idx, t_start, t_end, dur, str(dst), r.stderr

    segs = []
    with _fu.ThreadPoolExecutor(max_workers=min(4, len(keep_segs))) as ex:
        futs = [ex.submit(_cut, i, s, e) for i, (s, e) in enumerate(keep_segs)]
        for ok, idx, ts, te, dur, path, _err in [f.result() for f in futs]:
            if ok:
                segs.append({"index": idx, "start_sec": round(ts, 4),
                             "end_sec": round(te, 4), "duration_sec": round(dur, 4),
                             "path": path})
    return sorted(segs, key=lambda x: x["index"])


def _append_manifest(manifest_path: str, stem: str, out_subdir: str,
                     n_parts: int) -> None:
    """向 <output_root>/.completed.jsonl 追加一条完成记录。

    Linux POSIX 下 `open(..., 'a')` 单次 `write(<PIPE_BUF=4096B)` 是原子的，
    所以多 split worker 并发 append 不会撕裂行。每条记录 < 4KB 远低于此阈值。
    """
    line = json.dumps({
        "stem": stem,
        "out_subdir": out_subdir,
        "parts": n_parts,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False) + "\n"
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(line)


def _load_manifest(manifest_path: Path) -> set[tuple[str, str]]:
    """读取 manifest 返回 (out_subdir, stem) 已完成集合; 文件不存在返回空集。

    用 (out_subdir, stem) 而非裸 stem 作 key, 因为同 stem 可能出现在不同
    rel 子目录下 (典型: egodex 各 category 都有 0.mp4)。
    """
    done: set[tuple[str, str]] = set()
    if not manifest_path.is_file():
        return done
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["out_subdir"], rec["stem"]))
            except Exception:
                # 损坏行(并发撕裂或人工编辑)跳过不抛, 大不了重做一次
                continue
    return done


def _split_pool_worker(
    in_q: mp.Queue, out_q: mp.Queue, copy_audio: bool, keep_work: bool, dry_run: bool,
    manifest_path: str,
):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _set_parent_death_signal()
    while True:
        job = in_q.get()
        if job is None:
            return
        if not job.get("ok"):
            out_q.put(job)
            continue
        try:
            if dry_run:
                job["segments"] = job.get("keep_segs", [])
                job["t_split"] = 0.0
                out_q.put(job)
                continue
            t0 = time.time()
            segs = _split_one_video(
                job["video"], job["keep_segs"], job["out_subdir"], copy_audio, job["stem"],
            )
            job["segments"] = segs
            job["t_split"] = time.time() - t0

            # 集中 manifest 记录完成 (替代旧的 per-stem .done 文件)
            _append_manifest(manifest_path, job["stem"], job["out_subdir"], len(segs))
            if not keep_work:
                shutil.rmtree(job["video_work"], ignore_errors=True)
            out_q.put(job)
        except Exception as e:
            import traceback
            job["ok"] = False
            job["error"] = f"split: {type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"
            out_q.put(job)


# ═══════════════════════════════════════════════════════════════════════════════
# 主调度
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_runtime_profile(args, n_videos: int, gpu_ids: list[int],
                             available_cpu_count: int) -> dict:
    nproc = available_cpu_count
    n_videos = max(1, n_videos)
    n_gpu = max(1, len(gpu_ids))
    profile = args.profile

    if profile == "aggressive":
        batch = args.batch if args.batch is not None else 64
        # 抽帧是 CPU + IO 限制，瓶颈和 GPU 数无关；按 nproc 分配，保底 n_gpu*2 与 8。
        default_extract = min(n_videos, max(n_gpu * 2, nproc // 4, 8))
        extract_workers = args.extract_workers or default_extract
        tracker_workers = args.tracker_workers or min(n_videos, max(8, n_gpu * 2, nproc // 6))
        split_workers = args.split_workers or min(n_videos, max(8, n_gpu * 2, nproc // 6))
        extract_queue = args.extract_queue_size or max(128, batch * n_gpu, extract_workers * 2)
        yolo_queue = args.yolo_queue_size or max(32, n_gpu * 8, tracker_workers * 2)
        track_queue = args.track_queue_size or max(32, split_workers * 2)
        micro_batch_timeout_ms = args.micro_batch_timeout_ms if args.micro_batch_timeout_ms is not None else 40
        max_frames_inflight = args.max_frames_inflight if args.max_frames_inflight is not None else batch * 2
    else:
        batch = args.batch if args.batch is not None else 32
        extract_workers = args.extract_workers or min(n_videos, max(n_gpu * 4, nproc // 3, 4))
        tracker_workers = args.tracker_workers or min(n_videos, max(n_gpu, nproc // 8, 2))
        split_workers = args.split_workers or min(n_videos, max(n_gpu, nproc // 8, 2))
        extract_queue = args.extract_queue_size or max(batch * n_gpu, extract_workers * 2, 16)
        yolo_queue = args.yolo_queue_size or max(n_gpu * 4, tracker_workers * 2, 8)
        track_queue = args.track_queue_size or max(split_workers * 2, 8)
        micro_batch_timeout_ms = args.micro_batch_timeout_ms if args.micro_batch_timeout_ms is not None else 80
        max_frames_inflight = args.max_frames_inflight if args.max_frames_inflight is not None else batch * 2

    return {
        "profile": profile,
        "nproc": nproc,
        "n_gpu": n_gpu,
        "batch": batch,
        "extract_workers": extract_workers,
        "tracker_workers": tracker_workers,
        "split_workers": split_workers,
        "extract_queue_size": extract_queue,
        "yolo_queue_size": yolo_queue,
        "track_queue_size": track_queue,
        "micro_batch_timeout_ms": micro_batch_timeout_ms,
        "max_frames_inflight": max(batch, max_frames_inflight),
    }


def process_videos(items: list[tuple[Path, Path]], args) -> list[ProcessResult]:
    """批处理入口（main.py 优先调用本函数）。

    items: list[(video, out_subdir)]
        out_subdir 是 main.py 计算的 `<output_root>/<rel>`，本 op 直接把
        段写到 out_subdir/<stem>/...，与 reprocessed 的 _op_base(rel) 语义对齐。
    返回 list[ProcessResult]，按完成顺序。
    """
    from script_utils.resource_monitor import ResourceMonitor  # noqa: E402

    monitor = None
    interrupted = False

    output_root = Path(getattr(args, "output_dir", "") or ".").expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir \
        else output_root / ".work"
    # 集中 manifest: 用一行 JSONL 记录每个 stem 的完成状态, 取代旧的散落 .done 文件
    manifest_path = output_root / ".completed.jsonl"

    # SHM 命名空间：所有抽帧 worker 共用一个 session_prefix，便于残留兜底清理
    session_prefix = f"hand_filter_{os.getpid()}_{_uuid.uuid4().hex[:8]}"

    # GPU 解析
    if args.gpus == "auto":
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, check=True,
            )
            gpu_ids = [int(x.strip()) for x in r.stdout.strip().splitlines() if x.strip()]
        except Exception:
            gpu_ids = [0]
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    gpu_ids = list(dict.fromkeys(gpu_ids))

    total_cpu_count = os.cpu_count() or 0
    available_cpu_count = get_available_cpu_count()
    log("init", f"GPU: {gpu_ids}")
    log("init", f"CPU 总核数={total_cpu_count}  当前进程可用核数={available_cpu_count}")

    n_videos = len(items)
    watch_mode = bool(getattr(args, "watch", False))
    if not n_videos and not watch_mode:
        log("init", "[错误] 未提供视频")
        return []
    if watch_mode:
        log("init", f"[watch] 常驻流式：初始 {n_videos} 段，将持续轮询新段直到 .stream_done")
    else:
        log("init", f"{n_videos} 个视频")

    # watch 模式段数会持续增长：用 nominal 计数给 profile，避免 worker 池被初始小段数卡死
    _profile_n = max(n_videos, 64) if watch_mode else n_videos
    runtime = _resolve_runtime_profile(args, _profile_n, gpu_ids, available_cpu_count)
    n_gpu = runtime["n_gpu"]
    batch = runtime["batch"]
    n_extract = runtime["extract_workers"]
    n_tracker = runtime["tracker_workers"]
    n_split = runtime["split_workers"]

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
    log("init", f"输出: {output_root}")
    log("init", "抽帧走 SharedMemory（零落盘），YOLO 直读")
    log("init", "切分规则: 无手区间必切"
        + ("，超过两只手区间也切" if args.filter_multi_hands else "，默认不过滤多手区间"))
    log("init",
        f'profile={runtime["profile"]}  cpu={runtime["nproc"]}  gpu={gpu_ids}  '
        f'batch={batch}  inflight={runtime["max_frames_inflight"]}  '
        f'micro_batch_timeout_ms={runtime["micro_batch_timeout_ms"]}  '
        f"extract×{n_extract}  tracker×{n_tracker}  split×{n_split}")
    log("init",
        f'queue: extract→yolo={runtime["extract_queue_size"]}  '
        f'yolo→tracker={runtime["yolo_queue_size"]}  tracker→split={runtime["track_queue_size"]}')
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

    monitor = ResourceMonitor(output_root, gpu_ids=gpu_ids, interval=0.5)
    monitor.start()

    try:
        ctx = mp.get_context("spawn")
        q_extract_in = ctx.Queue()
        q_extract_out = ctx.Queue(maxsize=runtime["extract_queue_size"])
        q_yolo_out = ctx.Queue(maxsize=runtime["yolo_queue_size"])
        q_track_out = ctx.Queue(maxsize=runtime["track_queue_size"])
        q_split_out = ctx.Queue()

        procs: list[mp.Process] = []
        proc_meta: dict[int, dict] = {}

        extract_procs: list[mp.Process] = []
        yolo_procs: list[mp.Process] = []
        tracker_procs: list[mp.Process] = []
        split_procs: list[mp.Process] = []

        for idx in range(n_extract):
            p = ctx.Process(target=_extract_pool_worker,
                            args=(q_extract_in, q_extract_out, args.det_fps, session_prefix),
                            daemon=False)
            p.start(); extract_procs.append(p); procs.append(p)
            proc_meta[p.pid] = {"stage": "extract", "worker": idx}

        for gpu_id in gpu_ids:
            p = ctx.Process(
                target=_yolo_actor_main,
                args=(gpu_id, q_extract_out, q_yolo_out,
                      args.det_thresh, batch, args.imgsz, not args.no_half,
                      runtime["micro_batch_timeout_ms"], runtime["max_frames_inflight"]),
                daemon=False,
            )
            p.start(); yolo_procs.append(p); procs.append(p)
            proc_meta[p.pid] = {"stage": "yolo", "gpu_id": gpu_id}

        for idx in range(n_tracker):
            p = ctx.Process(
                target=_tracker_pool_worker,
                args=(q_yolo_out, q_track_out,
                      args.det_thresh, args.min_absent_sec, args.min_segment_sec,
                      args.filter_multi_hands, args.require_lr_pair),
                daemon=False,
            )
            p.start(); tracker_procs.append(p); procs.append(p)
            proc_meta[p.pid] = {"stage": "tracker", "worker": idx}

        for idx in range(n_split):
            p = ctx.Process(
                target=_split_pool_worker,
                args=(q_track_out, q_split_out,
                      not args.no_copy_audio, args.keep_work, args.dry_run,
                      str(manifest_path)),
                daemon=False,
            )
            p.start(); split_procs.append(p); procs.append(p)
            proc_meta[p.pid] = {"stage": "split", "worker": idx}

        def _kill_descendants(root_pid: int) -> int:
            """递归 kill 所有后代进程（包括 ffmpeg 等 grandchildren）。"""
            n = 0
            try:
                out = subprocess.run(["pgrep", "-P", str(root_pid)],
                                     capture_output=True, text=True, timeout=5)
                children = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
            except Exception:
                children = []
            for cpid in children:
                n += _kill_descendants(cpid)
                try:
                    os.kill(cpid, signal.SIGKILL)
                    n += 1
                except ProcessLookupError:
                    pass
                except Exception:
                    pass
            return n

        def _terminate_all():
            # 1) 先给 worker 发 SIGTERM，给它们机会清理 ffmpeg 和半写 SHM
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass
            for p in procs:
                try:
                    p.join(timeout=3)
                    if p.is_alive():
                        p.kill()
                except Exception:
                    pass
            # 2) 兜底：递归 kill 所有还活着的后代（孤儿 ffmpeg / 失控 worker）
            n_killed = _kill_descendants(os.getpid())
            if n_killed:
                log("sig", f"递归 kill 后代进程 {n_killed} 个")

        def _shutdown(sig, _frame):
            nonlocal interrupted, monitor
            interrupted = True
            log("sig", f"收到信号 {sig}，正在停止监控并退出")
            if monitor is not None:
                try:
                    monitor.stop()
                except Exception as e:
                    log("sig", f"monitor.stop 失败: {e}")
            _terminate_all()
            n_cleaned = _cleanup_session_shm(session_prefix)
            if n_cleaned:
                log("sig", f"清理 /dev/shm 残留 {n_cleaned} 块")
            sys.exit(1)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGHUP, _shutdown)

        t_wall = time.time()

        results: list[dict] = []

        def _shepherd(upstream: list[mp.Process], downstream_q: mp.Queue,
                      n_downstream: int, name: str):
            for p in upstream:
                p.join()
            for _ in range(n_downstream):
                downstream_q.put(None)

        state_lock = _th.Lock()
        feeder_state = {
            "seen_items": 0,
            "submitted_items": 0,
            "skipped_done": 0,
            "done_items": 0,
            "finished": False,
            "error": None,
        }

        # 用 items 而非自己扫源目录：main.py 已经做了 collect_videos 与 rel 映射
        items_to_feed = [(Path(v), Path(out_subdir)) for v, out_subdir in items]
        skipped_results: list[ProcessResult] = []
        # 启动时加载已完成集合 (out_subdir, stem) — 集中 manifest 取代散落 .done
        completed_set = _load_manifest(manifest_path)
        log("init", f"manifest: {manifest_path}  已记录 {len(completed_set)} 个完成项")

        import re as _re
        _PART_TMP_RE = _re.compile(r"_part\d{3,}\.(mp4|mkv|webm)$", _re.IGNORECASE)

        def _submit_one(video: Path, out_subdir: Path) -> bool:
            """构造 job 并投入抽帧队列；已完成则跳过。返回是否真正提交。"""
            stem = video.stem
            with state_lock:
                feeder_state["seen_items"] += 1
            legacy_done = out_subdir / stem / f"{stem}.done"
            if (str(out_subdir), stem) in completed_set or legacy_done.exists():
                with state_lock:
                    feeder_state["skipped_done"] += 1
                skipped_results.append(
                    ProcessResult(video, 0, 1, 0, "manifest/.done 已记录，跳过"))
                return False
            try:
                rel_parent = Path(out_subdir).resolve().relative_to(output_root)
            except ValueError:
                rel_parent = Path()
            video_work = work_dir / rel_parent / stem
            job = {
                "video": str(video),
                "stem": stem,
                "out_subdir": str(out_subdir),
                "video_work": str(video_work),
                "frames_dir": str(video_work / "frames"),
                "detection_dir": str(video_work / "detection"),
            }
            q_extract_in.put(job)
            with state_lock:
                feeder_state["submitted_items"] += 1
            return True

        def _feed_fixed_jobs():
            """普通模式：喂固定 items 列表，喂完投哨兵。"""
            try:
                for video, out_subdir in items_to_feed:
                    _submit_one(Path(video), Path(out_subdir))
            except Exception as e:
                with state_lock:
                    feeder_state["error"] = str(e)
            finally:
                for _ in range(n_extract):
                    q_extract_in.put(None)
                with state_lock:
                    feeder_state["finished"] = True

        def _feed_watch_jobs():
            """watch 模式：持续轮询 input_root 找新段，直到 <output_root>/.stream_done。
            out_subdir = output_root / (video.parent 相对 input_root)，与 main.py 一致。"""
            in_paths = args.input if isinstance(args.input, (list, tuple)) else [args.input]
            input_root = Path(in_paths[0]).expanduser().resolve()
            done_file = output_root / ".stream_done"
            submitted: set = set()
            log("watch", f"轮询 {input_root}  (停止哨兵: {done_file})")
            try:
                while True:
                    n_new = 0
                    if input_root.exists():
                        for video in input_root.rglob("*.mp4"):
                            if _PART_TMP_RE.search(video.name):
                                continue  # 跳过未改名的临时切分文件
                            try:
                                rel = video.parent.resolve().relative_to(input_root)
                            except ValueError:
                                continue
                            out_subdir = output_root if str(rel) in ("", ".") else output_root / rel
                            key = (str(out_subdir), video.stem)
                            if key in submitted:
                                continue
                            submitted.add(key)
                            if _submit_one(video, out_subdir):
                                n_new += 1
                    if done_file.exists() and n_new == 0:
                        log("watch", "收到 .stream_done 且无新段，收尾。")
                        break
                    time.sleep(max(0.2, getattr(args, "watch_poll", 2.0)))
            except Exception as e:
                with state_lock:
                    feeder_state["error"] = str(e)
            finally:
                for _ in range(n_extract):
                    q_extract_in.put(None)
                with state_lock:
                    feeder_state["finished"] = True

        _feed_extract_jobs = _feed_watch_jobs if watch_mode else _feed_fixed_jobs

        feeder = _th.Thread(target=_feed_extract_jobs, daemon=True)
        feeder.start()

        shepherds = [
            _th.Thread(target=_shepherd, args=(extract_procs, q_extract_out, len(yolo_procs),    "extract→yolo"), daemon=True),
            _th.Thread(target=_shepherd, args=(yolo_procs,    q_yolo_out,    len(tracker_procs), "yolo→tracker"), daemon=True),
            _th.Thread(target=_shepherd, args=(tracker_procs, q_track_out,   len(split_procs),   "tracker→split"), daemon=True),
        ]
        for t in shepherds:
            t.start()

        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        bar = None
        if _tqdm is not None and not args.dry_run:
            bar = _tqdm(total=(None if watch_mode else n_videos), desc="handFlt", unit="vid",
                       smoothing=0.05, dynamic_ncols=True, file=sys.stderr)

        def _bar_write(msg: str) -> None:
            if bar is not None:
                bar.write(msg, file=sys.stderr)
            else:
                log("main", msg)

        failed: list[dict] = []
        abnormal_shutdown = None
        total_parts_so_far = 0
        while True:
            dead_workers = []
            for p in procs:
                if p.exitcode is not None and p.exitcode != 0:
                    meta = proc_meta.get(p.pid, {})
                    dead_workers.append((p.pid, p.exitcode, meta))
            if dead_workers:
                pid, exitcode, meta = dead_workers[0]
                abnormal_shutdown = (pid, exitcode, meta)
                _bar_write(f"[worker 异常退出] pid={pid} exitcode={exitcode} meta={meta}")
                break

            with state_lock:
                feeder_error = feeder_state["error"]
                feeder_finished = feeder_state["finished"]
                submitted_items = feeder_state["submitted_items"]
                done_items = feeder_state["done_items"]
                skipped_done = feeder_state["skipped_done"]
            if feeder_error:
                abnormal_shutdown = (-1, -1, {"stage": "feeder", "error": feeder_error})
                _bar_write(f"[feeder 异常] {feeder_error}")
                break

            # 让进度条吸收 skipped_done（feeder 不会推到 q_split_out）
            if bar is not None:
                target_done = done_items + skipped_done
                delta = target_done - bar.n
                if delta > 0:
                    bar.update(delta)

            try:
                r = q_split_out.get(timeout=2)
            except queue.Empty:
                if feeder_finished and done_items >= submitted_items:
                    break
                continue
            if r is None:
                continue
            with state_lock:
                feeder_state["done_items"] += 1
                done_items = feeder_state["done_items"]
                submitted_items = feeder_state["submitted_items"]
            if not r.get("ok"):
                failed.append(r)
                _bar_write(
                    f'[失败] {r.get("stem","?")}: '
                    f'{str(r.get("error","")).splitlines()[-1][:160]}'
                )
            else:
                n_seg = len(r.get("segments", []) or r.get("keep_segs", []))
                total_parts_so_far += n_seg
                wait_detect = 0.0
                if r.get("ts_extract_done") and r.get("ts_detect_start"):
                    wait_detect = max(0.0, r["ts_detect_start"] - r["ts_extract_done"])
                r["wait_gpu"] = round(wait_detect, 4)
                results.append(r)
            if bar is not None:
                bar.update(1)
                bar.set_postfix(
                    ok=len(results), fail=len(failed),
                    skip=skipped_done, parts=total_parts_so_far,
                    refresh=False,
                )

        if abnormal_shutdown is not None:
            pid, exitcode, meta = abnormal_shutdown
            failed.append({
                "ok": False,
                "stem": meta.get("stage", "?"),
                "error": meta.get("error") or f"worker exited unexpectedly: pid={pid} exitcode={exitcode} meta={meta}",
            })
            _terminate_all()
        else:
            for p in procs:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()

        elapsed = time.time() - t_wall

        if bar is not None:
            bar.close()

        # 兜底清理本 session 残留的 SHM（正常路径 consumer 已 unlink）
        n_cleaned = _cleanup_session_shm(session_prefix)
        if n_cleaned:
            log("done", f"清理 /dev/shm 残留 {n_cleaned} 块")

        if not args.keep_work and not args.dry_run:
            try:
                work_dir.rmdir()
            except OSError:
                pass

        with state_lock:
            seen_items = feeder_state["seen_items"]
            submitted_items = feeder_state["submitted_items"]
            skipped_done = feeder_state["skipped_done"]
        total_parts = sum(len(r.get("segments", [])) for r in results)
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        elapsed_str = f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")
        avg_per_video = elapsed / max(1, submitted_items)
        log("done", f'{"="*50}')
        log("done",
            f"扫描={seen_items}  提交={submitted_items}  跳过已完成={skipped_done}  "
            f"成功={len(results)}  失败={len(failed)}  总段数={total_parts}")
        log("done", f"总耗时: {elapsed_str}  ({elapsed:.1f}s)  平均 {avg_per_video:.1f}s/视频")
        tot_extract = tot_detect = tot_track = tot_split = 0.0
        if results:
            tot_extract = sum(r.get("t_extract", 0) for r in results)
            tot_detect = sum(r.get("t_detect", 0) for r in results)
            tot_track = sum(r.get("t_track", 0) for r in results)
            tot_split = sum(r.get("t_split", 0) for r in results)
            log("done", f"累计 CPU 时间：extract={tot_extract:.1f}s  "
                        f"detect(GPU)={tot_detect:.1f}s  track(CPU)={tot_track:.1f}s  "
                        f"split={tot_split:.1f}s  "
                        f"(并行因子={(tot_extract+tot_detect+tot_track+tot_split)/max(elapsed,1e-6):.1f}x)")
        for r in failed:
            log("done", f'  错误 {r.get("stem")}: {str(r.get("error", ""))[-200:]}')

        if not args.dry_run:
            summary = output_root / "summary.json"
            payload = {
                "num_seen": seen_items,
                "num_skipped_done": skipped_done,
                "num_videos": submitted_items,
                "num_success": len(results),
                "num_failed": len(failed),
                "num_parts": total_parts,
                "total_sec": round(elapsed, 2),
                "total_str": elapsed_str,
                "avg_sec_per_video": round(avg_per_video, 2),
                "cpu_time_sec": {
                    "extract": round(tot_extract, 2),
                    "detect_gpu": round(tot_detect, 2),
                    "track": round(tot_track, 2),
                    "split": round(tot_split, 2),
                },
                "results": results + failed,
                "started_at": datetime.fromtimestamp(t_wall).isoformat(),
            }
            summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
            log("done", f"summary → {summary}")
    finally:
        if monitor is not None and not interrupted:
            monitor.stop()

    # 汇总 ProcessResult
    out_results: list[ProcessResult] = list(skipped_results)
    for r in results:
        n_seg = len(r.get("segments", []) or r.get("keep_segs", []))
        out_results.append(ProcessResult(Path(r["video"]), n_seg, 0, 0,
                                         f"{n_seg} 段"))
    for r in failed:
        v = r.get("video") or r.get("stem") or "?"
        out_results.append(ProcessResult(Path(str(v)), 0, 0, 1,
                                         str(r.get("error", ""))[:200]))
    return out_results
