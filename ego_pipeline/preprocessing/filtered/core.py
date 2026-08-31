"""共享工具：视频元数据、文件收集、时间格式化、ffmpeg 调用、结果结构、并发/日志小工具

被 filtered/main.py 与 ops/*.py 复用。与 reprocessed/core.py 保持同构（仅去掉
不需要的重编码相关部分），方便两边维护者切换。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading as _threading
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


@dataclass
class ProcessResult:
    """模块处理单个视频后的统一返回结构

    success / skipped / failed 统计的是"输出条目数"（hand_filter 一般为
    切出的段数，跳过 .done 时为 skipped=1）。
    """
    video: Path
    success: int = 0
    skipped: int = 0
    failed: int = 0
    msg: str = ""


# ── ffprobe ───────────────────────────────────────────────────────────────────

def get_duration(path: Path) -> float:
    """返回视频时长（秒），失败返回 0.0"""
    cmd = ["ffprobe", "-v", "error",
           "-show_entries", "format=duration",
           "-of", "json", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0


_EMPTY_META = {"fps": 0.0, "width": 0, "height": 0, "duration": 0.0, "codec": ""}


def get_meta(path: Path) -> dict:
    """{fps, width, height, duration, codec}；任一字段失败时为 0/""

    一次 ffprobe 取齐：调用方不需要再单独跑额外 probe。
    """
    cmd = ["ffprobe", "-v", "error",
           "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,width,height,codec_name:format=duration",
           "-of", "json", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return dict(_EMPTY_META)
    if r.returncode != 0:
        return dict(_EMPTY_META)
    try:
        data = json.loads(r.stdout)
    except Exception:
        return dict(_EMPTY_META)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format", {})
    raw = stream.get("r_frame_rate", "0/1")
    try:
        num, den = map(int, raw.split("/"))
        fps = num / den if den else 0.0
    except Exception:
        fps = 0.0
    return {
        "fps": fps,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": float(fmt.get("duration") or 0),
        "codec": (stream.get("codec_name") or "").lower(),
    }


# ── 标签 ──────────────────────────────────────────────────────────────────────

def format_time(seconds: int) -> str:
    """整数秒 → 标签字符串（与 reprocessed/core.format_time 完全一致）

        < 60            → "{s}s"
        == 60           → "60s"
        60 < t < 3600
          整分钟        → "{m}m"
          含余数        → "{m}m{s}s"
        == 3600         → "60m"
        > 3600
          整小时        → "{h}h"
          含余数        → "{h}h[{m}m][{s}s]"
    """
    if seconds < 60:
        return f"{seconds}s"
    if seconds == 60:
        return "60s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m" if s == 0 else f"{m}m{s}s"
    if seconds == 3600:
        return "60m"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    out = f"{h}h"
    if m:
        out += f"{m}m"
    if s:
        out += f"{s}s"
    return out


# ── 文件收集 ──────────────────────────────────────────────────────────────────

def collect_videos(paths: list[str]) -> list[tuple[Path, Path]]:
    """从输入路径（目录递归 / 视频文件）收集去重后的视频列表

    返回 (video, rel_subdir) 元组列表：
        - 目录输入：rel_subdir = video.parent.relative_to(input_root)
        - 文件输入：rel_subdir = Path(".")
    rel_subdir 用于在输出端镜像源目录结构，避免不同子目录下同名 stem 撞车。
    """
    items: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for p in paths:
        pp = Path(p).expanduser().resolve()
        if not pp.exists():
            print(f"[warn] 路径不存在: {pp}", file=sys.stderr)
            continue
        if pp.is_dir():
            print(f"扫描: {pp} ...", file=sys.stderr, flush=True)
            before = len(items)
            count = 0
            for root, _dirs, files in os.walk(str(pp), followlinks=True):
                root_p = Path(root)
                rel = root_p.relative_to(pp)
                for fname in files:
                    if fname.lower().endswith(".mp4"):
                        f = root_p / fname
                        if f not in seen:
                            seen.add(f)
                            items.append((f, rel))
                            count += 1
                            if count % 50 == 0:
                                short = str(rel) if str(rel) != "." else "."
                                print(f"\r  [.mp4] 已找到: {count} 个  ({short})",
                                      file=sys.stderr, end="", flush=True)
            print(file=sys.stderr)
            mp4_found = len(items) - before
            if mp4_found > 0:
                print(f"  → 找到 {mp4_found} 个 .mp4，开始处理", file=sys.stderr, flush=True)
            else:
                print(f"  未找到 .mp4，扫描其他格式 ...", file=sys.stderr, flush=True)
                other_exts = VIDEO_EXTS - {".mp4"}
                for root, _dirs, files in os.walk(str(pp), followlinks=True):
                    root_p = Path(root)
                    rel = root_p.relative_to(pp)
                    for fname in files:
                        if Path(fname).suffix.lower() in other_exts:
                            f = root_p / fname
                            if f not in seen:
                                seen.add(f)
                                items.append((f, rel))
                                count += 1
                                if count % 50 == 0:
                                    short = str(rel) if str(rel) != "." else "."
                                    print(f"\r  已找到: {count} 个  ({short})",
                                          file=sys.stderr, end="", flush=True)
                print(file=sys.stderr)
                print(f"  → 找到 {len(items) - before} 个视频", file=sys.stderr, flush=True)
        elif pp.is_file() and pp.suffix.lower() in VIDEO_EXTS:
            if pp not in seen:
                seen.add(pp)
                items.append((pp, Path(".")))
        else:
            print(f"[warn] 跳过: {pp}", file=sys.stderr)
    return items


# ── ffmpeg 调用 ───────────────────────────────────────────────────────────────

# Ctrl+C 协作：跟踪所有在跑的 ffmpeg subprocess，收到信号后能立刻 SIGKILL
_LIVE_PROCS: set[subprocess.Popen] = set()
_LIVE_LOCK = _threading.Lock()
_INTERRUPTED = _threading.Event()


def is_interrupted() -> bool:
    """主线程已收到 Ctrl+C 信号？"""
    return _INTERRUPTED.is_set()


def kill_all_subprocesses() -> int:
    """SIGKILL 所有正在跑的 ffmpeg。返回被杀进程数。

    供 main.py 在 SIGINT handler 里调用。多次调用安全。
    """
    _INTERRUPTED.set()
    with _LIVE_LOCK:
        procs = list(_LIVE_PROCS)
    n = 0
    for p in procs:
        try:
            p.kill()
            n += 1
        except Exception:
            pass
    return n


def run_ffmpeg(cmd: list[str], timeout: int = 3600) -> tuple[int, str]:
    """运行 ffmpeg；返回 (returncode, stderr 最后一行)

    用 Popen 而非 subprocess.run，把句柄登记到 _LIVE_PROCS，
    以便 Ctrl+C 时主线程能直接 kill 所有 ffmpeg，worker 立刻返回。
    若已收到中断信号，直接早退不再起新进程。
    """
    if _INTERRUPTED.is_set():
        return (-1, "interrupted")
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
    except Exception as e:
        return (-1, str(e))

    with _LIVE_LOCK:
        _LIVE_PROCS.add(p)
    try:
        try:
            _, stderr = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            return (-1, "timeout")
    finally:
        with _LIVE_LOCK:
            _LIVE_PROCS.discard(p)

    err = (stderr or "").strip().splitlines()
    return (p.returncode, err[-1] if err else "")


# ── 并发 / 日志小工具（从 batch_split_video.py 抽出的通用基础设施）─────────

def get_available_cpu_count() -> int:
    """返回当前进程可用的 CPU 数；优先 sched_getaffinity，回退 os.cpu_count()"""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 8
    except Exception:
        return os.cpu_count() or 8


def log(prefix: str, msg: str) -> None:
    """统一前缀日志（flush=True，避免被 buffer 吞掉）"""
    print(f"[{prefix}] {msg}", flush=True)
