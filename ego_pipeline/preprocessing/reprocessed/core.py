"""共享工具：视频元数据、文件收集、时间格式化、ffmpeg 调用、结果结构

被 main.py 和各 ops/*.py 复用。
"""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


@dataclass
class ProcessResult:
    """模块处理单个视频后的统一返回结构

    success / skipped / failed 统计的是"输出条目数"（切分时是片段数，
    降采样/降分辨率时一般是 0 或 1）。
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


def get_src_codec(path: Path) -> str:
    """返回视频流 codec 名（小写），失败返回空串。如 "h264", "hevc", "av1"。"""
    cmd = ["ffprobe", "-v", "error",
           "-select_streams", "v:0",
           "-show_entries", "stream=codec_name",
           "-of", "json", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
        streams = json.loads(r.stdout).get("streams") or []
        return (streams[0].get("codec_name") or "").lower() if streams else ""
    except Exception:
        return ""


# (encoder, ext, default_pix_fmt, default_args_template)
# args_template 中 {crf}/{preset} 可被 pick_encoder 替换；未提供时使用 template 默认值
_ENCODER_TABLE: dict[str, dict] = {
    "h264": {
        "encoder": "libx264", "ext": ".mp4", "pix_fmt": "yuv420p",
        "default_crf": 18, "default_preset": "veryfast",
        "args": lambda crf, preset: ["-preset", preset, "-crf", str(crf)],
    },
    "hevc": {
        "encoder": "libx265", "ext": ".mp4", "pix_fmt": "yuv420p",
        "default_crf": 22, "default_preset": "medium",
        "args": lambda crf, preset: ["-preset", preset, "-crf", str(crf)],
    },
    "av1": {
        "encoder": "libsvtav1", "ext": ".mp4", "pix_fmt": "yuv420p",
        "default_crf": 30, "default_preset": "8",
        "args": lambda crf, preset: ["-preset", str(preset), "-crf", str(crf)],
    },
    "vp9": {
        "encoder": "libvpx-vp9", "ext": ".webm", "pix_fmt": "yuv420p",
        "default_crf": 31, "default_preset": "good",
        "args": lambda crf, preset: ["-deadline", preset, "-crf", str(crf), "-b:v", "0"],
    },
    "ffv1": {
        "encoder": "ffv1", "ext": ".mkv", "pix_fmt": None,
        "default_crf": None, "default_preset": None,
        "args": lambda crf, preset: ["-level", "3", "-coder", "1", "-context", "1",
                                       "-g", "1", "-slicecrc", "1"],
    },
}

# 把源 codec 名映射到 _ENCODER_TABLE 的 key
_SRC_TO_TARGET = {
    "h264": "h264", "avc1": "h264",
    "hevc": "hevc", "h265": "hevc",
    "av1": "av1",
    "vp9": "vp9", "vp8": "vp9",
    "ffv1": "ffv1",
}


def gop_args(gop: int | None) -> list[str]:
    """构造关键帧间隔相关的 ffmpeg 选项。

    -g <gop>          : GOP 上限（帧数）
    -keyint_min <gop> : 最小 GOP，与 -g 一致 → 严格固定间隔
    -sc_threshold 0   : 禁用 scene-cut 强插 IDR，避免变长 GOP

    gop=15 + fps=30 ≈ 0.5s 一关键帧（与 densify_keyframes.py 默认一致）。
    传 None / <=0 时返回空 list（不设置 GOP，沿用编码器默认）。

    注意：本函数返回的 -g 通常放在 -c:v <encoder> 之后、编码器专属 args 之前；
    对 ffv1（args 中已带 -g 1，全 I 帧）让其覆盖即可。
    """
    if not gop or gop <= 0:
        return []
    return ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]


def pick_encoder(src_codec: str, force_codec: str | None = None,
                 crf: int | None = None, preset: str | None = None) -> dict:
    """选择编码器配置。

    优先级:
        force_codec 指定 → 用它；
        否则按 src_codec 映射；
        都没匹配 → 退回 h264。

    返回 {"encoder", "ext", "pix_fmt", "args"}，args 可直接拼到 ffmpeg 命令里。
    """
    target = (force_codec or "").lower() or _SRC_TO_TARGET.get((src_codec or "").lower(), "h264")
    spec = _ENCODER_TABLE.get(target) or _ENCODER_TABLE["h264"]
    eff_crf = crf if crf is not None else spec["default_crf"]
    eff_preset = preset if preset is not None else spec["default_preset"]
    return {
        "encoder": spec["encoder"],
        "ext": spec["ext"],
        "pix_fmt": spec["pix_fmt"],
        "args": spec["args"](eff_crf, eff_preset),
    }


_EMPTY_META = {"fps": 0.0, "width": 0, "height": 0, "duration": 0.0, "codec": ""}


def get_meta(path: Path) -> dict:
    """{fps, width, height, duration, codec}；任一字段失败时为 0/""

    一次 ffprobe 取齐：调用方不需要再单独跑 get_src_codec()。
    按 (路径, mtime) 缓存：同一视频在「按真实分辨率路由」与 process_video 里会被
    探测两次，缓存让第二次免去 ffprobe。返回副本避免调用方改动污染缓存。
    """
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return dict(_get_meta_cached(str(path), mtime))


@functools.lru_cache(maxsize=8192)
def _get_meta_cached(path_str: str, _mtime: float) -> dict:
    path = path_str
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
    """整数秒 → 标签字符串

    s / m 关系，与 m / h 关系完全对称：

        < 60            → "{s}s"                  例: 30 → "30s"
        == 60           → "60s"                   边界（保持秒单位）
        60 < t < 3600
          整分钟        → "{m}m"                  例: 120 → "2m"
          含余数        → "{m}m{s}s"              例: 80  → "1m20s"
        == 3600         → "60m"                   边界（保持分钟单位）
        > 3600
          整小时        → "{h}h"                  例: 7200 → "2h"
          含余数        → "{h}h[{m}m][{s}s]"      为 0 的部分跳过
                                                 例: 3660 → "1h1m"
                                                     3601 → "1h1s"
                                                     7321 → "2h2m1s"
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
            # 优先只扫 .mp4；找到就直接用，不再扫其他格式
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
                # 回退：扫描其他视频格式
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

import threading as _threading

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


