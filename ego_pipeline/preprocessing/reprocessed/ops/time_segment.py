"""segment op：按固定时长切分视频

输出结构:
    <output>/<video_stem>/<video_stem>_<start>_<end>.<ext>

时间标签:
    < 60s         → "Xs"
    == 60s        → "60s"
    > 60 整分钟   → "Xm"
    > 60 含余数   → "XmYs"

示例:
    20 分钟视频按 60s 切 → vid_0s_60s.mp4, vid_60s_2m.mp4, vid_2m_3m.mp4 ...
    按 40s 切            → vid_0s_40s.mp4, vid_40s_1m20s.mp4, vid_1m20s_2m.mp4 ...

实现:
    用 ffmpeg 的 -f segment muxer 一次性切完，启动 1 个 ffmpeg 进程而非 N 个。
    reencode 模式加 -force_key_frames 让切点对齐请求边界；stream copy 模式由
    自然关键帧对齐，段实际时长可能略偏（与 PipelineOp 行为一致）。
    段时长用 nominal (total // duration) 推算，不再逐段 probe。
"""
from __future__ import annotations

import sys
from pathlib import Path

from core import (
    ProcessResult,
    format_time,
    get_duration,
    gop_args,
    is_interrupted,
    run_ffmpeg,
)

NAME = "segment"
DESCRIPTION = "按固定时长切分视频"
IS_TERMINAL = True  # pipeline 中由本 op 控制输出阶段（-f segment muxer）


def add_arguments(parser) -> None:
    parser.add_argument("--duration", type=int, default=60,
                        help="每段时长（整数秒），默认 60")
    parser.add_argument("--reencode", action="store_true",
                        help="重新编码（精确切分；不加则用 stream copy，按关键帧对齐）")
    parser.add_argument("--min_tail", type=float, default=1.0,
                        help="尾段最短保留时长（秒），低于则丢弃。默认 1.0")


def default_output_suffix(args) -> str:
    return f"_Tseg{args.duration}"


def output_label(args) -> str:
    return f"Tseg{args.duration}"


def describe_config(args) -> str:
    mode = "reencode" if args.reencode else "stream copy"
    return f"duration={args.duration}s, mode={mode}, min_tail={args.min_tail}s"


def plan(video: Path, args) -> str:
    total = get_duration(video)
    if total <= 0:
        return "(ffprobe 失败)"
    n = int(total // args.duration)
    if total - n * args.duration >= args.min_tail:
        n += 1
    return f"{total:.1f}s → {n} 段"


def process_video(video: Path, output_dir: Path, args) -> ProcessResult:
    if is_interrupted():
        return ProcessResult(video, 0, 0, 0, "已中断，跳过")
    total = get_duration(video)
    if total <= 0:
        return ProcessResult(video, 0, 0, 0, "ffprobe 读取时长失败")
    if total < args.min_tail:
        return ProcessResult(
            video, 0, 0, 0,
            f"视频太短 ({total:.2f}s) < min_tail ({args.min_tail}s)，跳过")

    stem = video.stem
    ext = video.suffix
    D = args.duration

    # 短视频不需要切分：直接放到 output_dir/<原名>，不建子文件夹也不改名
    if total <= D + 1e-3:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / video.name
        if out_path.exists() and not args.overwrite:
            return ProcessResult(video, 0, 1, 0, "已存在，跳过")
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(video)]
        if args.reencode:
            cmd += ["-c:v", "libx264",
                    *gop_args(getattr(args, "gop", 15)),
                    "-preset", "veryfast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        cmd += [str(out_path)]
        rc, err = run_ffmpeg(cmd)
        if rc != 0:
            out_path.unlink(missing_ok=True)
            print(f"[fail] {video.name}: {err}", file=sys.stderr)
            return ProcessResult(video, 0, 0, 1, f"ffmpeg 失败：{err}")
        return ProcessResult(video, 1, 0, 0, "短视频原样输出")

    out_folder = output_dir / stem
    out_folder.mkdir(parents=True, exist_ok=True)

    # 视频级 skip：扫描已 rename 完成的段（与 PipelineOp 一致的 glob 风格）
    existing = sorted(out_folder.glob(f"{stem}_*s_*s{ext}")) + \
               sorted(out_folder.glob(f"{stem}_*s_*m{ext}")) + \
               sorted(out_folder.glob(f"{stem}_*s_*h{ext}"))
    existing = list({p for p in existing})
    if existing and not args.overwrite:
        return ProcessResult(video, 0, len(existing), 0,
                             f"已存在 {len(existing)} 段，跳过")

    # 清扫上次未完成的中间文件：本版 _part%04d.<ext> 与旧版 *.part.<ext>
    stale = 0
    for p in list(out_folder.glob(f"{stem}_part*{ext}")):
        try:
            p.unlink()
            stale += 1
        except Exception:
            pass
    for p in list(out_folder.glob("*.part.*")):
        try:
            p.unlink()
            stale += 1
        except Exception:
            pass

    pattern = str(out_folder / f"{stem}_part%04d{ext}")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video)]
    if args.reencode:
        cmd += ["-c:v", "libx264",
                *gop_args(getattr(args, "gop", 15)),
                "-preset", "veryfast", "-crf", "18",
                "-force_key_frames", f"expr:gte(t,n_forced*{D})",
                "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    cmd += ["-f", "segment",
            "-segment_time", str(D),
            "-reset_timestamps", "1",
            "-segment_start_number", "0",
            pattern]

    rc, err = run_ffmpeg(cmd)
    if rc != 0:
        for p in list(out_folder.glob(f"{stem}_part*{ext}")):
            p.unlink(missing_ok=True)
        print(f"[fail] {video.name}: {err}", file=sys.stderr)
        return ProcessResult(video, 0, 0, 1, f"ffmpeg 失败：{err}")

    parts = sorted(out_folder.glob(f"{stem}_part*{ext}"))
    if not parts:
        return ProcessResult(video, 0, 0, 1, "segment 无输出")

    success = failed = 0
    last_idx = len(parts) - 1
    for i, p in enumerate(parts):
        start = i * D
        if i < last_idx:
            end = start + D
        else:
            nominal_tail = total - start
            if nominal_tail < args.min_tail:
                p.unlink(missing_ok=True)
                continue
            end = start + D if nominal_tail >= D - 0.5 \
                else start + int(round(nominal_tail))
        target = out_folder / f"{stem}_{format_time(start)}_{format_time(end)}{ext}"
        try:
            p.rename(target)
            success += 1
        except Exception as e:
            p.unlink(missing_ok=True)
            failed += 1
            print(f"[fail] rename {p.name}: {e}", file=sys.stderr)

    msg = f"{success}成功 / 0已存在 / {failed}失败"
    if stale:
        msg += f" (清扫 {stale} 残留)"
    return ProcessResult(video, success, 0, failed, msg)
