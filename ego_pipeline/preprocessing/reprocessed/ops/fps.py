"""fps op：帧率降采样

策略:
    源 fps ≤ --max_fps 不动（流复制）
    源 fps > --max_fps   降到 --max_fps（重编码）

编码:
    - 默认按源 codec 选 encoder
    - 公共 --reencode <codec> 强制统一目标编码

输出:
    <output>/<video_stem>.<ext>
"""
from __future__ import annotations

import sys
from pathlib import Path

from core import (
    ProcessResult,
    get_duration,
    get_meta,
    gop_args,
    pick_encoder,
    run_ffmpeg,
)

NAME = "fps"
DESCRIPTION = "降帧率"
IS_TERMINAL = False


def add_arguments(parser) -> None:
    parser.add_argument("--max_fps", type=float, default=30.0,
                        help="目标帧率上限。源 fps ≤ 此值时保留，> 时降到此值。默认 30")


def default_output_suffix(args) -> str:
    return f"_fps{_fps_tag(args.max_fps)}{args.codec or 'auto'}"


def output_label(args) -> str:
    return f"fps{_fps_tag(args.max_fps)}"


def predicted_fps(meta: dict, args) -> float | None:
    """预报该视频的真实输出帧率（= ffmpeg 实际写出的 fps）。

    与 process_video/vf_chain 同一套策略 dst_fps = min(src_fps, max_fps)，
    故路由用它分桶 100% 等于最终写出的帧率。源 probe 失败返回 None（调用方退回预算命名）。
    """
    src_fps = meta.get("fps", 0.0)
    if src_fps <= 0:
        return None
    return min(src_fps, args.max_fps)


def real_output_label(meta: dict, args) -> str:
    """真实帧率目录标签 `fps<tag>`；预报不出时退回预算标签 output_label()。"""
    fps = predicted_fps(meta, args)
    return f"fps{_fps_tag(fps)}" if fps is not None else output_label(args)


def describe_config(args) -> str:
    policy = f"force→{args.codec}" if args.codec else "preserve-source-codec"
    return f"max_fps={args.max_fps}, encode={policy}"


def plan(video: Path, args) -> str:
    meta = get_meta(video)
    if meta["width"] == 0:
        return "(ffprobe 失败)"
    src_fps = meta["fps"]
    dst_fps = min(src_fps, args.max_fps) if src_fps > 0 else args.max_fps
    return f"{src_fps:.2f}fps {meta['duration']:.1f}s → {dst_fps:.2f}fps"


def vf_chain(meta: dict, args) -> str | None:
    """多 op 路径调用：返回此 op 贡献的 -vf 片段。无变化返回 None。"""
    src_fps = meta["fps"]
    if src_fps <= 0:
        return None
    if src_fps <= args.max_fps + 1e-3:
        return None
    return f"fps={args.max_fps}"


def process_video(video: Path, output_dir: Path, args) -> ProcessResult:
    meta = get_meta(video)
    total = meta["duration"] or get_duration(video)
    if total <= 0 or meta["width"] == 0:
        return ProcessResult(video, 0, 0, 0, "ffprobe 失败")

    src_fps = meta["fps"]
    dst_fps = min(src_fps, args.max_fps) if src_fps > 0 else args.max_fps

    needs_fps = src_fps > 0 and src_fps > args.max_fps + 1e-3
    reencode = bool(args.codec) or needs_fps

    if reencode:
        enc = pick_encoder(meta["codec"], args.codec,
                           crf=getattr(args, "crf", None),
                           preset=getattr(args, "preset", None))
        out_ext = enc["ext"]
    else:
        out_ext = video.suffix.lower()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video.stem}{out_ext}"

    src_str = f"{src_fps:.1f}fps"
    dst_str = f"{dst_fps:.1f}fps"

    if out_path.exists() and not args.overwrite:
        return ProcessResult(video, 0, 1, 0, f"已存在，跳过 ({src_str} → {dst_str})")

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video)]

    if reencode:
        vf_parts = [f"fps={dst_fps}", "setsar=1"]
        cmd += [
            "-vf", ",".join(vf_parts),
            "-r", f"{dst_fps}",
            "-c:v", enc["encoder"],
            *gop_args(getattr(args, "gop", 15)),
            *enc["args"],
        ]
        if enc["pix_fmt"]:
            cmd += ["-pix_fmt", enc["pix_fmt"]]
        if not args.keep_audio:
            cmd += ["-an"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        mode = f"reencode→{enc['encoder']}"
    else:
        cmd += ["-c", "copy"]
        if not args.keep_audio:
            cmd += ["-an"]
        mode = "copy"

    if out_ext == ".mp4":
        cmd += ["-movflags", "+faststart"]
    cmd += [str(out_path)]

    rc, err = run_ffmpeg(cmd)
    if rc != 0:
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        print(f"[fail] {out_path.name}: {err}", file=sys.stderr)
        return ProcessResult(video, 0, 0, 1, f"ffmpeg 失败：{err}")

    return ProcessResult(video, 1, 0, 0, f"ok [{mode}] ({src_str} → {dst_str})")


def _fps_tag(fps: float) -> str:
    """fps → 文件/目录名片段：四舍五入到 3 位小数后整数省略小数、去尾零。
    源 r_frame_rate 常是分数除法（30000/1001=29.97003…），不 round 会得到超长小数目录名。
    30.0 → '30'；29.97003 → '29.97'；23.976023 → '23.976'"""
    fps = round(float(fps), 3)
    if fps.is_integer():
        return str(int(fps))
    return f"{fps:.3f}".rstrip("0").rstrip(".")
