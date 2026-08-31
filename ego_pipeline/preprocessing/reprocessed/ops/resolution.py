"""resolution op：分辨率档位选择 + 缩放（保比例 / 不放大）

策略（--scale_mode fit，默认）:
    1. 找最接近源比例的 ratio bucket；偏差超过 --ratio_tol 则认为「无匹配比例」
    2. 在该 bucket 内取「面积 ≤ src 面积」的候选（不放大）
    3. 候选里挑「面积 ≤ --max_width × --max_height 像素预算」中最大的那一档
    4. 若所有候选都超预算，挑面积最小的（仍不放大）
    5. 无比例匹配 / bucket 无候选 → 回退到等比缩放，两边都不超预算，不放大，偶数取整

策略（--scale_mode pad）:
    1. 输出尺寸恒为 --max_width × --max_height
    2. 源内容按原比例等比缩放至能放进目标框（不放大）
    3. 剩余区域用黑边填充（letterbox/pillarbox）

策略（--scale_mode resize）:
    1. 输出尺寸恒为 --max_width × --max_height
    2. 直接拉伸到目标尺寸，不保比例（会变形）

编码:
    - 默认按源 codec 选 encoder（h264→libx264, hevc→libx265 ...）
    - 公共 --reencode <codec> 强制统一目标编码

输出:
    <output>/<video_stem>.<ext>
    - 流复制（无需缩放）：沿用源扩展名
    - 重编码：按目标 codec 选（h264/hevc/av1 → .mp4，vp9 → .webm，ffv1 → .mkv）
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

NAME = "resolution"
DESCRIPTION = "调分辨率（按档位选择，不放大）"
IS_TERMINAL = False


RATIO_BUCKETS: list[tuple[str, float, list[tuple[int, int]]]] = [
    # (label, ratio, [resolutions from large to small])
    ("16:9",  16 / 9, [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720),
                       (854, 480), (640, 360), (426, 240), (320, 180)]),
    ("4:3",   4 / 3,  [(2560, 1920), (1920, 1440), (1600, 1200), (1440, 1080),
                       (1280, 960), (1024, 768), (800, 600), (640, 480), (512, 384)]),
    ("16:10", 16 / 10, [(1920, 1200), (1680, 1050), (1440, 900), (1280, 800), (640, 400)]),
    ("5:4",   5 / 4,  [(1920, 1536), (1280, 1024), (960, 768)]),
    ("21:9",  21 / 9, [(2560, 1080), (1920, 800), (1280, 540)]),
    ("8:3",   8 / 3,  [(2880, 1080), (1920, 720), (1280, 480), (640, 240)]),
    ("2:1",   2 / 1,  [(5760, 2880), (2880, 1440), (1920, 960), (1280, 640)]),
    ("1:1",   1 / 1,  [(1080, 1080), (720, 720), (480, 480)]),
    ("9:16",  9 / 16, [(2160, 3840), (1080, 1920), (720, 1280), (480, 854), (360, 640)]),
    ("3:4",   3 / 4,  [(1200, 1600), (960, 1280), (768, 1024), (600, 800), (480, 640)]),
    ("9:21",  9 / 21, [(1080, 2560), (800, 1920)]),
    ("1:2",   1 / 2,  [(1440, 2880), (720, 1440), (640, 1280), (360, 720)]),
]


def add_arguments(parser) -> None:
    parser.add_argument("--max_width", type=int, default=512,
                        help="分辨率宽度预算（fit 模式参与档位选择；pad 模式即目标宽度）。默认 512")
    parser.add_argument("--max_height", type=int, default=384,
                        help="分辨率高度预算（fit 模式参与档位选择；pad 模式即目标高度）。默认 384")
    parser.add_argument("--ratio_tol", type=float, default=0.05,
                        help="纵横比容差，源比例与档位比例相差超过此值则跳过该档位。默认 0.05")
    parser.add_argument("--scale_mode", choices=["fit", "pad", "resize"], default="fit",
                        help="缩放模式：fit=取最接近源比例的同比例档位（不放大、不填边）；"
                             "pad=保持内容比例缩放后用黑边填到 max_width×max_height；"
                             "resize=直接拉伸到 max_width×max_height（不保比例，会变形）。默认 fit")


def default_output_suffix(args) -> str:
    return f"_{args.max_width}x{args.max_height}P{args.codec or 'auto'}"


def output_label(args) -> str:
    return f"{args.max_width}x{args.max_height}P"


def predicted_resolution(meta: dict, args) -> tuple[int, int] | None:
    """预报该视频的真实输出分辨率（= ffmpeg 实际缩放到的尺寸）。

    与 process_video/vf_chain 用同一套 pick_resolution，故路由用它分桶 100% 等于
    最终写出的分辨率。源 probe 失败返回 None（调用方退回预算命名）。
    """
    sw, sh = meta.get("width", 0), meta.get("height", 0)
    if sw <= 0 or sh <= 0:
        return None
    return pick_resolution(sw, sh, args.max_width, args.max_height,
                           args.ratio_tol, args.scale_mode)


def real_output_label(meta: dict, args) -> str:
    """真实分辨率目录标签 `<w>x<h>P`；预报不出时退回预算标签 output_label()。"""
    wh = predicted_resolution(meta, args)
    return f"{wh[0]}x{wh[1]}P" if wh else output_label(args)


def describe_config(args) -> str:
    policy = f"force→{args.codec}" if args.codec else "preserve-source-codec"
    size_role = "budget" if args.scale_mode == "fit" else "target"
    return (f"max_size={args.max_width}x{args.max_height} ({size_role}), "
            f"scale_mode={args.scale_mode}, "
            f"ratio_tol=±{args.ratio_tol:.0%}, encode={policy}")


def plan(video: Path, args) -> str:
    meta = get_meta(video)
    if meta["width"] == 0:
        return "(ffprobe 失败)"
    src_w, src_h = meta["width"], meta["height"]
    dst_w, dst_h = pick_resolution(src_w, src_h, args.max_width, args.max_height,
                                    args.ratio_tol, args.scale_mode)
    tag = f" [{args.scale_mode}]" if args.scale_mode != "fit" else ""
    return f"{src_w}x{src_h} {meta['duration']:.1f}s → {dst_w}x{dst_h}{tag}"


def _build_scale_filter(src_w: int, src_h: int, dst_w: int, dst_h: int,
                        scale_mode: str) -> str | None:
    """生成 -vf 片段；无需变换返回 None。"""
    if scale_mode == "pad":
        # 内容按原比例等比缩放至放进目标框（不放大），剩余区域填黑
        scale = (f"scale=w='min({dst_w},iw)':h='min({dst_h},ih)':"
                 f"force_original_aspect_ratio=decrease:flags=lanczos")
        pad = f"pad={dst_w}:{dst_h}:(ow-iw)/2:(oh-ih)/2:color=black"
        return f"{scale},{pad}"
    if scale_mode == "resize":
        if (src_w, src_h) == (dst_w, dst_h):
            return None
        return f"scale={dst_w}:{dst_h}:flags=lanczos"
    if (src_w, src_h) == (dst_w, dst_h):
        return None
    return f"scale={dst_w}:{dst_h}:flags=lanczos"


def vf_chain(meta: dict, args) -> str | None:
    """多 op 路径调用：返回此 op 贡献的 -vf 片段。无变化返回 None。"""
    src_w, src_h = meta["width"], meta["height"]
    if src_w <= 0 or src_h <= 0:
        return None
    dst_w, dst_h = pick_resolution(src_w, src_h, args.max_width, args.max_height,
                                    args.ratio_tol, args.scale_mode)
    return _build_scale_filter(src_w, src_h, dst_w, dst_h, args.scale_mode)


def process_video(video: Path, output_dir: Path, args) -> ProcessResult:
    meta = get_meta(video)
    total = meta["duration"] or get_duration(video)
    if total <= 0 or meta["width"] == 0:
        return ProcessResult(video, 0, 0, 0, "ffprobe 失败")

    src_w, src_h = meta["width"], meta["height"]
    dst_w, dst_h = pick_resolution(src_w, src_h, args.max_width, args.max_height,
                                    args.ratio_tol, args.scale_mode)

    scale_filter = _build_scale_filter(src_w, src_h, dst_w, dst_h, args.scale_mode)
    needs_scale = scale_filter is not None
    reencode = bool(args.codec) or needs_scale

    if reencode:
        enc = pick_encoder(meta["codec"], args.codec,
                           crf=getattr(args, "crf", None),
                           preset=getattr(args, "preset", None))
        out_ext = enc["ext"]
    else:
        out_ext = video.suffix.lower()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video.stem}{out_ext}"

    src_str = f"{src_w}x{src_h}"
    dst_str = f"{dst_w}x{dst_h}"

    if out_path.exists() and not args.overwrite:
        return ProcessResult(video, 0, 1, 0, f"已存在，跳过 ({src_str} → {dst_str})")

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video)]

    if reencode:
        vf_parts = [scale_filter] if scale_filter else []
        vf_parts.append("setsar=1")
        cmd += [
            "-vf", ",".join(vf_parts),
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


def pick_resolution(src_w: int, src_h: int,
                    max_w: int, max_h: int,
                    ratio_tol: float = 0.05,
                    scale_mode: str = "fit") -> tuple[int, int]:
    """挑选目标分辨率

    scale_mode == "pad" 或 "resize":
      直接返回 (max_w, max_h) —— 输出始终为目标尺寸；缩放细节由滤镜处理。

    scale_mode == "fit"（默认）:
      1. 找最接近源比例的 ratio bucket；若偏差超过 ratio_tol，认为「无匹配比例」
      2. 在该 bucket 内取「面积 ≤ src 面积」的候选（不放大）
      3. 候选里挑「面积 ≤ max_w*max_h」中最大的
      4. 若所有候选面积都超出预算，挑面积最小的（仍不放大、保证非空）
      5. 无比例匹配 / bucket 无候选 → 回退到等比缩放使两边都不超过预算；不放大；偶数取整

    返回 (w, h)，保证都是正偶数。
    """
    if scale_mode in ("pad", "resize"):
        return max_w, max_h
    if src_w <= 0 or src_h <= 0:
        return max_w, max_h
    src_ratio = src_w / src_h
    src_area = src_w * src_h
    budget = max_w * max_h

    _, best_ratio, bucket = min(RATIO_BUCKETS, key=lambda b: abs(b[1] / src_ratio - 1))
    if abs(best_ratio / src_ratio - 1) <= ratio_tol:
        candidates = [(w, h) for w, h in bucket if w * h <= src_area]
        if candidates:
            fitting = [(w, h) for w, h in candidates if w * h <= budget]
            return max(fitting, key=lambda x: x[0] * x[1]) if fitting \
                else min(candidates, key=lambda x: x[0] * x[1])

    # 回退：等比缩放 + 两边都不超预算 + 不放大
    scale = min(max_w / src_w, max_h / src_h, 1.0)
    new_w = max(2, int(src_w * scale) // 2 * 2)
    new_h = max(2, int(src_h * scale) // 2 * 2)
    return new_w, new_h
