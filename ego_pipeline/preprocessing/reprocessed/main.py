#!/usr/bin/env python3
"""批量视频处理主入口（模块化）

每个处理操作（切分 / 降采样 / 降分辨率 ...）实现为 ops/ 下的一个文件，
通过 `--ops <name>` 选择。本脚本只负责：参数解析、视频收集、并发调度、统一日志。

当前已实现的 op:
    fps         降帧率（reencode 触发，源 fps 已满足则不动）
    resolution  调分辨率（按档位选择，不放大）
    segment     按固定时长切分视频

组合 op:
    `--ops a+b+c` 按声明顺序串联：非终端 op (fps/resolution) 拼进 -vf 链，
    终端 op (segment) 自动落到末端，整体 1 decode + 1 encode + N 输出。
    每个 op 独立判断已满足并跳过；全部满足且未强制 --codec → 整体跳过。

用法:
    python main.py --list                                                # 列出可用 op
    python main.py --ops segment --help                                  # 查看 op 详细参数
    python main.py --ops segment --input /data/videos --duration 60
    python main.py --ops segment+fps+resolution --input /data \\
        --duration 60 --max_fps 30 --max_width 1280 --max_height 720 \\
        --codec h264 --crf 18 --preset ultrafast
    python main.py --ops segment --input /data --duration 60 --dry_run

多阶段（逗号分隔，前一阶段输出作为后一阶段输入）:
    python main.py --ops "segment+fps+resolution,undistort" --input /data \\
        --output /out --duration 60 --max_fps 30 --max_width 1280 --max_height 720 \\
        --gpus 0,1,2,3 --rotation 90 --f_out 600 --calib_root /data
"""
from __future__ import annotations

import argparse
import copy
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 把本目录加入 sys.path，让 ops.time_segment 等子模块能 `from core import ...`
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from core import (
    ProcessResult,
    VIDEO_EXTS,
    collect_videos,
    format_time,
    get_duration,
    get_meta,
    gop_args,
    is_interrupted,
    kill_all_subprocesses,
    pick_encoder,
    run_ffmpeg,
)
from ops import OPS


def _run_verify(src: Path, rel: Path, out_root: Path, args) -> dict | None:
    """处理完成后立即验收单个视频；通过返回 None，失败返回 issue dict。"""
    try:
        from verify import probe, expected_segments
    except ImportError:
        return {"status": "import_error", "src": src, "msg": "无法导入 verify.py"}

    ops_str = getattr(args, "ops", "")
    is_seg = "segment" in ops_str
    check_fps = "fps" not in ops_str        # fps op 会主动改帧率，不与源对比
    check_res = "resolution" not in ops_str  # resolution op 会主动改分辨率

    duration = getattr(args, "duration", 10 ** 9)
    min_tail = getattr(args, "min_tail", 1.0)
    fps_tol = getattr(args, "fps_tol", 0.01)

    src_meta = probe(src)
    if "error" in src_meta:
        return None  # 源 probe 失败，跳过

    total = src_meta["duration"]
    src_fps = src_meta.get("fps", 0.0)
    src_w, src_h = src_meta.get("width", 0), src_meta.get("height", 0)
    src_frames = src_meta.get("frames", 0)
    if total <= 0:
        return None

    out_subdir = out_root if str(rel) in ("", ".") else out_root / rel

    # ── 找输出文件 ──────────────────────────────────────────────────────────────
    if is_seg and total > duration + 1e-3:
        seg_folder = out_subdir / src.stem
        if not seg_folder.is_dir():
            return {"status": "miss", "src": src,
                    "msg": f"段文件夹不存在: {seg_folder}"}
        out_paths = sorted(p for p in seg_folder.iterdir()
                           if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    else:
        # 按 stem 匹配（允许重编码后扩展名变化）
        out_paths = [p for p in out_subdir.glob(f"{src.stem}.*")
                     if p.suffix.lower() in VIDEO_EXTS]

    if not out_paths:
        return {"status": "miss", "src": src, "msg": "输出文件不存在"}

    # ── probe 输出 ─────────────────────────────────────────────────────────────
    metas = []
    for p in out_paths:
        m = probe(p)
        if "error" in m:
            return {"status": "probe_out", "src": src,
                    "msg": f"输出 ffprobe 失败 ({p.name}): {m['error']}"}
        metas.append(m)

    # ── segment：段数 & 命名 ───────────────────────────────────────────────────
    if is_seg and total > duration + 1e-3:
        pairs = expected_segments(total, duration, min_tail)
        expected_stems = {
            f"{src.stem}_{format_time(s)}_{format_time(e)}" for s, e in pairs
        }
        found_stems = {p.stem for p in out_paths}
        missing = expected_stems - found_stems
        extra = found_stems - expected_stems
        if missing or extra:
            bits = []
            if missing:
                bits.append("缺 " + ", ".join(sorted(missing)[:3])
                             + (" ..." if len(missing) > 3 else ""))
            if extra:
                bits.append("多 " + ", ".join(sorted(extra)[:3])
                             + (" ..." if len(extra) > 3 else ""))
            return {"status": "seg_miss" if missing else "seg_extra", "src": src,
                    "msg": f"段数 期望{len(expected_stems)}/实际{len(found_stems)} "
                           f"({'; '.join(bits)})"}

    # ── segment：总时长 & 帧数 ─────────────────────────────────────────────────
    if is_seg:
        sum_dur = sum(m["duration"] for m in metas)
        sum_frames = sum(m["frames"] for m in metas)
        dur_tol = max(1.0, total * 0.01)
        frame_tol = max(5, int(src_frames * 0.01))
        if abs(sum_dur - total) > dur_tol:
            return {"status": "duration", "src": src,
                    "msg": f"总时长 期望{total:.2f}s 实际{sum_dur:.2f}s "
                           f"(Δ={sum_dur - total:+.2f}s, 容差±{dur_tol:.2f}s)"}
        if src_frames > 0 and abs(sum_frames - src_frames) > frame_tol:
            return {"status": "frames", "src": src,
                    "msg": f"总帧数 期望{src_frames} 实际{sum_frames} "
                           f"(Δ={sum_frames - src_frames:+d}, 容差±{frame_tol})"}

    # ── fps & 分辨率（各 op 只检查自己不会改变的属性）─────────────────────────
    for m, p in zip(metas, out_paths):
        if check_fps and src_fps > 0:
            if abs(m.get("fps", 0) - src_fps) > fps_tol:
                return {"status": "fps", "src": src,
                        "msg": f"帧率不一致 ({p.name}): "
                               f"期望{src_fps:.3f} 实际{m.get('fps', 0):.3f}"}
        if check_res and src_w > 0 and src_h > 0:
            w, h = m.get("width", 0), m.get("height", 0)
            if (w, h) != (src_w, src_h):
                return {"status": "resolution", "src": src,
                        "msg": f"分辨率不一致 ({p.name}): "
                               f"期望{src_w}x{src_h} 实际{w}x{h}"}

    return None  # ok


def _install_sigint_handler() -> None:
    """首次 Ctrl+C：立刻 SIGKILL 所有 ffmpeg，让 worker 立刻返回；
    第二次 Ctrl+C：恢复默认，强制中断 Python。"""
    def _handler(signum, frame):
        n = kill_all_subprocesses()
        print(f"\n[interrupted] 收到 Ctrl+C，已 SIGKILL {n} 个 ffmpeg "
              f"（再按一次强制退出）", file=sys.stderr, flush=True)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGINT, _handler)


# ── 组合 op (pipeline) ────────────────────────────────────────────────────────

class PipelineOp:
    """`a+b+c` 形式的组合 op。鸭子接口与单 op 兼容。"""

    def __init__(self, op_names: list[str]):
        self.op_names = op_names
        self.ops = [OPS[n] for n in op_names]
        self.NAME = "+".join(op_names)
        self.DESCRIPTION = "组合: " + " → ".join(o.NAME for o in self.ops)

        # 终端 op 不限位置：ffmpeg 管线天然 filter→muxer，无论 --ops 里写在哪
        terminals = [o for o in self.ops if getattr(o, "IS_TERMINAL", False)]
        if len(terminals) > 1:
            raise SystemExit(
                f"error: 组合 op 中含多个终端 op: "
                f"{[o.NAME for o in terminals]}（最多一个）")
        self.terminal_op = terminals[0] if terminals else None
        self.vf_ops = [o for o in self.ops if not getattr(o, "IS_TERMINAL", False)]

    def add_arguments(self, parser) -> None:
        for op_mod in self.ops:
            op_mod.add_arguments(parser)

    def describe_config(self, args) -> str:
        parts = []
        for op_mod in self.ops:
            if hasattr(op_mod, "describe_config"):
                parts.append(f"{op_mod.NAME}{{{op_mod.describe_config(args)}}}")
        return "; ".join(parts)

    def output_label(self, args) -> str:
        labels = []
        for op_mod in self.ops:
            if hasattr(op_mod, "output_label"):
                labels.append(op_mod.output_label(args))
            else:
                labels.append(op_mod.NAME)
        return "_".join(labels)

    def default_output_suffix(self, args) -> str:
        return "_" + self.output_label(args)

    def plan(self, video: Path, args) -> str:
        meta = get_meta(video)
        if meta["width"] == 0:
            return "(ffprobe 失败)"
        steps = []
        for op_mod in self.ops:
            if self._is_segment(op_mod):
                if meta["duration"] > args.duration + 1e-3:
                    n = self._count_segments(meta["duration"], args)
                    steps.append(f"segment×{n}")
                else:
                    steps.append(f"segment:skip({meta['duration']:.1f}s≤{args.duration}s)")
            elif hasattr(op_mod, "vf_chain"):
                snippet = op_mod.vf_chain(meta, args)
                steps.append(f"{op_mod.NAME}:{snippet or 'skip'}")
            else:
                steps.append(op_mod.NAME)
        return f"{meta['duration']:.1f}s → " + " | ".join(steps)

    def process_video(self, video: Path, output_dir: Path, args) -> ProcessResult:
        if is_interrupted():
            return ProcessResult(video, 0, 0, 0, "已中断，跳过")

        meta = get_meta(video)
        total = meta["duration"]
        if total <= 0 or meta["width"] == 0:
            return ProcessResult(video, 0, 0, 0, "ffprobe 失败")

        vf_parts: list[str] = []
        for op_mod in self.vf_ops:
            if not hasattr(op_mod, "vf_chain"):
                continue
            snippet = op_mod.vf_chain(meta, args)
            if snippet:
                vf_parts.append(snippet)
        needs_vf = bool(vf_parts)

        is_segment = self._is_segment(self.terminal_op)
        need_segment = is_segment and total > args.duration + 1e-3
        if is_segment and total < getattr(args, "min_tail", 1.0):
            return ProcessResult(
                video, 0, 0, 0,
                f"视频太短 ({total:.2f}s) < min_tail ({args.min_tail}s)，跳过")

        # --codec 只在 fps/分辨率实际需要重编码时生效，不对已达标的视频强制 reencode
        force_codec = bool(getattr(args, "codec", None)) and needs_vf
        if not needs_vf and not need_segment:
            # 即使无操作,也要在输出位置写出文件,下游(yolo/label)才能扫到。
            # 用 symlink 节省 IO,失败回退 hardlink/复制。
            # 命名沿用 segment 风格 "<stem>_0s_<dur>s.mp4" 以便 label_attach 解析。
            dur_int = int(round(total))
            out_dir_actual = output_dir / video.stem
            out_dir_actual.mkdir(parents=True, exist_ok=True)
            out_path = out_dir_actual / f"{video.stem}_0s_{dur_int}s.mp4"
            if not out_path.exists():
                src = video.resolve()
                try:
                    out_path.symlink_to(src)
                except OSError:
                    try:
                        os.link(str(src), str(out_path))
                    except OSError:
                        import shutil as _sh
                        _sh.copy2(str(src), str(out_path))
            return ProcessResult(video, 1, 0, 0, "已满足，symlink 到输出")

        reencode = needs_vf or force_codec

        if reencode:
            enc = pick_encoder(
                meta["codec"],
                getattr(args, "codec", None),
                crf=getattr(args, "crf", None),
                preset=getattr(args, "preset", None),
            )
            out_ext = enc["ext"]
        else:
            enc = None
            out_ext = video.suffix.lower()

        if need_segment:
            out_folder = output_dir / video.stem
            out_folder.mkdir(parents=True, exist_ok=True)
            for p in list(out_folder.glob(f"{video.stem}_part*{out_ext}")):
                p.unlink(missing_ok=True)
            for p in list(out_folder.glob("*.part.*")):
                p.unlink(missing_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_folder = output_dir

        if need_segment:
            existing = sorted(out_folder.glob(f"{video.stem}_*s_*s{out_ext}")) + \
                       sorted(out_folder.glob(f"{video.stem}_*s_*m{out_ext}")) + \
                       sorted(out_folder.glob(f"{video.stem}_*s_*h{out_ext}"))
            existing = list({p for p in existing})  # dedup
            if existing and not args.overwrite:
                return ProcessResult(video, 0, len(existing), 0,
                                     f"已存在 {len(existing)} 段，跳过")
        else:
            out_path = out_folder / f"{video.stem}{out_ext}"
            if out_path.exists() and not args.overwrite:
                return ProcessResult(video, 0, 1, 0, "已存在，跳过")

        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(video)]

        if reencode:
            vf_chain = ",".join(vf_parts + ["setsar=1"]) if vf_parts else "setsar=1"
            cmd += ["-vf", vf_chain, "-c:v", enc["encoder"],
                    *gop_args(getattr(args, "gop", 15)), *enc["args"]]
            if enc["pix_fmt"]:
                cmd += ["-pix_fmt", enc["pix_fmt"]]
            if not getattr(args, "keep_audio", False):
                cmd += ["-an"]
            else:
                cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
            # reencode + segment：强制关键帧对齐分段起点，否则会回溯到前一关键帧
            if need_segment:
                cmd += ["-force_key_frames",
                        f"expr:gte(t,n_forced*{args.duration})"]
        else:
            cmd += ["-c", "copy"]
            if not getattr(args, "keep_audio", False):
                cmd += ["-an"]

        if need_segment:
            cmd += [
                "-f", "segment",
                "-segment_time", str(args.duration),
                "-reset_timestamps", "1",
                "-segment_start_number", "0",
            ]
            # segment muxer 只认 printf 模式；跑完再 rename 成 segment 标签
            pattern = str(out_folder / f"{video.stem}_part%04d{out_ext}")
            cmd += [pattern]
        else:
            if out_ext == ".mp4":
                cmd += ["-movflags", "+faststart"]
            cmd += [str(out_path)]

        rc, err = run_ffmpeg(cmd)
        if rc != 0:
            if need_segment:
                for p in list(out_folder.glob(f"{video.stem}_part*{out_ext}")):
                    p.unlink(missing_ok=True)
            else:
                out_path.unlink(missing_ok=True)
            print(f"[fail] {video.name}: {err}", file=sys.stderr)
            return ProcessResult(video, 0, 0, 1, f"ffmpeg 失败：{err}")

        if need_segment:
            return self._rename_segments(video, out_folder, out_ext, total, args)
        return ProcessResult(video, 1, 0, 0, "ok (1 decode + 1 encode)")

    @staticmethod
    def _is_segment(op_mod) -> bool:
        return op_mod is not None and getattr(op_mod, "NAME", "") == "segment"

    @staticmethod
    def _count_segments(total: float, args) -> int:
        D = args.duration
        n = int(total // D)
        rem = total - n * D
        if rem >= getattr(args, "min_tail", 1.0):
            n += 1
        return n

    def _rename_segments(self, video: Path, out_folder: Path, out_ext: str,
                         total: float, args) -> ProcessResult:
        """%04d 索引 → stem_<start>_<end> 标签；尾段长度 = total % duration，只 probe 它"""
        D = args.duration
        min_tail = getattr(args, "min_tail", 1.0)
        parts = sorted(out_folder.glob(f"{video.stem}_part*{out_ext}"))
        if not parts:
            return ProcessResult(video, 0, 0, 1, "segment 无输出")

        success = failed = 0
        last_idx = len(parts) - 1
        for i, p in enumerate(parts):
            start = i * D
            if i < last_idx:
                end = start + D
            else:
                seg_dur = get_duration(p)
                if seg_dur <= 0:
                    p.unlink(missing_ok=True)
                    failed += 1
                    continue
                if seg_dur < min_tail:
                    p.unlink(missing_ok=True)
                    continue
                end = start + D if seg_dur >= D - 0.5 else start + int(round(seg_dur))
            target = out_folder / f"{video.stem}_{format_time(start)}_{format_time(end)}{out_ext}"
            try:
                p.rename(target)
                success += 1
            except Exception as e:
                p.unlink(missing_ok=True)
                failed += 1
                print(f"[fail] rename {p.name}: {e}", file=sys.stderr)
        msg = f"{success} 段 (1 decode + 1 encode)"
        if failed:
            msg += f"  失败 {failed}"
        return ProcessResult(video, success, 0, failed, msg)


def resolve_op(name: str):
    """把 `--ops` 字符串解析成单 op 或 PipelineOp。未知子名抛 SystemExit。

    本函数只处理 **单阶段** 字符串（如 'segment' 或 'segment+fps+resolution'）。
    多阶段（带 ','）由 split_stages() 先切开再逐阶段调本函数。
    """
    if "+" in name:
        sub_names = [s for s in name.split("+") if s]
        if not sub_names:
            raise SystemExit(f"error: --ops 解析为空: {name!r}")
        for n in sub_names:
            if n not in OPS:
                raise SystemExit(
                    f"error: 未知子 op '{n}'（可选: {', '.join(OPS.keys())}）")
        return PipelineOp(sub_names)
    if name not in OPS:
        raise SystemExit(
            f"error: 未知 op '{name}'（可选: {', '.join(OPS.keys())}）")
    return OPS[name]


def split_stages(ops_str: str) -> list[str]:
    """把 `--ops` 整串按 ',' 切成多个阶段。每阶段仍是单 op 或 'a+b+c' 形式。

    'segment+fps+resolution,undistort' → ['segment+fps+resolution', 'undistort']
    'segment'                          → ['segment']
    """
    return [s.strip() for s in ops_str.split(",") if s.strip()]


# ── 参数解析 ──────────────────────────────────────────────────────────────────

def _add_common_encoder_args(p: argparse.ArgumentParser) -> None:
    """编码相关的公共参数（fps/resolution/pipeline 共享）"""
    p.add_argument("--codec", default=None,
                   help="强制目标编码（h264/hevc/av1/vp9/ffv1）。"
                        "不指定则按源 codec 推断 encoder")
    p.add_argument("--crf", type=int, default=None,
                   help="CRF 质量参数（数字越小质量越高）。默认按 encoder 取值")
    p.add_argument("--preset", default=None,
                   help="编码 preset（如 ultrafast/veryfast/medium/slow）。"
                        "默认按 encoder 取值")
    p.add_argument("--gop", type=int, default=15,
                   help="关键帧间隔（GOP，单位：帧）。仅在重编码时生效。"
                        "默认 15（30fps 源 ≈ 0.5s 一关键帧，方便后续 stream-copy 切片对齐）。"
                        "传 0 / 负数禁用此选项，沿用编码器默认。")
    p.add_argument("--keep_audio", action="store_true",
                   help="保留音频流（默认丢弃，-an）")


def build_parser(ops: list | None = None) -> argparse.ArgumentParser:
    """构建完整 CLI 解析器。

    ops 已知时把每个阶段 op 的参数都挂到主解析器（多阶段共享同一组参数命名空间，
    不同 op 间名字不冲突）；ops 为 None（--help / --list 路径）时只挂公共参数。
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 不用 choices=：组合/多阶段 op 不属于 OPS key，resolve_op / split_stages 校验
    p.add_argument("--ops", metavar="NAME",
                   help=f"要执行的 op（可选: {', '.join(OPS.keys())}；"
                        f"用 '+' 串联单阶段 vf-chain，如 fps+resolution+segment；"
                        f"用 ',' 串联多阶段，如 segment+fps+resolution,undistort）")
    p.add_argument("--list", action="store_true",
                   help="列出所有可用 op 后退出")
    p.add_argument("--input", nargs="+",
                   help="输入路径（目录递归或视频文件，可多个）")
    p.add_argument("--output", default=None,
                   help="输出根目录（不指定则按 op 默认推断）")
    p.add_argument("--output_dir", default=None,
                   help="字面量输出目录；给了就直接用，不再走 <args.output>/<base.name>_<label> 拼装")
    p.add_argument("--workers", type=int, default=32,
                   help="并发处理的视频数（默认 32）。"
                        "stream copy 模式以 I/O 为主，可开 16-32+；"
                        "--reencode/组合 op 每个 ffmpeg 内部已用多核，"
                        "建议 workers × 单视频线程数 ≲ CPU 核数")
    p.add_argument("--overwrite", action="store_true",
                   help="覆盖已存在的输出文件")
    p.add_argument("--dry_run", action="store_true",
                   help="只打印将要执行的处理计划，不实际运行")
    p.add_argument("--verify", action="store_true",
                   help="每个视频处理完后立即验收输出；出错跳过继续，最后汇总报告")
    p.add_argument("--name_pattern", default=None,
                   help="按文件名 fnmatch 过滤（如 'aria01_214-1.mp4'）。"
                        "多阶段时只对首阶段输入应用，后续阶段扫描首阶段输出不再过滤")
    _add_common_encoder_args(p)
    if ops:
        for op in ops:
            op.add_arguments(p)
    return p


def _routing_ops(op) -> list:
    """返回用于「按真实值分桶」的 op 列表（每个 op 的 real_output_label 决定其目录 token）。

    每个视频会把目录名里各 op 的预算 token 换成该视频的真实 token，于是不同真实
    帧率 / 分辨率的视频落到不同兄弟目录（如 fps30 与 fps29.97、512x384P 与 1024x768P）。

    - fps op：始终参与——真实帧率 min(src,max) 可廉价预报，且 undistort 不改帧率。
    - resolution op：仅当链里无 undistort 时参与——含 undistort 时输出尺寸由 fisheye
      标定决定、无法廉价预报，退回预算命名。
    单 op 与 PipelineOp 都适配；无可路由 op 时返回 []（保持原单一预算目录行为）。
    """
    subs = list(getattr(op, "ops", None) or [op])
    has_undistort = any(getattr(s, "NAME", "") == "undistort" for s in subs)
    routing = []
    for s in subs:
        if not hasattr(s, "real_output_label"):
            continue
        if getattr(s, "NAME", "") == "resolution" and has_undistort:
            continue
        routing.append(s)
    return routing


def resolve_output_dir(args, op) -> Path:
    if getattr(args, "output_dir", None):
        return Path(args.output_dir).expanduser().resolve()
    first = Path(args.input[0]).expanduser().resolve()
    base = first if first.is_dir() else first.parent
    if args.output:
        label = op.output_label(args) if hasattr(op, "output_label") \
            else op.default_output_suffix(args).lstrip("_")
        return Path(args.output).expanduser().resolve() / f"{base.name}_{label}"
    suffix = op.default_output_suffix(args)
    return base.parent / f"{base.name}{suffix}"


# ── 单阶段执行 ────────────────────────────────────────────────────────────────

def _stage_supports_verify(ops_str: str) -> bool:
    """undistort 会改分辨率/朝向，现有 _run_verify 默认对比源/出 res，会 false-positive，
    所以含 undistort 的阶段直接跳过 verify。"""
    return "undistort" not in ops_str


def _run_stage(op, args, stage_label: str = "") -> tuple[Path, int, int, list[dict]]:
    """执行单个阶段（args.ops/args.input 已经按阶段就位）。

    返回 (output_dir, tot_ok, tot_fl, verify_issues)。tot_fl 兼带 verify 失败计数（由调用方决定 exit）。
    """
    op_str = args.ops
    videos = collect_videos(args.input)

    pattern = getattr(args, "name_pattern", None)
    if pattern:
        import fnmatch
        n0 = len(videos)
        videos = [(v, r) for v, r in videos if fnmatch.fnmatch(v.name, pattern)]
        print(f"[name_pattern={pattern}] 过滤 {n0} → {len(videos)} 个视频", file=sys.stderr)

    if not videos:
        print(f"[{stage_label or op_str}] 没有找到视频文件", file=sys.stderr)
        return Path(args.output) if args.output else Path("."), 0, 1, []

    output_dir = resolve_output_dir(args, op)

    # ── 按真实值分桶：把输出根名里各路由 op 的预算 token（如 fps30 / 1280x720P）换成
    #    该视频的真实 token（如 fps29.97 / 1024x768P），于是不同真实帧率 / 分辨率的视频
    #    落到不同兄弟目录。含 undistort 的链其分辨率无法廉价预报 → 该 op 不参与路由。
    #    每个 (op, budget_tok) 独立替换；fps 与分辨率 token 互不为子串，顺序替换安全。
    route_pairs = [(o, o.output_label(args)) for o in _routing_ops(op)]
    route_pairs = [(o, tok) for o, tok in route_pairs if tok and tok in output_dir.name]
    route = bool(route_pairs)

    def _real_root(video: Path) -> Path:
        if not route:
            return output_dir
        meta = get_meta(video)
        name = output_dir.name
        for o, budget_tok in route_pairs:
            real_tok = o.real_output_label(meta, args)
            if real_tok != budget_tok:
                name = name.replace(budget_tok, real_tok)
        if name == output_dir.name:
            return output_dir
        return output_dir.parent / name

    if not route:
        output_dir.mkdir(parents=True, exist_ok=True)  # 路由时各真实根由 worker 按需建

    # 输出端镜像输入子目录，按 (rel_subdir, stem) 检测真冲突
    keys: dict[tuple[str, str], Path] = {}
    for v, rel in videos:
        k = (str(rel), v.stem)
        if k in keys:
            print(f"[warn] 同位置同 stem 冲突：\n"
                  f"       {keys[k]}\n       {v}", file=sys.stderr)
        else:
            keys[k] = v

    header = f"[stage {stage_label}] " if stage_label else ""
    print(f"\n{header}op    : {op_str}  ({op.DESCRIPTION})")
    print(f"{header}输入  : {len(videos)} 个视频")
    if route:
        print(f"{header}输出  : {output_dir.parent}/  (按真实帧率/分辨率分桶，镜像源目录结构)")
    else:
        print(f"{header}输出  : {output_dir}  (镜像源目录结构)")
    if hasattr(op, "describe_config"):
        print(f"{header}配置  : {op.describe_config(args)}")
    print(f"{header}并发  : {args.workers}")

    def _op_base(video: Path, rel: Path) -> Path:
        root = _real_root(video)
        return root if str(rel) in ("", ".") else root / rel

    if args.dry_run:
        print(f"\n{header}[dry_run]")
        for v, rel in videos:
            plan = op.plan(v, args) if hasattr(op, "plan") else "(plan 未实现)"
            print(f"  {v}  →  {_op_base(v, rel)}/  ({plan})")
        return output_dir, 0, 0, []

    verify_enabled = args.verify and _stage_supports_verify(op_str)
    if args.verify and not verify_enabled:
        print(f"{header}[verify] 本阶段含 undistort，跳过验收（res 会被改写）", file=sys.stderr)

    results: list[tuple[Path, int, int, int]] = []
    verify_issues: list[dict] = []

    if hasattr(op, "process_videos"):
        # op 自带批处理调度（如 undistort 的多 GPU 线程池）；让 op 全权负责并发与信号处理
        items = [(v, _op_base(v, rel)) for v, rel in videos]
        rel_by_video = {v.resolve(): rel for v, rel in videos}
        op_results = op.process_videos(items, args)
        for r in op_results:
            results.append((r.video, r.success, r.skipped, r.failed))
            if verify_enabled and r.success > 0:
                rel = rel_by_video.get(Path(r.video).resolve(), Path("."))
                issue = _run_verify(r.video, rel, _real_root(Path(r.video)), args)
                if issue:
                    verify_issues.append(issue)
    else:
        _install_sigint_handler()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(op.process_video, v, _op_base(v, rel), args): (v, rel)
                for v, rel in videos
            }
            it = as_completed(futures)
            if tqdm:
                it = tqdm(it, total=len(futures), desc=op_str)
            for fut in it:
                video, rel = futures[fut]
                try:
                    r = fut.result()
                    results.append((r.video, r.success, r.skipped, r.failed))
                    if verify_enabled and r.success > 0:
                        issue = _run_verify(video, rel, _real_root(video), args)
                        if issue:
                            verify_issues.append(issue)
                except Exception:
                    results.append((video, 0, 0, 1))

    tot_ok = sum(r[1] for r in results)
    tot_sk = sum(r[2] for r in results)
    tot_fl = sum(r[3] for r in results)
    status = "中断" if is_interrupted() else "完成"
    print(f"\n{header}{status}: 视频 {len(videos)} | 输出 成功 {tot_ok} / 已存在 {tot_sk} / 失败 {tot_fl}")
    if route:
        pat = output_dir.name
        for _o, budget_tok in route_pairs:
            pat = pat.replace(budget_tok, "*")
        produced = sorted(p.name for p in output_dir.parent.glob(pat) if p.is_dir())
        print(f"{header}输出目录: {output_dir.parent}/  (真实帧率/分辨率分桶: {', '.join(produced) or '无'})")
    else:
        print(f"{header}输出目录: {output_dir}")

    if verify_enabled:
        if verify_issues:
            print(f"\n{header}验收失败: {len(verify_issues)}")
            for issue in verify_issues:
                print(f"  [{issue['status']:<10}] {issue['src']}  {issue.get('msg', '')}")
        else:
            print(f"{header}验收通过: {tot_ok} 个")

    return output_dir, tot_ok, tot_fl, verify_issues


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    # 第一遍解析：先取出 --ops / --list（不触发 help/错误），用来决定后续要挂哪些参数
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--ops", default=None)
    bootstrap.add_argument("--list", action="store_true", default=False)
    args0, _ = bootstrap.parse_known_args()

    if args0.list:
        print("可用 op:")
        for name, op_mod in OPS.items():
            print(f"  {name:<12} {op_mod.DESCRIPTION}")
        print("\n组合用法: --ops a+b+c  （如 fps+resolution+segment）")
        print("多阶段:  --ops a+b,c    （如 segment+fps+resolution,undistort，前一阶段输出 → 后一阶段输入）")
        return

    # 把多阶段全部 op 的参数都挂上，让 argparse 一次性解析
    stage_strs = split_stages(args0.ops) if args0.ops else []
    stage_ops_for_help = [resolve_op(s) for s in stage_strs] if stage_strs else None
    parser = build_parser(stage_ops_for_help)
    args = parser.parse_args()  # 此时会处理 --help、未知参数等

    if not args.ops:
        parser.print_help()
        print("\nerror: 必须指定 --ops <name>，例如 --ops segment 或 "
              "--ops fps+resolution+segment 或 --ops a+b,c（用 --list 查看）", file=sys.stderr)
        sys.exit(2)
    if not args.input:
        parser.print_help()
        print("\nerror: 必须指定 --input <path>", file=sys.stderr)
        sys.exit(2)

    stages = split_stages(args.ops)
    if not stages:
        print(f"\nerror: --ops 解析为空: {args.ops!r}", file=sys.stderr)
        sys.exit(2)

    # 逐阶段执行：每阶段的 input = 上一阶段的 output_dir
    current_input = args.input
    total_fl = 0
    total_verify_issues: list[dict] = []

    for i, op_str in enumerate(stages):
        op = resolve_op(op_str)
        args_stage = copy.copy(args)
        args_stage.ops = op_str
        args_stage.input = current_input
        if i > 0:
            # 首阶段输出文件名已经被切段/重编码改写，--name_pattern 只应用于首阶段输入
            args_stage.name_pattern = None
        label = f"{i + 1}/{len(stages)}" if len(stages) > 1 else ""

        output_dir, _ok, fl, issues = _run_stage(op, args_stage, stage_label=label)
        total_fl += fl
        total_verify_issues.extend(issues)

        if is_interrupted():
            print(f"\n[stage {label}] 已中断，跳过后续阶段", file=sys.stderr)
            sys.exit(130)

        # dry_run 下后续阶段输入目录还不存在，没法继续 plan；提示一下停在这里
        if args.dry_run and i + 1 < len(stages):
            remaining = ", ".join(stages[i + 1:])
            print(f"\n[stage {label}] --dry_run 下只展示首阶段计划；"
                  f"后续阶段 ({remaining}) 需先实跑当前阶段才能扫描其输出",
                  file=sys.stderr)
            break

        # 上一阶段挂了就不要往下接（否则后一阶段会去扫不完整的输出）
        if fl > 0 and i + 1 < len(stages):
            print(f"\n[stage {label}] 有 {fl} 个失败，跳过后续阶段", file=sys.stderr)
            break

        current_input = [str(output_dir)]

    if total_fl or total_verify_issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
