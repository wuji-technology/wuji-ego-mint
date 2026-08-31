#!/usr/bin/env python3
"""校验视频处理输出是否与源对齐（针对 segment / *+segment 组合 op）

期望输出结构（与 main.py / ops/time_segment.py 一致）:
    长视频  → <output_root>/<rel>/<stem>/<stem>_<start>_<end>.<ext>
    短视频  → <output_root>/<rel>/<原文件名>

校验项:
    1. 每个源视频是否有对应输出（长视频是子文件夹，短视频是同名文件）
    2. 长视频段数 = floor(T/D) + (1 if T%D >= min_tail else 0)
    3. 段命名覆盖 [0, T) 的整数秒区间
    4. 所有段总时长 / 总帧数 与源接近（默认容差 1.0s / 1.0%）
    5. 多余文件检测：output 树里出现源中不存在的视频

用法:
    python verify.py --input /data/src --output /data/src_Tseg60 \\
        --duration 60 --min_tail 1.0

退出码: 0 全通过；1 存在不一致
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core import VIDEO_EXTS, collect_videos, format_time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ── ffprobe ───────────────────────────────────────────────────────────────────

def probe(path: Path) -> dict:
    """返回 {duration, frames, fps, width, height}；失败返回 {error}"""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames,width,height:format=duration",
        "-of", "json", str(path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return {"path": path, "error": "ffprobe 未安装"}
    except subprocess.TimeoutExpired:
        return {"path": path, "error": "ffprobe 超时"}
    if r.returncode != 0:
        return {"path": path, "error": r.stderr.strip()[-120:]}
    try:
        data = json.loads(r.stdout)
    except Exception:
        return {"path": path, "error": "json parse"}
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format", {})
    raw = stream.get("r_frame_rate", "0/1")
    try:
        num, den = map(int, raw.split("/"))
        fps = num / den if den else 0.0
    except Exception:
        fps = 0.0
    duration = float(fmt.get("duration") or 0.0)
    nb = stream.get("nb_frames")
    if nb and nb != "N/A":
        frames = int(nb)
    else:
        frames = int(round(fps * duration)) if fps > 0 else 0
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    return {"path": path, "duration": duration, "frames": frames,
            "fps": fps, "width": width, "height": height}


def probe_many(paths: list[Path], workers: int, desc: str) -> dict[Path, dict]:
    out: dict[Path, dict] = {}
    if not paths:
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(probe, p): p for p in paths}
        it = as_completed(futs)
        if tqdm:
            it = tqdm(it, total=len(futs), desc=desc, unit="个",
                      dynamic_ncols=True)
        for fut in it:
            r = fut.result()
            out[r["path"]] = r
    return out


# ── 期望段区间（与 PipelineOp._rename_segments 命名一致）─────────────────────

def expected_segments(total: float, duration: int,
                      min_tail: float) -> list[tuple[int, int]]:
    D = duration
    n = int(total // D)
    rem = total - n * D
    pairs = [(i * D, (i + 1) * D) for i in range(n)]
    if rem >= min_tail:
        start = n * D
        end = start + D if rem >= D - 0.5 else start + int(round(rem))
        pairs.append((start, end))
    return pairs


# ── 单视频校验 ────────────────────────────────────────────────────────────────

def verify_one(src: Path, rel: Path, out_root: Path,
               src_meta: dict, seg_metas: dict[Path, dict],
               args) -> dict:
    """返回 {status, src, msg, ...}

    status:
        ok            通过
        probe_src     源 ffprobe 失败
        miss          输出不存在
        seg_miss      缺少段
        seg_extra     段命名匹配不上预期（含多余 / 重命名失败的 *_part*）
        frames        段总帧数与源差异超过容差
        duration      段总时长与源差异超过容差
        fps           段帧率与源不一致
        resolution    段分辨率与源不一致
        probe_out     输出 ffprobe 失败
    """
    if "error" in src_meta:
        return {"status": "probe_src", "src": src,
                "msg": f"源 ffprobe 失败: {src_meta['error']}"}
    total = src_meta["duration"]
    src_frames = src_meta["frames"]
    src_fps = src_meta.get("fps", 0.0)
    src_w = src_meta.get("width", 0)
    src_h = src_meta.get("height", 0)
    if total <= 0:
        return {"status": "probe_src", "src": src, "msg": "源时长 ≤ 0"}

    out_subdir = out_root if str(rel) in ("", ".") else out_root / rel
    stem = src.stem

    # 短视频：output_root/<rel>/<原名>
    if total <= args.duration + 1e-3:
        out_path = out_subdir / src.name
        if not out_path.exists():
            return {"status": "miss", "src": src,
                    "msg": f"短视频未输出: {out_path}"}
        m = seg_metas.get(out_path, {})
        if "error" in m:
            return {"status": "probe_out", "src": src,
                    "msg": f"输出 ffprobe 失败 ({out_path.name}): {m['error']}"}
        return _check_totals(src, total, src_frames, src_fps, src_w, src_h,
                             [m], expected_count=1,
                             args=args, out_paths=[out_path])

    # 长视频：output_root/<rel>/<stem>/<stem>_<start>_<end>.<ext>
    seg_folder = out_subdir / stem
    if not seg_folder.is_dir():
        return {"status": "miss", "src": src,
                "msg": f"段文件夹不存在: {seg_folder}"}

    pairs = expected_segments(total, args.duration, args.min_tail)
    expected_stems = {
        f"{stem}_{format_time(s)}_{format_time(e)}" for s, e in pairs
    }
    found = sorted(p for p in seg_folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    found_stems = {p.stem for p in found}

    missing = expected_stems - found_stems
    extra = found_stems - expected_stems

    if missing or extra:
        bits = []
        if missing:
            bits.append(f"缺 {len(missing)}: " + ", ".join(sorted(missing)[:3])
                        + (" ..." if len(missing) > 3 else ""))
        if extra:
            bits.append(f"多 {len(extra)}: " + ", ".join(sorted(extra)[:3])
                        + (" ..." if len(extra) > 3 else ""))
        return {"status": "seg_miss" if missing else "seg_extra",
                "src": src,
                "msg": f"段数 期望{len(expected_stems)}/实际{len(found_stems)} "
                       f"({'; '.join(bits)})"}

    metas = [seg_metas.get(p, {"error": "未 probe"}) for p in found]
    for p, m in zip(found, metas):
        if "error" in m:
            return {"status": "probe_out", "src": src,
                    "msg": f"段 ffprobe 失败 ({p.name}): {m['error']}"}

    return _check_totals(src, total, src_frames, src_fps, src_w, src_h,
                         metas, expected_count=len(pairs),
                         args=args, out_paths=found)


def _check_totals(src: Path, src_dur: float, src_frames: int,
                  src_fps: float, src_w: int, src_h: int,
                  metas: list[dict], expected_count: int,
                  args, out_paths: list[Path]) -> dict:
    sum_dur = sum(m["duration"] for m in metas)
    sum_frames = sum(m["frames"] for m in metas)

    dur_tol = max(args.dur_tol, src_dur * args.dur_pct / 100.0)
    frame_tol = max(args.frame_tol,
                    int(round(src_frames * args.frame_pct / 100.0)))

    dur_diff = sum_dur - src_dur
    frame_diff = sum_frames - src_frames

    if abs(dur_diff) > dur_tol:
        return {"status": "duration", "src": src,
                "msg": f"总时长 期望{src_dur:.2f}s 实际{sum_dur:.2f}s "
                       f"(Δ={dur_diff:+.2f}s, 容差±{dur_tol:.2f}s, "
                       f"{expected_count}段)"}
    if abs(frame_diff) > frame_tol:
        return {"status": "frames", "src": src,
                "msg": f"总帧数 期望{src_frames} 实际{sum_frames} "
                       f"(Δ={frame_diff:+d}, 容差±{frame_tol}, "
                       f"{expected_count}段)"}

    # 逐段检查帧率和分辨率
    for m, p in zip(metas, out_paths):
        if src_fps > 0 and abs(m.get("fps", 0) - src_fps) > args.fps_tol:
            return {"status": "fps", "src": src,
                    "msg": f"帧率不一致 ({p.name}): "
                           f"期望{src_fps:.3f} 实际{m.get('fps', 0):.3f}"}
        w, h = m.get("width", 0), m.get("height", 0)
        if src_w > 0 and src_h > 0 and (w != src_w or h != src_h):
            return {"status": "resolution", "src": src,
                    "msg": f"分辨率不一致 ({p.name}): "
                           f"期望{src_w}x{src_h} 实际{w}x{h}"}

    return {"status": "ok", "src": src, "count": expected_count,
            "sum_dur": sum_dur, "sum_frames": sum_frames,
            "out_paths": out_paths}


# ── 多余输出扫描 ─────────────────────────────────────────────────────────────

def find_orphans(out_root: Path, expected_files: set[Path]) -> list[Path]:
    """output_root 下不属于任何已知源的视频文件"""
    orphans = []
    for p in out_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            if p not in expected_files:
                orphans.append(p)
    return orphans


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True,
                    help="源视频目录 / 文件（与 main.py --input 一致）")
    ap.add_argument("--output", required=True,
                    help="待校验的输出根目录（main.py 写入的根）")
    ap.add_argument("--duration", type=int, default=60,
                    help="切分时长（秒），默认 60")
    ap.add_argument("--min_tail", type=float, default=1.0,
                    help="尾段最短保留时长（秒），默认 1.0")
    ap.add_argument("--dur_tol", type=float, default=1.0,
                    help="单视频总时长绝对容差（秒），默认 1.0")
    ap.add_argument("--dur_pct", type=float, default=1.0,
                    help="单视频总时长相对容差（%%），默认 1.0")
    ap.add_argument("--frame_tol", type=int, default=5,
                    help="单视频总帧数绝对容差（帧），默认 5")
    ap.add_argument("--frame_pct", type=float, default=1.0,
                    help="单视频总帧数相对容差（%%），默认 1.0")
    ap.add_argument("--fps_tol", type=float, default=0.01,
                    help="帧率绝对容差（fps），默认 0.01")
    ap.add_argument("--workers", type=int, default=16,
                    help="并发 probe 线程数，默认 16")
    ap.add_argument("--show_ok", action="store_true",
                    help="同时打印通过的条目（默认只打印不一致）")
    ap.add_argument("--check_orphan", action="store_true",
                    help="额外扫描 output 下多余的视频文件（源里没有对应）")
    args = ap.parse_args()

    out_root = Path(args.output).expanduser().resolve()
    if not out_root.is_dir():
        print(f"error: --output 目录不存在: {out_root}", file=sys.stderr)
        sys.exit(2)

    videos = collect_videos(args.input)
    if not videos:
        print("error: 源目录未找到视频", file=sys.stderr)
        sys.exit(2)

    print(f"源    : {len(videos)} 个视频", file=sys.stderr)
    print(f"输出根: {out_root}", file=sys.stderr)
    print(f"参数  : duration={args.duration}s, min_tail={args.min_tail}s, "
          f"容差时长±max({args.dur_tol}s, {args.dur_pct}%), "
          f"帧±max({args.frame_tol}, {args.frame_pct}%)", file=sys.stderr)

    # 第一遍 probe 源
    src_metas = probe_many([v for v, _ in videos], args.workers, "probe 源")

    # 推断每个源对应的输出路径集合，方便统一 probe
    output_paths: list[Path] = []
    src_to_outs: dict[Path, list[Path]] = {}
    for v, rel in videos:
        m = src_metas.get(v, {})
        total = m.get("duration", 0.0) if "error" not in m else 0.0
        out_subdir = out_root if str(rel) in ("", ".") else out_root / rel
        if total <= 0:
            src_to_outs[v] = []
            continue
        if total <= args.duration + 1e-3:
            p = out_subdir / v.name
            outs = [p] if p.exists() else []
        else:
            seg_folder = out_subdir / v.stem
            outs = sorted(p for p in seg_folder.iterdir()
                          if p.is_file() and p.suffix.lower() in VIDEO_EXTS
                          ) if seg_folder.is_dir() else []
        src_to_outs[v] = outs
        output_paths.extend(outs)

    seg_metas = probe_many(output_paths, args.workers, "probe 输出")

    # 校验
    issues: list[dict] = []
    oks: list[dict] = []
    for v, rel in videos:
        r = verify_one(v, rel, out_root, src_metas.get(v, {}), seg_metas, args)
        if r["status"] == "ok":
            oks.append(r)
        else:
            issues.append(r)

    # 输出
    if args.show_ok:
        for r in oks:
            print(f"[ok]    {r['src']}  {r['count']} 段  "
                  f"({r['sum_dur']:.2f}s, {r['sum_frames']} frames)")
    for r in issues:
        print(f"[{r['status']:<10}] {r['src']}  {r['msg']}")

    orphans: list[Path] = []
    if args.check_orphan:
        expected = {p for ps in src_to_outs.values() for p in ps}
        orphans = find_orphans(out_root, expected)
        for p in orphans:
            print(f"[orphan    ] {p}")

    # 汇总
    by_status: dict[str, int] = {}
    for r in issues:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    print()
    print("=" * 60)
    print(f"源视频    : {len(videos)}")
    print(f"通过      : {len(oks)}")
    if issues:
        print(f"不一致    : {len(issues)}")
        for st, n in sorted(by_status.items(), key=lambda x: -x[1]):
            print(f"  {st:<10}: {n}")
    if args.check_orphan:
        print(f"多余文件  : {len(orphans)}")

    if issues or orphans:
        sys.exit(1)


if __name__ == "__main__":
    main()
