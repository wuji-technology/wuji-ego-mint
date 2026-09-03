#!/usr/bin/env python3
"""批量视频过滤主入口（模块化）

与 reprocessed/ 同构：通过 `--ops <name>` 选择具体过滤 op。当前已实现的 op:
    hand_filter   基于 YOLO 手部检测过滤无手区间并切片

后续可在 ops/ 下新增过滤类 op（如基于其他模态的过滤），在 ops/__init__.py
注册即可，无需改 main.py。

用法:
    python main.py --list                                       # 列出可用 op
    python main.py --ops hand_filter --help                     # 查看 op 详细参数
    python main.py --ops hand_filter --input /data/videos \\
        --output /data/filtered --gpus 0,1,2,3
    python main.py --ops hand_filter --input /data --dry_run
"""
from __future__ import annotations

import argparse
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 把本目录加入 sys.path，让 ops.hand_filter 等子模块能 `from core import ...`
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from core import (
    ProcessResult,
    collect_videos,
    is_interrupted,
    kill_all_subprocesses,
)
from ops import OPS


def _install_sigint_handler() -> None:
    """首次 Ctrl+C：立刻 SIGKILL 所有 ffmpeg，让 worker 立刻返回；
    第二次 Ctrl+C：恢复默认，强制中断 Python。"""
    def _handler(signum, frame):
        n = kill_all_subprocesses()
        print(f"\n[interrupted] 收到 Ctrl+C，已 SIGKILL {n} 个 ffmpeg "
              f"（再按一次强制退出）", file=sys.stderr, flush=True)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGINT, _handler)


# ── 参数解析 ──────────────────────────────────────────────────────────────────

def resolve_op(name: str):
    """把 `--ops` 字符串解析成单 op。未知名抛 SystemExit。

    目前 filtered/ 只支持单 op；预留 `+` / `,` 组合的位置，等有第二个 op
    后按 reprocessed/main.py 的 PipelineOp / split_stages 范式回填即可。
    """
    if name not in OPS:
        raise SystemExit(
            f"error: 未知 op '{name}'（可选: {', '.join(OPS.keys())}）")
    return OPS[name]


def build_parser(op=None) -> argparse.ArgumentParser:
    """构建完整 CLI 解析器。

    op 已知时把其 op 参数也挂到主解析器；op 为 None（--help / --list 路径）
    时只挂公共参数。
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ops", metavar="NAME",
                   help=f"要执行的 op（可选: {', '.join(OPS.keys())}）")
    p.add_argument("--list", action="store_true",
                   help="列出所有可用 op 后退出")
    p.add_argument("--input", nargs="+",
                   help="输入路径（目录递归或视频文件，可多个）")
    p.add_argument("--output", default=None,
                   help="输出根目录（不指定则按 op 默认推断）")
    p.add_argument("--output_dir", default=None,
                   help="字面量输出目录；给了就直接用，不再走 <args.output>/<base.name>_<label> 拼装")
    p.add_argument("--workers", type=int, default=32,
                   help="并发处理的视频数（默认 32）。op 自带调度（如 hand_filter "
                        "的多进程流水线）时本参数被忽略")
    p.add_argument("--overwrite", action="store_true",
                   help="覆盖已存在的输出文件")
    p.add_argument("--dry_run", action="store_true",
                   help="只打印将要执行的处理计划，不实际运行")
    p.add_argument("--verify", action="store_true",
                   help="（保留位）跑完后调用同目录 verify.py 做后处理校验。"
                        "本入口当前不内嵌验收逻辑，请显式运行 verify.py")
    p.add_argument("--name_pattern", default=None,
                   help="按文件名 fnmatch 过滤（如 'aria01_214-1.mp4'）")
    if op:
        op.add_arguments(p)
    return p


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

def _run_stage(op, args, stage_label: str = "") -> tuple[Path, int, int]:
    """执行单个 op 阶段。返回 (output_dir, tot_ok, tot_fl)。

    目前 filtered/main.py 只跑一个阶段；保留这一抽象是为了将来加 pipeline /
    多阶段时复用，调用形态与 reprocessed/main.py._run_stage 一致。
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
        return Path(args.output) if args.output else Path("."), 0, 1

    output_dir = resolve_output_dir(args, op)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)  # 让 op 通过 args 拿到全局输出根

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
    print(f"{header}输出  : {output_dir}  (镜像源目录结构)")
    if hasattr(op, "describe_config"):
        print(f"{header}配置  : {op.describe_config(args)}")

    def _op_base(rel: Path) -> Path:
        return output_dir if str(rel) in ("", ".") else output_dir / rel

    if args.dry_run:
        print(f"\n{header}[dry_run]")
        for v, rel in videos:
            plan = op.plan(v, args) if hasattr(op, "plan") else "(plan 未实现)"
            print(f"  {v}  →  {_op_base(rel)}/  ({plan})")
        return output_dir, 0, 0

    results: list[tuple[Path, int, int, int]] = []

    if hasattr(op, "process_videos"):
        # op 自带批处理调度（如 hand_filter 的多进程流水线）；让 op 全权负责并发与信号处理
        items = [(v, _op_base(rel)) for v, rel in videos]
        op_results = op.process_videos(items, args)
        for r in op_results:
            results.append((r.video, r.success, r.skipped, r.failed))
    else:
        _install_sigint_handler()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(op.process_video, v, _op_base(rel), args): (v, rel)
                for v, rel in videos
            }
            it = as_completed(futures)
            if tqdm:
                it = tqdm(it, total=len(futures), desc=op_str)
            for fut in it:
                video, _rel = futures[fut]
                try:
                    r = fut.result()
                    results.append((r.video, r.success, r.skipped, r.failed))
                except Exception:
                    results.append((video, 0, 0, 1))

    tot_ok = sum(r[1] for r in results)
    tot_sk = sum(r[2] for r in results)
    tot_fl = sum(r[3] for r in results)
    status = "中断" if is_interrupted() else "完成"
    print(f"\n{header}{status}: 视频 {len(videos)} | 输出 成功 {tot_ok} / 已存在 {tot_sk} / 失败 {tot_fl}")
    print(f"{header}输出目录: {output_dir}")

    if args.verify:
        print(f"{header}[verify] 请显式运行: python {Path(__file__).parent / 'verify.py'} "
              f"--input <src> --output {output_dir}", file=sys.stderr)

    return output_dir, tot_ok, tot_fl


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
            print(f"  {name:<14} {op_mod.DESCRIPTION}")
        return

    op_for_help = resolve_op(args0.ops) if args0.ops else None
    parser = build_parser(op_for_help)
    args = parser.parse_args()  # 此时会处理 --help、未知参数等

    if not args.ops:
        parser.print_help()
        print("\nerror: 必须指定 --ops <name>，例如 --ops hand_filter（用 --list 查看）",
              file=sys.stderr)
        sys.exit(2)
    if not args.input:
        parser.print_help()
        print("\nerror: 必须指定 --input <path>", file=sys.stderr)
        sys.exit(2)

    op = resolve_op(args.ops)
    output_dir, _ok, fl = _run_stage(op, args)

    if is_interrupted():
        sys.exit(130)
    if fl > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
