#!/usr/bin/env python3
"""Run the local Ray video-processing pipeline and export LeRobot datasets."""

import argparse
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path







import os as _os
_RAY_DIR  = Path(__file__).resolve().parent
_PROJ_DIR = str(_RAY_DIR.parent)
_RAY_DIR  = str(_RAY_DIR)
_existing = _os.environ.get('PYTHONPATH', '')
_os.environ['PYTHONPATH'] = ':'.join(
    p for p in [_RAY_DIR, _PROJ_DIR, _existing] if p
)
if _RAY_DIR not in sys.path:
    sys.path.insert(0, _RAY_DIR)
_RAY_DIR = Path(_RAY_DIR)











_os.environ.setdefault('MALLOC_ARENA_MAX', '2')
_os.environ.setdefault('MALLOC_TRIM_THRESHOLD_', '134217728')
if __name__ == '__main__' and _os.environ.get('MINT_MALLOC_TUNED') != '1':
    _os.environ['MINT_MALLOC_TUNED'] = '1'
    print(f'[pipeline]'
          f'[pipeline]  {_os.environ["MALLOC_ARENA_MAX"]}.', flush=True)
    _os.execv(sys.executable, [sys.executable] + sys.argv)


_tmp_dir = _RAY_DIR.parent / 'output' / 'tmp'
_tmp_dir.mkdir(parents=True, exist_ok=True)
_os.environ['TMPDIR'] = str(_tmp_dir)


def _fix_ptxas() -> None:
    import stat
    import sysconfig


    site_packages = Path(sysconfig.get_path('purelib'))
    nvcc_ptxas = site_packages / 'nvidia' / 'cuda_nvcc' / 'bin' / 'ptxas'
    if nvcc_ptxas.exists():
        if not _os.access(str(nvcc_ptxas), _os.X_OK):
            try:
                nvcc_ptxas.chmod(nvcc_ptxas.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except PermissionError:
                pass
        if _os.access(str(nvcc_ptxas), _os.X_OK):
            _os.environ['TRITON_PTXAS_PATH'] = str(nvcc_ptxas)
            print(f'[pipeline]  {nvcc_ptxas}.')
            return


    conda_prefix = _os.environ.get('CONDA_PREFIX', '')
    if not conda_prefix:
        return
    ptxas = Path(conda_prefix) / 'lib/python3.10/site-packages/triton/third_party/cuda/bin/ptxas'
    if ptxas.exists() and not _os.access(str(ptxas), _os.X_OK):
        try:
            ptxas.chmod(ptxas.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            print(f'[pipeline]  {ptxas}.')
        except PermissionError:
            print(f'[pipeline]  {ptxas}.')

_fix_ptxas()

from runtime.cluster import init as init_ray, shutdown as shutdown_ray   # noqa: E402
from runtime.runner  import run_pipeline_multi                           # noqa: E402
from monitoring      import start as start_monitor, stop as stop_monitor # noqa: E402
from manifest        import (                                            # noqa: E402
    scan_input, prepare_manifest, format_summary, Status,
    mark_done, mark_failed, rollback_running,
    mark_running, mark_clip_stage, scene_key,
)
from tasks.lerobot_aggregator import IncrementalAggregator                # noqa: E402


def _start_per_video_lerobot_export(parent_work_dir: Path,
                                    use_raw_traj: bool = False) -> subprocess.Popen:
    script = Path(__file__).resolve().parent / 'tasks' / 'result_to_lerobot.py'
    target = parent_work_dir / 'lerobot_v3'
    log_path = parent_work_dir / 'result_to_lerobot.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        '--result', str(parent_work_dir),
        '--output', str(target),
        '--overwrite',
    ]
    if use_raw_traj:
        cmd.append('--use_raw_traj')
    with open(log_path, 'ab') as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f'[pipeline]  {parent_work_dir.name}; {proc.pid}; {target}.')
    return proc


def _cleanup_part_all_frames(part_dir: Path) -> None:
    import shutil

    def _sv_ok(clip_dir: Path) -> bool:
        sv = clip_dir / 'result' / 'source_video.mp4'
        try:
            return sv.is_symlink() and sv.resolve(strict=True).is_file()
        except (OSError, RuntimeError):
            return False

    clip_dirs = sorted({p.parent.parent for p in part_dir.glob('*/result/source_video.mp4')})
    any_ok = False
    for clip in clip_dirs:
        if not _sv_ok(clip):
            print(f'[pipeline]  {clip.name}.')
            continue
        any_ok = True
        link = clip / 'result' / 'frames'
        if link.is_symlink():
            try:
                link.unlink()
            except Exception as exc:
                print(f'[pipeline]  {link}; {exc}.')
        frames = clip / 'frames'
        if frames.is_dir() and not frames.is_symlink():
            try:
                shutil.rmtree(frames)
                print(f'[delete_temp] removed {frames}')
            except Exception as exc:
                print(f'[pipeline]  {frames}; {exc}.')

    frames_dir = part_dir / '_all_frames'
    if frames_dir.exists():
        if not any_ok:
            print(f'[pipeline]  {frames_dir}.')
            return
        try:
            shutil.rmtree(frames_dir)
            print(f'[delete_temp] removed {frames_dir}')
        except Exception as exc:
            print(f'[pipeline]  {frames_dir}; {exc}.')


def _resume_clip_stages_for_entry(entry) -> dict:
    out: dict[str, dict[str, bool]] = {}
    clips = entry.clips if isinstance(entry.clips, dict) else {}
    for clip_key, clip_state in clips.items():
        if not isinstance(clip_state, dict):
            continue
        stages: dict[str, bool] = {}
        for stage in ('geo', 'moge', 'hawor', 'megasam'):
            state = clip_state.get(stage)
            status = state.get('status') if isinstance(state, dict) else state
            if status == 'done':
                stages[stage] = True
        if stages:
            out[clip_key] = stages
    return out


def _strip_argv(argv: list[str], opts: set[str]) -> list[str]:
    """Internal helper."""
    out, i = [], 0
    while i < len(argv):
        tok = argv[i]
        if tok.split('=', 1)[0] in opts:
            if '=' in tok:
                i += 1
                continue
            i += 1
            while i < len(argv) and not argv[i].startswith('-'):
                i += 1
            continue
        out.append(tok)
        i += 1
    return out



def _used_mem_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().used / (1024 ** 3)
    except Exception:
        pass
    try:
        info: dict[str, int] = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, _, rest = line.partition(':')
                info[k] = int(rest.strip().split()[0])    # kB
        used_kb = info['MemTotal'] - info.get('MemAvailable', info.get('MemFree', 0))
        return used_kb / (1024 ** 2)
    except Exception:
        return None



def _scan_cache_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / 'scan_cache.txt'


def _write_scan_cache(output_dir: str | Path, input_root, videos) -> None:
    path = _scan_cache_path(output_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(f'#input_root={Path(input_root).resolve().as_posix()}\n')
            for v in videos:
                f.write(f'{Path(v).resolve().as_posix()}\n')
        tmp.replace(path)
        print(f'[pipeline]  {path}; {len(videos)}.', flush=True)
    except Exception as e:
        print(f'[pipeline]  {path}; {e}.', flush=True)


def _read_scan_cache(output_dir: str | Path):
    """Internal helper."""
    path = _scan_cache_path(output_dir)
    if not path.exists():
        return None
    try:
        input_root = None
        videos: list[Path] = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('#input_root='):
                    input_root = Path(line[len('#input_root='):]).expanduser()
                    continue
                if line.startswith('#'):
                    continue
                videos.append(Path(line))
        if input_root is None or not videos:
            return None
        return input_root, videos
    except Exception as e:
        print(f'[pipeline]  {path}; {e}.', flush=True)
        return None


def _filter_videos_by_label(
    videos: list,
    label_dir: str,
    output_dir: str,
) -> list:
    import re as _re
    label_root = Path(label_dir)
    if not label_root.is_dir():
        print(f'[pipeline]  {label_dir}.')
        return videos


    _fixed_re = _re.compile(r'_fixed_[0-9]+(?:\.[0-9]+)?s\.json$')
    label_stems: set[str] = set()
    for p in label_root.rglob('*_fixed_*s.json'):
        base = _fixed_re.sub('', p.name)
        if base:
            label_stems.add(base)

    has_label: list = []
    no_label:  list = []
    for video in videos:
        if video.stem in label_stems:
            has_label.append(video)
        else:
            no_label.append(video)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / 'no_label_videos.txt'
    if no_label:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f'# label_dir: {label_dir}\n')
            f.write(f'[pipeline]  {len(no_label)}.')
            for v in no_label:
                f.write(f'{v}\n')
        print(f'[pipeline]  {len(no_label)}; {len(videos)}.'
              f'[pipeline]  {log_path}.')
    else:

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f'# label_dir: {label_dir}\n')
            f.write(f'[pipeline]  {len(videos)}.')
        print(f'[pipeline]  {len(videos)}.')

    return has_label


def _filter_videos_by_sidecar_label(videos: list, log_dir: Path) -> list:
    from concurrent.futures import ThreadPoolExecutor
    from steps.cpu.result_sidecars import _merged_label_paths
    from tqdm import tqdm

    def _has_label(video) -> bool:


        candidates = _merged_label_paths(video, video.stem)
        return any(p.exists() for p in candidates)






    try:
        n_workers = int(_os.environ.get('MINT_LABEL_SCAN_WORKERS', '0'))
    except ValueError:
        n_workers = 0
    if n_workers <= 0:
        n_workers = max(1, min(32, (_os.cpu_count() or 8) // 2))

    no_label: list = []
    kept:     list = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        flags = tqdm(ex.map(_has_label, videos), total=len(videos),
                     desc=f'[pipeline]  {n_workers}.',
                     unit='vid')
        for video, ok in zip(videos, flags):
            (kept if ok else no_label).append(video)

    if no_label:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'no_label_videos.txt'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f'[pipeline]  {len(no_label)}.')
            for v in no_label:
                f.write(f'{v}\n')
        print(f'[pipeline]  {len(no_label)}; {len(videos)}.'
              f'[pipeline]  {log_path}.')
    else:
        print(f'[pipeline]  {len(videos)}.')
    return kept


def _dispatch_multi_inputs(args) -> None:
    """Internal helper."""
    inputs = list(args.input)
    do_resume = args.resume is not None


    if do_resume:
        parent = Path(args.resume).expanduser().resolve()
        if not parent.is_dir():
            raise SystemExit(f'[pipeline]  {parent}.')
    elif args.output is not None:
        parent = Path(args.output).expanduser().resolve()
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_root = _os.environ.get('MINT_OUTPUT_ROOT') or str(_RAY_DIR.parent / 'output' / 'ray')
        parent = Path(out_root) / ts
    print(f'[pipeline]  {len(inputs)}; {parent}.')


    used: dict[str, int] = {}
    plan: list[tuple[str, Path]] = []
    for ip in inputs:
        name = Path(ip).expanduser().resolve().name or 'input'
        seen = used.get(name, 0)
        used[name] = seen + 1
        sub = name if seen == 0 else f'{name}_{seen + 1:02d}'
        plan.append((ip, parent / sub))
        print(f'[pipeline]  {ip}; {parent / sub}.')


    base = _strip_argv(sys.argv[1:], {'--input', '--output', '--resume', '--manifest'})
    self_path = str(Path(__file__).resolve())

    fails: list[tuple[str, int]] = []
    for i, (ip, sub) in enumerate(plan, 1):
        print(f'[pipeline]  {"=" * 72}; {i}; {len(plan)}; {ip}; {sub}; {"=" * 72}.')
        child = [sys.executable, self_path, *base, '--input', ip]
        child += (['--resume', str(sub)] if do_resume else ['--output', str(sub)])
        rc = subprocess.run(child).returncode
        if rc != 0:
            fails.append((ip, rc))
            print(f'[pipeline]  {i}; {len(plan)}; {rc}.')
    ok = len(plan) - len(fails)
    print(f'[pipeline]  {ok}; {len(plan)}.' + (f'[pipeline]  {fails}.' if fails else ''))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Process egocentric videos with the MINT Ray pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--conda', help='Optional Conda environment for Ray workers')
    parser.add_argument('--input', required=True, nargs='+', metavar='PATH',
                        help='Input video, directory, or multiple input roots')
    parser.add_argument('--output', default=None,
                        help='Output directory; defaults to output/ray/<timestamp>')
    parser.add_argument('--num_gpus',          type=int,   default=None,
                        help='Number of GPUs available to Ray')
    parser.add_argument('--frame_stride', type=int, default=1,
                        help='Decode every Nth source frame')
    parser.add_argument('--geocalib_interval', type=float, default=5.0,
                        help='GeoCalib sampling interval in seconds')
    parser.add_argument('--moge_model', default=None,
                        help='Optional MoGe model or checkpoint override')
    parser.add_argument('--ba_steps1',         type=int,   default=10,
                        help='First bundle-adjustment iteration count')
    parser.add_argument('--ba_steps2',         type=int,   default=15,
                        help='Second bundle-adjustment iteration count')
    parser.add_argument('--ba_steps3',         type=int,   default=3,
                        help='Final bundle-adjustment iteration count')
    parser.add_argument('--long_video_threshold_s', type=float, default=float('inf'),
                        help='Split videos longer than this many seconds')
    parser.add_argument('--clip_duration_s',   type=float, default=10.0,
                        help='Duration of each split clip in seconds')
    parser.add_argument('--clip_overlap_s',    type=float, default=0.0,
                        help='Overlap between adjacent split clips in seconds')
    parser.add_argument('--log',               action='store_true',
                        help='Enable verbose Ray worker logs')
    parser.add_argument('--compile',            action='store_true',
                        help='Enable supported model compilation paths')
    parser.add_argument('--no_monitor',        action='store_true',
                        help='Disable resource monitoring')
    parser.add_argument('--slam_start_delay',  type=float, default=0.0,
                        help='Delay SLAM actor startup by this many seconds')
    parser.add_argument('--max_open_videos', type=int, default=12,
                        help='High watermark for concurrently open videos')
    parser.add_argument('--low_open_videos', type=int, default=6,
                        help='Low watermark for concurrently open videos')
    parser.add_argument('--max_open_clip_credit', type=int, default=48,
                        help='High watermark for queued clip credits')
    parser.add_argument('--low_open_clip_credit', type=int, default=24,
                        help='Low watermark for queued clip credits')
    parser.add_argument('--slam_steal_moge', action='store_true',
                        help='Allow SLAM workers to use idle MoGe GPU capacity')
    parser.add_argument('--delete_temp', choices=['yes', 'no'], default='yes',
                        help='Delete intermediate frame and clip files after export')
    parser.add_argument('--resume', default=None, metavar='OUTPUT_DIR',
                        help='Resume an existing output directory')
    parser.add_argument('--retry_failed', action='store_true',
                        help='Retry entries marked failed in the manifest')
    parser.add_argument('--manifest', default=None,
                        help='Custom run manifest path')
    parser.add_argument('--aggregate_lerobot', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Build an aggregate LeRobot v3 dataset')
    parser.add_argument('--label_dir', default=None,
                        help='Optional directory of approved video labels')
    parser.add_argument('--test', action='store_true',
                        help='Run a short pipeline validation configuration')
    parser.add_argument('--use_raw_traj', action='store_true',
                        help='Export raw trajectories instead of cleaned trajectories')
    parser.add_argument('--per_video_lerobot', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Export one LeRobot dataset per source video')
    parser.add_argument('--recycle_mem_gb', type=float, default=None,
                        help='Restart the driver above this memory threshold; 0 disables it')
    args = parser.parse_args()


    if args.recycle_mem_gb is None:
        try:
            args.recycle_mem_gb = float(_os.environ.get('MINT_RECYCLE_MEM_GB', '0'))
        except ValueError:
            args.recycle_mem_gb = 0.0




    if len(args.input) > 1:
        return _dispatch_multi_inputs(args)
    args.input = args.input[0]




    if args.test and args.long_video_threshold_s == float('inf'):
        args.long_video_threshold_s = 0.0
        print(f'[pipeline]  {args.clip_duration_s}.'
              f'(long_video_threshold_s=0)')





    do_resume = args.resume is not None
    if do_resume:
        resume_dir = Path(args.resume).expanduser().resolve()
        if not resume_dir.is_dir():
            parser.error(f'[pipeline]  {resume_dir}.')
        if args.output is not None:
            if Path(args.output).expanduser().resolve() != resume_dir:
                parser.error(
                    f'[pipeline]  {resume_dir}; {args.output}.'
                    '[pipeline]'
                )
        args.output = str(resume_dir)
        print(f'[pipeline]  {args.output}.')

    if args.output is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        out_root = _os.environ.get('MINT_OUTPUT_ROOT') or str(_RAY_DIR.parent / 'output' / 'ray')
        args.output = str(Path(out_root) / ts)
        print(f'[pipeline]  {args.output}.')




    cached = None
    if do_resume and _os.environ.get('MINT_SCAN_CACHE', '1') != '0':
        cached = _read_scan_cache(args.output)

    if cached is not None:
        input_root, scanned_videos = cached
        print(f'[pipeline]  {len(scanned_videos)}.'
              f'{_scan_cache_path(args.output)}', flush=True)
        if not scanned_videos:
            print('[pipeline]')
            return
    else:
        try:
            input_root, scanned_videos = scan_input(args.input)
        except ValueError as e:
            parser.error(str(e))

        if not scanned_videos:
            print('[pipeline]')
            return






        if args.test:
            print(f'[pipeline]  {len(scanned_videos)}.'
                  '[pipeline]')
        elif args.label_dir:
            scanned_videos = _filter_videos_by_label(
                scanned_videos, args.label_dir, args.output)
        else:


            log_dir = (_RAY_DIR.parent / 'output' / 'logs' / 'ray'
                       / datetime.now().strftime('%Y%m%d_%H%M%S'))
            scanned_videos = _filter_videos_by_sidecar_label(scanned_videos, log_dir)
        if not scanned_videos:
            print('[pipeline]')
            return


        _write_scan_cache(args.output, input_root, scanned_videos)


    manifest_path = args.manifest or str(Path(args.output) / 'run_batch_manifest.json')
    Path(args.output).mkdir(parents=True, exist_ok=True)
    if do_resume and not Path(manifest_path).exists() and args.manifest is None:
        parser.error(
            f'[pipeline]  {manifest_path}.'
            '[pipeline]'
        )
    if not do_resume and Path(manifest_path).exists():
        print(f'[pipeline]  {manifest_path}.'
              '[pipeline]')
    manifest, recon = prepare_manifest(
        input_root, scanned_videos,
        manifest_path = manifest_path,
        resume        = do_resume,
        retry_failed  = args.retry_failed,
    )
    print(f'[pipeline]  {manifest.path}.')
    print(format_summary(recon, manifest, scanned_total=len(scanned_videos)))
    if recon.missing:
        print(f'[pipeline]  {len(recon.missing)}.'
              f'[pipeline]  {recon.missing[:3]}; {"..." if len(recon.missing) > 3 else ""}.')

    if not recon.pending:
        print('[pipeline]'
              '[pipeline]')
        return



    pending_keys = {scene_key(p, input_root): k for k, p in recon.pending}
    videos = [str(p) for _, p in recon.pending]






    resume_clip_stages: dict[str, dict] = {}
    for key, path in recon.pending:
        entry = manifest.videos.get(key)
        if entry is None:
            continue
        resume_clip_stages[scene_key(path, input_root)] =\
            _resume_clip_stages_for_entry(entry)

    if args.long_video_threshold_s == float('inf'):
        print('[pipeline]')
    else:
        print(f'[pipeline]  {args.long_video_threshold_s}.'
              f'[pipeline]  {args.clip_duration_s}; {args.clip_overlap_s}.')

    # NOTE Env vars are captured to propagate on Ray initialization, i.e.
    # either `ray start --head` in shell or `ray.init()` in code, whoever
    # happens earlier. Because a cluster always has Ray started before
    # our code, it is always necessary to use Ray runtime env instead.
    runtime_env = {
        "env_vars": {
            'PYTHONPATH': _os.environ['PYTHONPATH'],
            'TMPDIR': _os.environ['TMPDIR'],
            'MINT_LONG_VIDEO_THRESHOLD_S': str(args.long_video_threshold_s),
            'MINT_CLIP_DURATION_S': str(args.clip_duration_s),
            'MINT_CLIP_OVERLAP_S': str(args.clip_overlap_s),






            'MALLOC_ARENA_MAX': _os.environ.get('MALLOC_ARENA_MAX', '2'),
            'MALLOC_TRIM_THRESHOLD_': _os.environ.get('MALLOC_TRIM_THRESHOLD_', '134217728'),


            'MINT_MALLOC_TRIM_EVERY': _os.environ.get('MINT_MALLOC_TRIM_EVERY', '24'),



            'MINT_TRACEMALLOC': _os.environ.get('MINT_TRACEMALLOC', '0'),
            'MINT_TRACEMALLOC_FRAMES': _os.environ.get('MINT_TRACEMALLOC_FRAMES', '1'),
        },
    }
    # NOTE Read cpu nums and divide evenly for four stage. Whereas Ray option
    # `num_cpus` is purely logical that does not limit actual CPU usage, not
    # specifying the option will cause OMP_NUM_THREADS be set to 1, which may
    # affect torch and numpy.
    match = re.search(r'--num-cpus=(\d+)', _os.environ.get('KUBERAY_GEN_RAY_START_CMD', ''))
    if match:
        runtime_env['env_vars']['OMP_NUM_THREADS'] = str(int(match[1]) // 4)

    if args.conda:
        runtime_env['conda'] = args.conda

    init_ray(num_gpus=args.num_gpus, log=args.log, runtime_env=runtime_env)






    launch_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.output) / 'logs' / launch_ts
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f'[pipeline]  {log_dir}.')

    monitor = start_monitor(log_dir) if not args.no_monitor else None




    interrupted = {'active': False}


    recycle = {'requested': False}

    def _force_exit(sig: int, _frame) -> None:
        name = signal.Signals(sig).name if hasattr(signal, 'Signals') else str(sig)
        print(f'[pipeline]  {name}.', flush=True)
        _os._exit(128 + sig)

    def _teardown_and_reexec() -> None:
        used = _used_mem_gb()
        used_s = f'{used:.1f}GB' if used is not None else '?'
        print(f'[pipeline]  {used_s}; {args.recycle_mem_gb}.'
              f'[pipeline]  {args.output}.', flush=True)
        if not started_scenes:
            print('[pipeline]'
                  '[pipeline]',
                  flush=True)
        try:
            n = rollback_running(manifest)
            if n:
                print(f'[pipeline]  {n}.', flush=True)
        except Exception as e:
            print(f'[pipeline]  {e}.', flush=True)
        try:
            if monitor is not None:
                stop_monitor(monitor, final_plot=False)
        except Exception as e:
            print(f'[pipeline]  {e}.', flush=True)
        try:
            shutdown_ray()
        except Exception:
            pass


        base = _strip_argv(sys.argv[1:], {'--input', '--output', '--resume', '--manifest'})
        self_path = str(Path(__file__).resolve())
        new_argv = [sys.executable, self_path, *base,
                    '--input', str(args.input), '--resume', str(args.output)]
        _os.environ['MINT_RECYCLED'] = '1'
        print(f'[recycle] re-exec: {" ".join(new_argv[1:])}', flush=True)
        _os.execv(sys.executable, new_argv)

    def _on_signal(sig: int, _frame) -> None:
        name = signal.Signals(sig).name if hasattr(signal, 'Signals') else str(sig)

        if recycle['requested'] and not interrupted['active']:
            interrupted['active'] = True
            _teardown_and_reexec()
            return
        if interrupted['active']:
            _force_exit(sig, _frame)
        interrupted['active'] = True
        signal.signal(signal.SIGINT, _force_exit)
        signal.signal(signal.SIGTERM, _force_exit)
        print(f'[pipeline]  {name}.', flush=True)

        try:
            n = rollback_running(manifest)
            if n:
                print(f'[pipeline]  {n}.')
        except Exception as e:
            print(f'[pipeline]  {e}.')
        try:
            if monitor is not None:
                stop_monitor(monitor, final_plot=False)
        except Exception as e:
            print(f'[pipeline]  {e}.')
        try:
            shutdown_ray()
        except Exception:
            pass

        _os._exit(128 + sig)

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    import ray as _ray
    n_gpus_actual = int(_ray.cluster_resources().get('GPU', 0))



    per_video_procs: list[subprocess.Popen] = []




    started_scenes: set[str] = set()





    if args.recycle_mem_gb and args.recycle_mem_gb > 0:
        try:
            _poll_s = float(_os.environ.get('MINT_RECYCLE_POLL_S', '15'))
        except ValueError:
            _poll_s = 15.0
        try:
            _min_uptime = float(_os.environ.get('MINT_RECYCLE_MIN_UPTIME_S', '300'))
        except ValueError:
            _min_uptime = 300.0

        def _recycle_watchdog() -> None:
            t0 = time.monotonic()
            while not recycle['requested'] and not interrupted['active']:
                time.sleep(_poll_s)
                if recycle['requested'] or interrupted['active']:
                    return
                used = _used_mem_gb()
                if used is None:
                    continue
                if used >= args.recycle_mem_gb and (time.monotonic() - t0) >= _min_uptime:
                    recycle['requested'] = True
                    print(f'[pipeline]  {used:.1f}.'
                          f'[pipeline]  {args.recycle_mem_gb}.', flush=True)
                    _os.kill(_os.getpid(), signal.SIGTERM)
                    return

        threading.Thread(target=_recycle_watchdog,
                         name='recycle-watchdog', daemon=True).start()
        print(f'[pipeline]  {args.recycle_mem_gb}.'
              f'[pipeline]  {_poll_s:.0f}; {_min_uptime:.0f}.', flush=True)



    aggregator = (
        IncrementalAggregator(Path(args.output), log_dir=log_dir)
        if args.aggregate_lerobot else None
    )






    def _on_clip_done(result: dict) -> None:
        parent_scene = result.get('parent_scene') or result.get('scene')
        key = pending_keys.get(parent_scene)
        if key is None:
            print(f'[pipeline]  {parent_scene}.')
            return

        if parent_scene not in started_scenes:
            started_scenes.add(parent_scene)
            mark_running(manifest, key)
        clip_idx = int(result.get('clip_idx', 0))

        entry = manifest.get(key)
        status_before = entry.status
        if result.get('error'):
            mark_failed(manifest, key, clip_idx=clip_idx, error=str(result['error']))
        else:
            mark_done(
                manifest, key, clip_idx=clip_idx,
                output_dir = result.get('cleaned') or result.get('work_dir'),
                duration_s = result.get('elapsed') or result.get('t_post'),
            )
        if (args.per_video_lerobot
                and entry.status == Status.DONE
                and status_before != Status.DONE):
            parent_work = Path(args.output) / parent_scene
            if parent_work.is_dir():
                try:
                    proc = _start_per_video_lerobot_export(parent_work, args.use_raw_traj)
                    per_video_procs.append(proc)
                    def _watch(p=proc, d=parent_work):
                        try:
                            p.wait()
                        except Exception:
                            return
                        if p.returncode == 0:
                            if args.delete_temp == 'yes':
                                _cleanup_part_all_frames(d)
                            if aggregator is not None:
                                aggregator.notify(d)
                        elif args.delete_temp == 'yes':
                            print(f'[pipeline]  {d.name}.'
                                  f'[pipeline]  {p.returncode}.')
                    threading.Thread(target=_watch, daemon=True).start()
                except Exception as e:
                    print(f'[pipeline]  {parent_scene}; {e}.')

    def _on_stage_done(event: dict) -> None:
        scene = event.get('scene')
        key = pending_keys.get(scene)
        if key is None:
            print(f'[pipeline]  {scene}.')
            return

        if scene not in started_scenes:
            started_scenes.add(scene)
            mark_running(manifest, key)
        mark_clip_stage(
            manifest, key,
            clip_idx=int(event.get('clip_idx', 0)),
            stage=str(event.get('stage')),
            status=str(event.get('status', 'failed')),
            error=event.get('error'),
            extra={
                k: event[k]
                for k in ('item_scene', 'path')
                if k in event and event[k] is not None
            },
        )

    try:
        run_pipeline_multi(
            videos            = videos,
            output_dir        = args.output,
            frame_stride      = args.frame_stride,
            geocalib_interval = args.geocalib_interval,
            moge_model        = args.moge_model,
            ba_steps1         = args.ba_steps1,
            ba_steps2         = args.ba_steps2,
            ba_steps3         = args.ba_steps3,
            use_compile       = args.compile,
            n_gpus            = n_gpus_actual,
            monitor           = monitor,
            slam_start_delay  = args.slam_start_delay,
            max_open_videos   = args.max_open_videos,
            low_open_videos   = args.low_open_videos,
            max_open_clip_credit = args.max_open_clip_credit,
            low_open_clip_credit = args.low_open_clip_credit,
            slam_steal_moge = args.slam_steal_moge,
            delete_temp     = (args.delete_temp == 'yes'),
            on_video_done   = _on_clip_done,
            on_stage_done   = _on_stage_done,
            resume_clip_stages = resume_clip_stages,
            label_dir       = args.label_dir,
            input_root      = str(input_root),
        )
    finally:
        if not interrupted['active']:

            try:
                n = rollback_running(manifest)
                if n:
                    print(f'[pipeline]  {n}.')
            except Exception as e:
                print(f'[pipeline]  {e}.')
            if monitor is not None:
                stop_monitor(monitor)
            shutdown_ray()


    if per_video_procs:
        print(f'[pipeline]  {len(per_video_procs)}.')
        for p in per_video_procs:
            try:
                p.wait()
            except Exception as e:
                print(f'[pipeline]  {p.pid}; {e}.')
        ok_n = sum(1 for p in per_video_procs if p.returncode == 0)
        print(f'[pipeline]  {ok_n}; {len(per_video_procs)}.')


    if aggregator is not None:
        print('[pipeline]')
        aggregator.shutdown()
        if aggregator.last_returncode not in (None, 0):
            print(f'[pipeline]  {aggregator.last_returncode}.')

if __name__ == '__main__':
    main()
