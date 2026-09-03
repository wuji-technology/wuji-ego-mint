"""Shared MegaSAM runtime used by regular and promoted tail SLAM actors."""

from __future__ import annotations

import os
import queue
import sys
import time
import traceback
from pathlib import Path

import ray
from ray.util.queue import Queue as RayQueue

from ._cuda import configure_memory_cap, move_to_cpu, release_cuda_cache
from ._utils import emit_stage_event, _wait_frames, MallocTrimmer, TraceMallocProbe

_NO_FIRST_ITEM = object()


def _cleanup_clip_moge_depth(moge_dir, scene: str) -> None:
    if not moge_dir or not scene:
        return
    import shutil
    base = Path(moge_dir)
    scene = Path(scene).name
    for sub in ('Depth-Anything', 'UniDepth'):
        target = base / sub / scene
        if target.exists():
            try:
                shutil.rmtree(target)
                print(f'[delete_temp] removed {target}')
            except Exception as exc:
                print(f'[pipeline]  {target}; {exc}.')


def _slam_weights_path() -> str:
    return str(Path(__file__).resolve().parents[2] / 'model' / 'megasam' / 'megasam_final.pth')


def _build_droid_net_cpu_without_megasam(weights: str):
    """Load DroidNet on CPU without importing the MegaSAM backend."""
    import importlib
    import sys
    from collections import OrderedDict as _OD

    import torch

    root = Path(__file__).resolve().parents[2]
    megasam_root = root / 'third_party' / 'mega-sam'
    droid_slam_path = megasam_root / 'base' / 'droid_slam'
    for path in (root, megasam_root / 'UniDepth', droid_slam_path):
        path_s = str(path)
        if path_s in sys.path:
            sys.path.remove(path_s)
        sys.path.insert(0, path_s)

    modules_pkg = sys.modules.get('modules')
    expected_modules_dir = (droid_slam_path / 'modules').resolve()
    loaded_paths = (
        {Path(path).resolve() for path in getattr(modules_pkg, '__path__', ())}
        if modules_pkg is not None
        else set()
    )
    if modules_pkg is not None and expected_modules_dir not in loaded_paths:
        for name in list(sys.modules):
            if name == 'modules' or name.startswith('modules.'):
                sys.modules.pop(name, None)
    importlib.import_module('modules')

    from droid_net import DroidNet as _DroidNet
    net = _DroidNet()
    sd = _OD([
        (k.replace('module.', ''), v)
        for k, v in torch.load(weights, map_location='cpu').items()
    ])
    sd['update.weight.2.weight'] = sd['update.weight.2.weight'][:2]
    sd['update.weight.2.bias']   = sd['update.weight.2.bias'][:2]
    sd['update.delta.2.weight']  = sd['update.delta.2.weight'][:2]
    sd['update.delta.2.bias']    = sd['update.delta.2.bias'][:2]
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


def start_slam_cpu_prewarm(owner) -> None:
    """Start a CPU-only MegaSAM weight preload in this actor process."""
    import threading

    if getattr(owner, '_slam_cpu_prewarm_started', False):
        return
    owner._slam_cpu_prewarm_started = True

    def _worker() -> None:
        gpu = getattr(owner, '_gpu', os.environ.get('CUDA_VISIBLE_DEVICES', '?'))
        try:
            owner._slam_cpu_net = _build_droid_net_cpu_without_megasam(
                _slam_weights_path())
            print(f'[pipeline]  {gpu}.')
        except Exception as exc:
            owner._slam_cpu_net = None
            owner._slam_cpu_prewarm_error = exc
            print(f'[pipeline]  {gpu}; {exc}.')

    thread = threading.Thread(
        target=_worker,
        daemon=True,
        name=f'slam-cpu-prewarm-gpu{getattr(owner, "_gpu", "?")}',
    )
    owner._slam_cpu_prewarm_thread = thread
    thread.start()


def _build_moge_cpu(moge_model: str | None):
    from steps.gpu.moge import load_moge_model
    model, _ = load_moge_model(moge_model, device='cpu')
    return model


def start_moge_cpu_prewarm(owner, moge_model: str | None) -> None:
    """Start a CPU-only MoGe weight preload in the actor process.

    Geo / HaWoR actors steal MoGe work when their input queue drains; without
    this prewarm, the first steal pays a synchronous disk + state_dict load.
    Started at run_loop entry (not __init__) so it doesn't race against the
    actor's main-model load for disk bandwidth.
    """
    import threading

    if getattr(owner, '_moge_cpu_prewarm_started', False):
        return
    owner._moge_cpu_prewarm_started = True
    owner._moge_cpu_prewarm_moge_model = moge_model
    owner._moge_cpu_net = None

    def _worker() -> None:
        gpu = getattr(owner, '_gpu', os.environ.get('CUDA_VISIBLE_DEVICES', '?'))
        try:
            owner._moge_cpu_net = _build_moge_cpu(moge_model)
            print(f'[pipeline]  {gpu}.')
        except Exception as exc:
            owner._moge_cpu_net = None
            owner._moge_cpu_prewarm_error = exc
            print(f'[pipeline]  {gpu}; {exc}.')

    thread = threading.Thread(
        target=_worker,
        daemon=True,
        name=f'moge-cpu-prewarm-gpu{getattr(owner, "_gpu", "?")}',
    )
    owner._moge_cpu_prewarm_thread = thread
    thread.start()


def adopt_moge_cpu_prewarm(owner):
    """Join the prewarm thread and pop the CPU model off owner; None if absent."""
    thread = getattr(owner, '_moge_cpu_prewarm_thread', None)
    if thread is not None and thread.is_alive():
        thread.join()
    net = getattr(owner, '_moge_cpu_net', None)
    if net is None:
        return None
    try:
        owner._moge_cpu_net = None
    except Exception:
        pass
    return net


def pin_module_(module) -> None:
    import os

    if os.environ.get('MINT_PIN_CPU_CACHE', '0') != '1':
        return
    if module is None:
        return
    try:
        import torch  # noqa: F401
    except Exception:
        return

    state = None
    try:
        state = module.state_dict(keep_vars=True)
    except Exception:
        pass
    if not state:
        return

    for tensor in state.values():
        try:
            if not hasattr(tensor, 'pin_memory'):
                continue
            if tensor.is_cuda:
                continue
            if getattr(tensor, 'is_pinned', None) and tensor.is_pinned():
                continue
            pinned = tensor.pin_memory()
            tensor.data = pinned.data
        except Exception:
            continue


def _adopt_slam_cpu_prewarm(owner, weights: str) -> None:
    thread = getattr(owner, '_slam_cpu_prewarm_thread', None)
    if thread is not None and thread.is_alive():
        thread.join()
    net = getattr(owner, '_slam_cpu_net', None)
    if net is None:
        return
    try:
        from ego_pipeline.backends import megasam as _megasam
        if weights not in _megasam._DROID_NET_CACHE:
            _megasam._DROID_NET_CPU_CACHE[weights] = net
        owner._slam_cpu_net = None
    except Exception:
        pass


def ensure_slam_model_loaded(owner) -> None:
    """Load MegaSAM on this actor's current GPU after the previous module exited."""
    if getattr(owner, '_slam_ready', False):
        return

    gpu = getattr(owner, '_gpu', os.environ.get('CUDA_VISIBLE_DEVICES', '?'))
    configure_memory_cap('SLAM', gpu)
    print(f'[pipeline]  {gpu}.')
    weights = _slam_weights_path()
    cwd = os.getcwd()
    from ego_pipeline.backends.megasam import _get_droid_net
    try:
        os.chdir(cwd)
    except Exception:
        pass

    _adopt_slam_cpu_prewarm(owner, weights)
    _get_droid_net(weights)
    owner._slam_ready = True
    print(f'[pipeline]  {gpu}.')


def offload_slam_model_cache(owner) -> None:
    """Move the cached DroidNet off GPU but keep it in process CPU cache."""
    gpu = getattr(owner, '_gpu', os.environ.get('CUDA_VISIBLE_DEVICES', '?'))
    try:
        from ego_pipeline.backends.megasam import _DROID_NET_CACHE, _DROID_NET_CPU_CACHE
        for weights, net in list(_DROID_NET_CACHE.items()):
            move_to_cpu(net)
            _DROID_NET_CPU_CACHE[weights] = net
        _DROID_NET_CACHE.clear()
    except Exception:
        pass
    owner._slam_ready = False
    release_cuda_cache('SLAM', gpu, log=True)


def cleanup_slam_model_cache(owner) -> None:
    """Release the process-local MegaSAM cache before this actor exits."""
    gpu = getattr(owner, '_gpu', os.environ.get('CUDA_VISIBLE_DEVICES', '?'))
    save_error: Exception | None = None
    megasam_mod = sys.modules.get('ego_pipeline.backends.megasam')
    if megasam_mod is not None:
        try:
            megasam_mod.wait_for_pending_saves()
        except Exception as exc:
            save_error = exc
            print(f'[pipeline]  {gpu}; {exc}.')
        try:
            droid_cache = getattr(megasam_mod, '_DROID_NET_CACHE', {})
            droid_cpu_cache = getattr(megasam_mod, '_DROID_NET_CPU_CACHE', {})
            for net in list(droid_cache.values()):
                move_to_cpu(net)
            droid_cache.clear()
            droid_cpu_cache.clear()
        except Exception:
            pass
    try:
        owner._slam_cpu_net = None
    except Exception:
        pass
    owner._slam_ready = False
    release_cuda_cache('SLAM', gpu, log=True)
    if save_error is not None:
        raise save_error


def run_slam_item_once(
    owner,
    item: dict,
    out_q: RayQueue,
    ba_steps1: int,
    ba_steps2: int,
    ba_steps3: int,
    finalize_ex=None,
    stage_q: RayQueue | None = None,
    delete_temp: bool = True,
) -> object | None:
    """Run exactly one SLAM item, for reversible idle stealing workers."""
    from ego_pipeline.backends.hawor_no_filler import finalize_cam2world
    from steps.gpu.megasam import prefetch_megasam_alignment, run_megasam_step

    gpu = getattr(owner, '_gpu', os.environ.get('CUDA_VISIBLE_DEVICES', '?'))
    gpu_id = getattr(owner, '_gpu_id', -1)

    def _emit_error() -> None:
        events = list(item.get('events') or [])
        emit_stage_event(stage_q, item, 'megasam', 'failed', error=item.get('error'))
        out_q.put({**item, 'cam_c2w': None, 'K': None, 'slam_hw': None, 'pred_result': None,
                   'slam_events': events})

    def _finalize_and_emit(cam_c2w, K, slam_hw, elapsed, slam_err, slam_events) -> None:
        scene = item['scene']
        err = slam_err or item.get('error')
        pred_result = None
        if not err:
            try:
                work_dir = Path(item['work_dir'])
                save_path = str(work_dir / 'hawor' / 'pred_result.pkl')
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                t_fin = time.perf_counter()
                t_fin_wall = time.time()
                pred_result = finalize_cam2world(
                    item['hawor_video'], item['focal'], cam_c2w,
                    item['detect_meta'], item['fc_map'], item['cs_map'],
                    save_path, item.get('image_dir'),
                )
                slam_events.append({
                    'gpu_id': gpu_id, 'task': 'SLAM-Final',
                    't_start': t_fin_wall, 't_end': time.time(),
                })
                print(f'[SLAM-Idle GPU{gpu}] {scene} finalize  ({time.perf_counter()-t_fin:.1f}s)')
            except Exception:
                err = traceback.format_exc()
        if not err and delete_temp:
            _cleanup_clip_moge_depth(item.get('moge_dir'), scene)
        out_q.put({
            **item,
            'cam_c2w': cam_c2w, 'K': K, 'slam_hw': slam_hw, 't_slam': elapsed,
            'pred_result': pred_result,
            'slam_events': slam_events,
            'error': err,
        })
        emit_stage_event(
            stage_q, item, 'megasam',
            'failed' if err else 'done',
            error=err,
            extra={'path': str(Path(item['work_dir']) / 'megasam'
                               / f'{Path(item["scene"]).name}.npz')},
        )

    if item.get('error'):
        _emit_error()
        return None

    scene = item['scene']
    slam_events: list = list(item.get('events') or [])
    cam_c2w = None
    K = None
    slam_hw = None
    elapsed = 0.0
    slam_err = None
    try:
        _wait_frames(item)
        precomputed = None
        try:
            align_s = time.time()
            precomputed = prefetch_megasam_alignment(
                item['image_dir'], scene,
                str(Path(item['work_dir']) / 'megasam'),
                item['focal'], item.get('moge_dir'),
            )
            slam_events.append({
                'gpu_id': gpu_id, 'task': 'SLAM-Align',
                't_start': align_s, 't_end': time.time(),
            })
        except Exception:
            # run_megasam_step can recompute alignment; keep the idle worker useful.
            precomputed = None

        ensure_slam_model_loaded(owner)
        parent_s = time.time()
        gpu_sub_events: list = []
        cam_c2w, K, slam_hw, elapsed = run_megasam_step(
            item['image_dir'], scene,
            str(Path(item['work_dir']) / 'megasam'),
            item['focal'], item['moge_dir'],
            ba_steps1, ba_steps2, ba_steps3,
            _precomputed=precomputed,
            _sub_events=gpu_sub_events,
        )
        parent_e = time.time()
        slam_events.append({'gpu_id': gpu_id, 'task': 'SLAM-Idle',
                            't_start': parent_s, 't_end': parent_e})
        for ev in gpu_sub_events:
            ev['gpu_id'] = gpu_id
            slam_events.append(ev)
        print(f'[SLAM-Idle GPU{gpu}] {scene}  ({elapsed:.1f}s)')
    except Exception:
        slam_err = traceback.format_exc()
    finally:
        release_cuda_cache('SLAM-Idle', gpu)

    if finalize_ex is None:
        _finalize_and_emit(cam_c2w, K, slam_hw, elapsed, slam_err, slam_events)
        return None
    return finalize_ex.submit(_finalize_and_emit, cam_c2w, K, slam_hw, elapsed, slam_err, slam_events)


def _queue_get_nowait(q) -> tuple[bool, object]:
    try:
        return True, q.get(block=False)
    except queue.Empty:
        return False, None
    except Exception as exc:
        if 'Empty' in exc.__class__.__name__:
            return False, None
        raise


def _shutdown_img_executor(model) -> None:
    executor = getattr(model, '_img_executor', None)
    if executor is None:
        return
    try:
        executor.shutdown(wait=False)
    except Exception:
        pass
    try:
        model._img_executor = None
    except Exception:
        pass


def run_slam_loop(
    owner,
    in_q: RayQueue,
    out_q: RayQueue,
    moge_q: RayQueue,
    join_store,
    ba_steps1: int,
    ba_steps2: int,
    ba_steps3: int,
    start_delay: float = 0.0,
    steal_moge: bool = True,
    moge_model: str | None = None,
    stage_q: RayQueue | None = None,
    first_item=_NO_FIRST_ITEM,
    delete_temp: bool = True,
) -> None:
    """Run the SLAM consumer loop inside any GPU actor process."""
    import concurrent.futures
    import threading

    from ego_pipeline.backends.hawor_no_filler import finalize_cam2world
    from steps.gpu.megasam import prefetch_megasam_alignment, run_megasam_step

    gpu = getattr(owner, '_gpu', os.environ.get('CUDA_VISIBLE_DEVICES', '?'))
    gpu_id = getattr(owner, '_gpu_id', -1)

    if start_delay > 0:
        print(f'[pipeline]  {gpu}; {start_delay:.1f}.')
        time.sleep(start_delay)
        print(f'[pipeline]  {gpu}.')

    cpu_ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    finalize_ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    finalize_futs: list = []
    trimmer = MallocTrimmer(f'SLAM GPU{gpu}')
    probe = TraceMallocProbe(f'SLAM GPU{gpu}')

    def _reap_finalize_futs(wait: bool = False) -> None:




        pending = []
        for fut in finalize_futs:
            if wait or fut.done():
                try:
                    fut.result()
                except Exception as exc:
                    print(f'[pipeline]  {gpu}; {exc}.')
            else:
                pending.append(fut)
        finalize_futs[:] = pending

    moge_runtime = {'model': None, 'device': None, 'disabled': not steal_moge}

    def _cleanup_stolen_moge(log: bool = False) -> None:
        if moge_runtime.get('model') is None:
            return
        move_to_cpu(moge_runtime.get('model'))
        moge_runtime['model'] = None
        moge_runtime['device'] = None
        release_cuda_cache('MoGe-Steal', gpu, log=log)

    def _prepare(item):
        if (item.get('resume_stages') or {}).get('megasam'):
            return None, time.time(), time.time()
        _wait_frames(item)
        t_s = time.time()
        result = prefetch_megasam_alignment(
            item['image_dir'], item['scene'],
            str(Path(item['work_dir']) / 'megasam'),
            item['focal'], item.get('moge_dir'),
        )
        return result, t_s, time.time()

    def _finalize_bg(item, cam_c2w, K, slam_hw, elapsed, slam_err, slam_events):
        scene = item['scene']
        err = slam_err or item.get('error')
        pred_result = None
        if not err:
            try:
                work_dir = Path(item['work_dir'])
                save_path = str(work_dir / 'hawor' / 'pred_result.pkl')
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                t_fin = time.perf_counter()
                t_fin_wall = time.time()
                pred_result = finalize_cam2world(
                    item['hawor_video'], item['focal'], cam_c2w,
                    item['detect_meta'], item['fc_map'], item['cs_map'],
                    save_path, item.get('image_dir'),
                )
                slam_events.append({
                    'gpu_id': gpu_id, 'task': 'SLAM-Final',
                    't_start': t_fin_wall, 't_end': time.time(),
                })
                print(f'[SLAM GPU{gpu}] {scene} finalize  ({time.perf_counter()-t_fin:.1f}s)')
            except Exception:
                err = traceback.format_exc()
        if not err and delete_temp:
            _cleanup_clip_moge_depth(item.get('moge_dir'), scene)
        out_q.put({
            **item,
            'cam_c2w': cam_c2w, 'K': K, 'slam_hw': slam_hw, 't_slam': elapsed,
            'pred_result': pred_result,
            'slam_events': slam_events,
            'error': err,
        })
        emit_stage_event(
            stage_q, item, 'megasam',
            'failed' if err else 'done',
            error=err,
            extra={'path': str(Path(item['work_dir']) / 'megasam'
                               / f'{Path(item["scene"]).name}.npz')},
        )

    def _emit_error_bg(item):
        events = list(item.get('events') or [])
        emit_stage_event(stage_q, item, 'megasam', 'failed', error=item.get('error'))
        out_q.put({**item, 'cam_c2w': None, 'K': None, 'slam_hw': None, 'pred_result': None,
                   'slam_events': events})

    def _queue_get_nowait(q) -> tuple[bool, object]:
        try:
            return True, q.get(block=False)
        except queue.Empty:
            return False, None
        except Exception as exc:
            if 'Empty' in exc.__class__.__name__:
                return False, None
            raise

    def _run_stolen_moge_once() -> bool:
        if moge_runtime['disabled']:
            return False
        got, item = _queue_get_nowait(moge_q)
        if not got:
            return False
        if item is None:
            moge_runtime['disabled'] = True
            moge_q.put(None)
            return False
        if item.get('error'):
            ray.get(join_store.mark_moge.remote(
                {**item, 'moge_dir': None}, in_q, stage_q))
            return True

        try:
            from steps.gpu.moge import load_moge_model, run_moge_step
            if moge_runtime['model'] is None:
                if getattr(owner, '_slam_ready', False):
                    print(f'[pipeline]  {gpu}.')
                    offload_slam_model_cache(owner)
                configure_memory_cap('MoGe-Steal', gpu)
                print(f'[pipeline]  {gpu}.')
                model, device = load_moge_model(moge_model)
                moge_runtime['model'] = model
                moge_runtime['device'] = device
            _wait_frames(item)
            t_s = time.time()
            sub_events: list = []
            moge_dir, elapsed = run_moge_step(
                item['image_dir'], item['scene'], item['work_dir'],
                None, item['focal'],
                model=moge_runtime['model'], device=moge_runtime['device'],
                _sub_events=sub_events,
            )
            t_e = time.time()
            events = list(item.get('events') or [])
            events.append({'gpu_id': gpu_id, 'task': 'MoGe-Steal',
                           't_start': t_s, 't_end': t_e})
            for ev in sub_events:
                ev['gpu_id'] = gpu_id
                events.append(ev)
            print(f'[SLAM GPU{gpu}] steal MoGe {item["scene"]}  ({elapsed:.1f}s)')
            ray.get(join_store.mark_moge.remote(
                {**item, 'moge_dir': moge_dir, 't_moge': elapsed,
                 'events': events, 'error': None},
                in_q, stage_q,
            ))
            return True
        except Exception:
            err = traceback.format_exc()
            print(f'[pipeline]  {gpu}; {err.splitlines()[-1]}.')
            moge_runtime['disabled'] = True
            _cleanup_stolen_moge(log=True)
            moge_q.put(item)
            return False

    # Keep only a tiny local lookahead. If early SLAM workers drain the
    # shared q_slam into private queues, late tail workers cannot help.
    fut_q: 'queue.Queue[tuple]' = queue.Queue(maxsize=2)

    def _producer():
        forced_item = first_item
        if forced_item is not _NO_FIRST_ITEM:
            fut = (
                cpu_ex.submit(_prepare, forced_item)
                if forced_item is not None and not forced_item.get('error') else None
            )
            fut_q.put((forced_item, fut))
            if forced_item is None:
                return

        while True:
            got, item = _queue_get_nowait(in_q)
            if not got:
                if _run_stolen_moge_once():
                    continue
                time.sleep(0.05)
                continue
            fut = (
                cpu_ex.submit(_prepare, item)
                if item is not None and not item.get('error') else None
            )
            fut_q.put((item, fut))
            if item is None:
                return

    pf_thread = threading.Thread(target=_producer, daemon=True,
                                 name=f'slam-prefetch-gpu{gpu}')
    pf_thread.start()

    try:
        item, fut = fut_q.get()
        if item is None:
            return

        while True:
            scene = item['scene']
            if not item.get('error'):
                slam_events: list = list(item.get('events') or [])
                precomputed = None
                if fut is not None:
                    try:
                        precomputed, align_s, align_e = fut.result()
                        slam_events.append({
                            'gpu_id': gpu_id, 'task': 'SLAM-Align',
                            't_start': align_s, 't_end': align_e,
                        })
                    except Exception:
                        pass
                try:
                    if moge_runtime.get('model') is not None:
                        print(f'[pipeline]  {gpu}.')
                        _cleanup_stolen_moge(log=True)
                    ensure_slam_model_loaded(owner)
                    gpu_sub_events: list = []
                    cam_c2w, K, slam_hw, elapsed = run_megasam_step(
                        item['image_dir'], scene,
                        str(Path(item['work_dir']) / 'megasam'),
                        item['focal'], item['moge_dir'],
                        ba_steps1, ba_steps2, ba_steps3,
                        _precomputed=precomputed,
                        _sub_events=gpu_sub_events,
                    )
                    for ev in gpu_sub_events:
                        ev['gpu_id'] = gpu_id
                        slam_events.append(ev)
                    print(f'[SLAM GPU{gpu}] {scene}  ({elapsed:.1f}s)')
                    slam_err = None
                except Exception:
                    cam_c2w = None
                    K = None
                    slam_hw = None
                    elapsed = 0.0
                    slam_err = traceback.format_exc()
                finally:
                    release_cuda_cache('SLAM', gpu)
                finalize_futs.append(
                    finalize_ex.submit(
                        _finalize_bg, item, cam_c2w, K, slam_hw, elapsed,
                        slam_err, slam_events,
                    )
                )
            else:
                finalize_futs.append(finalize_ex.submit(_emit_error_bg, item))


            _reap_finalize_futs(wait=False)
            trimmer.tick()
            probe.tick()

            next_item, next_fut = fut_q.get()
            if next_item is None:
                break
            item, fut = next_item, next_fut


        _reap_finalize_futs(wait=True)
    finally:
        _cleanup_stolen_moge(log=False)
        cpu_ex.shutdown(wait=False)
        finalize_ex.shutdown(wait=False)
        pf_thread.join(timeout=1.0)
        cleanup_slam_model_cache(owner)
        print(f'[pipeline]  {gpu}.')


def run_tail_slam_after_release_loop(
    owner,
    in_q: RayQueue,
    out_q: RayQueue,
    moge_q: RayQueue,
    join_store,
    ba_steps1: int,
    ba_steps2: int,
    ba_steps3: int,
    moge_model: str | None = None,
    stage_q: RayQueue | None = None,
    delete_temp: bool = True,
) -> None:
    """Before loading SLAM, fill the tail gap with MoGe work if q_slam is empty."""
    from steps.gpu.moge import load_moge_model, run_moge_step

    gpu = getattr(owner, '_gpu', os.environ.get('CUDA_VISIBLE_DEVICES', '?'))
    gpu_id = getattr(owner, '_gpu_id', -1)
    moge_runtime = {'model': None, 'device': None, 'disabled': False}

    def _cleanup_moge_helper() -> None:
        move_to_cpu(moge_runtime.get('model'))
        moge_runtime['model'] = None
        moge_runtime['device'] = None
        release_cuda_cache('MoGe-Tail', gpu, log=True)

    def _run_moge_once() -> bool:
        if moge_runtime['disabled']:
            return False
        got, item = _queue_get_nowait(moge_q)
        if not got:
            return False
        if item is None:
            moge_runtime['disabled'] = True
            moge_q.put(None)
            return False
        if item.get('error'):
            ray.get(join_store.mark_moge.remote(
                {**item, 'moge_dir': None}, in_q, stage_q))
            return True

        if moge_runtime['model'] is None:
            configure_memory_cap('MoGe-Tail', gpu)
            owner_cpu = getattr(owner, '_moge_idle_model_cpu', None)
            if owner_cpu is not None:

                print(f'[pipeline]  {gpu}.')
                model = owner_cpu.to('cuda', non_blocking=True)
                try:
                    owner._moge_idle_model_cpu = None
                except Exception:
                    pass
                import torch as _torch
                device = _torch.device('cuda')
            else:
                cpu_net = adopt_moge_cpu_prewarm(owner)
                if cpu_net is not None:
                    print(f'[pipeline]  {gpu}.')
                    model = cpu_net.to('cuda', non_blocking=True)
                    import torch as _torch
                    device = _torch.device('cuda')
                else:
                    print(f'[pipeline]  {gpu}.')
                    model, device = load_moge_model(moge_model)
            moge_runtime['model'] = model
            moge_runtime['device'] = device

        try:
            _wait_frames(item)
            t_s = time.time()
            sub_events: list = []
            moge_dir, elapsed = run_moge_step(
                item['image_dir'], item['scene'], item['work_dir'],
                None, item['focal'],
                model=moge_runtime['model'], device=moge_runtime['device'],
                _sub_events=sub_events,
            )
            t_e = time.time()
            events = list(item.get('events') or [])
            events.append({'gpu_id': gpu_id, 'task': 'MoGe-Tail',
                           't_start': t_s, 't_end': t_e})
            for ev in sub_events:
                ev['gpu_id'] = gpu_id
                events.append(ev)
            print(f'[MoGe-Tail GPU{gpu}] {item["scene"]}  ({elapsed:.1f}s)')
            ray.get(join_store.mark_moge.remote(
                {**item, 'moge_dir': moge_dir, 't_moge': elapsed,
                 'events': events, 'error': None},
                in_q, stage_q,
            ))
            return True
        except Exception:
            err = traceback.format_exc()
            ray.get(join_store.mark_moge.remote(
                {**item, 'moge_dir': None, 'error': err}, in_q, stage_q))
            return True
        finally:
            release_cuda_cache('MoGe-Tail', gpu)

    first_item = _NO_FIRST_ITEM
    try:
        while True:
            got, item = _queue_get_nowait(in_q)
            if got:
                first_item = item
                break
            if _run_moge_once():
                continue
            time.sleep(0.05)

        if first_item is None:
            return

        if moge_runtime['model'] is not None:
            print(f'[pipeline]  {gpu}.')
        _cleanup_moge_helper()
        run_slam_loop(
            owner, in_q, out_q, moge_q, join_store,
            ba_steps1, ba_steps2, ba_steps3,
            0.0, False, moge_model, stage_q,
            first_item=first_item,
            delete_temp=delete_temp,
        )
    finally:
        _cleanup_moge_helper()
        cleanup_slam_model_cache(owner)
