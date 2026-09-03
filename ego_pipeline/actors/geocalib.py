"""Internal helper."""

from __future__ import annotations

import os
import queue as _queue
import time
import traceback
from collections import deque

import ray
from ray.util.queue import Queue as RayQueue

from ._cuda import configure_memory_cap, move_to_cpu, release_cuda_cache
from ._utils import _wait_frames, MallocTrimmer
from ._utils import _build_clip_items, emit_stage_event, _fanout
from .slam_runtime import (
    adopt_moge_cpu_prewarm,
    offload_slam_model_cache,
    pin_module_,
    run_slam_item_once,
    run_tail_slam_after_release_loop,
    start_moge_cpu_prewarm,
    start_slam_cpu_prewarm,
)


_NO_GEO_ITEM = object()


def _queue_get_nowait(q) -> tuple[bool, object]:
    try:
        return True, q.get(block=False)
    except _queue.Empty:
        return False, None
    except Exception as exc:
        if 'Empty' in exc.__class__.__name__:
            return False, None
        raise


@ray.remote
class GeoCalibWorker:
    def __init__(self):
        import torch
        from pathlib import Path as _Path
        self._gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
        try:
            self._gpu_id = int(str(self._gpu).split(',')[0])
        except (ValueError, AttributeError):
            self._gpu_id = -1
        self._slam_ready = False
        self._moge_idle_model = None
        self._moge_idle_device = None


        self._model_cpu = None
        self._moge_idle_model_cpu = None
        configure_memory_cap('GeoCalib', self._gpu)
        print(f'[pipeline]  {self._gpu}; {torch.cuda.is_available()}.'
              f'  device_count={torch.cuda.device_count()}')


        import sys as _sys
        _vitra_dir = str(_Path(__file__).resolve().parents[2])
        _geocalib_root = str(_Path(_vitra_dir) / 'third_party' / 'GeoCalib')
        for _p in (_vitra_dir, _geocalib_root):
            if _p not in _sys.path:
                _sys.path.insert(0, _p)

        self._vitra_dir = _vitra_dir
        self._geocalib_weights = str(_Path(_vitra_dir) / 'model' / 'geocalib' / 'pinhole.tar')
        self._model = None
        self._ensure_geo_model_loaded()
        start_slam_cpu_prewarm(self)

    def _ensure_geo_model_loaded(self) -> None:
        if self._model is not None:
            return

        self._offload_moge_idle(log=True)
        if getattr(self, '_slam_ready', False):
            print(f'[pipeline]  {self._gpu}.')
            offload_slam_model_cache(self)
        import torch
        configure_memory_cap('GeoCalib', self._gpu)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t0 = time.perf_counter()
        if self._model_cpu is not None:
            self._model = self._model_cpu.to(device, non_blocking=True)
            self._model_cpu = None
            print(f'[pipeline]  {self._gpu}.'
                  f'{time.perf_counter()-t0:.2f}s)')
        else:
            from geocalib import GeoCalib
            self._model = GeoCalib(weights=self._geocalib_weights).to(device)
            print(f'[pipeline]  {self._gpu}.'
                  f'({time.perf_counter()-t0:.1f}s)')

    def _offload_geo_model(self, *, log: bool = False) -> None:
        """Internal helper."""
        if self._model is None:
            return
        move_to_cpu(self._model)
        pin_module_(self._model)
        self._model_cpu = self._model
        self._model = None
        release_cuda_cache('GeoCalib', self._gpu, log=log)

    def _cleanup_geo_model(self, *, log: bool = False) -> None:
        """Internal helper."""
        had_model = self._model is not None or self._model_cpu is not None
        if self._model is not None:
            move_to_cpu(self._model)
            self._model = None
        if self._model_cpu is not None:
            self._model_cpu = None
        release_cuda_cache('GeoCalib', self._gpu, log=log and had_model)

    def _offload_moge_idle(self, *, log: bool = False) -> None:
        """Internal helper."""
        if self._moge_idle_model is None:
            return
        move_to_cpu(self._moge_idle_model)
        pin_module_(self._moge_idle_model)
        self._moge_idle_model_cpu = self._moge_idle_model
        self._moge_idle_model = None
        self._moge_idle_device = None
        release_cuda_cache('MoGe-GeoIdle', self._gpu, log=log)

    def _cleanup_moge_idle(self, *, log: bool = False) -> None:
        """Internal helper."""
        had_model = (self._moge_idle_model is not None
                     or self._moge_idle_model_cpu is not None)
        if self._moge_idle_model is not None:
            move_to_cpu(self._moge_idle_model)
            self._moge_idle_model = None
            self._moge_idle_device = None
        if self._moge_idle_model_cpu is not None:
            self._moge_idle_model_cpu = None
        release_cuda_cache('MoGe-GeoIdle', self._gpu, log=log and had_model)

    def _ensure_moge_idle_loaded(self, moge_model: str | None = None) -> None:
        if self._moge_idle_model is not None:
            return
        if self._model is not None:
            print(f'[pipeline]  {self._gpu}.')
            self._offload_geo_model(log=True)
        if getattr(self, '_slam_ready', False):
            print(f'[pipeline]  {self._gpu}.')
            offload_slam_model_cache(self)
        import torch
        configure_memory_cap('MoGe-GeoIdle', self._gpu)
        t0 = time.perf_counter()
        if self._moge_idle_model_cpu is not None:
            self._moge_idle_model = self._moge_idle_model_cpu.to(
                'cuda', non_blocking=True)
            self._moge_idle_model_cpu = None
            self._moge_idle_device = torch.device('cuda')
            print(f'[pipeline]  {self._gpu}.'
                  f'{time.perf_counter()-t0:.2f}s)')
            return
        cpu_net = adopt_moge_cpu_prewarm(self)
        if cpu_net is not None:
            self._moge_idle_model = cpu_net.to('cuda', non_blocking=True)
            self._moge_idle_device = torch.device('cuda')
            print(f'[pipeline]  {self._gpu}.'
                  f'{time.perf_counter()-t0:.2f}s)')
            return
        from steps.gpu.moge import load_moge_model
        print(f'[pipeline]  {self._gpu}.')
        self._moge_idle_model, self._moge_idle_device = load_moge_model(moge_model)

    def run_loop(
        self,
        in_q:        RayQueue,
        out_moge_q:  RayQueue,
        out_hawor_q: RayQueue,
        geocalib_interval: float,
        slam_q:      RayQueue | None = None,
        post_q:      RayQueue | None = None,
        join_store = None,
        stage_q:     RayQueue | None = None,
        ba_steps1: int = 10,
        ba_steps2: int = 20,
        ba_steps3: int = 15,
        moge_model: str | None = None,
        idle_steal: bool = True,
        delete_temp: bool = True,
    ) -> None:
        import concurrent.futures
        from steps.gpu.geocalib import read_frames, run_inference
        from steps.gpu.moge import run_moge_step



        start_moge_cpu_prewarm(self, moge_model)

        geo_ex = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        finalize_ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        finalize_futs: list = []
        trimmer = MallocTrimmer(f'GeoCalib GPU{self._gpu}')
        geo_pending = deque()
        geo_input_closed = False
        moge_input_closed = False
        slam_input_closed = False

        def _read_geo_frames(pre: dict):
            if pre.get('error'):
                return pre, None
            resume = pre.get('resume_clip_stages') or {}
            if resume and all(v.get('geo') for v in resume.values()):
                return pre, []
            try:

                return pre, read_frames(pre['video'], geocalib_interval)
            except Exception:
                return {**pre, 'error': traceback.format_exc()}, None

        def _fill_geo_prefetch() -> None:
            nonlocal geo_input_closed
            while not geo_input_closed and len(geo_pending) < 2:
                got, pre = _queue_get_nowait(in_q)
                if not got:
                    break
                if pre is None:
                    geo_input_closed = True
                    geo_pending.append((None, None))
                    break
                geo_pending.append((pre, geo_ex.submit(_read_geo_frames, pre)))

        def _next_geo_item():
            _fill_geo_prefetch()
            if not geo_pending:
                return _NO_GEO_ITEM
            pre, fut = geo_pending.popleft()
            if pre is None:
                return None
            result = fut.result()
            _fill_geo_prefetch()
            return result

        def _geo_work_waiting() -> bool:
            _fill_geo_prefetch()
            return bool(geo_pending)

        def _run_geo_once(pre: dict, frames) -> None:
            scene = pre['scene']

            geo_event = None
            if pre.get('error') or frames is None:
                items = _build_clip_items(pre, None, 0.0,
                                          pre.get('error') or '[pipeline]')
            else:
                try:
                    self._ensure_geo_model_loaded()
                    t_s = time.time()
                    focal, calib, elapsed = run_inference(
                        frames, pre['work_dir'], self._model)
                    t_e = time.time()
                    n_clips = len(pre['clips']) if pre.get('is_long') else 1
                    print(f'[GeoCalib GPU{self._gpu}] {scene}  focal={focal:.1f}px  '
                          f'[pipeline]  {elapsed:.1f}; {n_clips}.')
                    items = _build_clip_items(pre, focal, elapsed, None)
                    geo_event = {'gpu_id': self._gpu_id, 'task': 'GeoCalib',
                                 't_start': t_s, 't_end': t_e}
                except Exception:
                    items = _build_clip_items(pre, None, 0.0, traceback.format_exc())
                finally:
                    del frames
                    release_cuda_cache('GeoCalib', self._gpu)



            if items and geo_event is not None:
                items[0].setdefault('events', []).append(geo_event)

            for it in items:
                emit_stage_event(
                    stage_q, it, 'geo',
                    'failed' if it.get('error') else 'done',
                    error=it.get('error'),
                )
                _fanout(out_moge_q, out_hawor_q, it)

        def _reap_finalize_futs(wait: bool = False) -> None:
            pending = []
            for fut in finalize_futs:
                if wait or fut.done():
                    try:
                        fut.result()
                    except Exception as exc:
                        print(f'[pipeline]  {self._gpu}; {exc}.')
                else:
                    pending.append(fut)
            finalize_futs[:] = pending

        def _run_idle_moge_once() -> bool:
            nonlocal moge_input_closed
            if not idle_steal or out_moge_q is None or join_store is None or slam_q is None:
                return False
            if moge_input_closed:
                return False
            got, item = _queue_get_nowait(out_moge_q)
            if not got:
                return False
            if item is None:
                moge_input_closed = True
                out_moge_q.put(None)
                return False
            if _geo_work_waiting():
                out_moge_q.put(item)
                return True
            if item.get('error'):
                ray.get(join_store.mark_moge.remote(
                    {**item, 'moge_dir': None}, slam_q, stage_q))
                return True

            try:
                self._ensure_moge_idle_loaded(moge_model)
                _wait_frames(item)
                t_s = time.time()
                sub_events: list = []
                moge_dir, elapsed = run_moge_step(
                    item['image_dir'], item['scene'], item['work_dir'],
                    None, item['focal'],
                    model=self._moge_idle_model, device=self._moge_idle_device,
                    _sub_events=sub_events,
                )
                t_e = time.time()
                events = list(item.get('events') or [])
                events.append({'gpu_id': self._gpu_id, 'task': 'MoGe-GeoIdle',
                               't_start': t_s, 't_end': t_e})
                for ev in sub_events:
                    ev['gpu_id'] = self._gpu_id
                    events.append(ev)
                print(f'[MoGe-GeoIdle GPU{self._gpu}] {item["scene"]}  ({elapsed:.1f}s)')
                ray.get(join_store.mark_moge.remote(
                    {**item, 'moge_dir': moge_dir, 't_moge': elapsed,
                     'events': events, 'error': None},
                    slam_q, stage_q,
                ))
                return True
            except Exception:
                err = traceback.format_exc()
                ray.get(join_store.mark_moge.remote(
                    {**item, 'moge_dir': None, 'error': err}, slam_q, stage_q))
                return True
            finally:
                release_cuda_cache('MoGe-GeoIdle', self._gpu)

        def _run_idle_slam_once() -> bool:
            nonlocal slam_input_closed
            if not idle_steal or slam_q is None or post_q is None:
                return False
            if slam_input_closed:
                return False
            got, item = _queue_get_nowait(slam_q)
            if not got:
                return False
            if item is None:
                slam_input_closed = True
                slam_q.put(None)
                return False
            if _geo_work_waiting():
                slam_q.put(item)
                return True
            if self._model is not None:
                print(f'[pipeline]  {self._gpu}.')
                self._offload_geo_model(log=True)
            self._offload_moge_idle(log=True)
            fut = run_slam_item_once(
                self, item, post_q,
                ba_steps1, ba_steps2, ba_steps3,
                finalize_ex=finalize_ex,
                stage_q=stage_q,
                delete_temp=delete_temp,
            )
            if fut is not None:
                finalize_futs.append(fut)
            _reap_finalize_futs(wait=False)
            return True

        try:
            while True:
                got = _next_geo_item()
                if got is None:
                    break
                if got is not _NO_GEO_ITEM:
                    pre, frames = got
                    _run_geo_once(pre, frames)
                    _reap_finalize_futs(wait=False)
                    trimmer.tick()
                    continue




                if _run_idle_slam_once():
                    trimmer.tick()
                    continue
                if _run_idle_moge_once():
                    trimmer.tick()
                    continue
                _reap_finalize_futs(wait=False)
                time.sleep(0.05)
        finally:
            _reap_finalize_futs(wait=True)
            geo_ex.shutdown(wait=False)
            finalize_ex.shutdown(wait=True)
            self._cleanup_moge_idle(log=True)
            if getattr(self, '_slam_ready', False):
                offload_slam_model_cache(self)
            self._cleanup_geo_model(log=True)
        print(f'[pipeline]  {self._gpu}.')

    def run_tail_slam_after_release(
        self,
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
        print(f'[pipeline]  {self._gpu}.')
        run_tail_slam_after_release_loop(
            self, in_q, out_q, moge_q, join_store,
            ba_steps1, ba_steps2, ba_steps3,
            moge_model, stage_q,
            delete_temp=delete_temp,
        )
