"""Internal helper."""

from __future__ import annotations

import os
import queue as _queue
import time
import traceback
from collections import deque
from pathlib import Path

import ray
from ray.util.queue import Queue as RayQueue

from ._cuda import configure_memory_cap, move_to_cpu, release_cuda_cache
from ._utils import _wait_frames, MallocTrimmer, TraceMallocProbe
from .slam_runtime import (
    adopt_moge_cpu_prewarm,
    offload_slam_model_cache,
    pin_module_,
    run_slam_item_once,
    run_tail_slam_after_release_loop,
    start_moge_cpu_prewarm,
    start_slam_cpu_prewarm,
)


_NO_HAWOR_ITEM = object()


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
class HaWorWorker:
    def __init__(self, use_compile: bool = False):
        self._gpu         = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
        try:
            self._gpu_id = int(str(self._gpu).split(',')[0])
        except (ValueError, AttributeError):
            self._gpu_id = -1
        self._slam_ready = False
        self._moge_idle_model = None
        self._moge_idle_device = None


        self._model_cpu = None
        self._yolo_cpu = None
        self._moge_idle_model_cpu = None
        configure_memory_cap('HaWoR', self._gpu)
        self._model       = None
        self._yolo        = None
        self._use_compile = use_compile
        print(f'[pipeline]  {self._gpu}.')
        self._load_model()
        print(f'[pipeline]  {self._gpu}.')
        start_slam_cpu_prewarm(self)

    def _load_model(self):
        import torch
        from concurrent.futures import ThreadPoolExecutor
        if self._model is None:
            t0 = time.perf_counter()
            if self._model_cpu is not None:
                self._model = self._model_cpu.to('cuda', non_blocking=True)
                self._model_cpu = None
                print(f'[pipeline]  {self._gpu}.'
                      f'{time.perf_counter()-t0:.2f}s)')
            else:
                from ego_pipeline.backends.hawor_no_filler import load_hawor_model
                self._model = load_hawor_model(device=torch.device('cuda'),
                                               use_compile=self._use_compile)


            if getattr(self._model, '_img_executor', None) is None:
                self._model._img_executor = ThreadPoolExecutor(max_workers=8)
        if self._yolo is None:
            t0 = time.perf_counter()
            if self._yolo_cpu is not None:
                self._yolo = self._yolo_cpu.to('cuda', non_blocking=True)
                self._yolo_cpu = None
                print(f'[pipeline]  {self._gpu}.'
                      f'{time.perf_counter()-t0:.2f}s)')
            else:
                from ultralytics import YOLO
                from lib.pipeline.tools import _resolve_detector_weights
                self._yolo = YOLO(_resolve_detector_weights())
                print(f'[pipeline]  {self._gpu}.')

    def _offload_models(self, *, log: bool = False) -> None:
        """Internal helper."""
        had_model = self._model is not None or self._yolo is not None
        if self._model is not None:
            executor = getattr(self._model, '_img_executor', None)
            if executor is not None:
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass
                try:
                    self._model._img_executor = None
                except Exception:
                    pass
            move_to_cpu(self._model)
            pin_module_(self._model)
            self._model_cpu = self._model
            self._model = None
        if self._yolo is not None:
            move_to_cpu(self._yolo)
            pin_module_(self._yolo)
            self._yolo_cpu = self._yolo
            self._yolo = None
        release_cuda_cache('HaWoR', self._gpu, log=log and had_model)

    def _cleanup_models(self) -> None:
        """Internal helper."""
        had_model = (self._model is not None or self._yolo is not None
                     or self._model_cpu is not None or self._yolo_cpu is not None)
        if self._model is not None:
            executor = getattr(self._model, '_img_executor', None)
            if executor is not None:
                try:
                    executor.shutdown(wait=False)
                except Exception:
                    pass
                try:
                    self._model._img_executor = None
                except Exception:
                    pass
            move_to_cpu(self._model)
            self._model = None
        if self._model_cpu is not None:
            self._model_cpu = None
        if self._yolo is not None:
            move_to_cpu(self._yolo)
            self._yolo = None
        if self._yolo_cpu is not None:
            self._yolo_cpu = None
        release_cuda_cache('HaWoR', self._gpu, log=had_model)

    def _offload_moge_idle(self, *, log: bool = False) -> None:
        """Internal helper."""
        if self._moge_idle_model is None:
            return
        move_to_cpu(self._moge_idle_model)
        pin_module_(self._moge_idle_model)
        self._moge_idle_model_cpu = self._moge_idle_model
        self._moge_idle_model = None
        self._moge_idle_device = None
        release_cuda_cache('MoGe-HaWoRIdle', self._gpu, log=log)

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
        release_cuda_cache('MoGe-HaWoRIdle', self._gpu, log=log and had_model)

    def _ensure_hawor_model_loaded(self) -> None:
        if self._model is not None and self._yolo is not None:
            return

        self._offload_moge_idle(log=True)
        if getattr(self, '_slam_ready', False):
            print(f'[pipeline]  {self._gpu}.')
            offload_slam_model_cache(self)
        configure_memory_cap('HaWoR', self._gpu)
        print(f'[pipeline]  {self._gpu}.')
        self._load_model()

    def _ensure_moge_idle_loaded(self, moge_model: str | None = None) -> None:
        if self._moge_idle_model is not None:
            return
        if self._model is not None or self._yolo is not None:
            print(f'[pipeline]  {self._gpu}.')
            self._offload_models(log=True)
        if getattr(self, '_slam_ready', False):
            print(f'[pipeline]  {self._gpu}.')
            offload_slam_model_cache(self)
        import torch
        configure_memory_cap('MoGe-HaWoRIdle', self._gpu)
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
        in_q:  RayQueue,
        join_store,
        slam_q: RayQueue,
        moge_q: RayQueue | None = None,
        post_q: RayQueue | None = None,
        stage_q: RayQueue | None = None,
        ba_steps1: int = 10,
        ba_steps2: int = 20,
        ba_steps3: int = 15,
        moge_model: str | None = None,
        idle_steal: bool = True,
        delete_temp: bool = True,
    ) -> None:
        import concurrent.futures
        from steps.cpu.file_ops import copy_video_for_hawor
        from ego_pipeline.backends.hawor_no_filler import (
            run_stage1_detect, run_stage1_track, run_stage2_from_meta,
        )
        from steps.gpu.moge import run_moge_step



        start_moge_cpu_prewarm(self, moge_model)

        self._ensure_hawor_model_loaded()










        io_ex    = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        cpu_ex   = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        track_ex = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        def _prepare(item, io_fut):
            """Internal helper."""
            if io_fut is not None:
                io_fut.result()
            t_s1_s = time.time()
            hawor_dir   = Path(item['work_dir']) / 'hawor'
            hawor_video = copy_video_for_hawor(item['video'], hawor_dir)
            detect_result = run_stage1_detect(
                hawor_video, item['focal'], item.get('image_dir'),
                hand_det_model=self._yolo,
            )
            t_yolo_end = time.time()
            track_fut  = track_ex.submit(run_stage1_track, detect_result)
            return hawor_video, track_fut, t_s1_s, t_yolo_end

        def _submit_prep(itm):
            """Internal helper."""
            if itm is None or itm.get('error'):
                return None, None
            if (itm.get('resume_stages') or {}).get('hawor'):
                io_fut = None
            else:
                io_fut = io_ex.submit(_wait_frames, itm)
            fut    = cpu_ex.submit(_prepare, itm, io_fut)
            return io_fut, fut

        pending_hawor = deque()
        hawor_input_closed = False
        moge_input_closed = False
        slam_input_closed = False
        finalize_ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        finalize_futs: list = []
        trimmer = MallocTrimmer(f'HaWoR GPU{self._gpu}')
        probe = TraceMallocProbe(f'HaWoR GPU{self._gpu}')

        def _fill_hawor_prefetch() -> None:
            nonlocal hawor_input_closed
            while not hawor_input_closed and len(pending_hawor) < 8:
                got, it = _queue_get_nowait(in_q)
                if not got:
                    break
                if it is None:
                    hawor_input_closed = True
                    pending_hawor.append((None, None))
                    break
                if it.get('error'):
                    pending_hawor.append((it, None))
                    continue
                self._ensure_hawor_model_loaded()
                _, fut = _submit_prep(it)
                pending_hawor.append((it, fut))

        def _next_hawor_item():
            _fill_hawor_prefetch()
            if not pending_hawor:
                return _NO_HAWOR_ITEM, None
            it, fut = pending_hawor.popleft()
            _fill_hawor_prefetch()
            if it is None:
                return None, None
            return it, fut

        def _hawor_work_waiting() -> bool:
            _fill_hawor_prefetch()
            return bool(pending_hawor)

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

        def _run_hawor_once(item, fut) -> None:
            scene = item['scene']
            if item.get('error'):
                ray.get(join_store.mark_hawor.remote(
                    {**item, 'hawor_video': None}, slam_q, stage_q))
                return
            try:
                self._ensure_hawor_model_loaded()

                hawor_video, track_fut, t_s1_s, t_yolo_end = fut.result()
                detect_meta, args = track_fut.result()
                t_s1_e = time.time()
                t = time.perf_counter()
                t_s2_s = time.time()
                fc_map, cs_map = run_stage2_from_meta(self._model, detect_meta, args)  # GPU
                release_cuda_cache('HaWoR', self._gpu)
                elapsed = time.perf_counter() - t
                t_s2_e = time.time()
                print(f'[HaWoR GPU{self._gpu}] {scene} Stage1+2  ({elapsed:.1f}s)')
                result = {
                    **item,
                    'hawor_video': hawor_video,
                    'detect_meta': detect_meta, 'fc_map': fc_map, 'cs_map': cs_map,
                    't_hawor12': elapsed, 'error': None,


                    'events': [
                        {'gpu_id': self._gpu_id, 'task': 'HaWoR',
                         't_start': t_s1_s, 't_end': t_s2_e},
                        {'gpu_id': self._gpu_id, 'task': 'HaWoR-S1',
                         't_start': t_s1_s, 't_end': t_s1_e},
                        {'gpu_id': self._gpu_id, 'task': 'HaWoR-S2',
                         't_start': t_s2_s, 't_end': t_s2_e},
                    ],
                }
            except Exception:
                result = {
                    **item,
                    'hawor_video': None, 'detect_meta': None,
                    'fc_map': None, 'cs_map': None,
                    'error': traceback.format_exc(),
                }
            finally:
                release_cuda_cache('HaWoR', self._gpu)
            ray.get(join_store.mark_hawor.remote(result, slam_q, stage_q))

        def _run_idle_moge_once() -> bool:
            nonlocal moge_input_closed
            if not idle_steal or moge_q is None or join_store is None or slam_q is None:
                return False
            if moge_input_closed:
                return False
            got, item = _queue_get_nowait(moge_q)
            if not got:
                return False
            if item is None:
                moge_input_closed = True
                moge_q.put(None)
                return False
            if _hawor_work_waiting():
                moge_q.put(item)
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
                events.append({'gpu_id': self._gpu_id, 'task': 'MoGe-HaWoRIdle',
                               't_start': t_s, 't_end': t_e})
                for ev in sub_events:
                    ev['gpu_id'] = self._gpu_id
                    events.append(ev)
                print(f'[MoGe-HaWoRIdle GPU{self._gpu}] {item["scene"]}  ({elapsed:.1f}s)')
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
                release_cuda_cache('MoGe-HaWoRIdle', self._gpu)

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
            if _hawor_work_waiting():
                slam_q.put(item)
                return True
            if self._model is not None or self._yolo is not None:
                print(f'[pipeline]  {self._gpu}.')
                self._offload_models(log=True)
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

        t_done = time.time()
        try:
            while True:
                item, fut = _next_hawor_item()
                if item is None:
                    break
                if item is not _NO_HAWOR_ITEM:
                    _run_hawor_once(item, fut)
                    _reap_finalize_futs(wait=False)
                    trimmer.tick()
                    probe.tick()
                    continue




                if _run_idle_slam_once():
                    trimmer.tick()
                    probe.tick()
                    continue
                if _run_idle_moge_once():
                    trimmer.tick()
                    probe.tick()
                    continue
                _reap_finalize_futs(wait=False)
                time.sleep(0.05)

            t_done = time.time()
            return t_done
        finally:
            _reap_finalize_futs(wait=True)
            finalize_ex.shutdown(wait=True)
            io_ex.shutdown(wait=False)
            cpu_ex.shutdown(wait=False)
            track_ex.shutdown(wait=True)
            self._cleanup_moge_idle(log=True)
            if getattr(self, '_slam_ready', False):
                offload_slam_model_cache(self)
            self._cleanup_models()
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
