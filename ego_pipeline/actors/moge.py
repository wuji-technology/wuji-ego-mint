"""Internal helper."""

from __future__ import annotations

import os
import time
import traceback

import ray
from ray.util.queue import Queue as RayQueue

from ._cuda import configure_memory_cap, move_to_cpu, release_cuda_cache
from ._utils import _wait_frames
from .slam_runtime import run_tail_slam_after_release_loop, start_slam_cpu_prewarm


@ray.remote
class MoGeWorker:
    def __init__(self, moge_model: str | None = None):
        self._gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
        try:
            self._gpu_id = int(str(self._gpu).split(',')[0])
        except (ValueError, AttributeError):
            self._gpu_id = -1
        self._slam_ready = False
        configure_memory_cap('MoGe', self._gpu)
        from steps.gpu.moge import load_moge_model
        self._model, self._device = load_moge_model(moge_model)
        print(f'[pipeline]  {self._gpu}.')
        start_slam_cpu_prewarm(self)

    def run_loop(
        self,
        in_q:  RayQueue,
        join_store,
        slam_q: RayQueue,
        stage_q: RayQueue | None = None,
    ) -> None:
        from steps.gpu.moge import run_moge_step
        try:
            while True:
                item = in_q.get()
                if item is None:
                    break
                scene = item['scene']
                if item.get('error'):
                    ray.get(join_store.mark_moge.remote(
                        {**item, 'moge_dir': None}, slam_q, stage_q))
                    continue
                try:
                    _wait_frames(item)
                    t_s = time.time()
                    sub_events: list = []
                    moge_dir, elapsed = run_moge_step(
                        item['image_dir'], scene, item['work_dir'],
                        None, item['focal'],
                        model=self._model, device=self._device,
                        _sub_events=sub_events,
                    )
                    t_e = time.time()
                    print(f'[MoGe GPU{self._gpu}] {scene}  ({elapsed:.1f}s)')
                    events = list(item.get('events') or [])

                    events.append({'gpu_id': self._gpu_id, 'task': 'MoGe',
                                   't_start': t_s, 't_end': t_e})

                    for ev in sub_events:
                        ev['gpu_id'] = self._gpu_id
                        events.append(ev)
                    ray.get(join_store.mark_moge.remote(
                        {**item, 'moge_dir': moge_dir, 't_moge': elapsed,
                         'events': events, 'error': None},
                        slam_q, stage_q,
                    ))
                except Exception:
                    ray.get(join_store.mark_moge.remote(
                        {**item, 'moge_dir': None, 'error': traceback.format_exc()},
                        slam_q, stage_q,
                    ))
                finally:
                    release_cuda_cache('MoGe', self._gpu)
        finally:
            move_to_cpu(getattr(self, '_model', None))
            self._model = None
            self._device = None
            release_cuda_cache('MoGe', self._gpu, log=True)
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
