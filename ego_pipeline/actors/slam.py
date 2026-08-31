"""Internal helper."""

from __future__ import annotations

import os

import ray
from ray.util.queue import Queue as RayQueue

from .slam_runtime import (
    cleanup_slam_model_cache,
    ensure_slam_model_loaded,
    run_slam_loop,
)


@ray.remote
class SlamWorker:
    def __init__(self):
        self._gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
        try:
            self._gpu_id = int(str(self._gpu).split(',')[0])
        except (ValueError, AttributeError):
            self._gpu_id = -1
        self._slam_ready = False
        ensure_slam_model_loaded(self)

    def _cleanup_slam_model_cache(self) -> None:
        cleanup_slam_model_cache(self)

    def run_loop(
        self,
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
        delete_temp: bool = True,
    ) -> None:
        run_slam_loop(
            self, in_q, out_q, moge_q, join_store,
            ba_steps1, ba_steps2, ba_steps3,
            start_delay, steal_moge, moge_model, stage_q,
            delete_temp=delete_temp,
        )
