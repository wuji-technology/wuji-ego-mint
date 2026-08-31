"""Actor pool creation and teardown."""

from __future__ import annotations

from dataclasses import dataclass

import ray

from actors import (
    ClipJoinStore,
    GeoCalibWorker,
    HaWorWorker,
    MoGeWorker,
    SlamWorker,
)

from .config import SchedulePlan, SchedulerConfig
from .queues import PipelineQueues


@dataclass
class WorkerPool:
    join_store: object
    geo_ws: list
    moge_ws: list
    haw_ws: list
    tail_slam_ws: list
    slam_ws: list
    geo_refs: list
    moge_refs: list
    haw_refs: list
    tail_slam_refs: list
    slam_refs: list


def start_worker_pool(
    plan: SchedulePlan,
    queues: PipelineQueues,
    config: SchedulerConfig,
) -> WorkerPool:
    join_store = ClipJoinStore.remote()
    slam_ws = None

    if plan.upstream_frac is not None:
        geo_ws = [GeoCalibWorker.options(num_gpus=plan.upstream_frac).remote()
                  for _ in range(plan.n_geo)]
        moge_ws = [MoGeWorker.options(num_gpus=plan.upstream_frac).remote(config.moge_model)
                   for _ in range(plan.n_moge)]
        haw_ws = [HaWorWorker.options(num_gpus=plan.upstream_frac).remote(
            use_compile=config.use_compile)
                  for _ in range(plan.n_hawor)]
    elif any(frac is not None for frac in (plan.geo_frac, plan.moge_frac, plan.hawor_frac)):
        geo_gpus = plan.geo_frac if plan.geo_frac is not None else 1
        moge_gpus = plan.moge_frac if plan.moge_frac is not None else 1
        hawor_gpus = plan.hawor_frac if plan.hawor_frac is not None else 1
        geo_ws = []
        moge_ws = []
        haw_ws = []
        slam_ws = []
        group_count = min(plan.n_geo, plan.n_moge, plan.n_hawor, plan.n_slam)
        for _ in range(group_count):
            geo_ws.append(GeoCalibWorker.options(num_gpus=geo_gpus).remote())
            moge_ws.append(MoGeWorker.options(num_gpus=moge_gpus).remote(config.moge_model))
            haw_ws.append(HaWorWorker.options(num_gpus=hawor_gpus).remote(
                use_compile=config.use_compile))
            slam_ws.append(SlamWorker.options(num_gpus=1).remote())
        for _ in range(plan.n_geo - group_count):
            geo_ws.append(GeoCalibWorker.options(num_gpus=geo_gpus).remote())
        for _ in range(plan.n_moge - group_count):
            moge_ws.append(MoGeWorker.options(num_gpus=moge_gpus).remote(config.moge_model))
        for _ in range(plan.n_hawor - group_count):
            haw_ws.append(HaWorWorker.options(num_gpus=hawor_gpus).remote(
                use_compile=config.use_compile))
        for _ in range(plan.n_slam - group_count):
            slam_ws.append(SlamWorker.options(num_gpus=1).remote())
    else:
        geo_ws = []
        moge_ws = []
        haw_ws = []
        slam_ws = []
        group_count = min(plan.n_geo, plan.n_moge, plan.n_hawor, plan.n_slam)
        for _ in range(group_count):
            geo_ws.append(GeoCalibWorker.options(num_gpus=1).remote())
            moge_ws.append(MoGeWorker.options(num_gpus=1).remote(config.moge_model))
            haw_ws.append(HaWorWorker.options(num_gpus=1).remote(
                use_compile=config.use_compile))
            slam_ws.append(SlamWorker.options(num_gpus=1).remote())
        for _ in range(plan.n_geo - group_count):
            geo_ws.append(GeoCalibWorker.options(num_gpus=1).remote())
        for _ in range(plan.n_moge - group_count):
            moge_ws.append(MoGeWorker.options(num_gpus=1).remote(config.moge_model))
        for _ in range(plan.n_hawor - group_count):
            haw_ws.append(HaWorWorker.options(num_gpus=1).remote(
                use_compile=config.use_compile))
        for _ in range(plan.n_slam - group_count):
            slam_ws.append(SlamWorker.options(num_gpus=1).remote())

    if slam_ws is None:
        slam_ws = [SlamWorker.options(num_gpus=1).remote()
                   for _ in range(plan.n_slam)]

    geo_idle_steal = plan.upstream_frac is None
    geo_refs = [w.run_loop.remote(
        queues.q_pre, queues.q_moge, queues.q_hawor, config.geocalib_interval,
        queues.q_slam, queues.q_post, join_store, queues.q_stage,
        config.ba_steps1, config.ba_steps2, config.ba_steps3,
        config.moge_model, geo_idle_steal, config.delete_temp)
        for w in geo_ws]
    moge_refs = [w.run_loop.remote(queues.q_moge, join_store, queues.q_slam, queues.q_stage)
                 for w in moge_ws]
    haw_refs = [w.run_loop.remote(
        queues.q_hawor, join_store, queues.q_slam,
        queues.q_moge, queues.q_post, queues.q_stage,
        config.ba_steps1, config.ba_steps2, config.ba_steps3,
        config.moge_model, geo_idle_steal, config.delete_temp)
                for w in haw_ws]

    if config.slam_start_delay > 0:
        print(f'[pipeline]  {config.slam_start_delay:.1f}.'
              f'[pipeline]')
    slam_refs = [w.run_loop.remote(
        queues.q_slam, queues.q_post,
        queues.q_moge, join_store,
        config.ba_steps1, config.ba_steps2, config.ba_steps3,
        config.slam_start_delay, config.slam_steal_moge,
        config.moge_model, queues.q_stage, config.delete_temp)
        for w in slam_ws]
    return WorkerPool(
        join_store=join_store,
        geo_ws=geo_ws,
        moge_ws=moge_ws,
        haw_ws=haw_ws,
        tail_slam_ws=[],
        slam_ws=slam_ws,
        geo_refs=geo_refs,
        moge_refs=moge_refs,
        haw_refs=haw_refs,
        tail_slam_refs=[],
        slam_refs=slam_refs,
    )


def promote_released_worker_to_slam(
    pool: WorkerPool,
    queues: PipelineQueues,
    config: SchedulerConfig,
    worker,
    source: str,
) -> None:
    """Reuse a finished upstream actor as a tail SLAM worker on the same GPU."""
    ref = worker.run_tail_slam_after_release.remote(
        queues.q_slam, queues.q_post,
        queues.q_moge, pool.join_store,
        config.ba_steps1, config.ba_steps2, config.ba_steps3,
        config.moge_model, queues.q_stage, config.delete_temp,
    )
    pool.tail_slam_ws.append(worker)
    pool.tail_slam_refs.append(ref)
    pool.slam_ws.append(worker)
    pool.slam_refs.append(ref)
    print(f'[pipeline]  {source}.'
          f'total_slam={len(pool.slam_refs)}')


def release_workers(workers: list) -> None:
    """Terminate completed Ray Actors and release their GPU memory promptly."""
    for w in workers:
        try:
            ray.kill(w)
        except Exception:
            pass
