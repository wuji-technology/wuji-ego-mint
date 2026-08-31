"""GPU resource planning for the Ray pipeline."""

from __future__ import annotations

import ray

from .config import SchedulePlan


def build_schedule_plan(n_gpus: int | None = None) -> SchedulePlan:
    """Return the static worker/GPU layout used by the current pipeline."""
    if n_gpus is None:
        n_gpus = int(ray.cluster_resources().get('GPU', 4))

    if n_gpus == 2:
        return SchedulePlan(
            n_gpus=n_gpus,
            n_geo=1,
            n_moge=1,
            n_hawor=1,
            n_slam=1,
            upstream_frac=0.33,
            geo_frac=None,
            hawor_frac=None,
            moge_frac=None,
            geo_gpu_ids=[0],
            moge_gpu_ids=[0],
            haw_gpu_ids=[0],
            slam_gpu_ids=[1],
        )

    if n_gpus == 4:
        return SchedulePlan(
            n_gpus=n_gpus,
            n_geo=1,
            n_moge=1,
            n_hawor=1,
            n_slam=1,
            upstream_frac=None,
            geo_frac=1.0,
            hawor_frac=1.0,
            moge_frac=1.0,
            geo_gpu_ids=[0],
            moge_gpu_ids=[1],
            haw_gpu_ids=[2],
            slam_gpu_ids=[3],
        )

    if n_gpus > 4 and n_gpus % 4 != 0:
        raise ValueError(
            f'[pipeline]  {n_gpus}.'
            '[pipeline]'
        )

    groups = max(1, n_gpus // 4)
    n_geo = groups
    n_moge = groups
    n_hawor = groups
    n_slam = groups

    return SchedulePlan(
        n_gpus=n_gpus,
        n_geo=n_geo,
        n_moge=n_moge,
        n_hawor=n_hawor,
        n_slam=n_slam,
        upstream_frac=None,
        geo_frac=None,
        hawor_frac=None,
        moge_frac=None,
        geo_gpu_ids=[i * 4 for i in range(groups)],
        moge_gpu_ids=[i * 4 + 1 for i in range(groups)],
        haw_gpu_ids=[i * 4 + 2 for i in range(groups)],
        slam_gpu_ids=[i * 4 + 3 for i in range(groups)],
    )


def print_schedule_plan(plan: SchedulePlan, video_count: int, output_dir: str) -> None:
    sep = '=' * 60
    print(f'\n{sep}')
    print(f'[pipeline]  {video_count}; {plan.n_gpus}.')
    print(f'[pipeline]  {plan.n_geo}; {plan.geo_gpu_ids}.'
          f'[pipeline]  {plan.n_moge}; {plan.moge_gpu_ids}.'
          f'[pipeline]  {plan.n_hawor}; {plan.haw_gpu_ids}.'
          f'[pipeline]  {plan.n_slam}; {plan.slam_gpu_ids}.')
    print('[pipeline]')
    print(f'[pipeline]  {output_dir}.')
    print(f'{sep}\n')
