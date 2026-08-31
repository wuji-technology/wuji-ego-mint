"""Configuration objects for the pipeline scheduler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerConfig:
    output_dir: str
    frame_stride: int = 1
    geocalib_interval: float = 5.0
    moge_model: str | None = None
    ba_steps1: int = 10
    ba_steps2: int = 20
    ba_steps3: int = 15
    n_gpus: int | None = None
    use_compile: bool = False
    slam_start_delay: float = 0.0
    max_open_videos: int = 12
    low_open_videos: int = 6
    max_open_clip_credit: int = 48
    low_open_clip_credit: int = 24
    slam_steal_moge: bool = False
    delete_temp: bool = True
    resume_clip_stages: dict | None = None

    label_dir: str | None = None


    input_root: str | None = None


@dataclass(frozen=True)
class SchedulePlan:
    n_gpus: int
    n_geo: int
    n_moge: int
    n_hawor: int
    n_slam: int
    upstream_frac: float | None
    geo_frac: float | None
    hawor_frac: float | None
    moge_frac: float | None
    geo_gpu_ids: list[int]
    moge_gpu_ids: list[int]
    haw_gpu_ids: list[int]
    slam_gpu_ids: list[int]
