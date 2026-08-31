
from __future__ import annotations

from typing import Callable

from scheduling import RuntimeScheduler, SchedulerConfig




def run_pipeline_multi(
    videos:            list[str],
    output_dir:        str,
    frame_stride:      int        = 1,
    geocalib_interval: float      = 5.0,
    moge_model:        str | None = None,
    ba_steps1:         int        = 10,
    ba_steps2:         int        = 20,
    ba_steps3:         int        = 15,
    n_gpus:            int | None = None,
    monitor=None,
    use_compile:       bool       = False,
    slam_start_delay:  float      = 0.0,
    max_open_videos:   int        = 12,
    low_open_videos:   int        = 6,
    max_open_clip_credit: int     = 48,
    low_open_clip_credit: int     = 24,
    slam_steal_moge: bool         = False,
    delete_temp: bool             = True,
    on_video_done: Callable[[dict], None] | None = None,
    on_stage_done: Callable[[dict], None] | None = None,
    resume_clip_stages: dict | None = None,
    label_dir: str | None = None,
    input_root: str | None = None,
) -> list[dict]:
    config = SchedulerConfig(
        output_dir=output_dir,
        frame_stride=frame_stride,
        geocalib_interval=geocalib_interval,
        moge_model=moge_model,
        ba_steps1=ba_steps1,
        ba_steps2=ba_steps2,
        ba_steps3=ba_steps3,
        n_gpus=n_gpus,
        use_compile=use_compile,
        slam_start_delay=slam_start_delay,
        max_open_videos=max_open_videos,
        low_open_videos=low_open_videos,
        max_open_clip_credit=max_open_clip_credit,
        low_open_clip_credit=low_open_clip_credit,
        slam_steal_moge=slam_steal_moge,
        delete_temp=delete_temp,
        resume_clip_stages=resume_clip_stages,
        label_dir=label_dir,
        input_root=input_root,
    )
    return RuntimeScheduler(
        config, monitor=monitor,
        on_video_done=on_video_done,
        on_stage_done=on_stage_done,
    ).run(videos)
