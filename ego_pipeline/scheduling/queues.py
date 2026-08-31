"""Queue wiring and queue-level helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ray.util.queue import Queue as RayQueue


@dataclass
class PipelineQueues:
    q_pre: RayQueue
    q_moge: RayQueue
    q_hawor: RayQueue
    q_slam: RayQueue
    q_post: RayQueue
    q_stage: RayQueue


def create_pipeline_queues() -> PipelineQueues:
    """Create unbounded Ray queues for the current static pipeline.

    q_post is intentionally unbounded: long videos expand to many clip-items,
    and the driver only drains q_post after GPU stages finish.
    """
    return PipelineQueues(
        q_pre=RayQueue(maxsize=0),
        q_moge=RayQueue(maxsize=0),
        q_hawor=RayQueue(maxsize=0),
        q_slam=RayQueue(maxsize=0),
        q_post=RayQueue(maxsize=0),
        q_stage=RayQueue(maxsize=0),
    )


def inject_sentinels(q: RayQueue, count: int) -> None:
    for _ in range(count):
        q.put(None)
