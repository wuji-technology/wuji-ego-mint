"""Bounded video decoding for public inference entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoBatch:
    frames_rgb: np.ndarray
    fps: float
    source_fps: float
    source_frames: int
    width: int
    height: int


def read_video(path: str | Path, max_frames: int | None = None, target_fps: float | None = None) -> VideoBatch:
    """Decode a video into RGB frames while enforcing explicit resource limits."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video does not exist: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"OpenCV cannot decode this video: {source.name}")
    # Phone footage carries a display-matrix rotation that OpenCV ignores unless
    # asked; without this the model sees sideways frames.
    if not capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1):
        raise RuntimeError(
            "OpenCV did not apply the display-matrix rotation; "
            "phone footage would be decoded sideways."
        )

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    requested_fps = source_fps if not target_fps or target_fps >= source_fps else float(target_fps)
    stride = max(1, int(round(source_fps / requested_fps)))
    output_fps = source_fps / stride

    frames: list[np.ndarray] = []
    source_index = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if source_index % stride == 0:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                if max_frames is not None and len(frames) >= max_frames:
                    break
            source_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames were decoded from {source.name}")
    return VideoBatch(
        frames_rgb=np.stack(frames),
        fps=output_fps,
        source_fps=source_fps,
        source_frames=source_count,
        width=width,
        height=height,
    )

