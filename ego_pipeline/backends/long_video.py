
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np




LONG_VIDEO_THRESHOLD_S = float(os.environ.get('MINT_LONG_VIDEO_THRESHOLD_S', '15.0'))
CLIP_DURATION_S        = float(os.environ.get('MINT_CLIP_DURATION_S',        '15.0'))
CLIP_OVERLAP_S         = float(os.environ.get('MINT_CLIP_OVERLAP_S',         '1.0'))






def get_video_info(video_path: str) -> tuple[float, float, int]:
    """Internal helper."""
    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration_s = total_frames / fps if fps > 0 else 0.0
    return fps, duration_s, total_frames


def _write_clip_video(video_path: str, out_path: str,
                      start_frame: int, end_frame: int, fps: float) -> None:
    """Internal helper."""
    cap    = cv2.VideoCapture(video_path)
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    print(f'[backend]  {Path(out_path).name}; {start_frame}; {end_frame}.')


def split_video_into_clips(video_path: str, out_dir: Path,
                            clip_duration_s: float, overlap_s: float,
                            fps: float, total_frames: int) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_frames    = int(clip_duration_s * fps)
    overlap_frames = int(overlap_s * fps)
    stride_frames  = clip_frames - overlap_frames

    clips = []
    start = 0
    idx   = 0
    while start < total_frames:
        end       = min(start + clip_frames, total_frames)
        clip_path = out_dir / f'clip_{idx:03d}.mp4'
        if not clip_path.exists():
            _write_clip_video(video_path, str(clip_path), start, end, fps)
        clips.append({'path': str(clip_path), 'start_frame': start,
                      'end_frame': end, 'clip_idx': idx})
        if end >= total_frames:
            break
        start += stride_frames
        idx   += 1

    duration_s = total_frames / fps
    print(f'[backend]  {duration_s:.1f}; {len(clips)}.'
          f'[backend]  {clip_duration_s}; {overlap_s}.'
          f'[backend]  {clip_duration_s - overlap_s}.')
    return clips

