"""Internal helper."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import ray


TAIL_MIN_DURATION_S = 3.0


def _plan_clips(fps: float, total_frames: int, out_dir: Path,
                base_stem: str,
                clip_duration_s: float, overlap_s: float) -> tuple[list[dict], list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_frames    = int(clip_duration_s * fps)
    overlap_frames = int(overlap_s * fps)
    stride_frames  = max(1, clip_frames - overlap_frames)

    clips: list[dict] = []
    start = 0
    idx   = 0
    while start < total_frames:
        end = min(start + clip_frames, total_frames)
        start_s = start / fps
        end_s   = end   / fps
        clip_name = (f'{base_stem}_subclip{idx:03d}'
                     f'_{start_s:.1f}s_{end_s:.1f}s.mp4')
        clip_path = out_dir / clip_name
        clips.append({'path': str(clip_path), 'start_frame': start,
                      'end_frame': end, 'clip_idx': idx,
                      'start_s': start_s, 'end_s': end_s})
        if end >= total_frames:
            break
        start += stride_frames
        idx   += 1

    dropped: list[dict] = []

    if len(clips) > 1:
        tail = clips[-1]
        tail_dur = (tail['end_frame'] - tail['start_frame']) / fps
        if tail_dur < TAIL_MIN_DURATION_S:
            dropped.append({**tail, 'reason': f'short_tail<{TAIL_MIN_DURATION_S}s',
                            'duration_s': tail_dur})
            clips.pop()
    return clips, dropped


@ray.remote
def preprocess_meta(video_path: str, output_dir: str,
                    input_root: str | None = None) -> dict:
    from ego_pipeline.backends.long_video import (
        CLIP_DURATION_S,
        CLIP_OVERLAP_S,
        LONG_VIDEO_THRESHOLD_S,
    )
    from manifest.scanner import scene_key
    from steps.cpu.video_io import get_video_info_step

    t0 = time.perf_counter()
    scene_name = scene_key(video_path, input_root)
    work_dir   = Path(output_dir) / scene_name
    work_dir.mkdir(parents=True, exist_ok=True)
    tag = f'[CPU] [{scene_name}]'

    try:
        fps, duration_s, total_frames = get_video_info_step(video_path)
        print(f'[pipeline]  {tag}; {fps:.2f}; {duration_s:.1f}.'
              f'[pipeline]  {total_frames}; {time.perf_counter()-t0:.3f}.')

        if duration_s > LONG_VIDEO_THRESHOLD_S:
            clips_dir = work_dir / 'clips'
            clips, dropped = _plan_clips(
                fps, total_frames, clips_dir,


                base_stem=Path(video_path).stem,
                clip_duration_s=CLIP_DURATION_S, overlap_s=CLIP_OVERLAP_S,
            )
            for clip in clips:
                (work_dir / f'clip_{clip["clip_idx"]:03d}').mkdir(parents=True, exist_ok=True)
            if dropped:
                names = [Path(c['path']).stem for c in dropped]
                print(f'[pipeline]  {tag}; {names}.')
            return {
                'video': video_path, 'scene': scene_name, 'work_dir': str(work_dir),
                'is_long': True, 'clips': clips, 'dropped_clips': dropped,
                'fps': fps, 'duration_s': duration_s, 'total_frames': total_frames,
                't_pre': time.perf_counter() - t0, 'error': None,
            }
        return {
            'video': video_path, 'scene': scene_name, 'work_dir': str(work_dir),
            'is_long': False, 'image_dir': str(work_dir / 'frames'),
            'fps': fps, 'duration_s': duration_s, 'total_frames': total_frames,
            'dropped_clips': [],
            't_pre': time.perf_counter() - t0, 'error': None,
        }
    except Exception:
        err = traceback.format_exc()
        print(f'[pipeline]  {tag}; {err.splitlines()[-1]}.')
        return {
            'video': video_path, 'scene': scene_name, 'work_dir': str(work_dir),
            'is_long': False, 'fps': 0, 'duration_s': 0, 'total_frames': 0,
            'dropped_clips': [],
            't_pre': time.perf_counter() - t0, 'error': err,
        }


@ray.remote
def extract_all_frames_step(video_path: str, work_dir: str) -> str:
    from steps.cpu.video_io import extract_frames_step

    all_frames_dir = Path(work_dir) / '_all_frames'
    t = time.perf_counter()
    extract_frames_step(video_path, all_frames_dir, stride=1)
    print(f'[CPU] [extract-all] {Path(video_path).name}  '
          f'[pipeline]  {time.perf_counter()-t:.2f}; {all_frames_dir}.')
    return str(all_frames_dir)


@ray.remote
def link_clip_frames_step(
    all_frames_dir: str,
    video_path:     str,
    clip:           dict,
    work_dir:       str,
    stride:         int = 1,
) -> str:
    work_dir_p       = Path(work_dir)
    all_frames_dir_p = Path(all_frames_dir)
    ci               = clip['clip_idx']
    frames_dir       = work_dir_p / f'clip_{ci:03d}' / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)


    for p in frames_dir.iterdir():
        if p.is_symlink() or p.suffix.lower() in ('.jpg', '.png'):
            try: p.unlink()
            except FileNotFoundError: pass

    saved = 0
    for src_idx in range(clip['start_frame'], clip['end_frame'], stride):
        src = all_frames_dir_p / f'{src_idx:06d}.jpg'
        if not src.exists():
            break
        dst = frames_dir / f'{saved:06d}.jpg'
        dst.symlink_to(os.path.relpath(src, frames_dir))
        saved += 1



    abs_video = os.path.abspath(video_path)
    clip_mp4  = Path(clip['path'])
    clip_mp4.parent.mkdir(parents=True, exist_ok=True)
    if clip_mp4.is_symlink() or clip_mp4.exists():
        try: clip_mp4.unlink()
        except FileNotFoundError: pass
    clip_mp4.symlink_to(abs_video)

    print(f'[pipeline]  {ci:03d}; {saved}; {stride}; {frames_dir}.')
    return str(frames_dir)


@ray.remote
def extract_short_frames_step(video_path: str, frames_dir: str, stride: int = 1) -> str:
    """Internal helper."""
    from steps.cpu.video_io import extract_frames_step
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    extract_frames_step(video_path, frames_dir, stride=stride)
    print(f'[CPU] [extract] {Path(video_path).name}  stride={stride}  '
          f'[pipeline]  {time.perf_counter()-t:.2f}; {frames_dir}.')
    return frames_dir
