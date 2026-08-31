
from __future__ import annotations

import copy
import ctypes
import ctypes.util
import os
import time
from pathlib import Path

import ray
from ray.util.queue import Queue as RayQueue





_libc = None
_libc_resolved = False


def _get_libc():
    global _libc, _libc_resolved
    if not _libc_resolved:
        _libc_resolved = True
        try:
            lib = ctypes.CDLL(ctypes.util.find_library('c') or 'libc.so.6')
            _libc = lib if hasattr(lib, 'malloc_trim') else None
        except OSError:
            _libc = None
    return _libc


def _rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except Exception:
        return None


class MallocTrimmer:
    
    def __init__(self, label: str, every: int | None = None):
        self.label = label
        if every is None:
            try:
                every = int(os.environ.get('MINT_MALLOC_TRIM_EVERY', '24'))
            except ValueError:
                every = 24
        self.every = every
        self._n = 0
        self._libc = _get_libc() if every > 0 else None

    def tick(self) -> None:
        if self._libc is None or self.every <= 0:
            return
        self._n += 1
        if self._n % self.every:
            return
        rss0 = _rss_gb()
        t0 = time.perf_counter()
        self._libc.malloc_trim(0)
        dt_ms = (time.perf_counter() - t0) * 1e3
        rss1 = _rss_gb()
        if rss0 is not None and rss1 is not None:

            print(f'[{self.label}] malloc_trim #{self._n}: '
                  f'RSS {rss0:.2f} -> {rss1:.2f} GB (freed {rss0 - rss1:.2f}, {dt_ms:.0f}ms)')


class TraceMallocProbe:
    
    def __init__(self, label: str, topn: int = 12):
        self.label = label
        self.topn = topn
        raw = os.environ.get('MINT_TRACEMALLOC', '0').strip()
        self.enabled = raw.lower() not in ('0', '', 'false', 'off', 'no')
        try:
            self.every = max(1, int(raw))
        except ValueError:
            self.every = 50
        self._n = 0
        self._prev = None
        if self.enabled:
            try:
                import tracemalloc
                if not tracemalloc.is_tracing():
                    nframes = int(os.environ.get('MINT_TRACEMALLOC_FRAMES', '1'))
                    tracemalloc.start(max(1, nframes))
                print(f'[pipeline]  {label}; {self.every}.'
                      f'[pipeline]  {topn}.')
            except Exception as exc:
                print(f'[pipeline]  {label}; {exc}.')
                self.enabled = False

    def tick(self) -> None:
        if not self.enabled:
            return
        self._n += 1
        if self._n % self.every:
            return
        import gc
        import tracemalloc
        gc.collect()
        snap = tracemalloc.take_snapshot()
        cur, peak = tracemalloc.get_traced_memory()
        rss = _rss_gb()
        rss_s = f'{rss:.2f}GB' if rss is not None else '?'
        if self._prev is None:
            print(f'[pipeline]  {self.label}; {self._n}.'
                  f'(RSS={rss_s}, traced={cur / 1048576:.1f}MiB)')
        else:
            diffs = sorted(snap.compare_to(self._prev, 'lineno'),
                           key=lambda s: s.size_diff, reverse=True)[:self.topn]
            print(f'[{self.label} tracemalloc] clip#{self._n}  RSS={rss_s}  '
                  f'traced={cur / 1048576:.1f}MiB(peak {peak / 1048576:.1f})  '
                  f'[pipeline]  {self.topn}.')
            for st in diffs:
                fr = st.traceback[0]
                print(f'[pipeline]  {st.size_diff / 1024:+9.1f}; {st.count_diff:+d}.'
                      f'{fr.filename}:{fr.lineno}')
        self._prev = snap


def _fanout(q1: RayQueue, q2: RayQueue, item: dict) -> None:
    q1.put(item)
    q2.put(copy.copy(item))


def parent_scene(item: dict) -> str:
    parent = item.get('_parent') if isinstance(item, dict) else None
    return parent['scene'] if parent else item.get('scene', '')


def clip_idx(item: dict) -> int:
    parent = item.get('_parent') if isinstance(item, dict) else None
    if parent:
        return int(parent.get('clip_idx', 0))
    return int(item.get('clip_idx', 0))


def emit_stage_event(
    stage_q: RayQueue | None,
    item: dict,
    stage: str,
    status: str,
    *,
    error: str | None = None,
    extra: dict | None = None,
) -> None:
    if stage_q is None:
        return
    event = {
        'scene': parent_scene(item),
        'item_scene': item.get('scene'),
        'clip_idx': clip_idx(item),
        'stage': stage,
        'status': status,
        'error': error,
        'ts': time.time(),
    }
    if extra:
        event.update(extra)
    stage_q.put(event)


def _wait_frames(item: dict) -> None:
    ref = item.get('frames_ref')
    if ref is None:
        return
    ray.get(ref)
    item['frames_ref'] = None


def _build_clip_items(pre: dict, focal, t_geo: float, err) -> list[dict]:
    if not pre.get('is_long'):
        resume = pre.get('resume_clip_stages') or {}
        return [{**pre, 'focal': focal, 't_geo': t_geo,
                 'error': err or pre.get('error'),
                 'frames_ref': pre.get('frames_ref'),
                 'clip_idx': 0,
                 'resume_stages': resume.get('clip_000', {})}]

    work_dir = Path(pre['work_dir'])



    scene_prefix = str(Path(pre['scene']).parent)
    items    = []
    for clip in pre['clips']:
        ci        = clip['clip_idx']
        clip_work = work_dir / f'clip_{ci:03d}'
        clip_stem = Path(clip['path']).stem
        items.append({


            'video':        clip['path'],
            'scene':        (clip_stem if scene_prefix in ('', '.')
                             else f'{scene_prefix}/{clip_stem}'),
            'work_dir':     str(clip_work),
            'is_long':      False,
            'image_dir':    str(clip_work / 'frames'),
            'fps':          pre['fps'],
            'duration_s':   pre['duration_s'],
            'total_frames': clip['end_frame'] - clip['start_frame'],
            't_pre':        0.0,
            'focal':        focal,
            't_geo':        t_geo if ci == 0 else 0.0,
            'error':        err,
            'frames_ref':   clip.get('frames_ref'),
            'resume_stages': (pre.get('resume_clip_stages') or {}).get(
                f'clip_{ci:03d}', {}),

            'label_dir':    pre.get('label_dir'),
            '_parent': {
                'scene':        pre['scene'],
                'video':        pre['video'],
                'work_dir':     str(work_dir),
                'total_frames': pre['total_frames'],
                'fps':          pre['fps'],
                't_pre':        pre.get('t_pre', 0.0),
                'n_clips':      len(pre['clips']),
                'clip_info':    clip,
                'clip_idx':     ci,
            },
        })
    return items
