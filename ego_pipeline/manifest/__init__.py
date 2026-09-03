
from __future__ import annotations

import time
from pathlib import Path

from .scanner    import scan_input, relpath_key, scene_key, VIDEO_EXTS
from .state      import (
    ProcessingManifest, VideoEntry,
    DEFAULT_MANIFEST_NAME, MANIFEST_VERSION,
)
from .status     import Status
from .reconciler import reconcile, format_summary, ReconcileResult

__all__ = [
    'scan_input', 'relpath_key', 'scene_key', 'VIDEO_EXTS',
    'ProcessingManifest', 'VideoEntry',
    'DEFAULT_MANIFEST_NAME', 'MANIFEST_VERSION',
    'Status',
    'reconcile', 'format_summary', 'ReconcileResult',
    'prepare_manifest',
    'mark_running', 'mark_done', 'mark_failed', 'mark_skipped',
    'init_clip_stages', 'mark_clip_stage',
    'clip_stage_is_done', 'video_all_clip_stages_done',
    'rollback_running',
]

CLIP_STAGES = ('geo', 'moge', 'hawor', 'megasam')



def prepare_manifest(
    input_root: Path,
    scanned_videos: list[Path],
    *,
    manifest_path: str | Path | None = None,
    resume: bool = True,
    retry_failed: bool = False,
) -> tuple[ProcessingManifest, ReconcileResult]:
    path = Path(manifest_path) if manifest_path else input_root / 'run_batch_manifest.json'


    manifest = ProcessingManifest.load(path)
    if manifest is None:
        manifest = ProcessingManifest(path=path, input_root=str(input_root.resolve()))

    result = reconcile(
        manifest, input_root, scanned_videos,
        resume=resume, retry_failed=retry_failed,
    )
    manifest.save()
    return manifest, result



def mark_running(manifest: ProcessingManifest, key: str) -> None:
    entry = manifest.get(key)
    entry.status = Status.RUNNING
    entry.started_at = _now_iso()
    entry.attempts += 1
    entry.error = None
    manifest.save(force=False)


def mark_done(
    manifest: ProcessingManifest,
    key: str,
    *,
    clip_idx: int | None = None,
    output_dir: str | None = None,
    duration_s: float | None = None,
    extra: dict | None = None,
) -> None:
    entry = manifest.get(key)
    if clip_idx is None:
        entry.status = Status.DONE
        entry.finished_at = _now_iso()
        if output_dir is not None:
            entry.output_dir = output_dir
        if duration_s is not None:
            entry.duration_s = duration_s
        if extra:
            entry.extra.update(extra)
        entry.error = None
    else:
        _set_clip_finalized(entry, clip_idx, 'done',
                            output_dir=output_dir, duration_s=duration_s, error=None)
        if extra:
            entry.extra.update(extra)
        _refresh_video_status_from_clips(entry)
    manifest.save(force=False)


def mark_failed(
    manifest: ProcessingManifest,
    key: str,
    *,
    clip_idx: int | None = None,
    error: str,
) -> None:
    entry = manifest.get(key)
    err_trunc = _truncate(error)
    if clip_idx is None:
        entry.status = Status.FAILED
        entry.finished_at = _now_iso()
        entry.error = err_trunc
    else:
        _set_clip_finalized(entry, clip_idx, 'failed', error=err_trunc)

        _refresh_video_status_from_clips(entry)
        if entry.status == Status.FAILED and not entry.error:
            entry.error = err_trunc
    manifest.save(force=False)


def mark_skipped(
    manifest: ProcessingManifest,
    key: str,
    *,
    clip_idx: int | None = None,
    reason: str = '',
) -> None:
    entry = manifest.get(key)
    if clip_idx is None:
        entry.status = Status.SKIPPED
        if reason:
            entry.extra['skip_reason'] = reason
    else:
        _set_clip_finalized(entry, clip_idx, 'skipped',
                            error=None, output_dir=None, duration_s=None,
                            skip_reason=reason or None)
        _refresh_video_status_from_clips(entry)
    manifest.save(force=False)


def init_clip_stages(
    manifest: ProcessingManifest,
    key: str,
    *,
    clips_count: int,
    is_long: bool,
    save: bool = True,
) -> None:
    """Ensure clip/stage bookkeeping exists for one video entry."""
    entry = manifest.get(key)
    entry.is_long = bool(is_long) or entry.is_long
    entry.clips_count = max(1, int(clips_count), int(entry.clips_count or 0))
    clips = entry.clips if isinstance(entry.clips, dict) else {}
    for idx in range(entry.clips_count):
        clip_key = _clip_key(idx)
        clip_state = clips.setdefault(clip_key, {})
        for stage in CLIP_STAGES:
            val = clip_state.get(stage)
            if isinstance(val, dict):
                val.setdefault('status', 'pending')
            elif isinstance(val, str):
                clip_state[stage] = {'status': val}
            else:
                clip_state[stage] = {'status': 'pending'}
    entry.clips = clips
    _refresh_clips_done(entry)
    if save:
        manifest.save(force=False)


def mark_clip_stage(
    manifest: ProcessingManifest,
    key: str,
    *,
    clip_idx: int,
    stage: str,
    status: str,
    error: str | None = None,
    extra: dict | None = None,
    save: bool = True,
) -> None:
    """Update one clip stage. Driver-only; Ray workers must not call this."""
    if stage not in CLIP_STAGES:
        raise ValueError(f'unknown clip stage: {stage}')
    entry = manifest.get(key)
    if entry.clips_count <= clip_idx:
        init_clip_stages(
            manifest, key,
            clips_count=clip_idx + 1,
            is_long=entry.is_long or clip_idx > 0,
            save=False,
        )
    clip_key = _clip_key(clip_idx)
    clip_state = entry.clips.setdefault(clip_key, {})
    stage_state = clip_state.setdefault(stage, {})
    if not isinstance(stage_state, dict):
        stage_state = {'status': str(stage_state)}
        clip_state[stage] = stage_state
    stage_state['status'] = status
    stage_state['updated_at'] = _now_iso()
    if error:
        stage_state['error'] = _truncate(error)
    else:
        stage_state.pop('error', None)
    if extra:
        stage_state.update(extra)
    _refresh_clips_done(entry)
    if save:
        manifest.save(force=False)


def clip_stage_is_done(manifest: ProcessingManifest, key: str, clip_idx: int, stage: str) -> bool:
    entry = manifest.get(key)
    clip_state = entry.clips.get(_clip_key(clip_idx), {})
    stage_state = clip_state.get(stage)
    if isinstance(stage_state, dict):
        return stage_state.get('status') == 'done'
    return stage_state == 'done'


def video_all_clip_stages_done(entry: VideoEntry) -> bool:
    if entry.clips_count <= 0:
        return False
    for idx in range(entry.clips_count):
        clip_state = entry.clips.get(_clip_key(idx), {})
        for stage in CLIP_STAGES:
            stage_state = clip_state.get(stage)
            if isinstance(stage_state, dict):
                ok = stage_state.get('status') == 'done'
            else:
                ok = stage_state == 'done'
            if not ok:
                return False
    return True


def rollback_running(manifest: ProcessingManifest) -> int:
    n = 0
    for entry in manifest.videos.values():
        if entry.status == Status.RUNNING:
            entry.status = Status.PENDING
            entry.started_at = None
            n += 1
        clips = entry.clips if isinstance(entry.clips, dict) else {}
        for clip_state in clips.values():
            if not isinstance(clip_state, dict):
                continue
            for stage in CLIP_STAGES:
                ss = clip_state.get(stage)
                if isinstance(ss, dict) and ss.get('status') == 'running':
                    ss['status'] = 'pending'
                    ss['updated_at'] = _now_iso()


    manifest.save(force=True)
    return n



def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec='seconds')


def _clip_key(clip_idx: int) -> str:
    return f'clip_{clip_idx:03d}'


def _truncate(error: str, limit: int = 2000) -> str:
    return error if len(error) < limit else error[:limit] + '...[truncated]'


def _refresh_clips_done(entry: VideoEntry) -> None:
    done: list[str] = []
    for idx in range(max(0, entry.clips_count)):
        clip_key = _clip_key(idx)
        clip_state = entry.clips.get(clip_key, {})
        all_done = True
        for stage in CLIP_STAGES:
            stage_state = clip_state.get(stage)
            if isinstance(stage_state, dict):
                ok = stage_state.get('status') == 'done'
            else:
                ok = stage_state == 'done'
            if not ok:
                all_done = False
                break
        if all_done:
            done.append(clip_key)
    entry.clips_done = done


def _set_clip_finalized(entry: VideoEntry, clip_idx: int, status: str,
                        **fields) -> None:
    if entry.clips_count <= clip_idx:
        entry.clips_count = clip_idx + 1
    clips = entry.clips if isinstance(entry.clips, dict) else {}
    clip_state = clips.setdefault(_clip_key(clip_idx), {})
    fin = clip_state.get('finalized')
    if not isinstance(fin, dict):
        fin = {}
        clip_state['finalized'] = fin
    fin['status'] = status
    fin['updated_at'] = _now_iso()
    for k, v in fields.items():
        if v is None:
            fin.pop(k, None)
        else:
            fin[k] = v
    entry.clips = clips


def _refresh_video_status_from_clips(entry: VideoEntry) -> None:
    if entry.clips_count <= 0:
        return
    done = failed = skipped = pending = 0
    for idx in range(entry.clips_count):
        clip_state = entry.clips.get(_clip_key(idx)) if isinstance(entry.clips, dict) else None
        fin = clip_state.get('finalized') if isinstance(clip_state, dict) else None
        s = fin.get('status') if isinstance(fin, dict) else None
        if   s == 'done':    done    += 1
        elif s == 'failed':  failed  += 1
        elif s == 'skipped': skipped += 1
        else:                pending += 1

    if pending > 0:

        return
    if failed > 0:
        entry.status = Status.FAILED
    elif done > 0:
        entry.status = Status.DONE
    elif skipped > 0:
        entry.status = Status.SKIPPED
    entry.finished_at = _now_iso()
    if entry.status == Status.DONE:
        _strip_clip_detail(entry)


def _strip_clip_detail(entry: VideoEntry) -> None:
    entry.clips = {}
    entry.clips_done = []
