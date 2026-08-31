
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .scanner import relpath_key
from .state import ProcessingManifest
from .status import Status


@dataclass
class ReconcileResult:
    """Internal helper."""

    pending: list[tuple[str, Path]]
    skipped_done: int
    skipped_failed: int
    skipped_manual: int
    rolled_back: int
    newly_added: int
    missing: list[str]


def reconcile(
    manifest: ProcessingManifest,
    input_root: Path,
    scanned_videos: list[Path],
    *,
    resume: bool = True,
    retry_failed: bool = False,
) -> ReconcileResult:
    key_to_path: dict[str, Path] = {
        relpath_key(v, input_root): v for v in scanned_videos
    }

    newly_added = 0
    rolled_back = 0



    for key in key_to_path:
        entry = manifest.videos.get(key)
        if entry is None:
            newly_added += 1
            continue
        if entry.status == Status.RUNNING:

            entry.status = Status.PENDING
            entry.started_at = None
            rolled_back += 1



    missing = [k for k in manifest.videos if k not in key_to_path]


    pending: list[tuple[str, Path]] = []
    skipped_done = skipped_failed = skipped_manual = 0

    for key, path in key_to_path.items():
        entry = manifest.videos.get(key)
        st = entry.status if entry is not None else Status.PENDING

        if st == Status.SKIPPED:
            skipped_manual += 1
            continue
        if st == Status.DONE and resume:
            skipped_done += 1
            continue
        if st == Status.FAILED and not retry_failed:
            skipped_failed += 1
            continue

        pending.append((key, path))

    return ReconcileResult(
        pending        = pending,
        skipped_done   = skipped_done,
        skipped_failed = skipped_failed,
        skipped_manual = skipped_manual,
        rolled_back    = rolled_back,
        newly_added    = newly_added,
        missing        = missing,
    )


def format_summary(result: ReconcileResult, manifest: ProcessingManifest,
                   *, scanned_total: int | None = None) -> str:
    counts = manifest.counts()
    stored = sum(counts.values())
    total = scanned_total if scanned_total is not None else stored
    return (
        f'[pipeline]  {len(result.pending)}.'
        f'[pipeline]  {result.newly_added}; {result.rolled_back}.'
        f'[pipeline]  {result.skipped_done}; {result.skipped_failed}.'
        f'skipped={result.skipped_manual})  '
        f'[pipeline]  {len(result.missing)}.'
        f'[pipeline]  {stored}; {total}.'
    )
