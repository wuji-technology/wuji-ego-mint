
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .status import Status


MANIFEST_VERSION = 2
DEFAULT_MANIFEST_NAME = '.run_batch_manifest.json'





_SAVE_INTERVAL_S = float(os.environ.get('MINT_MANIFEST_SAVE_INTERVAL_S', '30'))


def _now() -> str:
    """Internal helper."""
    return datetime.now().isoformat(timespec='seconds')


@dataclass
class VideoEntry:
    """Internal helper."""

    status: Status = Status.PENDING
    output_dir: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    error: str | None = None
    attempts: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    is_long: bool = False
    clips_count: int = 0
    clips: dict[str, Any] = field(default_factory=dict)   # clip_000 -> stage state
    clips_done: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'VideoEntry':
        d = dict(d)
        d['status'] = Status(d.get('status', 'pending'))
        d.setdefault('extra', {})

        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ProcessingManifest:
    """Internal helper."""

    path: Path
    input_root: str
    videos: dict[str, VideoEntry] = field(default_factory=dict)
    version: int = MANIFEST_VERSION
    updated_at: str = field(default_factory=_now)


    _last_save_t: float = field(default=0.0, repr=False, compare=False)
    _dirty: bool = field(default=False, repr=False, compare=False)
    _min_save_interval: float = field(
        default=_SAVE_INTERVAL_S, repr=False, compare=False)


    def to_dict(self) -> dict[str, Any]:
        return {
            'version':    self.version,
            'input_root': self.input_root,
            'updated_at': self.updated_at,
            'videos':     {k: v.to_dict() for k, v in self.videos.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], path: Path) -> 'ProcessingManifest':
        return cls(
            path       = path,
            input_root = d.get('input_root', ''),
            version    = d.get('version', MANIFEST_VERSION),
            updated_at = d.get('updated_at', _now()),
            videos     = {
                k: VideoEntry.from_dict(v)
                for k, v in d.get('videos', {}).items()
            },
        )


    @classmethod
    def load(cls, path: Path) -> 'ProcessingManifest | None':
        """Internal helper."""
        if not path.exists():
            return None
        with open(path) as f:
            return cls.from_dict(json.load(f), path)

    def save(self, force: bool = True) -> None:
        now = time.monotonic()
        if not force and (now - self._last_save_t) < self._min_save_interval:
            self._dirty = True
            return

        self._dirty = False
        self._last_save_t = now
        self.updated_at = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + '.',
            suffix='.tmp',
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False)
                f.flush()
                if force:
                    os.fsync(f.fileno())
            os.replace(tmp_name, self.path)
        except Exception:

            try: os.unlink(tmp_name)
            except OSError: pass
            raise


    def get(self, key: str) -> VideoEntry:
        """Internal helper."""
        return self.videos.setdefault(key, VideoEntry())

    def keys_by_status(self, *statuses: Status) -> list[str]:
        wanted = set(statuses)
        return [k for k, v in self.videos.items() if v.status in wanted]

    def counts(self) -> dict[str, int]:
        """Internal helper."""
        out: dict[str, int] = {s.value: 0 for s in Status}
        for v in self.videos.values():
            out[v.status.value] += 1
        return out
