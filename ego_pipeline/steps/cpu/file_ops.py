
from __future__ import annotations

import os
from pathlib import Path


def copy_video_for_hawor(video_path: str | Path, hawor_dir: str | Path) -> str:
    hawor_dir = Path(hawor_dir)
    hawor_dir.mkdir(parents=True, exist_ok=True)
    hawor_video = hawor_dir / Path(video_path).name

    src = Path(video_path).resolve()
    if hawor_video.is_symlink() or hawor_video.exists():
        try: hawor_video.unlink()
        except FileNotFoundError: pass
    os.symlink(str(src), str(hawor_video))
    return str(hawor_video)
