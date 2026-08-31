
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


VIDEO_EXTS = {'.mp4'}


def _scan_workers() -> int:
    try:
        n = int(os.environ.get('MINT_SCAN_WORKERS', '0'))
    except ValueError:
        n = 0
    if n > 0:
        return n
    return max(1, min(32, (os.cpu_count() or 8) // 2))


def _walk_videos_concurrent(root: Path, exts: set[str]) -> list[Path]:
    out: list[Path] = []
    out_lock = threading.Lock()
    cond = threading.Condition()
    pending = [1]
    n_workers = _scan_workers()
    ex = ThreadPoolExecutor(max_workers=n_workers,
                            thread_name_prefix='scan-walk')

    def _scan(d: str) -> None:
        local: list[Path] = []
        subdirs: list[str] = []
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:

                        name = e.name
                        dot = name.rfind('.')
                        if dot >= 0 and name[dot:].lower() in exts:
                            local.append(Path(e.path))
                            continue
                        if e.is_dir(follow_symlinks=False):
                            subdirs.append(e.path)
                    except OSError:
                        continue
        except OSError:
            pass
        if local:
            with out_lock:
                out.extend(local)

        with cond:
            pending[0] += len(subdirs)
        for sd in subdirs:
            ex.submit(_scan, sd)
        with cond:
            pending[0] -= 1
            if pending[0] == 0:
                cond.notify_all()

    print(f'[pipeline]  {n_workers}.', flush=True)
    ex.submit(_scan, str(root))
    with cond:
        while pending[0] != 0:
            cond.wait()
    ex.shutdown(wait=True)
    print(f'[pipeline]  {len(out)}.', flush=True)
    return out


def scan_input(input_path: str | Path) -> tuple[Path, list[Path]]:
    p = Path(input_path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f'[pipeline]  {p}.')

    if p.is_dir():
        root = p
        videos = _walk_videos_concurrent(p, VIDEO_EXTS)

    elif p.suffix.lower() == '.txt':
        root = p.parent
        videos = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                videos.append(Path(line).expanduser().resolve())

    elif p.suffix.lower() in VIDEO_EXTS:
        root = p.parent
        videos = [p]

    else:
        raise ValueError(
            f'[pipeline]  {p}.'
        )


    videos = sorted(set(videos))
    return root, videos


def relpath_key(video: Path, root: Path) -> str:
    try:
        return video.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return video.resolve().as_posix()


def scene_key(video: str | Path, input_root: str | Path | None) -> str:
    video = Path(video)
    if input_root:
        try:
            rel = video.resolve().relative_to(Path(input_root).resolve())
            return rel.with_suffix('').as_posix()
        except ValueError:
            pass
    return video.stem
