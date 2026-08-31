"""Write lightweight LeRobot-v3 conversion sidecars under ``result/``.

The heavy pipeline artifacts stay in the scene work directory.  The result
directory gets stable small files and symlinks so later dataset conversion does
not need to reverse-engineer the Ray work layout.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any


def _to_numpy(x):
    import numpy as np

    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _frame_count(pred_result: list) -> int:
    arr = _to_numpy(pred_result[0])
    return int(arr.shape[1])


def _replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(target)


def _first_image_hw(frames_dir: Path) -> tuple[int, int] | None:
    first = next(iter(sorted(frames_dir.glob("*.jpg"))), None)
    if first is None:
        first = next(iter(sorted(frames_dir.glob("*.png"))), None)
    if first is None:
        return None
    import cv2

    img = cv2.imread(str(first))
    if img is None:
        return None
    h, w = img.shape[:2]
    return int(h), int(w)


def _video_hw(video_path: Path) -> tuple[int, int] | None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if h <= 0 or w <= 0:
        return None
    return h, w


def _scale_k_to_video(
    K_slam,
    slam_hw: tuple[int, int] | None,
    video_hw: tuple[int, int] | None,
):
    import numpy as np

    K = np.asarray(K_slam if K_slam is not None else np.eye(3), dtype=np.float32).copy()
    if slam_hw is None or video_hw is None:
        return K
    slam_h, slam_w = int(slam_hw[0]), int(slam_hw[1])
    video_h, video_w = int(video_hw[0]), int(video_hw[1])
    if slam_h <= 0 or slam_w <= 0 or video_h <= 0 or video_w <= 0:
        return K
    sx = video_w / slam_w
    sy = video_h / slam_h
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] *= sx
    K[1, 2] *= sy
    return K.astype("float32")


def _k_to_fov(K_video, video_hw: tuple[int, int] | None):
    import numpy as np

    if video_hw is None:
        return np.array([0.0, 0.0], dtype=np.float32)
    h, w = int(video_hw[0]), int(video_hw[1])
    fx, fy = float(K_video[0, 0]), float(K_video[1, 1])
    if fx <= 0 or fy <= 0 or h <= 0 or w <= 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.array([
        2.0 * np.arctan(w / (2.0 * fx)),
        2.0 * np.arctan(h / (2.0 * fy)),
    ], dtype=np.float32)


def _candidate_label_paths(video_path: Path) -> list[Path]:
    parent = video_path.parent
    stem = video_path.stem
    names = [
        f"{stem}.json",
        f"{stem}_tasks.json",
        f"{stem}_video2tasks.json",
        f"{stem}_vlm_label.json",
        f"{stem}_label.json",
        f"{stem}_result_v2.json",
    ]
    dirs = [
        parent,
        parent / "vlm_label",
        parent / "vlm_labels",
        parent / "labels",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        for name in names:
            path = directory / name
            if path not in seen:
                out.append(path)
                seen.add(path)
    return out


def _merged_label_paths(video_path: Path, base_stem: str) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    parents = [video_path.parent]
    try:
        real = video_path.resolve(strict=True)
        if real.parent not in parents:
            parents.append(real.parent)
    except (OSError, RuntimeError):
        pass
    stems = [video_path.stem]
    if base_stem and base_stem not in stems:
        stems.append(base_stem)
    for directory in parents:
        for stem in stems:
            path = directory / f"merged_{stem}.json"
            if path not in seen:
                out.append(path)
                seen.add(path)
    return out


def _read_label_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows


def _extract_segments(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, dict):
        return []

    if isinstance(raw.get("merged_actions"), list):
        return raw["merged_actions"]
    if isinstance(raw.get("actions"), list):
        return raw["actions"]
    if isinstance(raw.get("tasks"), list):
        return raw["tasks"]
    segs = raw.get("segments")
    if isinstance(segs, list):
        return segs
    if isinstance(segs, dict) and isinstance(segs.get("segments"), list):
        return segs["segments"]
    keys = set(raw)
    if keys & {"start_frame", "start_idx", "start", "start_time"}:
        return [raw]
    return []


def _normalize_hand(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"left", "l", "0"}:
        return "left"
    if s in {"right", "r", "1"}:
        return "right"
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return None


def _normalize_segment(item: Any, total_frames: int,
                       *, fps: float | None = None) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    def _first(*names: str):
        for name in names:
            if name in item:
                return item[name]
        return None

    task = _first("task", "instruction", "description", "prompt")
    task = None if task is None else str(task)
    hand = _normalize_hand(_first("main_type", "main_hand", "hand", "dominant_hand"))

    start = _first("start_frame", "start_idx", "start")
    end = _first("end_frame", "end_idx", "end")

    if start is None:
        start_s = _first("start_time")
        if start_s is not None and fps:
            start = int(round(float(start_s) * fps))
    if end is None:
        end_s = _first("end_time")
        if end_s is not None and fps:
            end = int(round(float(end_s) * fps))
    if start is None:
        start = 0
    if end is None:
        end = total_frames
    start_i = max(0, min(int(start), int(total_frames)))
    end_i = max(0, min(int(end), int(total_frames)))
    if end_i <= start_i:
        return None
    return {
        "task": task,
        "main_type": hand,
        "start_frame": start_i,
        "end_frame": end_i,
    }



_PART_RE = re.compile(r"^(?P<base>.+?)_part\d+_[0-9.]+s_[0-9.]+s$")


_SUBCLIP_RE = re.compile(
    r"^(?P<base>.+?)_subclip(?P<idx>\d+)_(?P<start>[0-9.]+)s_(?P<end>[0-9.]+)s$"
)

_TAIL_ALIGN_TOLERANCE_S = 0.6


_FIXED_LABEL_RE = re.compile(r"_fixed_(?P<n>[0-9]+(?:\.[0-9]+)?)s\.json$")


def _find_label_in_external_dir(label_dir: Path, base_stem: str,
                                clip_duration_s: float | None) -> Path | None:
    if not label_dir or not label_dir.is_dir():
        return None
    matches: list[Path] = sorted(label_dir.rglob(f"{base_stem}_fixed_*s.json"))
    if not matches:
        return None
    if clip_duration_s is None or len(matches) == 1:
        return matches[0]
    best = None
    best_n = -1.0
    for path in matches:
        m = _FIXED_LABEL_RE.search(path.name)
        if not m:
            continue
        n = float(m.group("n"))
        if abs(n - clip_duration_s) <= _TAIL_ALIGN_TOLERANCE_S:
            return path
        if n > best_n:
            best, best_n = path, n
    return best or matches[0]


def _load_atomic_segments(merged_path: Path, merged_raw: Any, total_frames: int,
                          *, fps: float | None = None) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    if merged_path.name.startswith("merged_"):
        candidates.append(
            merged_path.parent / ("aligned_" + merged_path.name[len("merged_"):]))
    src = merged_raw.get("source_merged_json") if isinstance(merged_raw, dict) else None
    if src:
        candidates.append(merged_path.parent / str(src))
    aligned_path = next((p for p in candidates if p.exists()), None)
    if aligned_path is None:
        return []
    raw = _read_label_file(aligned_path)
    actions = _extract_segments(raw)
    return [
        seg for seg in (
            _normalize_segment(item, total_frames, fps=fps) for item in actions
        )
        if seg is not None
    ]


def _missing_annotation(total_frames: int,
                        source_label_path: str | None = None) -> dict[str, Any]:
    return {
        "label_missing": True,
        "source_label_path": source_label_path,
        "segments": [{
            "task": None,
            "main_type": None,
            "start_frame": 0,
            "end_frame": int(total_frames),
        }],
        "atomic_segments": [],
    }


def _load_annotation(video_path: Path, total_frames: int,
                     *, fps: float | None = None,
                     label_dir: str | None = None) -> dict[str, Any]:
    m_sub = _SUBCLIP_RE.match(video_path.stem)



    def _resolved_parent(p: Path) -> Path | None:
        try:
            real = p.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if real == p.absolute() or real.parent == p.absolute().parent:
            return None
        return real.parent

    if m_sub:
        clip_idx     = int(m_sub.group("idx"))
        clip_start_s = float(m_sub.group("start"))
        clip_end_s   = float(m_sub.group("end"))
        base_stem    = m_sub.group("base")
        base_video   = video_path.with_name(base_stem + video_path.suffix)
        candidates   = _candidate_label_paths(base_video)

        real_parent = _resolved_parent(video_path)
        if real_parent is not None:
            candidates.extend(_candidate_label_paths(
                real_parent / (base_stem + video_path.suffix)))
    else:
        base_stem    = video_path.stem
        candidates = _candidate_label_paths(video_path)
        real_parent = _resolved_parent(video_path)
        if real_parent is not None and real_parent != video_path.parent:
            candidates.extend(_candidate_label_paths(
                real_parent / video_path.name))

        m_part = _PART_RE.match(video_path.stem)
        if m_part:
            base_stem = m_part.group("base")
            candidates.extend(_candidate_label_paths(
                video_path.with_name(base_stem + video_path.suffix)))




    merged_path = next(
        (p for p in _merged_label_paths(video_path, base_stem) if p.exists()),
        None,
    )
    if merged_path is not None:
        raw = _read_label_file(merged_path)
        actions = _extract_segments(raw)
        segments = [
            seg for seg in (
                _normalize_segment(item, total_frames, fps=fps)
                for item in actions
            )
            if seg is not None
        ]
        if not segments:
            segments = [{
                "task": None,
                "main_type": None,
                "start_frame": 0,
                "end_frame": int(total_frames),
            }]
        atomic_segments = _load_atomic_segments(
            merged_path, raw, total_frames, fps=fps)
        return {
            "label_missing": False,
            "source_label_path": str(merged_path.resolve()),
            "segments": segments,
            "atomic_segments": atomic_segments,
        }


    label_path: Path | None = None
    if label_dir:
        clip_dur_s = (clip_end_s - clip_start_s) if m_sub else None
        label_path = _find_label_in_external_dir(
            Path(label_dir), base_stem, clip_dur_s)
        if label_path is None:
            print(f"[pipeline]  {video_path.stem}; {label_dir}."
                  f"[pipeline]  {base_stem}.")
    if label_path is None:
        label_path = next((p for p in candidates if p.exists()), None)
    if label_path is None:
        return _missing_annotation(total_frames)

    raw = _read_label_file(label_path)
    actions = _extract_segments(raw)

    if m_sub:

        label_interval = (raw.get("interval_seconds")
                          if isinstance(raw, dict) else None)
        clip_dur_s = clip_end_s - clip_start_s
        if label_interval and abs(float(label_interval) - clip_dur_s) > _TAIL_ALIGN_TOLERANCE_S:
            print(f"[pipeline]  {video_path.stem}."
                  f"[pipeline]  {label_interval}; {clip_dur_s:.1f}."
                  "[pipeline]")
        if clip_idx >= len(actions):
            print(f"[pipeline]  {video_path.stem}; {clip_idx}."
                  f"[pipeline]  {len(actions)}.")
            return _missing_annotation(total_frames,
                                       source_label_path=str(label_path.resolve()))
        action = actions[clip_idx]
        if isinstance(action, dict):
            seg_start_s = action.get("start_time")
            if seg_start_s is not None and abs(float(seg_start_s) - clip_start_s) > _TAIL_ALIGN_TOLERANCE_S:
                print(f"[pipeline]  {video_path.stem}; {clip_idx}."
                      f"[pipeline]  {seg_start_s}; {clip_start_s:.1f}.")
            task = action.get("task", action.get("instruction"))
            hand_raw = action.get("main_type", action.get("main_hand"))
        else:
            task = None
            hand_raw = None
        return {
            "label_missing": False,
            "source_label_path": str(label_path.resolve()),
            "segments": [{
                "task": None if task is None else str(task),
                "main_type": _normalize_hand(hand_raw),
                "start_frame": 0,
                "end_frame": int(total_frames),
            }],
            "atomic_segments": [],
        }


    segments = [
        seg for seg in (
            _normalize_segment(item, total_frames, fps=fps)
            for item in actions
        )
        if seg is not None
    ]
    if not segments:
        segments = [{
            "task": None,
            "main_type": None,
            "start_frame": 0,
            "end_frame": int(total_frames),
        }]
    return {
        "label_missing": False,
        "source_label_path": str(label_path.resolve()),
        "segments": segments,
        "atomic_segments": [],
    }


def _json_safe_path(path: Path | None) -> str | None:
    return str(path.resolve()) if path is not None else None


def _json_link_path(path: Path | None) -> str | None:
    return str(path.absolute()) if path is not None else None


def write_lerobot3_sidecars(
    *,
    pred_result: list,
    result_dir: str | Path,
    scene_name: str,
    source_video: str | Path,
    work_dir: str | Path,
    cam_c2w,
    K,
    slam_hw: tuple[int, int] | None,
    fps: float | None,
    raw_pth_path: str | Path | None,
    cleaned_pth_path: str | Path | None,
    label_dir: str | None = None,
) -> dict[str, Any]:
    """Write result-local files used by ``result_to_lerobot_v3.py``."""
    import numpy as np

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir)
    source_video = Path(source_video)
    total_frames = _frame_count(pred_result)

    frames_dir = work_dir / "_all_frames"
    if not frames_dir.is_dir():
        frames_dir = work_dir / "frames"
    frames_dir_out: Path | None = None
    if frames_dir.is_dir():
        frames_dir_out = result_dir / "frames"
        _replace_symlink(frames_dir_out, frames_dir.resolve())

    source_link = result_dir / "source_video.mp4"
    if source_video.exists():
        _replace_symlink(source_link, source_video.resolve())

    video_hw = _first_image_hw(frames_dir) if frames_dir.is_dir() else None
    if video_hw is None and source_video.exists():
        video_hw = _video_hw(source_video)

    K_slam = np.asarray(K if K is not None else np.eye(3), dtype=np.float32)
    K_video = _scale_k_to_video(K_slam, slam_hw, video_hw)
    fov = _k_to_fov(K_video, video_hw)
    cam = np.asarray(cam_c2w, dtype=np.float32)

    camera_path = result_dir / "megasam.npz"
    np.savez(
        camera_path,
        cam_c2w=cam,
        K=K_slam,
        K_video=K_video.astype(np.float32),
        fov=fov.astype(np.float32),
        slam_hw=np.asarray(slam_hw if slam_hw is not None else [-1, -1], dtype=np.int32),
        video_hw=np.asarray(video_hw if video_hw is not None else [-1, -1], dtype=np.int32),
    )

    annotation = _load_annotation(source_video, total_frames, fps=fps,
                                  label_dir=label_dir)
    annotation_path = result_dir / "annotation.json"
    annotation_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "scene": scene_name,
        "source_video": _json_safe_path(source_video),
        "source_video_link": _json_link_path(source_link) if source_link.exists() or source_link.is_symlink() else None,
        "work_dir": _json_safe_path(work_dir),
        "result_dir": _json_safe_path(result_dir),
        "raw_pth": _json_safe_path(Path(raw_pth_path)) if raw_pth_path else None,
        "cleaned_pth": _json_safe_path(Path(cleaned_pth_path)) if cleaned_pth_path else None,
        "frames_dir": _json_link_path(frames_dir_out) if frames_dir_out is not None else None,
        "camera_npz": _json_safe_path(camera_path),
        "annotation_json": _json_safe_path(annotation_path),
        "fps": float(fps) if fps is not None else None,
        "total_frames": int(total_frames),
        "camera": {
            "slam_hw": list(slam_hw) if slam_hw is not None else None,
            "video_hw": list(video_hw) if video_hw is not None else None,
            "fov": fov.astype(float).tolist(),
        },
        "annotations": annotation["segments"],
    }
    manifest_path = result_dir / "lerobot3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
