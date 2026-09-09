#!/usr/bin/env python3
"""Create bounded, metadata-free samples from clips approved for redistribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2


def load_review(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    clips = document.get("clips")
    if not isinstance(clips, list):
        raise ValueError("Review manifest must contain a 'clips' list.")
    return clips


def checked_source(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Reviewed clip is outside the input directory or missing: {relative}")
    return path


def blur_region(frame, normalized_box) -> None:
    height, width = frame.shape[:2]
    x, y, w, h = [float(value) for value in normalized_box]
    left, top = max(0, int(x * width)), max(0, int(y * height))
    right, bottom = min(width, int((x + w) * width)), min(height, int((y + h) * height))
    if right <= left or bottom <= top:
        return
    region = frame[top:bottom, left:right]
    kernel = max(31, (min(region.shape[:2]) // 5) | 1)
    frame[top:bottom, left:right] = cv2.GaussianBlur(region, (kernel, kernel), 0)


def transcode(source: Path, output: Path, masks: list, max_seconds: float, max_height: int) -> dict:
    capture = cv2.VideoCapture(str(source))
    # Phone footage carries a display-matrix rotation that OpenCV ignores
    # unless asked; without this the frames are decoded sideways.
    if capture.isOpened() and not capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1):
        capture.release()
        raise RuntimeError(
            "OpenCV did not apply the display-matrix rotation; "
            "phone footage would be decoded sideways."
        )
    if not capture.isOpened():
        raise ValueError(f"Cannot decode reviewed clip: {source.name}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_height / max(height, 1))
    target_width = max(2, int(round(width * scale / 2) * 2))
    target_height = max(2, int(round(height * scale / 2) * 2))
    frame_limit = max(1, int(round(max_seconds * source_fps)))

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{target_width}x{target_height}",
        "-r", f"{source_fps:.8f}", "-i", "-", "-an", "-map_metadata", "-1",
        "-c:v", "libx264", "-crf", "24", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    frames = 0
    try:
        while frames < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            if (target_width, target_height) != (width, height):
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            for box in masks:
                blur_region(frame, box)
            process.stdin.write(frame.tobytes())
            frames += 1
    finally:
        capture.release()
        if process.stdin:
            process.stdin.close()
    return_code = process.wait()
    if return_code != 0 or frames == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg failed while preparing {source.name}")
    return {"frames": frames, "fps": source_fps, "width": target_width, "height": target_height}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-manifest", required=True, type=Path)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--max-height", type=int, default=720)
    args = parser.parse_args(argv)
    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    released = []
    counters: dict[str, int] = {}
    for entry in load_review(args.review_manifest):
        if entry.get("redistribution_approved") is not True or entry.get("privacy_reviewed") is not True:
            raise ValueError(f"Clip lacks explicit redistribution/privacy approval: {entry.get('file', '<unknown>')}")
        dataset = str(entry.get("dataset", "sample")).lower().replace("_", "-")
        if dataset not in {"ego4d", "epic-kitchens", "egodex"}:
            raise ValueError(f"Unsupported public dataset label: {dataset}")
        counters[dataset] = counters.get(dataset, 0) + 1
        public_name = f"{dataset}-{counters[dataset]:02d}.mp4"
        output_path = output_root / public_name
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing sample: {public_name}")
        details = transcode(
            checked_source(input_root, str(entry["file"])),
            output_path,
            list(entry.get("normalized_masks", [])),
            args.max_seconds,
            args.max_height,
        )
        released.append(
            {
                "file": public_name,
                "dataset": dataset,
                "sha256": sha256(output_path),
                "privacy_masks": len(entry.get("normalized_masks", [])),
                "post_transform_reviewed": False,
                **details,
            }
        )

    manifest = {"schema_version": 1, "samples": released}
    (output_root / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(released)} clip(s). Complete a final frame-by-frame review before release.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

