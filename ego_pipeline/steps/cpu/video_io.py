from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path


def get_video_info_step(video_path: str) -> tuple[float, float, int]:
    from ego_pipeline.backends.long_video import get_video_info
    return get_video_info(video_path)


def extract_frames_step(
    video_path: str,
    output_dir: str | Path,
    stride: int = 1,
) -> tuple[str, float]:
    import os
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob('*.jpg')) or sorted(output_dir.glob('*.png'))
    if existing:
        img = cv2.imread(str(existing[0]))
        H, W = img.shape[:2]
        print(f'[pipeline]  {len(existing)}; {W}; {H}.')
        return str(output_dir), 0.0

    t = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    # Phone footage carries a display-matrix rotation that OpenCV ignores
    # unless asked; without this the frames are decoded sideways.
    if cap.isOpened() and not cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1):
        cap.release()
        raise RuntimeError(
            "OpenCV did not apply the display-matrix rotation; "
            "phone footage would be decoded sideways."
        )
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    n_workers = min(8, os.cpu_count() or 4)

    idx = saved = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % stride == 0:
                path = str(output_dir / f'{saved:06d}.jpg')
                futures.append(ex.submit(cv2.imwrite, path, frame.copy(), jpeg_params))
                saved += 1
            idx += 1
        cap.release()
        for f in futures:
            f.result()

    elapsed = time.perf_counter() - t
    print(f'[pipeline]  {saved}; {output_dir}; {W}; {H}.')
    return str(output_dir), elapsed
