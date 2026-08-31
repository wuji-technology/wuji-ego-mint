from __future__ import annotations

import gc
import json
import sys
from pathlib import Path


def _ensure_path(vitra_dir: Path) -> None:
    geocalib_root = str(vitra_dir / 'third_party' / 'GeoCalib')
    if geocalib_root not in sys.path:
        sys.path.insert(0, geocalib_root)


def read_frames(
    video_path: str,
    geocalib_interval: float = 5.0,
) -> list:
    import cv2
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    vitra_dir = Path(__file__).resolve().parents[3]
    _ensure_path(vitra_dir)
    from geocalib.utils import numpy_image_to_torch

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return []

    percents = np.clip(np.arange(0, 100 + geocalib_interval, geocalib_interval), 0, 100)
    indices  = sorted(set(int(round(p / 100.0 * (total - 1))) for p in percents))

    def _grab(i: int):
        c = cv2.VideoCapture(str(video_path))
        try:
            c.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = c.read()
        finally:
            c.release()
        if not ok:
            return None
        return numpy_image_to_torch(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    with ThreadPoolExecutor(max_workers=min(8, len(indices))) as ex:
        results = list(ex.map(_grab, indices))
    return [t for t in results if t is not None]


def run_inference(
    frame_tensors: list,
    work_dir: str | Path,
    model=None,
) -> tuple[float, dict, float]:
    import time
    import torch

    work_dir    = Path(work_dir)
    calib_cache = work_dir / 'geocalib_result.json'

    if calib_cache.exists():
        with open(calib_cache) as f:
            calib = json.load(f)
        print(f'[pipeline]  {calib["fx"]:.1f}; {calib["fy"]:.1f}.')
        return calib['fx'], calib, 0.0

    if not frame_tensors:
        raise RuntimeError(f'[pipeline]  {work_dir}.')

    t = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _own_model = model is None
    if _own_model:
        vitra_dir = Path(__file__).resolve().parents[3]
        _ensure_path(vitra_dir)
        from geocalib import GeoCalib
        weights = str(vitra_dir / 'model' / 'geocalib' / 'pinhole.tar')
        model   = GeoCalib(weights=weights).to(device)

    batch = torch.stack(frame_tensors, dim=0).to(device)
    try:
        res = model.calibrate(batch, shared_intrinsics=True)
        camera = res['camera']
        fx = float(camera.f[0, 0].item())
        fy = float(camera.f[0, 1].item())
        cx = float(camera.c[0, 0].item())
        cy = float(camera.c[0, 1].item())
        calib = {
            'fx':       fx,
            'fy':       fy,
            'cx':       cx,
            'cy':       cy,
            'vfov_deg': float(torch.rad2deg(camera.vfov[0]).item()),

            'K':        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        }
    finally:
        del batch
        try:
            del res, camera
        except Exception:
            pass
        if _own_model:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    with open(calib_cache, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f'[pipeline]  {calib["fx"]:.1f}; {calib["fy"]:.1f}; {calib["vfov_deg"]:.1f}.')
    return calib['fx'], calib, time.perf_counter() - t
