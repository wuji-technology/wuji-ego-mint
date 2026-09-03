from __future__ import annotations

import concurrent.futures
import gc
import os
import queue as _queue
import sys
import threading
import time
from pathlib import Path


def load_moge_model(moge_model: str | None = None, device=None):
    import sys
    import torch
    from pathlib import Path as _Path

    vitra_dir  = _Path(__file__).resolve().parents[3]
    third_party = str(vitra_dir / 'third_party')
    if third_party not in sys.path:
        sys.path.insert(0, third_party)

    from MoGe.moge.model.v2 import MoGeModel
    _model_path = moge_model or str(vitra_dir / 'model' / 'moge2' / 'model.pt')
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    model  = MoGeModel.from_pretrained(_model_path).to(device).eval()
    return model, device


def run_moge_step(
    image_dir: str,
    scene_name: str,
    output_dir: str | Path,
    moge_model: str | None,
    focal_px: float | None,
    model=None,
    device=None,
    _sub_events: list | None = None,
) -> tuple[str, float]:
    import cv2
    import numpy as np
    import torch

    output_dir = Path(output_dir)


    scene_name = Path(scene_name).name
    img_dir    = Path(image_dir)
    exts       = {'.jpg', '.jpeg', '.png', '.bmp'}
    img_paths  = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)

    da_dir  = output_dir / 'moge_depth' / 'Depth-Anything' / scene_name
    uni_dir = output_dir / 'moge_depth' / 'UniDepth'       / scene_name

    if da_dir.exists() and len(list(da_dir.glob('*.npy'))) >= len(img_paths):
        print(f'[pipeline]  {scene_name}; {len(img_paths)}.')
        return str(output_dir / 'moge_depth'), 0.0

    vitra_dir  = Path(__file__).resolve().parents[3]
    third_party = str(vitra_dir / 'third_party')
    if third_party not in sys.path:
        sys.path.insert(0, third_party)

    _model_path = moge_model or str(vitra_dir / 'model' / 'moge2' / 'model.pt')
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _own_model = model is None
    if _own_model:
        from MoGe.moge.model.v2 import MoGeModel
        model = MoGeModel.from_pretrained(_model_path).to(device).eval()
    t_start = time.perf_counter()

    da_dir.mkdir(parents=True, exist_ok=True)
    uni_dir.mkdir(parents=True, exist_ok=True)

    n_io    = min(6, os.cpu_count() or 4)
    read_ex  = concurrent.futures.ThreadPoolExecutor(max_workers=n_io)
    write_ex = concurrent.futures.ThreadPoolExecutor(max_workers=n_io)

    def _read(p):
        img = cv2.imread(str(p))
        if img is None:
            return None, None, None
        H, W = img.shape[:2]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), H, W  # uint8

    def _write(da_path, uni_path, depth, fov_x_deg):
        disp = 1.0 / np.maximum(depth, 1e-6)
        np.save(da_path, disp.astype(np.float32))
        np.savez(uni_path, depth=depth.astype(np.float32), fov=np.float32(fov_x_deg))

    read_futs = [read_ex.submit(_read, p) for p in img_paths]
    read_ex.shutdown(wait=False)



    pf_q: _queue.Queue = _queue.Queue(maxsize=3)

    def _prefetch() -> None:
        for fut, p in zip(read_futs, img_paths):
            rgb, H, W = fut.result()
            if rgb is None:
                pf_q.put(None)
                continue
            cpu_t = torch.from_numpy(np.ascontiguousarray(rgb))  # uint8, CPU only
            pf_q.put((cpu_t, H, W, p.stem))
        pf_q.put('DONE')

    pf_thread = threading.Thread(target=_prefetch, daemon=True)
    pf_thread.start()


    pending: list  = []   # (stem, W, H, depth_cpu_t, K_cpu_t)
    write_futs_out = []
    i = 0

    t_infer_s_wall = time.time()
    while True:
        item = pf_q.get()
        if item == 'DONE':
            break
        if item is None:
            i += 1
            continue

        cpu_t, H, W, stem = item


        gpu_t = cpu_t.to(device)

        t_img = gpu_t.permute(2, 0, 1).float().div_(255.0)
        fov_x_hint = float(np.degrees(2 * np.arctan(W / (2 * focal_px)))) if focal_px else None
        with torch.no_grad():
            out = model.infer(t_img, fov_x=fov_x_hint)


        depth_cpu = out['depth'].to('cpu', non_blocking=True)
        K_cpu     = out['intrinsics'].to('cpu', non_blocking=True)
        pending.append((stem, W, H, depth_cpu, K_cpu))
        del out, t_img, gpu_t, cpu_t


        if len(pending) > 2:
            s, w, h, dc, kc = pending.pop(0)
            d_np   = dc.numpy()
            d_np   = np.where(np.isinf(d_np), 0.0, d_np).astype(np.float32)
            K_np   = kc.numpy()
            fx_px  = float(K_np[0, 0] * w)
            fov_x  = float(np.degrees(2 * np.arctan(w / (2 * fx_px))))
            write_futs_out.append(write_ex.submit(
                _write, str(da_dir / f'{s}.npy'), str(uni_dir / f'{s}.npz'), d_np, fov_x))
            del dc, kc, d_np, K_np

        i += 1
        if i % 50 == 0 or i == len(img_paths):
            print(f'  [moge] [{i}/{len(img_paths)}]')

    pf_thread.join()
    t_infer_e_wall = time.time()


    t_flush_s_wall = time.time()
    torch.cuda.synchronize(device)
    for s, w, h, dc, kc in pending:
        d_np  = dc.numpy()
        d_np  = np.where(np.isinf(d_np), 0.0, d_np).astype(np.float32)
        K_np  = kc.numpy()
        fx_px = float(K_np[0, 0] * w)
        fov_x = float(np.degrees(2 * np.arctan(w / (2 * fx_px))))
        write_futs_out.append(write_ex.submit(
            _write, str(da_dir / f'{s}.npy'), str(uni_dir / f'{s}.npz'), d_np, fov_x))
        del dc, kc, d_np, K_np

    write_ex.shutdown(wait=True)
    for f in write_futs_out:
        f.result()

    if _own_model:
        del model
    pending.clear()
    write_futs_out.clear()
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    t_flush_e_wall = time.time()

    if _sub_events is not None:
        _sub_events.append({'task': 'MoGe-Infer',
                            't_start': t_infer_s_wall, 't_end': t_infer_e_wall})
        _sub_events.append({'task': 'MoGe-Flush',
                            't_start': t_flush_s_wall, 't_end': t_flush_e_wall})

    elapsed = time.perf_counter() - t_start
    print(f'[pipeline]  {scene_name}; {da_dir.parent}; {elapsed:.1f}.')
    return str(output_dir / 'moge_depth'), elapsed
