
from __future__ import annotations

import glob as _glob
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
_SENTINEL = object()


def _probe_frame_count(video_path: str) -> int:
    """Internal helper."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(n, 0)
    except Exception:
        return 0


def _build_ucm_grid(
    W: int, H: int, f_px: float, xi: float, device: torch.device
) -> torch.Tensor:
    u0, v0 = W / 2.0, H / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    Xc = (xs - u0) / f_px
    Yc = (ys - v0) / f_px
    Zc = torch.ones_like(Xc)
    a1 = torch.sqrt(Zc * Zc + Xc * Xc + Yc * Yc)
    a2 = Xc * Xc + Yc * Yc + Zc * Zc
    alpha = a1 / a2
    Xs_, Ys_, Zs_ = Xc * alpha, Yc * alpha, Zc * alpha
    den = xi * torch.sqrt(Xs_ * Xs_ + Ys_ * Ys_ + Zs_ * Zs_) + Zs_
    Xd = Xs_ * f_px / den + u0
    Yd = Ys_ * f_px / den + v0

    grid_x = (2.0 * Xd / (W - 1)) - 1.0
    grid_y = (2.0 * Yd / (H - 1)) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == 'cuda' and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device)


class UCMUndistorterGPU:
    """Internal helper."""

    def __init__(
        self,
        W: int,
        H: int,
        f_px: float,
        xi: float,
        device: str | torch.device = 'cuda',
    ):
        self.W = int(W)
        self.H = int(H)
        self.f_px = float(f_px)
        self.xi = float(xi)
        self.device = _resolve_device(device)
        self._grid = _build_ucm_grid(self.W, self.H, self.f_px, self.xi, self.device)

    @torch.no_grad()
    def undistort_tensor(self, img: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if img.dim() == 3:
            img = img.unsqueeze(0)
            squeeze = True
        if img.shape[-2] != self.H or img.shape[-1] != self.W:
            raise ValueError(
                f'[backend]  {tuple(img.shape[-2:])}; {self.H}; {self.W}.'
            )

        in_dtype = img.dtype
        if img.device != self.device:
            img = img.to(self.device, non_blocking=True)
        grid = self._grid.expand(img.shape[0], -1, -1, -1)
        out = F.grid_sample(
            img.float(), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        )
        if not in_dtype.is_floating_point:
            out = out.clamp(0, 255).to(in_dtype)
        else:
            out = out.to(in_dtype)
        return out.squeeze(0) if squeeze else out

    @torch.no_grad()
    def undistort_bgr_np(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Internal helper."""
        t = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous()         # (1, 3, H, W)
        out = self.undistort_tensor(t)                            # (1, 3, H, W)
        out = out.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
        return out




def run_image(
    input_image: str,
    f_px: float,
    xi: float,
    output_dir: str = 'output/undistort',
    xi_thresh: float = 0.05,
    device: str = 'cuda',
) -> dict:
    """Internal helper."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(input_image)
    if img is None:
        raise ValueError(f'[backend]  {input_image}.')
    H, W = img.shape[:2]

    apply = abs(xi) > xi_thresh
    if apply:
        u = UCMUndistorterGPU(W=W, H=H, f_px=f_px, xi=xi, device=device)
        t0 = time.perf_counter()
        und = u.undistort_bgr_np(img)
        elapsed = time.perf_counter() - t0
    else:
        und = img
        elapsed = 0.0

    und_path = out_dir / 'undistorted.jpg'
    params_path = out_dir / 'params.json'
    cv2.imwrite(str(und_path), und)

    result = {
        'f_pixels': float(f_px),
        'xi': float(xi),
        'image_W': W,
        'image_H': H,
        'undistorted': str(und_path),
        'distortion_applied': apply,
        'elapsed_sec': elapsed,
        'device': str(_resolve_device(device)),
    }
    with open(params_path, 'w') as fp:
        json.dump(result, fp, indent=2)
    print(f'[undistort_gpu] image {W}x{H} apply={apply} '
          f'[backend]  {elapsed*1000:.1f}; {und_path}.')
    return result




def _run_video_paths(
    input_video: str,
    output_video: str,
    params_path: str | None,
    f_px: float,
    xi: float,
    xi_thresh: float = 0.05,
    device: str | torch.device = 'cuda',
    batch_size: int = 16,
    fourcc: str = 'mp4v',
    log_every_batches: int = 20,
    log_prefix: str = '[undistort_gpu]',
    verbose: bool = True,
) -> dict:
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f'[backend]  {input_video}.')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    fourcc_int = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(output_video, fourcc_int, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            f'[backend]  {fourcc}; {W}; {H}; {fps}; {output_video}.'
        )

    apply = abs(xi) > xi_thresh
    dev   = _resolve_device(device)
    if verbose:
        print(f'[backend]  {log_prefix}; {Path(input_video).name}; {W}; {H}; {total}.'
              f'[backend]  {fps:.2f}; {f_px:.1f}; {xi:.4f}; {dev}.'
              f'apply={apply}  batch={batch_size}')

    written = 0
    t0 = time.perf_counter()

    if not apply:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            written += 1
    else:
        u   = UCMUndistorterGPU(W=W, H=H, f_px=f_px, xi=xi, device=dev)
        q_in:  queue.Queue = queue.Queue(maxsize=batch_size * 4)
        q_out: queue.Queue = queue.Queue(maxsize=batch_size * 4)
        stop  = threading.Event()
        errors: list[tuple[str, str]] = []

        def _decode() -> None:
            try:
                while not stop.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    while not stop.is_set():
                        try:
                            q_in.put(frame, timeout=0.5)
                            break
                        except queue.Full:
                            continue
            except BaseException:
                errors.append(('decode', traceback.format_exc()))
                stop.set()
            finally:

                while True:
                    try:
                        q_in.put(_SENTINEL, timeout=0.5)
                        break
                    except queue.Full:
                        try:
                            q_in.get_nowait()
                        except queue.Empty:
                            pass

        def _compute() -> None:
            processed = 0
            batch_idx = 0
            buf: list[np.ndarray] = []

            def _flush() -> None:
                nonlocal processed, batch_idx
                if not buf:
                    return
                arr = np.stack(buf, axis=0)                              # (B,H,W,3) uint8
                t = torch.from_numpy(arr).to(dev, non_blocking=True)
                t = t.permute(0, 3, 1, 2).contiguous()                   # (B,3,H,W)
                out_t = u.undistort_tensor(t)                            # (B,3,H,W)
                out_np = out_t.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                for fr in out_np:
                    while not stop.is_set():
                        try:
                            q_out.put(fr, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                processed += len(out_np)
                batch_idx += 1
                buf.clear()
                if verbose and batch_idx % log_every_batches == 0:
                    elapsed = time.perf_counter() - t0
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    print(f'  ...{processed}/{total} ({rate:.1f} fps) '
                          f'[{Path(input_video).name}]')

            try:
                while True:
                    item = q_in.get()
                    if item is _SENTINEL:
                        break
                    if stop.is_set():
                        continue
                    buf.append(item)
                    if len(buf) >= batch_size:
                        _flush()
                if not stop.is_set() and buf:
                    _flush()
            except BaseException:
                errors.append(('compute', traceback.format_exc()))
                stop.set()
            finally:
                while True:
                    try:
                        q_out.put(_SENTINEL, timeout=0.5)
                        break
                    except queue.Full:
                        try:
                            q_out.get_nowait()
                        except queue.Empty:
                            pass

        def _encode() -> None:
            nonlocal written
            try:
                while True:
                    item = q_out.get()
                    if item is _SENTINEL:
                        break
                    if stop.is_set():
                        continue
                    writer.write(item)
                    written += 1
            except BaseException:
                errors.append(('encode', traceback.format_exc()))
                stop.set()

        t_dec = threading.Thread(target=_decode, name='decode', daemon=True)
        t_cmp = threading.Thread(target=_compute, name='compute', daemon=True)
        t_enc = threading.Thread(target=_encode, name='encode', daemon=True)
        t_dec.start(); t_cmp.start(); t_enc.start()
        t_dec.join();  t_cmp.join();  t_enc.join()

        if errors:
            cap.release(); writer.release()
            msg = '\n'.join(f'[{name}]\n{tb}' for name, tb in errors)
            raise RuntimeError(f'[backend]  {msg}.')

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0
    fps_proc = (written / elapsed) if elapsed > 0 else 0.0

    result = {
        'f_pixels':           float(f_px),
        'xi':                 float(xi),
        'image_W':            W,
        'image_H':            H,
        'fps':                fps,
        'total_frames':       total,
        'frames_written':     written,
        'undistorted_video':  str(output_video),
        'distortion_applied': apply,
        'elapsed_sec':        elapsed,
        'throughput_fps':     fps_proc,
        'device':             str(dev),
        'batch_size':         batch_size,
        'fourcc':             fourcc,
    }
    if params_path is not None:
        Path(params_path).parent.mkdir(parents=True, exist_ok=True)
        with open(params_path, 'w') as fp:
            json.dump(result, fp, indent=2)

    if verbose:
        print(f'[backend]  {log_prefix}; {written}; {elapsed:.1f}.'
              f'[backend]  {fps_proc:.1f}; {output_video}.')
    return result


def run_video(
    input_video: str,
    f_px: float,
    xi: float,
    output_dir: str = 'output/undistort',
    xi_thresh: float = 0.05,
    device: str = 'cuda',
    batch_size: int = 16,
    fourcc: str = 'mp4v',
    log_every_batches: int = 20,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(input_video).suffix or '.mp4'
    return _run_video_paths(
        input_video=input_video,
        output_video=str(out_dir / f'undistorted{suffix}'),
        params_path=str(out_dir / 'params.json'),
        f_px=f_px, xi=xi,
        xi_thresh=xi_thresh, device=device,
        batch_size=batch_size, fourcc=fourcc,
        log_every_batches=log_every_batches,
    )




def _sibling_paths(input_video: str) -> tuple[str, str]:
    """Internal helper."""
    p = Path(input_video)
    out_video = p.parent / f'{p.stem}_undistorted{p.suffix}'
    out_params = p.parent / f'{p.stem}_undistorted.params.json'
    return str(out_video), str(out_params)


def _worker_main(
    worker_id: int,
    gpu_id: int,
    task_q: 'mp.Queue',
    result_q: 'mp.Queue',
    f_px: float,
    xi: float,
    xi_thresh: float,
    batch_size: int,
    fourcc: str,
    overwrite: bool,
    log_every_batches: int,
    verbose: bool,
) -> None:
    """Internal helper."""
    try:
        torch.cuda.set_device(gpu_id)
    except Exception:

        pass

    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    log_prefix = f'[w{worker_id}/gpu{gpu_id}]'

    while True:
        task = task_q.get()
        if task is None:
            break
        in_path = task
        out_video, out_params = _sibling_paths(in_path)

        if Path(out_video).exists() and not overwrite:
            result_q.put({
                'input': in_path, 'status': 'skipped',
                'output': out_video, 'worker': worker_id, 'gpu': gpu_id,
            })
            if verbose:
                print(f'{log_prefix} [skip] {in_path} (already exists)')
            continue

        try:
            res = _run_video_paths(
                input_video=in_path,
                output_video=out_video,
                params_path=out_params,
                f_px=f_px, xi=xi, xi_thresh=xi_thresh,
                device=device, batch_size=batch_size, fourcc=fourcc,
                log_every_batches=log_every_batches,
                log_prefix=log_prefix,
                verbose=verbose,
            )
            result_q.put({
                'input': in_path, 'status': 'ok', 'output': out_video,
                'worker': worker_id, 'gpu': gpu_id,
                'frames_written': res['frames_written'],
                'elapsed_sec': res['elapsed_sec'],
                'throughput_fps': res['throughput_fps'],
            })
        except BaseException as e:
            result_q.put({
                'input': in_path, 'status': 'error',
                'error': repr(e), 'traceback': traceback.format_exc(),
                'worker': worker_id, 'gpu': gpu_id,
            })
            print(f'{log_prefix} [error] {in_path}: {e!r}', file=sys.stderr)


def _resolve_gpus(gpus: Sequence[int] | str | None) -> list[int]:
    """Internal helper."""
    if gpus is None or gpus == '':
        n = torch.cuda.device_count()
        if n == 0:
            raise RuntimeError('[backend]')
        return list(range(n))
    if isinstance(gpus, str):
        return [int(x) for x in gpus.split(',') if x.strip() != '']
    return [int(x) for x in gpus]


def run_pool(
    inputs: Iterable[str],
    f_px: float,
    xi: float,
    gpus: Sequence[int] | str | None = None,
    workers_per_gpu: int = 2,
    overwrite: bool = False,
    xi_thresh: float = 0.05,
    batch_size: int = 16,
    fourcc: str = 'mp4v',
    log_every_batches: int = 20,
    summary_path: str | None = None,
    show_progress: bool = True,
    verbose_workers: bool = False,
) -> dict:
    inputs = [str(p) for p in inputs]
    if not inputs:
        raise ValueError('[backend]')
    gpu_ids = _resolve_gpus(gpus)
    n_workers = max(1, len(gpu_ids) * int(workers_per_gpu))


    if show_progress:
        try:
            from tqdm import tqdm as _tqdm
            scan_iter = _tqdm(inputs, desc='[pool] scanning', unit='clip', leave=False)
        except ImportError:
            scan_iter = inputs
    else:
        scan_iter = inputs
    frame_counts = {p: _probe_frame_count(p) for p in scan_iter}
    total_frames = sum(frame_counts.values())

    print(f'[backend]  {len(inputs)}; {total_frames}.'
          f'[backend]  {len(gpu_ids)}; {workers_per_gpu}; {n_workers}.'
          f'(gpus={gpu_ids}, overwrite={overwrite})')

    ctx = mp.get_context('spawn')
    task_q:   mp.Queue = ctx.Queue()
    result_q: mp.Queue = ctx.Queue()

    for inp in inputs:
        task_q.put(inp)
    for _ in range(n_workers):
        task_q.put(None)

    procs: list[mp.Process] = []
    for wid in range(n_workers):
        gpu = gpu_ids[wid % len(gpu_ids)]
        p = ctx.Process(
            target=_worker_main,
            args=(wid, gpu, task_q, result_q,
                  f_px, xi, xi_thresh, batch_size, fourcc,
                  overwrite, log_every_batches, verbose_workers),
            daemon=False,
        )
        p.start()
        procs.append(p)


    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(
                total=total_frames if total_frames > 0 else None,
                desc='[undistort]', unit='frame', unit_scale=True,
                smoothing=0.1, dynamic_ncols=True,
            )
        except ImportError:
            pbar = None

    results: list[dict] = []
    n_done = n_ok = n_skip = n_err = 0
    n_total = len(inputs)
    t0 = time.perf_counter()
    cum_frames_done = 0
    cum_frames_bar  = 0

    def _refresh_postfix() -> None:
        if pbar is None:
            return
        elapsed_now = time.perf_counter() - t0
        rate = cum_frames_done / elapsed_now if elapsed_now > 0 else 0.0
        pbar.set_postfix_str(
            f'clips {n_done}/{n_total} ok={n_ok} skip={n_skip} err={n_err} '
            f'@ {rate:.1f} fps'
        )

    try:
        while n_done < n_total:
            r = result_q.get()
            results.append(r)
            n_done += 1
            in_path = r.get('input', '')
            n_frames = frame_counts.get(in_path, 0)
            status = r.get('status')

            if status == 'ok':
                n_ok += 1
                got = r.get('frames_written', 0) or n_frames
                cum_frames_done += got

                cum_frames_bar += n_frames
                if pbar is not None:
                    pbar.update(n_frames)
            elif status == 'skipped':
                n_skip += 1
                cum_frames_bar += n_frames
                if pbar is not None:
                    pbar.update(n_frames)
            else:
                n_err += 1
                msg = (f'[pool] FAIL {Path(in_path).name}: '
                       f'{r.get("error", "unknown")}')
                if pbar is not None:
                    pbar.write(msg)
                else:
                    print(msg, file=sys.stderr)
            _refresh_postfix()
    finally:
        if pbar is not None:
            pbar.close()
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    elapsed = time.perf_counter() - t0
    summary = {
        'inputs':          n_total,
        'ok':              n_ok,
        'skipped':         n_skip,
        'errors':          n_err,
        'elapsed_sec':     elapsed,
        'cum_frames':      cum_frames_done,
        'cum_throughput_fps': cum_frames_done / elapsed if elapsed > 0 else 0.0,
        'gpus':            gpu_ids,
        'workers_per_gpu': int(workers_per_gpu),
        'results':         results,
    }
    print(f'[pool] all done: ok={n_ok} skip={n_skip} err={n_err} '
          f'in {elapsed:.1f}s ({summary["cum_throughput_fps"]:.1f} fps cum)')

    if summary_path is not None:
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as fp:
            json.dump(summary, fp, indent=2)
        print(f'[backend]  {summary_path}.')

    return summary




def _collect_inputs(args) -> list[str]:
    if args.inputs_glob:
        paths = sorted(_glob.glob(args.inputs_glob, recursive=True))
        return [p for p in paths if Path(p).suffix.lower() in _VIDEO_EXTS]
    if args.inputs_list:
        with open(args.inputs_list) as fp:
            return [ln.strip() for ln in fp if ln.strip()]
    return []


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description='[backend]')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--input',        help='[backend]')
    src.add_argument('--inputs_glob',  help='[backend]')
    src.add_argument('--inputs_list',  help='[backend]')

    p.add_argument('--output_dir', default='output/undistort',
                   help='[backend]')
    p.add_argument('--f_px',       type=float, required=True, help='[backend]')
    p.add_argument('--xi',         type=float, required=True, help='[backend]')
    p.add_argument('--xi_thresh',  type=float, default=0.05)
    p.add_argument('--device',     default='cuda', help="[backend]")
    p.add_argument('--batch_size', type=int,   default=16)
    p.add_argument('--fourcc',     default='mp4v')


    p.add_argument('--gpus',            default=None,
                   help='[backend]')
    p.add_argument('--workers_per_gpu', type=int, default=2)
    p.add_argument('--overwrite',       action='store_true',
                   help='[backend]')
    p.add_argument('--summary_path',    default=None,
                   help='[backend]')
    p.add_argument('--no_progress',     action='store_true',
                   help='[backend]')
    p.add_argument('--verbose_workers', action='store_true',
                   help='[backend]')

    args = p.parse_args()

    if args.input:
        ext = Path(args.input).suffix.lower()
        if ext in _VIDEO_EXTS:
            run_video(
                input_video=args.input,
                f_px=args.f_px, xi=args.xi,
                output_dir=args.output_dir, xi_thresh=args.xi_thresh,
                device=args.device, batch_size=args.batch_size, fourcc=args.fourcc,
            )
        else:
            run_image(
                input_image=args.input,
                f_px=args.f_px, xi=args.xi,
                output_dir=args.output_dir, xi_thresh=args.xi_thresh,
                device=args.device,
            )
    else:
        inputs = _collect_inputs(args)
        if not inputs:
            raise SystemExit('[backend]')
        summary = run_pool(
            inputs=inputs,
            f_px=args.f_px, xi=args.xi,
            gpus=args.gpus, workers_per_gpu=args.workers_per_gpu,
            overwrite=args.overwrite, xi_thresh=args.xi_thresh,
            batch_size=args.batch_size, fourcc=args.fourcc,
            summary_path=args.summary_path,
            show_progress=not args.no_progress,
            verbose_workers=args.verbose_workers,
        )
        if summary['errors'] > 0:
            sys.exit(1)


if __name__ == '__main__':
    main()
