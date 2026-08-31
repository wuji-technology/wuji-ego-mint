
from __future__ import annotations

import glob
import contextlib
import os
import sys
import threading as _threading
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
_MEGASAM_ROOT = _ROOT / 'third_party' / 'mega-sam'


os.chdir(str(_MEGASAM_ROOT))
sys.path.insert(0, str(_MEGASAM_ROOT / 'base' / 'droid_slam'))
sys.path.insert(0, str(_MEGASAM_ROOT / 'UniDepth'))




_droid_modules_dir = str(_MEGASAM_ROOT / 'base' / 'droid_slam' / 'modules')
_modules_pkg = sys.modules.get('modules')
if (
    _modules_pkg is not None
    and hasattr(_modules_pkg, '__path__')
    and _droid_modules_dir not in _modules_pkg.__path__
):
    _modules_pkg.__path__.append(_droid_modules_dir)

from droid import Droid          # noqa: E402
from lietorch import SE3         # noqa: E402




import atexit as _atexit
import concurrent.futures as _cf
_SAVE_EXECUTOR = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix='megasam-save')
_SAVE_FUTURES: list[_cf.Future] = []
_SAVE_LOCK = _threading.Lock()


def _atomic_savez(out_path: str, **arrays) -> None:
    """Write an npz without exposing a half-written zip file to readers."""
    out = Path(out_path)
    tmp = out.with_name(f'.{out.name}.tmp.{os.getpid()}.{_threading.get_ident()}')
    try:
        with open(tmp, 'wb') as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out)
        try:
            dir_fd = os.open(out.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _async_savez(out_path: str, **arrays) -> None:
    try:
        _atomic_savez(out_path, **arrays)
        print(f'[backend]  {out_path}.')
    except Exception as _e:
        print(f'[backend]  {out_path}; {_e}.')
        raise


def reap_completed_saves() -> None:
    with _SAVE_LOCK:
        if not _SAVE_FUTURES:
            return
        done, pending = [], []
        for fut in _SAVE_FUTURES:
            (done if fut.done() else pending).append(fut)
        if not done:
            return
        _SAVE_FUTURES[:] = pending
    for fut in done:
        exc = fut.exception()
        if exc is not None:
            print(f'[backend]  {exc}.')


def _submit_savez(out_path: str, **arrays) -> None:


    reap_completed_saves()
    fut = _SAVE_EXECUTOR.submit(_async_savez, out_path, **arrays)
    with _SAVE_LOCK:
        _SAVE_FUTURES.append(fut)


def wait_for_pending_saves() -> None:
    """Wait for all MegaSAM npz writes submitted in this process."""
    errors: list[BaseException] = []
    while True:
        with _SAVE_LOCK:
            futures = list(_SAVE_FUTURES)
            _SAVE_FUTURES.clear()
        if not futures:
            break
        for fut in futures:
            try:
                fut.result()
            except BaseException as exc:
                errors.append(exc)
    if errors:
        raise RuntimeError(f'[backend]  {len(errors)}; {errors[0]}.')


_atexit.register(_SAVE_EXECUTOR.shutdown, wait=True)
_atexit.register(wait_for_pending_saves)




_DROID_NET_CACHE: dict = {}      # weights_path -> DroidNet instance (on cuda)
_DROID_NET_CPU_CACHE: dict = {}  # weights_path -> DroidNet instance (on cpu)
_DROID_NET_LOCK = _threading.Lock()


def _build_droid_net_cpu(weights: str):
    from collections import OrderedDict as _OD
    from droid_net import DroidNet as _DroidNet

    net = _DroidNet()
    sd = _OD([
        (k.replace('module.', ''), v)
        for k, v in torch.load(weights, map_location='cpu').items()
    ])
    sd['update.weight.2.weight'] = sd['update.weight.2.weight'][:2]
    sd['update.weight.2.bias']   = sd['update.weight.2.bias'][:2]
    sd['update.delta.2.weight']  = sd['update.delta.2.weight'][:2]
    sd['update.delta.2.bias']    = sd['update.delta.2.bias'][:2]
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


def preload_droid_net_cpu(weights: str) -> None:
    """Internal helper."""
    with _DROID_NET_LOCK:
        if weights in _DROID_NET_CACHE or weights in _DROID_NET_CPU_CACHE:
            return
        _DROID_NET_CPU_CACHE[weights] = _build_droid_net_cpu(weights)
        print(f'[backend]  {weights}.')


def _get_droid_net(weights: str):
    """Internal helper."""
    with _DROID_NET_LOCK:
        if weights not in _DROID_NET_CACHE:
            net = _DROID_NET_CPU_CACHE.pop(weights, None)
            if net is None:
                net = _build_droid_net_cpu(weights)
            net.to('cuda:0').eval()
            _DROID_NET_CACHE[weights] = net
            print(f'[backend]  {weights}.')
        return _DROID_NET_CACHE[weights]




def _image_stream(image_list, mono_disp_paths, aligns, K, use_depth=True):
    import queue as _queue
    import threading

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    align_scale, align_shift, normalize_scale = aligns


    pf_q: _queue.Queue = _queue.Queue(maxsize=4)

    def _prefetch():
        """Internal helper."""
        for idx, (img_path, mono_disp_path) in enumerate(zip(image_list, mono_disp_paths)):
            img_bgr = cv2.imread(img_path)
            mono_disp = np.load(mono_disp_path)
            depth = np.clip(
                1.0 / ((1.0 / normalize_scale) * (align_scale * mono_disp + align_shift)),
                1e-4, 1e4,
            )
            depth[depth < 1e-2] = 0.0

            cpu_img = torch.from_numpy(np.ascontiguousarray(img_bgr)).pin_memory()
            cpu_dep = torch.from_numpy(
                np.ascontiguousarray(depth.astype(np.float32))
            ).pin_memory()
            pf_q.put((idx, cpu_img, cpu_dep, img_bgr.shape[:2]))
        pf_q.put(None)


    threading.Thread(target=_prefetch, daemon=True).start()

    def _iter():
        while True:
            item = pf_q.get()
            if item is None:
                break

            t, cpu_img, cpu_dep, (h0, w0) = item
            scale = np.sqrt((384 * 512) / (h0 * w0))
            h1 = int(h0 * scale); h1 -= h1 % 8
            w1 = int(w0 * scale); w1 -= w1 % 8



            gpu_img = cpu_img.cuda(non_blocking=True)
            gpu_dep = cpu_dep.cuda(non_blocking=True)


            img_f = gpu_img.permute(2, 0, 1).unsqueeze(0).float()  # (1,3,H,W)
            image = F.interpolate(img_f, (h1, w1), mode='bilinear', align_corners=False).byte()

            depth_r = F.interpolate(
                gpu_dep.unsqueeze(0).unsqueeze(0), (h1, w1), mode='nearest-exact',
            ).squeeze()

            mask = torch.ones(h1, w1, device='cuda')
            intrinsics = torch.tensor(
                [fx * w1 / w0, fy * h1 / h0, cx * w1 / w0, cy * h1 / h0],
                device='cuda',
            )

            if use_depth:
                yield t, image, depth_r, intrinsics, mask
            else:
                yield t, image, intrinsics, mask

    return _iter()


def _predict_slam_image_size(h0: int, w0: int) -> tuple[int, int]:
    """Internal helper."""
    s = np.sqrt((384 * 512) / (h0 * w0))
    h1 = int(h0 * s); h1 -= h1 % 8
    w1 = int(w0 * s); w1 -= w1 % 8
    return h1, w1




def compute_depth_alignment(
    image_dir: str,
    mono_disp_dir: str,
    metric_depth_dir: str,
    scene_name: str,
    focal_px: float | None = None,
) -> dict:
    image_list         = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    image_list        += sorted(glob.glob(os.path.join(image_dir, '*.png')))
    mono_disp_paths    = sorted(glob.glob(os.path.join(mono_disp_dir, '*.npy')))
    metric_depth_paths = sorted(glob.glob(
        os.path.join(metric_depth_dir, scene_name, '*.npz')
    ))

    assert len(image_list) == len(mono_disp_paths) == len(metric_depth_paths), (
        f'[backend]  {len(image_list)}; {len(mono_disp_paths)}.'
        f'[backend]  {len(metric_depth_paths)}.'
    )

    img_0 = cv2.imread(image_list[0])
    h0, w0 = img_0.shape[:2]



    def _per_frame(mono_path: str, metric_path: str):
        da_disp      = np.float32(np.load(mono_path))
        uni_data     = np.load(metric_path)
        metric_depth = uni_data['depth']
        fov          = float(uni_data['fov'])

        da_disp = cv2.resize(
            da_disp, (metric_depth.shape[1], metric_depth.shape[0]),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        gt_disp = 1.0 / (metric_depth + 1e-8)

        valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
        gt_disp[valid_mask] = 1e-2

        sky_ratio = np.sum(da_disp < 0.01) / da_disp.size
        if sky_ratio > 0.5:
            m     = da_disp > 0.01
            gt_ms = gt_disp[m] - np.median(gt_disp[m]) + 1e-8
            da_ms = da_disp[m] - np.median(da_disp[m]) + 1e-8
            scale = float(np.median(gt_ms / da_ms))
            shift = float(np.median(gt_disp[m] - scale * da_disp[m]))
        else:
            gt_ms = gt_disp - np.median(gt_disp) + 1e-8
            da_ms = da_disp - np.median(da_disp) + 1e-8
            scale = float(np.median(gt_ms / da_ms))
            shift = float(np.median(gt_disp - scale * da_disp))

        return da_disp, scale, shift, fov

    import concurrent.futures as _cf_align
    n_workers = min(8, max(1, len(mono_disp_paths)))
    with _cf_align.ThreadPoolExecutor(max_workers=n_workers) as _ex:
        _results = list(_ex.map(_per_frame, mono_disp_paths, metric_depth_paths))

    mono_disp_list = [r[0] for r in _results]
    scales         = [r[1] for r in _results]
    shifts         = [r[2] for r in _results]
    fovs           = [r[3] for r in _results]

    ss_product      = np.array(scales) * np.array(shifts)
    med_idx         = int(np.argmin(np.abs(ss_product - np.median(ss_product))))
    align_scale     = scales[med_idx]
    align_shift     = shifts[med_idx]
    normalize_scale = (
        np.percentile((align_scale * np.array(mono_disp_list) + align_shift), 98) / 2.0
    )
    aligns = (align_scale, align_shift, normalize_scale)



    del mono_disp_list, _results

    K = np.eye(3, dtype=np.float32)
    K[0, 2] = w0 / 2.0
    K[1, 2] = h0 / 2.0
    if focal_px is not None:
        K[0, 0] = K[1, 1] = float(focal_px)
        opt_intr = False
        print(f'[backend]  {focal_px:.1f}.')
    else:
        fov_median = float(np.median(fovs))
        K[0, 0] = K[1, 1] = w0 / (2 * np.tan(np.radians(fov_median / 2.0)))
        opt_intr = True
        print(f'[backend]  {fov_median:.1f}; {K[0,0]:.1f}.')
    print(f'[MegaSAM] K={K}')

    return {
        'aligns':          aligns,
        'K':               K,
        'opt_intr':        opt_intr,
        'mono_disp_paths': mono_disp_paths,
        'image_list':      image_list,
        'h0':              int(h0),
        'w0':              int(w0),
    }




def run_slam(
    image_dir: str,
    mono_disp_dir: str,
    metric_depth_dir: str,
    scene_name: str,
    weights: str | None = None,
    focal_px: float | None = None,
    buffer: int = 756,
    filter_thresh: float = 2.0,
    keyframe_thresh: float = 2.0,
    frontend_thresh: float = 12.0,
    frontend_window: int = 25,
    frontend_radius: int = 2,
    frontend_nms: int = 1,
    backend_thresh: float = 16.0,
    backend_radius: int = 2,
    backend_nms: int = 3,
    ba_steps1: int = 10,
    ba_steps2: int = 20,
    ba_steps3: int = 15,
    skip_diag: bool = False,
    mono_alpha: float = 1e-5,
    _sub_timings: list | None = None,
    _sub_events:  list | None = None,
    _precomputed: dict | None = None,
    return_dense: bool = True,
) -> dict:
    import time as _tm_slam
    import time as _wall

    if weights is None:
        weights = str(_ROOT / 'model/megasam/megasam_final.pth')


    if _precomputed is not None:
        aligns         = _precomputed['aligns']
        K              = _precomputed['K']
        opt_intr       = _precomputed['opt_intr']
        mono_disp_paths = _precomputed['mono_disp_paths']
        image_list     = _precomputed['image_list']
        _h0            = _precomputed.get('h0')
        _w0            = _precomputed.get('w0')
        _dt_align      = 0.0
    else:
        _t_align      = _tm_slam.perf_counter()
        _wall_align_s = _wall.time()
        _pc       = compute_depth_alignment(
            image_dir, mono_disp_dir, metric_depth_dir, scene_name, focal_px,
        )
        aligns         = _pc['aligns']
        K              = _pc['K']
        opt_intr       = _pc['opt_intr']
        mono_disp_paths = _pc['mono_disp_paths']
        image_list     = _pc['image_list']
        _h0            = _pc.get('h0')
        _w0            = _pc.get('w0')
        _dt_align      = _tm_slam.perf_counter() - _t_align
        if _sub_events is not None:
            _sub_events.append({'task': 'SLAM-Align',
                                't_start': _wall_align_s, 't_end': _wall.time()})


    args = types.SimpleNamespace(
        weights=weights,
        buffer=buffer,
        image_size=None,
        disable_vis=True,
        beta=0.3,
        filter_thresh=filter_thresh,
        warmup=8,
        keyframe_thresh=keyframe_thresh,
        frontend_thresh=frontend_thresh,
        frontend_window=frontend_window,
        frontend_radius=frontend_radius,
        frontend_nms=frontend_nms,
        stereo=False,
        depth=False,
        upsample=False,
        scene_name=scene_name,
        backend_thresh=backend_thresh,
        backend_radius=backend_radius,
        backend_nms=backend_nms,
        mono_depth_path='',
        metric_depth_path='',
    )
    torch.multiprocessing.set_start_method('spawn', force=True)

    rgb_list = [] if return_dense else None
    depth_list = [] if return_dense else None
    last_t = last_image = last_depth = last_intrinsics = last_mask = None
    _dt_droid_init = 0.0
    _wall_init_s = _wall_init_e = None


    _t_track      = _tm_slam.perf_counter()
    _wall_track_s = _wall.time()


    stream = _image_stream(image_list, mono_disp_paths, aligns, K)



    droid = None
    if _h0 is not None and _w0 is not None:
        _h1, _w1 = _predict_slam_image_size(int(_h0), int(_w0))
        args.image_size = [_h1, _w1]
        _t_droid_init_start = _tm_slam.perf_counter()
        _wall_init_s        = _wall.time()
        droid = Droid(args, net=_get_droid_net(weights))
        _dt_droid_init = _tm_slam.perf_counter() - _t_droid_init_start
        _wall_init_e        = _wall.time()

    for t, image, depth, intrinsics, mask in tqdm(stream, desc='[backend]', total=len(image_list)):
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            _t_droid_init_start = _tm_slam.perf_counter()
            _wall_init_s        = _wall.time()
            droid = Droid(args, net=_get_droid_net(weights))
            _dt_droid_init = _tm_slam.perf_counter() - _t_droid_init_start
            _wall_init_e        = _wall.time()
        droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)
        if return_dense:
            rgb_list.append(image[0].cpu())
            # Dense output is a compatibility feature, not part of BA state.
            depth_list.append(depth.detach().cpu())
        last_t, last_image, last_depth, last_intrinsics, last_mask = (
            t, image, depth, intrinsics, mask,
        )

    droid.track_final(last_t, last_image, last_depth,
                      intrinsics=last_intrinsics, mask=last_mask)
    _dt_track     = _tm_slam.perf_counter() - _t_track
    _wall_track_e = _wall.time()
    if _sub_events is not None:
        if _wall_init_s is not None:
            _sub_events.append({'task': 'SLAM-Init',
                                't_start': _wall_init_s, 't_end': _wall_init_e})
        _sub_events.append({'task': 'SLAM-Track',
                            't_start': _wall_track_s, 't_end': _wall_track_e})


    _t_ba      = _tm_slam.perf_counter()
    _wall_ba_s = _wall.time()
    print('[backend]')
    traj_est, depth_est, motion_prob = droid.terminate(
        _image_stream(image_list, mono_disp_paths, aligns, K),
        _opt_intr=opt_intr,
        full_ba=True,
        scene_name=scene_name,
        ba_steps1=ba_steps1,
        ba_steps2=ba_steps2,
        ba_steps3=ba_steps3,
        skip_diag=skip_diag,
        mono_alpha=mono_alpha,
    )
    _dt_ba = _tm_slam.perf_counter() - _t_ba
    if _sub_events is not None:
        _sub_events.append({'task': 'SLAM-BA',
                            't_start': _wall_ba_s, 't_end': _wall.time()})

    T = traj_est.shape[0]
    intr_raw = droid.video.intrinsics[:T].cpu().numpy()[0] * 8.0
    del droid

    K_out = np.eye(3, dtype=np.float32)
    K_out[0, 0] = intr_raw[0]
    K_out[1, 1] = intr_raw[1]
    K_out[0, 2] = intr_raw[2]
    K_out[1, 2] = intr_raw[3]

    poses_th = torch.as_tensor(traj_est, device='cpu')
    cam_c2w = SE3(poses_th).inv().matrix().numpy().astype(np.float32)  # (T, 4, 4)

    if return_dense:
        images_np = np.uint8(
            np.array(rgb_list[:T])[:, ::-1, ...].transpose(0, 2, 3, 1)
        )
        _dl = depth_list[:T]
        if _dl and all(d.shape == _dl[0].shape for d in _dl):
            depths_np = np.float32(1.0 / (torch.stack(_dl).cpu().numpy() + 1e-6))
        else:
            depths_np = np.float32(
                1.0 / (np.array([d.cpu().numpy() for d in _dl]) + 1e-6)
            )
    else:
        _dl = None
        images_np = depths_np = None









    import gc
    del depth_list, _dl, rgb_list
    del traj_est, depth_est, motion_prob
    del last_t, last_image, last_depth, last_intrinsics, last_mask
    del stream
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if _sub_timings is not None:
        n = len(image_list)
        _sub_timings.append((f'[backend]  {n}.',    _dt_align,      'CPU'))
        _sub_timings.append((f'[backend]',          _dt_droid_init, '[backend]'))
        _sub_timings.append((f'[backend]  {n}.',        _dt_track - _dt_droid_init, 'GPU'))
        _sub_timings.append(('[backend]',                   _dt_ba,         'GPU'))

    result = {
        'cam_c2w': cam_c2w,
        'K': K_out,
        'slam_hw': (int(args.image_size[0]), int(args.image_size[1])),
    }
    if return_dense:
        result.update(depths=depths_np, images=images_np)
    return result




def run(
    image_dir: str,
    scene_name: str,
    output_path: str | None = None,
    slam_weights: str | None = None,
    depth_base_dir: str | None = None,
    focal_px: float | None = None,
    ba_steps1: int = 10,
    ba_steps2: int = 20,
    ba_steps3: int = 15,
    skip_diag: bool = False,
    mono_alpha: float = 1e-5,
    _sub_timings: list | None = None,
    _sub_events:  list | None = None,
    _precomputed: dict | None = None,
    return_dense: bool = True,
) -> dict:
    image_dir = str(Path(image_dir).resolve())

    if depth_base_dir is not None:
        _depth_base = Path(depth_base_dir)
    else:
        _depth_base = _MEGASAM_ROOT
    mono_outdir   = str(_depth_base / 'Depth-Anything' / scene_name)
    metric_outdir = str(_depth_base / 'UniDepth')



    _n_frames = len(glob.glob(os.path.join(image_dir, '*.jpg'))) +\
                len(glob.glob(os.path.join(image_dir, '*.png')))
    _buffer = max(64, _n_frames)
    print(f'[backend]  {_n_frames}; {_buffer}.')
    _slam_sub: list = [] if _sub_timings is not None else None
    result = run_slam(
        image_dir=image_dir,
        mono_disp_dir=mono_outdir,
        metric_depth_dir=metric_outdir,
        scene_name=scene_name,
        weights=slam_weights,
        focal_px=focal_px,
        buffer=_buffer,
        ba_steps1=ba_steps1,
        ba_steps2=ba_steps2,
        ba_steps3=ba_steps3,
        skip_diag=skip_diag,
        mono_alpha=mono_alpha,
        _sub_timings=_slam_sub,
        _sub_events=_sub_events,
        _precomputed=_precomputed,
        return_dense=return_dense,
    )

    if _sub_timings is not None and _slam_sub:
        _sub_timings.extend(_slam_sub)


    if output_path is None:
        out = _MEGASAM_ROOT / 'outputs' / f'{scene_name}_droid.npz'
    else:
        out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)







    _slam_ref = result.get('images', result.get('depths'))
    _slam_hw_arr = np.asarray(
        result.get('slam_hw', _slam_ref.shape[1:3] if _slam_ref is not None else ()),
        dtype=np.int32,
    )
    _save_kwargs = dict(cam_c2w=result['cam_c2w'], K=result['K'])
    if _slam_hw_arr.size == 2:
        _save_kwargs['slam_hw'] = _slam_hw_arr
    _submit_savez(str(out), **_save_kwargs)

    T = result['cam_c2w'].shape[0]
    K = result['K']
    print(f'[backend]')
    print(f'[backend]  {T}.')
    print(f'fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}')
    print(f'cam_c2w : shape={result["cam_c2w"].shape}')
    print(f'[backend]  {out}.')

    return result
