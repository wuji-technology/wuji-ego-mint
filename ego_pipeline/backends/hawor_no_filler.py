
from __future__ import annotations

import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
from glob import glob
from natsort import natsorted


_THIRD_PARTY = Path(__file__).resolve().parents[2] / 'third_party'
HAWOR_DIR   = _THIRD_PARTY / 'HaWoR'
_DROID_DIR  = _THIRD_PARTY / 'mega-sam' / 'base' / 'droid_slam'

for _p in [str(_DROID_DIR), str(HAWOR_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_droid_modules = str(_DROID_DIR / 'modules')
_modules_pkg = sys.modules.get('modules')
if (
    _modules_pkg is not None
    and hasattr(_modules_pkg, '__path__')
    and _droid_modules not in _modules_pkg.__path__
):
    _modules_pkg.__path__.append(_droid_modules)

from scripts.scripts_test_video.detect_track_video import detect_track_video
from contextlib import contextmanager
from scripts.scripts_test_video.hawor_slam import hawor_slam
from scripts.scripts_test_video.hawor_video import load_hawor
from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
from lib.eval_utils.custom_utils import cam2world_convert, load_slam_cam, interpolate_bboxes
from lib.pipeline.tools import parse_chunks

_DEFAULT_HAWOR_CKPT = str(_THIRD_PARTY.parent / 'model/hawor/hawor.ckpt')


@contextmanager
def _hawor_cwd():
    """Internal helper."""
    _cwd = os.getcwd()
    os.chdir(str(HAWOR_DIR))
    try:
        yield
    finally:
        os.chdir(_cwd)




def load_hawor_model(
    hawor_ckpt: str = _DEFAULT_HAWOR_CKPT,
    device: torch.device | None = None,
    use_compile: bool = False,
):
    """Internal helper."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with _hawor_cwd():
        print('Loading HAWOR model ...')
        hawor_model, _ = load_hawor(hawor_ckpt)
        hawor_model = hawor_model.to(device).eval()

    if use_compile and hasattr(torch, 'compile'):
        try:
            torch._dynamo.config.suppress_errors = True
            hawor_model.backbone  = torch.compile(hawor_model.backbone,  mode='reduce-overhead')
            hawor_model.mano_head = torch.compile(hawor_model.mano_head, mode='reduce-overhead')
            if hawor_model.st_module is not None:
                hawor_model.st_module = torch.compile(hawor_model.st_module, mode='reduce-overhead')
            if hawor_model.motion_module is not None:
                hawor_model.motion_module = torch.compile(hawor_model.motion_module, mode='reduce-overhead')
            print('[backend]')
        except Exception as e:
            print(f'[backend]  {e}.')

    return hawor_model




def _run_motion_estimation(hawor_model, args, start_idx, end_idx, seq_folder):
    file       = args.video_path
    video_root = os.path.dirname(file)
    video      = os.path.basename(file).split('.')[0]
    img_folder = getattr(args, 'img_folder', None) or f'{video_root}/{video}/extracted_images'
    imgfiles   = np.array(natsorted(glob(f'{img_folder}/*.jpg')))

    tracks = np.load(
        f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy',
        allow_pickle=True,
    ).item()

    img_focal = args.img_focal
    if img_focal is None:
        try:
            with open(os.path.join(seq_folder, 'est_focal.txt')) as f:
                img_focal = float(f.read())
        except Exception:
            img_focal = 600
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'w') as f:
                f.write(str(img_focal))

    cache_path = f'{seq_folder}/tracks_{start_idx}_{end_idx}/frame_chunks_all.npy'
    if os.path.exists(cache_path):
        print('  skip motion estimation (cached)')
        return joblib.load(cache_path), img_focal, None

    print('  running motion estimation ...')
    tid = np.array([tr for tr in tracks])

    left_trk, right_trk = [], []
    for idx in tid:
        trk   = tracks[idx]
        valid = np.array([t['det'] for t in trk])
        is_r  = np.concatenate([t['det_handedness'] for t in trk])[valid]
        if is_r.sum() / max(len(is_r), 1) < 0.5:
            left_trk.extend(trk)
        else:
            right_trk.extend(trk)
    left_trk  = sorted(left_trk,  key=lambda x: x['frame'])
    right_trk = sorted(right_trk, key=lambda x: x['frame'])
    final_tracks = {0: left_trk, 1: right_trk}
    tid = [0, 1]

    img = cv2.imread(imgfiles[0])
    H, W = img.shape[:2]
    img_center  = [W / 2, H / 2]
    frame_chunks_all = defaultdict(list)
    cam_space_data   = {}

    for idx in tid:
        trk   = final_tracks[idx]
        valid = np.array([t['det'] for t in trk])
        if valid.sum() < 2:
            continue
        boxes    = np.concatenate([t['det_box'] for t in trk])
        non_zero = np.where(np.any(boxes != 0, axis=1))[0]
        first_nz, last_nz = non_zero[0], non_zero[-1]
        boxes[first_nz:last_nz+1] = interpolate_bboxes(boxes[first_nz:last_nz+1])
        valid[first_nz:last_nz+1] = True
        boxes = boxes[first_nz:last_nz+1]
        is_r  = np.concatenate([t['det_handedness'] for t in trk])[valid]
        frame = np.array([t['frame'] for t in trk])[valid]
        is_r  = (np.ones((len(boxes), 1)) if is_r.sum() / len(is_r) >= 0.5
                 else np.zeros((len(boxes), 1)))

        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)
        frame_chunks_all[idx] = frame_chunks
        if len(frame_chunks) == 0:
            continue

        do_flip = (is_r[0] <= 0)
        for frame_ck, boxes_ck in zip(frame_chunks, boxes_chunks):
            results  = hawor_model.inference(
                imgfiles[frame_ck], boxes_ck,
                img_focal=img_focal, img_center=img_center, do_flip=do_flip,
            )
            data_out = {
                'init_root_orient': results['pred_rotmat'][None, :, 0],
                'init_hand_pose':   results['pred_rotmat'][None, :, 1:],
                'init_trans':       results['pred_trans'][None, :, 0],
                'init_betas':       results['pred_shape'][None, :],
            }
            init_root      = rotation_matrix_to_angle_axis(data_out['init_root_orient'])
            init_hand_pose = rotation_matrix_to_angle_axis(data_out['init_hand_pose'])
            if do_flip:
                init_root[..., 1] *= -1;  init_root[..., 2] *= -1
                init_hand_pose[..., 1] *= -1;  init_hand_pose[..., 2] *= -1
            data_out['init_root_orient'] = angle_axis_to_rotation_matrix(init_root)
            data_out['init_hand_pose']   = angle_axis_to_rotation_matrix(init_hand_pose)


            pred_path = os.path.join(seq_folder, 'cam_space', str(idx),
                                     f"{frame_ck[0]}_{frame_ck[-1]}.npz")
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            np.savez(pred_path, **{k: v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                                   for k, v in data_out.items()})






    if torch.cuda.is_available():
        torch.cuda.synchronize()

    joblib.dump(frame_chunks_all, cache_path)
    return frame_chunks_all, img_focal, cam_space_data




def _run_cam2world(args, start_idx, end_idx, frame_chunks_all, save_path,
                   cam_space_data=None):
    file       = args.video_path
    video_root = os.path.dirname(file)
    video      = os.path.basename(file).split('.')[0]
    seq_folder = os.path.join(video_root, video)
    img_folder = getattr(args, 'img_folder', None) or f'{video_root}/{video}/extracted_images'
    imgfiles   = np.array(natsorted(glob(f'{img_folder}/*.jpg')))

    if os.path.exists(save_path):
        print('  skip cam2world (cached)')
        return joblib.load(save_path)

    fpath = os.path.join(seq_folder, f'SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz')
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(fpath)

    T = len(imgfiles)
    pred_trans     = torch.zeros(2, T, 3)
    pred_rot       = torch.zeros(2, T, 3)
    pred_hand_pose = torch.zeros(2, T, 45)
    pred_betas     = torch.zeros(2, T, 10)
    pred_valid     = torch.zeros(2, T)

    for idx in [0, 1]:
        for frame_ck in frame_chunks_all[idx]:
            key = (idx, frame_ck[0], frame_ck[-1])
            if cam_space_data is not None and key in cam_space_data:
                data_out = cam_space_data[key]
            else:
                base = os.path.join(seq_folder, 'cam_space', str(idx),
                                    f"{frame_ck[0]}_{frame_ck[-1]}")
                if os.path.exists(base + '.npz'):
                    raw = np.load(base + '.npz')
                    data_out = {k: torch.from_numpy(raw[k].copy()) for k in raw}
                else:
                    with open(base + '.json') as f:
                        data_out = {k: torch.tensor(v) for k, v in json.load(f).items()}
            data_world = cam2world_convert(
                R_c2w[frame_ck], t_c2w[frame_ck], data_out,
                'right' if idx > 0 else 'left',
            )
            pred_trans[[idx], frame_ck]     = data_world['init_trans']
            pred_rot[[idx], frame_ck]       = data_world['init_root_orient']
            pred_hand_pose[[idx], frame_ck] = data_world['init_hand_pose'].flatten(-2)
            pred_betas[[idx], frame_ck]     = data_world['init_betas']
            pred_valid[[idx], frame_ck]     = 1

    result = [pred_trans, pred_rot, pred_hand_pose, pred_betas, (pred_valid > 0).numpy()]
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    joblib.dump(result, save_path)
    return result




import cv2  # noqa: E402  (after sys.path setup)


def run_one(
    hawor_model,
    video_path: str,
    img_focal:  float,
    save_path:  str,
    skip_done:  bool = True,
) -> list:
    if skip_done and os.path.exists(save_path):
        print(f'  skip (already done): {save_path}')
        return joblib.load(save_path)

    class _Args:
        pass

    args = _Args()
    args.video_path = str(video_path)
    args.img_focal  = float(img_focal)
    args.checkpoint = _DEFAULT_HAWOR_CKPT

    # Stage 1
    print('  Stage 1: detect & track')
    start_idx, end_idx, seq_folder, _ = detect_track_video(args)

    # Stage 2
    print('  Stage 2: motion estimation')
    frame_chunks_all, _, cam_space_data = _run_motion_estimation(
        hawor_model, args, start_idx, end_idx, seq_folder)

    # Stage 3
    slam_path = os.path.join(seq_folder, f'SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz')
    if not os.path.exists(slam_path):
        print('  Stage 3: SLAM')
        hawor_slam(args, start_idx, end_idx)
    else:
        print('  Stage 3: SLAM (skip, cached)')


    print('[backend]')
    result = _run_cam2world(args, start_idx, end_idx, frame_chunks_all, save_path,
                            cam_space_data=cam_space_data)

    return result


def run_batch(
    hawor_model,
    video_paths: Sequence[str],
    focals:      Sequence[float],
    save_dir:    str,
    skip_done:   bool = True,
    image_dirs:  Sequence[str] | None = None,
    pre_meta:    dict | None = None,
) -> dict[str, list]:
    with _hawor_cwd():
        return _run_batch_impl(hawor_model, video_paths, focals, save_dir,
                               skip_done, image_dirs, pre_meta)


def _run_batch_impl(
    hawor_model,
    video_paths: Sequence[str],
    focals:      Sequence[float],
    save_dir,
    skip_done:   bool = True,
    image_dirs:  Sequence[str] | None = None,
    pre_meta:    dict | None = None,
) -> dict[str, list]:
    assert len(video_paths) == len(focals), '[backend]'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    class _Args:
        pass

    args_list = []
    for i, (vp, fx) in enumerate(zip(video_paths, focals)):
        a = _Args()
        a.video_path = str(vp)
        a.img_focal  = float(fx)
        a.checkpoint = _DEFAULT_HAWOR_CKPT
        if image_dirs is not None:
            a.img_folder = str(image_dirs[i])
        args_list.append(a)

    sep = '=' * 60


    print(f'\n{sep}\nStage 1 / 3  Detect & Track\n{sep}')
    meta = {}
    for args in args_list:
        stem = Path(args.video_path).stem
        print(f'\n  [{stem}]')
        if pre_meta and args.video_path in pre_meta:
            meta[args.video_path] = pre_meta[args.video_path]
            print('  skip detect_track (pre_meta provided)')
        else:
            start, end, seq_folder, _ = detect_track_video(args)
            meta[args.video_path] = dict(start=start, end=end, seq_folder=seq_folder)


    print(f'\n{sep}\nStage 2 / 3  Motion Estimation\n{sep}')
    fc_map = {}
    cs_map = {}
    for args in args_list:
        stem = Path(args.video_path).stem
        m    = meta[args.video_path]
        print(f'\n  [{stem}]')
        fc_all, _, cs_data = _run_motion_estimation(
            hawor_model, args, m['start'], m['end'], m['seq_folder'])
        fc_map[args.video_path] = fc_all
        cs_map[args.video_path] = cs_data


    print(f'\n{sep}\nStage 3 / 3  SLAM\n{sep}')
    for args in args_list:
        stem      = Path(args.video_path).stem
        m         = meta[args.video_path]
        slam_path = os.path.join(m['seq_folder'],
                                 f"SLAM/hawor_slam_w_scale_{m['start']}_{m['end']}.npz")
        print(f'\n  [{stem}]')
        if skip_done and os.path.exists(slam_path):
            print('  skip SLAM (cached)')
            continue
        hawor_slam(args, m['start'], m['end'])


    print(f'[backend]  {sep}; {sep}.')
    results = {}
    for args in args_list:
        stem      = Path(args.video_path).stem
        m         = meta[args.video_path]
        save_path = str(save_dir / f'{stem}.pth')
        print(f'\n  [{stem}]')
        result = _run_cam2world(
            args, m['start'], m['end'],
            fc_map[args.video_path], save_path,
            cam_space_data=cs_map[args.video_path],
        )
        results[stem] = result

    print(f'[backend]  {save_dir}.')
    return results






class _HaWorArgs:
    """Internal helper."""
    pass


def run_stage1_only(
    video_path: str,
    img_focal: float,
    image_dir: str | None = None,
    *,
    hand_det_model=None,
) -> tuple:
    with _hawor_cwd():
        a = _HaWorArgs()
        a.video_path = str(video_path)
        a.img_focal  = float(img_focal)
        a.checkpoint = _DEFAULT_HAWOR_CKPT
        if image_dir is not None:
            a.img_folder = str(image_dir)

        print('  [Stage 1] detect & track')
        start, end, seq_folder, _ = detect_track_video(a, hand_det_model=hand_det_model)
        return dict(start=start, end=end, seq_folder=seq_folder), a


def run_stage1_detect(
    video_path: str,
    img_focal: float,
    image_dir: str | None = None,
    *,
    hand_det_model=None,
) -> dict:
    from lib.pipeline.tools import detect_phase1

    with _hawor_cwd():
        a = _HaWorArgs()
        a.video_path = str(video_path)
        a.img_focal  = float(img_focal)
        a.checkpoint = _DEFAULT_HAWOR_CKPT
        if image_dir is not None:
            a.img_folder = str(image_dir)

        file = str(video_path)
        root = os.path.dirname(file)
        seq  = os.path.basename(file).split('.')[0]
        seq_folder = f'{root}/{seq}'
        img_folder = str(image_dir) if image_dir else f'{seq_folder}/extracted_images'
        os.makedirs(seq_folder, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)

        imgfiles  = natsorted(glob(f'{img_folder}/*.jpg'))
        start_idx = 0
        end_idx   = len(imgfiles)

        cached_boxes = f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy'
        if os.path.exists(cached_boxes):
            print(f'  [Stage 1-detect] skip (cached) {start_idx}_{end_idx}')
            return {'cached': True, 'args': a,
                    'seq_folder': seq_folder, 'start': start_idx, 'end': end_idx,
                    'frame_dets': None}

        os.makedirs(f'{seq_folder}/tracks_{start_idx}_{end_idx}', exist_ok=True)
        print('  [Stage 1-detect] YOLO batch')
        frame_dets = detect_phase1(imgfiles, thresh=0.2, hand_det_model=hand_det_model)
        return {'cached': False, 'args': a,
                'seq_folder': seq_folder, 'start': start_idx, 'end': end_idx,
                'frame_dets': frame_dets}


def run_stage1_track(detect_result: dict) -> tuple:
    from lib.pipeline.tools import track_phase2

    a          = detect_result['args']
    seq_folder = detect_result['seq_folder']
    start_idx  = detect_result['start']
    end_idx    = detect_result['end']

    with _hawor_cwd():
        if not detect_result['cached']:
            print('  [Stage 1-track] BYTETracker')
            boxes_, tracks_ = track_phase2(detect_result['frame_dets'], thresh=0.2)
            track_dir = f'{seq_folder}/tracks_{start_idx}_{end_idx}'
            os.makedirs(track_dir, exist_ok=True)
            np.save(f'{track_dir}/model_boxes.npy',  boxes_)
            np.save(f'{track_dir}/model_tracks.npy', tracks_)

    return dict(start=start_idx, end=end_idx, seq_folder=seq_folder), a


def run_stage2_from_meta(
    hawor_model,
    detect_meta: dict,
    args,
) -> tuple:
    with _hawor_cwd():
        print('  [Stage 2] motion estimation')
        fc_all, _, cs_data = _run_motion_estimation(
            hawor_model, args,
            detect_meta['start'], detect_meta['end'], detect_meta['seq_folder'],
        )
        return fc_all, cs_data


def run_stage12(
    hawor_model,
    video_path: str,
    img_focal: float,
    image_dir: str | None = None,
) -> tuple:
    detect_meta, a  = run_stage1_only(video_path, img_focal, image_dir)
    fc_all, cs_data = run_stage2_from_meta(hawor_model, detect_meta, a)
    return detect_meta, fc_all, cs_data


def finalize_cam2world(
    video_path: str,
    img_focal: float,
    cam_c2w_np: np.ndarray,
    detect_meta: dict,
    fc_map,
    cs_map,
    save_path: str,
    image_dir: str | None = None,
) -> list:
    from scipy.spatial.transform import Rotation as _Rotation

    start      = detect_meta['start']
    end        = detect_meta['end']
    seq_folder = detect_meta['seq_folder']


    T_mega = cam_c2w_np.shape[0]
    R_all  = cam_c2w_np[:, :3, :3].astype(np.float64)
    t_all  = cam_c2w_np[:, :3,  3].astype(np.float64)
    quats  = _Rotation.from_matrix(R_all).as_quat()           # (T,4) [qx,qy,qz,qw]
    traj   = np.concatenate([t_all, quats], axis=1).astype(np.float64)

    slam_dir  = os.path.join(seq_folder, 'SLAM')
    slam_path = os.path.join(slam_dir, f'hawor_slam_w_scale_{start}_{end}.npz')
    os.makedirs(slam_dir, exist_ok=True)
    if not os.path.exists(slam_path):
        np.savez(slam_path,
                 traj=traj,
                 scale=np.float64(1.0),
                 tstamp=np.arange(T_mega, dtype=np.int32),
                 disps=np.ones(T_mega, dtype=np.float32),
                 img_focal=np.float64(img_focal),
                 img_center=np.array([0.0, 0.0]))
        print(f'[backend]  {os.path.basename(slam_path)}; {T_mega}.')
    else:
        print(f'[backend]')

    class _Args:
        pass
    a = _Args()
    a.video_path = str(video_path)
    a.img_focal  = float(img_focal)
    if image_dir is not None:
        a.img_folder = str(image_dir)

    with _hawor_cwd():
        print('[backend]')
        return _run_cam2world(a, start, end, fc_map, save_path, cam_space_data=cs_map)
