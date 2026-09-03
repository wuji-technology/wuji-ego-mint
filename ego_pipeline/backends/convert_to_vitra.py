#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation

MINT_ROOT = Path(__file__).resolve().parents[2]
HAWOR_DIR = MINT_ROOT / 'third_party' / 'HaWoR'
for _p in [str(HAWOR_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

MANO_RIGHT_PATH = str(HAWOR_DIR / '_DATA' / 'data' / 'mano' / 'MANO_RIGHT.pkl')






def aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Internal helper."""
    shape = aa.shape[:-1]
    flat = aa.reshape(-1, 3)
    R = Rotation.from_rotvec(flat).as_matrix()
    return R.reshape(*shape, 3, 3)






def compute_joints_mano(
    global_orient_world: np.ndarray,   # (T, 3, 3) float64
    hand_pose: np.ndarray,             # (T, 15, 3, 3) float64
    betas: np.ndarray,                 # (10,) float64
    transl_world: np.ndarray,          # (T, 3) float64
    valid: np.ndarray,                 # (T,) bool
) -> np.ndarray:
    import torch
    from lib.models.mano_wrapper import MANO

    T = global_orient_world.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mano = MANO(
        MANO_RIGHT_PATH,
        use_pca=False,
        flat_hand_mean=False,
    ).to(device).eval()

    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return np.zeros((T, 21, 3), dtype=np.float32)

    go  = torch.tensor(global_orient_world[valid_idx], dtype=torch.float32, device=device)
    hp  = torch.tensor(hand_pose[valid_idx], dtype=torch.float32, device=device)
    bt  = torch.tensor(np.tile(betas, (len(valid_idx), 1)), dtype=torch.float32, device=device)
    tr  = torch.tensor(transl_world[valid_idx], dtype=torch.float32, device=device)

    with torch.no_grad():
        out = mano(
            global_orient=go.reshape(-1, 1, 3, 3),
            hand_pose=hp.reshape(-1, 15, 3, 3),
            betas=bt,
            transl=tr,
            pose2rot=False,
        )
    joints_valid = out.joints.cpu().float().numpy()  # (N_valid, 21, 3)



    tr_np = tr.cpu().float().numpy()   # (N_valid, 3)
    offset = joints_valid[:, 0:1, :] - tr_np[:, None, :]  # (N_valid, 1, 3)
    joints_valid = joints_valid - offset

    joints = np.zeros((T, 21, 3), dtype=np.float32)
    joints[valid_idx] = joints_valid
    return joints






def _avg_speed(transl_world: np.ndarray, valid: np.ndarray) -> float:
    """Internal helper."""
    if valid.sum() < 2:
        return 0.0
    diffs = []
    for t in range(len(valid) - 1):
        if valid[t] and valid[t + 1]:
            diffs.append(np.linalg.norm(transl_world[t + 1] - transl_world[t]))
    return float(np.mean(diffs)) if diffs else 0.0


def _camera_stats(cam_c2w: np.ndarray) -> tuple[float, float]:
    T = cam_c2w.shape[0]
    if T < 2:
        return 0.0, 0.0

    total_rot = 0.0
    total_trans = 0.0
    for t in range(T - 1):
        R_rel = cam_c2w[t, :3, :3].T @ cam_c2w[t + 1, :3, :3]
        angle = float(Rotation.from_matrix(R_rel).magnitude()) * (180 / np.pi)
        total_rot += angle
        dt = np.linalg.norm(cam_c2w[t + 1, :3, 3] - cam_c2w[t, :3, 3])
        total_trans += float(dt)

    return total_rot, total_trans






def convert_to_vitra(
    pred_result: list,
    cam_c2w: np.ndarray,
    K: np.ndarray,
    video_name: str,
    video_decode_frame: np.ndarray | None = None,
    compute_joints: bool = True,
    anno_type: str = 'right',
) -> dict:
    import torch


    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = pred_result

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    pred_trans     = to_np(pred_trans)      # (2, T, 3)
    pred_rot       = to_np(pred_rot)        # (2, T, 3) axis-angle
    pred_hand_pose = to_np(pred_hand_pose)  # (2, T, 45) axis-angle flat
    pred_betas     = to_np(pred_betas)      # (2, T, 10)
    pred_valid     = np.asarray(pred_valid, dtype=bool)  # (2, T)

    T = pred_trans.shape[1]
    cam_c2w = cam_c2w.astype(np.float64)   # (T, 4, 4)


    T_slam = cam_c2w.shape[0]
    if T_slam != T:

        idx = np.linspace(0, T_slam - 1, T)
        lo = np.floor(idx).astype(int).clip(0, T_slam - 1)
        hi = np.ceil(idx).astype(int).clip(0, T_slam - 1)
        w  = (idx - lo)[:, None, None]
        cam_c2w = cam_c2w[lo] * (1 - w) + cam_c2w[hi] * w

    # world2cam
    world2cam = np.linalg.inv(cam_c2w)     # (T, 4, 4)

    if video_decode_frame is None:
        video_decode_frame = np.arange(T, dtype=np.int64)


    hand_data = {}
    for hand_idx, hand_name in [(0, 'left'), (1, 'right')]:
        valid = pred_valid[hand_idx]               # (T,) bool
        transl_world = pred_trans[hand_idx].astype(np.float64)  # (T, 3)


        rot_aa  = pred_rot[hand_idx]               # (T, 3)
        R_world = aa_to_rotmat(rot_aa).astype(np.float64)  # (T, 3, 3)


        pose_aa   = pred_hand_pose[hand_idx].reshape(T, 15, 3)  # (T, 15, 3)
        pose_rmtx = aa_to_rotmat(pose_aa).astype(np.float64)    # (T, 15, 3, 3)


        betas_all = pred_betas[hand_idx]  # (T, 10)
        if valid.sum() > 0:
            beta = betas_all[valid].mean(axis=0).astype(np.float64)
        else:
            beta = betas_all.mean(axis=0).astype(np.float64)


        R_w2c = world2cam[:, :3, :3]  # (T, 3, 3)
        t_w2c = world2cam[:, :3, 3]   # (T, 3)

        transl_cam = np.einsum('tij,tj->ti', R_w2c, transl_world) + t_w2c  # (T, 3)
        R_cam      = np.einsum('tij,tjk->tik', R_w2c, R_world)             # (T, 3, 3)

        # 5. MANO joints
        if compute_joints:
            joints_world = compute_joints_mano(
                R_world, pose_rmtx, beta, transl_world, valid)  # (T, 21, 3) float32
            joints_cam = (
                np.einsum('tij,tnj->tni', R_w2c, joints_world.astype(np.float64))
                + t_w2c[:, None, :]
            ).astype(np.float32)  # (T, 21, 3)
        else:
            joints_world = np.zeros((T, 21, 3), dtype=np.float32)
            joints_cam   = np.zeros((T, 21, 3), dtype=np.float32)


        avg_speed = _avg_speed(transl_world, valid)

        hand_data[hand_name] = {
            'beta':                        beta,
            'global_orient_camspace':      R_cam.astype(np.float64),
            'global_orient_worldspace':    R_world.astype(np.float64),
            'hand_pose':                   pose_rmtx.astype(np.float64),
            'transl_camspace':             transl_cam.astype(np.float64),
            'transl_worldspace':           transl_world.astype(np.float64),
            'kept_frames':                 valid.astype(np.int64),
            'joints_camspace':             joints_cam,
            'joints_worldspace':           joints_world.astype(np.float64),

            'wrist':                       transl_world[:, None, :].astype(np.float32),
            'max_translation_movement':    avg_speed,
            'max_wrist_rotation_movement': None,
            'max_finger_joint_angle_movement': None,
        }


    total_rot_deg, total_trans_m = _camera_stats(cam_c2w)


    if anno_type == 'left':
        avg_speed_main = _avg_speed(
            pred_trans[0].astype(np.float64), pred_valid[0])
    else:
        avg_speed_main = _avg_speed(
            pred_trans[1].astype(np.float64), pred_valid[1])


    episode_info = {
        'video_clip_id_segment': np.arange(T, dtype=np.int64),   # deprecated
        'extrinsics':            world2cam.astype(np.float64),
        'intrinsics':            K.astype(np.float64),
        'video_decode_frame':    video_decode_frame.astype(np.int64),
        'video_name':            video_name,
        'avg_speed':             float(avg_speed_main),
        'total_rotvec_degree':   float(total_rot_deg),
        'total_transl_dist':     float(total_trans_m),
        'anno_type':             anno_type,
        'text':         {'left': [], 'right': []},
        'text_rephrase': {'left': [], 'right': []},
        'left':  hand_data['left'],
        'right': hand_data['right'],
    }

    return episode_info



# CLI


def main():
    parser = argparse.ArgumentParser(
        description='[backend]',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--pth',       required=True,
                        help='[backend]')
    parser.add_argument('--megasam',   required=True,
                        help='[backend]')
    parser.add_argument('--video_name', default=None,
                        help='[backend]')
    parser.add_argument('--output',    required=True,
                        help='[backend]')
    parser.add_argument('--no_joints', action='store_true',
                        help='[backend]')
    parser.add_argument('--anno_type', default='right',
                        choices=['left', 'right', 'both'],
                        help='[backend]')
    args = parser.parse_args()

    pth_path     = Path(args.pth).resolve()
    megasam_path = Path(args.megasam).resolve()
    output_path  = Path(args.output).resolve()
    video_name   = args.video_name or pth_path.stem

    print(f'[backend]  {pth_path}.')
    pred_result = joblib.load(str(pth_path))

    print(f'[backend]  {megasam_path}.')
    megasam = np.load(str(megasam_path), allow_pickle=True)
    cam_c2w = megasam['cam_c2w']
    K       = megasam['K']

    print(f'[convert] pred_result: T={pred_result[0].shape[1]}  cam_c2w: {cam_c2w.shape}')

    episode_info = convert_to_vitra(
        pred_result=pred_result,
        cam_c2w=cam_c2w,
        K=K,
        video_name=video_name,
        compute_joints=not args.no_joints,
        anno_type=args.anno_type,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), episode_info)
    print(f'[backend]  {output_path}.')


    loaded = np.load(str(output_path), allow_pickle=True).item()
    T = loaded['extrinsics'].shape[0]
    for hand in ['left', 'right']:
        kf = loaded[hand]['kept_frames']
        print(f'  {hand}: T={T}  valid={kf.sum()}/{T} ({kf.mean()*100:.1f}%)')
    print('[backend]')


if __name__ == '__main__':
    main()
