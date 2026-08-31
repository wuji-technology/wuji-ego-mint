from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

_MIN_VALID = 4


_HMP_MIN_VALID = 5
_HMP_SCALE = 1.4826


def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().copy()
    except Exception:
        pass
    return np.array(x, copy=True)



def block_spike(pred_result,
                pos_thresh: float = 0.15,
                rot_thresh: float = 45.0,
                max_block: int = 15,
                pos_recovery: float = 0.15):
    pos_thresh = float(pos_thresh)
    rot_thresh = float(rot_thresh)
    max_block = int(max_block)
    pos_recovery = float(pos_recovery)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = pred_result
    trans = _np(pred_trans)
    rot = _np(pred_rot)
    valid = _np(pred_valid).astype(bool)

    for h in range(2):
        idx = np.where(valid[h])[0]
        if len(idx) < _MIN_VALID:
            continue
        gaps = np.diff(idx).astype(float)
        N = len(idx)


        pos_step = np.linalg.norm(
            np.diff(trans[h][idx], axis=0), axis=1) / gaps  # (N-1,)

        rot_obj = Rotation.from_rotvec(rot[h][idx])
        rel = rot_obj[:-1].inv() * rot_obj[1:]
        rot_step = np.degrees(np.abs(rel.magnitude())) / gaps  # (N-1,)

        bad = np.zeros(N, dtype=bool)

        i = 1
        while i < N - 1:

            if not (pos_step[i - 1] > pos_thresh
                    or rot_step[i - 1] > rot_thresh):
                i += 1
                continue


            end = min(i + max_block, N - 1)
            for j in range(i, end):

                if not (pos_step[j] > pos_thresh
                        or rot_step[j] > rot_thresh):
                    continue


                pre_pos = trans[h][idx[i - 1]]
                post_pos = trans[h][idx[j + 1]]
                round_trip = float(np.linalg.norm(post_pos - pre_pos))

                if round_trip < pos_recovery:

                    bad[i:j + 1] = True
                    i = j + 1
                else:

                    i += 1
                break
            else:

                i += 1

        valid[h][idx[bad]] = False

    dtype = pred_valid.dtype if isinstance(pred_valid, np.ndarray) else np.float32
    return [trans, rot, pred_hand_pose, pred_betas, valid.astype(dtype)]



def _hmp_thresh(speed: np.ndarray, k: float, floor: float) -> float:
    """Internal helper."""
    med = float(np.median(speed))
    sigma = _HMP_SCALE * float(np.median(np.abs(speed - med)))
    return max(floor, med + k * sigma)


def _hmp_spike_mask(speed: np.ndarray, thr: float) -> np.ndarray:
    big = speed > thr                       # (N-1,)
    n = len(speed) + 1
    spike = np.zeros(n, dtype=bool)
    if n < 3:
        return spike
    spike[1:-1] = big[:-1] & big[1:]
    spike[0] = big[0] & ~big[1]
    spike[-1] = big[-1] & ~big[-2]
    return spike


def hampel(pred_result, k: float = 3.0, pos_floor: float = 0.10,
           rot_floor: float = 30.0):
    """Internal helper."""
    k = float(k)
    pos_floor = float(pos_floor)
    rot_floor = float(rot_floor)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = pred_result
    trans = _np(pred_trans)
    rot = _np(pred_rot)
    valid = _np(pred_valid).astype(bool)

    for h in range(2):
        idx = np.where(valid[h])[0]
        if len(idx) < _HMP_MIN_VALID:
            continue
        gaps = np.diff(idx).astype(float)


        pos_step = np.linalg.norm(np.diff(trans[h][idx], axis=0), axis=1) / gaps


        quats = Rotation.from_rotvec(rot[h][idx]).as_quat()
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i - 1]) < 0:
                quats[i] = -quats[i]
        dots = np.clip(np.abs(np.sum(quats[:-1] * quats[1:], axis=1)), 0.0, 1.0)
        rot_step = np.degrees(2.0 * np.arccos(dots)) / gaps

        bad = (_hmp_spike_mask(pos_step, _hmp_thresh(pos_step, k, pos_floor))
               | _hmp_spike_mask(rot_step, _hmp_thresh(rot_step, k, rot_floor)))
        valid[h][idx[bad]] = False

    dtype = pred_valid.dtype if isinstance(pred_valid, np.ndarray) else np.float32
    return [trans, rot, pred_hand_pose, pred_betas, valid.astype(dtype)]



def block_hampel(pred_result,
                 pos_thresh: float = 0.15,
                 rot_thresh: float = 45.0,
                 max_block: int = 15,
                 pos_recovery: float = 0.15):
    result = block_spike(pred_result,
                         pos_thresh=pos_thresh,
                         rot_thresh=rot_thresh,
                         max_block=max_block,
                         pos_recovery=pos_recovery)
    result = hampel(result)
    return result
