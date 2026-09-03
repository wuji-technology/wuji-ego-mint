
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def fill_slerp_rot(pred_result: list, max_gap: int = 30) -> list:
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = pred_result

    try:
        import torch
        def _np(x): return x.cpu().numpy().copy() if isinstance(x, torch.Tensor) else x.copy()
    except ImportError:
        def _np(x): return x.copy()

    trans     = _np(pred_trans)
    rot       = _np(pred_rot)
    hand_pose = _np(pred_hand_pose)
    betas     = _np(pred_betas)
    valid     = _np(pred_valid).astype(bool)

    for h in range(2):
        for lo, hi in _gaps(valid[h]):
            if hi - lo - 1 > max_gap:
                continue
            fill = np.arange(lo + 1, hi)
            t    = np.array([lo, hi], dtype=float)
            tf   = fill.astype(float)


            for arr in (trans[h], hand_pose[h], betas[h]):
                for d in range(arr.shape[1]):
                    arr[fill, d] = np.interp(tf, t, arr[[lo, hi], d])


            key_rots = Rotation.from_rotvec(rot[h, [lo, hi]])
            rot[h, fill] = Slerp(t, key_rots)(tf).as_rotvec()

            valid[h, fill] = True

    return [trans, rot, hand_pose, betas, valid.astype(pred_valid.dtype
            if isinstance(pred_valid, np.ndarray) else np.float32)]


def _gaps(valid: np.ndarray) -> list[tuple[int, int]]:
    """Internal helper."""
    gaps, last = [], -1
    for i, v in enumerate(valid):
        if v:
            if last >= 0 and i - last > 1:
                gaps.append((last, i))
            last = i
    return gaps
