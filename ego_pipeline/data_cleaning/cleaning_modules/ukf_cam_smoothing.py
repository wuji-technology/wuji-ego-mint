from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

_MIN_VALID = 4
_ALPHA = 1.0
_BETA_UT = 2.0
_JITTER = 1e-9


def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().copy()
    except Exception:
        pass
    return np.array(x, copy=True)


def _sigma_meas_batch(Y: np.ndarray) -> np.ndarray:
    """Internal helper."""
    d2 = Y[:, :-2] - 2.0 * Y[:, 1:-1] + Y[:, 2:]
    med = np.median(d2, axis=1, keepdims=True)
    mad = np.median(np.abs(d2 - med), axis=1)
    return np.maximum(1.4826 * mad / np.sqrt(6.0), 1e-6)


def _sigma_points_batch(m: np.ndarray, P: np.ndarray, lam: float) -> np.ndarray:
    """Internal helper."""
    C, n = m.shape
    M = (n + lam) * P
    M = 0.5 * (M + M.transpose(0, 2, 1)) + _JITTER * np.eye(n)
    L = np.linalg.cholesky(M)
    pts = np.empty((C, 2 * n + 1, n))
    pts[:, 0] = m
    for i in range(n):
        col = L[:, :, i]
        pts[:, 1 + i] = m + col
        pts[:, 1 + n + i] = m - col
    return pts


def _urtss_core_sr(t: np.ndarray, Y: np.ndarray, Rt: np.ndarray, sa2: np.ndarray,
                   backward: bool = True) -> np.ndarray:
    """Internal helper."""
    C, N = Y.shape
    if N < _MIN_VALID:
        return Y.copy()

    n = 2
    lam = _ALPHA ** 2 * (n + (3 - n)) - n
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = lam / (n + lam) + (1.0 - _ALPHA ** 2 + _BETA_UT)
    I2 = np.eye(n)

    mf = np.zeros((N, C, n))
    mp = np.zeros((N, C, n))
    Pp = np.zeros((N, C, n, n))
    Cc = np.zeros((N, C, n, n))

    mcur = np.zeros((C, n)); mcur[:, 0] = Y[:, 0]
    v0 = np.maximum(np.var(np.diff(Y, axis=1), axis=1), 1e-6)
    Pcur = np.zeros((C, n, n))
    Pcur[:, 0, 0] = np.maximum(Rt[:, 0], 1e-12)
    Pcur[:, 1, 1] = v0
    mf[0] = mcur

    for k in range(1, N):
        dt = float(t[k] - t[k - 1]) or 1.0
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Qb = np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
        Q = sa2[:, None, None] * Qb
        X = _sigma_points_batch(mcur, Pcur, lam)
        Ys = np.einsum('ij,ckj->cki', F, X)
        m_ = np.einsum('k,ckj->cj', Wm, Ys)
        dY = Ys - m_[:, None, :]
        P_ = np.einsum('k,cki,ckj->cij', Wc, dY, dY) + Q
        P_ = 0.5 * (P_ + P_.transpose(0, 2, 1)) + _JITTER * I2
        dX = X - mcur[:, None, :]
        Cc[k] = np.einsum('k,cki,ckj->cij', Wc, dX, dY)
        mp[k] = m_; Pp[k] = P_
        Xu = _sigma_points_batch(m_, P_, lam)
        Z = Xu[:, :, 0]
        zh = np.einsum('k,ck->c', Wm, Z)
        dZ = Z - zh[:, None]
        S = np.einsum('k,ck->c', Wc, dZ * dZ) + Rt[:, k]
        Cxz = np.einsum('k,cki,ck->ci', Wc, Xu - m_[:, None, :], dZ)
        K = Cxz / S[:, None]
        mcur = m_ + K * (Y[:, k] - zh)[:, None]
        Pcur = P_ - S[:, None, None] * np.einsum('ci,cj->cij', K, K)
        Pcur = 0.5 * (Pcur + Pcur.transpose(0, 2, 1)) + _JITTER * I2
        mf[k] = mcur

    if not backward:
        return mf[:, :, 0].T

    ms = mf.copy()
    for k in range(N - 2, -1, -1):
        G = np.linalg.solve(Pp[k + 1].transpose(0, 2, 1),
                            Cc[k + 1].transpose(0, 2, 1)).transpose(0, 2, 1)
        ms[k] = mf[k] + np.einsum('cij,cj->ci', G, ms[k + 1] - mp[k + 1])

    return ms[:, :, 0].T


def _speed_weight(t: np.ndarray, Y: np.ndarray, beta: float) -> np.ndarray:
    """Internal helper."""
    trans = Y[:3]                                              # (3,N)
    dt = np.maximum(np.diff(t), 1e-6)
    v = np.linalg.norm(np.diff(trans, axis=1), axis=0) / dt    # (N-1,)
    v = np.concatenate([v[:1], v])
    v = uniform_filter1d(v, size=5, mode='nearest')
    vmed = np.median(v) + 1e-9
    return (1.0 + beta * np.maximum(v / vmed - 1.0, 0.0)) ** 2


def _apply_channels(pred_result, smooth_fn):
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = pred_result
    trans = _np(pred_trans); rot = _np(pred_rot)
    hand_pose = _np(pred_hand_pose); betas = _np(pred_betas)
    valid = _np(pred_valid).astype(bool)

    n_t, n_hp, n_bt = 3, hand_pose.shape[2], betas.shape[2]
    for h in range(2):
        idx = np.where(valid[h])[0]
        if len(idx) < _MIN_VALID:
            continue
        t = idx.astype(float)

        quats = Rotation.from_rotvec(rot[h][idx]).as_quat()      # (Nv,4)
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i - 1]) < 0:
                quats[i] = -quats[i]

        Y = np.vstack([trans[h][idx].T, quats.T,
                       hand_pose[h][idx].T, betas[h][idx].T])     # (C,Nv)
        Ys = smooth_fn(t, Y)

        i0 = 0
        trans[h][idx] = Ys[i0:i0 + n_t].T;       i0 += n_t
        q_s = Ys[i0:i0 + 4].T;                   i0 += 4
        hand_pose[h][idx] = Ys[i0:i0 + n_hp].T;  i0 += n_hp
        betas[h][idx] = Ys[i0:i0 + n_bt].T

        norms = np.linalg.norm(q_s, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        rot[h][idx] = Rotation.from_quat(q_s / norms).as_rotvec()

    dtype = pred_valid.dtype if isinstance(pred_valid, np.ndarray) else np.float32
    return [trans, rot, hand_pose, betas, valid.astype(dtype)]


def _resolve_cam_poses(cam_c2w) -> tuple[np.ndarray, np.ndarray] | None:
    """Internal helper."""
    if cam_c2w is None:
        return None
    if isinstance(cam_c2w, (str, Path)):
        p = Path(cam_c2w)
        if not p.exists():
            return None
        try:
            z = np.load(p)
            if 'cam_c2w' not in z:
                return None
            cam_c2w = z['cam_c2w']
        except Exception as e:
            print(f'[clean]  {p}; {e}.')
            return None
    c2w = np.asarray(_np(cam_c2w), dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or len(c2w) < 2:
        print(f'[clean]  {c2w.shape}.')
        return None
    return c2w[:, :3, :3], c2w[:, :3, 3]


def _quat_sign_continuous(q: np.ndarray) -> np.ndarray:
    """Internal helper."""
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def smooth_ukf_cam(pred_result, cam_c2w=None, q: float = 0.6, r: float = 0.6,
                   beta: float = 2.0, rts: float = 1.0):
    """Internal helper."""
    q, r, beta = float(q), float(r), float(beta)
    backward = float(rts) >= 0.5

    cam = _resolve_cam_poses(cam_c2w)
    if cam is None:
        if cam_c2w is not None:
            print('[clean]')
    else:
        print(f'[clean]  {len(cam[0])}.')

    def _smooth_sr(t, Y):
        """Internal helper."""
        sm = _sigma_meas_batch(Y)
        w = _speed_weight(t, Y, beta)
        Rt = (r * sm)[:, None] ** 2 * w[None, :]
        return _urtss_core_sr(t, Y, Rt, (q * sm) ** 2, backward)

    def smooth_fn(t, Y):
        if cam is None:
            return _smooth_sr(t, Y)
        Rm, tv = cam
        fi = np.clip(np.rint(t).astype(int), 0, len(Rm) - 1)
        Rc, tc = Rm[fi], tv[fi]
        RcT = Rc.transpose(0, 2, 1)


        Yc = Y.copy()
        Yc[:3] = np.einsum('nij,nj->ni', RcT, Y[:3].T - tc).T
        Rw = Rotation.from_quat(Y[3:7].T).as_matrix()
        qc = Rotation.from_matrix(
            np.einsum('nij,njk->nik', RcT, Rw)).as_quat()
        Yc[3:7] = _quat_sign_continuous(qc).T


        Ys = _smooth_sr(t, Yc)


        out = Ys.copy()
        out[:3] = (np.einsum('nij,nj->ni', Rc, Ys[:3].T) + tc).T
        qs = Ys[3:7].T
        norms = np.linalg.norm(qs, axis=1, keepdims=True)
        qs = qs / np.where(norms < 1e-8, 1.0, norms)
        qw = Rotation.from_matrix(np.einsum(
            'nij,njk->nik', Rc, Rotation.from_quat(qs).as_matrix())).as_quat()
        out[3:7] = _quat_sign_continuous(qw).T
        return out

    return _apply_channels(pred_result, smooth_fn)
