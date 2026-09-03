#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""世界系 3D payload 构建（网页端 canvas 用，纯 numpy，JSON 安全）。

GT 与 PRED 各调一次 build_world_payload：输入都是「已解算好的世界系 joints + 该侧相机」，
输出前端 plot3d 所需字段（骨架 joints / 手腕轨迹 traj / 骨架连接 conn / 逐帧有效 valid /
逐帧 world→cam 旋转 cam_R / 相机位置 cam_t）。单位统一 cm。

不复用 tools/viewer/render_world.py（其 cam_R/fov_list 变量未定义、且数据源为 lerobot 字段）。
"""
from __future__ import annotations

import numpy as np

from ..reproj_core import geometry as geom


def _round_joints(J: np.ndarray, valid_h: np.ndarray, ndigits: int = 2) -> list:
    """单手 joints (T,N,3) cm → 逐帧 list；无效/含 NaN 帧置 None（前端不画该帧该手）。"""
    out = []
    for f in range(J.shape[0]):
        if not valid_h[f] or not np.all(np.isfinite(J[f])):
            out.append(None)
        else:
            out.append([[round(float(c), ndigits) for c in J[f, j]] for j in range(J.shape[1])])
    return out


def _round_traj(P: np.ndarray, valid_h: np.ndarray, ndigits: int = 2) -> list:
    """单手手腕轨迹 (T,3) cm → 逐帧 list；无效/NaN 帧置 None。"""
    out = []
    for f in range(P.shape[0]):
        p = P[f]
        if not valid_h[f] or not np.all(np.isfinite(p)):
            out.append(None)
        else:
            out.append([round(float(c), ndigits) for c in p])
    return out


def build_world_payload(world: dict, valid_lr: np.ndarray, cam_c2w: np.ndarray,
                        *, fps: float) -> dict:
    """解算单侧（GT 或 PRED）世界系 3D payload。

    world:    hands_to_world 产物 {'left'|'right': {'joints'(T,21,3) m, ...}}。
    valid_lr: (T,2) bool，[左,右] 逐帧有效。
    cam_c2w:  (T,4,4) camera→world。
    返回 JSON 安全 dict（单位 cm；无效帧 None）。
    """
    jl = np.asarray(world["left"]["joints"], dtype=np.float64) * 100.0    # (T,21,3) m→cm
    jr = np.asarray(world["right"]["joints"], dtype=np.float64) * 100.0
    T = jl.shape[0]
    valid_lr = np.asarray(valid_lr, dtype=bool)
    vl, vr = valid_lr[:, 0], valid_lr[:, 1]

    # 手腕轨迹取 21 关节的腕节点（索引 0）。
    traj_l, traj_r = jl[:, 0], jr[:, 0]

    # 逐帧 world→cam 旋转 = inv(cam_c2w)[:3,:3]，行主序 9 元组（前端对齐相机用）。
    cam_c2w = np.asarray(cam_c2w, dtype=np.float64)                       # (T,4,4)
    cam_w2c = np.linalg.inv(cam_c2w)
    cam_R = [[round(float(x), 6) for x in cam_w2c[f][:3, :3].reshape(-1)] for f in range(T)]

    # 逐帧相机在世界中的位置 = cam_c2w[:3,3]（m→cm，与 joints 同单位）。
    # 前端用 cam_R + cam_t 既能把手变到相机系(p_cam = cam_R·(p - cam_t))，又能画相机轨迹/朝向。
    cam_t = [[round(float(c), 2) for c in cam_c2w[f][:3, 3] * 100.0] for f in range(T)]

    return {
        "nframes": int(T),
        "fps": float(fps),
        "joints": [_round_joints(jl, vl), _round_joints(jr, vr)],
        "conn": [list(c) for c in geom.MANO_CONNECTIONS],
        "valid": [vl.tolist(), vr.tolist()],
        "traj": {"left": _round_traj(traj_l, vl), "right": _round_traj(traj_r, vr)},
        "cam_R": cam_R,
        "cam_t": cam_t,
    }
