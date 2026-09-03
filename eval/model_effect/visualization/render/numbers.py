#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐帧数值(GT vs PRED)：相机位姿 + 内参 FoV + 手腕位姿 + 手形 betas，供网页「整体」块下方综合数字表按帧查。

设计与 render/metrics.py 一致：一次算好逐帧数组，前端按当前帧取值渲染；另给整段平均值(betas/fov)供固定对比。
- 手/相机位姿为**世界系**(整体块语义：GT×GT 相机、PRED×PRED 相机)。手形 betas、内参 fov 与坐标系无关。
- 旋转矩阵不直观 → 一律转**欧拉角(度, 'xyz')**；FoV 弧度→度。手只给手腕平移/朝向 + betas，不含手指 pose。
- 无效帧(kept 掩码 False / 值非有限)置 None，保证 JSON 安全(不产生 NaN)。

依赖惰性 import(与 viewer 其它模块方法内 import 惯例一致)：reproj_core.geometry / render.compare / scipy。
"""
from __future__ import annotations

import numpy as np

_RAD2DEG = 180.0 / np.pi


def _euler_deg(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 (T,3,3) → 欧拉角度数 (T,3) 'xyz'。非有限帧(无效)占位单位阵，结果由 valid 置 None。"""
    from scipy.spatial.transform import Rotation
    R = np.asarray(R, np.float64).copy()
    bad = ~np.isfinite(R).all(axis=(-2, -1))
    R[bad] = np.eye(3)
    return Rotation.from_matrix(R).as_euler("xyz", degrees=True)


def _pos_eul_rows(pos_cm: np.ndarray, eul_deg: np.ndarray, valid) -> dict:
    """(T,3) 位置cm + (T,3) 欧拉度 + (T,)valid → {pos:[..|None], eul:[..|None]}；无效/非有限帧 None。"""
    T = pos_cm.shape[0]
    pos, eul = [], []
    for f in range(T):
        ok = ((valid is None or bool(valid[f]))
              and np.isfinite(pos_cm[f]).all() and np.isfinite(eul_deg[f]).all())
        if ok:
            pos.append([round(float(x), 2) for x in pos_cm[f]])
            eul.append([round(float(x), 1) for x in eul_deg[f]])
        else:
            pos.append(None)
            eul.append(None)
    return {"pos": pos, "eul": eul}


def _fov_rows(fov_rad) -> list:
    """(T,2) 弧度 FoV → 每帧 [度,度]|None（非有限置 None）。"""
    fov_deg = np.asarray(fov_rad, np.float64) * _RAD2DEG
    return [([round(float(x), 2) for x in fov_deg[f]] if np.isfinite(fov_deg[f]).all() else None)
            for f in range(fov_deg.shape[0])]


def _cam_block(c2w, fov_rad=None) -> dict:
    """相机 c2w (T,4,4) → {pos:世界系cm, eul:c2w 欧拉度, fov:每帧[度,度]|None}。"""
    c2w = np.asarray(c2w, np.float64)
    rec = _pos_eul_rows(c2w[:, :3, 3] * 100.0, _euler_deg(c2w[:, :3, :3]), None)
    rec["fov"] = _fov_rows(fov_rad) if fov_rad is not None else [None] * c2w.shape[0]
    return rec


def _hand_block(hands: dict, valid_lr) -> dict:
    """{left/right:{transl_cam,orient6d,betas,...}} → {left/right:{pos(cm),eul(°),betas}}(每帧原始值)。
    transl 米→cm；orient6d→矩阵→欧拉度；betas 每帧原样。无效帧(valid_lr[:,side])置 None。"""
    from ..reproj_core import geometry as geom
    out = {}
    for si, side in enumerate(("left", "right")):
        h = hands[side]
        pos_cm = np.asarray(h["transl_cam"], np.float64) * 100.0
        eul = _euler_deg(geom.rot6d_to_mat(np.asarray(h["orient6d"], np.float32)))
        betas = np.asarray(h["betas"], np.float64)
        v = valid_lr[:, si] if valid_lr is not None else None
        rec = _pos_eul_rows(pos_cm, eul, v)
        T = pos_cm.shape[0]
        rec["betas"] = [
            ([round(float(b), 3) for b in betas[f]]
             if ((v is None or bool(v[f])) and np.isfinite(betas[f]).all()) else None)
            for f in range(T)
        ]
        out[side] = rec
    return out


def _hand_world_to_cam(hands: dict, cam_c2w: np.ndarray) -> dict:
    """世界系双手 6D → 相机系（reproj_core.geometry.hand6d_cam_to_world 的逆），betas/pose 原样。
    orient_cam = R_c2w^T · orient_world；transl_cam = R_c2w^T · (transl_world - t_c2w)。
    仅用于少见的 hand_frame!='camera'（GT 已是世界系）时反推相机系；主流程直接用原始相机系值。"""
    from ..reproj_core import geometry as geom
    c2w = np.asarray(cam_c2w, np.float32)
    Rcw, tcw = c2w[:, :3, :3], c2w[:, :3, 3]                # (T,3,3),(T,3)
    out = {}
    for side in ("left", "right"):
        h = hands[side]
        o_world = geom.rot6d_to_mat(np.asarray(h["orient6d"], np.float32))   # (T,3,3)
        o_cam = np.einsum("tji,tjk->tik", Rcw, o_world)                      # R^T · o_world
        t_cam = np.einsum("tji,tj->ti", Rcw, np.asarray(h["transl_cam"], np.float32) - tcw)
        out[side] = {
            "transl_cam": t_cam.astype(np.float32),
            "orient6d": geom.mat_to_6d(o_cam).astype(np.float32),
            "pose6d": np.asarray(h["pose6d"], np.float32),
            "betas": np.asarray(h["betas"], np.float32),
        }
    return out


def _mean_betas(hands) -> dict | None:
    """{left/right:{betas(T,10)}} → {left:[10],right:[10]} 整段平均。"""
    if not hands:
        return None
    out = {}
    for s in ("left", "right"):
        m = np.asarray(hands[s]["betas"], np.float64).mean(0)
        out[s] = [round(float(x), 3) for x in m] if np.isfinite(m).all() else None
    return out


def _mean_fov(fov_rad) -> list | None:
    """(T,2) 弧度 → [度,度] 整段平均。"""
    if fov_rad is None:
        return None
    m = np.asarray(fov_rad, np.float64).mean(0) * _RAD2DEG
    return [round(float(x), 2) for x in m] if np.isfinite(m).all() else None


def frame_numbers(raw: dict, pred: dict, decode) -> dict:
    """综合逐帧数值 dict(JSON 安全，无效帧 None)，供网页两个独立面板按帧查：
      cam:      {gt:{pos,eul,fov}|None, pred:{pos,eul,fov}|None}          —— 世界系(cm/°)，fov=度
      hand:     {left/right:{gt:{pos,eul,betas}|None, pred:{...}|None}}   —— 世界系(GT×GT相机、PRED×PRED相机)
      hand_cam: {left/right:{gt:{pos,eul,betas}|None, pred:{...}|None}}   —— 相机系(手腕原始 transl_cam/orient6d)
      mean:     {gt_fov,pred_fov, gt_betas:{l,r}, pred_betas:{l,r}}       —— 整段平均(固定)
    「逐帧数值（世界系）」面板用 cam+hand+mean；「逐帧数值（相机系）」面板用 hand_cam(+betas 平均)。
    decode = reproj_core.geometry.decode_camera_pose_enc(pose_enc,H,W)→(c2w,K)。
    """
    from . import compare
    from ..reproj_core import geometry as geom

    frames = raw["frames"]
    H, W = int(frames.shape[1]), int(frames.shape[2])
    kept = raw.get("kept")
    valid_lr = np.asarray(kept, bool) if kept is not None else None

    # ---- 相机(pos/eul 世界系 + fov)：GT 用 raw["cam_c2w"]/raw["cam_pose_enc"]，PRED 用 pred["pose_enc"] ----
    gt_c2w = raw.get("cam_c2w")
    gt_pe = raw.get("cam_pose_enc")                        # [T,9] 或 None(裸集/旧数据)
    pred_pe = pred.get("pose_enc")
    pred_c2w = decode(pred_pe, H, W)[0] if pred_pe is not None else None
    gt_fov = np.asarray(gt_pe)[:, 7:9] if gt_pe is not None else None
    pred_fov = np.asarray(pred_pe)[:, 7:9] if pred_pe is not None else None
    cam = {
        "gt": _cam_block(gt_c2w, gt_fov) if gt_c2w is not None else None,
        "pred": _cam_block(pred_c2w, pred_fov) if pred_c2w is not None else None,
    }

    # ---- 手(世界系)：GT×GT相机、PRED×PRED相机 ----
    hand_frame = raw.get("hand_frame", "camera")
    gt_hands = raw.get("hands")
    pred_hands = compare.pred_hand_to_schema(pred["hand"]) if pred.get("hand") is not None else None
    gt_w = None
    if gt_hands:
        gh = (geom.hand6d_cam_to_world(gt_hands, gt_c2w)
              if (hand_frame == "camera" and gt_c2w is not None) else gt_hands)
        gt_w = _hand_block(gh, valid_lr)
    pred_w = (_hand_block(geom.hand6d_cam_to_world(pred_hands, pred_c2w), None)
              if (pred_hands is not None and pred_c2w is not None) else None)
    hand = {side: {"gt": (gt_w[side] if gt_w else None), "pred": (pred_w[side] if pred_w else None)}
            for side in ("left", "right")}

    # ---- 手(相机系)：手腕原始 transl_cam/orient6d（未转世界）。PRED 天然相机系；
    #      GT 在 hand_frame=='camera' 时即原始相机系，否则(已是世界系)做 world→cam 逆推。----
    gt_cam_hands = None
    if gt_hands:
        if hand_frame == "camera":
            gt_cam_hands = gt_hands
        elif gt_c2w is not None:
            gt_cam_hands = _hand_world_to_cam(gt_hands, gt_c2w)
    gt_c = _hand_block(gt_cam_hands, valid_lr) if gt_cam_hands is not None else None
    pred_c = _hand_block(pred_hands, None) if pred_hands is not None else None
    hand_cam = {side: {"gt": (gt_c[side] if gt_c else None), "pred": (pred_c[side] if pred_c else None)}
                for side in ("left", "right")}

    # ---- 整段平均(固定，与"每帧"列并排对比) ----
    mean = {"gt_fov": _mean_fov(gt_fov), "pred_fov": _mean_fov(pred_fov),
            "gt_betas": _mean_betas(gt_hands), "pred_betas": _mean_betas(pred_hands)}

    return {"cam": cam, "hand": hand, "hand_cam": hand_cam, "mean": mean}
