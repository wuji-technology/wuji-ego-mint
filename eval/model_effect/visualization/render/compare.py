#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT / 预测 的共用解算与 2D 重投影 overlay 渲染。

hand_reproj.py（离线导出 mp4）与 viewer_web.py（网页端按需渲染缓存）共用本模块，避免重复：
  · 预测 hand[T,218] → 与 GT 同 schema 的双手 6D；
  · 6D → 世界系 MANO verts/joints；
  · 单帧 sides 组装 + RGB→BGR；
  · render_compare_overlay：lerobot 单 episode「GT｜PRED 并排」mp4；
  · render_pred_overlay：裸视频「仅预测」mp4。
两个 overlay 写完就地转 H.264，返回 out_path。
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..reproj_core import geometry as geom
from . import draw

# 每手 218→2×109 的切片（与 data/lerobot_v3.py 的 cat 顺序一致）。
_PER_HAND = 109
_HAND_SLICES = {"transl_cam": (0, 3), "orient6d": (3, 9), "pose6d": (9, 99), "betas": (99, 109)}
CACHE_TAG = "allpred_2d_v6_explicit_presence"
_RENDER_WORKERS = max(1, int(os.environ.get("VIEWER_RENDER_WORKERS") or min(2, os.cpu_count() or 2)))
_RENDER_INFLIGHT = max(
    _RENDER_WORKERS,
    int(os.environ.get("VIEWER_RENDER_INFLIGHT") or (_RENDER_WORKERS * 2)),
)


def _write_rendered_frames(vw, total: int, render_one, *, on_step=None,
                           progress: bool = True, label: str = "渲染") -> None:
    """Draw frames in parallel while writing them to ffmpeg in exact input order."""
    total = int(total)
    if total <= 0:
        return
    step = max(1, total // 10)

    def _report(done):
        if on_step is not None:
            try:
                on_step(done, total)
            except Exception:  # noqa: BLE001
                pass
        if progress and (done % step == 0 or done == total):
            print(f"\r[compare] {label} {100 * done // total:3d}% ({done}/{total})",
                  end="\n" if done == total else "", flush=True)

    if _RENDER_WORKERS == 1 or total == 1:
        for frame_index in range(total):
            vw.write(render_one(frame_index))
            _report(frame_index + 1)
        return

    workers = min(_RENDER_WORKERS, total)
    inflight = min(total, max(workers, _RENDER_INFLIGHT))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="frame-render") as pool:
        pending = {index: pool.submit(render_one, index) for index in range(inflight)}
        next_submit = inflight
        for frame_index in range(total):
            frame = pending.pop(frame_index).result()
            if next_submit < total:
                pending[next_submit] = pool.submit(render_one, next_submit)
                next_submit += 1
            vw.write(frame)
            _report(frame_index + 1)


def pred_hand_to_schema(hand_218: np.ndarray) -> dict:
    """预测 hand[T,218] → {'left'|'right': {transl_cam,orient6d,pose6d,betas}}（与 GT 同 schema）。"""
    out = {}
    for base, side in ((0, "left"), (_PER_HAND, "right")):
        seg = hand_218[:, base:base + _PER_HAND]
        out[side] = {k: seg[:, lo:hi].astype(np.float32) for k, (lo, hi) in _HAND_SLICES.items()}
    return out


def hands_to_world(hands: dict, cam_c2w=None, hand_frame: str = "world",
                   betas_mean: bool = False) -> dict:
    """{'left'|'right': 6D schema} → {'left'|'right': {'verts'(T,778,3), 'joints'(T,21,3)}}。

    hand_frame='camera' 时(数据/预测的手为相机系),先用 cam_c2w 把手 6D 逆变换回世界系
    (geom.hand6d_cam_to_world),再 decode+MANO;使下游 world 系渲染/3D 逻辑完全不变。
    'world'(默认,旧数据)则原样解算。cam_c2w:(T,4,4) camera→world,相机系时必传。
    betas_mean=True 时先把每手 betas 换成**整段平均**(广播回每帧),去逐帧手形抖动。
    """
    from ..reproj_core import mano
    if betas_mean:   # 手形 betas 逐帧抖(手不该变大变小) → 用整段均值广播回每帧，去抖
        hands = {s: {**h, "betas": np.repeat(np.asarray(h["betas"]).mean(0, keepdims=True),
                                             np.asarray(h["betas"]).shape[0], axis=0)}
                 for s, h in hands.items()}
    if hand_frame == "camera":
        if cam_c2w is None:
            raise ValueError("hands_to_world: hand_frame='camera' 需传 cam_c2w 才能转回 world")
        hands = geom.hand6d_cam_to_world(hands, cam_c2w)
    out = {}
    for side in ("left", "right"):
        h = hands[side]
        is_right = side == "right"
        dec = mano.decode_hand_6d(h["transl_cam"], h["orient6d"], h["pose6d"], h["betas"], is_right)
        verts, joints = mano.run_mano(dec["trans"], dec["rot"], dec["hand_pose"], dec["betas"], is_right)
        out[side] = {"verts": verts, "joints": joints[:, :21]}
    return out


def kpt21_cam_to_world(kpt21: np.ndarray, cam_c2w: np.ndarray) -> dict:
    """Camera-frame OpenPose 21 points [T,2,21,3] -> render world schema."""
    points = np.asarray(kpt21, dtype=np.float32)
    cameras = np.asarray(cam_c2w, dtype=np.float32)
    if points.ndim != 4 or points.shape[1:] != (2, 21, 3):
        raise ValueError(f"kpt21_gt 应为 [T,2,21,3],实际为 {points.shape}")
    if cameras.shape != (points.shape[0], 4, 4):
        raise ValueError(f"cam_c2w 应为 [{points.shape[0]},4,4],实际为 {cameras.shape}")
    rotation = cameras[:, :3, :3]
    translation = cameras[:, :3, 3]
    world = np.einsum("tij,thkj->thki", rotation, points) + translation[:, None, None, :]
    return {
        side: {"verts": None, "joints": world[:, side_index]}
        for side_index, side in enumerate(("left", "right"))
    }


def gt_to_world(raw: dict, *, betas_mean: bool = False) -> dict:
    """Decode whichever GT source is available: MANO parameters first, then kp21."""
    hands = raw.get("hands")
    if hands:
        return hands_to_world(
            hands, raw["cam_c2w"], raw.get("hand_frame", "world"),
            betas_mean=betas_mean,
        )
    if raw.get("kpt21_gt") is not None:
        return kpt21_cam_to_world(raw["kpt21_gt"], raw["cam_c2w"])
    raise RuntimeError("当前 episode 既没有 MANO GT,也没有 21 点 GT")


def panel_sides(world: dict, i: int, valid_l: bool, valid_r: bool) -> dict:
    """取第 i 帧、组装 render_frame 需要的 sides dict。"""
    return {
        side: {
            "verts": None if world[side].get("verts") is None else world[side]["verts"][i],
            "joints": world[side]["joints"][i],
            "valid": bool(valid),
        }
        for side, valid in (("left", valid_l), ("right", valid_r))
    }


def to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame_rgb[:, :, ::-1])


def predicted_presence(pred: dict, length: int) -> np.ndarray | None:
    """Return thresholded left/right predictions as ``[T,2]``, if available."""
    if pred.get("hand_presence_logits") is not None:
        values = np.asarray(pred["hand_presence_logits"], dtype=np.float32)
        threshold = 0.0
    elif pred.get("hand_confidence") is not None:
        # Compatibility for old cached predictions that did not persist logits.
        values = np.asarray(pred["hand_confidence"], dtype=np.float32)
        threshold = 0.5
    else:
        return None
    if values.shape != (length, 2):
        return None
    return values >= threshold


def prediction_render_mask(pred: dict, length: int) -> np.ndarray:
    """Return the per-hand draw mask, keeping legacy checkpoints fully visible."""
    presence = predicted_presence(pred, length)
    if presence is None:
        return np.ones((length, 2), dtype=bool)
    return presence


def _presence_at(presence: np.ndarray | None, frame_idx: int):
    if presence is None:
        return None, None
    return bool(presence[frame_idx, 0]), bool(presence[frame_idx, 1])


def render_compare_overlay(raw: dict, pred: dict, out_path, *,
                           mode: str = "mesh_skel", alpha: float = 0.6,
                           fps: float = 30.0, progress: bool = True) -> Path:
    """lerobot 单 episode「GT｜PRED 并排」overlay mp4，写完转 H.264，返回 out_path。

    raw:  load_episode_raw 产物（frames/cam_c2w/K/kept/hands）。
    pred: predictor.predict 产物（pose_enc[T,9]，可选 hand[T,218]）。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faces_right, faces_left = draw.get_faces()
    faces_lr = (faces_left, faces_right)

    frames = raw["frames"]                       # (T,H,W,3) RGB
    T, H, W = frames.shape[:3]
    gt_c2w = raw["cam_c2w"]
    gt_world = gt_to_world(raw)
    gt_kept = raw["kept"]                         # (T,2)
    has_hand = "hand" in pred
    pred_c2w, pred_K = geom.decode_camera_pose_enc(pred["pose_enc"], H, W)
    # 模型的 hand 输出定义在预测相机系；PRED 路径从解算到投影只使用预测值。
    pred_world = (hands_to_world(
        pred_hand_to_schema(pred["hand"]), pred_c2w, "camera"
    ) if has_hand else None)
    pred_kept = predicted_presence(pred, T)
    pred_render = prediction_render_mask(pred, T)

    sep = 6
    vw = draw.H264PipeWriter(out_path, float(fps), (W * 2 + sep, H))
    step = max(1, T // 10)
    for i in range(T):
        base_bgr = to_bgr(frames[i])
        # 左：GT（GT 相机 + GT 手，按逐帧 kept）
        gt_sides = panel_sides(gt_world, i, gt_kept[i, 0], gt_kept[i, 1])
        left_panel = draw.render_frame(base_bgr.copy(), raw["cam_c2w"][i], raw["K"],
                                       gt_sides, faces_lr, mode=mode, alpha=alpha)
        draw.label(left_panel, "GT")
        draw.presence_label(left_panel, [("GT", gt_kept[i, 0], gt_kept[i, 1])])
        # 右：预测（预测相机 + 预测手，presence=N 的手不画）
        if pred_world is not None:
            pr_sides = panel_sides(
                pred_world, i, pred_render[i, 0], pred_render[i, 1]
            )
            right_panel = draw.render_frame(base_bgr.copy(), pred_c2w[i], pred_K,
                                            pr_sides, faces_lr, mode=mode, alpha=alpha)
        else:
            right_panel = base_bgr.copy()
        draw.label(right_panel, "PRED")
        draw.presence_label(right_panel, [("PRED", *_presence_at(pred_kept, i))])

        canvas = np.full((H, W * 2 + sep, 3), 32, dtype=np.uint8)
        canvas[:, :W] = left_panel
        canvas[:, W + sep:] = right_panel
        vw.write(canvas)
        if progress and ((i + 1) % step == 0 or i + 1 == T):
            print(f"[compare] 渲染 {i + 1}/{T}", flush=True)
    vw.close()
    return out_path


def render_2d(raw: dict, pred: dict, out_path, *,
              mode: str = "mesh_skel", alpha: float = 0.6, fps: float = 30.0,
              progress: bool = True, layout: str = "overlay", content: str = "both",
              on_step=None, gt_betas_mean: bool = False,
              pred_betas_mean: bool = False, pred_fov_mean: bool = False,
              gt_world_data=None, pred_world_data=None) -> Path:
    """lerobot 单 episode 端到端 2D 重投影，写完转 H.264。

    GT 路只使用 GT 手、存在性、相机外参和内参；PRED 路只使用对应预测值。layout：
    overlay=同一画面叠加（GT 绿 / PRED 红）；side=左右并排。模型无 hand 输出时只画 GT。
    """
    if content != "both":
        raise ValueError(f"2D 重投影只支持端到端 content='both'，收到 {content!r}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faces_right, faces_left = draw.get_faces()
    faces_lr = (faces_left, faces_right)

    frames = raw["frames"]                       # (T,H,W,3) RGB
    T, H, W = frames.shape[:3]
    gt_c2w, gt_K = raw["cam_c2w"], raw["K"]
    gt_world = (gt_world_data if gt_world_data is not None
                else gt_to_world(raw, betas_mean=gt_betas_mean))
    gt_kept = raw["kept"]                         # (T,2)
    has_hand = "hand" in pred
    pred_c2w, pred_K = geom.decode_camera_pose_enc(pred["pose_enc"], H, W, fov_mean=pred_fov_mean)
    pred_world = pred_world_data
    if pred_world is None and has_hand:
        pred_world = hands_to_world(
            pred_hand_to_schema(pred["hand"]), pred_c2w, "camera",
            betas_mean=pred_betas_mean,
        )
    pred_kept = predicted_presence(pred, T)
    pred_render = prediction_render_mask(pred, T)

    # 每路 source = (world, c2w序列, K, valid_fn(i)->(vl,vr), 标签)。
    def gt_kept_fn(i):  return (bool(gt_kept[i, 0]), bool(gt_kept[i, 1]))
    def pred_kept_fn(i): return (bool(pred_render[i, 0]), bool(pred_render[i, 1]))
    A = (gt_world, gt_c2w, gt_K, gt_kept_fn, "GT")
    B = (pred_world, pred_c2w, pred_K, pred_kept_fn, "PRED") if pred_world is not None else None

    def draw_src(panel, src, i, palette):
        world, c2w, K, vfn, _lbl = src
        vl, vr = vfn(i)
        return draw.render_frame(panel, c2w[i], K, panel_sides(world, i, vl, vr),
                                 faces_lr, mode=mode, alpha=alpha, palette=palette)

    sep = 6
    size = (W * 2 + sep, H) if layout == "side" else (W, H)
    vw = draw.H264PipeWriter(out_path, float(fps), size)

    def render_one(i):
        base = to_bgr(frames[i])
        if layout == "side":
            left = draw_src(base, A, i, None); draw.label(left, A[4])
            draw.presence_label(left, [("GT", gt_kept[i, 0], gt_kept[i, 1])])
            if B is not None:
                right = draw_src(base, B, i, None); draw.label(right, B[4])
            else:
                right = base.copy(); draw.label(right, "-")
            draw.presence_label(right, [("PRED", *_presence_at(pred_kept, i))])
            canvas = np.full((H, W * 2 + sep, 3), 32, dtype=np.uint8)
            canvas[:, :W] = left; canvas[:, W + sep:] = right
            return canvas
        else:
            panel = draw_src(base, A, i, draw.PALETTE_GT)
            if B is not None:
                panel = draw_src(panel, B, i, draw.PALETTE_PRED)
            draw.label(panel, f"{A[4]}(green) | {(B[4] if B else '-')}(red)")
            draw.presence_label(panel, [
                ("GT", gt_kept[i, 0], gt_kept[i, 1]),
                ("PRED", *_presence_at(pred_kept, i)),
            ])
            return panel

    _write_rendered_frames(
        vw, T, render_one, on_step=on_step, progress=progress,
        label=f"2D {content}/{layout} 渲染",
    )
    vw.close()
    return out_path


def render_gt_overlay(raw: dict, out_path, *,
                      mode: str = "mesh_skel", alpha: float = 0.6,
                      fps: float = 30.0, progress: bool = True, on_step=None,
                      betas_mean: bool = False, gt_world_data=None,
                      label_text: str = "GT",
                      presence_text: str | None = None) -> Path:
    """单 episode「仅 GT」overlay mp4（GT 手 + GT 相机，不涉及任何模型预测），写完转 H.264。

    供网页端「仅原始数据」模式用：只看原始标注的手/相机重投影效果，无需 ckpt / 不跑推理。
    手为相机系(hand_frame='camera')时用 GT 真实相机转回 world，与 render_2d 的 GT 侧一致。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faces_right, faces_left = draw.get_faces()
    faces_lr = (faces_left, faces_right)

    frames = raw["frames"]                        # (T,H,W,3) RGB
    T, H, W = frames.shape[:3]
    gt_c2w, gt_K = raw["cam_c2w"], raw["K"]
    hf = raw.get("hand_frame", "world")
    gt_world = (gt_world_data if gt_world_data is not None
                else gt_to_world(raw, betas_mean=betas_mean))
    gt_kept = raw["kept"]                          # (T,2)

    vw = draw.H264PipeWriter(out_path, float(fps), (W, H))

    def render_one(i):
        sides = panel_sides(gt_world, i, gt_kept[i, 0], gt_kept[i, 1])
        panel = draw.render_frame(to_bgr(frames[i]), gt_c2w[i], gt_K, sides,
                                  faces_lr, mode=mode, alpha=alpha)
        draw.label(panel, label_text)
        presence_name = label_text if presence_text is None else presence_text
        draw.presence_label(panel, [(presence_name, gt_kept[i, 0], gt_kept[i, 1])])
        return panel

    _write_rendered_frames(
        vw, T, render_one, on_step=on_step, progress=progress,
        label="GT-only 渲染",
    )
    vw.close()
    return out_path


def render_pred_overlay(frames: np.ndarray, pred: dict, out_path, *,
                        mode: str = "mesh_skel", alpha: float = 0.6,
                        fps: float = 30.0, progress: bool = True,
                        hand_frame: str = "world", on_step=None,
                        betas_mean: bool = False, fov_mean: bool = False,
                        pred_world_data=None) -> Path:
    """裸视频「仅预测」overlay mp4（无 GT），写完转 H.264，返回 out_path。

    hand_frame='camera' 时预测手为相机系;裸视频无 GT 相机,只能用预测相机(pred_c2w)转回 world。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faces_right, faces_left = draw.get_faces()
    faces_lr = (faces_left, faces_right)

    T, H, W = frames.shape[:3]
    pred_c2w, pred_K = geom.decode_camera_pose_enc(pred["pose_enc"], H, W, fov_mean=fov_mean)
    pred_world = (
        pred_world_data if pred_world_data is not None
        else hands_to_world(
            pred_hand_to_schema(pred["hand"]), pred_c2w, hand_frame,
            betas_mean=betas_mean,
        )
    )
    pred_kept = predicted_presence(pred, T)
    pred_render = prediction_render_mask(pred, T)

    vw = draw.H264PipeWriter(out_path, float(fps), (W, H))

    def render_one(i):
        base_bgr = to_bgr(frames[i])
        sides = panel_sides(
            pred_world, i, pred_render[i, 0], pred_render[i, 1]
        )
        panel = draw.render_frame(base_bgr, pred_c2w[i], pred_K, sides, faces_lr, mode=mode, alpha=alpha)
        draw.label(panel, "PRED")
        draw.presence_label(panel, [("", *_presence_at(pred_kept, i))])
        return panel

    _write_rendered_frames(
        vw, T, render_one, on_step=on_step, progress=progress,
        label="PRED 渲染",
    )
    vw.close()
    return out_path
