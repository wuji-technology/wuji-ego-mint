#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable MuJoCo MP4 renderer for the web viewer.

The standalone ``mujoco_view.py`` remains the feature-rich CLI. This module
extracts its render path into a data-source-independent function. The web viewer
uses a fixed third-person camera fitted to the complete motion by default.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


_WEB_FLOOR_CLEARANCE = 0.08
_ROBOT_VIDEO_PRESET = os.environ.get("VIEWER_ROBOT_VIDEO_PRESET", "superfast")

try:
    _ENCODE_BUFFER_FRAMES = max(
        0, int(os.environ.get("VIEWER_ROBOT_ENCODE_BUFFER_FRAMES", "3")))
except ValueError:
    _ENCODE_BUFFER_FRAMES = 3


def _first_valid(mask: np.ndarray) -> int:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    return int(indices[0]) if len(indices) else 0


def _rot_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source) + 1e-9
    target /= np.linalg.norm(target) + 1e-9
    cross = np.cross(source, target)
    cosine = float(np.dot(source, target))
    if np.linalg.norm(cross) < 1e-8:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + cosine))


def _gravity_axis(cam_c2w: np.ndarray, hand_points: np.ndarray) -> np.ndarray:
    cameras = np.asarray(cam_c2w, dtype=float)
    points = np.asarray(hand_points, dtype=float).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    forward = cameras[:, :3, 2].mean(0)
    forward /= np.linalg.norm(forward) + 1e-9
    centers = cameras[:, :3, 3]
    span_all = (float(np.linalg.norm(
        np.percentile(points, 98, axis=0) - np.percentile(points, 2, axis=0)))
        + 1e-9)

    best, best_score = None, None
    for axis in np.concatenate([np.eye(3), -np.eye(3)], axis=0):
        pitch = -np.degrees(np.arcsin(np.clip(float(np.dot(forward, axis)), -1.0, 1.0)))
        if not 5.0 <= pitch <= 60.0:
            continue
        if float(np.dot(centers.mean(0), axis)) <= float(np.percentile(points @ axis, 95)):
            continue
        height = points @ axis
        score = -(float(np.percentile(height, 98) - np.percentile(height, 2)) / span_all)
        if best_score is None or score > best_score:
            best, best_score = axis, score
    if best is not None:
        return best

    # 没有可靠物理判据时也只选世界坐标轴，避免用手点云拟合出每段不同的斜地面。
    camera_up = (-cameras[:, :3, 1]).mean(0)
    axis_index = int(np.argmax(np.abs(camera_up)))
    up = np.zeros(3, dtype=float)
    up[axis_index] = 1.0 if camera_up[axis_index] >= 0 else -1.0
    return up


def _upright_rotation(cam_c2w: np.ndarray, hand_points: np.ndarray) -> np.ndarray:
    # 与 Wuji renderer 的 _gravity_up 同口径：用全段相机 +Y(下)平均值的反向。
    down = np.asarray(cam_c2w, dtype=np.float64)[:, :3, 1]
    down = down[np.all(np.isfinite(down), axis=1)]
    if len(down) and float(np.linalg.norm(down.mean(axis=0))) >= 1e-6:
        up = -down.mean(axis=0)
    else:
        up = _gravity_axis(cam_c2w, hand_points)
    return _rot_align(up, np.array([0.0, 0.0, 1.0]))


def render_world_video(world: dict, cam_c2w: np.ndarray, kept: np.ndarray,
                       output: str | Path, *, fps: float,
                       intrinsics: np.ndarray | None = None,
                       image_size: tuple[int, int] | None = None,
                       width: int = 960, height: int | None = None,
                       view: str = "third",
                       label_text: str | None = None,
                       on_step=None) -> Path:
    """Render one GT or prediction from a fixed full-motion third-person view."""
    from . import draw, mujoco_scene

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cameras = np.asarray(cam_c2w, dtype=np.float64).copy()
    validity = np.asarray(kept, dtype=bool)
    if validity.ndim != 2 or validity.shape[1] != 2:
        raise ValueError(f"kept 应为 [T,2]，实际为 {validity.shape}")

    copied = {}
    for side in ("left", "right"):
        vertices = world[side].get("verts")
        if vertices is None:
            raise RuntimeError("MuJoCo 需要 MANO 网格；当前结果只有 21 点关节")
        copied[side] = {
            "verts": np.asarray(vertices, dtype=np.float64).copy(),
            "joints": np.asarray(world[side]["joints"], dtype=np.float64).copy(),
        }
    frames = len(cameras)
    if frames <= 0 or validity.shape[0] != frames:
        raise ValueError("MuJoCo 相机、手部与有效掩码帧数不一致")
    K = None if intrinsics is None else np.asarray(intrinsics, dtype=np.float64)
    if K is not None and (K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError(f"相机内参无效: {K.shape}")
    if image_size is None:
        source_size = None
    else:
        source_size = tuple(map(int, image_size))
        if len(source_size) != 2 or min(source_size) <= 0:
            raise ValueError(f"原视频尺寸无效: {image_size}")
    if view == "ego" and (K is None or source_size is None):
        raise ValueError("视频视角 MuJoCo 渲染需要相机内参和原视频尺寸")
    render_width = max(2, int(width))
    render_width += render_width % 2
    if height is None:
        if source_size is None:
            render_height = 540
        else:
            render_height = max(2, int(round(render_width * source_size[1] / source_size[0])))
            render_height += render_height % 2
    else:
        render_height = max(2, int(height))

    hand_points = np.concatenate([
        copied["left"]["joints"].reshape(-1, 3),
        copied["right"]["joints"].reshape(-1, 3),
    ], axis=0)
    upright, origin, cameras = mujoco_scene.canonicalize_robot_coordinates(
        cameras, hand_points)
    for side in ("left", "right"):
        copied[side]["verts"] = copied[side]["verts"] @ upright.T
        copied[side]["joints"] = copied[side]["joints"] @ upright.T
    for side in ("left", "right"):
        copied[side]["verts"] -= origin
        copied[side]["joints"] -= origin

    vertices_left = copied["left"]["verts"]
    vertices_right = copied["right"]["verts"]
    faces_right, faces_left = draw.get_faces()
    scene = mujoco_scene.HandWorldScene(
        faces_left, faces_right,
        vertices_left[_first_valid(validity[:, 0])],
        vertices_right[_first_valid(validity[:, 1])],
        width=render_width, height=render_height, floor=True,
        operation_mat=False, ego_floor_screen_level=True,
    )
    writer = None
    try:
        fit_parts = []
        for hand_index, side in enumerate(("left", "right")):
            if validity[:, hand_index].any():
                fit_parts.append(
                    copied[side]["joints"][validity[:, hand_index]].reshape(-1, 3))
        hand_finite = (np.concatenate(fit_parts, axis=0) if fit_parts else
                       np.concatenate([
                           copied["left"]["joints"].reshape(-1, 3),
                           copied["right"]["joints"].reshape(-1, 3),
                       ], axis=0))
        hand_finite = hand_finite[np.all(np.isfinite(hand_finite), axis=1)]

        forward = cameras[:, :3, 2]
        forward = forward[np.all(np.isfinite(forward), axis=1)]
        mean_forward = forward.mean(0) if len(forward) else np.array([1.0, 0.0, 0.0])
        forward_xy = mean_forward[:2]
        forward_azimuth = (float(np.arctan2(forward_xy[1], forward_xy[0]))
                           if np.linalg.norm(forward_xy) > 1e-6 else 0.0)
        # Web 画面不放桌面/操作垫；地面固定为世界 XY 水平面，并与手留出距离。
        # Use hands for the contact height, but include the full camera path
        # when sizing the shared ground so MuJoCo and Wuji show the same extent.
        scene.place_floor(
            hand_finite, margin=_WEB_FLOOR_CLEARANCE, fwd_az=None,
            extent_points=np.concatenate([
                hand_finite,
                cameras[:, :3, 3][np.all(np.isfinite(cameras[:, :3, 3]), axis=1)],
            ], axis=0))
        scene.fit_fixed_third_camera(
            np.asarray([0.0, 0.0, 1.0]), hand_finite, cameras)

        base = float(np.degrees(forward_azimuth))
        views = {
            "third": (base + 205.0, -22.0),
            "front": (base + 180.0, -22.0),
            "back": (base, -22.0),
            "right": (base + 90.0, -22.0),
            "left": (base - 90.0, -22.0),
            "top": (base, -88.0),
        }
        if view not in views:
            if view != "ego":
                raise ValueError(f"未知 MuJoCo 视角: {view}")
            azimuth = elevation = 0.0
        else:
            azimuth, elevation = views[view]
        fovy_deg = (None if K is None or source_size is None else
                    float(np.degrees(2.0 * np.arctan(
                        (source_size[1] / 2.0) / float(K[1, 1])))))
        writer = draw.H264PipeWriter(
            output, float(fps), (render_width, render_height),
            preset=_ROBOT_VIDEO_PRESET,
            buffered_frames=_ENCODE_BUFFER_FRAMES)
        if view != "ego":
            # The web renderer uses one fixed ground pose for the whole clip.
            # Avoid restoring it and running mj_forward again on every frame.
            scene.set_floor_ground()
        for frame in range(frames):
            scene.bake_frame(
                vertices_left[frame], vertices_right[frame],
                bool(validity[frame, 0]), bool(validity[frame, 1]),
            )
            if view == "ego":
                scene.hide_ghosts()
                scene.set_floor_ego_ground(cameras[frame], fovy_deg)
                scene.set_ego(cameras[frame], fovy_deg)
                image = scene.render_ego(K, source_size)
            elif view == "third":
                image = scene.render_fixed_third(frame)
            else:
                image = scene.render_free(
                    azimuth, elevation, None, frame, draw_trail=False)
            image_bgr = np.ascontiguousarray(image[:, :, ::-1])
            if label_text:
                draw.label(image_bgr, str(label_text))
            writer.write(image_bgr)
            if on_step is not None:
                on_step(frame + 1, frames)
        writer.close()
        writer = None
    finally:
        if writer is not None:
            writer.close()
        scene.close()
    return output
