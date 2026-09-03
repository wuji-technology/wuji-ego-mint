#!/usr/bin/env python3
"""Render the browser fixed-world Canvas view as a timeline-aligned MP4."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import draw


CACHE_TAG = "fixed_world_canvas_v7_clean_export_hud"
DEFAULT_SIZE = (960, 540)
COORD_MODES = ("z_up", "opencv")
DEFAULT_COORD_MODE = "z_up"
Z_UP_FROM_CV = np.asarray([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)
MAX_TRAJECTORY_POINTS = 1000
# 内部按 SUPERSAMPLE× 绘制、写帧前 INTER_AREA 缩回：线条/文字明显更锐（论文宣传片用）。
SUPERSAMPLE = 2
# 默认 3/4 俯视（弧度），与前端 app.js 的 newView() 必须一致；纯正视图读不出纵深。
DEFAULT_AZ, DEFAULT_EL = -0.61, 0.31
GRID_SPAN = 2.3             # 网格盘半径 = GRID_SPAN × 场景半径
GRID_TARGET_CELLS = 10      # 2×场景半径 目标覆盖格数（再取 1/2/5×10^k 的整齐步长）
GRID_MAJOR_EVERY = 5
GROUND_CLEARANCE = 0.28     # 地面放在最低手点下方 max(15cm, GROUND_CLEARANCE×半径)
SHADOW_ALPHA = 0.55
HUD_ALPHA = 0.78
AXIS_ALPHA = 0.28
AXIS_LABEL_ALPHA = 0.68


def _bgr(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (4, 2, 0))


BG_CENTER = np.asarray(_bgr("#16262b"), dtype=np.float32)
BG_EDGE = np.asarray(_bgr("#05090b"), dtype=np.float32)
BG_MIX = tuple(float(x) for x in (BG_CENTER * 0.45 + BG_EDGE * 0.55))
TEXT = _bgr("#e6edf3")
MUTED = _bgr("#8b98a8")
OUTLINE = _bgr("#05070a")
LABEL_BG = _bgr("#0b0f15")
CHIP_BG = _bgr("#080d11")
GRID_MINOR = _bgr("#3d5b66")
GRID_MAJOR = _bgr("#6d95a3")
SHADOW = _bgr("#02060a")
LEFT = _bgr("#4dd2ff")
RIGHT = _bgr("#ffd34d")
CAMERA = _bgr("#ff8adf")
AXIS_COLORS = (_bgr("#ff6b6b"), _bgr("#51cf66"), _bgr("#5c9dff"))


def normalize_views(views: dict | None) -> dict[str, dict[str, float]]:
    """Return finite, JSON/cache-safe view values matching the browser defaults."""
    views = views if isinstance(views, dict) else {}
    result = {}
    for name in ("vov", "vgt", "vpred"):
        source = views.get(name) if isinstance(views.get(name), dict) else {}

        def finite(key: str, default: float) -> float:
            try:
                value = float(source.get(key, default))
            except (TypeError, ValueError):
                return default
            return value if math.isfinite(value) else default

        result[name] = {
            "az": finite("az", DEFAULT_AZ),
            "el": max(-1.5, min(1.5, finite("el", DEFAULT_EL))),
            "zoom": max(0.05, min(50.0, finite("zoom", 1.0))),
            "panX": finite("panX", 0.0),
            "panY": finite("panY", 0.0),
        }
    return result


def view_cache_tuple(views: dict | None) -> tuple:
    normalized = normalize_views(views)
    return tuple(
        round(normalized[name][key], 6)
        for name in ("vov", "vgt", "vpred")
        for key in ("az", "el", "zoom", "panX", "panY")
    )


@lru_cache(maxsize=8)
def _background(width: int, height: int) -> np.ndarray:
    """椭圆径向暗角：中心亮、四角沉下去，比竖向渐变更像一个「舞台」。"""
    xs = (np.arange(width, dtype=np.float32) - (width - 1) * 0.5) / max(1.0, width * 0.5)
    ys = (np.arange(height, dtype=np.float32) - (height - 1) * 0.5) / max(1.0, height * 0.5)
    radius = np.sqrt((xs[None, :] * 0.82) ** 2 + (ys[:, None] * 1.0) ** 2)
    blend = np.clip(radius / 1.2, 0.0, 1.0)[:, :, None] ** 1.35
    frame = BG_CENTER[None, None, :] * (1.0 - blend) + BG_EDGE[None, None, :] * blend
    return frame.astype(np.uint8)


def _mix(color, other, weight: float):
    """weight=1 → color，weight=0 → other；把淡色预混进背景，省掉逐笔 alpha 混合。"""
    weight = max(0.0, min(1.0, float(weight)))
    return tuple(float(c) * weight + float(o) * (1.0 - weight)
                 for c, o in zip(color, other))


def _nice_step(raw: float) -> float:
    """取 1/2/5×10^k 中不超过 raw 的最大「整齐」值（网格步长与比例尺共用）。"""
    if not math.isfinite(raw) or raw <= 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(raw))
    nice = power
    for multiplier in (1, 2, 5, 10):
        if multiplier * power <= raw:
            nice = multiplier * power
    return nice


def _point(value) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        return None
    return result


def normalize_coord_mode(coord_mode: str | None) -> str:
    value = str(coord_mode or DEFAULT_COORD_MODE).lower()
    if value not in COORD_MODES:
        raise ValueError(f"unsupported fixed-world coordinate mode: {coord_mode}")
    return value


class _Coordinates:
    def __init__(self, payload: dict, coord_mode: str = DEFAULT_COORD_MODE):
        self.coord_mode = normalize_coord_mode(coord_mode)
        camera_t = payload.get("cam_t") or []
        camera_r = payload.get("cam_R") or []
        first = next(
            (index for index, value in enumerate(camera_t)
             if _point(value) is not None and index < len(camera_r)
             and np.asarray(camera_r[index]).size == 9),
            None,
        )
        if first is None:
            self.origin = np.zeros(3, dtype=np.float64)
            camera_zero_rotation = np.eye(3, dtype=np.float64)
        else:
            self.origin = _point(camera_t[first])
            camera_zero_rotation = np.asarray(
                camera_r[first], dtype=np.float64).reshape(3, 3)
        self.rotation = (Z_UP_FROM_CV @ camera_zero_rotation
                         if self.coord_mode == "z_up" else camera_zero_rotation)

    def point(self, value) -> np.ndarray | None:
        value = _point(value)
        return None if value is None else self.rotation @ (value - self.origin)

    def camera(self, payload: dict, frame_index: int):
        camera_t = payload.get("cam_t") or []
        camera_r = payload.get("cam_R") or []
        if not camera_t or not camera_r:
            return None, None
        index = min(frame_index, len(camera_t) - 1, len(camera_r) - 1)
        position = self.point(camera_t[index])
        rotation = np.asarray(camera_r[index], dtype=np.float64)
        if position is None or rotation.size != 9 or not np.all(np.isfinite(rotation)):
            return position, None
        return position, rotation.reshape(3, 3) @ self.rotation.T


def _sampled(points) -> list:
    if not points:
        return []
    stride = max(1, int(math.ceil(len(points) / MAX_TRAJECTORY_POINTS)))
    indices = list(range(0, len(points), stride))
    if indices[-1] != len(points) - 1:
        indices.append(len(points) - 1)
    return [points[index] for index in indices]


def _draw_segment(frame: np.ndarray, start, end, color, thickness: int,
                  dash: tuple[int, int] | None = None) -> None:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return
    if not dash:
        cv2.line(frame, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)),
                 color, thickness, cv2.LINE_AA)
        return
    on, off = dash
    cursor = 0.0
    while cursor < length:
        finish = min(length, cursor + on)
        p0 = start + delta * (cursor / length)
        p1 = start + delta * (finish / length)
        cv2.line(frame, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)),
                 color, thickness, cv2.LINE_AA)
        cursor += on + off


def _draw_polyline(frame: np.ndarray, points: list, color, thickness: int,
                   dash: tuple[int, int] | None = None) -> None:
    previous = None
    for point in points:
        if point is None:
            previous = None
            continue
        if previous is not None:
            _draw_segment(frame, previous, point, color, thickness, dash)
        previous = point


def _draw_axis_labels(frame: np.ndarray, labels: list, ss: float) -> None:
    """Draw positive-axis labels last so camera/hand overlays cannot hide them."""
    if not labels:
        return
    height, width = frame.shape[:2]
    layer = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52 * ss
    thickness = _scaled(1, ss)
    for label, origin, projected, color in labels:
        direction = np.asarray(projected) - np.asarray(origin)
        direction /= max(1e-6, float(np.linalg.norm(direction)))
        normal = np.asarray([-direction[1], direction[0]])
        position = np.rint(
            projected + direction * _scaled(7, ss) + normal * _scaled(2, ss)
        ).astype(int)
        size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        x = max(1, min(width - size[0] - 2, int(position[0])))
        y = max(size[1] + 1, min(height - baseline - 2, int(position[1])))
        cv2.putText(layer, label, (x, y), font, font_scale,
                    OUTLINE, thickness + _scaled(2, ss), cv2.LINE_AA)
        cv2.putText(layer, label, (x, y), font, font_scale,
                    color, thickness, cv2.LINE_AA)
    cv2.addWeighted(
        layer, AXIS_LABEL_ALPHA, frame,
        1.0 - AXIS_LABEL_ALPHA, 0.0, dst=frame)


def _draw_trail(frame: np.ndarray, points: list, color, thickness: int,
                dash: tuple[int, int] | None = None) -> None:
    """轨迹尾迹：越早的一段越淡（颜色直接向背景预混，无需逐笔 alpha 混合）。"""
    total = max(1, len(points) - 1)
    previous = None
    for index, point in enumerate(points):
        if point is None:
            previous = None
            continue
        if previous is not None:
            weight = 0.22 + 0.78 * (index / total) ** 1.4
            _draw_segment(frame, previous, point, _mix(color, BG_MIX, weight),
                          thickness, dash)
        previous = point


def _scaled(value: float, ss: float, minimum: int = 1) -> int:
    return max(minimum, int(round(value * ss)))


def _rounded_rect(image: np.ndarray, p0, p1, radius: int, color) -> None:
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    radius = max(0, min(int(radius), (x1 - x0) // 2, (y1 - y0) // 2))
    if radius <= 0:
        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)
        return
    cv2.rectangle(image, (x0 + radius, y0), (x1 - radius, y1), color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0 + radius), (x1, y1 - radius), color, -1, cv2.LINE_AA)
    for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                   (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
        cv2.circle(image, (cx, cy), radius, color, -1, cv2.LINE_AA)


def _put_label(frame: np.ndarray, text: str, position, color=TEXT,
               scale: float = 0.42, ss: float = 1.0) -> None:
    x, baseline_y = (int(round(position[0])), int(round(position[1])))
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = _scaled(1.0, ss)
    size, baseline = cv2.getTextSize(text, font, scale * ss, thickness)
    pad = _scaled(3, ss)
    x = max(pad, min(frame.shape[1] - size[0] - 2 * pad, x))
    baseline_y = max(size[1] + pad + 1, min(frame.shape[0] - baseline - pad, baseline_y))
    _rounded_rect(frame, (x - pad, baseline_y - size[1] - pad),
                  (x + size[0] + pad, baseline_y + baseline + pad),
                  _scaled(3, ss), LABEL_BG)
    cv2.putText(frame, text, (x, baseline_y), font, scale * ss, color,
                thickness, cv2.LINE_AA)


def _chip(frame: np.ndarray, rows: list[tuple[str, tuple, float]], anchor,
          ss: float, *, align: str = "tl", swatch: bool = False,
          alpha: float = HUD_ALPHA) -> tuple[int, int]:
    """圆角半透明小卡片：rows = [(文本, 颜色, 字号)]；返回卡片右下角，便于纵向堆叠。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = _scaled(1.0, ss)
    pad = _scaled(7, ss)
    gap = _scaled(5, ss)
    dot = _scaled(4, ss)
    swatch_gap = (dot * 2 + _scaled(6, ss)) if swatch else 0
    sizes = [cv2.getTextSize(text, font, scale * ss, thickness)[0]
             for text, _color, scale in rows]
    width = max(size[0] for size in sizes) + 2 * pad + swatch_gap
    height = sum(size[1] for size in sizes) + gap * (len(rows) - 1) + 2 * pad
    x0 = int(anchor[0]) if "l" in align else int(anchor[0]) - width
    y0 = int(anchor[1]) if "t" in align else int(anchor[1]) - height
    x0 = max(0, min(frame.shape[1] - width, x0))
    y0 = max(0, min(frame.shape[0] - height, y0))

    layer = frame.copy()
    _rounded_rect(layer, (x0, y0), (x0 + width, y0 + height), _scaled(6, ss), CHIP_BG)
    cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0.0, dst=frame)

    cursor = y0 + pad
    for (text, color, scale), size in zip(rows, sizes):
        baseline = cursor + size[1]
        if swatch:
            cv2.circle(frame, (x0 + pad + dot, baseline - size[1] // 2), dot,
                       color, -1, cv2.LINE_AA)
        cv2.putText(frame, text, (x0 + pad + swatch_gap, baseline), font,
                    scale * ss, color, thickness, cv2.LINE_AA)
        cursor = baseline + gap
    return x0 + width, y0 + height


def _items_for_payload(payload: dict, layout: str):
    gt, pred = payload.get("gt"), payload.get("pred")
    if layout == "side":
        if gt is None:
            return [([{"d": pred, "dash": False, "tag": "PRED"}] if pred else [], "PRED"),
                    ([], "")]
        if pred is None:
            return [([{"d": gt, "dash": False, "tag": "GT"}], "GT"), ([], "")]
        return [([{"d": gt, "dash": False, "tag": "GT"}], "GT"),
                ([{"d": pred, "dash": False, "tag": "PRED"}], "PRED")]
    if gt is None:
        return [([{"d": pred, "dash": False, "tag": "PRED"}] if pred else [], "PRED")]
    if pred is None:
        return [([{"d": gt, "dash": False, "tag": "GT"}], "GT")]
    return [([{"d": gt, "dash": False, "tag": "GT"},
              {"d": pred, "dash": True, "tag": "PRED"}], "GT solid / PRED dashed")]


def _extent(items: list[dict], coordinates: dict[int, _Coordinates]):
    points = [np.zeros(3, dtype=np.float64)]
    for item in items:
        payload = item["d"]
        coord = coordinates[id(payload)]
        for side in ("left", "right"):
            for value in (payload.get("traj") or {}).get(side, []):
                transformed = coord.point(value)
                if transformed is not None:
                    points.append(transformed)
        for value in payload.get("cam_t") or []:
            transformed = coord.point(value)
            if transformed is not None:
                points.append(transformed)
    array = np.asarray(points, dtype=np.float64)
    minimum, maximum = array.min(axis=0), array.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = max(float(np.max(maximum - minimum)) * 0.5, 0.5)
    radius = max(radius, float(np.linalg.norm(center)))
    return center, radius, array


def _up_direction(items: list[dict], coordinates: dict[int, _Coordinates]) -> np.ndarray:
    """固定世界系没有天然重力轴：用全段相机 +Y(下) 的平均值取反当「上」。

    cam_R 是 world→cam 行主序，第 1 行 = 相机 +Y 在原世界中的方向；显示系再左乘首帧旋转。
    """
    accumulated = np.zeros(3, dtype=np.float64)
    for item in items:
        payload = item["d"]
        coord = coordinates[id(payload)]
        rows = []
        for value in payload.get("cam_R") or []:
            rotation = np.asarray(value, dtype=np.float64)
            if rotation.size == 9 and np.all(np.isfinite(rotation)):
                rows.append(rotation.reshape(3, 3)[1])
        if rows:
            accumulated += coord.rotation @ np.asarray(rows).mean(axis=0)
    norm = float(np.linalg.norm(accumulated))
    if norm < 1e-6:
        return np.asarray([0.0, -1.0, 0.0])      # OpenCV 相机系里 -Y 朝上
    return -accumulated / norm


def _ground_plane(points: np.ndarray, center: np.ndarray, radius: float,
                  up: np.ndarray) -> dict:
    """地面：过「沿 up 的最低点再下移一点」的水平面 + 平面内正交基 + 整齐格距。"""
    heights = points @ up
    clearance = max(15.0, GROUND_CLEARANCE * radius)
    level = float(heights.min()) - clearance
    origin = center + up * (level - float(center @ up))
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(reference @ up)) > 0.9:
        reference = np.asarray([0.0, 0.0, 1.0])
    e1 = reference - up * float(reference @ up)
    e1 /= max(1e-9, float(np.linalg.norm(e1)))
    e2 = np.cross(up, e1)
    return {
        "origin": origin, "e1": e1, "e2": e2, "up": up,
        "step": _nice_step(2.0 * radius / GRID_TARGET_CELLS),
        "half": GRID_SPAN * radius,
    }


def _flatten(point: np.ndarray, plane: dict) -> np.ndarray:
    """把 3D 点沿 up 压到地面上（落影/垂线用）。"""
    up = plane["up"]
    return point - up * float((point - plane["origin"]) @ up)


def _draw_arrow(frame: np.ndarray, start, end, color, thickness: int,
                dash: tuple[int, int] | None = None,
                outline_color=OUTLINE) -> None:
    start_i = tuple(np.rint(start).astype(int))
    end_i = tuple(np.rint(end).astype(int))
    if not dash:
        cv2.arrowedLine(frame, start_i, end_i, outline_color, thickness + 4, cv2.LINE_AA,
                        tipLength=0.18)
        cv2.arrowedLine(frame, start_i, end_i, color, thickness, cv2.LINE_AA,
                        tipLength=0.18)
        return
    _draw_segment(frame, start, end, outline_color, thickness + 3, dash)
    _draw_segment(frame, start, end, color, thickness, dash)
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return
    direction = delta / length
    normal = np.asarray([-direction[1], direction[0]])
    arrow_length = min(length * 0.35, max(7.0, thickness * 2.6))
    half_width = max(3.0, thickness * 1.25)
    base = np.asarray(end, dtype=np.float64) - direction * arrow_length
    outline_triangle = np.rint([
        end, base + normal * (half_width + 2), base - normal * (half_width + 2),
    ]).astype(np.int32)
    color_triangle = np.rint([
        end, base + normal * half_width, base - normal * half_width,
    ]).astype(np.int32)
    cv2.fillConvexPoly(frame, outline_triangle, outline_color, cv2.LINE_AA)
    cv2.fillConvexPoly(frame, color_triangle, color, cv2.LINE_AA)


def _draw_dashed_rect(frame: np.ndarray, p0, p1, color, thickness: int,
                      dash: tuple[int, int]) -> None:
    x0, y0 = p0
    x1, y1 = p1
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    for index, start in enumerate(corners):
        _draw_segment(frame, start, corners[(index + 1) % 4], color, thickness, dash)


def _draw_scale_bar(frame: np.ndarray, scale: float, ss: float = 1.0) -> None:
    if not math.isfinite(scale) or scale <= 0:
        return
    nice = _nice_step(90.0 * ss / scale)
    length = int(round(nice * scale))
    margin = _scaled(16, ss)
    x1, y = frame.shape[1] - margin, frame.shape[0] - _scaled(17, ss)
    x0 = x1 - length
    thickness = _scaled(2, ss)
    tick = _scaled(4, ss)
    cv2.line(frame, (x0, y), (x1, y), TEXT, thickness, cv2.LINE_AA)
    cv2.line(frame, (x0, y - tick), (x0, y + tick), TEXT, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y - tick), (x1, y + tick), TEXT, thickness, cv2.LINE_AA)
    label = f"{nice / 100:g} m" if nice >= 100 else f"{nice:g} cm"
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38 * ss,
                           _scaled(1, ss))[0]
    cv2.putText(frame, label, ((x0 + x1 - size[0]) // 2, y - _scaled(7, ss)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38 * ss, TEXT, _scaled(1, ss), cv2.LINE_AA)


def _prepare_scene_geometry(items: list[dict], view: dict,
                            width: int, height: int,
                            coord_mode: str = DEFAULT_COORD_MODE) -> dict:
    coord_mode = normalize_coord_mode(coord_mode)
    coordinates = {
        id(item["d"]): _Coordinates(item["d"], coord_mode) for item in items
    }
    center, radius, points = _extent(items, coordinates)
    scale = min(width, height) * 0.42 * view["zoom"] / radius
    ca, sa = math.cos(view["az"]), math.sin(view["az"])
    ce, se = math.cos(view["el"]), math.sin(view["el"])

    def project(point):
        x, y, z = np.asarray(point, dtype=np.float64) - center
        if coord_mode == "z_up":
            x1, depth = x * ca + y * sa, -x * sa + y * ca
            y2 = -z * ce - depth * se
        else:
            x1, depth = x * ca + z * sa, -x * sa + z * ca
            y2 = y * ce - depth * se
        return np.asarray([
            width * 0.5 + view["panX"] + x1 * scale,
            height * 0.5 + view["panY"] + y2 * scale,
        ])

    up = (np.asarray([0.0, 0.0, 1.0]) if coord_mode == "z_up"
          else _up_direction(items, coordinates))
    return {
        "coordinates": coordinates, "center": center, "radius": radius,
        "scale": scale, "project": project, "static_ready": False,
        "plane": _ground_plane(points, center, radius, up),
        "coord_mode": coord_mode,
    }


def _draw_ground(frame: np.ndarray, prepared: dict, ss: float) -> None:
    """静态网格地面：圆盘状、离中心越远越淡，major 线每 GRID_MAJOR_EVERY 格加亮。"""
    plane, project = prepared["plane"], prepared["project"]
    step, half = plane["step"], plane["half"]
    count = int(min(48, max(2, math.floor(half / step))))
    thin, thick = _scaled(1.0, ss), _scaled(1.7, ss)
    for axis, other in ((plane["e1"], plane["e2"]), (plane["e2"], plane["e1"])):
        for index in range(-count, count + 1):
            offset = index * step
            base = plane["origin"] + axis * offset
            major = index % GRID_MAJOR_EVERY == 0
            color_base = GRID_MAJOR if major else GRID_MINOR
            for cell in range(-count, count):
                start, finish = cell * step, (cell + 1) * step
                distance = math.hypot(offset, (start + finish) * 0.5)
                if distance > half:
                    continue
                fade = (1.0 - distance / half) ** 1.15
                fade *= 1.0 if major else 0.66
                if fade < 0.03:
                    continue
                _draw_segment(frame, project(base + other * start),
                              project(base + other * finish),
                              _mix(color_base, BG_MIX, fade),
                              thick if major else thin)

    # 世界原点（首帧相机）落到地面的垂线 + 落点环：一眼读出相机离地多高。
    origin_ground = _flatten(np.zeros(3), plane)
    _draw_segment(frame, project(np.zeros(3)), project(origin_ground),
                  _mix(TEXT, BG_MIX, 0.35), thin, (_scaled(4, ss), _scaled(4, ss)))
    cv2.circle(frame, tuple(np.rint(project(origin_ground)).astype(int)),
               _scaled(4, ss), _mix(TEXT, BG_MIX, 0.5), thin, cv2.LINE_AA)


def _draw_hand(frame: np.ndarray, projected: list, connections, color,
               dash: tuple[int, int] | None, ss: float) -> None:
    """骨架：深色描边 + 本色芯 + 圆头端点，腕节点加环。"""
    outline = _scaled(3.4, ss)
    core = _scaled(1.5, ss)
    for start_index, end_index in connections or []:
        if start_index >= len(projected) or end_index >= len(projected):
            continue
        _draw_segment(frame, projected[start_index], projected[end_index],
                      OUTLINE, outline, dash)
    for start_index, end_index in connections or []:
        if start_index >= len(projected) or end_index >= len(projected):
            continue
        _draw_segment(frame, projected[start_index], projected[end_index],
                      color, core, dash)
    for index, value in enumerate(projected):
        point = tuple(np.rint(value).astype(int))
        if index == 0:
            cv2.circle(frame, point, _scaled(4.2, ss), OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(frame, point, _scaled(3.3, ss), color, -1, cv2.LINE_AA)
            cv2.circle(frame, point, _scaled(5.4, ss), color,
                       _scaled(0.8, ss), cv2.LINE_AA)
        else:
            cv2.circle(frame, point, _scaled(2.35, ss), OUTLINE, -1, cv2.LINE_AA)
            cv2.circle(frame, point, _scaled(1.45, ss), color, -1, cv2.LINE_AA)


def _draw_shadow_skeleton(layer: np.ndarray, transformed: list, connections,
                          plane: dict, project, ss: float) -> None:
    flattened = [project(_flatten(value, plane)) for value in transformed]
    thickness = _scaled(3.0, ss)
    for start_index, end_index in connections or []:
        if start_index >= len(flattened) or end_index >= len(flattened):
            continue
        _draw_segment(layer, flattened[start_index], flattened[end_index],
                      SHADOW, thickness)
    if flattened:
        cv2.circle(layer, tuple(np.rint(flattened[0]).astype(int)),
                   _scaled(5, ss), SHADOW, -1, cv2.LINE_AA)


def _draw_scene(frame: np.ndarray, items: list[dict], view: dict,
                frame_index: int, total_frames: int,
                show_traj: bool, show_cam_hand: bool, caption: str,
                *, prepared: dict | None = None,
                static_only: bool = False, ss: float = 1.0) -> None:
    height, width = frame.shape[:2]
    if not items:
        if static_only or prepared is None:
            cv2.putText(frame, "(no data)", (_scaled(12, ss), _scaled(26, ss)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ss, MUTED,
                        _scaled(1, ss), cv2.LINE_AA)
        return

    prepared = prepared or _prepare_scene_geometry(items, view, width, height)
    coordinates = prepared["coordinates"]
    radius, scale, project = prepared["radius"], prepared["scale"], prepared["project"]
    plane = prepared["plane"]
    draw_static = static_only or not prepared.get("static_ready", False)

    if draw_static:
        _draw_ground(frame, prepared, ss)
        origin = project(np.zeros(3))
        axis_length = radius * 0.6
        axis_layer = frame.copy()
        axis_labels = []
        for label, endpoint, color in zip(
                ("X", "Y", "Z"), np.eye(3) * axis_length, AXIS_COLORS):
            projected = project(endpoint)
            cv2.arrowedLine(
                axis_layer,
                tuple(np.rint(origin).astype(int)),
                tuple(np.rint(projected).astype(int)),
                color, _scaled(1.5, ss), cv2.LINE_AA, tipLength=0.14)
            axis_labels.append((f"+{label}", origin, projected, color))
        cv2.circle(axis_layer, tuple(np.rint(origin).astype(int)), _scaled(2.5, ss),
                   TEXT, -1, cv2.LINE_AA)
        cv2.addWeighted(
            axis_layer, AXIS_ALPHA, frame, 1.0 - AXIS_ALPHA, 0.0, dst=frame)

        prepared["axis_labels"] = axis_labels

    # 逐帧落影统一画在一张副本上，最后一次性混合：既保留网格透过阴影的层次，也只付一次混合开销。
    shadow_layer = None if static_only else frame.copy()
    camera_overlays = []
    for item in items:
        payload, dashed = item["d"], bool(item.get("dash"))
        coord = coordinates[id(payload)]
        camera_position, camera_rotation = coord.camera(payload, frame_index)
        camera_trajectory_dash = (_scaled(8, ss), _scaled(5, ss))
        hand_trajectory_dash = (8, 5) if dashed else None
        skeleton_dash = (_scaled(7, ss), _scaled(5, ss)) if dashed else None
        if hand_trajectory_dash is not None:
            hand_trajectory_dash = (_scaled(hand_trajectory_dash[0], ss),
                                    _scaled(hand_trajectory_dash[1], ss))

        if draw_static and show_traj:
            camera_points = [
                coord.point(value) for value in _sampled(payload.get("cam_t") or [])]
            _draw_trail(frame,
                        [project(point) if point is not None else None
                         for point in camera_points],
                        CAMERA, _scaled(1.4, ss), camera_trajectory_dash)
        if not static_only and camera_position is not None:
            if show_cam_hand:
                _draw_segment(shadow_layer, project(camera_position),
                              project(_flatten(camera_position, plane)),
                              _mix(CAMERA, BG_MIX, 0.45), _scaled(1, ss),
                              (_scaled(4, ss), _scaled(4, ss)))
                cv2.circle(shadow_layer,
                           tuple(np.rint(project(_flatten(camera_position, plane))).astype(int)),
                           _scaled(4, ss), SHADOW, -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(np.rint(project(camera_position)).astype(int)),
                       _scaled(3 if dashed else 4, ss), CAMERA, -1, cv2.LINE_AA)
            if show_cam_hand and camera_rotation is not None:
                camera_overlays.append((camera_position, camera_rotation, item.get("tag") == "PRED"))

        joints_by_hand = payload.get("joints") or [[], []]
        trajectories = payload.get("traj") or {}
        for hand_index, (side, color) in enumerate((("left", LEFT), ("right", RIGHT))):
            trajectory = trajectories.get(side) or []
            if draw_static and show_traj:
                transformed_trajectory = [
                    coord.point(value) for value in _sampled(trajectory)]
                _draw_trail(frame,
                            [project(point) if point is not None else None
                             for point in transformed_trajectory],
                            color, _scaled(2.0, ss), hand_trajectory_dash)
            if not static_only and show_traj:
                if trajectory:
                    current = coord.point(trajectory[min(frame_index, len(trajectory) - 1)])
                    if current is not None:
                        cv2.circle(frame, tuple(np.rint(project(current)).astype(int)),
                                   _scaled(4, ss), color, -1, cv2.LINE_AA)

            if static_only:
                continue
            if hand_index >= len(joints_by_hand) or not joints_by_hand[hand_index]:
                continue
            joints = joints_by_hand[hand_index]
            values = joints[min(frame_index, len(joints) - 1)]
            if values is None:
                continue
            transformed = [coord.point(value) for value in values]
            if any(value is None for value in transformed):
                continue
            connections = payload.get("conn") or []
            _draw_shadow_skeleton(shadow_layer, transformed, connections,
                                  plane, project, ss)
            _draw_segment(shadow_layer, project(transformed[0]),
                          project(_flatten(transformed[0], plane)),
                          _mix(color, BG_MIX, 0.4), _scaled(1, ss),
                          (_scaled(4, ss), _scaled(4, ss)))
            _draw_hand(frame, [project(value) for value in transformed],
                       connections, color, skeleton_dash, ss)

            if show_cam_hand and camera_position is not None and transformed:
                wrist = transformed[0]
                camera_px, wrist_px = project(camera_position), project(wrist)
                _draw_segment(frame, camera_px, wrist_px, color, _scaled(1.4, ss))
                distance = float(np.linalg.norm(wrist - camera_position))
                prefix = (item.get("tag", "") + " ") if len(items) > 1 else ""
                label = f"{prefix}{'L' if hand_index == 0 else 'R'} {distance:.1f} cm"
                offset = (11 if dashed else -7) * ss
                _put_label(frame, label,
                           (camera_px + wrist_px) * 0.5 + (5 * ss, offset),
                           color, 0.42, ss)

    if shadow_layer is not None:
        cv2.addWeighted(shadow_layer, SHADOW_ALPHA, frame, 1.0 - SHADOW_ALPHA,
                        0.0, dst=frame)

    if static_only:
        if draw_static:
            _draw_hud(frame, caption, plane, ss)
            _draw_scale_bar(frame, scale, ss)
        return

    for position, rotation, predicted in camera_overlays:
        center_px = project(position)
        pose_colors = ((_bgr("#ff922b"), _bgr("#ffd43b"), _bgr("#b197fc"))
                       if predicted else
                       (_bgr("#ff595e"), _bgr("#69db7c"), _bgr("#4dabf7")))
        directions = (rotation[0], -rotation[1], rotation[2])
        lengths = (max(7.0, min(radius * 0.22, 22.0)),
                   max(7.0, min(radius * 0.22, 22.0)),
                   max(12.0, min(radius * 0.50, 48.0)))
        # 三轴只给视线轴标注（并注明 GT/PRED）：GT+PRED 各三条标签会把画面中心糊成一团。
        labels = (None, None, "PRED view" if predicted else "GT view")
        camera_dash = None
        faded_outline = _mix(OUTLINE, BG_MIX, 0.34)
        for direction, length, color, label in zip(directions, lengths, pose_colors, labels):
            endpoint = project(position + direction * length)
            faded_color = _mix(color, BG_MIX, 0.58)
            _draw_arrow(frame, center_px, endpoint, faded_color,
                        _scaled(3.2 if label else 2.4, ss), camera_dash,
                        outline_color=faded_outline)
            if label:
                _put_label(frame, label, endpoint + (_scaled(4, ss), -_scaled(4, ss)),
                           faded_color, 0.36, ss)
        marker_size = max(2, int(round((16 if predicted else 12) * view["zoom"] * ss)))
        x0, y0 = np.rint(center_px - marker_size * 0.5).astype(int)
        x1, y1 = x0 + marker_size, y0 + marker_size
        marker_color = _bgr("#ff6b6b") if predicted else _bgr("#51cf66")
        cv2.rectangle(frame, (x0, y0), (x1, y1), faded_outline,
                      _scaled(2.6, ss), cv2.LINE_AA)
        cv2.rectangle(frame, (x0, y0), (x1, y1),
                      _mix(marker_color, BG_MIX, 0.55),
                      _scaled(1.2, ss), cv2.LINE_AA)

    _draw_axis_labels(frame, prepared.get("axis_labels") or [], ss)
    if draw_static:
        _draw_hud(frame, caption, plane, ss)
        _draw_scale_bar(frame, scale, ss)


def _draw_hud(frame: np.ndarray, caption: str, plane: dict, ss: float) -> None:
    """Keep exported 3D HUD limited to source identity and scale."""
    step = plane["step"]
    grid = f"grid {step / 100:g} m" if step >= 100 else f"grid {step:g} cm"
    title = str(caption or "").strip()
    if "GT" in title and "PRED" in title:
        title = "GT / PRED"
    else:
        title = title.upper()
    prefix = f"{title}  |  " if title else ""
    _chip(frame, [(f"{prefix}cm  |  {grid}", TEXT, 0.42)],
          (_scaled(12, ss), _scaled(12, ss)), ss, align="tl")


def _prepare_render(payload: dict, *, layout: str, views: dict | None,
                    coord_mode: str,
                    show_traj: bool, show_cam_hand: bool,
                    size: tuple[int, int]) -> dict:
    out_width, out_height = map(int, size)
    if out_width < 160 or out_height < 120:
        raise ValueError("fixed-world render size is too small")
    ss = float(max(1, int(SUPERSAMPLE)))
    width, height = int(out_width * ss), int(out_height * ss)
    layout = "side" if layout == "side" else "overlay"
    coord_mode = normalize_coord_mode(coord_mode)
    normalized_views = normalize_views(views)
    panels = _items_for_payload(payload, layout)
    total = max(1, int(payload.get("nframes", 1)))
    specs = []
    if layout == "overlay":
        items, caption = panels[0]
        raw_specs = [(0, width, items, caption, normalized_views["vov"], False)]
    else:
        gap = int(10 * ss)
        panel_width = (width - gap) // 2
        raw_specs = []
        for index, ((items, caption), view_name) in enumerate(
                zip(panels, ("vgt", "vpred"))):
            x0 = index * (panel_width + gap)
            x1 = width if index == 1 else x0 + panel_width
            raw_specs.append(
                (x0, x1, items, caption, normalized_views[view_name], True))

    for x0, x1, items, caption, view, show_caption in raw_specs:
        base = _background(x1 - x0, height).copy()
        geometry = (_prepare_scene_geometry(
            items, view, x1 - x0, height, coord_mode)
                    if items else None)
        _draw_scene(
            base, items, view, 0, total, show_traj, show_cam_hand, caption,
            prepared=geometry, static_only=True, ss=ss)
        if show_caption and caption and geometry is None:
            _put_label(base, caption, (_scaled(10, ss), _scaled(22, ss)), TEXT, 0.48, ss)
        if geometry is not None:
            geometry["static_ready"] = True
        specs.append({
            "x0": x0, "x1": x1, "items": items, "caption": caption,
            "view": view, "base": base, "geometry": geometry,
        })
    return {
        "width": width, "height": height, "total": total,
        "out_size": (out_width, out_height), "ss": ss,
        "coord_mode": coord_mode,
        "show_traj": show_traj, "show_cam_hand": show_cam_hand,
        "specs": specs,
    }


def _render_prepared(prepared: dict, frame_index: int) -> np.ndarray:
    frame_index = max(0, min(int(frame_index), prepared["total"] - 1))
    output = _background(prepared["width"], prepared["height"]).copy()
    for spec in prepared["specs"]:
        panel = spec["base"].copy()
        if spec["items"]:
            _draw_scene(
                panel, spec["items"], spec["view"], frame_index, prepared["total"],
                prepared["show_traj"], prepared["show_cam_hand"], spec["caption"],
                prepared=spec["geometry"], ss=prepared["ss"])
        output[:, spec["x0"]:spec["x1"]] = panel
    out_width, out_height = prepared["out_size"]
    if (out_width, out_height) != (prepared["width"], prepared["height"]):
        output = cv2.resize(output, (out_width, out_height),
                            interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(output)


def render_fixed_world_frame(payload: dict, frame_index: int, *,
                             layout: str = "overlay", views: dict | None = None,
                             coord_mode: str = DEFAULT_COORD_MODE,
                             show_traj: bool = True, show_cam_hand: bool = True,
                             size: tuple[int, int] = DEFAULT_SIZE) -> np.ndarray:
    """Render one BGR frame using the same timeline index as the master video."""
    prepared = _prepare_render(
        payload, layout=layout, views=views,
        coord_mode=coord_mode,
        show_traj=show_traj, show_cam_hand=show_cam_hand, size=size)
    return _render_prepared(prepared, frame_index)


def render_fixed_world_video(payload: dict, output: str | Path, *,
                             fps: float | None = None,
                             layout: str = "overlay", views: dict | None = None,
                             coord_mode: str = DEFAULT_COORD_MODE,
                             show_traj: bool = True, show_cam_hand: bool = True,
                             size: tuple[int, int] = DEFAULT_SIZE,
                             on_step: Callable[[int, int], None] | None = None) -> Path:
    """Render every payload frame and encode it directly as browser-friendly H.264."""
    total = int(payload.get("nframes") or 0)
    if total <= 0:
        raise ValueError("fixed-world payload contains no frames")
    output = Path(output)
    prepared = _prepare_render(
        payload, layout=layout, views=views,
        coord_mode=coord_mode,
        show_traj=show_traj, show_cam_hand=show_cam_hand, size=size)
    writer = draw.H264PipeWriter(output, float(fps or payload.get("fps") or 30.0), size)
    try:
        for frame_index in range(total):
            writer.write(_render_prepared(prepared, frame_index))
            if on_step is not None:
                on_step(frame_index + 1, total)
    finally:
        writer.close()
    return output
