#!/usr/bin/env python3
"""Render 21-point hands as a fixed third-person first-generation Wuji Hand video."""
from __future__ import annotations

import math
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np


_REPO_DIR = Path(__file__).resolve().parents[4]
_WUJI_ROOT = _REPO_DIR / "eval" / "simulate" / "wuji-retargeting"
_DESCRIPTION_ROOT = _WUJI_ROOT / "wuji_retargeting" / "wuji-description"
_CONFIG_DIR = _WUJI_ROOT / "example" / "config"

# HUD 与手部材质同一套配色：左青、右金，与固定世界 3D 面板一致。
_PANEL_COLORS = {"left": (245, 197, 97), "right": (113, 209, 255)}          # BGR
_HAND_RGBA = {"left": (0.34, 0.73, 0.95, 1.0), "right": (1.0, 0.79, 0.40, 1.0)}
# 视觉 geom 逐手一个 group：按 presence 开关 mjvOption.geomgroup，未检测的手不出现也不投影。
_VISUAL_GROUP = {"left": 2, "right": 3}
_FLOOR_GROUP = 1
# 与 MuJoCo 面板共用较近参考地面，突出手部操作而不是空旷的离地空间。
_FLOOR_CLEARANCE = 0.08
_TEXT = (245, 246, 248)
_MUTED = (176, 186, 196)
_CARD = (14, 11, 8)
_CARD_ALPHA = 0.62
_ROBOT_VIDEO_PRESET = os.environ.get("VIEWER_ROBOT_VIDEO_PRESET", "superfast")

try:
    _ENCODE_BUFFER_FRAMES = max(
        0, int(os.environ.get("VIEWER_ROBOT_ENCODE_BUFFER_FRAMES", "3")))
except ValueError:
    _ENCODE_BUFFER_FRAMES = 3


def _hand_body() -> Path:
    body = _DESCRIPTION_ROOT / "hand" / "body"
    if all((body / kind / f"{side}.{ext}").is_file()
           for kind, ext in (("urdf", "urdf"), ("mjcf", "xml"))
           for side in ("left", "right")):
        return body
    raise FileNotFoundError(f"Wuji Hand URDF/MJCF assets not found under: {body}")


def _build_retargeter(side: str):
    if str(_WUJI_ROOT) not in sys.path:
        sys.path.insert(0, str(_WUJI_ROOT))
    from wuji_retargeting import Retargeter

    config_path = _CONFIG_DIR / f"adaptive_analytical_wuji_glove_{side}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Wuji Hand retarget config not found: {config_path}")
    return Retargeter.from_yaml(str(config_path), hand_side=side)


def _rgba(color) -> str:
    return " ".join(f"{value:.4f}" for value in color)


def _merged_hand_xml(body: Path, width: int, height: int) -> str:
    """把官方左右手 MJCF 合成一个带天空/网格地面/主光的场景（只在内存里拼，不改 vendored 文件）。

    单场景取代「左右手各一个 MjModel + 深度合成」：遮挡由 MuJoCo 自己处理，
    手才能互相遮挡并在地面上投影。两侧的 mesh/body/joint 名都带 left_/right_ 前缀，合并不会撞名。
    """
    assets, bodies = [], []
    for side in ("left", "right"):
        path = body / "mjcf" / f"{side}.xml"
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        mesh_root = (path.parent / (compiler.get("meshdir") or ""
                                    if compiler is not None else "")).resolve()
        for mesh in root.findall("./asset/mesh"):
            mesh.set("file", str(mesh_root / mesh.get("file")))   # 去 meshdir，改绝对路径
            assets.append(ET.tostring(mesh, encoding="unicode"))
        for hand in root.findall("./worldbody/body"):
            for geom in hand.iter("geom"):
                # 官方 MJCF：group1 = 视觉网格，无 group = 重复的碰撞网格（永远不显示）。
                if geom.get("group") == "1":
                    geom.set("group", str(_VISUAL_GROUP[side]))
                    geom.set("material", f"m_{side}")
                    geom.attrib.pop("rgba", None)                 # 统一材质，弃用逐 link 杂色
            bodies.append(ET.tostring(hand, encoding="unicode"))

    # 观感取值对齐 render/mujoco_scene.py 的 MuJoCo 面板：浅灰渐变天空 + checker 网格地面 +
    # 压低环境光的方向主光（ambient 太高会把接触阴影冲掉）。地面用薄 box 而非 plane：
    # 方向光在大 plane 上掠射会出 shadow acne，薄 box 的有限边界让阴影视锥贴合场景。
    from .mujoco_scene import (
        ROBOT_GROUND_RGB1, ROBOT_GROUND_RGB2, ROBOT_GROUND_TEXREPEAT,
        ROBOT_HAND_SHININESS, ROBOT_HAND_SPECULAR,
        ROBOT_HEADLIGHT_AMBIENT, ROBOT_HEADLIGHT_DIFFUSE,
        ROBOT_HEADLIGHT_SPECULAR, ROBOT_KEY_DIFFUSE, ROBOT_KEY_SPECULAR,
        ROBOT_SHADOWCLIP, ROBOT_SHADOWSCALE, ROBOT_ZFAR, ROBOT_ZNEAR,
    )
    rgb1 = " ".join(f"{value:.2f}" for value in ROBOT_GROUND_RGB1)
    rgb2 = " ".join(f"{value:.2f}" for value in ROBOT_GROUND_RGB2)
    ambient = " ".join(f"{value:.2f}" for value in ROBOT_HEADLIGHT_AMBIENT)
    diffuse = " ".join(f"{value:.2f}" for value in ROBOT_HEADLIGHT_DIFFUSE)
    specular = " ".join(f"{value:.2f}" for value in ROBOT_HEADLIGHT_SPECULAR)
    key_diffuse = " ".join(f"{value:.2f}" for value in ROBOT_KEY_DIFFUSE)
    key_specular = " ".join(f"{value:.2f}" for value in ROBOT_KEY_SPECULAR)
    return f"""<mujoco model="wuji-hand-scene">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="{int(width)}" offheight="{int(height)}"/>
    <quality shadowsize="4096" offsamples="2"/>
    <headlight ambient="{ambient}" diffuse="{diffuse}" specular="{specular}"/>
    <map shadowclip="{ROBOT_SHADOWCLIP}" shadowscale="{ROBOT_SHADOWSCALE}" znear="{ROBOT_ZNEAR}" zfar="{ROBOT_ZFAR}"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.95 0.96 0.98" rgb2="0.78 0.82 0.88"
             width="512" height="512"/>
    <texture name="ground_grid" type="2d" builtin="checker" rgb1="{rgb1}"
             rgb2="{rgb2}" width="512" height="512"/>
    <material name="ground" texture="ground_grid" texrepeat="{ROBOT_GROUND_TEXREPEAT} {ROBOT_GROUND_TEXREPEAT}" texuniform="true"
              rgba="1 1 1 1" reflectance="0.0" specular="0.0" shininess="0.0"/>
    <material name="m_left" rgba="{_rgba(_HAND_RGBA['left'])}" specular="{ROBOT_HAND_SPECULAR}" shininess="{ROBOT_HAND_SHININESS}"/>
    <material name="m_right" rgba="{_rgba(_HAND_RGBA['right'])}" specular="{ROBOT_HAND_SPECULAR}" shininess="{ROBOT_HAND_SHININESS}"/>
    {''.join(assets)}
  </asset>
  <worldbody>
    <light name="key" pos="0 0 2" dir="0 0 -1" directional="true" castshadow="true"
           diffuse="{key_diffuse}" specular="{key_specular}"/>
    <body name="floorb" pos="0 0 0">
      <geom name="floor" type="box" size="3 3 0.02" material="ground"
            group="{_FLOOR_GROUP}" contype="0" conaffinity="0"/>
    </body>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""


def _local_to_world_rotation(
    joints: np.ndarray, side: str, rotation_xyz: dict,
) -> np.ndarray:
    """Recover the robot-local to source-world rotation used by retargeting."""
    from scipy.spatial.transform import Rotation
    from wuji_retargeting.mediapipe import (
        OPERATOR2MANO_LEFT,
        OPERATOR2MANO_RIGHT,
        estimate_frame_from_hand_points,
    )

    centered = np.asarray(joints, dtype=np.float64) - joints[0]
    wrist_frame = estimate_frame_from_hand_points(centered)
    operator = OPERATOR2MANO_RIGHT if side == "right" else OPERATOR2MANO_LEFT
    adjustment = Rotation.from_euler(
        "xyz",
        [rotation_xyz.get(axis, 0.0) for axis in ("x", "y", "z")],
        degrees=True,
    ).as_matrix()
    return wrist_frame @ operator @ adjustment.T


def _root_pose(
    joints: np.ndarray,
    side: str,
    rotation_xyz: dict,
    wrist_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = _local_to_world_rotation(joints, side, rotation_xyz)
    translation = np.asarray(joints[0], dtype=np.float64) - rotation @ wrist_local
    return rotation, translation


def _configure_ego_camera(scene, cam_c2w: np.ndarray,
                          intrinsics: np.ndarray,
                          image_size: tuple[int, int]) -> None:
    """Apply the source OpenCV camera and off-axis intrinsics to an MjvScene."""
    camera_to_world = np.asarray(cam_c2w, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    width, height = map(float, image_size)
    if camera_to_world.shape != (4, 4):
        raise ValueError(f"camera pose must be [4,4], got {camera_to_world.shape}")
    if K.shape != (3, 3) or width <= 0 or height <= 0:
        raise ValueError("Wuji retargeting camera intrinsics or image size is invalid")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("Wuji retargeting camera focal lengths must be positive")

    # MuJoCo GL cameras look along ``forward`` with ``up`` pointing upward;
    # OpenCV uses +z forward and +y down.
    position = camera_to_world[:3, 3]
    forward = camera_to_world[:3, 2]
    up = -camera_to_world[:3, 1]
    for camera in scene.camera:
        camera.pos[:] = position
        camera.forward[:] = forward
        camera.up[:] = up
        near = float(camera.frustum_near)
        left, right = -cx / fx * near, (width - cx) / fx * near
        camera.frustum_center = (left + right) / 2.0
        camera.frustum_width = (right - left) / 2.0
        camera.frustum_bottom = -(height - cy) / fy * near
        camera.frustum_top = cy / fy * near


def _gravity_up(cam_c2w: np.ndarray) -> np.ndarray:
    """重力方向：全段相机 +Y(下) 的平均值取反（ego 相机大致随人体直立）。

    与固定世界 3D（fixed_world_video._up_direction / app.js upDirection）同一口径。
    """
    down = np.asarray(cam_c2w, dtype=np.float64)[:, :3, 1]
    down = down[np.all(np.isfinite(down), axis=1)]
    if not len(down):
        return np.asarray([0.0, 0.0, 1.0])
    mean = down.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:
        return np.asarray([0.0, 0.0, 1.0])
    return -mean / norm


def _floor_pose(up: np.ndarray, hand_points: np.ndarray,
                camera_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """地面位姿：中心沿 up 落在最低手点下方 _FLOOR_CLEARANCE，半边长按水平跨度扩展。

    返回 (中心, 四元数 wxyz 把 box 的 +Z 对到 up, 半边长)；纯 numpy，便于单测。
    """
    up = np.asarray(up, dtype=np.float64)
    hands = np.asarray(hand_points, dtype=np.float64).reshape(-1, 3)
    cameras = np.asarray(camera_points, dtype=np.float64).reshape(-1, 3)
    points = np.concatenate([hands, cameras], axis=0)
    level = float(np.percentile(hands @ up, 0.1)) - _FLOOR_CLEARANCE
    center = hands.mean(axis=0)
    center = center + up * (level - float(center @ up))
    # Size is based on the same hand + camera envelope as MuJoCo; the center
    # and contact height are based on hands so the ground does not drift with
    # the camera path.
    from .mujoco_scene import robot_ground_half_extent
    half = robot_ground_half_extent(points, up=up)

    axis = np.cross(np.asarray([0.0, 0.0, 1.0]), up)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        quat = np.asarray([1.0, 0.0, 0.0, 0.0] if float(up[2]) > 0
                          else [0.0, 1.0, 0.0, 0.0])
    else:
        angle = math.atan2(norm, float(up[2]))
        quat = np.concatenate([[math.cos(angle / 2.0)],
                               axis / norm * math.sin(angle / 2.0)])
    return center, quat, half


class _WujiHandScene:
    """左右手共处一个 MuJoCo 场景：单渲染器、原生遮挡、地面接触阴影。"""

    def __init__(self, body: Path, width: int, height: int):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(_merged_hand_xml(body, width, height))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[_FLOOR_GROUP] = 1
        self.option.sitegroup[:] = 0

        self.floor_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "floorb")
        self.floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.light = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "key")
        if min(self.floor_body, self.floor_geom, self.light) < 0:
            raise ValueError("Wuji Hand 场景缺少地面或主光")

        self.sides = {}
        for side in ("left", "right"):
            retargeter = _build_retargeter(side)
            from scipy.spatial.transform import Rotation
            from wuji_retargeting.mediapipe import (
                OPERATOR2MANO_LEFT,
                OPERATOR2MANO_RIGHT,
                estimate_frame_from_hand_points,
            )

            root = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_palm_link")
            if root < 0:
                raise ValueError(f"Wuji Hand {side} MJCF root body not found")
            pin_names = list(retargeter.optimizer.robot.dof_joint_names)
            pin_index = {name: index for index, name in enumerate(pin_names)}
            joint_names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
                for index in range(self.model.njnt)
            ]
            own = [index for index, name in enumerate(joint_names)
                   if name and name.startswith(f"{side}_")]
            try:
                permutation = np.asarray(
                    [pin_index[joint_names[index]] for index in own], dtype=int)
            except KeyError as exc:
                raise ValueError(
                    f"Wuji Hand {side} URDF/MJCF joint mismatch: {exc.args[0]}"
                ) from exc
            if len(own) != len(pin_names):
                raise ValueError(
                    f"Wuji Hand {side} DoF mismatch: URDF={len(pin_names)}, "
                    f"MJCF={len(own)}"
                )
            robot = retargeter.optimizer.robot
            wrist_id = robot.get_link_index(retargeter.optimizer.origin_link_name)
            robot.compute_forward_kinematics(np.zeros(len(pin_names), dtype=np.float64))
            wrist_local = robot.get_link_pose(wrist_id)[:3, 3].copy()
            adjustment = Rotation.from_euler(
                "xyz",
                [retargeter.rotation_xyz.get(axis, 0.0) for axis in ("x", "y", "z")],
                degrees=True,
            ).as_matrix()
            self.sides[side] = {
                "retargeter": retargeter,
                "root": root,
                "qpos_adr": np.asarray(
                    [self.model.jnt_qposadr[index] for index in own], dtype=int),
                "qpos_perm": permutation,
                "operator": (OPERATOR2MANO_RIGHT if side == "right"
                             else OPERATOR2MANO_LEFT),
                "adjustment": adjustment,
                "estimate_frame": estimate_frame_from_hand_points,
                "wrist_local": wrist_local,
            }

        self.third_position = None
        self.third_target = None
        self.third_up = None
        self.third_radius = 0.08
        self.third_camera_frusta = ()

    def place_floor(self, up: np.ndarray, hand_points: np.ndarray,
                    camera_points: np.ndarray) -> None:
        """摆地面与主光；手和相机都不动，因此像素对齐与改造前完全一致。"""
        center, quat, half = _floor_pose(up, hand_points, camera_points)
        # 位姿走 body_pos/body_quat（geom 挂在可动 body 上）：挂世界 body 的静态 geom
        # 改 geom_quat 后 mj_forward 不会重算 geom_xmat，朝向会永远停在编译值。
        self.model.body_pos[self.floor_body] = center - up * float(
            self.model.geom_size[self.floor_geom][2])          # box 顶面对齐地面高度
        self.model.body_quat[self.floor_body] = quat
        size = np.asarray(self.model.geom_size[self.floor_geom]).copy()
        size[0] = size[1] = half
        self.model.geom_size[self.floor_geom] = size
        from .mujoco_scene import robot_key_light_pose
        position, direction = robot_key_light_pose(up, hand_points)
        self.model.light_pos[self.light] = position
        self.model.light_dir[self.light] = direction

    def fit_third_camera(self, up: np.ndarray, hand_points: np.ndarray,
                         cameras: np.ndarray, image_size: tuple[int, int]) -> None:
        """Use the exact camera pose shared with the MuJoCo MANO renderer."""
        from .mujoco_scene import (
            fixed_third_camera_pose,
            prepare_fixed_camera_frusta,
        )

        result = fixed_third_camera_pose(
            up, hand_points, cameras,
            aspect=float(image_size[0]) / max(1.0, float(image_size[1])))
        self.third_position, self.third_target, self.third_up, self.third_radius = result
        self.third_camera_frusta = prepare_fixed_camera_frusta(
            cameras, self.third_radius)

    def _add_line(self, start: np.ndarray, end: np.ndarray,
                  width: float, rgba) -> None:
        scene = self.renderer.scene
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        self.mujoco.mjv_initGeom(
            geom, self.mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3), np.zeros(3), np.zeros(9),
            np.asarray(rgba, dtype=np.float32),
        )
        self.mujoco.mjv_connector(
            geom, self.mujoco.mjtGeom.mjGEOM_LINE, float(width),
            np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64),
        )
        scene.ngeom += 1

    def solve_pose(self, side: str, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """单手 retargeting，返回 MuJoCo qpos 与根节点位姿，不改共享场景。"""
        config = self.sides[side]
        retargeter = config["retargeter"]
        qpos = np.asarray(retargeter.retarget(joints), dtype=np.float64)
        if qpos.shape != config["qpos_perm"].shape or not np.isfinite(qpos).all():
            raise ValueError(
                f"Wuji Hand {side} retargeting returned invalid qpos {qpos.shape}"
            )
        centered = np.asarray(joints, dtype=np.float64) - joints[0]
        wrist_frame = config["estimate_frame"](centered)
        rotation = wrist_frame @ config["operator"] @ config["adjustment"].T
        translation = np.asarray(joints[0], dtype=np.float64) - \
            rotation @ config["wrist_local"]
        root_quat = np.zeros(4, dtype=np.float64)
        self.mujoco.mju_mat2Quat(
            root_quat, np.ascontiguousarray(rotation.reshape(-1)))
        return qpos[config["qpos_perm"]], translation, root_quat

    def apply_pose(self, side: str,
                   solution: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        config = self.sides[side]
        qpos, translation, root_quat = solution
        self.data.qpos[config["qpos_adr"]] = qpos
        self.model.body_pos[config["root"]] = translation
        self.model.body_quat[config["root"]] = root_quat

    def solve(self, side: str, joints: np.ndarray) -> None:
        """兼容单帧调用：解算后立即写入共享场景。"""
        self.apply_pose(side, self.solve_pose(side, joints))

    def render(self, validity: dict[str, bool], cam_c2w: np.ndarray,
               intrinsics: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        for side, group in _VISUAL_GROUP.items():
            self.option.geomgroup[group] = 1 if validity.get(side) else 0
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, scene_option=self.option)
        _configure_ego_camera(self.renderer.scene, cam_c2w, intrinsics, image_size)
        return np.asarray(self.renderer.render())

    def render_third(self, validity: dict[str, bool], current_frame: int,
                     image_size: tuple[int, int]) -> np.ndarray:
        from .mujoco_scene import (
            FIXED_THIRD_FOVY,
            configure_fixed_camera,
            fixed_camera_frusta,
        )

        if self.third_position is None:
            raise RuntimeError("Wuji Hand fixed third camera has not been fitted")
        for side, group in _VISUAL_GROUP.items():
            self.option.geomgroup[group] = 1 if validity.get(side) else 0
        self.mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, scene_option=self.option)
        configure_fixed_camera(
            self.renderer.scene, self.third_position, self.third_target,
            self.third_up, fovy=FIXED_THIRD_FOVY,
            aspect=float(image_size[0]) / max(1.0, float(image_size[1])))
        for start, end, width, rgba in fixed_camera_frusta(
                self.third_camera_frusta, current_frame):
            self._add_line(start, end, width, rgba)
        return np.asarray(self.renderer.render())

    def reset(self, side: str) -> None:
        self.sides[side]["retargeter"].reset()

    def close(self) -> None:
        self.renderer.close()


def _rounded_rect(image: np.ndarray, p0, p1, radius: int, color) -> None:
    x0, y0, x1, y1 = int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])
    radius = max(0, min(int(radius), (x1 - x0) // 2, (y1 - y0) // 2))
    if radius <= 0:
        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)
        return
    cv2.rectangle(image, (x0 + radius, y0), (x1 - radius, y1), color, -1, cv2.LINE_AA)
    cv2.rectangle(image, (x0, y0 + radius), (x1, y1 - radius), color, -1, cv2.LINE_AA)
    for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                   (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
        cv2.circle(image, (cx, cy), radius, color, -1, cv2.LINE_AA)


def _card(image: np.ndarray, rows, anchor, *, align: str = "tl",
          swatch: bool = False, alpha: float = _CARD_ALPHA):
    """圆角半透明信息卡：rows = [(文本, 颜色, 字号, 粗细)]；返回卡片左上/右下角。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad, gap, dot = 9, 6, 5
    swatch_gap = (dot * 2 + 8) if swatch else 0
    sizes = [cv2.getTextSize(text, font, scale, thick)[0]
             for text, _color, scale, thick in rows]
    width = max(size[0] for size in sizes) + pad * 2 + swatch_gap
    height = sum(size[1] for size in sizes) + gap * (len(rows) - 1) + pad * 2
    if align == "cc":                                    # 画面正中（无手提示用）
        x0, y0 = int(anchor[0]) - width // 2, int(anchor[1]) - height // 2
    else:
        x0 = int(anchor[0]) if "l" in align else int(anchor[0]) - width
        y0 = int(anchor[1]) if "t" in align else int(anchor[1]) - height
    x0 = max(0, min(image.shape[1] - width, x0))
    y0 = max(0, min(image.shape[0] - height, y0))

    roi = image[y0:y0 + height + 1, x0:x0 + width + 1]
    layer = roi.copy()
    _rounded_rect(layer, (0, 0), (width, height), 8, _CARD)
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    cursor = y0 + pad
    for (text, color, scale, thick), size in zip(rows, sizes):
        baseline = cursor + size[1]
        if swatch:
            cv2.circle(image, (x0 + pad + dot, baseline - size[1] // 3), dot,
                       color, -1, cv2.LINE_AA)
        cv2.putText(image, text, (x0 + pad + swatch_gap, baseline), font, scale,
                    color, thick, cv2.LINE_AA)
        cursor = baseline + gap
    return (x0, y0), (x0 + width, y0 + height)


def _decorate_frame(image: np.ndarray, validity: dict[str, bool],
                    label_text: str | None = None) -> np.ndarray:
    """Draw the method/source title without duplicating 2D hand-presence HUD."""
    image = np.ascontiguousarray(image)
    _height, width = image.shape[:2]
    compact = width <= 640
    title = ("WUJI RETARGET" if compact else "WUJI HAND  /  RETARGETING")
    if label_text:
        title += f"  /  {label_text}"
    subtitle = ("fixed third-person" if compact
                else "fixed third-person  |  start + live camera")
    _card(image, [(title, _TEXT, 0.54 if compact else 0.6, 2),
                  (subtitle, _MUTED, 0.36 if compact else 0.38, 1)],
          (16, 14), align="tl")
    if not any(validity.values()):
        _card(image, [("HANDS NOT DETECTED", _TEXT, 0.62, 2)],
              (width // 2, height // 2), align="cc", alpha=0.7)
    return image


def _solve_side_sequence(scene: _WujiHandScene, side: str, values: np.ndarray,
                         valid: np.ndarray, output_queue) -> None:
    """保持单手时序状态，把逐帧结果送进渲染流水；左右手各跑一条线程。"""
    previously_valid = False
    try:
        for frame, frame_valid in enumerate(valid):
            solution = None
            if frame_valid:
                if not previously_valid:
                    scene.reset(side)
                solution = scene.solve_pose(side, values[frame])
            previously_valid = bool(frame_valid)
            output_queue.put((frame, solution))
    except BaseException as exc:
        output_queue.put((None, exc))
        raise


def render_wuji_hand_video(
    world: dict,
    cam_c2w: np.ndarray,
    kept: np.ndarray,
    output: str | Path,
    *,
    fps: float,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    width: int = 960,
    height: int | None = None,
    label_text: str | None = None,
    on_step=None,
) -> Path:
    """Retarget model joints and render both hands from one fitted third-person view."""
    import queue

    from . import draw

    validity = np.asarray(kept, dtype=bool)
    if validity.ndim != 2 or validity.shape[1] != 2:
        raise ValueError(f"kept must have shape [T,2], got {validity.shape}")
    joints = {}
    for side in ("left", "right"):
        values = np.asarray((world.get(side) or {}).get("joints"), dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (21, 3):
            raise ValueError(
                f"{side} model joints must have shape [T,21,3], got {values.shape}"
            )
        joints[side] = values
    cameras = np.asarray(cam_c2w, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    source_size = tuple(map(int, image_size))
    frames = validity.shape[0]
    if (frames <= 0 or cameras.shape != (frames, 4, 4)
            or any(len(values) != frames for values in joints.values())):
        raise ValueError("Wuji retargeting camera, joints and validity frame counts differ")
    if K.shape != (3, 3) or len(source_size) != 2 or min(source_size) <= 0:
        raise ValueError("Wuji retargeting camera intrinsics or source size is invalid")

    # Match MuJoCo's upright/centered gauge before fitting the shared view and
    # solving the hand, so camera-origin frusta land at identical pixels.
    from .mujoco_scene import canonicalize_robot_coordinates
    all_points = np.concatenate([joints["left"], joints["right"]], axis=0)
    rotation, origin, cameras = canonicalize_robot_coordinates(cameras, all_points)
    for side in ("left", "right"):
        joints[side] = joints[side] @ rotation.T - origin

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    render_width = max(4, int(width))
    render_width += render_width % 2
    render_height = max(4, int(
        height if height is not None else render_width * source_size[1] / source_size[0]))
    render_height += render_height % 2
    scene = _WujiHandScene(_hand_body(), render_width, render_height)
    writer = None
    try:
        # 地面/主光只摆一次：手与相机都不旋转，像素对齐与改造前完全一致。
        up = _gravity_up(cameras)
        hand_points = np.concatenate([
            joints[side][validity[:, index]].reshape(-1, 3)
            if validity[:, index].any() else joints[side].reshape(-1, 3)
            for index, side in enumerate(("left", "right"))
        ], axis=0)
        hand_points = hand_points[np.all(np.isfinite(hand_points), axis=1)]
        camera_points = cameras[:, :3, 3]
        camera_points = camera_points[np.all(np.isfinite(camera_points), axis=1)]
        if not len(hand_points) or not len(camera_points):
            raise ValueError("Wuji retargeting 没有可用的手部或相机位置")
        scene.place_floor(up, hand_points, camera_points)
        scene.fit_third_camera(up, hand_points, cameras, (render_width, render_height))

        frame_validity = {
            side: validity[:, side_index] & np.all(np.isfinite(joints[side]), axis=(1, 2))
            for side_index, side in enumerate(("left", "right"))
        }
        writer = draw.H264PipeWriter(
            output, float(fps), (render_width, render_height),
            preset=_ROBOT_VIDEO_PRESET,
            buffered_frames=_ENCODE_BUFFER_FRAMES)
        solution_queues = {side: queue.SimpleQueue() for side in ("left", "right")}
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="retarget-hand") as pool:
            futures = {
                side: pool.submit(
                    _solve_side_sequence, scene, side, joints[side],
                    frame_validity[side], solution_queues[side])
                for side in ("left", "right")
            }
            for frame in range(frames):
                current_validity = {
                    side: bool(frame_validity[side][frame]) for side in ("left", "right")}
                for side in ("left", "right"):
                    solved_frame, solution = solution_queues[side].get()
                    if solved_frame is None:
                        raise solution
                    if solved_frame != frame:
                        raise RuntimeError(
                            f"Wuji Hand {side} retargeting 帧顺序错乱: "
                            f"expected {frame}, got {solved_frame}")
                    if solution is not None:
                        scene.apply_pose(side, solution)
                image = scene.render_third(
                    current_validity, frame, (render_width, render_height))
                writer.write(_decorate_frame(
                    image[:, :, ::-1], current_validity, label_text))
                if on_step is not None:
                    on_step(frame + 1, frames)
            for future in futures.values():
                future.result()
        writer.close()
        writer = None
    finally:
        if writer is not None:
            writer.close()
        scene.close()
    return output


__all__ = ["render_wuji_hand_video"]
