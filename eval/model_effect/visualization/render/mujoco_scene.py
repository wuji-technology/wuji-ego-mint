#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""世界系双手 MANO 网格 + 相机轨迹的 MuJoCo 渲染（离屏 mp4 / 交互 viewer）。

只负责「拿到世界系 verts/joints/相机外参」之后的 MuJoCo 场景搭建与渲染，与数据来源
(GT / 模型预测) 解耦——mujoco_view.py 复用 visualization 的解算链产出输入，再交本模块渲染。
对标 HaWoR「Hand and camera motion in world space」：左手蓝、右手粉，相机沿轨迹画紫色锥体。

关键点（MuJoCo mesh 逐帧变形）：
  MuJoCo 编译 mesh 时把顶点重表达到 mesh 自身参考系，渲染时 world = geom_xpos + geom_xmat @ local。
  故逐帧把「世界系顶点」写回 model.mesh_vert 前，需用编译期固定的 (R=geom_xmat, t=geom_xpos)
  逆变换成本地系：local = (W - t) @ R；法线同样按 R 旋回。写后 mjr_uploadMesh 重传 GPU。
  两只手各自一个 mesh，(R,t) 分别记录。
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")   # headless 服务器走 EGL 离屏渲染

import numpy as np
import mujoco

# 颜色（对齐 HaWoR 截图：左蓝 / 右粉 / 相机紫）。
LEFT_RGBA = (0.36, 0.55, 0.95, 1.0)
RIGHT_RGBA = (0.95, 0.55, 0.72, 1.0)
CAM_RGBA = (0.62, 0.40, 0.85, 1.0)
CUBE_RGBA = (0.88, 0.62, 0.24, 1.0)   # 参照方块（暖橙，与蓝手/粉手/灰垫布都拉得开）
FIXED_THIRD_FOVY = 42.0
FIXED_THIRD_VIEW_SHIFT_LEFT = 0.12
N_VERT = 778   # MANO 顶点数
# MuJoCo and Wuji retargeting use the same neutral background.  Keep these
# values here so the two XML scene builders cannot silently drift apart.
ROBOT_GROUND_RGB1 = (0.19, 0.23, 0.23)
ROBOT_GROUND_RGB2 = (0.34, 0.38, 0.37)
ROBOT_GROUND_TEXREPEAT = 12
ROBOT_GROUND_HALF_RANGE = (4.0, 8.0)
ROBOT_GROUND_RADIUS_SCALE = 1.5
ROBOT_GROUND_RADIUS_PADDING = 1.2
ROBOT_HEADLIGHT_AMBIENT = (0.40, 0.40, 0.40)
ROBOT_HEADLIGHT_DIFFUSE = (0.08, 0.08, 0.08)
ROBOT_HEADLIGHT_SPECULAR = (0.05, 0.05, 0.05)
ROBOT_KEY_DIFFUSE = (0.50, 0.50, 0.50)
ROBOT_KEY_SPECULAR = (0.10, 0.10, 0.10)
ROBOT_HAND_SPECULAR = 0.20
ROBOT_HAND_SHININESS = 0.30
ROBOT_SHADOWCLIP = 2.5
ROBOT_SHADOWSCALE = 1.0
ROBOT_ZNEAR = 0.02
ROBOT_ZFAR = 12.0
# ghost 残影的「向白混合」权重区间：最老 _GHOST_W0(最淡) → 最新 _GHOST_W1(最接近本色)。
_GHOST_W0, _GHOST_W1 = 0.28, 0.78
_EGO_FLOOR_HALF_RANGE = (12.0, 24.0)


def robot_ground_half_extent(points: np.ndarray, up=(0.0, 0.0, 1.0)) -> float:
    """Return the shared square ground half-size for MuJoCo and Wuji scenes."""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    values = values[np.all(np.isfinite(values), axis=1)]
    if not len(values):
        return float(ROBOT_GROUND_HALF_RANGE[0])
    normal = np.asarray(up, dtype=np.float64)
    normal /= max(1e-9, float(np.linalg.norm(normal)))
    # The mean is rotation-equivariant, unlike an axis-aligned bbox center;
    # MuJoCo first uprights the world while Wuji keeps the source basis.
    center = values.mean(axis=0)
    horizontal = values - np.outer((values - center) @ normal, normal)
    horizontal_center = center - normal * float(center @ normal)
    radius = float(np.percentile(
        np.linalg.norm(horizontal - horizontal_center, axis=1), 98.0))
    return float(np.clip(
        radius * ROBOT_GROUND_RADIUS_SCALE + ROBOT_GROUND_RADIUS_PADDING,
        *ROBOT_GROUND_HALF_RANGE))


def robot_key_light_pose(up: np.ndarray, hand_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the shared slanted key-light pose for both robot renderers."""
    normal = np.asarray(up, dtype=np.float64)
    normal /= max(1e-9, float(np.linalg.norm(normal)))
    lateral = np.cross(normal, np.asarray([1.0, 0.0, 0.0]))
    if float(np.linalg.norm(lateral)) < 1e-6:
        lateral = np.cross(normal, np.asarray([0.0, 1.0, 0.0]))
    lateral /= max(1e-9, float(np.linalg.norm(lateral)))
    direction = -normal + 0.35 * lateral + 0.2 * np.cross(normal, lateral)
    direction /= max(1e-9, float(np.linalg.norm(direction)))
    points = np.asarray(hand_points, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    center = points.mean(axis=0) if len(points) else np.zeros(3)
    return center + normal * 1.6 - lateral * 0.6, direction


def fixed_third_camera_pose(up: np.ndarray, hand_points: np.ndarray,
                            cameras: np.ndarray, *, aspect: float,
                            fovy: float = FIXED_THIRD_FOVY):
    """公共固定第三视角：采用 Wuji 的观察方向，只用完整双手范围决定取景大小。"""
    points = np.asarray(hand_points, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    camera_poses = np.asarray(cameras, dtype=np.float64)
    if not len(points) or camera_poses.ndim != 3:
        raise ValueError("固定第三视角缺少有效手点或相机位姿")
    up = np.asarray(up, dtype=np.float64)
    up /= max(1e-9, float(np.linalg.norm(up)))
    low, high = np.percentile(points, [1, 99], axis=0)
    target = (low + high) * 0.5
    radius = max(0.08, float(np.percentile(
        np.linalg.norm(points - target, axis=1), 99)))

    source_forward = camera_poses[:, :3, 2]
    source_forward = source_forward[np.all(np.isfinite(source_forward), axis=1)]
    source_forward = (source_forward.mean(axis=0) if len(source_forward)
                      else np.asarray([1.0, 0.0, 0.0]))
    source_forward -= up * float(source_forward @ up)
    if float(np.linalg.norm(source_forward)) < 1e-6:
        source_forward = np.cross(up, np.asarray([1.0, 0.0, 0.0]))
    source_forward /= max(1e-9, float(np.linalg.norm(source_forward)))
    right = np.cross(source_forward, up)
    right /= max(1e-9, float(np.linalg.norm(right)))
    observer = -source_forward + right * 0.45 + up * 0.38
    observer /= max(1e-9, float(np.linalg.norm(observer)))

    vertical_tangent = np.tan(np.radians(float(fovy)) / 2.0)
    limiting_tangent = min(vertical_tangent, vertical_tangent * float(aspect))
    distance = radius / max(0.1, limiting_tangent) * 1.12
    position = target + observer * distance
    # Pan the camera rig slightly toward screen-left.  Content near the left
    # crop edge then moves right into the safe area without zooming out.
    view_forward = target - position
    view_forward /= max(1e-9, float(np.linalg.norm(view_forward)))
    screen_right = np.cross(view_forward, up)
    screen_right /= max(1e-9, float(np.linalg.norm(screen_right)))
    shift = -screen_right * radius * FIXED_THIRD_VIEW_SHIFT_LEFT
    return position + shift, target + shift, up, radius


def _rotation_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= max(1e-9, float(np.linalg.norm(source)))
    target /= max(1e-9, float(np.linalg.norm(target)))
    cross = np.cross(source, target)
    cosine = float(np.dot(source, target))
    if float(np.linalg.norm(cross)) < 1e-8:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.asarray([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + cosine))


def canonicalize_robot_coordinates(cameras: np.ndarray,
                                    hand_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use one upright/centered gauge for MuJoCo and Wuji camera overlays."""
    transforms = np.asarray(cameras, dtype=np.float64).copy()
    points = np.asarray(hand_points, dtype=np.float64).reshape(-1, 3)
    finite_points = points[np.all(np.isfinite(points), axis=1)]
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("robot camera poses must have shape [T,4,4]")
    if not len(finite_points):
        finite_points = np.zeros((1, 3), dtype=np.float64)
    down = transforms[:, :3, 1]
    down = down[np.all(np.isfinite(down), axis=1)]
    if len(down) and float(np.linalg.norm(down.mean(axis=0))) >= 1e-6:
        up = -down.mean(axis=0)
        up /= max(1e-9, float(np.linalg.norm(up)))
    else:
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    rotation = _rotation_align(up, np.asarray([0.0, 0.0, 1.0]))
    transformed_points = finite_points @ rotation.T
    transforms[:, :3, :3] = rotation[None] @ transforms[:, :3, :3]
    transforms[:, :3, 3] = transforms[:, :3, 3] @ rotation.T
    origin = np.median(transformed_points, axis=0)
    transforms[:, :3, 3] -= origin
    return rotation, origin, transforms


def configure_fixed_camera(scene, position: np.ndarray, target: np.ndarray,
                           up: np.ndarray, *, fovy: float, aspect: float) -> None:
    """把同一台透视相机写入 MuJoCo MjvScene 的双目 GL camera。"""
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= max(1e-9, float(np.linalg.norm(forward)))
    camera_up = np.asarray(up, dtype=np.float64)
    camera_up = camera_up - forward * float(camera_up @ forward)
    camera_up /= max(1e-9, float(np.linalg.norm(camera_up)))
    for camera in scene.camera:
        camera.pos[:] = position
        camera.forward[:] = forward
        camera.up[:] = camera_up
        near = float(camera.frustum_near)
        half_height = np.tan(np.radians(float(fovy)) / 2.0) * near
        camera.frustum_center = 0.0
        camera.frustum_width = half_height * float(aspect)
        camera.frustum_bottom = -half_height
        camera.frustum_top = half_height


def prepare_fixed_camera_frusta(cameras: np.ndarray, hand_radius: float) -> dict:
    """Prepare one fixed start marker plus the live camera marker."""
    transforms = np.asarray(cameras, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        return {}
    valid_indices = np.flatnonzero(np.all(np.isfinite(transforms), axis=(1, 2)))
    if not len(valid_indices):
        return {}
    return {
        "transforms": transforms,
        "valid_indices": valid_indices,
        "size": max(0.04, float(hand_radius)) * 0.18,
    }


def fixed_camera_frusta(prepared: dict, current_frame: int) -> tuple:
    """Return the faded fixed start frustum and the current live frustum."""
    if not prepared:
        return ()
    transforms = prepared["transforms"]
    valid_indices = prepared["valid_indices"]
    valid_end = int(np.searchsorted(
        valid_indices, int(current_frame), side="right")) - 1
    if valid_end < 0:
        valid_end = 0
    current_index = int(valid_indices[valid_end])

    # At the first frame both candidates overlap, so only live is visible.
    selected = {}
    for index, alpha in (
            (int(valid_indices[0]), 0.18),
            (current_index, 0.98)):
        selected[index] = alpha

    size = float(prepared["size"])
    width = 2.0
    half = size * 0.6
    lines = []
    for index, alpha in selected.items():
        transform = transforms[index]
        center, rotation = transform[:3, 3], transform[:3, :3]
        corners = (rotation @ np.asarray([
            [half, half, size], [-half, half, size],
            [-half, -half, size], [half, -half, size],
        ]).T).T + center
        rgba = (*CAM_RGBA[:3], alpha)
        for corner in corners:
            lines.append((center.copy(), corner.copy(), width, rgba))
        for corner in range(4):
            lines.append((corners[corner].copy(),
                          corners[(corner + 1) % 4].copy(), width, rgba))
    return tuple(lines)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """按面法线累加求逐顶点法线 (V,3)，单位化。"""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    tri = verts[faces]                                   # (F,3,3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    indices = faces.reshape(-1)
    repeated = np.repeat(fn, 3, axis=0)
    n = np.stack([
        np.bincount(indices, weights=repeated[:, axis], minlength=len(verts))
        for axis in range(3)
    ], axis=1)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return (n / ln).astype(np.float32)


def _mesh_xml(name: str, verts: np.ndarray, faces: np.ndarray) -> str:
    v = " ".join(f"{x:.6f}" for x in np.asarray(verts, dtype=np.float64).reshape(-1))
    f = " ".join(str(int(x)) for x in np.asarray(faces).reshape(-1))
    return f'<mesh name="{name}" vertex="{v}" face="{f}"/>'


def _rgba(c) -> str:
    return " ".join(f"{v:.4f}" for v in c)


class HandWorldScene:
    """双手 MANO 网格 + 相机锥体轨迹的世界系 MuJoCo 场景。

    faces_left/right: (F,3) MANO 面片；init_l/init_r: (778,3) 用于编译的初始世界顶点(某帧)。
    operation_mat: 是否在地面上增加操作垫；Web 视频关闭它，只保留远处地面。
    ego_floor_screen_level: 第一人称地面消除相机滚转，保持画面中的地平线水平。
    cube_size: >0 时在垫布上放一个该边长(米)的**参照方块**，见 place_cube。
    """

    def __init__(self, faces_left, faces_right, init_l, init_r, *,
                 width=1280, height=720, floor=True, operation_mat=True,
                 ego_floor_screen_level=False, n_ghost=0, cube_size=0.0):
        self.faces = {"left": np.asarray(faces_left), "right": np.asarray(faces_right)}
        self.width, self.height = int(width), int(height)
        self.n_ghost = max(0, int(n_ghost))
        self.cube_size = max(0.0, float(cube_size))
        self._ego_floor_screen_level = bool(ego_floor_screen_level)

        # 对标 HaWoR：单块有限地台 "floor"（手落其上带阴影）。
        # 关键：地面用薄 box 而非 plane —— 方向光在无限/超大 plane 上会产生掠射 shadow acne 噪点，
        # 薄 box 有限边界让 shadow 视锥贴合场景 → 阴影干净。尺寸/位置随手部范围在 place_floor 里再调。
        # （曾用「浅灰地面 + 深灰垫布」两层，但两薄 box 近共面必 z-fight 出棋盘纹，改单块最稳、观感同样干净。）
        # 地台 geom 挂在独立 body 上（而非直接挂世界 body）：
        # MuJoCo 对挂世界 body(id=0) 的静态 geom，运行时改 model.geom_quat 后 mj_forward
        # 不会重算 data.geom_xmat（朝向永远停在编译值）——ego 背景板要「转向相机」就失效、板子恒水平。
        # 改挂到可动 body、位姿走 body_pos/body_quat，kinematics 才会把朝向正确传到 geom_xmat。
        # 两层：floorb=浅色衬底（ego 铺满画面的背景板 / orbit 大块水平地面）；
        #       matb =有边界的操作垫布（居中在双手下方/前方，比衬底略深，读出「在垫布上操作」）。
        floor_xml = (
            '<body name="floorb" pos="0 0 0">'
            '<geom name="floor" type="box" size="0.6 0.6 0.02" material="ground" '
            'contype="0" conaffinity="0"/></body>' if floor else ""
        )
        mat_xml = (
            '<body name="matb" pos="0 0 0">'
            '<geom name="mat" type="box" size="0.3 0.3 0.02" material="mat" '
            'contype="0" conaffinity="0"/></body>'
            if floor and operation_mat else ""
        )
        # 手部运动残影(ghost)：对标 HaWoR「手沿轨迹留一串由淡到实」。左右各 n_ghost 个额外 mesh 槽，
        # 逐帧把抽出的历史帧顶点 bake 进去。ghost geom **不挂 material、直接写 rgba**：主手靠
        # 「material + geom_rgba 非默认时覆盖」的机制上色(见 _set_visible)，而 ghost 要逐帧改深浅，
        # 走 geom_rgba 更直接。用**不透明浅色**(向白混合)而非 alpha 半透明——既避开 MuJoCo 半透明
        # 排序问题，观感也更接近 HaWoR 那种由淡到实的浅色渐变。
        ghost_assets = "".join(
            _mesh_xml(f"hand_Lg{i}", init_l, faces_left) + _mesh_xml(f"hand_Rg{i}", init_r, faces_right)
            for i in range(self.n_ghost)
        )
        ghost_bodies = "".join(
            f'<body name="bLg{i}"><geom name="gLg{i}" type="mesh" mesh="hand_Lg{i}" '
            f'rgba="{_rgba(LEFT_RGBA)}" contype="0" conaffinity="0"/></body>'
            f'<body name="bRg{i}"><geom name="gRg{i}" type="mesh" mesh="hand_Rg{i}" '
            f'rgba="{_rgba(RIGHT_RGBA)}" contype="0" conaffinity="0"/></body>'
            for i in range(self.n_ghost)
        )
        # 参照方块（cube_size>0 才建）：真实尺寸的静态 box，坐在垫布上、朝向对齐操作者前向。
        # **纯视觉参照物，不做物理**——contype/conaffinity=0 与地台一致，手会直接穿过它。
        # 用途：① 一眼读出场景**尺度**对不对（5cm 方块 vs 手掌，比看数字直观）；
        #       ② 相机移动时它在画面里应**纹丝不动**，若跟着手/相机漂 → 世界系解算有问题。
        # 位姿在 place_cube 里定；这里 pos 先给 0，尺寸写死进 geom_size。
        ch = self.cube_size / 2.0
        cube_xml = (
            f'<body name="cubeb" pos="0 0 0">'
            f'<geom name="cube" type="box" size="{ch:.6f} {ch:.6f} {ch:.6f}" material="mcube" '
            f'contype="0" conaffinity="0"/></body>' if self.cube_size > 0 else ""
        )
        xml = f"""
<mujoco model="hand_world">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="{self.width}" offheight="{self.height}" azimuth="140" elevation="20"/>
    <quality shadowsize="4096" offsamples="2"/>
    <!-- 环境光压低、方向光提亮 → 手在地面上的投影才有明暗差看得见。 -->
    <headlight ambient="{' '.join(f'{v:.2f}' for v in ROBOT_HEADLIGHT_AMBIENT)}" diffuse="{' '.join(f'{v:.2f}' for v in ROBOT_HEADLIGHT_DIFFUSE)}" specular="{' '.join(f'{v:.2f}' for v in ROBOT_HEADLIGHT_SPECULAR)}"/>
    <!-- shadowclip 收紧贴合场景、shadowscale 收窄阴影视锥 → 去掉 shadow acne 噪点、
         让方向光在垫布上投出干净的接触阴影（原 shadowclip=10 太大导致 shadow map 分辨率浪费、满屏噪点）。
         取景改为「手 + 相机轨迹一起入画」后场景跨度变大（~1m），1.6 会把远端阴影裁掉，放宽到 2.5。 -->
    <map shadowclip="{ROBOT_SHADOWCLIP}" shadowscale="{ROBOT_SHADOWSCALE}" znear="{ROBOT_ZNEAR}" zfar="{ROBOT_ZFAR}"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.95 0.96 0.98" rgb2="0.78 0.82 0.88"
             width="512" height="512"/>
    <texture name="ground_grid" type="2d" builtin="checker" rgb1="{' '.join(f'{v:.2f}' for v in ROBOT_GROUND_RGB1)}"
             rgb2="{' '.join(f'{v:.2f}' for v in ROBOT_GROUND_RGB2)}" width="512" height="512"/>
    <!-- 衬底(浅)与操作垫布(略深)分两材质：垫布比衬底暗一档，边界读得出、投影落上去也有明暗差。 -->
    <material name="ground" texture="ground_grid" texrepeat="{ROBOT_GROUND_TEXREPEAT} {ROBOT_GROUND_TEXREPEAT}" texuniform="true"
              rgba="1 1 1 1" reflectance="0.0" specular="0.0" shininess="0.0"/>
    <material name="mat" rgba="0.55 0.56 0.61 1.0" reflectance="0.0" specular="0.0" shininess="0.0"/>
    <material name="mL" rgba="{_rgba(LEFT_RGBA)}" specular="{ROBOT_HAND_SPECULAR}" shininess="{ROBOT_HAND_SHININESS}"/>
    <material name="mR" rgba="{_rgba(RIGHT_RGBA)}" specular="{ROBOT_HAND_SPECULAR}" shininess="{ROBOT_HAND_SHININESS}"/>
    <!-- 参照方块：暖橙，与蓝手/粉手/灰垫布都拉开；高光稍强 → 三个面明暗分明，一眼看出是立方体。 -->
    <material name="mcube" rgba="{_rgba(CUBE_RGBA)}" specular="0.35" shininess="0.45"/>
    {_mesh_xml("hand_L", init_l, faces_left)}
    {_mesh_xml("hand_R", init_r, faces_right)}
    {ghost_assets}
  </asset>
  <worldbody>
    <light name="key" pos="0.3 0.3 2" dir="-0.2 -0.2 -1" directional="true" castshadow="true"
           diffuse="{' '.join(f'{v:.2f}' for v in ROBOT_KEY_DIFFUSE)}" specular="{' '.join(f'{v:.2f}' for v in ROBOT_KEY_SPECULAR)}"/>
    {floor_xml}
    {mat_xml}
    {cube_xml}
    <body name="bL"><geom name="gL" type="mesh" mesh="hand_L" material="mL"
          contype="0" conaffinity="0"/></body>
    <body name="bR"><geom name="gR" type="mesh" mesh="hand_R" material="mR"
          contype="0" conaffinity="0"/></body>
    {ghost_bodies}
    <camera name="ego" pos="0 0 1" fovy="60"/>
  </worldbody>
</mujoco>"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self._floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor") if floor else -1
        self._floor_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "floorb") if floor else -1
        self._mat_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "mat") \
            if floor and operation_mat else -1
        self._mat_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "matb") \
            if floor and operation_mat else -1
        # 参照方块 id（cube_size=0 时为 -1，place_cube 直接空转）。方块挂独立 body、位姿走
        # body_pos/body_quat，理由同地台：挂世界 body 的静态 geom 改 geom_quat 不会重算 geom_xmat。
        self._cube_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube") \
            if self.cube_size > 0 else -1
        self._cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cubeb") \
            if self.cube_size > 0 else -1
        self._cube_center = None            # place_cube 填入；ego 背景板要退到方块之后才不会吞掉它
        self._cube_rad = 0.0
        self._mat_ground = None             # place_floor 记录垫布水平位姿，供 ego/orbit 切换还原
        self._hand_rad = 0.12               # place_floor 填入手部 3D 半径（ego 垫布尺寸/景深用）
        self._ego_cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
        # 方向光 id + 初始 dir/pos（ego 背景板会临时把光转向相机光轴，其它视角要还原）。
        self._light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, "key")
        self._light_dir0 = np.array(self.model.light_dir[self._light_id]).copy() if self._light_id >= 0 else None
        self._light_pos0 = np.array(self.model.light_pos[self._light_id]).copy() if self._light_id >= 0 else None
        self._floor_ground = None           # place_floor 记录水平地面位姿，供 ego/orbit 间切换还原
        self._bb_back = 0.25                 # 背景板放在双手中心后方多远（m），保证手不穿出板
        self._bb_margin = 2.2                 # 背景板外扩系数：需 >√(1+aspect²)≈2.04 才能在相机滚转下仍盖满四角

        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._fit_center = np.zeros(3)                       # autofit_camera 填入（供多视角 render_free 共用）
        self._fit_dist = 1.0
        self._mat_center = np.zeros(3)                       # place_floor 填入垫布中心/半尺寸，供取景留白
        self._mat_half = np.array([0.3, 0.3])

        # 记录每只手 mesh 的编译期刚体参考系 (R,t)、顶点段区间、geom id。
        self._mesh = {}
        for side, mesh_name, geom_name in (("left", "hand_L", "gL"), ("right", "hand_R", "gR")):
            self._mesh[side] = self._mesh_info(mesh_name, geom_name)
        # ghost 槽同样登记（key: "left#0" / "right#0" …），(R,t) 各自记录不共用。
        for i in range(self.n_ghost):
            self._mesh[f"left#{i}"] = self._mesh_info(f"hand_Lg{i}", f"gLg{i}")
            self._mesh[f"right#{i}"] = self._mesh_info(f"hand_Rg{i}", f"gRg{i}")
        self.hide_ghosts()

    def _mesh_info(self, mesh_name: str, geom_name: str) -> dict:
        """查一个 mesh/geom 的编译期参考系与顶点段（逐帧 _bake 要用）。"""
        mid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        v0 = int(self.model.mesh_vertadr[mid])
        n0 = int(self.model.mesh_normaladr[mid])
        return {
            "mid": mid, "gid": gid,
            "vslice": slice(v0, v0 + int(self.model.mesh_vertnum[mid])),
            "nslice": slice(n0, n0 + int(self.model.mesh_normalnum[mid])),
            "R": np.array(self.data.geom_xmat[gid]).reshape(3, 3),
            "t": np.array(self.data.geom_xpos[gid]),
            "rgba0": np.array(self.model.geom_rgba[gid]).copy(),
        }

    # ---- 逐帧写世界顶点（逆变换到 mesh 本地系）+ 重传 GPU ----
    def _bake(self, side: str, world_verts: np.ndarray):
        info = self._mesh[side]
        faces = self.faces[side.split("#")[0]]                # ghost key 形如 "left#0"，面片与本手同
        R, t = info["R"], info["t"]
        W = np.nan_to_num(np.asarray(world_verts, dtype=np.float64), nan=0.0)
        local = (W - t) @ R                                  # world = t + R@local  =>  local = R^T (W-t)
        nrm = vertex_normals(W, faces) @ R                   # 法线只旋转
        self.model.mesh_vert[info["vslice"]] = local.astype(np.float32)
        if info["nslice"].stop - info["nslice"].start == local.shape[0]:
            self.model.mesh_normal[info["nslice"]] = nrm.astype(np.float32)
        mujoco.mjr_uploadMesh(self.model, self.renderer._mjr_context, info["mid"])

    def _set_visible(self, side: str, visible: bool):
        info = self._mesh[side]
        a = info["rgba0"].copy()
        a[3] = info["rgba0"][3] if visible else 0.0
        self.model.geom_rgba[info["gid"]] = a

    def place_floor(self, points: np.ndarray, margin=0.03, fwd_az=None,
                    extent_points: np.ndarray | None = None):
        """把浅灰地面与深灰垫布放到手部点云正下方：地面接地、垫布居中且按手水平范围定尺寸。

        地面/垫布是薄 box（避免 plane 的掠射 shadow acne），box 的顶面 = 中心 z + 半高，
        故摆放时中心 z 要减去半高，让顶面落在目标接触高度 zmin。
        fwd_az（弧度，可选）：操作者前向在世界 XY 的方位角。给了就把垫布绕 z 转到「操作者坐标系」
        （长短边对齐前向/右向），第三人称各方位看过去垫布才和 ego 投影方向一致；否则退化为世界轴对齐。
        """
        if self._floor_gid < 0:
            return
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if not len(pts):
            return
        extent = pts if extent_points is None else np.asarray(
            extent_points, dtype=np.float64).reshape(-1, 3)
        extent = extent[np.all(np.isfinite(extent), axis=1)]
        if not len(extent):
            extent = pts
        # 稳健包围盒（2–98 百分位）定手水平运动范围，避免无效帧/离群点把垫布撑大。
        cxy = pts[:, :2].mean(axis=0)
        lo = np.percentile(pts, 2, axis=0)
        hi = np.percentile(pts, 98, axis=0)
        half = (hi[:2] - lo[:2]) / 2.0
        # z 下界用近最低分位（0.1%）：传入的是网格顶点，取近最低点让手最低接触帧真正贴地、
        # 又避开个别离群顶点；再减 margin 留极小间隙，保证网格不穿出地台。
        zmin = float(np.percentile(pts[:, 2], 0.1)) - margin
        # 地台(薄 box)：顶面落在 zmin（手落其上）、水平居中、按手运动范围放大成一大块地台。
        # 位姿走 body_pos/body_quat（geom 挂 floorb，见 __init__ 注释）；尺寸仍写 geom_size。
        # 衬底**必须和垫布一样绕 z 对齐 fwd_az**：它虽然够大总能盖住，但边缘会露在画面里——
        # 世界轴对齐时正方形的一个角朝着镜头，边缘斜切过画面，观感上「桌子是歪的/不水平」。
        # 对齐后从 front/back/left/right 四个正方位看过去，衬底边缘平行/垂直于画面，才读得出水平。
        yaw = float(fwd_az) - np.pi / 2.0 if fwd_az is not None else 0.0   # 局部 x = 操作者右向
        ground_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        fh = float(self.model.geom_size[self._floor_gid][2])          # 地台 box 半高
        bp = np.array([cxy[0], cxy[1], zmin - fh])                    # 顶面 = 中心 + 半高 = zmin
        self.model.body_pos[self._floor_bid] = bp
        self.model.body_quat[self._floor_bid] = ground_quat
        fs = np.array(self.model.geom_size[self._floor_gid])
        shared_half = robot_ground_half_extent(extent)
        fs[0] = fs[1] = shared_half
        self.model.geom_size[self._floor_gid] = fs
        # _mat_center/_mat_half 记手运动区域中心与半范围（供 mujoco_view 取景留白用；非地台整体尺寸）。
        self._mat_center = np.array([cxy[0], cxy[1], zmin])
        self._mat_half = half.copy()
        # 手部 3D 半径（稳健 90 分位）：ego 垫布尺寸/景深按它派生，保证垫布刚好裹住双手。
        self._hand_rad = float(np.percentile(np.linalg.norm(pts - pts.mean(0), axis=1), 90))
        light_pos, light_dir = robot_key_light_pose(np.asarray([0.0, 0.0, 1.0]), pts)
        if self._light_id >= 0:
            self.model.light_pos[self._light_id] = light_pos
            self.model.light_dir[self._light_id] = light_dir
        # 操作垫布(matb)：比衬底小一圈(裹住手运动范围)、顶面略高于衬底避免 z-fight。
        # 有 fwd_az 时按操作者坐标系(右向 ex、前向 ey)量取半范围并绕 z 旋转，使垫布对齐操作方向。
        if fwd_az is not None:
            ey = np.array([np.cos(fwd_az), np.sin(fwd_az)])           # 前向 → 垫布局部 y
            ex = np.array([ey[1], -ey[0]])                            # 右向 → 垫布局部 x（前向顺时针 90°）
            d = pts[:, :2] - cxy
            px, py = d @ ex, d @ ey
            hx = float(np.percentile(px, 98) - np.percentile(px, 2)) / 2.0
            hy = float(np.percentile(py, 98) - np.percentile(py, 2)) / 2.0
            cx_off = (np.percentile(px, 98) + np.percentile(px, 2)) / 2.0
            cy_off = (np.percentile(py, 98) + np.percentile(py, 2)) / 2.0
            mcxy = cxy + cx_off * ex + cy_off * ey                    # 旋转系里的真中心
            mh = np.array([hx * 1.4 + 0.05, hy * 1.4 + 0.05, 0.02])
            mquat = ground_quat.copy()                                # 与衬底同一个绕 z 朝向
        else:
            mcxy = cxy
            mh = np.array([float(half[0]) * 1.4 + 0.04, float(half[1]) * 1.4 + 0.04, 0.02])
            mquat = np.array([1.0, 0.0, 0.0, 0.0])
        mbp = np.array([mcxy[0], mcxy[1], zmin + 0.001 - mh[2]])       # 顶面 = zmin+0.001（贴在衬底上方一点）
        if self._mat_gid >= 0:
            self.model.geom_size[self._mat_gid] = mh
            self.model.body_pos[self._mat_bid] = mbp
            self.model.body_quat[self._mat_bid] = mquat
        # 记录水平地面位姿（orbit/top 用）；ego 会临时改成背景板，渲完各视角前按需切回。
        self._floor_ground = {
            "pos": bp.copy(),
            "quat": ground_quat.copy(),
            "size": np.array(self.model.geom_size[self._floor_gid]).copy(),
            "surface": np.array([cxy[0], cxy[1], zmin]),
        }
        self._mat_ground = ({
            "pos": mbp.copy(),
            "quat": mquat.copy(),
            "size": mh.copy(),
        } if self._mat_gid >= 0 else None)
        mujoco.mj_forward(self.model, self.data)

    def place_cube(self, points: np.ndarray, *, offset=None, fwd_az=None, margin=0.03, gap=0.04):
        """把参照方块坐在垫布表面。offset=None（默认）时自动摆到**手部活动范围前方、不碰手**。

        points 传**与 place_floor 相同的网格顶点**、margin 也传相同值：这里按同一口径复算落地高度
        （2–98 分位定水平中心、0.1 分位减 margin 定 zmin、顶面再 +0.001 对齐垫布），
        方块底面就正好贴在垫布上，不会陷进去或浮起来。朝向与垫布同一个绕 z 旋转，画面才整齐。

        offset=None：沿操作者前向退到手部前边缘（前向 98 分位）之外 gap+半边长处——摆在手正中央
        会被双手穿过去、糊成一团（方块无碰撞，见下），前置才既看得全又不打架。
        offset=(右, 前, 上)：显式指定，单位米，在「操作者坐标系」里相对手部活动中心偏移。

        方块**只是参照物**（无碰撞、不参与物理，手会直接穿过）：
          · 尺度自检——已知边长的方块摆在手边，一眼看出手/场景大小是否合理；
          · 世界系自检——它在世界里完全静止。第三人称看它必须钉死；ego 第一人称里它**会随相机滑动**，
            那是对的（相机在动），若它反而跟着手走，说明解算有问题。
            ⚠ 别拿 ego 的垫布当参照：ego 下垫布是每帧正对相机的假板（set_floor_billboard），
            它跟着相机转，方块不转，二者必然相对滑动。
        """
        if self._cube_gid < 0:
            return
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if not len(pts):
            return
        lo = np.percentile(pts, 2, axis=0)
        hi = np.percentile(pts, 98, axis=0)
        cxy = (lo[:2] + hi[:2]) / 2.0
        zmin = float(np.percentile(pts[:, 2], 0.1)) - float(margin)
        ztop = zmin + 0.001                                        # 垫布顶面（见 place_floor）
        half = float(self.model.geom_size[self._cube_gid][2])      # box 半边长
        if fwd_az is not None:
            ey = np.array([np.cos(fwd_az), np.sin(fwd_az)])        # 操作者前向
            ex = np.array([ey[1], -ey[0]])                         # 操作者右向（前向顺时针 90°）
        else:
            ex, ey = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        if offset is None:                                          # 自动前置：退到手前边缘外
            fwd_edge = float(np.percentile((pts[:, :2] - cxy) @ ey, 98))
            ox, oy, oz = 0.0, fwd_edge + float(gap) + half, 0.0
        else:
            ox, oy, oz = (float(v) for v in offset)
        c = cxy + ox * ex + oy * ey
        yaw = float(fwd_az) - np.pi / 2.0 if fwd_az is not None else 0.0   # 与衬底/垫布同朝向
        pos = np.array([c[0], c[1], ztop + half + oz])
        self.model.body_pos[self._cube_bid] = pos
        self.model.body_quat[self._cube_bid] = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        # 记给 set_floor_billboard：ego 的背景板必须退到方块**之后**，否则方块被板吞掉看不见。
        self._cube_center = pos.copy()
        self._cube_rad = half * np.sqrt(3.0)                        # 外接球半径（板要绕开整个方块）
        mujoco.mj_forward(self.model, self.data)
        return pos

    def extend_mat_for_cube(self, pad=0.025):
        """把操作垫布扩到能托住参照方块。

        垫布尺寸是按**手的活动范围**算的，而方块默认摆在该范围之外（前置避让），于是常有
        半个方块悬在垫布外、看着像踩空。这里只沿需要的方向单边外扩（另一侧不动），
        垫布仍紧贴手的活动区，不会平白撑大一圈。
        """
        if self._cube_gid < 0 or self._mat_gid < 0 \
                or self._mat_ground is None or self._cube_center is None:
            return
        q = self._mat_ground["quat"]
        yaw = 2.0 * np.arctan2(float(q[3]), float(q[0]))
        ex = np.array([np.cos(yaw), np.sin(yaw)])              # 垫布局部 x（=操作者右向）
        ey = np.array([-np.sin(yaw), np.cos(yaw)])             # 垫布局部 y（=操作者前向）
        mpos = self._mat_ground["pos"].copy()
        msz = self._mat_ground["size"].copy()
        d = self._cube_center[:2] - mpos[:2]
        r = float(self.model.geom_size[self._cube_gid][0]) + float(pad)   # 方块与垫布同 yaw → 局部轴对齐
        shift = np.zeros(2)
        for k, e in ((0, ex), (1, ey)):
            p = float(d @ e)
            lo, hi = min(-msz[k], p - r), max(msz[k], p + r)
            msz[k] = (hi - lo) / 2.0
            shift = shift + ((hi + lo) / 2.0) * e
        mpos[:2] = mpos[:2] + shift
        self.model.body_pos[self._mat_bid] = mpos
        self.model.geom_size[self._mat_gid] = msz
        self._mat_ground["pos"] = mpos.copy()                  # set_floor_ground 按它还原
        self._mat_ground["size"] = msz.copy()
        mujoco.mj_forward(self.model, self.data)

    def set_floor_ground(self):
        """把衬底+垫布切回世界水平地面（orbit/top 视角用），并还原方向光。"""
        if self._floor_gid < 0 or self._floor_ground is None:
            return
        self.model.body_pos[self._floor_bid] = self._floor_ground["pos"]
        self.model.body_quat[self._floor_bid] = self._floor_ground["quat"]
        self.model.geom_size[self._floor_gid] = self._floor_ground["size"]
        if self._mat_ground is not None:                            # 垫布也切回水平
            self.model.body_pos[self._mat_bid] = self._mat_ground["pos"]
            self.model.body_quat[self._mat_bid] = self._mat_ground["quat"]
            self.model.geom_size[self._mat_gid] = self._mat_ground["size"]
        if self._light_id >= 0:
            self.model.light_dir[self._light_id] = self._light_dir0
            self.model.light_pos[self._light_id] = self._light_pos0
        mujoco.mj_forward(self.model, self.data)

    def set_floor_ego_ground(self, c2w, fovy_deg):
        """布置第一人称地面；Web 模式额外消除相机滚转造成的斜地平线。"""
        if self._floor_gid < 0 or self._floor_ground is None:
            return
        c2w = np.asarray(c2w, dtype=np.float64)
        C = c2w[:3, 3]
        f = c2w[:3, 2] / (np.linalg.norm(c2w[:3, 2]) + 1e-9)
        fh = float(self.model.geom_size[self._floor_gid][2])
        if self._ego_floor_screen_level:
            # 从世界竖直方向中移除相机画面横轴分量：手/相机投影完全不动，
            # 只让参考地面的地平线在第一人称画面内始终水平。
            right = c2w[:3, 0] / (np.linalg.norm(c2w[:3, 0]) + 1e-9)
            normal = np.array([0.0, 0.0, 1.0])
            normal = normal - right * float(np.dot(normal, right))
            if np.linalg.norm(normal) < 1e-6:
                normal = -c2w[:3, 1]
            normal = normal / (np.linalg.norm(normal) + 1e-9)
            if normal[2] < 0:
                normal = -normal
            x_axis = right
            y_axis = np.cross(normal, x_axis)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
            x_axis = np.cross(y_axis, normal)
            rotation = np.column_stack([x_axis, y_axis, normal])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation.reshape(-1)))

            surface = self._floor_ground["surface"]
            denom = float(np.dot(f, normal))
            t = float(np.dot(surface - C, normal) / denom) if abs(denom) > 1e-4 else -1.0
            if t > 0:
                hit = C + f * t
            else:
                distance = max(0.8, float(np.linalg.norm(surface - C)))
                target = C + f * distance
                hit = target + normal * float(np.dot(surface - target, normal))
                t = float(np.linalg.norm(hit - C))
            # 浅俯角和宽视场下，画面底角与地面的交点远大于中心光轴交点。地面至少保留
            # 12 m 半边，并按光轴交点扩到 24 m，避免操作过程中手下方露出有限 box 的边缘。
            half = float(np.clip(
                abs(t) * 4.0 + 4.0, *_EGO_FLOOR_HALF_RANGE))
            self.model.body_pos[self._floor_bid] = hit - normal * fh
            self.model.body_quat[self._floor_bid] = quat
        else:
            base = self._floor_ground["pos"].copy()
            ground_z = base[2] + fh
            if f[2] < -1e-3:
                t = float((ground_z - C[2]) / f[2])
                hit = C + f * t
            else:
                t = max(0.3, float(np.linalg.norm(C[:2] - self._mat_center[:2])))
                hit = np.array([self._mat_center[0], self._mat_center[1], ground_z])
            half = float(np.clip(
                abs(t) * 2.2 + 0.4, float(self._floor_ground["size"][0]), 3.0))
            self.model.body_pos[self._floor_bid] = np.array([hit[0], hit[1], base[2]])
            self.model.body_quat[self._floor_bid] = self._floor_ground["quat"]
        fs = np.array(self.model.geom_size[self._floor_gid])
        fs[0] = fs[1] = half
        self.model.geom_size[self._floor_gid] = fs
        if self._mat_ground is not None:                         # 垫布回到（并留在）世界原位
            self.model.body_pos[self._mat_bid] = self._mat_ground["pos"]
            self.model.body_quat[self._mat_bid] = self._mat_ground["quat"]
            self.model.geom_size[self._mat_gid] = self._mat_ground["size"]
        if self._light_id >= 0:                                  # 顶光还原：水平地面要顶光才有正常阴影
            self.model.light_dir[self._light_id] = self._light_dir0
            self.model.light_pos[self._light_id] = self._light_pos0
        mujoco.mj_forward(self.model, self.data)

    def set_floor_billboard(self, c2w, fovy_deg):
        """ego 用：衬底=正对相机、铺满画面的背景板；垫布=正对相机、裹住双手的有边界矩形(四周露衬底)。

        双手已居中到世界原点附近；相机沿光轴 f 看向双手。两块都用相机旋转矩阵定朝向
        （box 局部 z=薄轴 对齐相机 z=光轴 → 大面正对镜头）：
          · 垫布放在双手最深顶点之后一点(Dm)，尺寸按手部半径派生并留四周衬底；
          · 衬底(背景板)再退到垫布之后(D)，放大到铺满整幅画面。
        方向光临时沿光轴照向板面，手影干净投在垫布/背景板上（竖直板+顶光会掠射生噪）。
        """
        if self._floor_gid < 0:
            return
        c2w = np.asarray(c2w, dtype=np.float64)
        C = c2w[:3, 3]
        f = c2w[:3, 2] / (np.linalg.norm(c2w[:3, 2]) + 1e-9)      # OpenCV +z 前向
        d_hand = max(0.2, float(np.dot(-C, f)))                    # 相机→双手中心沿光轴距离
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(c2w[:3, :3].reshape(-1)))
        tan_half = float(np.tan(np.radians(fovy_deg) / 2.0))
        aspect = self.width / self.height

        # 垫布：退到双手最深顶点(≈d_hand+_hand_rad)之后 0.03，裹住双手且四周留衬底。
        Dm = d_hand + self._hand_rad + 0.03
        # 参照方块摆在手前方 → 沿光轴比手更远，垫布/背景板必须再退到它之后，否则方块被板吞掉。
        if self._cube_center is not None:
            Dm = max(Dm, float(np.dot(self._cube_center - C, f)) + self._cube_rad + 0.03)
        mat_center = C + f * Dm
        frame_h = Dm * tan_half                                    # 该深度处画面半高
        # 投影约 1.6× 手部张角；上限 0.82×画面半高，保证四周始终露出衬底。
        mH = min(self._hand_rad * 1.6 * (Dm / d_hand), 0.82 * frame_h)
        mW = min(mH * 1.2, 0.82 * frame_h * aspect)
        if self._mat_gid >= 0:
            self.model.body_pos[self._mat_bid] = mat_center
            self.model.body_quat[self._mat_bid] = quat
            self.model.geom_size[self._mat_gid] = np.array([mW, mH, 0.02])

        # 衬底(背景板)：再退到垫布之后 0.15，放大铺满画面（_bb_margin> √(1+asp²) 覆盖滚转四角）。
        D = Dm + 0.15
        center = C + f * D
        halfH = D * tan_half * self._bb_margin
        halfW = halfH * aspect
        self.model.body_pos[self._floor_bid] = center
        self.model.body_quat[self._floor_bid] = quat
        self.model.geom_size[self._floor_gid] = np.array([halfW, halfH, 0.02])
        if self._light_id >= 0:
            self.model.light_dir[self._light_id] = f                # 光沿光轴照向板面
            self.model.light_pos[self._light_id] = C
        mujoco.mj_forward(self.model, self.data)

    def autofit_camera(self, points: np.ndarray, *, azimuth=140.0, elevation=-20.0,
                       lookat=None, margin=1.05):
        """按世界点云自动设定 free 相机：距离由 free 相机 fovy **几何反解**，而非拍脑袋的倍数。

        要把半径 rad 的包围球完整收进画面，相机到球心至少需要 rad/tan(fovy/2)
        （fovy=45° 时约 2.41·rad）——原来写死的 dist_scale=2.1 在数学上就装不下，
        手会被撑出画外。margin 是额外留白系数。
        lookat 可显式给（调用方希望画面重心偏向手时用），默认取点云包围盒中心。
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if not len(pts):
            pts = np.zeros((1, 3))
        c = np.asarray(lookat, dtype=np.float64) if lookat is not None \
            else (pts.min(0) + pts.max(0)) / 2.0
        rad = float(np.linalg.norm(pts - c, axis=1).max())
        fovy = float(self.model.vis.global_.fovy)             # free 相机竖直 FoV（度）
        self._fit_center = c
        self._fit_dist = max(0.3, rad / np.tan(np.radians(fovy) / 2.0) * float(margin))
        self.cam.lookat[:] = c
        self.cam.distance = self._fit_dist
        self.cam.azimuth = azimuth
        self.cam.elevation = elevation

    # ---- 装饰几何：相机锥体 + 轨迹线 ----
    @staticmethod
    def _add_line(scene, p0, p1, width, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.zeros(9),
                            np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64))
        scene.ngeom += 1

    @staticmethod
    def _add_screen_line(scene, p0, p1, width, rgba):
        """Add an unlit pixel-width line; unlike capsules it casts no shadow."""
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3), np.zeros(3), np.zeros(9),
            np.asarray(rgba, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_LINE, float(width),
            np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64),
        )
        scene.ngeom += 1

    def _add_frustum(self, scene, c2w, size, rgba, width):
        t = np.asarray(c2w[:3, 3], dtype=np.float64)
        R = np.asarray(c2w[:3, :3], dtype=np.float64)
        h = size * 0.6
        corners_cam = np.array([[h, h, size], [-h, h, size], [-h, -h, size], [h, -h, size]])
        cs = (R @ corners_cam.T).T + t                        # (4,3) world
        for c in cs:
            self._add_line(scene, t, c, width, rgba)          # apex → 四角
        for i in range(4):
            self._add_line(scene, cs[i], cs[(i + 1) % 4], width, rgba)  # 底面矩形

    def fit_fixed_third_camera(self, up, hand_points, cameras):
        """Fit the shared Wuji-style third-person camera to the complete hand motion."""
        position, target, camera_up, radius = fixed_third_camera_pose(
            up, hand_points, cameras, aspect=self.width / self.height)
        self._fixed_third = {
            "position": position, "target": target, "up": camera_up,
            "radius": radius,
        }
        self._fixed_camera_frusta = prepare_fixed_camera_frusta(cameras, radius)

    def render_fixed_third(self, current_frame: int):
        """Render one faded start camera pose plus the live pose."""
        if not hasattr(self, "_fixed_third"):
            raise RuntimeError("固定第三视角尚未完成取景")
        self.renderer.update_scene(self.data, self.cam)
        view = self._fixed_third
        configure_fixed_camera(
            self.renderer.scene, view["position"], view["target"], view["up"],
            fovy=FIXED_THIRD_FOVY, aspect=self.width / self.height)
        for start, end, width, rgba in fixed_camera_frusta(
                self._fixed_camera_frusta, current_frame):
            self._add_screen_line(self.renderer.scene, start, end, width, rgba)
        return self.renderer.render()

    def _draw_cam_trail(self, scene, cam_c2w, cur, *, max_frusta=8,
                        max_points=240, size=None, width=None, full_trail=False):
        traj = np.asarray(cam_c2w, dtype=np.float64)          # (T,4,4)
        # 锥体尺寸按手部 3D 半径派生（原来写死 0.05，场景尺度一变就要么看不见要么糊满屏）。
        size = float(self._hand_rad * 0.35) if size is None else float(size)
        width = size * 0.06 if width is None else float(width)
        centers = traj[:, :3, 3] if full_trail else traj[:cur + 1, :3, 3]
        stride = max(1, int(np.ceil(len(centers) / max(2, int(max_points)))))
        trail_indices = list(range(0, len(centers), stride))
        if trail_indices and trail_indices[-1] != len(centers) - 1:
            trail_indices.append(len(centers) - 1)
        for start, end in zip(trail_indices, trail_indices[1:]):
            if not np.isfinite(centers[[start, end]]).all():
                continue
            phase = end / max(1, len(centers) - 1)
            alpha = 0.16 + 0.78 * phase ** 1.4
            self._add_line(scene, centers[start], centers[end], width * 0.5,
                           (*CAM_RGBA[:3], alpha))
        # 按世界位置最小间距抽锥体：相邻入选锥体世界距离 < 阈值就跳过 → 相机位移小时不会把十几个锥体
        # 叠成一坨；位移大时自然沿轨迹排成一串。始终保留当前帧(末帧)锥体。
        min_gap = size * 1.5
        idxs = [0] if len(centers) else []
        for i in range(1, len(centers)):
            if np.linalg.norm(centers[i] - centers[idxs[-1]]) >= min_gap:
                idxs.append(i)
        # 全段轨迹模式仍把当前相机留到最后画，使当前帧锥体最醒目。
        if idxs and idxs[-1] != cur:
            idxs.append(cur)
        if len(idxs) > max_frusta:                            # 超上限沿链等距抽稀
            idxs = list(np.array(idxs)[np.unique(np.linspace(0, len(idxs) - 1, max_frusta).astype(int))])
        for k, i in enumerate(idxs):                          # 锥体：越近越实（由淡到实）
            a = 0.2 + 0.7 * (k + 1) / max(1, len(idxs))
            self._add_frustum(scene, traj[i], size, (*CAM_RGBA[:3], a), width)

    def _draw_hand_trails(self, scene, joints_l, joints_r, validity, *, max_points=240):
        """画完整左右腕轨迹；时间越晚越实，与 Fixed World 的渐变方向一致。"""
        valid = np.asarray(validity, dtype=bool)
        width = max(0.0008, float(self._hand_rad) * 0.012)
        for hand_index, (joints, color) in enumerate((
                (joints_l, LEFT_RGBA), (joints_r, RIGHT_RGBA))):
            points = np.asarray(joints, dtype=np.float64)[:, 0]
            stride = max(1, int(np.ceil(len(points) / max(2, int(max_points)))))
            indices = list(range(0, len(points), stride))
            if indices and indices[-1] != len(points) - 1:
                indices.append(len(points) - 1)
            for start, end in zip(indices, indices[1:]):
                if (not valid[start:end + 1, hand_index].all()
                        or not np.isfinite(points[[start, end]]).all()):
                    continue
                phase = end / max(1, len(points) - 1)
                alpha = 0.18 + 0.82 * phase ** 1.4
                self._add_line(scene, points[start], points[end], width,
                               (*color[:3], alpha))

    # ---- 逐帧把双手网格摆好（多视角渲染前只做一次，避免重复上传 GPU）----
    def bake_frame(self, verts_l, verts_r, valid_l, valid_r):
        self._set_visible("left", valid_l)
        self._set_visible("right", valid_r)
        if valid_l:
            self._bake("left", verts_l)
        if valid_r:
            self._bake("right", verts_r)

    # ---- 手部运动残影：把抽出的历史帧摆进 ghost 槽（对标 HaWoR 的一串由淡到实的手）----
    def hide_ghosts(self):
        """隐藏全部 ghost 槽（ego 第一人称渲染前调用：那是实时画面，不该有残影）。"""
        for i in range(self.n_ghost):
            for side in ("left", "right"):
                gid = self._mesh[f"{side}#{i}"]["gid"]
                rgba = np.array(self.model.geom_rgba[gid])
                rgba[3] = 0.0
                self.model.geom_rgba[gid] = rgba

    def set_ghosts(self, ghosts_l, ghosts_r):
        """ghosts_*: 由旧到新的历史帧世界顶点列表（每项 (778,3)，None=该槽不画）。

        颜色由基色**向白线性混合**：最老最淡、最新最接近本色，全程不透明——
        既避开 MuJoCo 半透明排序问题，观感也更贴 HaWoR 那串浅色渐变的手。
        ghost 一并投影，地上会留下一串手影，同样对齐 HaWoR。
        """
        for side, seq, base in (("left", ghosts_l, LEFT_RGBA), ("right", ghosts_r, RIGHT_RGBA)):
            seq = list(seq or [])
            n = min(len(seq), self.n_ghost)
            for i in range(self.n_ghost):
                info = self._mesh[f"{side}#{i}"]
                if i < n and seq[i] is not None:
                    self._bake(f"{side}#{i}", seq[i])
                    w = _GHOST_W0 + (_GHOST_W1 - _GHOST_W0) * (i + 1) / max(1, n)   # 由淡到实
                    rgb = np.asarray(base[:3]) * w + (1.0 - w)                       # 向白混合
                    self.model.geom_rgba[info["gid"]] = np.array([*rgb, 1.0])
                else:
                    rgba = np.array(self.model.geom_rgba[info["gid"]])
                    rgba[3] = 0.0
                    self.model.geom_rgba[info["gid"]] = rgba

    # ---- 渲染单帧（自由相机 + 相机轨迹锥体）；保持原签名供单测/交互复用 ----
    def render_frame(self, verts_l, verts_r, valid_l, valid_r, cam_c2w, cur) -> np.ndarray:
        self.bake_frame(verts_l, verts_r, valid_l, valid_r)
        self.renderer.update_scene(self.data, self.cam)
        if cam_c2w is not None:
            self._draw_cam_trail(self.renderer.scene, cam_c2w, cur)
        return self.renderer.render()

    # ---- 第三人称环视：给定方位角/仰角，lookat/distance 用 autofit 结果 ----
    def render_free(self, azimuth, elevation, cam_c2w, cur, *, draw_trail=True,
                    full_trail=False, hand_trails=None, hand_validity=None) -> np.ndarray:
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = self._fit_center
        self.cam.distance = self._fit_dist
        self.cam.azimuth = float(azimuth)
        self.cam.elevation = float(elevation)
        self.renderer.update_scene(self.data, self.cam)
        if draw_trail and cam_c2w is not None:
            self._draw_cam_trail(
                self.renderer.scene, cam_c2w, cur, full_trail=full_trail)
        if hand_trails is not None and hand_validity is not None:
            self._draw_hand_trails(
                self.renderer.scene, hand_trails[0], hand_trails[1], hand_validity)
        return self.renderer.render()

    # ---- 第一人称 ego：把 OpenCV cam→world 外参装到 MuJoCo 固定相机上（翻 y/z 轴）----
    def set_ego(self, c2w, fovy_deg):
        if self._ego_cid < 0:
            return
        c2w = np.asarray(c2w, dtype=np.float64)
        # MuJoCo 相机看 -z、+y 上（OpenGL）；数据是 OpenCV +z 前、+y 下 → 右乘 diag(1,-1,-1) 换轴。
        R = c2w[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R.reshape(-1)))
        self.model.cam_pos[self._ego_cid] = c2w[:3, 3]
        self.model.cam_quat[self._ego_cid] = quat
        self.model.cam_fovy[self._ego_cid] = float(fovy_deg)
        mujoco.mj_forward(self.model, self.data)

    def render_ego(self, intrinsics=None, image_size=None) -> np.ndarray:
        self.renderer.update_scene(self.data, self._ego_cid)   # 固定相机(=真实头戴相机位姿)，不画轨迹锥体
        if intrinsics is not None:
            if image_size is None:
                raise ValueError("设置 MuJoCo 相机内参时必须提供原视频尺寸")
            K = np.asarray(intrinsics, dtype=np.float64)
            width, height = map(float, image_size)
            if K.shape != (3, 3) or width <= 0 or height <= 0:
                raise ValueError("MuJoCo 相机内参或原视频尺寸无效")
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            if fx <= 0 or fy <= 0:
                raise ValueError("MuJoCo 相机焦距必须为正数")
            # MjvGLCamera exposes an off-axis frustum. Its width field is a
            # half-width, while bottom/top use the usual near-plane bounds.
            for camera in self.renderer.scene.camera:
                near = float(camera.frustum_near)
                left, right = -cx / fx * near, (width - cx) / fx * near
                camera.frustum_center = (left + right) / 2.0
                camera.frustum_width = (right - left) / 2.0
                camera.frustum_bottom = -(height - cy) / fy * near
                camera.frustum_top = cy / fy * near
        return self.renderer.render()

    def close(self):
        try:
            self.renderer.close()
        except Exception:      # noqa: BLE001  EGL 上下文销毁的无害告警
            pass
