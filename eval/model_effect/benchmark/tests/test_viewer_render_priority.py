import sys
import threading
import zipfile
from pathlib import Path


MODEL_EFFECT = Path(__file__).resolve().parents[2]
WEB_DIR = MODEL_EFFECT / "visualization" / "viewer" / "web"
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))


def test_robot_panels_wait_for_2d_and_compare_gt_pred_per_method():
    source = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    style = (WEB_DIR / "style.css").read_text(encoding="utf-8")

    assert "function _wireServerVideoPanel(pan,badge,{autoStart=true}={})" in source
    assert "const primary2DReady=new Promise" in source
    assert "const compare=!state.no_truth&&!state.rawOnly;" in source
    assert "const sources=compare?['gt','pred']" in source
    assert "_wireServerVideoPanel(child,badge,{autoStart:false})" in source
    assert "if(!pan.isActive||pan.isActive()) pan.releaseRender();" in source
    assert "pan.videos||[pan.video]" in source
    assert "video.classList.add('synced-follower-video')" in source
    assert ".robot-comparison-grid.is-comparison" in style
    assert ".primary-scene-row" not in style
    assert ".mujoco-wrap video.synced-follower-video { pointer-events: none; }" in style


def test_export_defaults_to_all_four_server_rendered_views():
    source = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    assert "'both_2d','world_motion_3d','mujoco_3d','wuji_retarget_3d'" in source
    assert "available:Boolean(state.loaded)" in source
    assert "Boolean(panel)" not in source[source.index("function _exportSourceItem"):source.index("function _selectedExportSources")]
    for source_id in ("both_2d", "world_motion_3d", "mujoco_3d", "wuji_retarget_3d"):
        assert f'data-export-source="{source_id}" checked' in html
    assert "导出选中 4 路" in html
    assert 'id="exportSeparateBtn"' in html
    assert "output_format:outputFormat" in source


def test_individual_export_bundle_contains_named_gt_pred_videos(tmp_path):
    from visualization.viewer.store import Store

    store = Store.__new__(Store)
    store.cache_dir = tmp_path / "cache"
    store._locks = {}
    store._reg_lock = threading.Lock()
    inputs = []
    for source in ("both_2d", "world_motion_3d", "mujoco_3d", "wuji_retarget_3d"):
        for kind in ("gt", "pred"):
            path = tmp_path / f"{source}_{kind}.mp4"
            path.write_bytes(f"{source}:{kind}".encode())
            inputs.append((f"{source}_{kind}", path))

    output = store._bundle_export(inputs, episode_index=7)

    assert output.suffix == ".zip"
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [
            "ep007-2d-gt.mp4", "ep007-2d-pred.mp4",
            "ep007-world-gt.mp4", "ep007-world-pred.mp4",
            "ep007-mujoco-gt.mp4", "ep007-mujoco-pred.mp4",
            "ep007-retarget-gt.mp4", "ep007-retarget-pred.mp4",
        ]


def test_individual_raw_export_uses_gt_filename(tmp_path):
    from visualization.viewer.store import Store

    store = Store.__new__(Store)
    store.cache_dir = tmp_path / "cache"
    store._locks = {}
    store._reg_lock = threading.Lock()
    source = tmp_path / "world.mp4"
    source.write_bytes(b"raw world render")

    output = store._bundle_export(
        [("world_motion_3d", source)], episode_index=2, default_source="gt")

    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["ep002-world-gt.mp4"]


def test_export_renders_overall_2d_before_robot_views_and_keeps_layout_order(tmp_path):
    from visualization.viewer.store import Store

    store = Store.__new__(Store)
    primary_done = threading.Event()
    calls = []
    composed = []
    compose_options = {}
    store.is_no_truth = lambda _eid: False

    def render_2d(source, path):
        def render(*args, **kwargs):
            calls.append(f"both_2d:{source}")
            if source == "pred":
                primary_done.set()
            return tmp_path / path

        return render

    def render_robot(name, path):
        def render(_eid, source, *args, **kwargs):
            assert primary_done.is_set()
            calls.append(f"{name}:{source}")
            assert kwargs["render_size"] == (512, 512)
            return tmp_path / f"{source}_{path}"

        return render

    def render_world(_eid, *args, **kwargs):
        assert primary_done.is_set()
        source = kwargs["render_source"]
        calls.append(f"world_motion_3d:{source}")
        assert kwargs["render_size"] == (512, 512)
        return tmp_path / f"{source}_world.mp4"

    store.mp4_gt = render_2d("gt", "2d_gt.mp4")
    store.mp4_pred = render_2d("pred", "2d_pred.mp4")
    store.mujoco_video = render_robot("mujoco_3d", "mujoco.mp4")
    store.retarget_video = render_robot("wuji_retarget_3d", "retarget.mp4")
    store.world_video = render_world
    store.raw = lambda _eid: {"frames": __import__("numpy").zeros((1, 512, 512, 3))}
    def compose(inputs, **kwargs):
        composed.extend(inputs)
        compose_options.update(kwargs)
        return tmp_path / "export.mp4"

    store._compose_export = compose

    result = store.export_video(
        0,
        ["both_2d", "world_motion_3d", "mujoco_3d", "wuji_retarget_3d"],
        mode="mesh_skel",
    )

    assert calls[:2] == ["both_2d:gt", "both_2d:pred"]
    assert set(calls[2:]) == {
        "mujoco_3d:gt", "mujoco_3d:pred",
        "wuji_retarget_3d:gt", "wuji_retarget_3d:pred",
        "world_motion_3d:gt", "world_motion_3d:pred",
    }
    assert [source_id for source_id, _ in composed] == [
        "both_2d_gt", "both_2d_pred",
        "world_motion_3d_gt", "world_motion_3d_pred",
        "mujoco_3d_gt", "mujoco_3d_pred",
        "wuji_retarget_3d_gt", "wuji_retarget_3d_pred",
    ]
    assert compose_options["tile_size"] == (512, 512)
    assert result == tmp_path / "export.mp4"


def test_export_without_2d_still_renders_selected_robot_views(tmp_path):
    from visualization.viewer.store import Store

    store = Store.__new__(Store)
    composed = []
    store.is_no_truth = lambda _eid: True
    store.raw = lambda _eid: {"frames": __import__("numpy").zeros((1, 512, 512, 3))}
    store.mujoco_video = lambda *args, **kwargs: tmp_path / "mujoco.mp4"
    store.retarget_video = lambda *args, **kwargs: tmp_path / "retarget.mp4"
    store._compose_export = lambda inputs, **kwargs: (
        composed.extend(inputs) or tmp_path / "export.mp4")

    store.export_video(
        0,
        ["mujoco_3d", "wuji_retarget_3d"],
        mode="mesh_skel",
    )

    assert [source_id for source_id, _ in composed] == [
        "mujoco_3d", "wuji_retarget_3d",
    ]


def test_fixed_world_export_uses_translucent_labeled_positive_axes():
    from visualization.render import fixed_world_video

    source = Path(fixed_world_video.__file__).read_text(encoding="utf-8")
    assert fixed_world_video.CACHE_TAG == "fixed_world_canvas_v7_clean_export_hud"
    assert fixed_world_video.AXIS_ALPHA < 0.4
    assert 'axis_labels.append((f"+{label}"' in source
    assert "cv2.arrowedLine(" in source
    assert "first camera = origin" not in source
    assert "FIXED WORLD Z-UP" not in source
    assert 'f"frame {frame_index}' not in source
    assert 'f"{prefix}cm  |  {grid}"' in source


def test_only_2d_video_uses_explicit_hand_presence_hud():
    from visualization.render import compare, draw, wuji_retargeting_video

    compare_source = Path(compare.__file__).read_text(encoding="utf-8")
    draw_source = Path(draw.__file__).read_text(encoding="utf-8")
    retarget_source = Path(wuji_retargeting_video.__file__).read_text(encoding="utf-8")
    assert compare.CACHE_TAG == "allpred_2d_v6_explicit_presence"
    assert "left: {mark(left)}  right: {mark(right)}" in draw_source
    assert "presence_label(" in compare_source
    assert "LIVE' if validity" not in retarget_source
    assert "MISS'" not in retarget_source


def test_prediction_labels_use_uppercase_pred():
    source = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert "PRED" in source
    assert "tag:'Pred'" not in source
    assert "Pred cam" not in source
    assert ">Pred<" not in source


def test_robot_renderers_share_ground_configuration():
    from visualization.render import mujoco_scene, wuji_retargeting_video

    source = Path(wuji_retargeting_video.__file__).read_text(encoding="utf-8")
    assert mujoco_scene.ROBOT_GROUND_RGB1 == (0.19, 0.23, 0.23)
    assert mujoco_scene.ROBOT_GROUND_RGB2 == (0.34, 0.38, 0.37)
    assert mujoco_scene.ROBOT_GROUND_HALF_RANGE == (4.0, 8.0)
    assert mujoco_scene.ROBOT_HAND_SPECULAR == 0.2
    assert mujoco_scene.ROBOT_HAND_SHININESS == 0.3
    assert "ROBOT_GROUND_RGB1" in source
    assert "ROBOT_GROUND_RGB2" in source
    assert "ROBOT_HAND_SPECULAR" in source
    assert "robot_ground_half_extent" in source
    assert "_vignette" not in source
    assert "canonicalize_robot_coordinates" in source


def test_eight_source_sized_tiles_export_as_four_columns_by_two_rows():
    from visualization.viewer.store import _export_grid_layout

    layout, canvas_size = _export_grid_layout(8, 512, 512)
    assert layout == (
        "0_0|512_0|1024_0|1536_0|"
        "0_512|512_512|1024_512|1536_512")
    assert canvas_size == (2048, 1024)


def test_fixed_third_camera_pans_view_slightly_left():
    import numpy as np

    from visualization.render.mujoco_scene import (
        FIXED_THIRD_VIEW_SHIFT_LEFT,
        fixed_third_camera_pose,
    )

    points = np.asarray([
        [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1],
        [-0.1, 0.1, 0.0], [0.1, -0.1, 0.0],
    ])
    cameras = np.repeat(np.eye(4)[None], 2, axis=0)
    position, target, up, radius = fixed_third_camera_pose(
        np.asarray([0.0, 0.0, 1.0]), points, cameras, aspect=1.0)
    unshifted_target = sum(np.percentile(points, [1, 99], axis=0)) * 0.5
    view_forward = target - position
    view_forward /= np.linalg.norm(view_forward)
    screen_right = np.cross(view_forward, up)
    screen_right /= np.linalg.norm(screen_right)
    shift = target - unshifted_target

    assert np.dot(shift, screen_right) < 0.0
    assert np.isclose(
        np.linalg.norm(shift), radius * FIXED_THIRD_VIEW_SHIFT_LEFT)


def test_robot_renderers_share_canonical_camera_origin_gauge():
    import numpy as np

    from visualization.render.mujoco_scene import canonicalize_robot_coordinates

    cameras = np.repeat(np.eye(4)[None], 3, axis=0)
    cameras[:, :3, 1] = [0.0, -1.0, 0.0]
    cameras[:, :3, 3] = [1.0, 2.0, 3.0]
    points = np.asarray([
        [0.1, 0.2, 0.3], [-0.2, 0.4, 0.1], [0.3, -0.1, 0.5],
    ])
    rotation, origin, canonical = canonicalize_robot_coordinates(cameras, points)
    transformed = points @ rotation.T - origin

    assert np.allclose(rotation @ np.asarray([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
    assert np.allclose(np.median(transformed, axis=0), [0.0, 0.0, 0.0])
    assert np.allclose(canonical[:, :3, 3],
                       np.asarray([1.0, 2.0, 3.0]) @ rotation.T - origin)
