"""Prediction-only hand overlay and compact 3D trajectory export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import draw, geometry, mano


PER_HAND = 109
HAND_SLICES = {
    "transl_cam": (0, 3),
    "orient6d": (3, 9),
    "pose6d": (9, 99),
    "betas": (99, 109),
}


def prediction_to_hands(hand_output: np.ndarray) -> dict:
    """Split the model's `[T, 218]` output into left/right MANO parameters."""
    values = np.asarray(hand_output, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != PER_HAND * 2:
        raise ValueError(f"Expected hand output [T, 218], received {values.shape}")
    result = {}
    for offset, side in ((0, "left"), (PER_HAND, "right")):
        segment = values[:, offset : offset + PER_HAND]
        result[side] = {
            name: segment[:, start:end].astype(np.float32)
            for name, (start, end) in HAND_SLICES.items()
        }
    return result


def hands_to_world(hands: dict, camera_c2w: np.ndarray, average_shape: bool = True) -> dict:
    """Decode camera-frame hand parameters into world-frame vertices and joints."""
    if average_shape:
        hands = {
            side: {
                **values,
                "betas": np.repeat(values["betas"].mean(0, keepdims=True), len(values["betas"]), axis=0),
            }
            for side, values in hands.items()
        }
    world_parameters = geometry.hand6d_cam_to_world(hands, camera_c2w)
    output = {}
    for side in ("left", "right"):
        values = world_parameters[side]
        is_right = side == "right"
        decoded = mano.decode_hand_6d(
            values["transl_cam"],
            values["orient6d"],
            values["pose6d"],
            values["betas"],
            is_right,
        )
        vertices, joints = mano.run_mano(
            decoded["trans"], decoded["rot"], decoded["hand_pose"], decoded["betas"], is_right
        )
        output[side] = {"verts": vertices, "joints": joints[:, :21]}
    return output


def predicted_presence(prediction: dict, length: int) -> np.ndarray | None:
    """Return thresholded left/right presence predictions when available."""
    if prediction.get("hand_presence_logits") is not None:
        values = np.asarray(prediction["hand_presence_logits"], dtype=np.float32)
        threshold = 0.0
    elif prediction.get("hand_confidence") is not None:
        values = np.asarray(prediction["hand_confidence"], dtype=np.float32)
        threshold = 0.5
    else:
        return None
    return values >= threshold if values.shape == (length, 2) else None


def presence_mask(prediction: dict, length: int) -> np.ndarray:
    """Return the draw mask, keeping both hands visible for legacy checkpoints."""
    presence = predicted_presence(prediction, length)
    return presence if presence is not None else np.ones((length, 2), dtype=bool)


def prepare_scene(frames_rgb: np.ndarray, prediction: dict) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """Decode all numeric values shared by overlay and trajectory rendering."""
    if prediction.get("hand") is None:
        raise ValueError("The selected checkpoint does not expose a hand prediction head.")
    mano.ensure_mano_weights()
    frame_count, height, width = frames_rgb.shape[:3]
    camera_c2w, camera_k = geometry.decode_camera_pose_enc(
        prediction["pose_enc"], height, width, fov_mean=True
    )
    world = hands_to_world(prediction_to_hands(prediction["hand"]), camera_c2w)
    return camera_c2w, camera_k, world, presence_mask(prediction, frame_count)


def render_prediction(
    frames_rgb: np.ndarray,
    prediction: dict,
    output_path: str | Path,
    fps: float,
    mode: str = "mesh_skel",
    alpha: float = 0.62,
    progress=None,
    label_text: str = "MINT / PREDICTION",
) -> Path:
    """Render a single prediction overlay without loading or displaying ground truth."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    camera_c2w, camera_k, world, visible = prepare_scene(frames_rgb, prediction)
    presence = predicted_presence(prediction, len(frames_rgb))
    faces_right, faces_left = draw.get_faces()
    faces = (faces_left, faces_right)
    frame_count, height, width = frames_rgb.shape[:3]
    writer = draw.H264PipeWriter(output, fps, (width, height))
    try:
        for index in range(frame_count):
            frame_bgr = np.ascontiguousarray(frames_rgb[index, :, :, ::-1])
            sides = {
                side: {
                    "verts": world[side]["verts"][index],
                    "joints": world[side]["joints"][index],
                    "valid": bool(visible[index, side_index]),
                }
                for side_index, side in enumerate(("left", "right"))
            }
            rendered = draw.render_frame(
                frame_bgr, camera_c2w[index], camera_k, sides, faces, mode=mode, alpha=alpha
            )
            draw.label(rendered, label_text)
            if presence is None:
                left_present, right_present = None, None
            else:
                left_present, right_present = map(bool, presence[index])
            draw.presence_label(
                rendered, [("", left_present, right_present)]
            )
            writer.write(rendered)
            if progress is not None:
                progress(index + 1, frame_count)
    finally:
        writer.close()
    return output


def trajectory_payload(frames_rgb: np.ndarray, prediction: dict, max_points: int = 180) -> dict:
    """Create a bounded JSON payload for the viewer's dependency-free 3D canvas."""
    camera_c2w, _, world, visible = prepare_scene(frames_rgb, prediction)
    frame_count = len(camera_c2w)
    stride = max(1, int(np.ceil(frame_count / max_points)))
    indices = np.arange(0, frame_count, stride)

    def rounded(values: np.ndarray) -> list:
        return np.round(np.asarray(values, dtype=np.float64), 5).tolist()

    return {
        "frames": indices.tolist(),
        "camera": rounded(camera_c2w[indices, :3, 3]),
        "left_wrist": rounded(world["left"]["joints"][indices, 0]),
        "right_wrist": rounded(world["right"]["joints"][indices, 0]),
        "left_visible": visible[indices, 0].tolist(),
        "right_visible": visible[indices, 1].tolist(),
    }
