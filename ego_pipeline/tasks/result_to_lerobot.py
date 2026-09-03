
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as _R

log = logging.getLogger("ego_pipeline.result_to_lerobot")

STATE_DIM = 122
ACTION_DIM = 102
DEFAULT_VIDEO_KEY = "observation.images.ego"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "lerobot_v3"


def _to_np(x, dtype=None):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def _aa_to_matrix(aa: np.ndarray) -> np.ndarray:
    leading = aa.shape[:-1]
    return _R.from_rotvec(aa.reshape(-1, 3)).as_matrix().astype(np.float32).reshape(
        leading + (3, 3)
    )


def _matrix_to_aa(mat: np.ndarray) -> np.ndarray:
    leading = mat.shape[:-2]
    return _R.from_matrix(mat.reshape(-1, 3, 3)).as_rotvec().astype(np.float32).reshape(
        leading + (3,)
    )


def _matrix_to_euler(mat: np.ndarray) -> np.ndarray:
    """Internal helper."""
    leading = mat.shape[:-2]
    return _R.from_matrix(mat.reshape(-1, 3, 3)).as_euler(
        "xyz", degrees=False).astype(np.float32).reshape(leading + (3,))







_MANO_J0_CACHE: dict = {}




KP21_DIM = 63
KP21_QUAT_DIM = 84

_MANO_PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])

_MANO_TO_OP = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])

_KP21_PARENT = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Internal helper."""
    shp = R.shape[:-2]
    q = _R.from_matrix(R.reshape(-1, 3, 3).astype(np.float64)).as_quat()   # xyzw
    return np.concatenate([q[:, 3:4], q[:, :3]], axis=1).reshape(shp + (4,))


def _hawor_data_dir() -> Path:
    """Internal helper."""
    cand = PROJECT_ROOT / "third_party" / "HaWoR" / "_DATA"
    if cand.exists():
        return cand
    return PROJECT_ROOT / "model" / "hawor" / "_DATA"


def _mano_model(is_right: bool):
    """Internal helper."""
    hawor_dir = PROJECT_ROOT / "third_party" / "HaWoR"
    if str(hawor_dir) not in sys.path:
        sys.path.insert(0, str(hawor_dir))
    from lib.models.mano_wrapper import MANO  # noqa: E402

    data_dir = _hawor_data_dir()
    cache_key = "right" if is_right else "left"
    if cache_key not in _MANO_J0_CACHE:
        if is_right:
            cfg = dict(data_dir=str(data_dir / "data") + "/",
                       model_path=str(data_dir / "data" / "mano"),
                       gender="neutral", num_hand_joints=15, create_body_pose=False)
        else:
            cfg = dict(data_dir=str(data_dir / "data_left") + "/",
                       model_path=str(data_dir / "data_left" / "mano_left"),
                       gender="neutral", num_hand_joints=15,
                       create_body_pose=False, is_rhand=False)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MANO(**cfg)
        if not is_right:
            m.shapedirs[:, 0, :] *= -1
        _MANO_J0_CACHE[cache_key] = m
    return _MANO_J0_CACHE[cache_key]


def _compute_J0_canon(betas: np.ndarray, is_right: bool) -> np.ndarray:
    import torch
    m = _mano_model(is_right)
    b = np.nan_to_num(np.asarray(betas, dtype=np.float32), nan=0.0).reshape(-1, 10)
    N = b.shape[0]
    g = torch.eye(3).view(1, 1, 3, 3).expand(N, 1, 3, 3).contiguous()
    h = torch.eye(3).view(1, 1, 3, 3).expand(N, 15, 3, 3).contiguous()
    t = torch.zeros(N, 3)
    out = m(global_orient=g, hand_pose=h,
            betas=torch.from_numpy(b).float(), transl=t, pose2rot=False)
    return out.joints[:, 0, :].detach().cpu().numpy().astype(np.float32)   # (N,3)


def _compute_kp21(orient_mat: np.ndarray, hp_mat: np.ndarray, transl: np.ndarray,
                  betas: np.ndarray, is_right: bool) -> tuple[np.ndarray, np.ndarray]:
    import torch
    T = orient_mat.shape[0]
    m = _mano_model(is_right)
    b = np.nan_to_num(np.asarray(betas, dtype=np.float32), nan=0.0).reshape(T, 10)
    out = m(global_orient=torch.from_numpy(orient_mat.astype(np.float32)).view(T, 1, 3, 3),
            hand_pose=torch.from_numpy(hp_mat.astype(np.float32)).view(T, 15, 3, 3),
            betas=torch.from_numpy(b).float(),
            transl=torch.from_numpy(transl.astype(np.float32)).view(T, 3),
            pose2rot=False)
    pos = out.joints[:, :21, :].detach().cpu().numpy().astype(np.float64)   # (T,21,3)

    Rg = np.empty((T, 16, 3, 3))
    Rg[:, 0] = orient_mat
    for j in range(1, 16):
        Rg[:, j] = Rg[:, _MANO_PARENTS[j]] @ hp_mat[:, j - 1]
    R21 = np.empty((T, 21, 3, 3))
    for k in range(21):
        mm = int(_MANO_TO_OP[k])
        R21[:, k] = Rg[:, mm] if mm < 16 else R21[:, _KP21_PARENT[k]]
    quat = _rotmat_to_quat_wxyz(R21)                                        # (T,21,4)
    return pos.reshape(T, KP21_DIM), quat.reshape(T, KP21_QUAT_DIM)


def _load_manifest(result_dir: Path) -> dict[str, Any]:
    path = result_dir / "lerobot3_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_result_path(result_dir: Path, manifest: dict[str, Any], key: str, fallback: str) -> Path:
    value = manifest.get(key) or str(result_dir / fallback)
    path = Path(value)
    if not path.is_absolute():
        path = result_dir / path
    return path


def _load_geocalib_intrinsics(result_dir: Path) -> np.ndarray:
    """Internal helper."""
    for parent in result_dir.parents:
        gc = parent / "geocalib_result.json"
        if gc.exists():
            K = json.loads(gc.read_text(encoding="utf-8"))["K"]
            return np.asarray(K, dtype=np.float64).reshape(9)
    raise FileNotFoundError(f"geocalib_result.json not found above {result_dir}")


def _load_result_bundle(result_dir: Path, *, use_raw_traj: bool = False) -> dict[str, Any]:
    result_dir = Path(result_dir).resolve()
    manifest = _load_manifest(result_dir)


    if use_raw_traj:
        pth = manifest.get("raw_pth") or manifest.get("cleaned_pth")
    else:
        pth = manifest.get("cleaned_pth") or manifest.get("raw_pth")
    if pth is None:
        raw_glob = sorted(p for p in result_dir.glob("*.pth")
                          if not p.name.endswith("_cleaned.pth"))
        cleaned_glob = sorted(result_dir.glob("*_cleaned.pth"))
        candidates = (raw_glob or cleaned_glob) if use_raw_traj\
            else (cleaned_glob or sorted(result_dir.glob("*.pth")))
        if not candidates:
            raise FileNotFoundError(f"no pth in {result_dir}")
        pth_path = candidates[0]
    else:
        pth_path = Path(pth)
    if not pth_path.is_absolute():
        pth_path = result_dir / pth_path

    camera_path = _resolve_result_path(result_dir, manifest, "camera_npz", "megasam.npz")
    annotation_path = _resolve_result_path(result_dir, manifest, "annotation_json", "annotation.json")
    frames_dir = _resolve_result_path(result_dir, manifest, "frames_dir", "frames")

    pred = joblib.load(pth_path)
    pred = [
        _to_np(pred[0], np.float32),
        _to_np(pred[1], np.float32),
        _to_np(pred[2], np.float32),
        _to_np(pred[3], np.float32),
        _to_np(pred[4], bool),
    ]
    with np.load(camera_path) as data:
        cam_c2w = np.asarray(data["cam_c2w"], dtype=np.float32)
        fov = np.asarray(data["fov"], dtype=np.float32) if "fov" in data.files else np.zeros(2, dtype=np.float32)

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    return {
        "result_dir": result_dir,
        "manifest": manifest,
        "pth": pth_path,
        "camera": camera_path,
        "annotation": annotation,
        "frames_dir": frames_dir,
        "pred": pred,
        "cam_c2w": cam_c2w,
        "fov": fov,
        "intrinsics": _load_geocalib_intrinsics(result_dir),
        "fps": float(manifest.get("fps") or 30.0),
        "scene": str(manifest.get("scene") or result_dir.parent.name),
    }


def _segments(annotation: dict[str, Any], total_frames: int) -> list[dict[str, Any]]:
    raw = annotation.get("segments") if isinstance(annotation, dict) else None
    if not isinstance(raw, list) or not raw:
        raw = [{"task": None, "main_type": None, "start_frame": 0, "end_frame": total_frames}]
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = int(item.get("start_frame", item.get("start_idx", 0)))
        end = int(item.get("end_frame", item.get("end_idx", total_frames)))
        start = max(0, min(start, total_frames))
        end = max(0, min(end, total_frames))
        if end <= start:
            continue
        main = item.get("main_type", item.get("main_hand"))
        if main is None:
            main_type = -1
        else:
            s = str(main).strip().lower()
            main_type = 0 if s == "left" else 1 if s == "right" else -1
        task = item.get("task", item.get("instruction"))
        out.append({
            "start": start,
            "end": end,
            "task": None if task is None else str(task),
            "main_type": main_type,
        })
    return out or [{"start": 0, "end": total_frames, "task": None, "main_type": -1}]


def _atomic_segments(annotation: dict[str, Any], total_frames: int) -> list[dict[str, Any]]:
    raw = annotation.get("atomic_segments") if isinstance(annotation, dict) else None
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = int(item.get("start_frame", item.get("start_idx", 0)))
        end = int(item.get("end_frame", item.get("end_idx", total_frames)))
        start = max(0, min(start, total_frames))
        end = max(0, min(end, total_frames))
        if end <= start:
            continue
        main = item.get("main_type", item.get("main_hand"))
        if main is None:
            main_type = -1
        else:
            s = str(main).strip().lower()
            main_type = 0 if s == "left" else 1 if s == "right" else -1
        task = item.get("task", item.get("instruction"))
        out.append({
            "start": start,
            "end": end,
            "task": None if task is None else str(task),
            "main_type": main_type,
        })
    return out


def _build_episode_rows(
    *,
    episode_index: int,
    task_index: Sequence[int],
    merged_task_index: int,
    atomic_index: Sequence[int],
    main_type: int,
    pred_slice: list[np.ndarray],
    cam_c2w: np.ndarray,
    fov: np.ndarray,
    intrinsics: np.ndarray,
    dataset_from: int,
    fps: float,
    mirror_left_pose: bool = True,
) -> list[dict[str, Any]]:
    trans, rot, hp, betas, valid = pred_slice
    hp_raw = hp

    if mirror_left_pose:




        hp = hp.copy()
        hp[0, :, 1::3] *= -1.0
        hp[0, :, 2::3] *= -1.0

    T = trans.shape[1]
    if cam_c2w.shape[0] != T:
        raise ValueError(f"cam_c2w length {cam_c2w.shape[0]} != episode length {T}")
    if len(task_index) != T or len(atomic_index) != T:
        raise ValueError(
            f"per-frame task_index/atomic_index length ({len(task_index)}/"
            f"{len(atomic_index)}) != episode length {T}")

    extr_w2c = np.linalg.inv(cam_c2w.astype(np.float64)).astype(np.float32)
    R_w2c = extr_w2c[:, :3, :3]
    t_w2c = extr_w2c[:, :3, 3]

    orient = {
        "left": _aa_to_matrix(rot[0]),
        "right": _aa_to_matrix(rot[1]),
    }
    hand_pose = {
        "left": _aa_to_matrix(hp[0].reshape(T, 15, 3)),
        "right": _aa_to_matrix(hp[1].reshape(T, 15, 3)),
    }




    J0 = np.stack([
        _compute_J0_canon(betas[0], is_right=False),
        _compute_J0_canon(betas[1], is_right=True),
    ], axis=0)                                   # (2, T, 3)
    wrist_world = trans + J0                      # (2, T, 3)



    kp21 = {}
    for h_idx, side in [(0, "left"), (1, "right")]:
        kp21[side] = _compute_kp21(
            orient_mat=_aa_to_matrix(rot[h_idx]),
            hp_mat=_aa_to_matrix(hp_raw[h_idx].reshape(T, 15, 3)),
            transl=trans[h_idx],
            betas=betas[h_idx],
            is_right=(side == "right"),
        )

    obs = np.zeros((T, STATE_DIM), dtype=np.float32)
    for h_idx, side, base in [(0, "left", 0), (1, "right", 61)]:
        wrist_cam = np.einsum("tij,tj->ti", R_w2c, wrist_world[h_idx]) + t_w2c
        orient_cam = np.einsum("tij,tjk->tik", R_w2c, orient[side])
        obs[:, base:base + 3] = wrist_cam.astype(np.float32)


        obs[:, base + 3:base + 6] = _matrix_to_euler(orient_cam)
        obs[:, base + 6:base + 51] = _matrix_to_euler(hand_pose[side]).reshape(T, 45)
        obs[:, base + 51:base + 61] = betas[h_idx]

    state_mask = np.stack([valid[0], valid[1]], axis=1).astype(bool)
    rows = []
    for t in range(T):
        rows.append({
            "frame_index": int(t),
            "episode_index": int(episode_index),
            "index": int(dataset_from + t),
            "task_index": int(task_index[t]),
            "merged_task_index": int(merged_task_index),
            "atomic_index": int(atomic_index[t]),
            "main_type": int(main_type),
            "observation.state": obs[t].tolist(),
            "state_mask": state_mask[t].tolist(),
            "fov": fov.astype(np.float32).tolist(),
            "intrinsics": np.asarray(intrinsics, dtype=np.float64).reshape(-1).tolist(),
            "extrinsics_w2c": extr_w2c[t].reshape(-1).tolist(),
            "left_transl_world": wrist_world[0, t].tolist(),
            "left_orient_world": orient["left"][t].reshape(-1).tolist(),
            "left_hand_pose": hand_pose["left"][t].reshape(-1).tolist(),
            "left_kept": bool(valid[0, t]),
            "left_seg_start": -1,
            "left_seg_end": -1,
            "right_transl_world": wrist_world[1, t].tolist(),
            "right_orient_world": orient["right"][t].reshape(-1).tolist(),
            "right_hand_pose": hand_pose["right"][t].reshape(-1).tolist(),
            "right_kept": bool(valid[1, t]),
            "right_seg_start": -1,
            "right_seg_end": -1,
            "left_hand_kp21_world": kp21["left"][0][t].tolist(),
            "left_hand_kp21_quat_world": kp21["left"][1][t].tolist(),
            "right_hand_kp21_world": kp21["right"][0][t].tolist(),
            "right_hand_kp21_quat_world": kp21["right"][1][t].tolist(),
            "timestamp": float(t) / fps if fps > 0 else 0.0,
        })
    return rows


def _schema() -> pa.Schema:
    f64 = pa.float64()
    i64 = pa.int64()
    b = pa.bool_()
    return pa.schema([
        ("frame_index", i64),
        ("episode_index", i64),
        ("index", i64),
        ("task_index", i64),
        ("merged_task_index", i64),
        ("atomic_index", i64),
        ("main_type", i64),
        ("observation.state", pa.list_(f64, STATE_DIM)),
        ("state_mask", pa.list_(b, 2)),
        ("fov", pa.list_(f64, 2)),
        ("intrinsics", pa.list_(f64, 9)),
        ("extrinsics_w2c", pa.list_(f64, 16)),
        ("left_transl_world", pa.list_(f64, 3)),
        ("left_orient_world", pa.list_(f64, 9)),
        ("left_hand_pose", pa.list_(f64, 135)),
        ("left_kept", b),
        ("left_seg_start", i64),
        ("left_seg_end", i64),
        ("right_transl_world", pa.list_(f64, 3)),
        ("right_orient_world", pa.list_(f64, 9)),
        ("right_hand_pose", pa.list_(f64, 135)),
        ("right_kept", b),
        ("right_seg_start", i64),
        ("right_seg_end", i64),
        ("left_hand_kp21_world", pa.list_(f64, KP21_DIM)),
        ("left_hand_kp21_quat_world", pa.list_(f64, KP21_QUAT_DIM)),
        ("right_hand_kp21_world", pa.list_(f64, KP21_DIM)),
        ("right_hand_kp21_quat_world", pa.list_(f64, KP21_QUAT_DIM)),
        ("timestamp", f64),
    ])


def _frame_files(frames_dir: Path, start: int, end: int) -> list[str]:
    files = sorted(frames_dir.glob("*.jpg"))
    if not files:
        files = sorted(frames_dir.glob("*.png"))
    if len(files) < end:
        raise RuntimeError(f"{frames_dir}: need frame {end}, found {len(files)}")
    return [str(p) for p in files[start:end]]


def _encode_video(frame_files: list[str], out_path: Path, fps: float) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(frame_files[0])
    if first is None:
        raise RuntimeError(f"cannot read frame: {frame_files[0]}")
    H, W = first.shape[:2]
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
        "-r", str(fps), "-i", "pipe:0",
        "-frames:v", str(len(frame_files)),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-g", str(max(1, int(round(fps)))),
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for p in frame_files:
            frame = cv2.imread(p)
            if frame is None:
                raise RuntimeError(f"cannot read frame: {p}")
            if frame.shape[:2] != (H, W):
                raise RuntimeError(f"frame size mismatch: {p} has {frame.shape[:2]}, expected {(H, W)}")
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.stdin = None
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    return {
        "video.fps": float(fps),
        "video.height": int(H),
        "video.width": int(W),
        "video.channel": 3,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }


def _collect_result_dirs(paths: list[str]) -> list[Path]:
    out = []
    for arg in paths:
        p = Path(arg).expanduser().resolve()
        if p.name == "result" and (p / "lerobot3_manifest.json").exists():
            out.append(p)
        elif (p / "lerobot3_manifest.json").exists():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(d for d in p.rglob("result") if (d / "lerobot3_manifest.json").exists()))
        else:
            raise FileNotFoundError(p)
    dedup = []
    seen = set()
    for p in out:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    if not dedup:
        raise FileNotFoundError("no result directories with lerobot3_manifest.json")
    return dedup


def convert(result_dirs: list[Path], output_dir: Path, *, overwrite: bool = False,
            video_key: str = DEFAULT_VIDEO_KEY, mirror_left_pose: bool = True,
            use_raw_traj: bool = False) -> Path:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

    tasks: dict[str, int] = {}
    episode_meta: list[dict[str, Any]] = []
    schema = _schema()
    total_frames = 0
    episode_index = 0
    video_info_last: dict[str, Any] | None = None

    for result_dir in result_dirs:
        bundle = _load_result_bundle(result_dir, use_raw_traj=use_raw_traj)
        trans, rot, hp, betas, valid = bundle["pred"]
        T = int(trans.shape[1])
        fps = float(bundle["fps"])
        if bundle["cam_c2w"].shape[0] != T:
            raise ValueError(f"{result_dir}: cam_c2w T={bundle['cam_c2w'].shape[0]} != pred T={T}")


        atomic_all = _atomic_segments(bundle["annotation"], T)

        for seg in _segments(bundle["annotation"], T):
            start, end = int(seg["start"]), int(seg["end"])
            length = end - start

            merged_text = seg["task"] if seg["task"] is not None else "null"
            merged_task_index = tasks.setdefault(merged_text, len(tasks))



            task_index = [merged_task_index] * length
            atomic_index = [-1] * length
            ep_tasks = [merged_text]
            ep_atomics: list[dict[str, Any]] = []
            ordinal = 0
            for a in atomic_all:
                a_s = max(int(a["start"]), start)
                a_e = min(int(a["end"]), end)
                if a_e <= a_s:
                    continue
                a_text = a["task"] if a["task"] is not None else "null"
                a_task_index = tasks.setdefault(a_text, len(tasks))
                for g in range(a_s, a_e):
                    task_index[g - start] = a_task_index
                    atomic_index[g - start] = ordinal
                ep_atomics.append({
                    "atomic_index": int(ordinal),
                    "task_index": int(a_task_index),
                    "start_frame": int(a_s - start),   # episode-local
                    "end_frame": int(a_e - start),
                })
                if a_text not in ep_tasks:
                    ep_tasks.append(a_text)
                ordinal += 1

            pred_slice = [
                trans[:, start:end],
                rot[:, start:end],
                hp[:, start:end],
                betas[:, start:end],
                valid[:, start:end],
            ]
            rows = _build_episode_rows(
                episode_index=episode_index,
                task_index=task_index,
                merged_task_index=merged_task_index,
                atomic_index=atomic_index,
                main_type=int(seg["main_type"]),
                pred_slice=pred_slice,
                cam_c2w=bundle["cam_c2w"][start:end],
                fov=bundle["fov"],
                intrinsics=bundle["intrinsics"],
                dataset_from=total_frames,
                fps=fps,
                mirror_left_pose=mirror_left_pose,
            )

            file_index = episode_index
            pq_path = output_dir / "data" / "chunk-000" / f"file-{file_index:03d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), pq_path)

            mp4_path = output_dir / "videos" / video_key / "chunk-000" / f"file-{file_index:03d}.mp4"
            video_info = _encode_video(_frame_files(bundle["frames_dir"], start, end), mp4_path, fps)
            video_info_last = video_info

            episode_meta.append({

                "episode_index": int(episode_index),
                "length": int(length),

                "tasks": ep_tasks,
                "atomic_segments": ep_atomics,
                "data/chunk_index": 0,
                "data/file_index": int(file_index),
                f"videos/{video_key}/chunk_index": 0,
                f"videos/{video_key}/file_index": int(file_index),
                "dataset_from_index": int(total_frames),
                "dataset_to_index": int(total_frames + length),

                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                f"videos/{video_key}/from_timestamp": 0.0,
                f"videos/{video_key}/to_timestamp": float(length) / fps if fps > 0 else 0.0,
            })
            total_frames += length
            episode_index += 1

    if video_info_last is None:
        raise RuntimeError("no episodes produced")

    tasks_pdf = pd.DataFrame(
        {"task_index": list(tasks.values())},
        index=pd.Index(list(tasks.keys()), name=None),
    )
    tasks_pdf.to_parquet(output_dir / "meta" / "tasks.parquet")
    pq.write_table(
        pa.Table.from_pylist(episode_meta),
        output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )

    fps_info = float(video_info_last["video.fps"])
    info = {
        "codebase_version": "v3.0",
        "robot_type": "ego_hand",
        "total_episodes": len(episode_meta),
        "total_frames": int(total_frames),
        "total_tasks": len(tasks),
        "total_videos": len(episode_meta),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "fps": fps_info,
        "splits": {"train": f"0:{len(episode_meta)}"},
        "features": {

            "action": {"dtype": "float64", "shape": [ACTION_DIM], "names": None},
            "observation.state": {"dtype": "float64", "shape": [STATE_DIM], "names": None},
            "state_mask": {"dtype": "bool", "shape": [2], "names": ["left", "right"]},
            "fov": {"dtype": "float64", "shape": [2], "names": None},
            "intrinsics": {"dtype": "float64", "shape": [9], "names": None},
            "extrinsics_w2c": {"dtype": "float64", "shape": [16], "names": None},
            video_key: {
                "dtype": "video",
                "shape": [video_info_last["video.height"], video_info_last["video.width"], 3],
                "names": ["height", "width", "channel"],
                "info": video_info_last,
            },

            "main_type": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "merged_task_index": {"dtype": "int64", "shape": [1], "names": None},
            "atomic_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "left_hand_kp21_world": {"dtype": "float64", "shape": [KP21_DIM], "names": None},
            "left_hand_kp21_quat_world": {"dtype": "float64", "shape": [KP21_QUAT_DIM], "names": None},
            "right_hand_kp21_world": {"dtype": "float64", "shape": [KP21_DIM], "names": None},
            "right_hand_kp21_quat_world": {"dtype": "float64", "shape": [KP21_QUAT_DIM], "names": None},
        },
    }
    (output_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


    return output_dir


def validate(dataset_dir: Path, *, num_frames: int = 16) -> None:
    dataset_dir = Path(dataset_dir)
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v3.0", info["codebase_version"]
    assert info["features"]["observation.state"]["shape"] == [STATE_DIM]
    assert info["features"]["action"]["shape"] == [ACTION_DIM]
    log.info(
        "info.json OK: fps=%.2f, %d eps, %d frames",
        info["fps"], info["total_episodes"], info["total_frames"],
    )

    ep_paths = sorted((dataset_dir / "meta" / "episodes").rglob("*.parquet"))
    if not ep_paths:
        raise FileNotFoundError(f"no episode parquet under {dataset_dir / 'meta' / 'episodes'}")
    ep_tbl = pa.concat_tables([pq.read_table(p) for p in ep_paths]).to_pandas()
    assert {"episode_index", "length", "data/chunk_index", "data/file_index"} <= set(ep_tbl.columns)
    log.info("episodes parquet OK: %d rows", len(ep_tbl))

    tasks_df = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet")
    task_idx = tasks_df["task_index"].to_numpy()
    task_str = tasks_df.index.to_numpy()
    task_map = dict(zip(task_idx.tolist(), task_str.tolist()))
    for text in task_map.values():
        assert isinstance(text, str) and text, f"invalid task text in tasks.parquet: {text!r}"
    log.info("tasks.parquet OK: %s", task_map)

    needed = {
        "frame_index", "episode_index", "main_type",
        "observation.state", "state_mask", "fov", "intrinsics", "task_index",
        "merged_task_index", "atomic_index",
        "extrinsics_w2c",
        "left_transl_world", "left_orient_world", "left_hand_pose",
        "left_kept", "left_seg_start", "left_seg_end",
        "right_transl_world", "right_orient_world", "right_hand_pose",
        "right_kept", "right_seg_start", "right_seg_end",
        "left_hand_kp21_world", "left_hand_kp21_quat_world",
        "right_hand_kp21_world", "right_hand_kp21_quat_world",
    }
    data_paths = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"no data parquet under {dataset_dir / 'data'}")
    for path in data_paths:
        missing = needed - set(pq.read_schema(path).names)
        assert not missing, f"{path}: missing columns {sorted(missing)}"
    log.info("data parquet columns OK: %d files", len(data_paths))

    ep0 = ep_tbl.sort_values("episode_index").iloc[0]
    length = int(ep0["length"])
    chunk_i = int(ep0["data/chunk_index"])
    file_i = int(ep0["data/file_index"])
    data_path = dataset_dir / info["data_path"].format(chunk_index=chunk_i, file_index=file_i)
    rows = pq.read_table(data_path, columns=list(needed)).to_pandas()
    assert len(rows) == length, f"{data_path}: rows={len(rows)} length={length}"

    state = np.asarray(rows["observation.state"].iloc[0], dtype=np.float32)
    state_mask = np.asarray(rows["state_mask"].iloc[0], dtype=bool)
    fov = np.asarray(rows["fov"].iloc[0], dtype=np.float32)
    assert state.shape == (STATE_DIM,), state.shape
    assert state_mask.shape == (2,), state_mask.shape
    assert fov.shape == (2,), fov.shape
    log.info("sample row OK: state=%s state_mask=%s fov=%s", state.shape, state_mask.tolist(), fov.tolist())

    window_t = min(num_frames, max(0, length - 1))
    if window_t > 0:
        win_ext = rows.iloc[:window_t + 1]
        extr0 = np.asarray(win_ext["extrinsics_w2c"].iloc[0], dtype=np.float64).reshape(4, 4)
        assert extr0.shape == (4, 4)
        for hand in ("left", "right"):
            transl = np.vstack(win_ext[f"{hand}_transl_world"].to_numpy()).astype(np.float32)
            orient = np.vstack(win_ext[f"{hand}_orient_world"].to_numpy()).astype(np.float32)
            pose = np.vstack(win_ext[f"{hand}_hand_pose"].to_numpy()).astype(np.float32)
            assert transl.shape == (window_t + 1, 3)
            assert orient.shape == (window_t + 1, 9)
            assert pose.shape == (window_t + 1, 135)
        log.info("window raw fields OK: T=%d", window_t)

    video_key = DEFAULT_VIDEO_KEY
    v_chunk = int(ep0[f"videos/{video_key}/chunk_index"])
    v_file = int(ep0[f"videos/{video_key}/file_index"])
    mp4_path = dataset_dir / info["video_path"].format(
        video_key=video_key, chunk_index=v_chunk, file_index=v_file,
    )
    cap = cv2.VideoCapture(str(mp4_path))
    assert cap.isOpened(), f"cannot open mp4: {mp4_path}"
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert abs(n_frames - length) <= 2, f"mp4 frames {n_frames} != episode length {length}"
    log.info("video OK: %s %dx%d %d frames", mp4_path.name, width, height, n_frames)
    log.info("validate OK: %s", dataset_dir)


def check_dataloader(
    *,
    result_dirs_args: list[str],
    dataset_dir: Path,
    video_key: str = DEFAULT_VIDEO_KEY,
    atol: float = 2e-5,
    rtol: float = 1e-5,
    check_video_pixels: bool = False,
    pixel_mae_tol: float = 8.0,
    num_frames: int = 16,
    video_num_frames: int = 1,
    video_stride: int = 1,
    window_stride: int = 1,
) -> None:
    try:
        from unit_test.data_to_lerobot.validate_lerobot_v3_semantics import (
            validate_semantics,
        )
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "[pipeline]"
            "[pipeline]"
            f"{e}"
        ) from e

    validate_semantics(
        result_args=list(result_dirs_args),
        dataset_dir=Path(dataset_dir).resolve(),
        video_key=video_key,
        atol=atol,
        rtol=rtol,
        check_video_pixels=check_video_pixels,
        pixel_mae_tol=pixel_mae_tol,
        check_dataloader=True,
        num_frames=num_frames,
        video_num_frames=video_num_frames,
        video_stride=video_stride,
        window_stride=window_stride,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", nargs="+", required=True,
                        help="result/ dirs, scene dirs, or an output root containing result/ dirs")
    parser.add_argument("--output", default=None,
                        help="[pipeline]")
    parser.add_argument("--dataset", default=None,
                        help="[pipeline]")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_mirror_left", action="store_true",
                        help="[pipeline]"
                             "[pipeline]")
    parser.add_argument("--video_key", default=DEFAULT_VIDEO_KEY)
    parser.add_argument("--use_raw_traj", action="store_true",
                        help="[pipeline]")
    parser.add_argument("--skip_validate", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")


    parser.add_argument("--check_dataloader", action="store_true",
                        help="[pipeline]"
                             "[pipeline]")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--check_video_pixels", action="store_true",
                        help="[pipeline]")
    parser.add_argument("--pixel_mae_tol", type=float, default=8.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--video_num_frames", type=int, default=1)
    parser.add_argument("--video_stride", type=int, default=1)
    parser.add_argument("--window_stride", type=int, default=1)

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.check_dataloader:
        dataset_arg = args.dataset or args.output
        if dataset_arg is None:
            parser.error("[pipeline]")
        check_dataloader(
            result_dirs_args=args.result,
            dataset_dir=Path(dataset_arg),
            video_key=args.video_key,
            atol=args.atol,
            rtol=args.rtol,
            check_video_pixels=args.check_video_pixels,
            pixel_mae_tol=args.pixel_mae_tol,
            num_frames=args.num_frames,
            video_num_frames=args.video_num_frames,
            video_stride=args.video_stride,
            window_stride=args.window_stride,
        )
        print(f"[OK] check_dataloader: {Path(dataset_arg).resolve()}")
        return 0

    result_dirs = _collect_result_dirs(args.result)
    out = Path(args.output) if args.output else DEFAULT_OUTPUT_ROOT / _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = convert(result_dirs, out, overwrite=args.overwrite, video_key=args.video_key,
                  mirror_left_pose=not args.no_mirror_left, use_raw_traj=args.use_raw_traj)
    if not args.skip_validate:
        validate(out)
    print(f"[OK] output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
