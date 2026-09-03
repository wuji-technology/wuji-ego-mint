#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os

import cv2
import numpy as np

from . import geometry as geom


COLOR_LEFT = (230, 180, 120)
COLOR_RIGHT = (120, 180, 230)
COLOR_SKEL_LEFT = (255, 230, 80)
COLOR_SKEL_RIGHT = (80, 200, 255)



PALETTE_GT = ((90, 210, 90), (120, 255, 120))
PALETTE_PRED = ((80, 80, 225), (100, 100, 255))

_faces_cache = None

_VIDEO_CRF = os.environ.get("VIEWER_VIDEO_CRF", "23")
_VIDEO_MAXRATE = os.environ.get("VIEWER_VIDEO_MAXRATE", "12M")
_VIDEO_BUFSIZE = os.environ.get("VIEWER_VIDEO_BUFSIZE", "24M")
_VIDEO_PRESET = os.environ.get("VIEWER_VIDEO_PRESET", "veryfast")


def _h264_web_args(fps: float | None = None, *, preset: str | None = None) -> list[str]:
    """H.264 settings with bounded bitrate and short GOPs for remote playback."""
    args = [
        "-c:v", "libx264", "-preset", preset or _VIDEO_PRESET, "-pix_fmt", "yuv420p",
        "-crf", _VIDEO_CRF, "-maxrate", _VIDEO_MAXRATE,
        "-bufsize", _VIDEO_BUFSIZE,
    ]
    if fps is not None:
        gop = max(12, int(round(float(fps) * 2)))
        args += ["-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0"]
    return args + ["-movflags", "+faststart"]


def get_faces():
    """Internal helper."""
    global _faces_cache
    if _faces_cache is None:
        from . import mano
        _faces_cache = mano.build_mano_faces()   # (faces_right, faces_left)
    return _faces_cache


def _draw_mesh(overlay: np.ndarray, verts_2d: np.ndarray, depth: np.ndarray,
               faces: np.ndarray, color: tuple) -> None:
    vfin = np.isfinite(verts_2d).all(axis=1)
    ok = (depth[faces].min(axis=1) > 0.01) & vfin[faces].all(axis=1)
    f = faces[ok]
    if len(f) == 0:
        return
    p = verts_2d[f]                                # (F,3,2)
    area = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
            - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    for tri in p[area > 0].astype(np.int32):
        cv2.fillConvexPoly(overlay, tri, color)


def _draw_skeleton(frame: np.ndarray, joints_world: np.ndarray,
                   cam_c2w: np.ndarray, K: np.ndarray, color: tuple) -> None:
    """Internal helper."""
    N = joints_world.shape[0]
    uv, depth = geom.project(joints_world, cam_c2w, K)
    ok = (depth > 0.01) & np.isfinite(uv).all(axis=1)
    for j1, j2 in geom.MANO_CONNECTIONS:
        if j1 >= N or j2 >= N or not ok[j1] or not ok[j2]:
            continue
        p1 = (int(uv[j1, 0]), int(uv[j1, 1]))
        p2 = (int(uv[j2, 0]), int(uv[j2, 1]))
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    for ji in range(N):
        if not ok[ji]:
            continue
        r = 2 if ji == 0 else 1
        cv2.circle(frame, (int(uv[ji, 0]), int(uv[ji, 1])), r, color, -1, cv2.LINE_AA)


def render_frame(frame_bgr: np.ndarray, cam_c2w: np.ndarray, K: np.ndarray,
                 sides: dict, faces_lr: tuple,
                 mode: str = "mesh_skel", alpha: float = 0.6,
                 palette: tuple | None = None) -> np.ndarray:
    faces_left, faces_right = faces_lr
    if palette is None:
        spec = [("left", COLOR_LEFT, COLOR_SKEL_LEFT, faces_left),
                ("right", COLOR_RIGHT, COLOR_SKEL_RIGHT, faces_right)]
    else:
        mc, sc = palette
        spec = [("left", mc, sc, faces_left), ("right", mc, sc, faces_right)]

    if mode in ("mesh", "mesh_skel"):
        overlay = np.zeros_like(frame_bgr)
        has_mesh = False
        for side, col, _skel_col, faces in spec:
            h = sides.get(side)
            if h and h.get("valid") and h.get("verts") is not None:
                has_mesh = True
                uv, dep = geom.project(h["verts"], cam_c2w, K)
                _draw_mesh(overlay, uv, dep, faces, col)
        out = cv2.addWeighted(frame_bgr, 1.0, overlay, alpha, 0)

        if mode == "mesh_skel" or not has_mesh:
            for side, _col, skel_col, _faces in spec:
                h = sides.get(side)
                if h and h.get("valid") and h.get("joints") is not None:
                    _draw_skeleton(out, h["joints"], cam_c2w, K, skel_col)
        return out

    # skeleton
    out = frame_bgr.copy()
    for side, _col, skel_col, _faces in spec:
        h = sides.get(side)
        if h and h.get("valid") and h.get("joints") is not None:
            _draw_skeleton(out, h["joints"], cam_c2w, K, skel_col)
    return out


def label(frame_bgr: np.ndarray, text: str) -> None:
    """Internal helper."""
    cv2.putText(frame_bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)


def presence_label(frame_bgr: np.ndarray, rows) -> None:
    """Draw explicit ``left: yes/no`` and ``right: yes/no`` 2D status rows."""
    rows = list(rows)
    if not rows:
        return

    def mark(value):
        # Legacy checkpoints without presence scores render both hands.
        return "yes" if value is None or bool(value) else "no"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.58, 1
    texts = [
        f"{name + '  ' if name else ''}left: {mark(left)}  right: {mark(right)}"
        for name, left, right in rows
    ]
    sizes = [cv2.getTextSize(text, font, scale, thickness)[0] for text in texts]
    line_h = max(height for _width, height in sizes) + 11
    panel_w = max(width for width, _height in sizes) + 16
    panel_h = line_h * len(rows) + 6
    x0 = max(0, frame_bgr.shape[1] - panel_w - 8)

    shade = frame_bgr.copy()
    cv2.rectangle(shade, (x0, 5), (frame_bgr.shape[1] - 5, 5 + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.58, frame_bgr, 0.42, 0, dst=frame_bgr)

    colors = {"GT": (120, 255, 120), "PRED": (100, 140, 255)}
    for row_idx, ((name, _left, _right), text, (width, height)) in enumerate(
            zip(rows, texts, sizes)):
        x = frame_bgr.shape[1] - width - 12
        y = 9 + row_idx * line_h + height
        cv2.putText(frame_bgr, text, (x, y), font, scale, (0, 0, 0),
                    thickness + 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (x, y), font, scale,
                    colors.get(name, (255, 255, 255)), thickness, cv2.LINE_AA)


def transcode_to_h264(path) -> None:
    """Internal helper."""
    import subprocess as sp
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return
    tmp = path.with_suffix(".h264.mp4")
    try:
        sp.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
                *_h264_web_args(), str(tmp)], check=True)
        tmp.replace(path)
    except Exception as e:   # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        print(f"[inference]  {path}; {e}.")


class H264PipeWriter:
    
    def __init__(self, path, fps, size, *, preset=None, buffered_frames=0):
        import queue
        import subprocess as sp
        import threading
        from pathlib import Path
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = int(size[0]), int(size[1])
        self._proc = None
        self._vw = None
        self._queue = None
        self._worker = None
        self._pipe_failed = False
        try:
            self._proc = sp.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.w}x{self.h}",
                 "-r", f"{float(fps)}", "-i", "-",
                 *_h264_web_args(float(fps), preset=preset), str(self.path)],
                stdin=sp.PIPE)
        except Exception:
            self._proc = None
            self._vw = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                       float(fps), (self.w, self.h))
        if self._proc is not None and int(buffered_frames) > 0:
            self._queue = queue.Queue(maxsize=int(buffered_frames))
            self._worker = threading.Thread(
                target=self._write_worker, name="h264-pipe-writer", daemon=True)
            self._worker.start()

    def _write_pipe(self, frame_bgr) -> None:
        if self._pipe_failed:
            return
        try:
            frame = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
            self._proc.stdin.write(memoryview(frame).cast("B"))
        except (BrokenPipeError, ValueError, OSError):

            self._pipe_failed = True

    def _write_worker(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            self._write_pipe(frame)

    def write(self, frame_bgr) -> None:
        if self._vw is not None:
            self._vw.write(frame_bgr)
            return
        frame = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        if self._queue is not None:

            self._queue.put(frame.copy())
        else:
            self._write_pipe(frame)

    def close(self) -> None:
        if self._vw is not None:
            self._vw.release()
            self._vw = None
            transcode_to_h264(self.path)
            return
        if self._proc is None:
            return
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join()
            self._worker = None
            self._queue = None
        try:
            self._proc.stdin.close()
        except Exception:   # noqa: BLE001
            pass
        ret = self._proc.wait()
        self._proc = None
        if ret != 0:
            print(f"[inference]  {ret}; {self.path}.")

    release = close
