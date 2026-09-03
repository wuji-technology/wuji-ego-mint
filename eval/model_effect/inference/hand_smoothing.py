#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline hand post-processing selected by MINT's production pipeline.

This is the camera-frame equivalent of ``smooth_ukf_cam`` from
``ego_pipeline/data_cleaning/cleaning_modules/ukf_cam_smoothing.py``. Model
hands already use camera-frame MANO parameters, so the world->camera->world
wrapper from the data-cleaning pipeline cancels out here. The filter itself is
kept the same: speed-adaptive observation noise, a constant-velocity UKF, and
an unscented RTS backward pass with q=0.6, r=0.6, beta=2.0 by default.
"""
from __future__ import annotations

import numpy as np


_PER_HAND = 109
_MIN_VALID = 4
_ALPHA = 1.0
_BETA_UT = 2.0
_JITTER = 1e-9
DEFAULT_UKF_PARAMS = {"q": 0.6, "r": 0.6, "beta": 2.0}
UKF_PARAM_LIMITS = {"q": (0.1, 2.0), "r": (0.1, 2.0), "beta": (0.0, 5.0)}


def encode_hand_smoothing_mode(params: dict | None = None) -> str:
    """Encode validated UKF parameters in a cache-safe hand-mode string."""
    values = dict(DEFAULT_UKF_PARAMS)
    if params:
        values.update(params)
    normalized = {}
    for name, (minimum, maximum) in UKF_PARAM_LIMITS.items():
        value = float(values[name])
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"UKF {name} must be between {minimum:g} and {maximum:g}, got {value}"
            )
        normalized[name] = value
    return "smooth@{q:.6g},{r:.6g},{beta:.6g}".format(**normalized)


def parse_hand_smoothing_mode(mode: str) -> tuple[str, dict | None]:
    """Return the base hand mode and optional UKF parameters."""
    value = str(mode or "")
    if value in {"hard", "blend"}:
        return value, None
    if value == "smooth":
        return value, dict(DEFAULT_UKF_PARAMS)
    if not value.startswith("smooth@"):
        raise ValueError(f"unknown hand mode: {value!r}")
    parts = value.partition("@")[2].split(",")
    if len(parts) != 3:
        raise ValueError(f"invalid UKF hand mode: {value!r}")
    params = dict(zip(("q", "r", "beta"), map(float, parts)))
    canonical = encode_hand_smoothing_mode(params)
    normalized = dict(zip(("q", "r", "beta"), map(
        float, canonical.partition("@")[2].split(","))))
    return "smooth", normalized


def _rot6d_to_mat(values: np.ndarray) -> np.ndarray:
    """Convert row-major 6D rotations to matrices with a stable fallback."""
    values = np.asarray(values, dtype=np.float64)
    a0, a1 = values[..., :3], values[..., 3:]
    n0 = np.linalg.norm(a0, axis=-1, keepdims=True)
    b0 = a0 / np.maximum(n0, 1e-12)
    a1p = a1 - np.sum(b0 * a1, axis=-1, keepdims=True) * b0
    n1 = np.linalg.norm(a1p, axis=-1, keepdims=True)
    b1 = a1p / np.maximum(n1, 1e-12)
    b2 = np.cross(b0, b1)
    matrices = np.stack([b0, b1, b2], axis=-2)
    bad = ((n0[..., 0] < 1e-8) | (n1[..., 0] < 1e-8)
           | ~np.isfinite(matrices).all(axis=(-2, -1)))
    if np.any(bad):
        matrices[bad] = np.eye(3, dtype=np.float64)
    return matrices


def _mat_to_rot6d(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices)
    return matrices[..., :2, :].reshape(*matrices.shape[:-2], 6)


def _quat_sign_continuous(quaternions: np.ndarray) -> np.ndarray:
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index], quaternions[index - 1]) < 0:
            quaternions[index] = -quaternions[index]
    return quaternions


def _sigma_meas_batch(values: np.ndarray) -> np.ndarray:
    second = values[:, :-2] - 2.0 * values[:, 1:-1] + values[:, 2:]
    median = np.median(second, axis=1, keepdims=True)
    mad = np.median(np.abs(second - median), axis=1)
    return np.maximum(1.4826 * mad / np.sqrt(6.0), 1e-6)


def _sigma_points_batch(mean: np.ndarray, covariance: np.ndarray,
                        lam: float) -> np.ndarray:
    channels, dims = mean.shape
    matrix = (dims + lam) * covariance
    matrix = (0.5 * (matrix + matrix.transpose(0, 2, 1))
              + _JITTER * np.eye(dims))
    chol = np.linalg.cholesky(matrix)
    points = np.empty((channels, 2 * dims + 1, dims))
    points[:, 0] = mean
    for index in range(dims):
        column = chol[:, :, index]
        points[:, 1 + index] = mean + column
        points[:, 1 + dims + index] = mean - column
    return points


def _urtss_core(timestamps: np.ndarray, values: np.ndarray,
                observation_noise: np.ndarray, process_noise: np.ndarray,
                backward: bool = True) -> np.ndarray:
    """Batched constant-velocity UKF with optional unscented RTS smoothing."""
    channels, frames = values.shape
    if frames < _MIN_VALID:
        return values.copy()

    dims = 2
    lam = _ALPHA ** 2 * (dims + (3 - dims)) - dims
    weights_mean = np.full(2 * dims + 1, 1.0 / (2.0 * (dims + lam)))
    weights_cov = weights_mean.copy()
    weights_mean[0] = lam / (dims + lam)
    weights_cov[0] = (lam / (dims + lam)
                      + 1.0 - _ALPHA ** 2 + _BETA_UT)
    identity = np.eye(dims)

    filtered_mean = np.zeros((frames, channels, dims))
    predicted_mean = np.zeros((frames, channels, dims))
    predicted_cov = np.zeros((frames, channels, dims, dims))
    cross_cov = np.zeros((frames, channels, dims, dims))

    current_mean = np.zeros((channels, dims))
    current_mean[:, 0] = values[:, 0]
    initial_velocity_var = np.maximum(
        np.var(np.diff(values, axis=1), axis=1), 1e-6)
    current_cov = np.zeros((channels, dims, dims))
    current_cov[:, 0, 0] = np.maximum(observation_noise[:, 0], 1e-12)
    current_cov[:, 1, 1] = initial_velocity_var
    filtered_mean[0] = current_mean

    for frame in range(1, frames):
        dt = float(timestamps[frame] - timestamps[frame - 1]) or 1.0
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        base_process = np.array(
            [[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
        process_cov = process_noise[:, None, None] * base_process

        sigma = _sigma_points_batch(current_mean, current_cov, lam)
        propagated = np.einsum("ij,ckj->cki", transition, sigma)
        mean_pred = np.einsum("k,ckj->cj", weights_mean, propagated)
        delta_pred = propagated - mean_pred[:, None, :]
        cov_pred = (np.einsum("k,cki,ckj->cij", weights_cov,
                              delta_pred, delta_pred) + process_cov)
        cov_pred = (0.5 * (cov_pred + cov_pred.transpose(0, 2, 1))
                    + _JITTER * identity)
        delta_sigma = sigma - current_mean[:, None, :]
        cross_cov[frame] = np.einsum(
            "k,cki,ckj->cij", weights_cov, delta_sigma, delta_pred)
        predicted_mean[frame] = mean_pred
        predicted_cov[frame] = cov_pred

        update_sigma = _sigma_points_batch(mean_pred, cov_pred, lam)
        observations = update_sigma[:, :, 0]
        observation_mean = np.einsum("k,ck->c", weights_mean, observations)
        delta_observation = observations - observation_mean[:, None]
        innovation = (np.einsum("k,ck->c", weights_cov,
                                delta_observation * delta_observation)
                      + observation_noise[:, frame])
        state_observation = np.einsum(
            "k,cki,ck->ci", weights_cov,
            update_sigma - mean_pred[:, None, :], delta_observation)
        gain = state_observation / innovation[:, None]
        current_mean = (mean_pred
                        + gain * (values[:, frame] - observation_mean)[:, None])
        current_cov = (cov_pred
                       - innovation[:, None, None]
                       * np.einsum("ci,cj->cij", gain, gain))
        current_cov = (0.5 * (current_cov + current_cov.transpose(0, 2, 1))
                       + _JITTER * identity)
        filtered_mean[frame] = current_mean

    if not backward:
        return filtered_mean[:, :, 0].T

    smoothed = filtered_mean.copy()
    for frame in range(frames - 2, -1, -1):
        gain = np.linalg.solve(
            predicted_cov[frame + 1].transpose(0, 2, 1),
            cross_cov[frame + 1].transpose(0, 2, 1),
        ).transpose(0, 2, 1)
        smoothed[frame] = (filtered_mean[frame]
                           + np.einsum("cij,cj->ci", gain,
                                       smoothed[frame + 1]
                                       - predicted_mean[frame + 1]))
    return smoothed[:, :, 0].T


def _speed_weight(timestamps: np.ndarray, values: np.ndarray,
                  beta: float) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d

    translation = values[:3]
    dt = np.maximum(np.diff(timestamps), 1e-6)
    speed = np.linalg.norm(np.diff(translation, axis=1), axis=0) / dt
    speed = np.concatenate([speed[:1], speed])
    speed = uniform_filter1d(speed, size=5, mode="nearest")
    median = np.median(speed) + 1e-9
    return (1.0 + beta * np.maximum(speed / median - 1.0, 0.0)) ** 2


def _smooth_channels(timestamps: np.ndarray, values: np.ndarray, *,
                     q: float, r: float, beta: float,
                     backward: bool) -> np.ndarray:
    sigma = _sigma_meas_batch(values)
    weight = _speed_weight(timestamps, values, beta)
    observation_noise = (r * sigma)[:, None] ** 2 * weight[None, :]
    return _urtss_core(
        timestamps, values, observation_noise, (q * sigma) ** 2, backward)


def smooth_hand_output(hand: np.ndarray, valid: np.ndarray | None = None, *,
                       q: float = DEFAULT_UKF_PARAMS["q"],
                       r: float = DEFAULT_UKF_PARAMS["r"],
                       beta: float = DEFAULT_UKF_PARAMS["beta"],
                       rts: float = 1.0) -> np.ndarray:
    """Smooth ``hand[T,218]`` camera-frame MANO output without changing validity."""
    from scipy.spatial.transform import Rotation

    source = np.asarray(hand)
    if source.ndim != 2 or source.shape[1] != 2 * _PER_HAND:
        raise ValueError(f"hand 应为 [T,218]，实际为 {source.shape}")
    frames = source.shape[0]
    if valid is None:
        valid_mask = np.ones((frames, 2), dtype=bool)
    else:
        valid_mask = np.asarray(valid, dtype=bool)
        if valid_mask.shape != (frames, 2):
            raise ValueError(f"valid 应为 [T,2]，实际为 {valid_mask.shape}")

    result = source.astype(np.float64, copy=True)
    for side, base in enumerate((0, _PER_HAND)):
        indices = np.flatnonzero(valid_mask[:, side])
        if len(indices) < _MIN_VALID:
            continue
        segment = result[indices, base:base + _PER_HAND]
        orientation = Rotation.from_matrix(
            _rot6d_to_mat(segment[:, 3:9])).as_quat()
        orientation = _quat_sign_continuous(orientation)
        pose_matrices = _rot6d_to_mat(segment[:, 9:99].reshape(-1, 15, 6))
        pose_rotvec = Rotation.from_matrix(
            pose_matrices.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 45)
        channels = np.vstack([
            segment[:, :3].T,
            orientation.T,
            pose_rotvec.T,
            segment[:, 99:109].T,
        ])
        smoothed = _smooth_channels(
            indices.astype(float), channels,
            q=float(q), r=float(r), beta=float(beta), backward=float(rts) >= 0.5)

        offset = 0
        segment[:, :3] = smoothed[offset:offset + 3].T
        offset += 3
        quat = smoothed[offset:offset + 4].T
        offset += 4
        quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
        segment[:, 3:9] = _mat_to_rot6d(
            Rotation.from_quat(quat).as_matrix())
        pose = smoothed[offset:offset + 45].T
        offset += 45
        segment[:, 9:99] = _mat_to_rot6d(
            Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix()
        ).reshape(-1, 90)
        segment[:, 99:109] = smoothed[offset:offset + 10].T
        result[indices, base:base + _PER_HAND] = segment

    return result.astype(source.dtype, copy=False)
