"""Internal helper."""

import cv2
import numpy as np


def is_camera_moving(video_path: str, threshold: float = 2.0, sample_interval: int = 5) -> bool:
    """Internal helper."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"[backend]  {video_path}.")

    ret, prev = cap.read()
    if not ret:
        raise ValueError("[backend]")
    prev_gray = cv2.cvtColor(cv2.resize(prev, (320, 180)), cv2.COLOR_BGR2GRAY)

    feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=10, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    mean_flows = []
    while True:
        for _ in range(sample_interval - 1):
            cap.grab()
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(cv2.resize(curr, (320, 180)), cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if p0 is None or len(p0) < 4:
            prev_gray = curr_gray
            continue
        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
        good = status.ravel() == 1
        if good.sum() >= 4:
            mean_flows.append(np.mean(np.linalg.norm(p1[good] - p0[good], axis=1)))
        prev_gray = curr_gray

    cap.release()
    return bool(mean_flows) and np.mean(mean_flows) >= threshold
