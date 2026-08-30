#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路径 + 枚举常量（各 viewer 子模块共用）。"""
from __future__ import annotations

import os
from pathlib import Path

VIS_DIR = Path(__file__).resolve().parents[1]               # viewer -> visualization
REPO_DIR = VIS_DIR.parents[2]                               # visualization -> model_effect -> eval -> <repo>
DEFAULT_CONFIG = (REPO_DIR / "configs" / "training"
                  / "mint_step2.yaml")
MODEL_TRAIN_ROOT = REPO_DIR / "output" / "model_train"      # 各训练 run（<run_ts>/step_*）根目录
DEFAULT_CHECKPOINT = REPO_DIR / "checkpoints" / "model.safetensors"
DEFAULT_CHECKPOINT_RUN = "checkpoints"

MODES = ["mesh", "skeleton", "mesh_skel"]
LAYOUTS = ["overlay", "side"]     # 2D 布局：overlay=叠加同画面；side=左右并排
DEFAULT_LAYOUT = "side"           # 推理后默认把 GT 与 PRED 分成左右两画面
CONTENTS = ["both"]   # 2D 只保留端到端 GT vs PRED；手/相机隔离诊断由 3D 面板负责
# 相机推理策略：chunked=训练窗长分窗+相邻窗 SE(3) 链式拼接（窗间误差累积）；
# max_chunked=以 exact full 安全上限为窗长（默认）；streaming=原生流式；full=限长整段单次普通前向。
CAM_MODES = ["chunked", "max_chunked", "streaming", "full"]
DEFAULT_CAM_MODE = "max_chunked"
# 手部窗口拼接：hard=原始硬切；blend=重叠线性融合；
# smooth=blend 后复用 wuji-data-infra 生产参数的相机系 UKF+RTS 双向平滑。
HAND_MODES = ["hard", "blend", "smooth"]
DEFAULT_HAND_MODE = "smooth"
# Viewer starts slightly lighter than production (0.6/0.6/2.0); users can tune all three.
DEFAULT_UKF_PARAMS = {"q": 0.7, "r": 0.5, "beta": 0.3}
# 手形 betas / 内参 FoV 可视化：per_frame=用每帧各自值（现状）；mean=用整段平均值（去逐帧抖动）。
PARAM_MODES = ["per_frame", "mean"]
DEFAULT_PARAM_MODE = "per_frame"
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")   # --no_truth 裸视频模式识别的扩展名

try:
    ROBOT_RENDER_WIDTH = max(320, int(os.environ.get("VIEWER_ROBOT_RENDER_WIDTH", "640")))
except ValueError:
    ROBOT_RENDER_WIDTH = 640
