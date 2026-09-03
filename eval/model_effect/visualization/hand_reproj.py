#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化离线入口：学生模型预测的双手 MANO + 相机，用预测相机重投影回 ego 视频（模型经 --model 选）。

两种输入自动判别：
  · lerobot v3（训练就绪 export）：左 = 教师 GT（GT 相机 + GT 手），右 = 预测（预测相机 + 预测手），
    左右并排同帧对比，肉眼检验对齐（核心验证，有数值意义）。
  · 裸视频（.mp4 等，没见过）：只出预测 overlay 单画面，看泛化（定性）。

输出：output/eval/<model>/hand_reproj/<时间戳>/compare.mp4（mp4v 写完转 H.264，可直接播）。

用法（使用新项目的 mint 环境）：
    PY=python
    # lerobot held-out：GT vs 预测
    $PY eval/model_effect/visualization/hand_reproj.py --input <lerobot_v3 目录> --ckpt <step_* 目录> \
        --episode 0 --max-frames 64 --mode mesh_skel
    # 裸视频：仅预测 overlay
    $PY eval/model_effect/visualization/hand_reproj.py --input <某视频.mp4> --ckpt <step_* 目录> \
        --fps 10 --max-frames 64
    # 管线 smoke（不给 --ckpt：骨干用 config 预训练、hand_head 随机，仅验证跑通）
    $PY eval/model_effect/visualization/hand_reproj.py --input <lerobot_v3 目录> --episode 0 --max-frames 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 让 visualization / inference / predictors 顶级包可 import（本文件作脚本跑时）：注入包根 model_effect。
_HERE = os.path.dirname(os.path.abspath(__file__))          # visualization/
_PKG_ROOT = os.path.dirname(_HERE)                          # model_effect/（包根，供跨顶级包绝对 import）
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

REPO_DIR = Path(_HERE).resolve().parents[2]   # visualization -> model_effect -> eval -> <repo>
_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def main():
    ap = argparse.ArgumentParser(description="lingbotmap 重投影验证：GT vs 预测 / 裸视频预测 overlay")
    ap.add_argument("--input", required=True, help="lerobot v3 数据集目录 或 单个视频文件")
    ap.add_argument("--model", default="lingbotmap", help="推理模型（inference.registry 注册名，如 lingbotmap / vggt）")
    ap.add_argument("--config", default=None, help="训练 config（取 model 结构 + size_hw）；不给则用该模型默认")
    ap.add_argument("--ckpt", default=None, help="训练 ckpt（accelerate step_* 目录或权重文件）；不给则走 smoke")
    ap.add_argument("--out", default=None, help="输出目录（默认 output/eval/<model>/hand_reproj/<ts>）")
    ap.add_argument("--episode", type=int, default=0, help="lerobot：选第几个 episode（列表序号）")
    ap.add_argument("--max-frames", type=int, default=None, help="最多帧数（长序列建议设置，控显存/耗时）")
    ap.add_argument("--window", type=int, default=None,
                    help="分窗前向窗口大小（帧）；默认自动=该 ckpt 训练 clip_len，保证与训练前向一致")
    ap.add_argument("--fps", type=float, default=30.0, help="裸视频抽帧帧率 / 输出回放帧率")
    ap.add_argument("--mode", default="mesh_skel", choices=["mesh", "skeleton", "mesh_skel"], help="绘制模式")
    ap.add_argument("--alpha", type=float, default=0.6, help="mesh 叠加透明度")
    ap.add_argument("--hand-frame", choices=["camera", "world"], default="world",
                    help="裸视频模式:预测手的坐标系(cam 集训模型填 camera)。"
                         "lerobot 模式忽略此项,由数据集 info.json 的 hand_frame 决定")
    args = ap.parse_args()

    from visualization.reproj_core import lerobot_io, mano
    from inference.registry import get_predictor
    from visualization.render import compare

    mano.ensure_mano_weights()

    inp = Path(args.input)
    is_video = inp.is_file() and inp.suffix.lower() in _VIDEO_EXTS
    ds_dir = None if is_video else lerobot_io.find_dataset(inp)
    if not is_video and ds_dir is None:
        print(f"[ERROR] --input 既不是视频文件也不是 lerobot v3 数据集（找不到 meta/info.json）：{inp}")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else REPO_DIR / "output" / "eval" / args.model / \
        "hand_reproj" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compare.mp4"

    predictor = get_predictor(args.model, config=args.config, ckpt=args.ckpt, window=args.window)

    # ---------------- lerobot：GT vs 预测 并排 ----------------
    if not is_video:
        eps = lerobot_io.discover_episodes(ds_dir)
        if not eps:
            print(f"[ERROR] 未在 {ds_dir} 找到 episode。")
            sys.exit(1)
        if args.episode < 0 or args.episode >= len(eps):
            print(f"[ERROR] --episode {args.episode} 越界（共 {len(eps)} 个）。")
            sys.exit(1)
        raw = lerobot_io.load_episode_raw(eps[args.episode], max_frames=args.max_frames)
        T, H, W = raw["frames"].shape[:3]
        print(f"[hand_reproj] episode={raw['episode_index']} T={T} HxW={H}x{W}")

        pred = predictor.predict(raw["frames"])
        if "hand" not in pred:
            print("[WARN] 模型未输出 hand（enable_hand 关？），预测侧将只重投影相机、无手部。")
        print("[hand_reproj] 解码 MANO + 渲染 GT｜PRED 并排...", flush=True)
        compare.render_compare_overlay(raw, pred, out_path, mode=args.mode, alpha=args.alpha, fps=args.fps)

    # ---------------- 裸视频：仅预测 overlay ----------------
    else:
        frames = lerobot_io.read_video_frames(str(inp), fps=args.fps, max_frames=args.max_frames)
        T, H, W = frames.shape[:3]
        print(f"[hand_reproj] 裸视频 {inp.name} T={T} HxW={H}x{W}")

        pred = predictor.predict(frames)
        if "hand" not in pred:
            print("[ERROR] 模型未输出 hand，裸视频模式无可视内容。")
            sys.exit(1)
        print("[hand_reproj] 解码 MANO（预测）+ 渲染 overlay...", flush=True)
        compare.render_pred_overlay(frames, pred, out_path, mode=args.mode, alpha=args.alpha,
                                    fps=args.fps, hand_frame=args.hand_frame)

    print(f"[hand_reproj] 完成 → {out_path}")


if __name__ == "__main__":
    main()
