#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网页端 lingbotmap 效果验证上位机（瘦入口）。

同页看「2D 重投影 overlay 视频」+ 世界系 3D 双视图，逐帧联动；远端无 X11，浏览器打开即用。
后端已拆到 viewer/ 子包：
  · viewer/const.py   路径 + 枚举常量
  · viewer/ckpts.py   ckpt 发现/浏览/tag/config 快照解析
  · viewer/netutil.py 端口自愈 + 自动开浏览器
  · viewer/store.py   Store（缓存/推理/GT/mp4/取消/进度/模型&视频懒加载）
  · viewer/routes.py  create_app(store) —— Flask 路由 + /video + 静态资源
  · viewer/web/       前端 index.html / style.css / app.js
本文件只做：解析参数 → 定位数据集/ckpt → 建 Store → create_app → 起后台线程(模型/视频统计/开浏览器) → run。

网页秒开：模型（含 4.6G 骨干）+ MANO 资产在 app.run 后台线程加载；启动不预扫描——前端逐级浏览
--input 根，浏览到哪层才登记条目 eid。有/无真值**按浏览到的路径判定**：进入含 meta/info.json 的
lerobot 数据集目录 → 出 episode 选择器（有 GT/loss）；进入普通目录 → 列该目录视频文件、点击即加载
（无真值、仅预测）。加载/推理可停止；换 ckpt 走文件目录浏览器。

环境：新项目的 mint python（需 flask + torch + smplx + cv2；overlay 转码需命令行 ffmpeg）。
  PY=python
  # --input 给任意根目录，前端浏览时自动分辨 lerobot（选 episode）/ 普通目录（点视频）：
  $PY eval/model_effect/visualization/viewer_web.py \
      --input /path/to/input --ckpt <step_* 目录> [--host 0.0.0.0 --port 8011]
  # 裸视频项的默认抽帧/坐标系（lerobot 项忽略，读各自 info.json）：[--hand_frame camera --fps 30]
启动后默认自动打开浏览器；使用 --no-open 可关闭这一行为。
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

# 让 visualization / inference / predictors 顶级包可 import（本文件作脚本跑时）：注入包根 model_effect。
_HERE = os.path.dirname(os.path.abspath(__file__))          # visualization/
_PKG_ROOT = os.path.dirname(_HERE)                          # model_effect/（包根，供跨顶级包绝对 import）
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from inference.registry import default_config              # noqa: E402
from visualization.viewer import ckpts, netutil            # noqa: E402
from visualization.viewer.const import (CONTENTS, DEFAULT_CHECKPOINT, LAYOUTS,  # noqa: E402
                                        MODEL_TRAIN_ROOT, MODES, REPO_DIR)
from visualization.viewer.routes import create_app         # noqa: E402
from visualization.viewer.store import Store               # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="网页端 lingbotmap 效果验证：2D GT|PRED 重投影视频 + 世界系 3D 双视图",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--input", default=str(REPO_DIR / "data" / "samples"),
                   help="浏览**起点**目录（默认指向 data/samples）；不再是天花板——"
                        "网页里点「⬆ 上一级」可上溯到任意系统路径。进入含 meta/info.json 的 lerobot 目录→选 "
                        "episode（有 GT），进入普通目录→点视频文件加载（无真值，仅预测）")
    p.add_argument("--hand_frame", choices=["camera", "world"], default="camera",
                   help="裸视频项预测手的坐标系（当前 camera 系 ckpt 用 camera；老 world 系 ckpt 用 world；lerobot 项忽略）")
    p.add_argument("--fps", type=float, default=30.0, help="裸视频项抽帧/回放帧率（默认 30；lerobot 项忽略，读各自 info.json）")
    p.add_argument("--model", default="lingbotmap", help="推理模型（inference.registry 注册名，如 lingbotmap / vggt）")
    p.add_argument("--config", default=None, help="训练 config（取 model 结构 + size_hw）；不给则用该模型默认")
    p.add_argument(
        "--ckpt", default=None,
        help="训练 ckpt（step_* 目录或权重文件）；不给则优先最新训练权重，再用 checkpoints/model.safetensors",
    )
    p.add_argument("--window", type=int, default=None,
                   help="分窗前向窗口大小（帧）；默认自动=该 ckpt 训练 clip_len，保证与训练前向一致")
    p.add_argument(
        "--full-max-frames", type=int, default=None,
        help="exact full 安全上限/最大分窗窗长；须为训练窗整数倍。默认按所选 GPU 最小显存取 1/2/3 倍",
    )
    p.add_argument(
        "--devices", default="auto",
        help="交互推理设备：auto=加载模型时使用全部可见 GPU（默认）；也可指定 0,1 或 cpu",
    )
    p.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"),
        default=None,
        help="可选 torch.compile 模式；固定分窗推荐 reduce-overhead（含 CUDA Graph）",
    )
    p.add_argument(
        "--fp8-mode", choices=("dynamic",), default=None,
        help="可选 torchao FP8 dynamic activation + weight，仅量化 aggregator 大 Linear",
    )
    p.add_argument("--max-frames", type=int, default=None, help="每 episode 最多帧数（控显存/耗时，默认全量）")
    p.add_argument(
        "--cache-dir", default=os.environ.get("VIEWER_CACHE_DIR"),
        help="交互视频/预测缓存根目录；默认使用本机临时盘 /tmp/wuji-viewer-cache，"
             "也可通过 VIEWER_CACHE_DIR 设置",
    )
    p.add_argument("--mode", default="mesh_skel", choices=MODES, help="overlay 默认绘制模式")
    p.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    p.add_argument("--port", type=int, default=8011, help="端口（默认 8011）")
    p.add_argument("--preload", action="store_true",
                   help="启动即后台预载 4.6G 模型（默认不加载：加载与推理已拆开，须在网页选模型后手动点「加载模型」才载入）")
    p.add_argument("--prerender", action="store_true", help="启动后后台把全部 episode 预测+预渲好（默认按需）")
    p.add_argument("--jobs", "-j", type=int, default=2,
                   help="批量视频的并发渲染/写盘数，也用于 --prerender（默认 2；GPU 推理会与这些任务流水并发）")
    p.add_argument("--no-open", dest="open_browser", action="store_false",
                   help="启动后不自动弹出浏览器（默认自动打开）")
    args = p.parse_args()

    # config 模板优先级：命令行 --config > 该模型适配器默认（default_config）。二者皆无（如 vggt 占位）
    # 则为 None，后续 ckpts 调用安全降级，模型真正构建时由适配器抛清晰的「需 --config」错误。
    cfg_tmpl = args.config or default_config(args.model)

    # 启动前自愈端口：固定端口、强制腾出——占用者不管是不是本脚本一律杀掉，始终复用该端口；
    # 实在杀不掉才退回换端口。
    # 重活（MANO、模型）挪到 app.run 之后的后台线程；数据集/视频均**不预扫描**，浏览到才登记。
    args.port = netutil.ensure_port(args.port)

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.is_dir():        # 默认起点不存在也不崩：回退仓库根，让用户从网页里自己上溯/下钻找
        print(f"[input] 起点 {in_path} 不是目录，回退到仓库根 {REPO_DIR}", flush=True)
        in_path = REPO_DIR.resolve()
    root = in_path
    scene = in_path.name or "root"
    print(f"[input] 浏览起点：{in_path}（网页可「⬆ 上一级」到任意路径；lerobot 目录选 episode / 普通目录点视频；不预扫描）"
          f" 裸视频默认 手坐标系={args.hand_frame} fps={args.fps}")

    cache_root = (Path(args.cache_dir).expanduser().resolve() if args.cache_dir
                  else Path(tempfile.gettempdir()) / "wuji-viewer-cache")
    cache_dir = cache_root / args.model / scene / time.strftime("%Y%m%d_%H%M%S")
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt is None:
        args.ckpt = ckpts.auto_pick_ckpt()
        if args.ckpt:
            if Path(args.ckpt).resolve() == DEFAULT_CHECKPOINT.resolve():
                print(f"[ckpt] 未指定 --ckpt，自动选用项目 checkpoint: {args.ckpt}")
            else:
                print(f"[ckpt] 未指定 --ckpt，自动选用 model_train 下最新: {args.ckpt}")
        else:
            print(f"[ckpt] {MODEL_TRAIN_ROOT} 和 {DEFAULT_CHECKPOINT} 均无可用权重，走 inspect/smoke（随机权重）")

    # 模型结构/size_hw：优先用 ckpt 所属 run 自带的 config 快照，回退到命令行 --config 模板。
    cfg_used = ckpts.config_for_ckpt(args.ckpt, cfg_tmpl)
    if cfg_used != cfg_tmpl:
        print(f"[config] 使用 ckpt 所属 run 自带 config: {cfg_used}")
    else:
        print(f"[config] 使用命令行/模型默认 config: {cfg_used}")
    loss_cfg = ckpts.load_loss_cfg(cfg_used)

    # predictor=None → 模型放后台线程加载（网页秒开）；首次推理前 ensure_predictor 阻塞等就绪。
    store = Store(None, root=root, default_mode=args.mode,
                  max_frames=args.max_frames, scene=scene, cache_dir=cache_dir,
                  loss_cfg=loss_cfg, config_path=cfg_tmpl, window=args.window,
                  full_max_frames=args.full_max_frames,
                  ckpt_path=args.ckpt, default_fps=float(args.fps),
                  default_hand_frame=args.hand_frame, init_config=cfg_used,
                  model=args.model, inference_devices=args.devices,
                  inference_compile_mode=args.compile_mode,
                  inference_fp8_mode=args.fp8_mode,
                  batch_workers=args.jobs)
    app = create_app(store)
    print(f"[info] 浏览根: {root}")
    print(f"[info] overlay/预测 缓存目录: {cache_dir}")
    print(f"[info] 打开浏览器访问:  http://localhost:{args.port}/")
    print("[info] 网页已可访问；4.6G 模型"
          + ("已按 --preload 启动即后台预载。" if args.preload
             else "默认不加载——在网页选模型后手动点 [⬇ 加载模型] 载入；加载与推理已拆开，加载中不可推理。"))

    # 后台预热：MANO 资产始终后台加载（轻，GT/仅看原始都要）；4.6G 模型默认不加载，
    # 须手动点 [⬇ 加载模型] 才载入（加载与推理拆开）。--preload 恢复启动即预载模型。
    def _boot():
        try:
            store.ensure_assets()
            if args.preload:
                store.ensure_predictor()
        except Exception as e:  # noqa: BLE001
            print(f"[model] 后台加载失败: {e}", flush=True)
    threading.Thread(target=_boot, name="boot", daemon=True).start()
    if args.prerender:
        threading.Thread(target=store.prerender,
                         args=(args.jobs, args.mode, LAYOUTS[0], CONTENTS[0]),
                         daemon=True).start()
    if args.open_browser:                   # 默认自动弹出浏览器
        threading.Thread(target=netutil.open_browser, args=(args.port,), daemon=True).start()

    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
