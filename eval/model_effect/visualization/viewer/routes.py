#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flask app 工厂：create_app(store) 绑定全部 /api 路由 + /video + 静态资源(web/)。

前端静态文件在 viewer/web/{index.html,style.css,i18n.js,app.js}，index.html 用相对路径外链
（兼容被反向代理到子路径的场景，如 code-server /proxy/8013/）。路由用闭包捕获 store，无全局单例。
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

from . import ckpts, diversity_analysis
from .const import (CAM_MODES, CONTENTS, DEFAULT_CAM_MODE, DEFAULT_HAND_MODE,
                    DEFAULT_LAYOUT, DEFAULT_PARAM_MODE, DEFAULT_UKF_PARAMS,
                    HAND_MODES, LAYOUTS, MODEL_TRAIN_ROOT, MODES, PARAM_MODES,
                    VIDEO_EXTS)
from .dataset_analysis import DatasetAnalysisManager

WEB_DIR = Path(__file__).resolve().parent / "web"
WORLD_COORD_MODES = {"z_up", "opencv"}


def _normalize_hand_mode(value: str) -> str:
    """Validate base or parameterized smooth mode and return a stable cache key."""
    from inference.hand_smoothing import (
        encode_hand_smoothing_mode,
        parse_hand_smoothing_mode,
    )

    base, params = parse_hand_smoothing_mode(value)
    return encode_hand_smoothing_mode(params) if params is not None else base


def _is_lerobot_dir(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def _default_input_path(root: Path) -> Path:
    """Enter a directly nested LeRobot dataset instead of opening its parent."""
    root = Path(root).expanduser().resolve()
    if _is_lerobot_dir(root):
        return root

    for name in ("lerobot_v3", "lerobot"):
        candidate = root / name
        if _is_lerobot_dir(candidate):
            return candidate.resolve()

    datasets = []
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                try:
                    if entry.is_dir() and _is_lerobot_dir(Path(entry.path)):
                        datasets.append(Path(entry.path).resolve())
                        if len(datasets) > 1:
                            break
                except OSError:
                    continue
    except OSError:
        return root
    return datasets[0] if len(datasets) == 1 else root


def create_app(store) -> Flask:
    app = Flask(__name__)
    dataset_analysis = DatasetAnalysisManager(
        default_root=Path(getattr(store, "root", None) or Path.cwd()),
    )
    export_files: dict[str, tuple[Path, str]] = {}
    export_jobs: dict[str, dict] = {}
    export_lock = threading.Lock()
    # 压掉 werkzeug 每请求的 200 访问日志（/api/progress 轮询等刷屏大头）；4xx/5xx 仍会以 WARNING 打出。
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.after_request
    def _no_cache(resp):
        """页面/API 禁缓存；带参数的视频允许浏览器复用已下载的 Range 分块。"""
        if request.endpoint in {"video", "world_video", "mujoco_video", "retarget_video", "export_file"} \
                and resp.status_code in (200, 206):
            # 视频 URL 已带 ckpt/渲染参数；允许浏览器复用 Range 分块，避免长视频
            # 播放或回看时反复从 CPFS 拉取同一段。private 防止共享代理串内容。
            resp.headers["Cache-Control"] = "private, max-age=3600"
            resp.headers.pop("Pragma", None)
            resp.headers["Accept-Ranges"] = "bytes"
            return resp
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    # ---- 静态资源（前端单页）----
    @app.get("/")
    def index():
        return send_file(WEB_DIR / "index.html")

    @app.get("/style.css")
    def style_css():
        return send_file(WEB_DIR / "style.css", mimetype="text/css")

    @app.get("/app.js")
    def app_js():
        return send_file(WEB_DIR / "app.js", mimetype="application/javascript")

    @app.get("/i18n.js")
    def i18n_js():
        return send_file(WEB_DIR / "i18n.js", mimetype="application/javascript")

    @app.get("/benchmark_suites.js")
    def benchmark_suites_js():
        return send_file(WEB_DIR / "benchmark_suites.js", mimetype="application/javascript")

    @app.get("/camera_baselines.js")
    def camera_baselines_js():
        return send_file(WEB_DIR / "camera_baselines.js", mimetype="application/javascript")

    # ---- 元信息 / 浏览 ----
    @app.get("/api/episodes")
    def api_episodes():
        # 启动不预扫描：只给前端默认起点（绝对路径）+ 默认叠加模式。有/无真值由浏览到的目录
        # 是否 lerobot 数据集动态判定（见 /api/ibrowse）；起点仅默认，网页可上溯任意路径。
        diversity_root = diversity_analysis.DEFAULT_ROOT.resolve()
        diversity_paths = [
            {"dataset": name, "label": config["label"],
             "path": str(diversity_root / config["dirs"][0] / "lerobot_v3")}
            for name, config in diversity_analysis.DATASETS.items()
        ]
        return jsonify(default_mode=store.default_mode,
                       default_path=str(_default_input_path(store.root)),
                       diversity_root=str(diversity_root),
                       diversity_paths=diversity_paths)

    @app.get("/api/ckpts")
    def api_ckpts():
        cur = store.ckpt_path
        cur_run, cur_step = ckpts.run_step_for_path(cur)
        return jsonify(root=str(MODEL_TRAIN_ROOT), runs=ckpts.list_runs(),
                       current={"run": cur_run,
                                "step": cur_step,
                                "path": cur, "tag": store.ckpt_tag})

    @app.get("/api/ckpts/<path:run>")   # run 名可含 '/'(多级:<gpu>/<ts>/<task>),须用 path 转换器
    def api_ckpt_steps(run: str):
        return jsonify(steps=ckpts.list_steps(run))

    @app.get("/api/browse")             # 逐级浏览:?path=<相对 model_train>,''=根。返回子目录 + 是否到 run 层(+steps)
    def api_browse():
        return jsonify(**ckpts.browse(request.args.get("path", "")))

    @app.get("/api/ckpt/browse")
    def api_ckpt_browse():
        """Browse arbitrary server directories and expose supported checkpoint files."""
        raw = (request.args.get("path") or "").strip()
        if raw:
            target = Path(raw).expanduser()
            if not target.is_absolute():
                target = ckpts.REPO_DIR / target
        elif store.ckpt_path:
            target = Path(store.ckpt_path).expanduser()
        else:
            target = ckpts.DEFAULT_CHECKPOINT.parent
        try:
            target = target.resolve()
        except OSError as exc:
            return jsonify(error=f"Checkpoint 路径无法解析: {exc}"), 400
        selected = target.name if target.is_file() else None
        directory = target.parent if target.is_file() else target
        if not directory.is_dir():
            return jsonify(error=f"Checkpoint 目录不存在或不可访问: {directory}"), 400

        dirs, files, truncated = [], [], False
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if len(dirs) + len(files) >= 2000:
                        truncated = True
                        break
                    try:
                        if entry.is_dir():
                            dirs.append(entry.name)
                        elif entry.is_file() and Path(entry.name).suffix.lower() in ckpts.CHECKPOINT_FILE_EXTS:
                            files.append(entry.name)
                    except OSError:
                        continue
        except OSError as exc:
            return jsonify(error=f"Checkpoint 目录不可读取: {exc}"), 400
        dirs.sort(key=str.lower)
        files.sort(key=str.lower)
        return jsonify(
            path=str(directory), parent=str(directory.parent), dirs=dirs, files=files,
            selected=selected, selectable=ckpts.checkpoint_weight_file(directory) is not None,
            current=store.ckpt_path, truncated=truncated,
        )

    @app.get("/api/ibrowse")            # 浏览任意**绝对**目录：列子目录 + 判定是否 lerobot 数据集（不锁根）
    def api_ibrowse():
        """path=绝对路径（空→默认起点 store.root）。不限制在 --input 根内，可上溯任意系统路径。
        返回 {path, parent, dirs, lerobot, episodes, fps, videos:[{name,eid}]}：
          · 当前目录直含 meta/info.json → lerobot=True，给 episode 数 + fps（前端出 episode 选择器），不列视频；
          · 否则 → lerobot=False，懒登记该目录下视频文件的 eid（前端点击只选中，无真值）。
        parent=父目录绝对路径（到文件系统根时 parent==path，前端据此隐藏「上一级」）。"""
        p = request.args.get("path", "")
        default_input = _default_input_path(store.root)
        d = Path(p).resolve() if p else default_input
        if not d.is_dir():                          # 路径无效 → 回默认起点
            d = default_input
        parent = str(d.parent)                      # 根时 d.parent == d（Path 语义），前端自行判断
        # 大目录提速：os.scandir 一次遍历同时分目录/文件，DirEntry 类型走目录缓存、不逐条 stat
        # （pathlib 的 is_dir/is_file 每条一次 stat，几万条目会卡）；再加 MAX_ENTRIES 上限截断，
        # 避免海量条目撑爆前端渲染与 eid 登记。d 已 resolve → e.path 即绝对路径，省去逐条 resolve。
        MAX_ENTRIES = 2000
        is_lerobot = (d / "meta" / "info.json").exists()
        dirs, videos, truncated = [], [], False
        try:
            with os.scandir(d) as it:
                for e in it:
                    if len(dirs) + len(videos) >= MAX_ENTRIES:
                        truncated = True
                        break
                    try:
                        if e.is_dir():
                            dirs.append(e.name)
                        elif (not is_lerobot) and e.is_file() \
                                and os.path.splitext(e.name)[1].lower() in VIDEO_EXTS:
                            videos.append({"name": e.name, "eid": store.video_eid(e.path)})
                    except OSError:
                        continue                    # 单条坏链接/无权限 → 跳过
        except OSError:                             # 目录不可读：仍给 path/parent，列表空
            return jsonify(path=str(d), parent=parent, dirs=[], lerobot=False, videos=[])
        dirs.sort(key=str.lower)
        videos.sort(key=lambda v: v["name"].lower())
        if is_lerobot:                              # 本目录即 lerobot 数据集：**异步**枚举，立即返回（不阻塞浏览）
            st = store.ensure_dataset_async(d)      # 已缓存→ready；否则后台枚举，前端轮询 /api/dataset_progress
            return jsonify(path=str(d), parent=parent, dirs=dirs, lerobot=True,
                           ready=bool(st.get("ready")), episodes=st.get("episodes"),
                           fps=st.get("fps"), videos=[], truncated=truncated)
        return jsonify(path=str(d), parent=parent, dirs=dirs, lerobot=False, videos=videos, truncated=truncated)

    @app.get("/api/dataset_progress")   # 前端轮询：某 lerobot 数据集（绝对路径）的 episode 枚举进度
    def api_dataset_progress():
        p = request.args.get("path", "")
        d = Path(p).resolve() if p else _default_input_path(store.root)
        return jsonify(**store.dataset_progress(d))

    @app.get("/api/dirbrowse")          # 批量任务目录选择器：只列子目录，不扫描/登记视频
    def api_dirbrowse():
        p = request.args.get("path", "")
        d = Path(p).resolve() if p else store.root.resolve()
        if not d.is_dir():
            return jsonify(error=f"目录不存在或不可访问: {d}"), 400
        dirs = []
        truncated = False
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir():
                            dirs.append(entry.name)
                            if len(dirs) >= 2000:
                                truncated = True
                                break
                    except OSError:
                        continue
        except OSError as exc:
            return jsonify(error=f"目录不可读取: {exc}"), 400
        dirs.sort(key=str.lower)
        return jsonify(path=str(d), parent=str(d.parent), dirs=dirs, truncated=truncated)

    @app.get("/api/lerobot_eid")        # 选中某 lerobot 数据集（绝对路径）的第 ep 个 episode → 分配/复用 eid
    def api_lerobot_eid():
        p = request.args.get("path", "")
        d = Path(p).resolve() if p else store.root.resolve()
        try:
            ep = int(request.args.get("ep", "0"))
            eid = store.lerobot_eid(d, ep)
        except (ValueError, IndexError) as e:
            abort(400, str(e))
        except Exception as e:  # noqa: BLE001
            abort(500, f"定位 episode 失败: {e}")
        return jsonify(eid=eid)

    # ---- ckpt 切换 / 取消 ----
    @app.post("/api/cancel/<int:eid>")  # 「停止」：置位事件，数据分块/推理窗口/payload 阶段协作式中断
    def api_cancel(eid: int):
        if not 0 <= eid < store.n_items():
            abort(404, "episode 越界")
        print(f"[action] 停止 eid={eid}（等待当前在途 I/O 或 GPU 调用结束）", flush=True)
        # 先发布 cancelling 状态，再置事件；即使 world 请求此刻才进入，也不会把这次取消
        # 当作上一轮残留而清掉。
        store.set_prog(eid, stage="cancelling")
        store.request_cancel(eid)
        return jsonify(ok=True)

    @app.post("/api/ckpt")
    def api_set_ckpt():
        body = request.get_json(force=True, silent=True) or {}
        if body.get("path"):
            ckpt = ckpts.resolve_checkpoint_path(body.get("path"))
        else:
            ckpt = ckpts.resolve_ckpt(body.get("run"), body.get("step"))
        if ckpt is None:
            abort(400, "Checkpoint 路径无效，或目录中没有受支持的权重文件")
        try:
            res = store.swap_ckpt(str(ckpt))       # swap_ckpt 内部已打印切换/重载方式
        except RuntimeError as e:
            abort(409, str(e))
        except Exception as e:  # noqa: BLE001
            abort(500, f"切换 ckpt 失败: {e}")
        return jsonify(ok=True, **res)

    # ---- 批量视频推理：递归扫描输入目录，镜像保存预测 NPZ 与渲染 MP4 ----
    @app.post("/api/batch/start")
    def api_batch_start():
        body = request.get_json(force=True, silent=True) or {}
        checkpoint = None
        checkpoint_run = body.get("checkpoint_run") or ""
        checkpoint_step = body.get("checkpoint_step") or ""
        if bool(checkpoint_run) != bool(checkpoint_step):
            return jsonify(ok=False, error="批量模型必须同时选择 Run 和 Step"), 400
        if checkpoint_run:
            checkpoint = ckpts.resolve_ckpt(checkpoint_run, checkpoint_step)
            if checkpoint is None:
                return jsonify(ok=False, error="批量模型 Run/Step 无效或 checkpoint 不存在"), 400
        mode = body.get("mode") or store.default_mode
        cam_mode = body.get("cam_mode") or DEFAULT_CAM_MODE
        try:
            hand_mode = _normalize_hand_mode(body.get("hand_mode") or DEFAULT_HAND_MODE)
        except (TypeError, ValueError) as exc:
            return jsonify(ok=False, error=str(exc)), 400
        pred_betas = body.get("pred_betas") or DEFAULT_PARAM_MODE
        pred_fov = body.get("pred_fov") or DEFAULT_PARAM_MODE
        if mode not in MODES:
            return jsonify(ok=False, error=f"未知渲染模式: {mode}"), 400
        if cam_mode not in CAM_MODES:
            return jsonify(ok=False, error=f"未知相机推理模式: {cam_mode}"), 400
        if pred_betas not in PARAM_MODES or pred_fov not in PARAM_MODES:
            return jsonify(ok=False, error="预测手形/内参模式无效"), 400
        res = store.start_batch_inference(
            input_dir=body.get("input_dir") or "",
            output_dir=body.get("output_dir") or "",
            name_template=body.get("name_template") or "{stem}_pred",
            mode=mode, cam_mode=cam_mode, hand_mode=hand_mode,
            pred_betas_mean=(pred_betas == "mean"),
            pred_fov_mean=(pred_fov == "mean"),
            overwrite=bool(body.get("overwrite", False)),
            checkpoint=str(checkpoint) if checkpoint else None,
        )
        if not res.get("ok"):
            return jsonify(**res), 409
        return jsonify(**res)

    @app.get("/api/batch/status")
    def api_batch_status():
        return jsonify(**store.batch_inference_status())

    @app.post("/api/batch/cancel")
    def api_batch_cancel():
        res = store.cancel_batch_inference()
        return (jsonify(**res), 200 if res.get("ok") else 409)

    # ---- 数据集分析：视频、LeRobot 内参与多样性统计、缓存与导出 ----
    @app.post("/api/dataset-analysis/start")
    def api_dataset_analysis_start():
        body = request.get_json(force=True, silent=True) or {}
        analysis_type = body.get("analysis_type") or "video"
        try:
            workers = int(body.get("workers") or (32 if analysis_type == "video" else 8))
            sample_files = int(body.get("sample_files") or 24)
        except (TypeError, ValueError):
            return jsonify(ok=False, error="并发数和轨迹采样文件数必须是整数"), 400
        selected_datasets = body.get("datasets")
        if selected_datasets is not None and not isinstance(selected_datasets, list):
            return jsonify(ok=False, error="datasets 必须是数组"), 400
        input_dirs = body.get("input_dirs")
        if input_dirs is not None and not isinstance(input_dirs, list):
            return jsonify(ok=False, error="input_dirs 必须是数组"), 400
        result = dataset_analysis.start(
            input_dir=body.get("input_dir") or "",
            workers=workers,
            refresh=bool(body.get("refresh", False)),
            analysis_type=analysis_type,
            selected_datasets=selected_datasets,
            sample_files=sample_files,
            input_dirs=input_dirs,
        )
        return (jsonify(**result), 200 if result.get("ok") else 409)

    @app.get("/api/dataset-analysis/status")
    def api_dataset_analysis_status():
        return jsonify(**dataset_analysis.status())

    @app.post("/api/dataset-analysis/cancel")
    def api_dataset_analysis_cancel():
        result = dataset_analysis.cancel()
        return (jsonify(**result), 200 if result.get("ok") else 409)

    @app.get("/api/dataset-analysis/result")
    def api_dataset_analysis_result():
        try:
            result = dataset_analysis.result_page(
                page=int(request.args.get("page", "1")),
                page_size=int(request.args.get("page_size", "100")),
                search=request.args.get("search", ""),
                anomaly=request.args.get("anomaly", ""),
                codec=request.args.get("codec", ""),
                orientation=request.args.get("orientation", ""),
                resolution=request.args.get("resolution", ""),
                sort=request.args.get("sort", "relative_path"),
                order=request.args.get("order", "asc"),
            )
        except (TypeError, ValueError) as exc:
            return jsonify(error=f"分页参数无效: {exc}"), 400
        except LookupError as exc:
            return jsonify(error=str(exc)), 404
        return jsonify(**result)

    @app.get("/api/dataset-analysis/export")
    def api_dataset_analysis_export():
        format_name = (request.args.get("format") or "json").lower()
        if format_name not in {"json", "csv", "txt"}:
            return jsonify(error="导出格式仅支持 json/csv/txt"), 400
        try:
            path, filename = dataset_analysis.export_path(format_name)
        except LookupError as exc:
            return jsonify(error=str(exc)), 404
        return send_file(path, as_attachment=True, download_name=filename)

    # ---- benchmark：对一个或多个 ckpt 跑相同量化评测（后台子进程），前端面板轮询 ----
    @app.get("/api/benchmark/history")
    def api_bench_history():
        from benchmark.cache import list_benchmark_history

        run = request.args.get("run") or None
        step = request.args.get("step") or None
        try:
            records = list_benchmark_history(run, step)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except FileNotFoundError as exc:
            return jsonify(error=str(exc)), 404
        return jsonify(root=str(MODEL_TRAIN_ROOT.resolve()), run=run, step=step, records=records)

    @app.get("/api/benchmark/history/result")
    def api_bench_history_result():
        from benchmark.cache import load_benchmark_history

        run = request.args.get("run") or ""
        record_id = request.args.get("record_id") or ""
        try:
            record = load_benchmark_history(run, record_id)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except FileNotFoundError as exc:
            return jsonify(error=str(exc)), 404
        return jsonify(**record)

    @app.post("/api/benchmark/start")
    def api_bench_start():
        body = request.get_json(force=True, silent=True) or {}
        checkpoints = None
        if "models" in body:
            requested = body.get("models")
            if not isinstance(requested, list) or not requested:
                return jsonify(ok=False, error="请至少添加一个 Benchmark 模型"), 400
            if len(requested) > 16:
                return jsonify(ok=False, error="一次最多比较 16 个 Benchmark 模型"), 400
            checkpoints = []
            seen = set()
            for item in requested:
                if not isinstance(item, dict):
                    return jsonify(ok=False, error="Benchmark 模型格式无效"), 400
                run, step = item.get("run") or "", item.get("step") or ""
                checkpoint = (ckpts.resolve_checkpoint_path(item.get("path"))
                              if item.get("path") else ckpts.resolve_ckpt(run, step))
                if checkpoint is None:
                    label = item.get("path") or f"{run}/{step}"
                    return jsonify(ok=False, error=f"Benchmark 模型无效: {label}"), 400
                if not run or not step:
                    run = checkpoint.parent.name
                    step = checkpoint.name
                key = str(checkpoint.resolve())
                if key in seen:
                    continue
                seen.add(key)
                checkpoints.append({
                    "run": run, "step": step, "ckpt": key,
                    "label": item.get("label") or f"{run} / {step}",
                })
            if not checkpoints:
                return jsonify(ok=False, error="请至少添加一个不重复的 Benchmark 模型"), 400
        res = store.start_benchmark(
            datasets=(body.get("datasets") or "all"),
            heads=(body.get("heads") or "all"),
            max_seqs=(body.get("max_seqs") or None),
            max_frames=(body.get("max_frames") or None),
            devices=(body.get("devices") or None),
            seq_start=body.get("seq_start", 0),
            seq_end=body.get("seq_end"),
            dataset_selection=body.get("dataset_selection"),
            checkpoints=checkpoints,
            reuse_cache=body.get("reuse_cache", True),
            auto_ukf_best=body.get("auto_ukf_best", True),
            backend=body.get("backend") or "local",
            aliyun=body.get("aliyun"))
        if not res.get("ok"):
            return jsonify(ok=False, error=res.get("error", "无法启动测评")), 409
        return jsonify(**res)

    @app.get("/api/benchmark/status")
    def api_bench_status():
        return jsonify(**store.benchmark_status())

    @app.get("/api/benchmark/aliyun/defaults")
    def api_bench_aliyun_defaults():
        try:
            return jsonify(**store.benchmark_aliyun_defaults())
        except (OSError, ValueError) as exc:
            return jsonify(error=f"读取 Aliyun 默认配置失败: {exc}"), 500

    # 能力清单:前端三态选择网格用(heads/datasets 需求与提供 + 是否实现 + 当前模型产出能力)
    @app.get("/api/benchmark/capabilities")
    def api_bench_caps():
        return jsonify(**store.benchmark_capabilities())

    # 每数据集规模(序列条数/总帧数):不加载模型,后台算一次并缓存;computing=True 时前端稍后再拉
    @app.get("/api/benchmark/sizes")
    def api_bench_sizes():
        return jsonify(**store.benchmark_sizes())

    # 本机可用 GPU 列表(index/name/显存/利用率):供面板显卡多选栏渲染,选卡后并行分片评测
    @app.get("/api/benchmark/gpus")
    def api_bench_gpus():
        return jsonify(**store.benchmark_gpus())

    @app.post("/api/benchmark/cancel")
    def api_bench_cancel():
        return jsonify(**store.cancel_benchmark())

    # ---- log_diff：对比两个训练 run 的配置和代码，后台子进程跑、前端面板轮询 ----
    @app.get("/api/logdiff/runs")   # 列含 logs/node*.log 的 run；无需已有 step_* checkpoint
    def api_logdiff_runs():
        cur = store.ckpt_path
        cur_run = None
        if cur:
            try:
                cur_run = Path(cur).resolve().parent.relative_to(MODEL_TRAIN_ROOT.resolve()).as_posix()
            except ValueError:
                cur_run = None
        return jsonify(runs=ckpts.list_log_runs(), current=cur_run)

    @app.post("/api/logdiff/start")
    def api_logdiff_start():
        body = request.get_json(force=True, silent=True) or {}
        res = store.start_logdiff(body.get("run_a") or "", body.get("run_b") or "",
                                  body.get("code_scope") or "")
        if not res.get("ok"):
            return jsonify(ok=False, error=res.get("error", "无法启动对比")), 409
        return jsonify(**res)

    @app.get("/api/logdiff/scopes")
    def api_logdiff_scopes():
        res = store.logdiff_scopes(request.args.get("run_a", ""), request.args.get("run_b", ""))
        if not res.get("ok"):
            return jsonify(**res), 400
        return jsonify(**res)

    @app.get("/api/logdiff/status")
    def api_logdiff_status():
        return jsonify(**store.logdiff_status())

    # ---- episode 元信息 / 世界系 payload / 进度 ----
    @app.get("/api/episode/<int:eid>")
    def api_episode(eid: int):
        # 兼具两用：init 用 /api/episode/0 取全局枚举（此时可能尚无条目 → fps/no_truth 用默认）；
        # 加载前也可查某已登记条目的 fps / 是否无真值。
        reg = 0 <= eid < store.n_items()
        return jsonify(fps=(store.item_fps(eid) if reg else store.default_fps),
                       no_truth=(store.is_no_truth(eid) if reg else False),
                       modes=MODES, default_mode=store.default_mode,
                       layouts=LAYOUTS, default_layout=DEFAULT_LAYOUT,
                       contents=CONTENTS, default_content=CONTENTS[0],
                       cam_modes=CAM_MODES, default_cam_mode=DEFAULT_CAM_MODE,
                       full_max_frames=getattr(
                           store, "predictor_full_max_frames", lambda: None
                       )(),
                       hand_modes=HAND_MODES, default_hand_mode=DEFAULT_HAND_MODE,
                       default_ukf_params=DEFAULT_UKF_PARAMS,
                       param_modes=PARAM_MODES, default_param_mode=DEFAULT_PARAM_MODE)

    @app.get("/api/world/<int:eid>")
    def api_world(eid: int):
        if not 0 <= eid < store.n_items():
            abort(404, "episode 越界")
        raw = bool(request.args.get("raw"))
        cam_mode = request.args.get("cam_mode", DEFAULT_CAM_MODE)
        if cam_mode not in CAM_MODES:
            cam_mode = DEFAULT_CAM_MODE
        try:
            hand_mode = _normalize_hand_mode(
                request.args.get("hand_mode", DEFAULT_HAND_MODE))
        except (TypeError, ValueError) as exc:
            abort(400, str(exc))
        gtb = request.args.get("gt_betas", DEFAULT_PARAM_MODE) == "mean"
        pdb = request.args.get("pred_betas", DEFAULT_PARAM_MODE) == "mean"
        pdf = request.args.get("pred_fov", DEFAULT_PARAM_MODE) == "mean"
        print(f"[action] {'仅看原始GT(不推理)' if raw else f'加载/推理(相机={cam_mode},手部={hand_mode})'} eid={eid} "
              f"模型就绪={store.predictor_ready()}", flush=True)
        from inference.base import FullSequenceTooLong, InferenceCancelled
        # 上一轮 cancelled/done 可以清；若停止请求抢先到达并已发布 cancelling，必须保留事件。
        progress = store.get_prog(eid)
        if progress.get("stage") != "cancelling":
            store.clear_cancel(eid)
        try:
            if raw:                      # 仅原始 GT：不跑推理，直接返回 GT-only payload
                if store.is_no_truth(eid):
                    abort(400, "裸视频项无 GT，无法「仅看原始」")
                return jsonify(**store.payload_gt(eid))
            return jsonify(**store.payload(eid, cam_mode, hand_mode, gtb, pdb, pdf))
        except InferenceCancelled:
            store.set_prog(eid, stage="cancelled")
            return jsonify(cancelled=True), 409   # 前端据此显示「已停止」，不写入 state
        except FullSequenceTooLong as exc:
            store.set_prog(eid, stage="rejected")
            return jsonify(
                code="full_too_long",
                error=str(exc),
                frames=exc.num_frames,
                full_max_frames=exc.max_frames,
                suggested_cam_mode="max_chunked",
            ), 413

    @app.get("/api/model_ready")
    def api_model_ready():
        # 前端在按钮旁显示「未加载/加载中/就绪」（默认不加载；须手动点[加载模型]才载入；加载中禁止推理）。
        return jsonify(ready=store.predictor_ready(), loading=store.loader_active(),
                       devices=store.predictor_devices(),
                       full_max_frames=getattr(
                           store, "predictor_full_max_frames", lambda: None
                       )())

    @app.post("/api/model/load")        # 「加载模型」：显式启动后台加载当前选中 ckpt（幂等、立即返回）
    def api_model_load():
        try:
            res = store.start_load()
        except Exception as e:  # noqa: BLE001
            abort(500, f"启动加载失败: {e}")
        return jsonify(**res)

    @app.get("/api/progress2d/<int:eid>")
    def api_progress2d(eid: int):
        # 前端按块轮询 2D 渲染进度：参数与 /video 一致（mode/layout/content/raw），后端按同一归一化取键。
        mode = request.args.get("mode", store.default_mode)
        layout = request.args.get("layout", DEFAULT_LAYOUT)
        content = request.args.get("content", CONTENTS[0])
        raw = bool(request.args.get("raw"))
        cam_mode = request.args.get("cam_mode", DEFAULT_CAM_MODE)
        if cam_mode not in CAM_MODES:
            cam_mode = DEFAULT_CAM_MODE
        try:
            hand_mode = _normalize_hand_mode(
                request.args.get("hand_mode", DEFAULT_HAND_MODE))
        except (TypeError, ValueError) as exc:
            abort(400, str(exc))
        pkey = (f"{int(request.args.get('gt_betas')=='mean')}"
                f"{int(request.args.get('pred_betas')=='mean')}"
                f"{int(request.args.get('pred_fov')=='mean')}")
        return jsonify(store.get_prog2d(
            store.prog2d_key(eid, mode, layout, content, raw, cam_mode, hand_mode, pkey)))

    @app.get("/api/progress/<int:eid>")
    def api_progress(eid: int):
        # 前端推理期间轮询：{stage: load|infer|render3d|done, done, total}。空=尚未开始。
        return jsonify(store.get_prog(eid))

    @app.get("/api/mujoco/progress/<int:eid>")
    def api_mujoco_progress(eid: int):
        if not 0 <= eid < store.n_items():
            abort(404, "episode 越界")
        source = request.args.get("source", "pred")
        cam_mode = request.args.get("cam_mode", DEFAULT_CAM_MODE)
        try:
            hand_mode = _normalize_hand_mode(
                request.args.get("hand_mode", DEFAULT_HAND_MODE))
        except (TypeError, ValueError) as exc:
            abort(400, str(exc))
        if source not in {"gt", "pred"}:
            abort(400, f"未知 source: {source}")
        if cam_mode not in CAM_MODES:
            abort(400, "MuJoCo 推理参数无效")
        return jsonify(store.mujoco_progress(
            eid, source, cam_mode, hand_mode,
            request.args.get("betas", DEFAULT_PARAM_MODE) == "mean",
            request.args.get("fov", DEFAULT_PARAM_MODE) == "mean"))

    def _world_query_options() -> dict:
        layout = request.args.get("layout", DEFAULT_LAYOUT)
        coord_mode = request.args.get("coord_mode", "z_up")
        cam_mode = request.args.get("cam_mode", DEFAULT_CAM_MODE)
        try:
            hand_mode = _normalize_hand_mode(
                request.args.get("hand_mode", DEFAULT_HAND_MODE))
        except (TypeError, ValueError) as exc:
            abort(400, str(exc))
        if (layout not in LAYOUTS or coord_mode not in WORLD_COORD_MODES
                or cam_mode not in CAM_MODES):
            abort(400, "固定世界渲染参数无效")
        views = {}
        for view_name in ("vov", "vgt", "vpred"):
            view = {}
            for field in ("az", "el", "zoom", "panX", "panY"):
                value = request.args.get(f"{view_name}_{field}")
                if value is not None:
                    try:
                        view[field] = float(value)
                    except ValueError:
                        abort(400, "固定世界视角参数无效")
            views[view_name] = view
        return {
            "layout": layout,
            "views": views,
            "coord_mode": coord_mode,
            "show_traj": request.args.get("show_traj", "1") != "0",
            "show_cam_hand": request.args.get("show_cam_hand", "1") != "0",
            "cam_mode": cam_mode,
            "hand_mode": hand_mode,
            "gt_betas_mean": request.args.get("gt_betas", DEFAULT_PARAM_MODE) == "mean",
            "pred_betas_mean": request.args.get("pred_betas", DEFAULT_PARAM_MODE) == "mean",
            "pred_fov_mean": request.args.get("pred_fov", DEFAULT_PARAM_MODE) == "mean",
            "raw": bool(request.args.get("raw")),
        }

    @app.get("/api/world/progress/<int:eid>")
    def api_world_progress(eid: int):
        if not 0 <= eid < store.n_items():
            abort(404, "episode 越界")
        return jsonify(store.world_video_progress(eid, **_world_query_options()))

    @app.get("/world/<int:eid>")
    def world_video(eid: int):
        if not 0 <= eid < store.n_items():
            abort(404, "episode 越界")
        try:
            path = store.world_video(eid, **_world_query_options())
        except (OSError, RuntimeError, ValueError) as exc:
            abort(400, str(exc))
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.get("/mujoco/<int:eid>")
    def mujoco_video(eid: int):
        if not 0 <= eid < store.n_items():
            abort(404, "episode 越界")
        source = request.args.get("source", "pred")
        cam_mode = request.args.get("cam_mode", DEFAULT_CAM_MODE)
        try:
            hand_mode = _normalize_hand_mode(
                request.args.get("hand_mode", DEFAULT_HAND_MODE))
        except (TypeError, ValueError) as exc:
            abort(404, str(exc))
        if source not in {"gt", "pred"}:
            abort(404, f"未知 source: {source}")
        if cam_mode not in CAM_MODES:
            abort(404, "MuJoCo 推理参数无效")
        try:
            path = store.mujoco_video(
                eid, source, cam_mode, hand_mode,
                request.args.get("betas", DEFAULT_PARAM_MODE) == "mean",
                request.args.get("fov", DEFAULT_PARAM_MODE) == "mean")
        except (ImportError, ModuleNotFoundError) as exc:
            abort(503, f"MuJoCo 依赖不可用: {exc}")
        except (RuntimeError, ValueError) as exc:
            abort(400, str(exc))
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.get("/api/retarget/progress/<int:eid>")
    def api_retarget_progress(eid: int):
        if not 0 <= eid < store.n_items():
            abort(404, "episode 越界")
        source = request.args.get("source", "pred")
        cam_mode = request.args.get("cam_mode", DEFAULT_CAM_MODE)
        try:
            hand_mode = _normalize_hand_mode(
                request.args.get("hand_mode", DEFAULT_HAND_MODE))
        except (TypeError, ValueError) as exc:
            abort(400, str(exc))
        if source not in {"gt", "pred"}:
            abort(400, f"未知 source: {source}")
        if cam_mode not in CAM_MODES:
            abort(400, "Wuji retargeting 推理参数无效")
        return jsonify(store.retarget_progress(
            eid, source, cam_mode, hand_mode,
            request.args.get("betas", DEFAULT_PARAM_MODE) == "mean",
            request.args.get("fov", DEFAULT_PARAM_MODE) == "mean"))

    @app.get("/retarget/<int:eid>")
    def retarget_video(eid: int):
        if not 0 <= eid < store.n_items():
            abort(404, "episode 越界")
        source = request.args.get("source", "pred")
        cam_mode = request.args.get("cam_mode", DEFAULT_CAM_MODE)
        try:
            hand_mode = _normalize_hand_mode(
                request.args.get("hand_mode", DEFAULT_HAND_MODE))
        except (TypeError, ValueError) as exc:
            abort(404, str(exc))
        if source not in {"gt", "pred"}:
            abort(404, f"未知 source: {source}")
        if cam_mode not in CAM_MODES:
            abort(404, "Wuji retargeting 推理参数无效")
        try:
            path = store.retarget_video(
                eid, source, cam_mode, hand_mode,
                request.args.get("betas", DEFAULT_PARAM_MODE) == "mean",
                request.args.get("fov", DEFAULT_PARAM_MODE) == "mean")
        except (ImportError, ModuleNotFoundError) as exc:
            abort(503, f"Wuji retargeting 依赖不可用: {exc}")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            abort(400, str(exc))
        return send_file(path, mimetype="video/mp4", conditional=True)

    @app.post("/api/export/<int:eid>")
    def api_export_video(eid: int):
        if not 0 <= eid < store.n_items():
            abort(404, "episode 越界")
        body = request.get_json(force=True, silent=True) or {}
        sources = body.get("sources")
        if not isinstance(sources, list) or not sources or not all(
                isinstance(source, str) for source in sources):
            abort(400, "至少选择一路导出画面")
        mode = body.get("mode", store.default_mode)
        layout = body.get("layout", DEFAULT_LAYOUT)
        content = body.get("content", CONTENTS[0])
        cam_mode = body.get("cam_mode", DEFAULT_CAM_MODE)
        try:
            hand_mode = _normalize_hand_mode(
                body.get("hand_mode", DEFAULT_HAND_MODE))
        except (TypeError, ValueError) as exc:
            abort(400, str(exc))
        if mode not in MODES or layout not in LAYOUTS or content not in CONTENTS:
            abort(400, "导出渲染参数无效")
        if cam_mode not in CAM_MODES:
            abort(400, "导出推理参数无效")
        world_views = body.get("world_views")
        if world_views is not None and not isinstance(world_views, dict):
            abort(400, "固定世界视角参数无效")
        world_coord_mode = body.get("world_coord_mode", "z_up")
        if world_coord_mode not in WORLD_COORD_MODES:
            abort(400, "固定世界坐标系参数无效")
        show_traj = body.get("show_traj", True)
        show_cam_hand = body.get("show_cam_hand", True)
        if not isinstance(show_traj, bool) or not isinstance(show_cam_hand, bool):
            abort(400, "固定世界显示参数无效")
        output_format = body.get("output_format", "grid")
        if output_format not in {"grid", "separate"}:
            abort(400, "导出文件格式无效")
        source_tag = "_".join(str(source) for source in dict.fromkeys(sources))
        filename = (f"wuji_item_{eid:03d}_individual_videos.zip"
                    if output_format == "separate" else f"wuji_{source_tag}.mp4")
        token = uuid.uuid4().hex
        options = {
            "mode": mode,
            "layout": layout,
            "content": content,
            "cam_mode": cam_mode,
            "hand_mode": hand_mode,
            "gt_betas_mean": body.get("gt_betas") == "mean",
            "pred_betas_mean": body.get("pred_betas") == "mean",
            "pred_fov_mean": body.get("pred_fov") == "mean",
            "world_views": world_views,
            "world_coord_mode": world_coord_mode,
            "show_traj": show_traj,
            "show_cam_hand": show_cam_hand,
            "raw": bool(body.get("raw")),
            "output_format": output_format,
        }
        with export_lock:
            export_jobs[token] = {
                "ok": True, "stage": "queued", "progress": 0.0,
                "message": "导出任务已提交", "filename": filename,
                "download": None, "error": None,
            }
            if len(export_jobs) > 128:
                for old_token in list(export_jobs):
                    if old_token == token:
                        continue
                    if export_jobs[old_token].get("stage") in {"done", "error"}:
                        export_jobs.pop(old_token, None)
                        export_files.pop(old_token, None)
                    if len(export_jobs) <= 128:
                        break

        def update_progress(values: dict) -> None:
            with export_lock:
                job = export_jobs.get(token)
                if job is None:
                    return
                job.update(values)
                job["progress"] = max(0.0, min(1.0, float(job.get("progress") or 0.0)))

        def run_export() -> None:
            try:
                path = store.export_video(
                    eid, list(sources), **options, on_progress=update_progress)
                with export_lock:
                    export_files[token] = (Path(path), filename)
                    export_jobs[token].update(
                        stage="done", progress=1.0, message="导出完成",
                        download=f"export/{token}", error=None)
            except Exception as exc:  # noqa: BLE001 - background failures are reported by progress API
                with export_lock:
                    job = export_jobs.get(token)
                    if job is not None:
                        job.update(stage="error", message="导出失败",
                                   error=str(exc), download=None)

        threading.Thread(
            target=run_export, name=f"viewer-export-{token[:8]}", daemon=True,
        ).start()
        return jsonify(
            ok=True, token=token, progress=f"api/export/progress/{token}",
            filename=filename), 202

    @app.get("/api/export/progress/<token>")
    def api_export_progress(token: str):
        with export_lock:
            job = export_jobs.get(token)
            snapshot = dict(job) if job is not None else None
        if snapshot is None:
            abort(404, "导出任务不存在或已过期")
        return jsonify(**snapshot)

    @app.get("/export/<token>")
    def export_file(token: str):
        with export_lock:
            exported = export_files.get(token)
        if exported is None or not exported[0].is_file():
            abort(404, "导出视频不存在或已过期")
        path, filename = exported
        return send_file(
            path, as_attachment=True, download_name=filename, conditional=True)

    @app.get("/video/<int:eid>")
    def video(eid: int):
        if not 0 <= eid < store.n_items():
            abort(404, "episode 越界")
        mode = request.args.get("mode", store.default_mode)
        if mode not in MODES:
            abort(404, f"未知 mode: {mode}")
        layout = request.args.get("layout", DEFAULT_LAYOUT)
        if layout not in LAYOUTS:
            abort(404, f"未知 layout: {layout}")
        content = request.args.get("content", CONTENTS[0])
        if content not in CONTENTS:
            abort(404, f"未知 content: {content}")
        cam_mode = request.args.get("cam_mode", DEFAULT_CAM_MODE)
        if cam_mode not in CAM_MODES:
            abort(404, f"未知 cam_mode: {cam_mode}")
        try:
            hand_mode = _normalize_hand_mode(
                request.args.get("hand_mode", DEFAULT_HAND_MODE))
        except (TypeError, ValueError) as exc:
            abort(404, str(exc))
        gtb = request.args.get("gt_betas", DEFAULT_PARAM_MODE) == "mean"
        pdb = request.args.get("pred_betas", DEFAULT_PARAM_MODE) == "mean"
        pdf = request.args.get("pred_fov", DEFAULT_PARAM_MODE) == "mean"
        raw = bool(request.args.get("raw"))
        # 不再逐请求打印（/video 会因 Range/多块被调多次刷屏）；渲染进度由 [compare] 单行刷新给出。
        if raw:                          # 仅原始 GT：单画面 GT overlay，不跑推理（与推理模式无关）
            mp4 = store.mp4_gt(eid, mode)
        else:
            mp4 = store.mp4(eid, mode, layout, content, cam_mode, hand_mode, gtb, pdb, pdf)
        if not mp4.exists():
            abort(404)
        return send_file(mp4, mimetype="video/mp4", conditional=True)   # conditional=支持 Range seek

    return app
