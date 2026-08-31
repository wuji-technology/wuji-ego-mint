# 视频预处理与过滤

EgoPipeline 正式各级（GeoCalib → MoGe-2 → MegaSaM → HaWoR）之前的两步：把来源各异的原始
视频**规格化**成统一的 fps / 分辨率 / 时长，再按**画面里有没有手**把它切成可用片段。
`README` 中「过滤与轨迹清理后保留 1,021.514 小时」里的「过滤」，第一道就是这里。

两个互相独立的框架，各自有 `main.py` 入口和一套 op：

| 目录 | 作用 | op |
| --- | --- | --- |
| `reprocessed/` | 规格化：降帧率、降分辨率、按时长切段 | `fps`、`resolution`、`time_segment` |
| `filtered/` | 按手部检测结果过滤并切片 | `hand_filter` |

`script_utils/` 是两者共用的资源监控。

## 过滤规则

判定在 `filtered/ops/hand_filter.py` 的 `_find_keep_segments()`：

```python
cut_mask = ~any_hand                                  # ① 没有手 → 切
if filter_multi_hands:
    cut_mask |= padded_hand_counts > 2                # ② 超过两只手 → 切
if require_lr_pair:
    valid = (nl <= 1) & (nr <= 1) & ((nl + nr) >= 1)  # ③ 左≤1、右≤1、至少一只
    cut_mask |= ~valid                                #    同侧重复（两只左手）也切
```

即**没有手、或超过两只手，就切掉**。`--require_lr_pair` 会更严一档：除了总数，还检查
左右各不超过一只，能挡住「检测出两只左手」这类同侧重复。单手区间始终保留。

切除和保留都有最短长度：一段无手区间必须持续 `--min_absent_sec` 以上才真的切除（避免
被逐帧的检测抖动切碎），切完剩下的片段短于 `--min_segment_sec` 则整段丢弃。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--det_fps` | `5.0` | 抽帧检测帧率，不是原视频帧率 |
| `--det_thresh` | `0.2` | YOLO 手部检测置信度，越低越少切 |
| `--min_absent_sec` | `0.5` | 无手区间至少这么长才切除 |
| `--min_segment_sec` | `2.0` | 保留段最短时长，短于此丢弃 |
| `--filter_multi_hands` | 关 | 打开后「超过两只手」也切；默认只切无手区间 |
| `--require_lr_pair` | 关 | 打开后要求左≤1、右≤1 且至少一只 |
| `--imgsz` | `640` | YOLO 推理边长 |
| `--gpus` | `0,1,2,3` | 逗号分隔卡号，或 `auto` |
| `--profile` | `aggressive` | `balanced` 批大小 32，`aggressive` 64 |

切片用 `ffmpeg -c:v copy`，**不重编码**，所以这一步不损失画质，也不改变编码参数。

## 规格化参数

`reprocessed/` 各 op 的默认值与生产实际用值不同 —— 生产上是在命令行显式指定的：

| op | 参数 | 默认 | 生产用值 |
| --- | --- | --- | --- |
| `fps` | `--max_fps` | `30.0` | `30` |
| `resolution` | `--max_width` / `--max_height` | `512` / `384` | `1280` / `720` |
| | `--scale_mode` | `fit` | `fit` |
| `time_segment` | `--duration` | `60` | `60` |
| | `--min_tail` | `1.0` | — |
| 公共 | `--codec` / `--crf` / `--preset` | 无 | `h264` / `18` / `ultrafast` |

多个 op 可以用 `--ops a+b+c` 组合成**一次** ffmpeg 调用（1 decode + 1 encode），滤镜链由各
op 的 `vf_chain()` 拼接，不落中间产物。

## 用法

```bash
# 规格化：降到 30 fps、1280×720，按 60 秒切段
python ego_pipeline/preprocessing/reprocessed/main.py \
    --ops fps+resolution+time_segment \
    --input  /path/to/raw_videos \
    --output /path/to/normalized \
    --max_fps 30 --max_width 1280 --max_height 720 \
    --duration 60 --codec h264 --crf 18 --preset ultrafast \
    --workers 64

# 过滤：切掉无手与多手区间
python ego_pipeline/preprocessing/filtered/main.py \
    --ops hand_filter \
    --input  /path/to/normalized \
    --output /path/to/filtered \
    --filter_multi_hands --require_lr_pair \
    --gpus 0,1,2,3
```

两个入口都支持 `--list` 查看可用 op、`--dry_run` 只打印计划。产物按源目录结构镜像输出，
`hand_filter` 每个视频写成 `<stem>/<stem>_partNNN_<start>s_<end>s.mp4` 加一个完成标记；
完成状态记在输出根的 `.completed.jsonl`，重跑会跳过已完成项。

## 依赖

`reprocessed/` 只要 `ffmpeg`。

`filtered/hand_filter` 另外需要 `torch`、`ultralytics`，以及 **HaWoR** 提供的检测权重与
`lib.pipeline.tools._DetAdaptor`：

| 资产 | 路径 |
| --- | --- |
| HaWoR 检测器权重 | `model/hawor/detector.pt` |
| HaWoR 源码 | `third_party/HaWoR/` |

HaWoR 受 CC BY-NC-ND 约束，不随本仓库分发，需按其许可条款自行获取，详见
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) 与
[模型训练与可选管线复现](../../README_ZH.md#️-模型训练与可选管线复现)。

## 未随本仓库发布的 op

上游还有几个 op 不在这里：

- `action_segment`（按动作切段）与 `label_rule_filter`（标签规则过滤）依赖 VLM 打标产物，
  属于打标流程而非视频预处理
- `fisheye_undistort`（鱼眼去畸变）在本仓库已由
  [`ego_pipeline/backends/undistort_gpu.py`](../backends/undistort_gpu.py) 提供 GPU 实现
