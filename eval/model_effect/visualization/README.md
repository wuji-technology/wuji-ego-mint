# eval/model_effect/visualization — 模型无关的共用可视化

把训练好的学生模型（经 `--model` 选：lingbotmap / vggt / …）预测的**双手 MANO + 相机**，用预测值重投影回 ego 视频，与教师真值（lerobot 预算列）重投影对比，肉眼检验对齐程度；对没见过的裸视频则只出预测 overlay 看泛化。GT 与 Pred 两条渲染路径彼此独立，Pred 不使用 GT 手、存在性、相机外参或内参。

本包**只负责可视化**，与具体模型解耦：任意模型只要经 `inference` 引擎产出统一契约（见 `inference/base.py`），即可复用这里的离线 mp4（`hand_reproj.py`）与网页上位机（`viewer_web.py`）。

## model_effect 整体架构（四个平级包）
```
eval/model_effect/                 # 包根（不放 __init__.py，仅作 sys.path 根）
├── visualization/                 # ★本包：共用可视化（模型无关）
├── inference/                     # 共用推理：StudentEngine + 契约 base + 注册表 registry
├── predictors/                    # 多模型推理适配器：predictors/<model>/adapter.py（注册名 + 默认 config）
└── benchmark/                     # 指标评测（复用 inference 引擎 + visualization.reproj_core）
```
- 推理引擎 `inference.engine.StudentEngine` 走 `model_train.build_model`，**本身模型无关**（对 model_train 注册的学生都通用，靠 config 选）；`predictors/<model>` 只是「默认 config + 注册名」的薄适配器。
- 逐帧 loss `render/metrics.py` 直接复用 checkpoint 所属 run 的配置快照与训练 Criterion，按 `camera / hand_presence / mano_param / mano_joint / camera_mano_consistency` 展示启用项；refine hand head 的 initial 中间输出会随分窗结果一并拼接，缺少模型辅助输出的旧模型只跳过对应项而不会中断页面。camera translation normalization 从当前 episode 所属 LeRobot 根读取。

## 本包目录结构
```
visualization/
├── hand_reproj.py        ★离线入口：判别 lerobot/裸视频，左右并排对比 → compare.mp4（--model 选模型）
├── viewer_web.py         ★网页入口：端到端 2D GT vs Pred 视频 + 分解的 3D 诊断面板
├── reproj_core/          自包含底座（不依赖训练框架；GT 与预测共用，benchmark 也复用）
│   ├── geometry.py       6D↔旋转矩阵、左手翻转、世界点投影、相机 pose_enc 反解
│   ├── mano.py           smplx MANO 前向封装、J0_canon、faces、decode_hand_6d
│   └── lerobot_io.py     定位数据集 / 枚举 episode / 读预算列 + 按行号抽帧（帧↔标签对齐）
├── render/
│   ├── draw.py           重投影绘制（mesh / skeleton / mesh_skel）
│   ├── compare.py        GT/预测独立解算 + 端到端 2D 渲染（离线并排 render_compare_overlay）
│   ├── world.py          世界系 3D payload 构建（joints/traj/conn/valid/cam_R/cam_t，网页 canvas 用）
│   ├── fixed_world_video.py  固定世界 Canvas 的逐帧 H.264 导出渲染器
│   ├── mujoco_video.py   网页按需 MuJoCo 离屏渲染（固定第三视角 H.264 MP4）
│   ├── numbers.py        逐帧数值面板（相机/手位姿、betas、FoV）
│   └── metrics.py        逐帧 loss 复算（共用；复用训练框架 losses，与训练同切窗）
├── viewer/               网页后端子包（const/ckpts/netutil/store/routes/web）
└── tools/
    └── check_metrics_win_consistency.py   面板 loss 与训练同切窗一致性校验
```
> 包内跨模块用相对 import（`from ..reproj_core import ...`）；跨顶级包用绝对名（`from inference.registry import ...`）。入口脚本负责把包根 `model_effect/` 注入 `sys.path`。

## 用法（使用新项目的 mint 环境）
```bash
PY=python

# 1) lerobot held-out：GT vs 预测 对比（核心验证，有数值意义）
$PY eval/model_effect/visualization/hand_reproj.py \
    --input <lerobot_v3 目录> --ckpt <训练 step_* 目录> \
    --episode 0 --max-frames 64 --mode mesh_skel

# 2) 没见过的裸视频：仅预测 overlay（泛化定性）
$PY eval/model_effect/visualization/hand_reproj.py \
    --input <某视频.mp4> --ckpt <训练 step_* 目录> --fps 10 --max-frames 64

# 3) 管线 smoke（不给 --ckpt：骨干用 config 预训练、hand_head 随机，仅验证能跑通）
$PY eval/model_effect/visualization/hand_reproj.py \
    --input <lerobot_v3 目录> --episode 0 --max-frames 16
```
- `--model`：推理模型（`inference.registry` 注册名，默认 `lingbotmap`）。`vggt` 为占位——暂无默认 distill config，须显式 `--config` 否则报错。
- `--config`：默认 `None`，由所选模型适配器补默认（lingbotmap = `model_train/configs/volcano/lingbotmap_distill_frozon.yaml`，取 model 结构 + `size_hw`）。
- `--ckpt`：accelerate `save_state` 目录（`output/model_train/<ts>/step_*`）或权重文件；不给则依次使用最新训练权重、`checkpoints/model.safetensors`，都不存在才走 smoke。
- `--mode`：`mesh` / `skeleton` / `mesh_skel`（默认）。`--window` 分窗前向窗口。`--alpha` mesh 透明度。
- 输出：`output/eval/<model>/hand_reproj/<时间戳>/compare.mp4`（mp4v 写完转 H.264，VSCode/浏览器可直接播）。

## 网页端可视化（交互，推荐 held-out 逐 episode 排查）
`viewer_web.py` 起一个网页上位机。页面生成一个端到端 **2D GT vs Pred** 视频，并依次显示**固定世界·3D**与**当前相机·3D**；GT 图层全部使用 GT，Pred 图层全部使用模型预测。**MuJoCo·仿真**与 **Wuji Hand·Retargeting** 默认关闭，不请求对应后端，也不会创建 MuJoCo/EGL 渲染上下文；用户从顶栏或右侧模块栏手动开启后，才检查已有缓存或开始后台渲染。前者显示 MANO 世界系结果，后者把同一份模型 21 点输出驱动一代手 URDF/MJCF。有 GT 时两个方法各占一栏、栏内分别显示 `GT | PRED`；无 GT 裸视频时每栏只显示 PRED。

顶部 Checkpoint 支持直接输入绝对路径或项目相对路径，也可从 `/` 开始浏览任意服务器目录；可选择单个 `.safetensors/.bin/.pt/.pth/.ckpt` 文件，或选择直接包含这些权重的 checkpoint 目录。

| 对比内容 | 2D 重投影 | 3D | 用途 |
|---|---|---|---|
| **整体** | GT手×GT相机 vs Pred手×Pred相机 | 固定世界：整体运动；当前相机：逐帧相对几何 | 端到端最终效果 + 世界系与相机系交叉核对 |

- **布局**：`叠加`（默认）把 GT/Pred 画在同一处（2D GT绿/Pred红；3D GT实线/Pred虚线，左手青右手金）；右上角分两行标注 GT/Pred 的 `L/R` 是否存在。`并排`回到左右两块，并在两个 panel 右上角分别标注各自状态。
- **相机推理**：交互推理和批量推理均默认选择“最大窗分窗 · Max Chunked”，以当前显卡 exact-full 安全上限作为窗长；仍可手动切换训练窗长分窗、流式或整段模式。
- 唯一的 2D 视频负责播放/拖动，并驱动固定世界、当前相机、手动开启的 MuJoCo/Retargeting 与逐帧 loss 面板。
- **固定世界·3D**：以各路首帧相机的完整位姿一次性建立共同显示坐标系（首帧相机位置为原点、朝向为世界轴），之后视图不再随相机旋转。默认视角是 `az=-0.61 / el=0.31`（约 -35°/18° 的 3/4 俯视，正视图读不出纵深），网页 `app.js` 的 `VIEW_AZ0/VIEW_EL0` 与 `render/fixed_world_video.py` 的 `DEFAULT_AZ/DEFAULT_EL` 必须一致（有 contract 测试盯着）。画面带**网格参考地面**：固定世界系没有天然重力轴，「上」取全段相机 `+Y(下)` 平均值的反向；地面过「沿上方向的最低点再下移 `max(15cm, 0.28×场景半径)`」，格距取 `1/2/5×10^k cm` 的整齐值（约 10 格覆盖 2×半径，格子本身即尺度参照），每 5 格加亮一条、离中心越远越淡成圆盘。手/相机会在地面上投出落影并画到地面的虚线垂线；轨迹改为随时间渐亮的尾迹（GT 实线、Pred 长虚线），骨架是深色描边 + 本色芯 + 圆头端点，背景为径向暗角，左上标题卡 / 左下图例与帧号 / 右下比例尺。导出 mp4 内部按 2× 超采样再 `INTER_AREA` 缩回，线条更锐（约 16 ms/帧）。世界轴与当前相机姿态轴全部使用低透明实线，三轴仍随真实相机旋转：GT 使用 `X红/Y绿/Z蓝`，Pred 使用明显不同的 `X橙/Y黄/Z紫`，但不会压过手部和轨迹。该窗口专属的「轨迹 / 相机姿态↔手」开关位于标题旁；青/金实线连接相机与左右手腕并标出真实三维距离（cm）。固定世界视图使用左键拖动平移、`Ctrl+左键`拖动旋转、滚轮缩放。
- **当前相机·3D**：每帧分别按 `X_cam = R_w2c · (X_world - C_world)` 把 GT 手变到 GT 相机系、Pred 手变到 Pred 相机系，再在统一的 OpenCV 轴约定（X 向右、Y 向下、Z 向前，光心为原点）下叠加或并排显示。该面板紧跟固定世界，默认只画当前帧双手骨架、相机原点三轴和比例尺，不连接跨帧世界轨迹；用于直接核对手相对 ego 相机的方向、深度与尺度。支持独立旋转、缩放、平移和当前帧 PNG 保存。
- **MuJoCo·仿真**：默认关闭；用户手动开启面板后，有 GT 时分别请求 GT 与 PRED 两路服务端离屏视频并在同一方法栏内并排比较，无 GT 时只显示 PRED；已有缓存则直接加载。Wuji Hand 面板采用相同的 GT/PRED 组织方式，两个方法不再互相并排。MuJoCo 与 Wuji Hand 共用由 Wuji 观察方向定义的同一台固定第三人称相机，取景范围只由完整双手运动决定，不把源相机轨迹计入包围盒。为了提高长视频渲染速度，这两个面板不再逐帧提交左腕/右腕/相机连续轨迹，只保留一个低透明度起点相机框和一个随当前帧移动的深色实时框；二者使用无光照屏幕线，不在地面投下框形阴影。完整轨迹仍在「固定世界·3D」面板中可视化。地面统一放在最低关节点下约 `0.08 m`，所有机器人视频都跟随「整体·2D」播放。
- **Wuji Hand·Retargeting**：默认关闭，用户手动开启面板后才启动解算与渲染。它使用模型的 21 点经 adaptive analytical IK 驱动 20-DoF Wuji Hand，左右手合并在同一个 MuJoCo 场景中原生处理遮挡。视角、取景大小和地面高度全部复用 MuJoCo 面板的公共实现；右上角不重复显示仅属于 2D 视频的左右手状态。运行环境需安装 `mujoco>=3.6`、`pin==3.8.0` 与 `nlopt`。

  ```bash
  $PY -m pip install 'mujoco>=3.6' 'pin==3.8.0' nlopt \
      'cmeel-urdfdom>=4,<5' 'cmeel-tinyxml2>=10,<11'
  ```
- **保存当前帧**：「保存当前帧」可从整体 2D、固定世界 3D、当前相机 3D、MuJoCo 或 Wuji Hand 中选择一个当前画面并下载 PNG。固定世界 3D 默认采用更常见的 Z-up 世界系（X 向右、Y 向前、Z 向上），也可在面板中切回 OpenCV 首帧相机系（X 向右、Y 向下、Z 向前）；坐标系切换会同步影响 Canvas、截图和视频导出。截图直接取网页当前 Canvas，因此保留当前播放帧、GT/Pred 叠加或并排布局、旋转、缩放、平移以及「轨迹 / 相机姿态↔手」开关状态；并排模式会按页面当时的横排或竖排布局合成一张图。视频类面板直接抓取各自当前已解码帧。文件名包含样本、画面类型和零填充帧号。
- **导出视频**：「导出」按钮与「开始推理」相邻，带 GT 的 LeRobot 对比默认选择原视频渲染、固定世界·3D、MuJoCo 和 Wuji Hand Retargeting 四类画面。MuJoCo 与 Retargeting 面板无需手动打开：导出任务会直接在服务端生成缺失视频。有 GT 时最终视频展开为两排八格：第一排为整体 2D `GT | PRED`、固定世界 3D `GT | PRED`，第二排为 MuJoCo `GT | PRED`、Wuji Retargeting `GT | PRED`。每一格都直接使用原视频宽高，不再固定为 `960×540`；例如 HOT3D 的 `512×512` 方形视频会得到八个相同的 `512×512` 方格，总分辨率 `2048×1024`。无 GT 时每类保持单个预测画面。固定世界由服务端按主视频的 `0..T-1` 帧逐帧复刻 Canvas 动画，保留当前视角与「轨迹 / 相机姿态↔手」开关；导出坐标轴使用低透明实线箭头并在正方向标注 `+X / +Y / +Z`。世界轴和完整轨迹只预渲一次，每帧只更新动态骨架与相机姿态。独立画面仍由两个受限 EGL 上下文并发槽处理。Retargeting 的左右手时序 IK 各自保持 warm-start/filter，但两条序列并行解算，随后在单个合并场景中渲染；渲染帧与 ffmpeg H.264 写入使用有界队列流水重叠。编码默认 `superfast`（`VIEWER_ROBOT_VIDEO_PRESET` 可覆盖），`VIEWER_ROBOT_ENCODE_BUFFER_FRAMES` 控制编码流水缓冲帧数。页面进度条按每一路的实际渲染帧数显示百分比和 `done/total`，最终显示合成阶段，完成后自动下载。过程不依赖浏览器播放或 `MediaRecorder`，暂停、拖动和倍速不会影响导出帧序列。
```bash
$PY eval/model_effect/visualization/viewer_web.py \
    --input <lerobot_v3 目录> --ckpt <训练 step_* 目录> --port 8000
```
- 点击「加载模型」时默认使用**当前进程可见的全部 GPU**，在每张卡常驻一个相同模型副本；
  `chunked` 推理把独立窗口并发分给这些卡，输出回 CPU 后仍按原顺序做相机拼接和手部融合，不改变评估口径。
  页面「模型就绪」状态会显示实际加载的设备。`streaming` 相机有跨帧 KV cache，仍只在主卡执行。
- 手部窗口模式有三种：`hard` 保留原始硬切，`blend` 使用最多 8 帧重叠做线性渐入/渐出，
  `smooth`（默认，页面显示「融合 + UKF平滑」）先执行同一套 `blend`，再复用
  `ego_pipeline/data_cleaning/cleaning_modules/ukf_cam_smoothing.py` 最终生产参数的相机系
  速度自适应 UKF + 无迹 RTS 双向平滑。生产兼容默认仍为 `q=0.6, r=0.6, beta=2.0, rts=1`；
  Viewer 默认采用较弱的 `q=0.7, r=0.5, beta=0.3`，并在「手部拼窗」下允许输入三项参数。
  建议范围为 `q=0.4–1.0, r=0.3–1.0, beta=0.2–3.0`（安全范围分别为 `0.1–2.0, 0.1–2.0, 0–5`）：
  `q` 越大越跟手、平滑越弱，`r` / `beta` 越大则平滑越强。参数会进入预测与渲染缓存键。
  UKF 处理
  `transl_cam`、腕部 quaternion、MANO pose rotvec 和 betas，最后把旋转转回模型的 6D 表示。
  模型输出本来就是相机系，因此这里不重复生产模块的 world→camera→world 外壳；只改写存在性有效帧，
  不改变 `hand_presence` 掩码。本后处理只移植生产清洗链的最终 `ukf_cam`，不会额外执行前置的
  `block_hampel` 离群剔除或 `slerp` 缺帧填充。
- 相机推理选择 `full` 时，会把安全帧数上限内的整段视频送入主卡做**一次普通模型前向**；相机、手部和存在性
  输出都不分窗、不拼接。默认上限按本次所选 GPU 中的最小显存保守取训练窗长的 1/2/3 倍：小于 40 GiB、40–80 GiB、
  不低于 80 GiB 分别对应 32/64/96 帧（训练窗为 32 时），也可用 `--full-max-frames` 覆盖且须为训练窗整数倍。
  超限输入不会启动 GPU 前向；交互页面会询问是否切到 `max_chunked`，以该上限为窗长分窗推理和拼接。
  `full` 允许测试 32 帧训练长度以外的外推效果，但全局 attention 的计算和显存开销会
  随帧数快速增长，其中计算时间近似按帧数平方增长，因此只适合作为短片/对照实验模式，不推荐批量处理长视频；
  它不会使用其余模型副本，也不会在 CUDA OOM 后自动回退到 `chunked`。单次 GPU 前向执行中无法即时停止，
  只能在这次前向开始前或完成后响应取消。由于没有手部拼窗，`hard` / `blend` 在该模式下等价并共享预测缓存；
  `smooth` 仍会对整段输出执行 UKF+RTS 后处理，因此使用独立预测缓存。
- `full` 会关闭普通 batch 前向中不会被后续帧复用的持久 SDPA K/V cache，并在调用结束后恢复缓存结构；这只
  释放冗余缓存，不裁剪 attention，也不影响 `streaming` 所需的跨调用 KV cache。Viewer 启动时还会默认启用
  CUDA expandable segments 来减少长序列临时张量造成的 allocator 碎片。
- 鼠标悬停「数据集分析」后可单独选择「视频分析」或「内参分析」：前者只递归读取 ffprobe 媒体
  属性，不再默认扫描 FOV；后者自动识别 LeRobot 根，并从 `data/**/*.parquet` 的 `cam_fov`
  （兼容旧列 `fov`）为每个 episode 只采样 `frame_index=0` 的首帧。页面展示垂直/水平 FOV、由
  数据集分辨率反解的 `fx/fy`、归一化焦距和 `fx/fy` 比值分布，并诊断首帧非法 FOV、极端视场角
  和非方形像素迹象。当前标注不含主点、skew 或畸变系数，因此页面明确把 `cx=W/2, cy=H/2`
  与 `skew=0` 标为假设，并将畸变显示为不可用。
- 「批量推理」可单独选择模型 Run / Step 并递归扫描一个服务器目录，仅在点击「开始批量推理」后自动
  切换、加载所选模型并执行。每个视频的分窗前向会使用该模型已加载的全部 GPU；某视频推理完成后立即
  进入后台渲染池，GPU 同时开始下一个视频，避免被 MP4 渲染阻塞。`-j/--jobs` 控制并发渲染/写盘数
  （默认 2，队列有界以限制内存）；结果按输入相对目录镜像写入可选输出根，并生成含完整 checkpoint
  路径的 `batch_manifest.json`。文件名支持
  `{stem}`、`{name}`、`{ext}`、`{parent}`、`{index:04d}` 等模板；
  输入根可手填、取当前 Viewer 浏览目录或用面板内的服务器目录选择器逐级选取；输出根必须位于输入根之外，
  避免生成的 MP4 在下次批次中被再次扫描。
- Benchmark 的「运行位置」可切换为 `Aliyun 远程`。节点数默认 2，每节点默认使用全部 8 卡（共 16 个
  全局 shard）；节点数和全部 PAI-DLC 参数均可在页面修改。远程容器通过 CPFS 共享请求、进度和结果，
  主节点等待所有节点完成后复用 `benchmark/dist/aggregate.py` 聚合。页面持续展示 DLC JobId/状态并支持
  停止作业；默认配置、CLI 和输出目录说明见 `benchmark/dist/aliyun/README.md`。
- 「Benchmark 测评」可连续添加多个 Run / Step；服务端按队列逐个模型执行，让每个模型独占同一组
  已选 GPU，并在每个模型完成后把相同 head / dataset / metric 汇入横向对比表。所有模型使用完全相同的
  功能、数据集与序列范围。默认开启「自动优选最佳模型并追加 UKF 融合测评」：原始模型全部完成后，
  对每个可比质量指标在模型间做 0–1 归一化（高优指标反向），平均归一化损失最低的 checkpoint 再以
  `smooth`（线性融合 + 生产参数 UKF + RTS）独立复测同一协议。计数、尺度系数与诊断相机分支不参与选优；
  单模型时直接选择该模型。UKF 结果作为独立 M 行、独立报告和独立缓存记录，可在运行前取消勾选。
  范围按每个数据集的稳定输入顺序采用左闭右开 `[start, end)`：`0–50`
  表示前 50 条，`20–53` 表示索引 20 到 52；范围会先于多卡分片应用，HOT3D 左右手等共享同一视频的
  多条 GT 不会被拆开。面板内「抽查预测可视化」可按数据集筛选并任选序列，直接复用 Viewer 的 2D
  overlay 与 3D 联动。公开 Benchmark GT 与 LeRobot 渲染契约不同，因此抽查区明确只展示预测，不伪造
  GT 对比；量化 GT 仍以 Benchmark 指标表为准。运行时每张 GPU 保持一条固定状态行，完成任一
  head/序列后立即更新「实时评测结果」中的滚动均值；详细日志默认折叠且只保留关键行和最新状态槽。
- HOT3D 量化优先读取 `output/scripts/data_processed/pipeline_lerobot/build_train_lerobot/hot3d/lerobot_v3`
  的 90 条去畸变 LeRobot 产物（可用 `HOT3D_LEROBOT_ROOT` 覆盖），直接复用其中的相机轨迹和双手
  真值，不再要求旧版 `anno.pth/ego_extrinsics.pkl/head_pose.pkl` 或 `joblib`。
- `CUDA_VISIBLE_DEVICES` 可限制网页进程可见的 GPU；`--devices 0,1` 可进一步指定当前进程内的逻辑卡，
  `--devices cpu` 可强制 CPU。模型加载后会固定使用当时选定的全部设备，修改可见卡范围后需重启网页进程。
- 数据加载会并行读取 episode Parquet 和解码 H.264；Decord 默认 8 解码线程、每批 96 帧，可用
  `VIEWER_DECORD_THREADS` / `VIEWER_DECODE_CHUNK` 按机器调整。当前 1408×1408 HOT3D 实测 Decord
  明显快于本机 ffmpeg/NVDEC，因此保持逐帧完全对齐的 Decord 路径。
- 3D/loss 阶段复用常驻 MANO 层并批量计算逐帧 loss；2D 视频直接复用 3D 阶段已解算的 GT/Pred
  世界系网格，不再重复跑 MANO。逐帧 OpenCV 绘制采用有界并行、仍按帧号顺序送入 ffmpeg；默认
  `VIEWER_RENDER_WORKERS=2`（本机实测甜点），可用 `VIEWER_RENDER_INFLIGHT` 调整待写帧上限。
- 预测（GPU 前向）与端到端 2D 视频按 episode 缓存到 `output/eval/<model>/viewer/<scene>/<ts>/`，首次访问某 (episode, mode, layout) 组合才渲染；`--prerender -j N` 启动后台批量预渲。
- `meta/episodes` 首次枚举只用外层 4 线程读取小 Parquet，并将7列清单缓存到
  `output/cache/viewer_episode_index/`；相同 export 再次打开只读单个索引文件。可用
  `VIEWER_EPISODE_READ_WORKERS` 和 `VIEWER_EPISODE_STAT_WORKERS` 针对其他存储重新调并发，
  `VIEWER_EPISODE_CACHE_DIR` 可改缓存目录。
- LeRobot 条目支持 GT vs 预测；普通视频和 Benchmark 序列使用仅预测模式。`--max-frames` 控每条输入帧数（显存/耗时）。

## 注意
- lerobot 输入须为 `build_train_lerobot.py` 的训练就绪 export：相机需 `cam_*`，手部 GT 可为
  `*_mano_*` 参数列或仅 `{left,right}_kpt21`。kp21-only 数据在 2D/3D 中显示骨架；选择 `mesh`
  时 GT 自动退化为骨架，预测侧仍可显示模型 MANO 网格。
- 长 episode 建议设 `--max-frames` 控制显存/耗时。
- 相机 K 用显示帧分辨率解码（FoV 与分辨率无关），GT/预测一致自洽。
- GT 网格/骨架按逐帧 `hand_kept` 绘制；预测侧直接按 `hand_presence_logits>=0` 逐手绘制，右上角同步显示 Pred `L/R`。旧缓存只有 `hand_confidence` 时兼容 `>=0.5`，旧 checkpoint 没有存在性头时显示 `?` 并兼容为两手都画。
