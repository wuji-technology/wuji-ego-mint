(() => {
  'use strict';

  const STORAGE_KEY = 'wuji-viewer-language';
  const CJK_RE = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;
  const CJK_RUN_RE = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+/g;
  const TRANSLATABLE_ATTRIBUTES = ['title', 'placeholder', 'aria-label', 'aria-description', 'alt'];
  const SKIP_TAGS = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT']);

  // Long, user-facing phrases come first; the word dictionary below handles dynamic combinations.
  const PHRASES = new Map([
    ['待操作', 'Ready'],
    ['选择输入与参数后，点击“开始推理”运行模型。', 'Choose an input and settings, then click Start Inference.'],
    ['当前进行到哪一步', 'Current workflow step'],
    ['加载进度（推理为确定进度，其余为不确定动画）', 'Loading progress (inference is determinate; other tasks are indeterminate)'],
    ['模型与样本', 'Model and Sample'],
    ['输入 checkpoint 文件或目录路径', 'Enter a checkpoint file or directory path'],
    ['打开输入路径所在目录', 'Open the directory containing this path'],
    ['打开路径', 'Open Path'],
    ['选择该 checkpoint；目录内须含模型权重', 'Select this checkpoint; a directory must contain model weights'],
    ['当前浏览目录，点某层可回退', 'Current directory; click a level to go back'],
    ['可浏览任意服务器目录；点权重文件直接选择，或进入含权重的目录后选择当前目录', 'Browse any server directory. Select a weight file directly, or enter its directory and select the directory.'],
    ['选择', 'Select'],
    ['上一个', 'Previous'],
    ['下一个', 'Next'],
    ['枚举 episode 进度', 'Episode discovery progress'],
    ['推理策略', 'Inference Strategy'],
    ['叠加模式', 'Overlay Mode'],
    ['网格 Mesh', 'Mesh'],
    ['骨架 Skeleton', 'Skeleton'],
    ['网格 + 骨架', 'Mesh + Skeleton'],
    ['相机推理', 'Camera Inference'],
    ['分窗=训练窗长；最大窗分窗=当前显卡 exact full 安全上限；流式=相机整段因果+KV cache；整段=上限内一次普通前向，超限时会先询问是否改用最大窗分窗', 'Chunked uses the training window; Max Chunked uses the current GPU safe exact-full limit; Streaming runs causal full-sequence inference with a KV cache; Full uses one forward pass within the limit and asks before falling back when exceeded.'],
    ['分窗 · Chunked', 'Chunked'],
    ['最大窗分窗 · Max Chunked', 'Max Chunked'],
    ['流式 · Streaming', 'Streaming'],
    ['整段 · Full', 'Full Sequence'],
    ['手部拼窗', 'Hand Window Merge'],
    ['切换只更新设置，不会自动推理；点击开始推理后生效', 'Changing this only updates the setting. Click Start Inference to apply it.'],
    ['原始硬切', 'Hard Cut'],
    ['线性融合', 'Linear Blend'],
    ['融合 + UKF平滑', 'Blend + UKF Smoothing'],
    ['线性融合后，使用 ego_pipeline 数据清理链的相机系 UKF+RTS 双向平滑', 'Apply camera-space UKF + RTS bidirectional smoothing from the ego_pipeline cleaning chain after linear blending.'],
    ['默认：较轻平滑', 'Default: lighter smoothing'],
    ['建议 q 0.4–1.0、r 0.3–1.0、beta 0.2–3.0；q ↑ 更跟手，r/beta ↑ 更平滑。安全范围 q/r 0.1–2.0、beta 0–5。', 'Suggested: q 0.4–1.0, r 0.3–1.0, beta 0.2–3.0. Higher q follows motion more closely; higher r/beta smooths more. Safe limits: q/r 0.1–2.0, beta 0–5.'],
    ['过程噪声；越大越允许快速运动，平滑越弱', 'Process noise: higher values allow faster motion and reduce smoothing.'],
    ['观测噪声；越大越不跟随逐帧预测，平滑越强', 'Observation noise: higher values trust per-frame predictions less and increase smoothing.'],
    ['快速运动的自适应平滑；越大越抑制快速运动抖动', 'Adaptive smoothing during fast motion: higher values suppress fast-motion jitter more.'],
    ['几何参数', 'Geometry Settings'],
    ['GT 手形', 'GT Hand Shape'],
    ['预测手形', 'Predicted Hand Shape'],
    ['预测内参', 'Predicted Intrinsics'],
    ['只更新设置，点击开始推理后生效', 'Only updates the setting; click Start Inference to apply it.'],
    ['只更新设置，点击开始推理后生效；2D、MuJoCo 与 Retargeting 共用', 'Only updates the setting; click Start Inference to apply it. Shared by 2D, MuJoCo, and Retargeting.'],
    ['每帧', 'Per Frame'],
    ['整段平均', 'Sequence Mean'],
    ['仅看原始 GT', 'Raw GT Only'],
    ['只看原始标注，不跑模型推理；再次点击返回已有 GT 对照', 'Show raw annotations without model inference; click again to restore the existing GT comparison.'],
    ['MuJoCo 仿真', 'MuJoCo Simulation'],
    ['默认关闭；点击后才按当前样本启动 MuJoCo 渲染', 'Off by default. Click to start MuJoCo rendering for the current sample.'],
    ['默认关闭；点击后才将当前 21 点驱动 Wuji Hand 并启动渲染', 'Off by default. Click to drive Wuji Hand from the current 21 joints and start rendering.'],
    ['数据集分析', 'Dataset Analysis'],
    ['视频分析', 'Video Analysis'],
    ['ffprobe 媒体属性与质量', 'ffprobe media properties and quality'],
    ['内参分析', 'Intrinsics Analysis'],
    ['LeRobot episode 首帧 FOV', 'First-frame FOV for each LeRobot episode'],
    ['多样性分析', 'Diversity Analysis'],
    ['场景、动作、活动范围与手监督', 'Scenes, actions, motion range, and hand supervision'],
    ['批量推理', 'Batch Inference'],
    ['递归推理指定目录的全部视频，并按镜像目录保存结果', 'Run inference recursively on all videos in a directory and save results using a mirrored layout.'],
    ['对当前 ckpt 跑量化评测', 'Run quantitative evaluation on the current checkpoint'],
    ['代码对比', 'Code Comparison'],
    ['选择两个 run 和代码模块，只展示真实变更', 'Select two runs and a code module to show only actual changes.'],
    ['在每张可见 GPU 上常驻一个当前 ckpt 模型副本；须加载完成后才能推理', 'Keep one copy of the current checkpoint on each visible GPU; inference is available after loading finishes.'],
    ['加载模型', 'Load Model'],
    ['只有点击此按钮才会运行模型；选择输入和参数不会自动推理', 'The model runs only when this button is clicked; selecting inputs and settings does not start inference.'],
    ['▶ 开始推理', '▶ Start Inference'],
    ['保存当前帧 ▾', 'Save Current Frame ▾'],
    ['当前帧保存来源', 'Current frame source'],
    ['整体 · 2D', 'Combined · 2D'],
    ['固定世界 · 3D', 'Fixed World · 3D'],
    ['当前相机 · 3D', 'Current Camera · 3D'],
    ['MuJoCo · 仿真', 'MuJoCo · Simulation'],
    ['默认选择四类画面；有 GT 时展开为两排八格，单格跟随原视频尺寸', 'Four view categories are selected by default. With ground truth, export expands to two rows and eight tiles, each matching the source video size.'],
    ['导出选中 4 路', 'Export 4 Selected'],
    ['导出范围', 'Export Scope'],
    ['导出范围 · 选择画面', 'Export Scope · Select Views'],
    ['原视频渲染', 'Source Video Render'],
    ['固定世界 · 3D（跟随视频）', 'Fixed World · 3D (follows video)'],
    ['Wuji Hand · Retargeting', 'Wuji Hand · Retargeting'],
    ['有 GT 时两排八格：2D、3D、MuJoCo、Wuji 均为 GT/PRED', 'With ground truth, export uses eight tiles across two rows: GT/PRED for 2D, 3D, MuJoCo, and Wuji.'],
    ['加载样本后可导出', 'Load a sample to enable export.'],
    ['导出时由服务端渲染并缓存', 'The server will render and cache this view during export.'],
    ['所选画面当前不可导出：', 'The selected views are not currently exportable: '],
    ['导出进度', 'Export progress'],
    ['0% · 等待导出', '0% · Waiting to Export'],
    ['逐帧 loss（与训练同公式，随当前帧联动）。权重=组权重×项权重；值/加权/占比同口径（旋转项按弧度，值列括号补度数便于观看）；加权=权重×值；占比=加权值÷加权总 loss，看谁主导。', 'Per-frame loss using the training formula and synchronized to the current frame. Weight = group weight × item weight; value, weighted value, and share use the same convention. Rotation values use radians with degrees shown in parentheses. Weighted value = weight × value; share = weighted value ÷ total weighted loss.'],
    ['只读 ffprobe 媒体属性，不解码视频帧', 'Read-only ffprobe media properties; video frames are not decoded'],
    ['收起（后台分析继续）', 'Collapse (analysis continues in the background)'],
    ['数据集目录', 'Dataset Directory'],
    ['服务器绝对路径', 'Absolute server path'],
    ['浏览选择', 'Browse'],
    ['使用当前浏览目录', 'Use Current Directory'],
    ['ffprobe 并发', 'ffprobe Workers'],
    ['忽略缓存，重新分析', 'Ignore cache and analyze again'],
    ['数据集目录（可多选）', 'Dataset Directories (multiple allowed)'],
    ['每条路径自动识别数据集；默认已加入三套 LeRobot 数据', 'Datasets are detected automatically for each path; three LeRobot datasets are included by default.'],
    ['＋ 浏览添加', '+ Browse and Add'],
    ['＋ 添加当前浏览目录', '+ Add Current Directory'],
    ['输入服务器绝对路径', 'Enter an absolute server path'],
    ['添加路径', 'Add Path'],
    ['可添加单个 lerobot_v3，也可添加包含多套数据的上级目录；重复数据会自动合并。', 'Add one lerobot_v3 directory or a parent containing multiple datasets. Duplicate data is merged automatically.'],
    ['轨迹采样', 'Trajectory Sampling'],
    ['快速 · 12 文件/集', 'Fast · 12 Files/Dataset'],
    ['标准 · 24 文件/集', 'Standard · 24 Files/Dataset'],
    ['精细 · 48 文件/集', 'Detailed · 48 Files/Dataset'],
    ['← 上一级', '← Parent Directory'],
    ['✓ 选择此目录', '✓ Select This Directory'],
    ['关闭目录选择器', 'Close directory picker'],
    ['▶ 开始视频分析', '▶ Start Video Analysis'],
    ['■ 取消', '■ Cancel'],
    ['待分析', 'Ready to Analyze'],
    ['摘要 TXT', 'Summary TXT'],
    ['构成概览', 'Composition Overview'],
    ['柱长表示该分组的视频数量占比', 'Bar length represents the share of videos in each group'],
    ['来源目录', 'Source Directories'],
    ['按一级子目录汇总文件、时长与容量', 'Summarized by top-level subdirectory: files, duration, and size'],
    ['质量诊断', 'Quality Diagnostics'],
    ['规则诊断用于定位值得复查的样本', 'Rule-based diagnostics identify samples worth reviewing'],
    ['逐视频明细', 'Per-Video Details'],
    ['点击表头排序 · 服务端分页', 'Click a column header to sort · Server-side pagination'],
    ['搜索路径 / 编码 / 分辨率', 'Search path / codec / resolution'],
    ['全部质量状态', 'All Quality States'],
    ['存在诊断项', 'Has Diagnostic Findings'],
    ['无诊断项', 'No Diagnostic Findings'],
    ['解析失败', 'Parse Failed'],
    ['全部编码', 'All Codecs'],
    ['全部分辨率', 'All Resolutions'],
    ['全部方向', 'All Orientations'],
    ['50 条/页', '50 per page'],
    ['100 条/页', '100 per page'],
    ['200 条/页', '200 per page'],
    ['500 条/页', '500 per page'],
    ['视频路径', 'Video Path'],
    ['时长', 'Duration'],
    ['帧数', 'Frames'],
    ['分辨率', 'Resolution'],
    ['编码', 'Codec'],
    ['码率', 'Bitrate'],
    ['容量', 'Size'],
    ['音频', 'Audio'],
    ['← 上一页', '← Previous Page'],
    ['下一页 →', 'Next Page →'],
    ['相机内参分布', 'Camera Intrinsics Distribution'],
    ['每个 episode 仅采样首帧 FOV，并结合视频分辨率反解焦距', 'Sample only the first-frame FOV of each episode and derive focal length from video resolution.'],
    ['批量视频推理', 'Batch Video Inference'],
    ['多 GPU 推理与多视频渲染流水并发，不再逐个视频等待 MP4', 'Multi-GPU inference and multi-video rendering run concurrently without waiting for each MP4 sequentially.'],
    ['收起（后台任务继续）', 'Collapse (task continues in the background)'],
    ['批量模型 Run', 'Batch Model Run'],
    ['为本次批量任务单独选择训练 Run；启动后自动切换并加载', 'Choose a training run for this batch task; it switches and loads automatically after starting.'],
    ['加载中…', 'Loading…'],
    ['请先选择 Run', 'Select a Run First'],
    ['本次批量任务实际使用的 checkpoint', 'Checkpoint used for this batch task'],
    ['输入目录', 'Input Directory'],
    ['输出根目录', 'Output Root Directory'],
    ['服务器绝对路径；按输入相对层级镜像保存', 'Absolute server path; saved using a mirrored relative directory structure'],
    ['文件名模板', 'Filename Template'],
    ['支持 {stem} {name} {ext} {parent} {index}，例如 {index:04d}_{stem}', 'Supports {stem} {name} {ext} {parent} {index}, for example {index:04d}_{stem}'],
    ['渲染模式', 'Render Mode'],
    ['Full 超过安全上限会失败；批量长视频请选择最大窗分窗', 'Full fails above the safe limit; use Max Chunked for long batch videos.'],
    ['手部拼窗 / 后处理', 'Hand Window Merge / Postprocessing'],
    ['覆盖已有 MP4 / NPZ', 'Overwrite Existing MP4 / NPZ'],
    ['▶ 开始批量推理', '▶ Start Batch Inference'],
    ['待运行', 'Ready to Run'],
    ['任务日志', 'Task Log'],
    ['Benchmark 测评', 'Benchmark Evaluation'],
    ['添加多个模型，以相同功能、数据和序列范围依次测评并横向对比', 'Add multiple models and evaluate them sequentially with the same features, data, and sequence range for side-by-side comparison.'],
    ['＋ 加入待测模型', '+ Add Model to Evaluate'],
    ['历史份量', 'History Data Fraction'],
    ['先按测评数据份量筛选历史记录', 'Filter history by evaluation data fraction first'],
    ['全部', 'All'],
    ['历史测评（自动扫描 logs）', 'Evaluation History (scan logs automatically)'],
    ['自动扫描默认 Benchmark 保存目录', 'Scan the default Benchmark output directory automatically'],
    ['扫描中…', 'Scanning…'],
    ['＋ 加入历史对比', '+ Add Historical Comparison'],
    ['本次测评总进度', 'Overall Evaluation Progress'],
    ['等待启动', 'Waiting to Start'],
    ['Benchmark 总进度', 'Overall Benchmark Progress'],
    ['完成', 'Completed'],
    ['当前已用', 'Elapsed'],
    ['预计总时长', 'Estimated Total'],
    ['预计还需', 'Estimated Remaining'],
    ['启动后先枚举评测单元，获得首批完成速度后开始估算时间。', 'Evaluation units are discovered after startup; time estimates begin after the first completed batch.'],
    ['对比协议', 'Comparison Protocol'],
    ['Benchmark 对比协议', 'Benchmark Comparison Protocol'],
    ['联合 SOTA', 'Combined SOTA'],
    ['相机轨迹对比', 'Camera Trajectory Comparison'],
    ['HOT3D 相机表', 'HOT3D Camera Table'],
    ['ARCTIC 相机表', 'ARCTIC Camera Table'],
    ['ViDiHand 对比', 'ViDiHand Comparison'],
    ['ViDiHand（已在大部分 ARCTIC/HOT3D 数据上进行过训练，仅供参考，不具备实际意义）', 'ViDiHand (trained on most ARCTIC/HOT3D data; for reference only and not practically meaningful)'],
    ['全部评测', 'All Evaluations'],
    ['导出四张结果表 · JPG', 'Export Four Result Tables · JPG'],
    ['多选表格后可分别保存，或汇总为一张长图；导出包含固定行及已有的本次模型结果。', 'Select one or more tables, then save separate files or one combined long image. Exports include fixed rows and available current-model results.'],
    ['选择要导出的 Benchmark 表格', 'Select Benchmark tables to export'],
    ['ViDiHand · ARCTIC', 'ViDiHand · ARCTIC'],
    ['ViDiHand · HOT3D', 'ViDiHand · HOT3D'],
    ['全选表格', 'Select All Tables'],
    ['清空选择', 'Clear Selection'],
    ['分别导出 JPG', 'Export Separate JPGs'],
    ['汇总导出 JPG', 'Export Combined JPG'],
    ['就绪 · 图片按完整指标列导出', 'Ready · Images include all metric columns'],
    ['请至少选择一张表格', 'Select at least one table'],
    ['公开 SOTA + 本次模型 · 实时', 'Published SOTA + Current Models · Live'],
    ['M1、M2… 分行显示，等待评测结果', 'M1, M2… are shown on separate rows while results are pending'],
    ['本次模型横向对比 · 实时', 'Current Model Comparison · Live'],
    ['每个模型完成后加入横向指标表', 'Each model is added to the comparison table after completion'],
    ['运行日志', 'Run Log'],
    ['📋 配置 / 代码变更', '📋 Configuration / Code Changes'],
    ['只展示改变的配置，并可把代码 diff 限制到指定模块', 'Show only changed configuration and optionally limit the code diff to a selected module.'],
    ['基准 run（相对 model_train）', 'Baseline run (relative to model_train)'],
    ['对比 run（相对 model_train）', 'Comparison run (relative to model_train)'],
    ['代码模块', 'Code Module'],
    ['只比较训练快照中选定的顶层代码模块', 'Compare only the selected top-level code module in the training snapshot'],
    ['全部代码', 'All Code'],
    ['▶ 对比', '▶ Compare'],
    ['收起（后台对比继续）', 'Collapse (comparison continues in the background)'],
    ['浏览与编排', 'Browse and Arrange'],
    ['栏目顺序', 'Panel Order'],
    ['拖动 ☰ 调整顺序；点 👁 隐藏/显示对应模块（视频 / 固定世界 / 当前相机 / MuJoCo / Wuji Hand Retargeting / 数值 / loss / 工具面板）。仿真、retargeting 与工具面板默认关闭；MuJoCo / Wuji Hand 仅在手动开启后开始渲染', 'Drag ☰ to reorder. Click 👁 to hide or show panels (video / fixed world / current camera / MuJoCo / Wuji Hand Retargeting / values / loss / tools). Simulation, retargeting, and tool panels are off by default; MuJoCo / Wuji Hand render only after manual activation.'],
    ['输入目录', 'Input Directory'],
    ['默认起点 output/…/lerobot；可沿面包屑上溯。到 lerobot 目录后选择 Episode；普通目录点击视频只会选中，不会自动推理。', 'The default start is output/…/lerobot. Use breadcrumbs to move upward. Select an episode in a lerobot directory; clicking a video in a regular directory selects it without starting inference.'],
    ['已选路径,点某层可回退', 'Selected path; click a level to go back'],
    ['返回上一级目录', 'Go to parent directory'],
    ['⬆ 上一级', '⬆ Parent Directory'],
    ['点目录进入 / 点视频文件选中输入（不会自动推理）', 'Click a directory to enter it or a video file to select it (inference does not start automatically).'],
    ['整体·2D', 'Combined · 2D'],
    ['等待整体·2D完成…', 'Waiting for Combined · 2D to finish…'],
    ['固定世界·3D', 'Fixed World · 3D'],
    ['当前相机·3D', 'Current Camera · 3D'],
    ['MuJoCo·仿真', 'MuJoCo · Simulation'],
    ['逐帧数值（世界系）', 'Per-Frame Values (World)'],
    ['逐帧数值（相机系）', 'Per-Frame Values (Camera)'],
    ['逐帧 loss', 'Per-Frame Loss'],
    ['数据集构成分析', 'Dataset Composition Analysis'],
    ['🏁 Benchmark 测评', '🏁 Benchmark Evaluation'],
    ['📋 配置/代码变更', '📋 Configuration / Code Changes'],
    ['轴', 'Axes'],
    ['显示/隐藏', 'Show / Hide'],
    ['（无数据）', '(No Data)'],
    ['加载失败', 'Load Failed'],
    ['加载失败:', 'Load Failed:'],
    ['推理失败', 'Inference Failed'],
    ['推理完成', 'Inference Complete'],
    ['模型加载完成', 'Model Loaded'],
    ['正在加载模型', 'Loading Model'],
    ['正在推理', 'Running Inference'],
    ['已取消', 'Cancelled'],
    ['错误', 'Error'],
    ['警告', 'Warning'],
    ['成功', 'Success'],
    ['未知', 'Unknown'],
    ['暂无数据', 'No Data'],
    ['无数据', 'No Data'],
    ['没有结果', 'No Results'],
    ['未选择', 'Not Selected'],
    ['未加载', 'Not Loaded'],
    ['模型未加载', 'Model Not Loaded'],
    ['重试', 'Retry'],
    ['关闭', 'Close'],
    ['取消', 'Cancel'],
    ['确认', 'Confirm'],
    ['删除', 'Delete'],
    ['添加', 'Add'],
    ['保存', 'Save'],
    ['导出', 'Export'],
    ['刷新', 'Refresh'],
    ['浏览', 'Browse'],
    ['详情', 'Details'],
    ['结果', 'Results'],
    ['设置', 'Settings'],
    ['状态', 'Status'],
    ['进度', 'Progress'],
    ['当前', 'Current'],
    ['总计', 'Total'],
    ['平均', 'Mean'],
    ['中位数', 'Median'],
    ['最小值', 'Minimum'],
    ['最大值', 'Maximum'],
    ['序列', 'Sequence'],
    ['样本', 'Sample'],
    ['模型', 'Model'],
    ['方法', 'Method'],
    ['指标', 'Metric'],
    ['单位', 'Unit'],
    ['数据集', 'Dataset'],
    ['相机', 'Camera'],
    ['左手', 'Left Hand'],
    ['右手', 'Right Hand'],
    ['双手', 'Both Hands'],
    ['真值', 'Ground Truth'],
    ['预测', 'Prediction'],
    ['原始', 'Raw'],
    ['平滑', 'Smoothing'],
    ['世界系', 'World Space'],
    ['相机系', 'Camera Space'],
    ['固定世界', 'Fixed World'],
    ['当前相机', 'Current Camera'],
    ['视频', 'Video'],
    ['文件', 'File'],
    ['目录', 'Directory'],
    ['路径', 'Path'],
    ['页面', 'Page'],
    ['上一页', 'Previous Page'],
    ['下一页', 'Next Page'],
    ['是', 'Yes'],
    ['否', 'No'],
    ['有', 'Yes'],
    ['无', 'None'],
  ]);

  const REPLACEMENTS = Array.from(PHRASES.entries())
    .filter(([source]) => source.length > 1)
    .sort((a, b) => b[0].length - a[0].length);

  const textState = new WeakMap();
  const attributeState = new WeakMap();
  let language = readInitialLanguage();

  function readInitialLanguage() {
    try {
      return localStorage.getItem(STORAGE_KEY) === 'zh' ? 'zh' : 'en';
    } catch (_) {
      return 'en';
    }
  }

  function containsCJK(value) {
    return CJK_RE.test(String(value == null ? '' : value));
  }

  function preserveOuterWhitespace(source, translated) {
    const leading = source.match(/^\s*/)?.[0] || '';
    const trailing = source.match(/\s*$/)?.[0] || '';
    return leading + translated + trailing;
  }

  function translatePatterns(value) {
    return value
      .replace(/第\s*(\d+)\s*页/g, 'Page $1')
      .replace(/共\s*(\d+)\s*页/g, '$1 pages total')
      .replace(/共\s*(\d+)\s*条/g, '$1 total')
      .replace(/共\s*(\d+)\s*个/g, '$1 total')
      .replace(/(\d+)\s*帧\/窗/g, '$1 frames/window')
      .replace(/(\d+)\s*帧/g, '$1 frames')
      .replace(/(\d+)\s*条\/页/g, '$1 per page')
      .replace(/(\d+)\s*文件\/集/g, '$1 files/dataset')
      .replace(/已完成\s*(\d+)/g, 'Completed $1')
      .replace(/失败\s*(\d+)/g, 'Failed $1')
      .replace(/耗时\s*([\d.]+)\s*秒/g, 'Elapsed $1 s')
      .replace(/([\d.]+)\s*秒/g, '$1 s');
  }

  function sanitizeEnglish(value) {
    let output = value
      .replace(CJK_RUN_RE, ' ')
      .replace(/，/g, ', ')
      .replace(/。/g, '. ')
      .replace(/；/g, '; ')
      .replace(/：/g, ': ')
      .replace(/！/g, '! ')
      .replace(/？/g, '? ')
      .replace(/、/g, ', ')
      .replace(/[“”]/g, '"')
      .replace(/[‘’]/g, "'")
      .replace(/（/g, ' (')
      .replace(/）/g, ') ')
      .replace(/[【《]/g, ' [')
      .replace(/[】》]/g, '] ')
      .replace(/\s+([,.;:!?%)\]])/g, '$1')
      .replace(/([(\[])\s+/g, '$1')
      .replace(/\s{2,}/g, ' ')
      .trim();
    if (!output || !/[A-Za-z0-9%+\-_/\\.]/.test(output)) output = 'Translation unavailable';
    return output;
  }

  function translateText(value) {
    const source = String(value == null ? '' : value);
    if (language !== 'en' || !containsCJK(source)) return source;
    const core = source.trim();
    if (!core) return source;
    const exact = PHRASES.get(core);
    if (exact) return preserveOuterWhitespace(source, exact);

    let translated = core;
    for (const [from, to] of REPLACEMENTS) {
      if (translated.includes(from)) translated = translated.split(from).join(to);
    }
    translated = translatePatterns(translated);
    if (containsCJK(translated)) translated = sanitizeEnglish(translated);
    return preserveOuterWhitespace(source, translated);
  }

  function isIgnored(node) {
    const element = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
    return Boolean(element && element.closest('[data-i18n-ignore]'));
  }

  function translateTextNode(node) {
    if (!node || node.nodeType !== Node.TEXT_NODE || isIgnored(node)) return;
    const parent = node.parentElement;
    if (parent && SKIP_TAGS.has(parent.tagName)) return;
    const current = node.nodeValue || '';
    let state = textState.get(node);

    if (language === 'zh') {
      if (state && current === state.rendered && current !== state.source) node.nodeValue = state.source;
      if (state) state.rendered = state.source;
      return;
    }

    if (!state || current !== state.rendered) {
      state = {source: current, rendered: current};
      textState.set(node, state);
    }
    const rendered = translateText(state.source);
    state.rendered = rendered;
    if (current !== rendered) node.nodeValue = rendered;
  }

  function translateAttribute(element, name) {
    if (!element.hasAttribute(name) || isIgnored(element)) return;
    const current = element.getAttribute(name) || '';
    let states = attributeState.get(element);
    if (!states) {
      states = new Map();
      attributeState.set(element, states);
    }
    let state = states.get(name);

    if (language === 'zh') {
      if (state && current === state.rendered && current !== state.source) element.setAttribute(name, state.source);
      if (state) state.rendered = state.source;
      return;
    }

    if (!state || current !== state.rendered) {
      state = {source: current, rendered: current};
      states.set(name, state);
    }
    const rendered = translateText(state.source);
    state.rendered = rendered;
    if (current !== rendered) element.setAttribute(name, rendered);
  }

  function translateInputValue(element) {
    if (!(element instanceof HTMLInputElement) || !['button', 'submit', 'reset'].includes(element.type)) return;
    translateAttribute(element, 'value');
  }

  function translateOwnElement(element) {
    if (!(element instanceof Element) || isIgnored(element) || SKIP_TAGS.has(element.tagName)) return;
    TRANSLATABLE_ATTRIBUTES.forEach(name => translateAttribute(element, name));
    translateInputValue(element);
  }

  function translateElement(root) {
    if (!root) return;
    if (root.nodeType === Node.TEXT_NODE) {
      translateTextNode(root);
      return;
    }
    if (!(root instanceof Element) && root !== document) return;
    if (root instanceof Element && (isIgnored(root) || SKIP_TAGS.has(root.tagName))) return;
    if (root instanceof Element) translateOwnElement(root);

    const walker = document.createTreeWalker(
      root,
      NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            if (SKIP_TAGS.has(node.tagName) || node.hasAttribute('data-i18n-ignore')) return NodeFilter.FILTER_REJECT;
            return NodeFilter.FILTER_ACCEPT;
          }
          return isIgnored(node) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
        },
      },
    );
    let node = walker.nextNode();
    while (node) {
      if (node.nodeType === Node.TEXT_NODE) translateTextNode(node);
      else translateOwnElement(node);
      node = walker.nextNode();
    }
  }

  function updateLanguageControl() {
    const select = document.getElementById('languageSelect');
    const label = document.getElementById('languageLabel');
    if (select) {
      select.value = language;
      const en = select.querySelector('option[value="en"]');
      const zh = select.querySelector('option[value="zh"]');
      if (en) en.textContent = language === 'zh' ? '英语' : 'English';
      if (zh) zh.textContent = language === 'zh' ? '中文' : 'Chinese';
    }
    if (label) label.textContent = language === 'zh' ? '语言' : 'Language';
  }

  function setLanguage(nextLanguage, options = {}) {
    language = nextLanguage === 'zh' ? 'zh' : 'en';
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    if (options.persist !== false) {
      try { localStorage.setItem(STORAGE_KEY, language); } catch (_) { /* Storage can be disabled. */ }
    }
    translateElement(document.documentElement);
    updateLanguageControl();
    window.dispatchEvent(new CustomEvent('viewer:languagechange', {detail: {language}}));
    window.dispatchEvent(new Event('resize'));
  }

  function patchDialogs() {
    for (const name of ['alert', 'confirm', 'prompt']) {
      const original = window[name];
      if (typeof original !== 'function') continue;
      window[name] = function translatedDialog(message, ...args) {
        return original.call(window, translateText(message), ...args);
      };
    }
  }

  function patchCanvasText() {
    if (typeof CanvasRenderingContext2D === 'undefined') return;
    const prototype = CanvasRenderingContext2D.prototype;
    for (const name of ['fillText', 'strokeText', 'measureText']) {
      const original = prototype[name];
      if (typeof original !== 'function' || original.__viewerI18nPatched) continue;
      const patched = function translatedCanvasText(text, ...args) {
        return original.call(this, translateText(text), ...args);
      };
      Object.defineProperty(patched, '__viewerI18nPatched', {value: true});
      prototype[name] = patched;
    }
  }

  const pendingRoots = new Set();
  let translationQueued = false;
  function queueTranslation(node) {
    if (!node) return;
    pendingRoots.add(node);
    if (translationQueued) return;
    translationQueued = true;
    queueMicrotask(() => {
      translationQueued = false;
      const roots = Array.from(pendingRoots);
      pendingRoots.clear();
      roots.forEach(translateElement);
    });
  }

  function observeDynamicContent() {
    const observer = new MutationObserver(records => {
      for (const record of records) {
        if (record.type === 'characterData') queueTranslation(record.target);
        else if (record.type === 'attributes') queueTranslation(record.target);
        else record.addedNodes.forEach(queueTranslation);
      }
    });
    observer.observe(document.documentElement, {
      subtree: true,
      childList: true,
      characterData: true,
      attributes: true,
      attributeFilter: TRANSLATABLE_ATTRIBUTES.concat('value'),
    });
  }

  patchDialogs();
  patchCanvasText();
  translateElement(document.documentElement);
  updateLanguageControl();
  observeDynamicContent();

  const languageSelect = document.getElementById('languageSelect');
  if (languageSelect) languageSelect.addEventListener('change', event => setLanguage(event.target.value));

  document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  document.documentElement.classList.remove('i18n-pending');

  window.ViewerI18n = {
    get language() { return language; },
    t: translateText,
    setLanguage,
    translateElement,
    containsCJK,
  };
})();
