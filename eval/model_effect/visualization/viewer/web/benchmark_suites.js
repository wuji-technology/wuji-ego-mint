// Paper-aligned comparison presets. Public rows stay separate from local runs
// because the available local HOT3D/ARCTIC splits are not the papers' test sets.
const CAMERA_BASELINES = typeof CAMERA_TRAJECTORY_BASELINES === 'undefined' ? {
  source: '相机离线基线尚未生成', note: '', twoDatasetRows: [],
  hot3dRows: [], arcticRows: [], exclusions: [],
} : CAMERA_TRAJECTORY_BASELINES;

const ICRA_CAMERA_METRICS = [
  {id: 'Coverage', label: '覆盖', direction: 'max', unit: 'sequence', detail: {
    name: 'Successful sequence coverage',
    definition: '成功输出完整、有限且与 GT 逐帧对齐的相机轨迹序列数 / 固定评测清单总序列数。',
    alignment: '覆盖率在对齐和指标计算前审核；失败、缺帧、NaN/Inf 都不静默删除。',
    protocol: 'HOT3D 固定 27 条，ARCTIC 固定 34 条；表中同时显示分子与分母。',
    local: '运行中的 M 行从 report counts.sequences 读取；完整结果应为 27/27 或 34/34。'}},
  {id: 'ATE-S mean', label: 'ATE均', direction: 'min', unit: 'mm', detail: {
    name: 'Mean whole-sequence metric-scale ATE-RMSE',
    definition: '每条序列先计算相机中心逐帧位置误差的 RMSE，再对固定清单内各成功序列等权取均值。',
    alignment: '每条完整视频只拟合一次 Umeyama SE(3) 的旋转和平移，尺度固定为 1；不按 32 帧窗口重新对齐。',
    protocol: '对应评测实现的 ATE_S_mm；图表中的 ATE 是“不吸收尺度”的 ATE-S 口径。',
    local: '直接惩罚模型输出的公制尺度误差；非公制方法只能作为固定输入尺度下的记录。'}},
  {id: 'ATE-S median', label: 'ATE中', direction: 'min', unit: 'mm', detail: {
    name: 'Median whole-sequence metric-scale ATE-RMSE',
    definition: '上述每序列 ATE-S 的中位数，降低少数困难长序列对均值的影响。',
    alignment: '与 ATE均完全相同：每条完整视频一次 SE(3)，尺度固定为 1。',
    protocol: '不是逐帧误差中位数，而是“每序列 RMSE”在数据集上的中位数。',
    local: '与 ATE均并列报告，用于判断误差是否由少数失效序列主导。'}},
  {id: 'RPE-T-S mean', label: 'RPE-T均', direction: 'min', unit: 'mm', detail: {
    name: 'Mean one-frame metric-scale relative translation RMSE',
    definition: '对相邻帧相对 SE(3) 的误差变换取平移模长，在每条序列内算 RMSE，再跨序列等权取均值。',
    alignment: '使用同一条尺度固定为 1 的整段 SE(3) 对齐轨迹，不吸收预测平移尺度。',
    protocol: 'delta=1 frame、所有相邻帧对，单位 mm；对应 RPE_T_S_mm。',
    local: '衡量短时局部相机平移；不能替代反映长程漂移的 ATE。'}},
  {id: 'RPE-T-S median', label: 'RPE-T中', direction: 'min', unit: 'mm', detail: {
    name: 'Median one-frame metric-scale relative translation RMSE',
    definition: '每条序列 RPE-T RMSE 在数据集上的中位数。',
    alignment: '尺度固定为 1，和 RPE-T均采用完全相同的相邻帧误差。',
    protocol: '不是相邻帧误差样本的全局中位数，而是序列级 RMSE 中位数。',
    local: '和均值差距大时，说明少数序列存在局部跟踪失效。'}},
  {id: 'RPE-R mean', label: 'RPE-R均', direction: 'min', unit: 'deg', detail: {
    name: 'Mean one-frame relative rotation RMSE',
    definition: '相邻帧相对旋转误差的 SO(3) 测地角，在每条序列内算 RMSE，再跨序列等权平均。',
    alignment: '全局 SE(3) 对齐不会改变相邻帧相对旋转误差。',
    protocol: 'delta=1 frame，单位 degree。',
    local: '用于检查旋转头、快速头动和窗口拼接处的姿态稳定性。'}},
  {id: 'RPE-R median', label: 'RPE-R中', direction: 'min', unit: 'deg', detail: {
    name: 'Median one-frame relative rotation RMSE',
    definition: '每条序列 RPE-R RMSE 在数据集上的中位数。',
    alignment: '与 RPE-R均使用相同相邻帧 SO(3) 测地角。',
    protocol: '序列级统计，单位 degree。',
    local: '与均值配合区分普遍旋转误差和少数灾难性序列。'}},
  {id: 'Path scale', label: '弧长比', direction: 'target', unit: 'x', detail: {
    name: 'GT/predicted path-length ratio',
    definition: '每条序列 GT 累计弧长除以预测累计弧长，再跨序列等权平均；目标为 1。',
    alignment: '直接用未缩放的原始预测轨迹计算，不依赖轨迹形状拟合。',
    protocol: '大于 1 表示预测路程整体偏短，小于 1 表示预测路程偏长。',
    local: '非公制方法的值只描述当前任意输出尺度，不代表恢复了物理尺度。'}},
  {id: 'ATE-S pct', label: 'ATE%', direction: 'min', unit: '%', detail: {
    name: 'Metric-scale ATE relative to GT path length',
    definition: '每条序列 ATE-S 除以该序列 GT 累计弧长并乘 100，再跨序列等权平均。',
    alignment: '分子为整段 SE(3)、尺度固定为 1 的 ATE-RMSE。',
    protocol: '用于减弱 HOT3D 与 ARCTIC 运动范围不同对绝对毫米误差的影响。',
    local: '仍是序列等权；不能由“平均 ATE / 平均弧长”代替。'}},
];

const ICRA_CAMERA_COLUMNS = [
  {key: 'coverage', label: '覆盖', metric: 'Coverage', direction: 'max', format: 'coverage'},
  {key: 'ateMean', label: 'ATE均', metric: 'ATE-S mean', direction: 'min', unit: 'mm', digits: 1},
  {key: 'ateMedian', label: 'ATE中', metric: 'ATE-S median', direction: 'min', unit: 'mm', digits: 1},
  {key: 'rpeTMean', label: 'RPE-T均', metric: 'RPE-T-S mean', direction: 'min', unit: 'mm', digits: 2},
  {key: 'rpeTMedian', label: 'RPE-T中', metric: 'RPE-T-S median', direction: 'min', unit: 'mm', digits: 2},
  {key: 'rpeRMean', label: 'RPE-R均', metric: 'RPE-R mean', direction: 'min', unit: 'deg', digits: 3},
  {key: 'rpeRMedian', label: 'RPE-R中', metric: 'RPE-R median', direction: 'min', unit: 'deg', digits: 3},
  {key: 'pathScale', label: '弧长比', metric: 'Path scale', direction: 'target', unit: 'x', digits: 3},
  {key: 'atePct', label: 'ATE%', metric: 'ATE-S pct', direction: 'min', unit: '%', digits: 2},
];

const ICRA_CAMERA_LIVE_VALUES = {
  coverage: {metric: 'sequences', source: 'counts'},
  ateMean: 'ATE_S_mm', ateMedian: 'ATE_S_median_mm',
  rpeTMean: 'RPE_T_S_mm', rpeTMedian: 'RPE_T_S_median_mm',
  rpeRMean: 'RPE_R_deg', rpeRMedian: 'RPE_R_median_deg',
  pathScale: 'path_scale', atePct: 'ATE_S_pct',
};

// Table 2 of the 2026-09-05 manuscript; MegaSaM HOT3D coverage corrected to 27/27.
const ICRA_HOT3D_CAMERA_ROWS = [
  {method: 'DROID-SLAM', values: ['27/27', 49.1, 39.6, 5.36, 3.52, 0.227, 0.146, 0.778, 0.36]},
  {method: 'HaWoR', values: ['27/27', 200.3, 179.2, 8.60, 7.33, 1.098, 0.928, 0.950, 1.28]},
  {method: 'InfiniteVGGT', values: ['27/27', 124.8, 100.0, 13.52, 9.67, 1.492, 0.392, 0.556, 0.80]},
  {method: 'LingBot-Map', values: ['27/27', 85.5, 51.0, 7.56, 6.18, 0.684, 0.253, 0.712, 0.55]},
  {method: 'MegaSaM†', values: ['27/27', 94.4, 65.3, 3.18, 2.13, 0.082, 0.063, 0.716, 0.69]},
  {method: 'MINT（未微调相机轨迹）', values: ['27/27', 524.7, 434.6, 8.75, 8.37, 0.234, 0.229, 0.466, 3.29]},
  {method: 'MINT（相机轨迹二阶段微调）', values: ['27/27', 181.7, 155.5, 4.69, 4.78, 0.284, 0.259, 1.094, 1.15]},
];

const ICRA_ARCTIC_CAMERA_ROWS = [
  {method: 'DROID-SLAM', values: ['34/34', 181.5, 49.6, 33.84, 14.25, 1.006, 0.423, 0.964, 8.07]},
  {method: 'HaWoR', values: ['34/34', 66.2, 29.4, 24.08, 4.16, 0.298, 0.151, 0.759, 2.87]},
  {method: 'InfiniteVGGT', values: ['34/34', 79.0, 69.1, 16.21, 12.63, 1.265, 0.655, 0.284, 3.23]},
  {method: 'LingBot-Map', values: ['34/34', 59.6, 62.0, 9.17, 8.47, 0.980, 0.717, 0.591, 2.46]},
  {method: 'MegaSaM†', values: ['34/34', 51.4, 50.4, 8.73, 5.58, 0.779, 0.725, 1.956, 2.15]},
  {method: 'MINT（未微调相机轨迹）', values: ['34/34', 63.7, 59.3, 3.53, 3.62, 0.251, 0.243, 0.755, 2.63]},
  {method: 'MINT（相机轨迹二阶段微调）', values: ['34/34', 81.9, 83.2, 3.39, 3.37, 0.256, 0.251, 1.412, 3.40]},
];

const VIDIHAND_COLUMNS = [
  {key: 'facc', label: 'FAcc', metric: 'FAcc', direction: 'max'},
  {key: 'recall', label: 'Recall', metric: 'Recall', direction: 'max'},
  {key: 'f1', label: 'F1', metric: 'F1', direction: 'max'},
  {key: 'mpjpe', label: 'MPJPE-p', metric: 'MPJPE-p', direction: 'min'},
  {key: 'pa', label: 'PA-MPJPE-p', metric: 'PA-MPJPE-p', direction: 'min'},
  {key: 'go', label: 'GO-p', metric: 'GO-p', direction: 'min'},
  {key: 'ct', label: 'CT-p', metric: 'CT-p', direction: 'min'},
  {key: 'jitter', label: 'Jitter', metric: 'Jitter', direction: 'min'},
];

const VIDIHAND_ARCTIC_ROWS = [
  {method: 'InterWild [16]', values: [0.878, 0.943, 0.959, 30.817, 15.952, 25.386, 0.097, 46.577]},
  {method: 'HaMeR [18]', values: [0.875, 0.943, 0.957, 29.197, 14.596, 24.907, 0.095, 18.279]},
  {method: 'Hamba [3]', values: [0.833, 0.912, 0.941, 31.233, 17.168, 27.822, 0.110, 15.357]},
  {method: 'WildHands [20]', values: [0.879, 0.946, 0.960, 25.704, 13.941, 22.320, 0.058, 12.972]},
  {method: 'OmniHands [14]', values: [0.866, 0.949, 0.954, 29.674, 14.203, 24.580, 0.087, 45.312]},
  {method: 'WiLoR [19]', values: [0.919, 0.951, 0.974, 22.012, 11.873, 17.358, 0.075, 24.091]},
  {method: 'Dyn-HaMR [32]', values: [0.842, 0.918, 0.951, 27.904, 17.017, 25.951, 0.121, 12.840]},
  {method: 'HaWoR [34]', values: [0.700, 0.817, 0.895, 45.357, 26.375, 43.325, 0.149, 19.789]},
  {method: 'ViDiHand（已在大部分 ARCTIC/HOT3D 数据上进行过训练，仅供参考，不具备实际意义）', values: [0.997, 0.999, 0.999, 21.668, 9.821, 14.642, 0.047, 3.183]},
  {section: '本地固定记录 · sota-fixed-v1 50%（非论文同 split）', method: 'MINT（相机轨迹二阶段微调）· UKF†', comparable: false,
   values: [0.9164, 0.9571, 0.9781, 51.0883, 27.7045, 24.2211, 0.1404, 2.5410]},
  {method: 'MINT（未微调相机轨迹）· raw†', comparable: false,
   values: [0.9164, 0.9571, 0.9781, 51.0271, 27.7103, 24.1860, 0.1404, 12.2614]},
];

const VIDIHAND_HOT3D_ROWS = [
  {method: 'InterWild [16]', values: [0.669, 0.881, 0.868, 77.168, 24.811, 58.501, 0.213, 101.164]},
  {method: 'HaMeR [18]', values: [0.692, 0.904, 0.883, 68.314, 21.455, 49.636, 0.102, 23.632]},
  {method: 'Hamba [3]', values: [0.632, 0.828, 0.853, 71.732, 29.620, 56.525, 0.128, 18.507]},
  {method: 'WildHands [20]', values: [0.655, 0.863, 0.844, 52.791, 28.946, 53.933, 0.157, 22.885]},
  {method: 'OmniHands [14]', values: [0.649, 0.895, 0.868, 63.281, 22.682, 49.120, 0.133, 69.510]},
  {method: 'WiLoR [19]', values: [0.827, 0.897, 0.937, 30.966, 19.980, 25.746, 0.098, 17.976]},
  {method: 'Dyn-HaMR [32]', values: [0.614, 0.811, 0.802, 74.214, 38.201, 43.851, 0.571, 44.942]},
  {method: 'HaWoR [34]', values: [0.348, 0.499, 0.654, 71.396, 66.031, 79.350, 0.262, 23.872]},
  {method: 'ViDiHand（已在大部分 ARCTIC/HOT3D 数据上进行过训练，仅供参考，不具备实际意义）', values: [0.948, 0.974, 0.983, 21.514, 11.383, 15.829, 0.040, 3.741]},
  {section: '本地固定记录 · sota-fixed-v1 50%（非论文同 split）', method: 'MINT（相机轨迹二阶段微调）· UKF†', comparable: false,
   values: [0.9404, 0.9767, 0.9501, 23.6182, 10.6857, 16.7675, 0.0731, 2.3875]},
  {method: 'MINT（未微调相机轨迹）· raw†', comparable: false,
   values: [0.9404, 0.9767, 0.9501, 23.6133, 10.7008, 16.7750, 0.0733, 11.5162]},
];

const VIDIHAND_LIVE_VALUES = {
  facc: 'FAcc', recall: 'Recall', f1: 'F1', mpjpe: 'MPJPE-p', pa: 'PA-MPJPE-p',
  go: 'GO-p', ct: 'CT-p', jitter: 'Jitter',
};
const BENCHMARK_SUITES = Object.freeze({
  custom: {
    id: 'custom',
    label: '全部评测',
    description: '保留完整功能 x 数据集网格，自由组合现有评测。',
    datasets: [],
    combos: null,
    metrics: [],
    tables: [],
  },
  sota: {
    id: 'sota',
    label: '联合 SOTA 对比',
    datasets: [
      {name: 'camera_hot3d', label: 'HOT3D · 全长相机轨迹表'},
      {name: 'camera_arctic', label: 'ARCTIC · 全长相机轨迹表'},
      {name: 'arctic_hand_coverage', label: 'ARCTIC · ViDiHand coverage-aware'},
      {name: 'hot3d_hand_coverage', label: 'HOT3D · ViDiHand coverage-aware'},
    ],
    combos: [
      ['camera_trajectory', 'camera_hot3d'],
      ['camera_trajectory', 'camera_arctic'],
      ['hands_coverage', 'arctic_hand_coverage'],
      ['hands_coverage', 'hot3d_hand_coverage'],
    ],
    liveDatasets: ['camera_hot3d', 'camera_arctic', 'arctic_hand_coverage', 'hot3d_hand_coverage'],
    metricSuiteIds: ['hot3d_camera', 'arctic_camera', 'vidihand'],
    tableSuiteIds: ['hot3d_camera', 'arctic_camera', 'vidihand'],
    metrics: [],
    tables: [],
  },
  camera: {
    id: 'camera',
    label: '相机轨迹对比',
    description: '同一面板比较一个或多个 checkpoint 在 HOT3D 27 条与 ARCTIC 34 条全长序列上的相机轨迹。',
    datasets: [
      {name: 'camera_hot3d', label: 'HOT3D · 全长相机轨迹', note: '27 条 · 94,978 帧',
       purpose: '使用已去畸变针孔图像和 Aria MPS 公制 c2w，整条视频只做一次 Sim(3) 对齐，检查长期轨迹形状、局部 RPE 与度量尺度。'},
      {name: 'camera_arctic', label: 'ARCTIC · 全长相机轨迹', note: 'P2 val · 34 条 · 25,883 帧',
       purpose: '使用 ARCTIC s05 protocol P2 validation 的完整操作序列和公制相机 GT；不使用 81 帧 hand-coverage 切片。'},
    ],
    guide: {
      title: 'ICRA 全长相机轨迹协议',
      summary: '每个 checkpoint 对两套完整视频逐帧回归 c2w。长视频沿用模型官方 benchmark 的训练窗分块与相邻窗 SE(3) 链式拼接，然后每条全局轨迹与 GT 做一次整段对齐。顶部卡片直接给出 HOT3D / ARCTIC 两栏 ATE，详细横向表保留全部诊断指标。',
      stages: [
        {label: '模型前向', text: '按训练 clip_len（通常 32 帧）分窗，窗口批量前向后用相邻重叠帧的 SE(3) 关系串成一条全局轨迹；不会尝试数千帧整段前向。'},
        {label: '整段对齐', text: '每条视频只拟合一个 Umeyama Sim(3)；ATE-S 只做 SE(3) 对齐，以保留模型自己的公制尺度误差。'},
        {label: '模型对比', text: 'M1/M2/… 使用相同序列、图像、GT 与指标实现；HOT3D 和 ARCTIC 分栏展示，避免把两个数据集平均成一个数。'},
      ],
      datasets: [
        {label: '相机轨迹数据', text: '默认读取 data/benchmark/camera_trajectory，也可通过 CAMERA_TRAJECTORY_ROOT 指定数据根目录；目录内是 images + gt.npz + meta.json。'},
        {label: 'hand_pose 数据', text: 'data/benchmark/hand_pose 是另一套原始/训练侧手部数据，用于手部姿态、世界系手部或 81 帧 coverage 协议，不作为本相机面板输入。'},
      ],
      warning: '这里的 HOT3D/ARCTIC 是本地 validation export，不等同于论文隐藏 test split；只有 checkpoint 间使用相同本地协议的结果可直接横向比较。',
    },
    combos: [
      ['camera_trajectory', 'camera_hot3d'],
      ['camera_trajectory', 'camera_arctic'],
    ],
    liveDatasets: ['camera_hot3d', 'camera_arctic'],
    liveMetricAliases: {
      ATE_mm: 'ATE', RPE_T_mm: 'RPE-T', RPE_R_deg: 'RPE-R',
      ATE_S_mm: 'ATE-S', ATE_pct: 'ATE%', scale: 'Scale',
      scale_error_pct: 'Scale error', path_scale: 'Path scale', FPS: 'FPS',
    },
    liveMetricUnits: {
      ATE_mm: 'mm', RPE_T_mm: 'mm', RPE_R_deg: 'deg', ATE_S_mm: 'mm',
      ATE_pct: '%', scale: 'x', scale_error_pct: '%', path_scale: 'x', FPS: 'frame/s',
    },
    metrics: [
      {id: 'ATE', label: 'ATE', direction: 'min', unit: 'mm', detail: {
        name: 'Whole-sequence Sim(3) ATE-RMSE',
        definition: '整条预测相机中心经一个 7-DoF Umeyama Sim(3) 对齐到 GT 后，逐帧位置误差的 RMSE。',
        alignment: '每条完整视频只拟合一次旋转、平移与尺度，不按 32 帧窗口重新对齐。',
        protocol: '与 ICRA 相机轨迹主表的 whole_sequence ATE 一致，单位 mm。',
        local: '用于同一 validation export 上的 checkpoint 横向选择。'}},
      {id: 'RPE-T', label: 'RPE-T', direction: 'min', unit: 'mm', detail: {
        name: 'One-frame relative translation error',
        definition: '对相邻帧相对 SE(3) 的误差变换取平移模长，并在整条序列上计算 RMSE。',
        alignment: '预测轨迹使用与 ATE 相同的整段 Sim(3) 对齐。',
        protocol: 'delta=1 frame、all consecutive pairs，单位 mm。',
        local: '反映逐帧局部相机运动，不替代长期漂移指标 ATE。'}},
      {id: 'RPE-R', label: 'RPE-R', direction: 'min', unit: 'deg', detail: {
        name: 'One-frame relative rotation error',
        definition: '相邻帧相对旋转误差的 SO(3) 测地角，并在整条序列上计算 RMSE。',
        alignment: '全局坐标系旋转不改变相对旋转误差。',
        protocol: 'delta=1 frame，单位 degree。',
        local: 'ARCTIC 出现异常大值时可用来定位旋转头或轨迹拼窗问题。'}},
      {id: 'ATE-S', label: 'ATE-S', direction: 'min', unit: 'mm', detail: {
        name: 'Metric-scale ATE',
        definition: '只用 SE(3) 消除任意世界原点和朝向，不吸收预测尺度，再计算相机中心 ATE-RMSE。',
        alignment: '尺度固定为 1，因此会惩罚模型自身的公制尺度误差。',
        protocol: '对应 ICRA 表里的 ATE-S 诊断列。',
        local: '应结合 Scale 与 Path scale 判断，不要只看单一回归尺度。'}},
      {id: 'ATE%', label: 'ATE%', direction: 'min', unit: '%', detail: {
        name: 'ATE relative to GT path length',
        definition: 'ATE 除以 GT 相机轨迹累计弧长并乘 100。',
        alignment: '分子为整段 Sim(3) ATE，分母为原始公制 GT 弧长。',
        protocol: '降低 HOT3D 与 ARCTIC 运动范围差异对绝对毫米值的影响。',
        local: '仍按序列等权聚合。'}},
      {id: 'Scale', label: '最优 Scale', direction: 'target', unit: 'x', detail: {
        name: 'Umeyama prediction-to-GT scale',
        definition: '使预测相机中心最接近 GT 的 Sim(3) 回归尺度，目标为 1。',
        alignment: '形状相关性差时该回归系数也会偏移。',
        protocol: '与 ATE 的整段 Sim(3) 拟合共享同一个尺度。',
        local: '必须和 Path scale 一起读。'}},
      {id: 'Scale error', label: 'Scale error', direction: 'min', unit: '%', detail: {
        name: 'Fitted scale error', definition: '|Scale - 1| × 100%。',
        alignment: '来自整段 Umeyama 回归系数。', protocol: '0% 最好。',
        local: '模型声明公制输出时才是有意义的部署诊断。'}},
      {id: 'Path scale', label: '弧长比', direction: 'target', unit: 'x', detail: {
        name: 'GT/predicted path-length ratio',
        definition: 'GT 累计弧长除以预测累计弧长；大于 1 表示预测整体偏小。',
        alignment: '不依赖轨迹形状拟合，是尺度大小的直接读数。',
        protocol: '与最优 Scale 并列报告。', local: '目标值为 1。'}},
      {id: 'FPS', label: 'FPS', direction: 'max', unit: 'frame/s', detail: {
        name: 'Model forward throughput', definition: '总预测帧数除以模型 GPU forward 累计秒数。',
        alignment: '不含读图、GT 加载和指标计算。', protocol: '同一硬件和窗口 batch 下比较。',
        local: '多卡仅并行不同序列，单序列 FPS 口径不因卡数改变。'}},
    ],
    tables: [
      {
        id: 'camera-two-dataset-ate',
        title: '全长相机轨迹 · 两数据集 ATE',
        source: CAMERA_BASELINES.source,
        referenceKind: 'local-baseline',
        note: CAMERA_BASELINES.note+' 每个 M 行仍对应一个 pipeline checkpoint。',
        columns: [
          {key: 'hot3d', label: 'HOT3D', metric: 'ATE', direction: 'min', unit: 'mm'},
          {key: 'arctic', label: 'ARCTIC', metric: 'ATE', direction: 'min', unit: 'mm'},
        ],
        rows: CAMERA_BASELINES.twoDatasetRows,
        liveRows: [
          {label: '相机轨迹 · whole · 27/27, 34/34', head: 'camera_trajectory', dataset: 'camera_hot3d', group: 'overall', comparable: false,
           values: {
             hot3d: {metric: 'ATE_mm', dataset: 'camera_hot3d'},
             arctic: {metric: 'ATE_mm', dataset: 'camera_arctic'},
           }},
        ],
      },
      {
        id: 'camera-hot3d-full-metrics',
        title: 'HOT3D · 离线基线完整指标',
        source: CAMERA_BASELINES.source,
        referenceKind: 'local-baseline',
        note: '27 条、94,978 帧。'+CAMERA_BASELINES.note,
        columns: [
          {key: 'ate', label: 'ATE', metric: 'ATE', direction: 'min', unit: 'mm'},
          {key: 'rpeT', label: 'RPE-T', metric: 'RPE-T', direction: 'min', unit: 'mm'},
          {key: 'rpeR', label: 'RPE-R', metric: 'RPE-R', direction: 'min', unit: 'deg'},
          {key: 'ateS', label: 'ATE-S', metric: 'ATE-S', direction: 'min', unit: 'mm'},
          {key: 'atePct', label: 'ATE%', metric: 'ATE%', direction: 'min', unit: '%'},
          {key: 'scale', label: 'Scale', metric: 'Scale', direction: 'target', unit: 'x'},
          {key: 'scaleError', label: 'Scale error', metric: 'Scale error', direction: 'min', unit: '%'},
          {key: 'pathScale', label: 'Path scale', metric: 'Path scale', direction: 'target', unit: 'x'},
          {key: 'fps', label: 'FPS', metric: 'FPS', direction: 'max', unit: 'frame/s'},
        ],
        rows: CAMERA_BASELINES.hot3dRows,
        liveRows: [
          {label: 'HOT3D · whole · 本次 checkpoint', head: 'camera_trajectory', dataset: 'camera_hot3d', group: 'overall', comparable: false,
           values: {ate: 'ATE_mm', rpeT: 'RPE_T_mm', rpeR: 'RPE_R_deg', ateS: 'ATE_S_mm', atePct: 'ATE_pct', scale: 'scale', scaleError: 'scale_error_pct', pathScale: 'path_scale', fps: 'FPS'}},
        ],
      },
      {
        id: 'camera-arctic-full-metrics',
        title: 'ARCTIC · 离线基线完整指标',
        source: CAMERA_BASELINES.source,
        referenceKind: 'local-baseline',
        note: 'P2 validation s05，34 条、25,883 帧。'+CAMERA_BASELINES.note,
        columns: [
          {key: 'ate', label: 'ATE', metric: 'ATE', direction: 'min', unit: 'mm'},
          {key: 'rpeT', label: 'RPE-T', metric: 'RPE-T', direction: 'min', unit: 'mm'},
          {key: 'rpeR', label: 'RPE-R', metric: 'RPE-R', direction: 'min', unit: 'deg'},
          {key: 'ateS', label: 'ATE-S', metric: 'ATE-S', direction: 'min', unit: 'mm'},
          {key: 'atePct', label: 'ATE%', metric: 'ATE%', direction: 'min', unit: '%'},
          {key: 'scale', label: 'Scale', metric: 'Scale', direction: 'target', unit: 'x'},
          {key: 'scaleError', label: 'Scale error', metric: 'Scale error', direction: 'min', unit: '%'},
          {key: 'pathScale', label: 'Path scale', metric: 'Path scale', direction: 'target', unit: 'x'},
          {key: 'fps', label: 'FPS', metric: 'FPS', direction: 'max', unit: 'frame/s'},
        ],
        rows: CAMERA_BASELINES.arcticRows,
        liveRows: [
          {label: 'ARCTIC · whole · 本次 checkpoint', head: 'camera_trajectory', dataset: 'camera_arctic', group: 'overall', comparable: false,
           values: {ate: 'ATE_mm', rpeT: 'RPE_T_mm', rpeR: 'RPE_R_deg', ateS: 'ATE_S_mm', atePct: 'ATE_pct', scale: 'scale', scaleError: 'scale_error_pct', pathScale: 'path_scale', fps: 'FPS'}},
        ],
      },
    ],
  },
  hot3d_camera: {
    id: 'hot3d_camera',
    label: 'HOT3D 相机表',
    description: '论文 Table 2 的 HOT3D 全长相机轨迹参考值，以及本次 checkpoint 的评测结果。',
    datasets: [
      {name: 'camera_hot3d', label: 'HOT3D · 全长相机轨迹', note: '27 条 · 94,978 帧',
       purpose: '使用去畸变针孔 JPEG 与逐帧 Aria MPS 公制 c2w；每条完整视频只做一次 SE(3) 对齐，统计序列级均值与中位数。'},
    ],
    guide: {
      title: 'HOT3D 表格评测条件',
      summary: '输入为固定 27 条 HOT3D validation export（共 94,978 帧）。MINT checkpoint 使用标准相机 benchmark runner：4 卡按帧数均衡分片，每卡加载一份学生模型；每条视频按训练 clip_len=32 分窗，window_batch_size=4、image_workers=8，相邻窗口以 SE(3) 链式拼成一条完整 c2w 轨迹。',
      stages: [
        {label: '输入与前向', text: '使用序列目录中的已去畸变 pinhole images、meta.json 原始 H/W，以及训练 run 自带 config；不做测试时调参，不把数千帧视频强行整段单次前向。'},
        {label: '轨迹对齐', text: '每条完整预测轨迹与 GT 只拟合一次 Umeyama SE(3) 的旋转和平移；scale 固定为 1，ATE 与 RPE-T 都不吸收预测尺度。'},
        {label: '数据集聚合', text: '先逐序列计算 RMSE，再对成功完成评测的序列等权计算均值与中位数；覆盖率分母保留全部 27 条序列，失败序列不填入有限误差值。'},
      ],
      datasets: [
        {label: 'GT', text: 'HOT3D rectified validation export；GT 为逐帧公制 camera-to-world，图像和轨迹严格同帧。'},
        {label: '方法行', text: '固定方法行与误差指标摘录自论文 Table 2，MegaSaM 覆盖率已更正为 27/27；本次 checkpoint 的实际评测结果另行追加。比较前应核对评测清单、预处理、对齐与聚合设置。'},
      ],
      warning: 'DROID-SLAM、InfiniteVGGT、LingBot-Map 不声明公制尺度，因此其不缩放 ATE/弧长比只记录当前输出尺度。该 validation export 也不等同于论文隐藏 test split。',
    },
    combos: [['camera_trajectory', 'camera_hot3d']],
    liveDatasets: ['camera_hot3d'],
    liveMetricAliases: {
      ATE_S_mm: 'ATE-S mean', ATE_S_median_mm: 'ATE-S median',
      RPE_T_S_mm: 'RPE-T-S mean', RPE_T_S_median_mm: 'RPE-T-S median',
      RPE_R_deg: 'RPE-R mean', RPE_R_median_deg: 'RPE-R median',
      path_scale: 'Path scale', ATE_S_pct: 'ATE-S pct',
    },
    liveMetricUnits: {
      ATE_S_mm: 'mm', ATE_S_median_mm: 'mm', RPE_T_S_mm: 'mm',
      RPE_T_S_median_mm: 'mm', RPE_R_deg: 'deg', RPE_R_median_deg: 'deg',
      path_scale: 'x', ATE_S_pct: '%',
    },
    metrics: ICRA_CAMERA_METRICS,
    tables: [
      {
        id: 'icra-hot3d-camera-comparison',
        title: 'HOT3D · 全长相机轨迹',
        source: 'MINT manuscript · Table 2 · 2026-09-05 · SE(3)-only',
        referenceKind: 'paper',
        note: '误差指标摘录自论文 Table 2，MegaSaM 覆盖率更正为 27/27。27 条、94,978 帧；长度单位 mm。ATE/RPE 为成功评测序列的 RMSE 均值/中位数，弧长比为 GT/pred。MegaSaM† 表示 no CVD。',
        expectedSequences: 27,
        columns: ICRA_CAMERA_COLUMNS,
        rows: ICRA_HOT3D_CAMERA_ROWS,
        liveRows: [
          {label: 'HOT3D · whole · 本次 checkpoint', head: 'camera_trajectory', dataset: 'camera_hot3d', group: 'overall', comparable: true,
           values: ICRA_CAMERA_LIVE_VALUES},
        ],
      },
    ],
  },
  arctic_camera: {
    id: 'arctic_camera',
    label: 'ARCTIC 相机表',
    description: '论文 Table 2 的 ARCTIC 全长相机轨迹参考值，以及本次 checkpoint 的评测结果。',
    datasets: [
      {name: 'camera_arctic', label: 'ARCTIC · 全长相机轨迹', note: 'P2 val · 34 条 · 25,883 帧',
       purpose: '使用 ARCTIC s05 protocol P2 validation 的完整 egocentric 操作序列和公制 c2w；不使用 P1 的 81 帧 hand-coverage 切片。'},
    ],
    guide: {
      title: 'ARCTIC 表格评测条件',
      summary: '输入为固定 34 条 ARCTIC s05 protocol P2 validation 全长序列（共 25,883 帧）。学生模型使用标准相机 benchmark runner 的 4 卡帧数均衡分片和 windowed32 SE(3)-chain 推理路径，参数为 window_batch_size=4、image_workers=8。',
      stages: [
        {label: '输入与前向', text: '使用 undistorted egocentric images、逐序列标定和完整公制 c2w；训练 config 与 checkpoint 配套读取，不使用 P1 手部片段或任何测试时调参。'},
        {label: '轨迹对齐', text: '每条完整视频只做一次 Umeyama SE(3) 对齐，旋转/平移可变但 scale=1；因此 ATE/RPE-T 会保留模型公制尺度误差。'},
        {label: '数据集聚合', text: '每条视频先算 ATE/RPE RMSE，34 条序列等权取均值与中位数；覆盖率独立审核，ARCTIC 表中所有固定方法均为 34/34。'},
      ],
      datasets: [
        {label: 'GT', text: 'ARCTIC protocol P2 validation，subject s05，34 条完整操作序列；共 25,883 帧。'},
        {label: '方法行', text: '固定方法行与数值摘录自论文 Table 2；本次 checkpoint 的实际评测结果另行追加。比较前应核对评测清单、预处理、对齐与聚合设置。'},
        {label: '与 ViDiHand 区别', text: 'ViDiHand 面板使用 P1 validation 的 81 帧相机系手部 coverage 协议；本面板只评完整相机轨迹，不能混用结果或输入。'},
      ],
      warning: 'DROID-SLAM、InfiniteVGGT、LingBot-Map 的输出本身没有公制度量保证；它们的 ATE/弧长比按当前尺度记录。这里只能在相同 P2 export、预处理和 whole-sequence SE(3)-only 指标下横向比较。',
    },
    combos: [['camera_trajectory', 'camera_arctic']],
    liveDatasets: ['camera_arctic'],
    liveMetricAliases: {
      ATE_S_mm: 'ATE-S mean', ATE_S_median_mm: 'ATE-S median',
      RPE_T_S_mm: 'RPE-T-S mean', RPE_T_S_median_mm: 'RPE-T-S median',
      RPE_R_deg: 'RPE-R mean', RPE_R_median_deg: 'RPE-R median',
      path_scale: 'Path scale', ATE_S_pct: 'ATE-S pct',
    },
    liveMetricUnits: {
      ATE_S_mm: 'mm', ATE_S_median_mm: 'mm', RPE_T_S_mm: 'mm',
      RPE_T_S_median_mm: 'mm', RPE_R_deg: 'deg', RPE_R_median_deg: 'deg',
      path_scale: 'x', ATE_S_pct: '%',
    },
    metrics: ICRA_CAMERA_METRICS,
    tables: [
      {
        id: 'icra-arctic-camera-comparison',
        title: 'ARCTIC · 全长相机轨迹',
        source: 'MINT manuscript · Table 2 · 2026-09-05 · SE(3)-only',
        referenceKind: 'paper',
        note: '固定结果摘录自论文 Table 2。P2 validation s05，34 条、25,883 帧；长度单位 mm。ATE/RPE 为成功评测序列的 RMSE 均值/中位数，弧长比为 GT/pred。MegaSaM† 表示 no CVD。',
        expectedSequences: 34,
        columns: ICRA_CAMERA_COLUMNS,
        rows: ICRA_ARCTIC_CAMERA_ROWS,
        liveRows: [
          {label: 'ARCTIC · whole · 本次 checkpoint', head: 'camera_trajectory', dataset: 'camera_arctic', group: 'overall', comparable: true,
           values: ICRA_CAMERA_LIVE_VALUES},
        ],
      },
    ],
  },
  vidihand: {
    id: 'vidihand',
    label: 'ViDiHand 对比',
    description: '两张表中的公开方法与数值直接引用自 ViDiHand 论文 Table 1，未由本项目重新实测或修改；MINT 两行是本项目 sota-fixed-v1 50% 的本地固定记录，与论文不是同一 split。用户运行后的本次模型结果会另行追加。ViDiHand 训练使用了 ARCTIC 和 HOT3D，相关论文指标仅作同域参考。',
    datasets: [
      {name: 'arctic_hand_coverage', label: 'ARCTIC · ViDiHand coverage-aware', note: 'P1 val · 81 帧',
       purpose: '专测 ARCTIC 操作场景中的双手出现检测和相机系姿态；81 帧单次前向，漏检进入姿态惩罚。'},
      {name: 'hot3d_hand_coverage', label: 'HOT3D · ViDiHand coverage-aware', note: 'local holdout · 81 帧',
       purpose: '用同一套 coverage-aware 指标检查 HOT3D 真实第一人称场景中的遮挡、出画和跨域泛化；当前不是论文官方 test split。'},
    ],
    guide: {
      title: 'ViDiHand coverage-aware 协议',
      summary: '公开方法行和数值全部原样引用自 ViDiHand 论文 Table 1，本项目没有重新运行或修改这些论文数值；MINT 两行来自本项目 sota-fixed-v1 50% 本地记录，并以非论文同 split 标注。Table 1 同时评测手是否被检测到、检测后的 3D pose、朝向/位置和时序稳定性；漏检会使用 canonical MANO 或图像对角线等惩罚。',
      stages: [
        {label: 'Detection', text: 'FAcc、Recall、F1 衡量左右手 presence；固定左右槽位，不做跨手匹配。'},
        {label: '3D Pose', text: 'MPJPE-p 与 PA-MPJPE-p 对所有屏幕内 GT 手统计，包括漏检惩罚。'},
        {label: '位置与时序', text: 'GO-p、CT-p 分别衡量全局朝向和相机系平移；Jitter 衡量连续检出轨迹的二阶差分。'},
      ],
      datasets: [
        {label: 'ARCTIC', text: 'ViDiHand 的训练使用了 ARCTIC；表中 ViDiHand 数值仅作为同域参考，不表示在未见数据上的泛化能力。本地使用 protocol P1 validation 81 帧片段。'},
        {label: 'HOT3D', text: 'ViDiHand 的训练使用了 HOT3D；表中 ViDiHand 数值仅作为同域参考，不表示在未见数据上的泛化能力。本地使用确定性的 local holdout 81 帧片段。'},
      ],
      warning: 'InterWild 至 ViDiHand 的固定行均为论文引用数据；MINT 两行是 sota-fixed-v1 50% 本地固定记录，与论文不是同一 split，不能视为严格同协议排名。用户启动评测后追加的“本次模型”行同样来自本地运行。ViDiHand 使用 HOT3D 和 ARCTIC 训练，因此其论文指标仅供同域参考，不具备跨数据集泛化意义。',
    },
    combos: [
      ['hands_coverage', 'arctic_hand_coverage'],
      ['hands_coverage', 'hot3d_hand_coverage'],
    ],
    liveDatasets: ['arctic_hand_coverage', 'hot3d_hand_coverage'],
    liveMetricAliases: {
      FAcc: 'FAcc', Recall: 'Recall', F1: 'F1',
      'MPJPE-p': 'MPJPE-p', 'PA-MPJPE-p': 'PA-MPJPE-p',
      'GO-p': 'GO-p', 'CT-p': 'CT-p', Jitter: 'Jitter',
    },
    liveMetricUnits: {
      FAcc: '', Recall: '', F1: '', 'MPJPE-p': 'mm', 'PA-MPJPE-p': 'mm',
      'GO-p': 'deg', 'CT-p': 'm', Jitter: 'mm/frame2',
    },
    metrics: [
      {id: 'FAcc', label: 'FAcc', direction: 'max', unit: '', detail: {
        name: 'Frame Accuracy',
        definition: '在至少有一只屏幕内 GT 手的帧上，只有左右手 presence 全部判断正确（无 FN 且无 FP）才记为正确帧。',
        alignment: '固定 left/right 槽位进行二分类，不允许左右手互换匹配。',
        protocol: '属于 ViDiHand Table 1 的 Detection 指标，越高越好。',
        local: '本地以 hand_presence_logits >= 0 判为存在，并按整个 corpus 累加正确帧数后计算。'}},
      {id: 'Recall', label: 'Recall', direction: 'max', unit: '', detail: {
        name: 'Hand Detection Recall',
        definition: '屏幕内 GT 手中被正确检出的比例：TP / (TP + FN)。',
        alignment: '按左右手固定槽位统计；完全出画的 GT 手不进入 presence GT。',
        protocol: '属于 Detection 指标，漏检直接降低 Recall，并进一步进入带 -p 的 pose 惩罚。',
        local: '本地在全部序列上先累计 TP/FN，再计算 corpus-level Recall。'}},
      {id: 'F1', label: 'F1', direction: 'max', unit: '', detail: {
        name: 'Hand Detection F1',
        definition: '检测 Precision 与 Recall 的调和平均，兼顾误检和漏检。',
        alignment: 'Precision = TP/(TP+FP)，Recall = TP/(TP+FN)，固定左右手槽位。',
        protocol: '属于 Detection 指标，越高越好。',
        local: '本地从整个 corpus 汇总的 TP、FP、FN 计算，不平均每段 F1。'}},
      {id: 'MPJPE-p', label: 'MPJPE-p', direction: 'min', unit: 'mm', detail: {
        name: 'Coverage-aware Penalized MPJPE',
        definition: '检出手计算腕部 root-relative 21 关节 3D 误差；漏检手以 canonical MANO 与 GT 的 root-relative MPJPE 作为惩罚。',
        alignment: '检出手只去腕部平移，不做旋转或尺度对齐。',
        protocol: '所有屏幕内 GT 手（TP + FN）都进入分母，因此低 detection coverage 不会得到虚假的低 pose error。',
        local: '本地按累计误差和 / pose hand 数计算 corpus-level mm。'}},
      {id: 'PA-MPJPE-p', label: 'PA-MPJPE-p', direction: 'min', unit: 'mm', detail: {
        name: 'Coverage-aware Penalized PA-MPJPE',
        definition: '检出手逐帧做 7-DoF Procrustes 后计算 MPJPE；漏检手沿用论文定义的 raw canonical-MANO MPJPE 惩罚。',
        alignment: 'TP 使用旋转、平移、尺度对齐；FN 没有预测可对齐，因此使用固定 canonical penalty。',
        protocol: '所有屏幕内 GT 手（TP + FN）进入分母，越低越好。',
        local: '本地实现显式保留 FN 惩罚，不能与只在检出帧算 PA-MPJPE 的普通协议比较。'}},
      {id: 'GO-p', label: 'GO-p', direction: 'min', unit: 'deg', detail: {
        name: 'Coverage-aware Global Orientation Error',
        definition: '检出手计算预测与 GT 手部全局旋转矩阵之间的测地线角；漏检使用 canonical orientation 惩罚。',
        alignment: '不额外对齐全局旋转，保留相机系朝向误差。',
        protocol: '对所有屏幕内 GT 手汇总，单位 degree，越低越好。',
        local: '本地从 MANO global orientation 计算 SO(3) geodesic angle。'}},
      {id: 'CT-p', label: 'CT-p', direction: 'min', unit: 'm', detail: {
        name: 'Coverage-aware Camera Translation Error',
        definition: '检出手计算预测与 GT 的相机坐标系 MANO 平移 L2；漏检使用 GT 平移到 canonical 原点的距离作为惩罚。',
        alignment: '不做平移或尺度拟合，保留绝对相机系位置误差。',
        protocol: '对所有屏幕内 GT 手汇总，单位 m，越低越好。',
        local: '本地直接使用模型输出的相机系 hand translation。'}},
      {id: 'Jitter', label: 'Jitter', direction: 'min', unit: 'mm/frame2', detail: {
        name: 'Temporal Joint Jitter',
        definition: '在连续正确检出的手轨迹上，对预测 3D 关节做逐帧二阶差分并取关节平均幅值。',
        alignment: '只在至少 3 帧的连续 TP track 内计算，避免跨漏检断点制造伪抖动。',
        protocol: '属于 Temporal 指标，单位 mm/frame²，越低越平滑。',
        local: '本地按整个 corpus 累计二阶差分样本；它不是乘 fps² 的 HaWoR Accel。'}},
    ],
    tables: [
      {
        id: 'vidihand-arctic',
        title: 'ARCTIC',
        source: '论文引用：ViDiHand Table 1；MINT：sota-fixed-v1 50%',
        note: 'InterWild 至 ViDiHand 的固定方法行和数值直接引用自 ViDiHand 论文，未由本项目重新实测或修改。MINT 两行为本项目本地固定记录，与论文不是同一 split。ViDiHand 使用了 ARCTIC 训练，其论文指标仅供同域参考，不具备跨数据集泛化意义；用户实际运行的结果会作为“本次模型”行单独追加。',
        columns: VIDIHAND_COLUMNS,
        rows: VIDIHAND_ARCTIC_ROWS,
        liveRows: [
          {label: '本项目实测 · 本次模型（非论文引用值）', head: 'hands_coverage', dataset: 'arctic_hand_coverage', group: 'overall', requiresSameSplit: true,
           values: VIDIHAND_LIVE_VALUES},
        ],
      },
      {
        id: 'vidihand-hot3d',
        title: 'HOT3D',
        source: '论文引用：ViDiHand Table 1；MINT：sota-fixed-v1 50%',
        note: 'InterWild 至 ViDiHand 的固定方法行和数值直接引用自 ViDiHand 论文，未由本项目重新实测或修改。MINT 两行为本项目本地固定记录，与论文不是同一 split。ViDiHand 使用了 HOT3D 训练，其论文指标仅供同域参考，不具备跨数据集泛化意义；用户实际运行的结果会作为“本次模型”行单独追加。',
        columns: VIDIHAND_COLUMNS,
        rows: VIDIHAND_HOT3D_ROWS,
        liveRows: [
          {label: '本项目实测 · 本次模型（非论文引用值）', head: 'hands_coverage', dataset: 'hot3d_hand_coverage', group: 'overall', requiresSameSplit: true,
           values: VIDIHAND_LIVE_VALUES},
        ],
      },
    ],
  },
});
