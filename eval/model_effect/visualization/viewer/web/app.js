const $ = s => document.querySelector(s);
const epSel = $('#ep'), modeSel = $('#mode'), camModeSel = $('#camMode'), info = $('#info');
const PERSISTENT_PANELS = Object.fromEntries(
  ['metricsWrap', 'batchWrap', 'benchWrap', 'ldWrap'].concat(['analysisWrap'])
    .map(id => [id, document.getElementById(id)]));

// 3D 坐标轴颜色图例（与 renderScene 画的三轴一致）：X红 / Y绿 / Z蓝。
const AXES = '<span class="axkey">轴 <b style="color:#ff6b6b">X</b> <b style="color:#51cf66">Y</b> <b style="color:#5c9dff">Z</b></span>';
// 2D 展示端到端 GT vs PRED；3D 同时保留固定世界整体运动与当前相机系手部。
const PANELS = [
  {id:'both_2d', name:'整体·2D',     kind:'video', content:'both'},
  {id:'world_motion_3d', name:'固定世界·3D', kind:'scene', content:'worldmotion'},
  {id:'current_camera_3d', name:'当前相机·3D', kind:'scene', content:'hand'},
  {id:'mujoco_3d', name:'MuJoCo·仿真', kind:'mujoco'},
  {id:'wuji_retarget_3d', name:'Wuji Hand·Retargeting', kind:'retarget'},
  {id:'nums',     name:'逐帧数值（世界系）', kind:'nums', frame:'world'},
  {id:'nums_cam', name:'逐帧数值（相机系）', kind:'nums', frame:'cam'},
  {id:'loss',    name:'逐帧 loss',   kind:'loss'},
  // 工具面板(全局,不随 episode/loaded)：持久 DOM 节点(index.html 里的 wrapId),buildPanels 按序搬移,
  // 与 loss/metricsWrap 同范式。默认隐藏(见 state.hidden),顶栏按钮或调节栏 👁 打开。
  {id:'analysis', name:'数据集构成分析',       kind:'tool', wrapId:'analysisWrap'},
  {id:'batch',   name:'批量视频推理',        kind:'tool', wrapId:'batchWrap'},
  {id:'bench',   name:'🏁 Benchmark 测评',  kind:'tool', wrapId:'benchWrap'},
  {id:'logdiff', name:'📋 配置/代码变更',   kind:'tool', wrapId:'ldWrap'},
];
const PANEL_BY = Object.fromEntries(PANELS.map(p => [p.id, p]));
const DEFAULT_ORDER = PANELS.map(p => p.id);
const EXPORT_SOURCES = [
  {id:'both_2d', label:'原视频渲染', file:'source_2d'},
  {id:'world_motion_3d', label:'固定世界 · 3D', file:'fixed_world'},
  {id:'mujoco_3d', label:'MuJoCo · 仿真', file:'mujoco'},
  {id:'wuji_retarget_3d', label:'Wuji Hand · Retargeting', file:'wuji_hand_1'},
];
const FRAME_CAPTURE_SOURCES = [
  {id:'both_2d', label:'整体 · 2D', file:'both_2d'},
  {id:'world_motion_3d', label:'固定世界 · 3D', file:'world_3d'},
  {id:'current_camera_3d', label:'当前相机 · 3D', file:'camera_3d'},
  {id:'mujoco_3d', label:'MuJoCo · 仿真', file:'mujoco_3d'},
  {id:'wuji_retarget_3d', label:'Wuji Hand · Retargeting', file:'wuji_retarget_3d'},
];
const MUJOCO_RENDER_TAG = 'shared_wuji_camera_v13_canonical_origin';
const RETARGET_RENDER_TAG = 'shared_wuji_camera_v13_no_presence_hud';
// 固定世界默认 3/4 俯视（弧度）：正对首帧相机光轴的正视图读不出纵深。
// 与 render/fixed_world_video.py 的 DEFAULT_AZ/DEFAULT_EL 必须一致，否则网页与导出视频不同视角。
const VIEW_AZ0 = -0.61, VIEW_EL0 = 0.31;
const newView = () => ({az:VIEW_AZ0, el:VIEW_EL0, zoom:1, panX:0, panY:0, drag:null});
const MAX_TRAJECTORY_POINTS = 1000;                  // 轨迹像素级降采样上限，避免长视频逐帧画数万点
const _sceneExtentCache = new WeakMap();             // payload 不变：完整包围盒只计算一次
const _sceneGroundCache = new WeakMap();             // 同理：地面高度(沿 up 的最低点)只算一次

let masterVideo = null;                              // 首个可见块的视频（带控件，驱动当前帧）
let _drawPending = true;
let _masterVideoCleanup = null;
let _videoFrameDriven = false;
let _fallbackFrame = -1;
let state = { eid:0, fps:30, mode:'mesh_skel', nframes:1,
              camMode:'max_chunked',                 // chunked / max_chunked / streaming / exact full
              fullMaxFrames:null,                    // 模型加载后由后端返回当前主卡的 exact full 安全上限
              handMode:'smooth',                     // hard / blend / smooth(blend + UKF/RTS 后处理)
              ukf:{q:0.7,r:0.5,beta:0.3},            // 轻于生产默认 0.6/0.6/2.0；仅 smooth 生效
              gtBetas:'per_frame', predBetas:'per_frame', predFov:'per_frame',  // 手形/内参 每帧 vs 整段平均
              gt:null, pred:null, metrics:null, metricsError:null, nums:null,
                                                        // nums=逐帧数值(每块下方面板)
              layout:'side',                         // 生效布局：'side'(默认，GT/PRED 左右并排) | 'overlay'(同画面叠加)
              layoutPref:'side',                     // 用户点选的布局偏好；裸视频被迫 overlay 时不改它
              order:DEFAULT_ORDER.slice(),           // 模块显示顺序（右侧调节栏拖拽可改）
              hidden:new Set(['mujoco_3d','wuji_retarget_3d','batch','bench','logdiff']),
                                                        // 仿真/重定向与工具面板均按需打开
              followCam:false, showTraj:true, showCamHand:true,
              worldCoordMode:'z_up',                 // 固定世界默认 X右 / Y前 / Z上；可切回 OpenCV
              views:{},                              // 模块 id -> {vov,vgt,vpred}（跨重建保留视角）
              panels:{},                             // 模块 id -> panel obj
              exportSelection:new Set([
                'both_2d','world_motion_3d','mujoco_3d','wuji_retarget_3d',
              ]),
              exporting:false, exportProgress:0,
              capturingFrame:false, frameCaptureMessage:'',
              comparisonSnapshot:null,               // 进入仅原始 GT 前保存当前 GT/PRED 对照，供再次点击恢复
              no_truth:false,                        // 无真值裸视频模式（仅预测，无 GT/loss）
              rawOnly:false,                         // 仅原始 GT（不跑推理，只看原数据）
              vidName:null, epIdx:null, epOrdinal:null, epTotal:null, sourceName:null, sourcePath:null,
              loaded:false, loading:false, stopped:false, cancelling:false,
              // loaded=已出结果；loading=活动请求；cancelling=停止请求已发出，避免重复提交
              inferenceDirty:true,                  // 输入/推理参数已改，须显式点「开始推理」才能应用
              modelReady:false,                      // 模型是否加载完毕（未就绪时禁用「推理」）
              modelLoading:false,                    // 模型是否正在后台加载（加载中禁止推理）
              modelDevices:[],                       // 加载模型时发现并常驻副本的推理设备
              analysis:{running:false, stage:'idle', phase:'待分析', input_dir:'', workers:32, analysis_type:'video',
                        diversity_root:'', diversity_choices:[], input_dirs:[], selected_datasets:[], sample_files:24,
                        discovered:0, total:0, done:0, failed:0, current:'', error:null,
                        cancelled:false, cached:false, result_ready:false, summary:null,
                        page:1, page_size:100, pages:1, page_total:0, rows:[], filters:null,
                        search:'', anomaly:'', codec:'', resolution:'', orientation:'',
                        sort:'relative_path', order:'asc'},
              batch:{running:false, phase:'', input_root:'', output_root:'', name_template:'{stem}_pred',
                     checkpoint:null, checkpoint_path:null,
                     total:0, done:0, succeeded:0, failed:0, skipped:0, current:null, current_index:0,
                     file_done:0, file_total:0, devices:[], workers:2, active:[], log:[], error:null, manifest:null,
                     cancelled:false, progress:0},
              bench:{running:false, phase:'', count:0, report:null, live_report:null, error:null, out:null, log:[], progress:{}, caps:null, sizes:null, gpus:null,
                     backend:'local', job:null, job_id:null, job_status:null, aliyun:null, aliyunDefaultsLoaded:false,
                     suite:'sota', metricSelections:{}, metricDetailOpen:null, comparisonGroup:'overall', liveUpdatedAt:null,
                     samplePreset:'half', autoUkf:true,
                     tableExportSelection:new Set(['hot3d-camera','arctic-camera','vidihand-arctic','vidihand-hot3d']), tableExporting:false,
                     selection:{}, selectedModels:[], modelRuns:[], modelResults:[], activeModel:null,
                     historyTier:'half', historyOptions:[], historyResults:[], historyLoading:false, historyRoot:'', historyError:null,
                     modelOptionsLoaded:false},
              logdiff:{running:false, phase:'', result:null, error:null, log:[], runs:[], scopes:[], scope:'', loaded:false} };
state.hidden.add('analysis');                         // 数据集分析同其他工具面板一样默认收起

// 相对当前页目录拼 URL：兼容被反向代理到子路径的场景（如 code-server 的 /proxy/8011/），
// 也兼容普通 http://localhost:8011/。绝对 '/api/..' 在子路径反代下会打到代理根而 404，故一律相对化。
const API_BASE = document.baseURI.replace(/[^/]*$/, '');   // 当前页所在目录（带尾斜杠）
const U = p => API_BASE + String(p).replace(/^\//, '');
async function getJSON(u){ const r = await fetch(U(u)); if(!r.ok) throw new Error(`${r.status} ${await r.text()}`); return r.json(); }

let EP_TOTAL = 0, CKPT_TAG = '';
const clampEp = v => EP_TOTAL ? Math.min(EP_TOTAL-1, Math.max(0, Math.floor(+v||0))) : 0;
function effectiveHandMode(){
  if(state.handMode!=='smooth') return state.handMode;
  const fmt=value=>Number(value).toFixed(6).replace(/0+$/,'').replace(/\.$/,'');
  return `smooth@${fmt(state.ukf.q)},${fmt(state.ukf.r)},${fmt(state.ukf.beta)}`;
}
function appendHandMode(params){ params.set('hand_mode',effectiveHandMode()); return params; }
function syncFullLimit(value){
  const limit = Math.max(0, Math.floor(+value||0));
  if(limit) state.fullMaxFrames = limit;
  const suffix = state.fullMaxFrames ? `（${state.fullMaxFrames}帧）` : '';
  for(const sel of [camModeSel, $('#batchCam')]){
    if(!sel) continue;
    const full = sel.querySelector('option[value="full"]');
    const maxChunked = sel.querySelector('option[value="max_chunked"]');
    if(full) full.textContent = `整段 · Full${state.fullMaxFrames ? `（≤${state.fullMaxFrames}帧）` : ''}`;
    if(maxChunked) maxChunked.textContent = `最大窗分窗 · Max Chunked${suffix}`;
  }
}
function markPending(msg){ info.textContent = msg || '已选择，点 [⬇ 加载模型] 载入后 [▶ 开始推理]'; }
function setRawOnly(on){
  state.rawOnly=Boolean(on);
  const button=$('#loadRawBtn'); if(!button) return;
  button.textContent=state.rawOnly?'返回 GT 对照':'仅看原始 GT';
  button.classList.toggle('on',state.rawOnly);
  button.setAttribute('aria-pressed',state.rawOnly?'true':'false');
}
function queueInference(msg){
  if(state.loading){ info.textContent = '推理正在运行，请先停止再修改输入或推理参数'; return false; }
  state.loaded = false; setRawOnly(false); state.comparisonSnapshot=null; state.inferenceDirty = true;
  state.gt = state.pred = state.metrics = state.nums = null;
  state.metricsError = null;
  state.epIdx = state.epOrdinal = state.epTotal = null;
  state.sourceName = state.sourcePath = null;
  buildPanels(false);
  setStep(state.modelReady ? '○ 设置待应用' : '○ 模型未加载', state.modelReady ? 'warn' : '');
  hideProg();
  markPending(msg || (state.modelReady
    ? '设置已更新，点击 [▶ 开始推理] 才会运行模型'
    : '设置已更新，先加载模型，再点击 [▶ 开始推理]'));
  return true;
}
// 按钮旁的「当前步骤」指示（busy黄/ok绿/warn橙/err红）。
const setStep = (txt, cls='') => { const el=$('#stepInfo'); if(el){ el.textContent=txt; el.className='step '+cls; } };
// 按钮旁进度条：推理(infer 有 done/total)→确定进度；其余阶段→不确定动画。
function setProg(p){
  const w=$('#progWrap'), b=$('#progBar'); if(!w||!b) return;
  w.style.display='inline-block';
  const st=p&&p.stage, tot=+(p&&p.total)||0, dn=+(p&&p.done)||0;
  if(tot>0){ w.classList.remove('indet'); b.style.width=Math.min(100,Math.round(dn/tot*100))+'%'; }  // load/infer 有 done/total → 确定进度
  else if(st==='done'){ w.classList.remove('indet'); b.style.width='100%'; }
  else { w.classList.add('indet'); }                 // model/render3d/未知 → 不确定动画
}
function hideProg(){ if(state.loading) return; const w=$('#progWrap'); if(w){ w.classList.remove('indet'); w.style.display='none'; $('#progBar').style.width='0%'; } }
// 加载与推理拆开：「加载模型」在未加载且非加载中时可点；「推理」仅模型就绪时可点，
// 推理进行中(=「停止」)恒可点。加载中禁止推理。
function updateLoadBtn(){
  const mbtn = $('#loadModelBtn'), btn = $('#loadBtn');
  if(mbtn){
    mbtn.disabled = state.modelReady || state.modelLoading;
    mbtn.textContent = state.modelLoading ? '⏳ 加载模型中…'
                     : (state.modelReady ? '✓ 模型已加载' : '⬇ 加载模型');
  }
  if(btn){
    btn.disabled = state.cancelling || (!state.loading && !state.modelReady);
    btn.title = state.modelReady ? '显式运行当前输入和设置；其他选择操作不会自动推理'
                                 : '请先点 [⬇ 加载模型] 载入模型后再推理';
  }
  for(const id of ['#camMode','#gtBetas','#predBetas','#predFov','#ukfQ','#ukfR','#ukfBeta']){
    const el=$(id); if(el) el.disabled=state.loading;
  }
  document.querySelectorAll('#handModeSeg button').forEach(el=>{ el.disabled=state.loading; });
  if(typeof renderBatch === 'function') renderBatch();
}
// 轮询模型状态：区分 未加载/加载中/就绪 更新按钮态与提示。就绪即停（点加载 / 换 ckpt 会重启轮询）。
async function pollModelReady(){
  try{ const {ready, loading, devices, full_max_frames} = await getJSON('/api/model_ready');
    state.modelDevices = Array.isArray(devices) ? devices : [];
    syncFullLimit(full_max_frames);
    state.modelReady = !!ready; state.modelLoading = !ready && !!loading; updateLoadBtn();
    if(ready){ if(!state.loading){
      const gpu = state.modelDevices.length ? ` · ${state.modelDevices.length}卡 ${state.modelDevices.join(',')}` : '';
      setStep('✓ 模型就绪'+gpu,'ok'); hideProg();
    } return; }
    if(!state.loading){
      if(loading){ setStep('⏳ 模型加载中…','warn'); setProg({stage:'model'}); }
      else { setStep('○ 模型未加载（点 [⬇ 加载模型] 载入）',''); hideProg(); }
    }
  }catch(e){}
  setTimeout(pollModelReady, 1500);
}

const _rel = (base, name) => base ? base + '/' + name : name;
const _enc = p => p.split('/').map(encodeURIComponent).join('/');   // 逐段编码,保留 '/' 给 <path:...>

let ckptBrowser = null;
let ckptCur = '';
let curDatasetRel = null;     // 当前浏览目录若是 lerobot 数据集则记其**绝对**路径，否则 null（普通目录=视频项）
let _dsScanToken = 0;         // 数据集枚举轮询代次：每次进目录 +1，失效上一目录仍在跑的轮询（防旧进度回填）

// 按当前加载/选择项切换「模式相关」UI：无真值(裸视频)隐藏 布局/说明/仅看原始，固定 overlay；有真值全开并恢复偏好。
function applyModeUI(nt){
  state.no_truth = nt;
  state.layout = nt ? 'overlay' : state.layoutPref;
  const rawButton=$('#loadRawBtn'); if(rawButton) rawButton.style.display=nt?'none':'';
  const mw = $('#metricsWrap'); if(mw && nt) mw.style.display = 'none';   // 无真值无 loss；有真值时 loadEpisode 里按需显隐
}

// 进入某目录后按其类型切换顶栏：lerobot→显示 Episode 选择器(记数据集 rel)；普通目录→隐藏，靠点视频文件选中输入。
// lerobot 枚举**异步**：已就绪直接放出选择器；未就绪显示进度条并轮询，枚举完再放出（不阻塞浏览）。
function onInputDir(info, cur){
  const token = ++_dsScanToken;              // 进目录即翻代次：失效上一目录仍在跑的枚举轮询
  if(info && info.lerobot){
    curDatasetRel = cur; applyModeUI(false);
    $('#epWrap').style.display = '';
    if(info.ready){ setDatasetReady(info.episodes || 0); }
    else { setDatasetScanning(); pollDatasetScan(cur, token); }
  }else{
    curDatasetRel = null;
    $('#epWrap').style.display = 'none';
    const hasVid = info && info.videos && info.videos.length;
    markPending(hasVid ? '点击视频只会选中；确认后由 [▶ 开始推理] 显式运行模型'
                       : '点目录进入：到 lerobot 目录（含 meta/info.json）选 episode，或普通目录点视频文件');
  }
}
// ── 数据集 episode 枚举：就绪/枚举中 UI + 轮询进度 ──
function _epCtlDisabled(dis){ for(const id of ['#ep','#epPrev','#epNext']){ const el=$(id); if(el) el.disabled=dis; } }
function setDatasetReady(total){
  EP_TOTAL = total; epSel.max = Math.max(0, EP_TOTAL - 1);
  $('#epTotal').textContent = `/ 共 ${EP_TOTAL} 个（序号 0 ~ ${Math.max(0, EP_TOTAL-1)}）`;
  epSel.value = clampEp(epSel.value); _epCtlDisabled(false);
  const s=$('#epScan'); if(s) s.style.display='none';
  const sp=$('#epScanProg'); if(sp){ sp.style.display='none'; sp.classList.remove('indet'); }
  markPending(EP_TOTAL > 0 ? '选择 Episode 后点击 [▶ 开始推理]（或随时 [仅看原始 GT]）'
                           : '该 lerobot 数据集无 episode');
}
function setDatasetScanning(){
  EP_TOTAL = 0; $('#epTotal').textContent = ''; _epCtlDisabled(true);   // 枚举中禁用 Episode 选择
  const s=$('#epScan'); if(s){ s.style.display=''; s.className='step busy'; s.textContent='⏳ 枚举 episode 中…'; }
  const sp=$('#epScanProg'); if(sp){ sp.style.display='inline-block'; sp.classList.add('indet'); }
  markPending('正在枚举该数据集的 episode…（大数据集稍候，完成后再选序号）');
}
const _scanStageText = st => st.stage==='read' ? '读元数据'
                          : st.stage==='resolve' ? '校验文件'
                          : st.stage==='build' ? '汇总' : (st.stage||'');
async function pollDatasetScan(dsAbs, token){
  while(token === _dsScanToken){
    let st = null;
    try{ st = await getJSON('/api/dataset_progress?path=' + _enc(dsAbs)); }catch(e){}
    if(token !== _dsScanToken) return;          // 期间切了目录 → 停，不回填旧进度
    if(st){
      if(st.error){
        const s=$('#epScan'); if(s){ s.className='step err'; s.textContent='✗ 枚举失败：'+st.error; }
        const sp=$('#epScanProg'); if(sp){ sp.style.display='none'; sp.classList.remove('indet'); }
        markPending('该数据集枚举失败：'+st.error); return;
      }
      if(st.ready){ setDatasetReady(st.episodes || 0); return; }
      const s=$('#epScan'); if(s){ s.className='step busy';
        s.textContent = '⏳ 枚举 episode… '+_scanStageText(st)+(st.total>0?` ${st.done}/${st.total}`:''); }
      const sp=$('#epScanProg'), bar=sp&&sp.querySelector('.progbar');
      if(sp){ sp.style.display='inline-block';
        if(st.total>0){ sp.classList.remove('indet'); if(bar) bar.style.width=Math.min(100,Math.round(st.done/st.total*100))+'%'; }
        else sp.classList.add('indet'); }
    }
    await new Promise(r=>setTimeout(r, 500));
  }
}

// 解析当前选择 → 全局 eid：lerobot 上下文按 (数据集绝对路径, episode 序号) 现取；视频项用点选时已登记的 eid。
async function resolveEid(){
  if(curDatasetRel !== null){
    const ep = clampEp(epSel.value);
    const {eid} = await getJSON('/api/lerobot_eid?path=' + _enc(curDatasetRel) + '&ep=' + ep);
    return eid;
  }
  return state.eid;
}

// 输入目录浏览器（**绝对路径**，可上溯任意系统路径）：面包屑各段可点回退 +「⬆ 上一级」；
// 到 lerobot 目录经 onInputDir 出 Episode 选择器，普通目录列视频文件；点击只选中，不触发推理。
let inCur = '';
let _inBrowseToken = 0;       // 目录请求代次：快速切目录时只接受最后一次响应，防旧目录晚到后覆盖新目录
async function inGo(abs){
  const token = ++_inBrowseToken;
  const crumb = $('#inCrumb'), list = $('#inList');
  list.innerHTML = '<span class="bmut">加载…</span>';
  let b;
  try{ b = await getJSON('/api/ibrowse?path=' + _enc(abs || '')); }
  catch(e){ if(token === _inBrowseToken) list.innerHTML = '<span class="bmut">读取失败</span>'; return; }
  if(token !== _inBrowseToken) return;
  const nextPath = b.path || abs || '';
  if(inCur && nextPath !== inCur && state.loaded){
    // 输入上下文已改变，旧画面不能继续冒充当前目录；模型仍保留就绪，可直接对新选择推理。
    state.loaded = false; state.gt = state.pred = state.metrics = state.nums = null;
    state.metricsError = null;
    state.epIdx = state.epOrdinal = state.epTotal = null;
    state.sourceName = state.sourcePath = null; state.vidName = null;
    info.title = '';
    buildPanels(false);
  }
  inCur = nextPath;
  // 面包屑：/ 根 + 各段（点任意段回退到该绝对路径）
  const segs = inCur.split('/').filter(Boolean);
  let acc = '', html = '<b data-abs="/">/</b>';
  for(const s of segs){ acc += '/' + s; html += `<b data-abs="${acc}">${s}</b><span class="bsep">/</span>`; }
  crumb.innerHTML = html;
  crumb.querySelectorAll('b').forEach(el => el.onclick = ()=> inGo(el.dataset.abs));
  onInputDir(b, inCur);      // 切顶栏上下文（lerobot→Episode 选择器 / 普通→点视频）
  // 「⬆ 上一级」固定在面包屑行右侧（不进列表，故不随文件名长短/滚动移动，避免误点）
  const upBtn = $('#inUp');
  if(b.parent && b.parent !== inCur){ upBtn.hidden = false; upBtn.onclick = ()=> inGo(b.parent); }
  else upBtn.hidden = true;
  // 列表：子目录 + 视频文件
  const dirs = (b.dirs||[]).map(d=>`<button class="bdir" data-dir="${d}">📁 ${d}</button>`).join('');
  const vids = (b.videos||[]).map(f=>`<button class="bleaf" data-eid="${f.eid}" data-name="${f.name}">${f.name}</button>`).join('');
  const trunc = b.truncated ? '<span class="bmut">⚠ 目录过大，仅列前 2000 项（请进更深子目录缩小范围）</span>' : '';
  list.innerHTML = trunc + (dirs + vids || '<span class="bmut">（空）</span>');
  list.querySelectorAll('.bdir[data-dir]').forEach(el => el.onclick = ()=>
    inGo((inCur === '/' ? '' : inCur) + '/' + el.dataset.dir));
  list.querySelectorAll('.bleaf').forEach(el => el.onclick = ()=>{
    if(state.loading){ info.textContent='推理正在运行，请先停止再选择其他视频'; return; }
    list.querySelectorAll('.bleaf').forEach(x=>x.classList.remove('on')); el.classList.add('on');
    const eid = +el.dataset.eid; console.log('[btn] 选视频 '+el.dataset.name+' eid='+eid);
    applyModeUI(true); state.vidName = el.dataset.name; state.eid = eid;
    queueInference(state.modelReady
      ? '已选视频「'+el.dataset.name+'」，点击 [▶ 开始推理] 才会运行模型'
      : '已选视频「'+el.dataset.name+'」，先加载模型，再点击 [▶ 开始推理]');
  });
}

const _joinAbs = (base,name) => base==='/' ? '/'+name : base.replace(/\/+$/,'')+'/'+name;

function renderCkptCrumb(path){
  const crumb=$('#ckptCrumb'); if(!crumb) return;
  const parts=String(path||'/').split('/').filter(Boolean);
  let acc='', html='<b data-path="/">/</b>';
  for(const part of parts){
    acc += '/'+part;
    html += ` <span>/</span> <b data-path="${_esc(acc)}">${_esc(part)}</b>`;
  }
  crumb.innerHTML=html;
  crumb.querySelectorAll('b').forEach(el=>el.onclick=()=>ckptGo(el.dataset.path));
}

function markCkptPath(path){
  const selected=String(path||'');
  document.querySelectorAll('#ckptList [data-ckpt-path]').forEach(el=>
    el.classList.toggle('on', el.dataset.ckptPath===selected));
}

async function ckptGo(path){
  const list=$('#ckptList'), input=$('#ckptPath');
  if(!list) return;
  list.innerHTML='<span class="bmut">加载…</span>';
  let data;
  try{ data=await getJSON('/api/ckpt/browse?path='+encodeURIComponent(path||'')); }
  catch(e){ list.innerHTML='<span class="bmut">读取失败：'+_esc(e.message)+'</span>'; return; }
  ckptCur=data.path||'/'; renderCkptCrumb(ckptCur);
  const selected=data.selected ? _joinAbs(ckptCur,data.selected) : '';
  if(input) input.value=selected||ckptCur;
  let html='';
  if(data.parent && data.parent!==ckptCur){
    html += `<button class="bup" data-up="${_esc(data.parent)}">⬆ 上一级</button>`;
  }
  if(data.selectable){
    html += `<button class="bselect" data-select-current="${_esc(ckptCur)}" data-ckpt-path="${_esc(ckptCur)}">✓ 选择当前目录</button>`;
  }
  html += (data.dirs||[]).map(name=>
    `<button class="bdir" data-dir="${_esc(name)}">📁 ${_esc(name)}</button>`).join('');
  html += (data.files||[]).map(name=>{
    const full=_joinAbs(ckptCur,name);
    return `<button class="bleaf" data-ckpt-path="${_esc(full)}">${_esc(name)}</button>`;
  }).join('');
  if(data.truncated) html += '<span class="bmut">目录过大，仅显示前 2000 项</span>';
  if(!html) html='<span class="bmut">（没有子目录或 checkpoint 权重）</span>';
  list.innerHTML=html;
  const up=list.querySelector('[data-up]'); if(up) up.onclick=()=>ckptGo(up.dataset.up);
  const current=list.querySelector('[data-select-current]');
  if(current) current.onclick=()=>switchCkptPath(current.dataset.selectCurrent);
  list.querySelectorAll('[data-dir]').forEach(el=>
    el.onclick=()=>ckptGo(_joinAbs(ckptCur,el.dataset.dir)));
  list.querySelectorAll('[data-ckpt-path]').forEach(el=>
    el.onclick=()=>switchCkptPath(el.dataset.ckptPath));
  markCkptPath(data.current||selected);
}

// 选 ckpt：校验成功后作废旧预测并置模型未就绪；不自动加载。
async function switchCkptPath(path){
  path=String(path||'').trim();
  if(!path) return;
  console.log('[btn] 选择 checkpoint → '+path);
  try{
    const r = await fetch(U('/api/ckpt'), {method:'POST', headers:{'Content-Type':'application/json'},
                                        body: JSON.stringify({path})});
    if(!r.ok) throw new Error(await r.text());
    const res = await r.json();
    const resolved=res.ckpt||path;
    state.gt = state.pred = state.metrics = null; state.metricsError = null; state.loaded = false;
    state.modelReady = false; state.modelLoading = false; state.modelDevices = [];
    updateLoadBtn(); buildPanels(false);
    CKPT_TAG = res.tag || resolved;
    const input=$('#ckptPath'); if(input) input.value=resolved;
    setStep('○ 模型未加载',''); hideProg();
    markPending(`已选择 ${resolved}，点 [⬇ 加载模型] 载入（或先 [👁 仅看原始]）`);
    await ckptGo(resolved); markCkptPath(resolved);
    pollModelReady();
  }catch(e){ info.textContent = '选择 checkpoint 失败：'+e.message; }
}

async function init(){
  // ── 全局配置：浏览根 + 默认起点（模式不再全局固定，由浏览到的目录动态判定）──
  const cfg = await getJSON('/api/episodes');
  state.analysis.diversity_root=cfg.diversity_root||'';
  state.analysis.diversity_choices=Array.isArray(cfg.diversity_paths)?cfg.diversity_paths:[];
  state.analysis.input_dirs=state.analysis.diversity_choices.map(item=>item.path).filter(Boolean);

  // ── checkpoint：可直接输入文件/目录，也可浏览任意服务器路径 ──
  const ck = await getJSON('/api/ckpts');
  CKPT_TAG = ck.current.tag || '';
  ckptBrowser={go:ckptGo, mark:markCkptPath, get cur(){return ckptCur;}};
  const initialCkpt=(ck.current&&ck.current.path)||((ck.runs||[]).length?ck.root:'');
  await ckptGo(initialCkpt); markCkptPath(initialCkpt);
  const ckptInput=$('#ckptPath'), ckptOpen=$('#ckptPathOpen'), ckptApply=$('#ckptPathApply');
  if(ckptOpen) ckptOpen.onclick=()=>ckptGo((ckptInput&&ckptInput.value)||'');
  if(ckptApply) ckptApply.onclick=()=>switchCkptPath((ckptInput&&ckptInput.value)||'');
  if(ckptInput) ckptInput.onkeydown=e=>{
    if(e.key==='Enter'){ e.preventDefault(); switchCkptPath(ckptInput.value); }
  };

  const meta = await getJSON('/api/episode/0');
  syncFullLimit(meta.full_max_frames);
  state.mode = meta.default_mode || 'mesh_skel'; modeSel.value = state.mode;
  modeSel.onchange = ()=>{ state.mode = modeSel.value; console.log('[btn] 叠加模式 → '+state.mode); if(state.loaded) buildPanels(true); };
  // 推理参数只更新选择并清空旧结果；唯一推理入口是「开始推理」按钮。
  state.camMode = meta.default_cam_mode || 'max_chunked'; camModeSel.value = state.camMode;
  camModeSel.onchange = ()=>{ state.camMode = camModeSel.value; console.log('[btn] 相机推理 → '+state.camMode);
    queueInference('相机推理模式已改为「'+camModeSel.options[camModeSel.selectedIndex].text+'」，点击 [▶ 开始推理] 应用'); };
  // 手部拼窗/后处理模式：hard、blend、blend 后生产 UKF+RTS 平滑分别独立缓存。
  state.handMode = meta.default_hand_mode || 'smooth';
  state.ukf = {...state.ukf,...(meta.default_ukf_params||{})};
  const handModeBtns = document.querySelectorAll('#handModeSeg [data-hand-mode]');
  const ukfControls=$('#ukfControls');
  const syncHandMode = ()=>{
    handModeBtns.forEach(btn => btn.classList.toggle('on', btn.dataset.handMode === state.handMode));
    if(ukfControls) ukfControls.hidden=state.handMode!=='smooth';
  };
  syncHandMode();
  handModeBtns.forEach(btn => btn.onclick = ()=>{
    const next = btn.dataset.handMode;
    if(next === state.handMode) return;
    state.handMode = next; syncHandMode(); console.log('[btn] 手部拼窗 → '+state.handMode);
    queueInference('手部拼窗模式已更新，点击 [▶ 开始推理] 应用');
  });
  const ukfFields=[['#ukfQ','q',0.1,2],['#ukfR','r',0.1,2],['#ukfBeta','beta',0,5]];
  for(const [selector,key,minimum,maximum] of ukfFields){
    const input=$(selector); if(!input) continue;
    input.value=String(state.ukf[key]);
    input.onchange=()=>{
      const parsed=Number(input.value);
      const value=Number.isFinite(parsed)?Math.max(minimum,Math.min(maximum,parsed)):state.ukf[key];
      state.ukf[key]=Math.round(value*1000)/1000; input.value=String(state.ukf[key]);
      console.log(`[btn] UKF ${key} → ${state.ukf[key]}`);
      queueInference(`UKF ${key} 已改为 ${state.ukf[key]}，点击 [▶ 开始推理] 应用`);
    };
  }
  // 手形/内参 每帧 vs 平均：只记录选项，不自动加载、不自动推理。
  const _pdef = meta.default_param_mode || 'per_frame';
  for(const [id, key] of [['#gtBetas','gtBetas'],['#predBetas','predBetas'],['#predFov','predFov']]){
    const sel = $(id); if(!sel) continue; state[key] = _pdef; sel.value = _pdef;
    sel.onchange = ()=>{ state[key] = sel.value; console.log('[btn] '+key+' → '+state[key]);
      queueInference(sel.previousElementSibling.textContent+'已改为「'+sel.options[sel.selectedIndex].text+'」，点击 [▶ 开始推理] 应用'); };
  }
  $('#loadModelBtn').onclick = onLoadModelClick;
  $('#loadBtn').onclick = onLoadClick;
  $('#loadRawBtn').onclick = onLoadRawClick;
  setRawOnly(state.rawOnly);
  $('#mujocoBtn').onclick = toggleMujocoPanel;
  $('#retargetBtn').onclick = toggleRetargetPanel;
  wireFrameCapturePicker();
  $('#exportBtn').onclick = exportVideo;
  wireExportPicker();
  updateExportButton();
  updateLoadBtn();                           // 首屏：未加载 → 禁用「推理」、放开「加载模型」
  pollModelReady();                          // 按钮旁常驻显示模型未加载/加载中/就绪 + 控制按钮可用性

  // ── 输入目录浏览器：点击视频/选择 Episode 只更新选择，不触发推理。──
  // episode 序号跳转（数据集可能几十万，用数字输入避免巨量 <option>）；仅 lerobot 上下文可见，处理器常驻。
  epSel.onchange = ()=>{ epSel.value = clampEp(epSel.value); console.log('[btn] 选 episode 序号='+epSel.value);
    queueInference('已选 Episode '+epSel.value+'，点击 [▶ 开始推理] 才会运行模型'); };
  $('#epPrev').onclick = ()=>{ epSel.value = clampEp((+epSel.value||0) - 1); console.log('[btn] 上一个 → '+epSel.value);
    queueInference('已选 Episode '+epSel.value+'，点击 [▶ 开始推理] 才会运行模型'); };
  $('#epNext').onclick = ()=>{ epSel.value = clampEp((+epSel.value||0) + 1); console.log('[btn] 下一个 → '+epSel.value);
    queueInference('已选 Episode '+epSel.value+'，点击 [▶ 开始推理] 才会运行模型'); };
  await inGo(cfg.default_path || '');
  markPending('浏览「输入」：可「⬆ 上一级」到任意路径；进 lerobot 数据集目录选 episode，或普通目录点视频文件');
  wireDatasetAnalysis();                     // 目录级视频构成分析：不依赖模型，也不自动启动
  wireBatch();                               // 批量目录推理：只绑定，不自动启动
  wireBench();                               // 顶部 Benchmark 按钮 + 结果区（独立于样例面板）
  wireLogDiff();                             // 顶部「📋 代码对比」按钮 + 结果区（独立面板）
  buildPanels(false);                        // 绑定持久 DOM 后再摘出隐藏面板，事件监听会随节点保留
  startDatasetAnalysisPoll();                // 接续显示服务端已有分析任务或最近结果
  startBatchPoll();                          // 接续显示服务端已有批量任务/结果
  startBenchPoll();                          // 拉一次 benchmark 状态（若服务端已有跑完的报告或在运行则接续显示）
  startLdPoll();                             // 接续显示服务端已有的对比结果（若在运行/已完成）
  // 不在任何选择事件中自动推理；唯一预测入口是「开始推理」。
}

function _stageText(p){
  if(!p || !p.stage) return '推理中…';
  if(p.stage==='model')    return '加载模型中（首次，稍候）…';
  if(p.stage==='load')     return p.total ? `加载数据 ${p.done}/${p.total} 帧` : '加载数据…';
  if(p.stage==='stream')   return '流式相机推理中（整段一次，较慢，无逐帧进度）…';
  if(p.stage==='full')     return '整段一次普通前向中（无分窗拼接，长视频耗时按帧数平方增长）…';
  if(p.stage==='infer')    return p.total ? `推理中 ${p.done}/${p.total} 窗` : '推理中…';
  if(p.stage==='render3d') return '解算 3D / loss…';
  if(p.stage==='done')     return '渲染中…';
  if(p.stage==='cancelling') return '正在停止…';
  if(p.stage==='cancelled')  return '已停止';
  return '推理中…';
}
// 「⬇ 加载模型」：显式启动后台加载当前选中 ckpt 的模型（不推理）。加载中禁止推理，就绪后方可推理。
async function onLoadModelClick(){
  if(state.modelReady || state.modelLoading) return;   // 已就绪/加载中 → 忽略（按钮此时本就置灰）
  console.log('[btn] 点击「加载模型」ckpt='+CKPT_TAG);
  state.modelLoading = true; updateLoadBtn();
  setStep('⏳ 模型加载中…','warn'); setProg({stage:'model'});
  info.textContent = '在全部可见 GPU 上加载模型副本中…（就绪后点击 [▶ 开始推理]）';
  try{
    const r = await fetch(U('/api/model/load'), {method:'POST'});
    if(!r.ok) throw new Error(await r.text());
  }catch(e){ state.modelLoading = false; updateLoadBtn();
    info.textContent = '启动加载失败：'+e.message; setStep('✗ 失败','err'); return; }
  pollModelReady();                                    // 轮询显示 加载中→就绪
}
// 「开始推理」是唯一允许进入预测路径的前端入口；推理中同一按钮变成停止。
function onLoadClick(){
  if(state.loading){                                   // 推理中 → 停止
    if(state.cancelling) return;
    console.log('[btn] 点击「停止」 eid='+state.eid+'（中断数据加载/推理）');
    state.cancelling = true; updateLoadBtn();
    const btn=$('#loadBtn'); if(btn) btn.textContent = '… 正在停止';
    info.textContent = '正在停止…';
    fetch(U('/api/cancel/'+state.eid), {method:'POST'}).then(r=>{
      if(!r.ok) throw new Error('HTTP '+r.status);
    }).catch(e=>{
      state.cancelling = false; updateLoadBtn();
      if(btn) btn.textContent = '■ 停止';
      info.textContent = '停止请求失败：'+e.message+'，可重试';
    });
    return;
  }
  if(!state.modelReady){                               // 防御：未就绪不推理（按钮此时本就置灰）
    info.textContent = '请先点 [⬇ 加载模型] 载入模型后再推理';
    return;
  }
  setRawOnly(false); state.comparisonSnapshot=null;     // 主按钮 = 跑推理
  console.log('[btn] 点击「推理」');
  resolveEid().then(eid => loadEpisode(eid, {explicitInference:true}))
    .catch(e => { info.textContent = '定位失败：' + (e && e.message || e); });
}
// 「仅看原始 GT」双向切换：进入前保存当前 GT/PRED；再次点击直接恢复，不重新跑模型。
function onLoadRawClick(){
  if(state.loading){                                   // 加载中不响应（用主按钮停止后再点）
    console.log('[btn] 「仅看原始(GT)」被忽略：正在加载中 eid='+state.eid+'，请先点「停止」');
    info.textContent = '正在加载中，先点 [■ 停止] 再看原始数据';
    return;
  }
  if(state.rawOnly){
    const snapshot=state.comparisonSnapshot;
    if(!snapshot){
      setRawOnly(false);
      queueInference('没有可恢复的 GT 对照，请点击 [▶ 开始推理] 生成');
      return;
    }
    Object.assign(state,snapshot);
    state.comparisonSnapshot=null;
    setRawOnly(false);
    const metrics=$('#metricsWrap'); if(metrics) metrics.style.display='';
    setStep('✓ 已返回 GT 对照','ok');
    buildPanels(false); requestDraw();
    console.log('[btn] 返回 GT/PRED 对照（复用已有结果）');
    return;
  }
  if(state.loaded&&state.pred){
    state.comparisonSnapshot={
      loaded:true, fps:state.fps, nframes:state.nframes,
      epIdx:state.epIdx, epOrdinal:state.epOrdinal, epTotal:state.epTotal,
      sourceName:state.sourceName, sourcePath:state.sourcePath,
      gt:state.gt, pred:state.pred, metrics:state.metrics,
      metricsError:state.metricsError, nums:state.nums,
      layout:state.layout, inferenceDirty:state.inferenceDirty,
    };
  }
  setRawOnly(true); state.layout='overlay';
  console.log('[btn] 点击「仅看原始(GT)」（不推理）');
  resolveEid().then(eid => loadEpisode(eid))
    .catch(e => { info.textContent = '定位失败：' + (e && e.message || e); });
}

async function loadEpisode(eid, {explicitInference=false}={}){
  if(!state.rawOnly && !explicitInference){
    queueInference('输入或设置已更新，点击 [▶ 开始推理] 才会运行模型');
    return;
  }
  const btn = $('#loadBtn'), rawBtn = $('#loadRawBtn');
  state.eid = eid; state.loading = true; state.stopped = false; state.cancelling = false;
  if(curDatasetRel !== null) _epCtlDisabled(true);             // 请求期间锁定 eid，避免结果与选择错位
  if(rawBtn) rawBtn.disabled = true;                          // 加载中禁用另一颗
  updateLoadBtn();                                            // 加载中主按钮=「停止」，保持可点
  const mw = $('#metricsWrap'); if(mw) mw.style.display = (state.rawOnly||state.no_truth)?'none':'';
  btn.classList.add('stopping'); btn.textContent = '■ 停止';   // 加载中：主按钮变「停止」（可打断）
  setStep(state.rawOnly ? '读取原始GT…' : '开始…', 'busy');    // 按钮旁：当前步骤
  setProg({stage: state.rawOnly ? 'load' : 'model'});          // 进度条起手（不确定动画）
  let polling = true, retryMode = null;                   // 超限确认后，在 finally 释放本轮状态再以最大窗重试
  const pollDone = (async ()=>{ while(polling){
    try{ const p = await getJSON('/api/progress/'+eid);
      if(polling && state.loading){ const t=_stageText(p);
        const hint = state.cancelling ? '' : '（点 [■ 停止] 可中断）';
        info.textContent = `序号 ${eid} · ${t}${hint}`; setStep(t, 'busy'); setProg(p); }
    }catch(e){}
    if(polling) await new Promise(r=>setTimeout(r, 350));
  }})();
  try{
    // raw=1 → 仅 GT；否则带推理/手部模式和手形/内参参数，后端分别缓存。
    const pq = `&gt_betas=${state.gtBetas}&pred_betas=${state.predBetas}&pred_fov=${state.predFov}`;
    const wq = state.rawOnly ? '?raw=1' :
      (`?cam_mode=${encodeURIComponent(state.camMode)}&hand_mode=${encodeURIComponent(effectiveHandMode())}${pq}`);
    const r = await fetch(U('/api/world/'+eid + wq));
    if(r.status === 409){ state.stopped = true; state.cancelling = false;
      info.textContent = '已停止'; setStep('■ 已停止', 'err'); }
    else if(r.status === 413){
      const d = await r.json(); syncFullLimit(d.full_max_frames);
      const msg = d.error || '视频超过 exact full 安全上限';
      if(d.code==='full_too_long' && window.confirm(`${msg}\n\n是否改用 ${d.full_max_frames} 帧/窗继续推理？`)){
        retryMode = d.suggested_cam_mode || 'max_chunked';
        info.textContent = `已切换为 ${d.full_max_frames} 帧/窗，准备重新推理…`;
        setStep('↻ 改用最大窗分窗', 'warn');
      }else{
        info.textContent = msg; setStep('○ Full 超限，未启动前向', 'warn');
      }
    }
    else if(!r.ok){ throw new Error(`${r.status} ${await r.text()}`); }
    else{
      setStep(state.rawOnly ? '✓ 原始GT就绪' : '✓ 完成', 'ok'); setProg({stage:'done'});
      const d = await r.json();
      state.fps = d.fps || 30;
      state.nframes = d.nframes || 1;
      state.epIdx = d.ep_idx;
      state.epOrdinal = d.episode_ordinal;
      state.epTotal = d.episode_total;
      state.sourceName = d.source_name || null;
      state.sourcePath = d.source_path || null;
      state.gt = d.gt; state.pred = d.pred; state.metrics = d.metrics || null;
      state.metricsError = d.metrics_error || null;
      state.nums = d.nums || null;
      state.loaded = true; state.inferenceDirty = false;
      _lastMF = -1;                                                          // 强制刷新误差面板
      for(const k in state.views){ const v=state.views[k];                   // 换输入复位各块视角
        for(const view of [v.vov,v.vgt,v.vpred]){
          view.az=VIEW_AZ0; view.el=VIEW_EL0; view.panX=view.panY=0;         // 复位到默认 3/4 俯视
        }
      }
      buildPanels(false);
    }
  }catch(e){ if(!state.stopped){ info.textContent = '加载失败：'+e.message; setStep('✗ 失败', 'err'); } }
  finally{ polling = false; try{ await pollDone; }catch(e){}   // 等轮询真正停下再恢复按钮
    state.loading = false; state.cancelling = false;
    btn.classList.remove('stopping'); btn.textContent = '▶ 开始推理';
    if(curDatasetRel !== null && EP_TOTAL > 0) _epCtlDisabled(false);
    if(rawBtn) rawBtn.disabled = false; updateLoadBtn();      // 恢复按钮可用性（按模型就绪与否）
    setTimeout(hideProg, 700); }                          // 成功时让 100% 闪一下再收起进度条
  if(retryMode){
    state.camMode = retryMode; camModeSel.value = retryMode;
    await loadEpisode(eid, {explicitInference:true});
  }
}

function _manualRenderBadge(pan,badge,message,buttonText='重试渲染'){
  if(!badge||pan.renderStarted) return;
  badge.classList.remove('rendering'); badge.classList.toggle('error',pan.failed);
  badge.style.display=''; badge.style.pointerEvents='auto';
  badge.innerHTML=`${message} <button type="button" class="render-video-btn">${buttonText}</button>`;
  const button=badge.querySelector('.render-video-btn');
  if(button) button.onclick=event=>{ event.preventDefault(); event.stopPropagation(); pan.startRender(true); };
}

function _loadServerVideo(pan,badge,retry){
  if(pan.loadRequested) return;
  pan.loadRequested=true;
  pan.renderStarted=true; pan.autoLoadFailed=false;
  badge.classList.remove('rendering','error');
  badge.style.display=''; badge.style.pointerEvents='none';
  badge.textContent=retry?`${pan.label}已重新提交，正在排队渲染…`:`${pan.label}正在自动排队渲染…`;
  const separator=pan.videoUrl.includes('?')?'&':'?';
  pan.video.src=pan.videoUrl+separator+'request='+Date.now();
  pan.video.load();
  updateExportButton();
}

function _wireServerVideoPanel(pan,badge,{autoStart=true}={}){
  const video=pan.video;
  pan.badge=badge;
  pan.startRender=(retry=false)=>{
    if(pan.renderDeferred){
      pan.pendingRetry=pan.pendingRetry||retry;
      badge.classList.remove('rendering','error');
      badge.style.display=''; badge.style.pointerEvents='none';
      badge.textContent=`${pan.label}等待整体·2D完成…`;
      return;
    }
    pan.failed=false; pan.renderDone=false; pan.loadRequested=false;
    _loadServerVideo(pan,badge,retry);
  };
  pan.renderDeferred=!autoStart;
  pan.releaseRender=()=>{
    if(!pan.renderDeferred) return;
    pan.renderDeferred=false;
    const retry=Boolean(pan.pendingRetry); pan.pendingRetry=false;
    pan.startRender(retry);
  };
  const onMediaReady=()=>{
    pan.renderDone=true; pan.renderStarted=false; pan.failed=false;
    pan.autoLoadFailed=false; badge.classList.remove('rendering','error');
    badge.style.display='none'; badge.style.pointerEvents='none';
    if(pan.onStatusChange) pan.onStatusChange();
    syncMujocoVideos(true); updateExportButton();
  };
  video.addEventListener('loadedmetadata',()=>{
    pan.renderDone=true;
    if(pan.onStatusChange) pan.onStatusChange();
    updateExportButton();
  });
  video.addEventListener('loadeddata',onMediaReady);
  video.addEventListener('canplay',onMediaReady);
  video.addEventListener('error',()=>{
    pan.failed=true; pan.renderStarted=false; pan.loadRequested=false;
    pan.autoLoadFailed=true;
    _manualRenderBadge(pan,badge,`${pan.label}视频加载或渲染失败`,'重试渲染');
    if(pan.onStatusChange) pan.onStatusChange();
    updateExportButton();
  });

  pan.showRenderProgress=progress=>{
    if(pan.renderDeferred){
      pan.lastRenderProgress=progress||null;
      badge.classList.remove('rendering','error');
      badge.style.display=''; badge.style.pointerEvents='none';
      badge.textContent=`${pan.label}等待整体·2D完成…`;
      return;
    }
    if(!progress){
      if(!pan.renderStarted&&!pan.renderDone){
        _manualRenderBadge(pan,badge,`${pan.label}尚未渲染`);
        updateExportButton();
      }
      return;
    }
    const total=+(progress.total??progress.frame_total)||0;
    const done=+(progress.done??progress.frame_done)||0;
    const fraction=total>0?done/total:(+progress.progress||0);
    const pct=Math.max(0,Math.min(100,Math.round(fraction*100)));
    pan.lastRenderProgress={...progress,total,done,progress:fraction};
    if(progress.stage==='error'){
      pan.failed=true; pan.renderStarted=false; pan.loadRequested=false;
      _manualRenderBadge(pan,badge,`${pan.label}渲染失败：${progress.error||'未知错误'}`,'重试渲染');
      if(pan.onStatusChange) pan.onStatusChange();
    }else if(progress.stage==='done'){
      pan.renderDone=true; pan.renderStarted=false; pan.failed=false;
      badge.classList.remove('rendering','error');
      if(!_mediaFrameReady(video)&&!pan.autoLoadFailed) _loadServerVideo(pan,badge,false);
      if(pan.onStatusChange) pan.onStatusChange();
      updateExportButton();
    }else if(total>0||progress.stage==='render'||progress.stage==='retarget'||progress.stage==='queued'){
      pan.renderStarted=true; pan.failed=false;
      badge.classList.add('rendering'); badge.classList.remove('error');
      badge.style.display=''; badge.style.pointerEvents='none';
      const frames=total>0?` · ${done}/${total} 帧`:'';
      badge.innerHTML=`${pan.label}渲染 ${pct}%${frames}<span class="vprog"><span style="width:${pct}%"></span></span>`;
      updateExportButton();
    }else if(!pan.renderStarted&&!pan.renderDone){
      _manualRenderBadge(pan,badge,`${pan.label}尚未渲染`);
      updateExportButton();
    }
  };
  (async()=>{
    while(!pan.isActive||pan.isActive()){
      try{
        const progress=await getJSON(pan.progressUrl);
        pan.showRenderProgress(progress);
      }catch(error){
        if(!pan.renderDeferred&&!pan.renderStarted&&!pan.renderDone){
          _manualRenderBadge(pan,badge,`${pan.label}状态暂时不可用`,'重试渲染');
        }
      }
      await new Promise(resolve=>setTimeout(resolve,500));
    }
  })();
  pan.startRender();
}

function _syncExportPanelProgress(sources){
  if(!sources||typeof sources!=='object') return;
  for(const [sourceId,progress] of Object.entries(sources)){
    const panel=state.panels[sourceId];
    if(panel&&typeof panel.showRenderProgress==='function') panel.showRenderProgress(progress);
  }
}

function _worldRenderQuery(){
  const params=new URLSearchParams({
    layout:state.layout, cam_mode:state.camMode, hand_mode:effectiveHandMode(),
    gt_betas:state.gtBetas, pred_betas:state.predBetas, pred_fov:state.predFov,
    coord_mode:state.worldCoordMode,
    show_traj:state.showTraj?'1':'0', show_cam_hand:state.showCamHand?'1':'0',
  });
  if(state.rawOnly) params.set('raw','1');
  const views=state.views.world_motion_3d||{};
  for(const viewName of ['vov','vgt','vpred']){
    const view=views[viewName]||newView();
    for(const field of ['az','el','zoom','panX','panY']) params.set(`${viewName}_${field}`,String(+view[field]||0));
  }
  return params.toString();
}

async function _checkWorldVideoCache(pan){
  const token=(pan.cacheCheckToken||0)+1; pan.cacheCheckToken=token;
  try{
    const progress=await getJSON(`/api/world/progress/${state.eid}?${_worldRenderQuery()}`);
    if(state.panels[pan.id]!==pan||pan.cacheCheckToken!==token) return;
    pan.renderDone=progress.stage==='done'; updateExportButton();
  }catch(error){
    if(state.panels[pan.id]===pan&&pan.cacheCheckToken===token){ pan.renderDone=false; updateExportButton(); }
  }
}

function _invalidateWorldVideo(pan){
  if(!pan||pan.content!=='worldmotion') return;
  pan.renderDone=false; pan.cacheCheckToken=(pan.cacheCheckToken||0)+1; updateExportButton();
}

// 按 state.order（跳过 hidden）渲染各模块。隐藏的机器人面板不会创建视频或请求渲染；
// 手动开启后，每种机器人方法各占一栏；有 GT 时栏内显示 GT | PRED，并等待整体·2D先完成。
// preserveTime：尽量保留主视频进度。渲染后同步右侧顺序调节栏。
function buildPanels(preserveTime){
  const cont = $('#blocks');
  const t = (preserveTime && masterVideo) ? (masterVideo.currentTime||0) : 0;
  watchMasterVideo(null);                            // 先取消旧 video 的帧回调，再销毁 DOM
  // 清空 #blocks 前先摘出持久节点(loss + 工具面板)保护:innerHTML='' 会连带销毁
  // 其子 DOM 与已绑事件;这些节点跨重建复用,摘出后按顺序再挂(隐藏则游离不挂,节点不丢)。
  const persist = {};
  for(const pid of ['metricsWrap', 'analysisWrap', 'batchWrap', 'benchWrap', 'ldWrap']){
    const el = PERSISTENT_PANELS[pid];
    if(el && el.parentNode) el.parentNode.removeChild(el);
    persist[pid] = el;
  }
  cont.innerHTML=''; state.panels={}; masterVideo=null;
  const loaded = state.loaded;
  const ov = state.layout==='overlay';
  const vids = [];
  let resolvePrimary2D;
  const primary2DReady=new Promise(resolve=>{ resolvePrimary2D=resolve; });
  let primary2DSettled=false, hasPrimary2D=false;
  const deferredRobotPanels=[];
  const settlePrimary2D=()=>{
    if(primary2DSettled) return;
    primary2DSettled=true; resolvePrimary2D();
  };
  const appendPanel = (sec,id)=>{
    cont.appendChild(sec);
  };
  for(const id of state.order){
    if(state.hidden.has(id)) continue;
    const P = PANEL_BY[id]; if(!P) continue;
    // 工具面板(bench/logdiff)与 loaded 无关,始终可渲染;数据面板(video/scene/nums/loss)需已加载 episode。
    if(!loaded && P.kind!=='tool') continue;
    const sec = document.createElement('section'); sec.className='panel'; sec.dataset.pid=id;
    if(P.kind==='video'){
      const comparisonTools=!state.no_truth&&!state.rawOnly ? `<span id="layoutWrap" class="tool-field gt-layout-controls"><label>GT 对照</label><span class="seg" id="layoutSeg">
        <button data-layout="overlay" class="${state.layout==='overlay'?'on':''}" title="在同一画面叠加 GT 与 PRED">GT/PRED 叠加</button><button data-layout="side" class="${state.layout==='side'?'on':''}" title="把 GT 与 PRED 分成左右画面">GT/PRED 并排</button></span></span>` : '';
      sec.innerHTML = `<div class="btitle"><b class="k">${P.name}</b>${comparisonTools}</div>
        <div class="vwrap"><video class="v2d" preload="auto"></video><div class="vbadge">2D 渲染中…</div></div>`;
      wireLayoutControls(sec);
      appendPanel(sec,id);
      const pan = { id, kind:'video', content:P.content, video:sec.querySelector('.v2d') };
      const badge = sec.querySelector('.vbadge'); let vdone=false;
      if(id==='both_2d') hasPrimary2D=true;
      pan.video.addEventListener('loadeddata', ()=>{
        vdone=true; if(id==='both_2d') settlePrimary2D();
        if(badge) badge.style.display='none'; requestDraw(); updateExportButton();
      });
      pan.video.addEventListener('error', ()=>{
        vdone=true; if(id==='both_2d') settlePrimary2D();
        if(badge){ badge.textContent='2D 渲染失败'; badge.style.display=''; } updateExportButton();
      });
      pan.video.controls = true;
      const rawq = state.rawOnly ? '&raw=1' : '';
      const camq = state.rawOnly ? '' :
        `&cam_mode=${encodeURIComponent(state.camMode)}&hand_mode=${encodeURIComponent(effectiveHandMode())}&gt_betas=${state.gtBetas}&pred_betas=${state.predBetas}&pred_fov=${state.predFov}`;
      const bust = state.rawOnly ? 'gt'
        : encodeURIComponent((CKPT_TAG||'x')+':'+state.camMode+':'+effectiveHandMode()+':'+state.gtBetas+state.predBetas+state.predFov);
      const vq = `mode=${encodeURIComponent(state.mode)}&layout=${state.layout}&content=${P.content}${rawq}${camq}`;
      pan.video.src = U(`/video/${state.eid}?${vq}&_=${bust}`);
      pan.video.load();
      state.panels[id] = pan;
      const eidNow = state.eid;
      (async ()=>{ while(!vdone && state.panels[id]===pan){
        try{ const p = await getJSON('/api/progress2d/'+eidNow+'?'+vq);
          if(!vdone && badge){
            if(p && p.total){ const pct=Math.min(100,Math.round(p.done/p.total*100));
              badge.innerHTML = `2D 渲染 ${pct}% <span class="vprog"><span style="width:${pct}%"></span></span>`; }
            else badge.textContent='2D 排队渲染…';
          }
        }catch(e){}
        await new Promise(r=>setTimeout(r,400));
      }})();
      if(preserveTime && t>0){ const vd=pan.video;
        const onm=()=>{ try{ vd.currentTime=Math.min(t, vd.duration||t); }catch(e){} vd.removeEventListener('loadedmetadata', onm); };
        vd.addEventListener('loadedmetadata', onm); }
      vids.push(pan.video);
    } else if(P.kind==='scene'){
      if(!state.views[id]) state.views[id] = {vov:newView(), vgt:newView(), vpred:newView()};
      const cap = P.content==='hand' ? '逐帧当前相机系；GT手×GT相机，PRED手×PRED相机'
        : (P.content==='camworld' ? '相机轨迹+世界轴(首帧对齐)'
        : (P.content==='worldmotion' ? '固定世界；GT=红绿蓝，PRED=橙黄紫；左键平移，Ctrl+左键旋转' : '世界系整体'));
      const axesKey = AXES;
      const worldTools = P.content==='worldmotion' ? `<span class="scene-local">
        <label>坐标</label><span class="seg world-coord-controls">
          <button data-world-coord="z_up" class="${state.worldCoordMode==='z_up'?'on':''}" title="X 向右、Y 向前、Z 向上">Z-up</button>
          <button data-world-coord="opencv" class="${state.worldCoordMode==='opencv'?'on':''}" title="OpenCV：X 向右、Y 向下、Z 向前">OpenCV</button>
        </span>
        <label>视图</label><span class="seg world-view-controls">
          <button class="world-traj${state.showTraj?' on':''}" title="同时显示或隐藏手部轨迹与相机轨迹">轨迹</button>
          <button class="world-link${state.showCamHand?' on':''}" title="显示相机姿态三轴及相机到左右手腕的实时距离">相机姿态↔手</button>
        </span></span>` : '';
      sec.innerHTML = `<div class="btitle"><b class="k">${P.name}</b> <span class="sub">${cap} ${axesKey}</span>${worldTools}</div>`
        + (ov ? `<div class="ov3d"><div class="chart"><canvas class="world cov"></canvas></div></div>`
              : `<div class="worlds"><div class="chart"><span class="cap">GT</span><canvas class="world cgt"></canvas></div>`
                + `<div class="chart"><span class="cap">PRED</span><canvas class="world cpred"></canvas></div></div>`);
      appendPanel(sec,id);
      const pan = { id, kind:'scene', content:P.content, v:state.views[id], renderDone:false,
                    cOV:sec.querySelector('.cov'), cGT:sec.querySelector('.cgt'), cPred:sec.querySelector('.cpred') };
      for(const c of [pan.cOV,pan.cGT,pan.cPred]) if(c)
        c.title='左键拖动平移；Ctrl+左键拖动旋转；滚轮缩放';
      const viewChanged=()=>_invalidateWorldVideo(pan);
      if(pan.cOV) attachOrbit(pan.cOV, pan.v.vov, viewChanged);
      if(pan.cGT) attachOrbit(pan.cGT, pan.v.vgt, viewChanged);
      if(pan.cPred) attachOrbit(pan.cPred, pan.v.vpred, viewChanged);
      state.panels[id] = pan;
      if(P.content==='worldmotion'){
        const traj=sec.querySelector('.world-traj'), link=sec.querySelector('.world-link');
        sec.querySelectorAll('[data-world-coord]').forEach(button=>button.onclick=()=>{
          const mode=button.dataset.worldCoord;
          if(!['z_up','opencv'].includes(mode)||state.worldCoordMode===mode) return;
          state.worldCoordMode=mode;
          sec.querySelectorAll('[data-world-coord]').forEach(option=>option.classList.toggle('on',option===button));
          _invalidateWorldVideo(pan); requestDraw();
          console.log('[btn] 固定世界坐标系 → '+mode);
        });
        traj.onclick=()=>{ state.showTraj=!state.showTraj; traj.classList.toggle('on',state.showTraj);
          _invalidateWorldVideo(pan); requestDraw(); console.log('[btn] 固定世界轨迹 → '+state.showTraj); };
        link.onclick=()=>{ state.showCamHand=!state.showCamHand; link.classList.toggle('on',state.showCamHand);
          _invalidateWorldVideo(pan); requestDraw(); console.log('[btn] 固定世界相机↔手 → '+state.showCamHand); };
        _checkWorldVideoCache(pan);
      }
    } else if(P.kind==='mujoco'||P.kind==='retarget'){
      const isMujoco=P.kind==='mujoco';
      const methodLabel=isMujoco?'MuJoCo':'Wuji Hand';
      const endpoint=isMujoco?'mujoco':'retarget';
      const renderTag=isMujoco?MUJOCO_RENDER_TAG:RETARGET_RENDER_TAG;
      const videoClass=isMujoco?'mujoco-video':'retarget-video';
      const compare=!state.no_truth&&!state.rawOnly;
      const sources=compare?['gt','pred']:[state.rawOnly?'gt':'pred'];
      const lead=compare?'GT vs PRED':(sources[0]==='gt'?'GT':'PRED');
      const methodDetail=isMujoco?'共用 Wuji 固定视角':'21点重定向 · 共用 Wuji 固定视角';
      const cells=sources.map(source=>`<div class="robot-video-cell" data-robot-source="${source}">
        <span class="robot-video-label">${source.toUpperCase()}</span>
        <div class="vwrap mujoco-wrap"><video class="${videoClass}" preload="metadata" muted playsinline></video><div class="vbadge">${methodLabel} ${source.toUpperCase()} 正在自动渲染…</div></div></div>`).join('');
      sec.innerHTML=`<div class="btitle"><b class="k">${P.name}</b><span class="sub">${lead} · ${methodDetail} · 起点框 + 实时相机框</span></div>
        <div class="robot-comparison-grid${compare?' is-comparison':''}">${cells}</div>`;
      appendPanel(sec,id);
      const parent={id,kind:P.kind,video:null,videos:[],children:[],renderDone:false,failed:false};
      state.panels[id]=parent;
      const updateParent=()=>{
        parent.renderDone=parent.children.length>0&&parent.children.every(child=>child.renderDone);
        parent.renderStarted=parent.children.some(child=>child.renderStarted);
        parent.failed=parent.children.some(child=>child.failed);
        updateExportButton();
      };
      for(const source of sources){
        const cell=sec.querySelector(`[data-robot-source="${source}"]`);
        const video=cell.querySelector('video'), badge=cell.querySelector('.vbadge');
        const betas=source==='gt'?state.gtBetas:state.predBetas;
        const fov=source==='gt'?'per_frame':state.predFov;
        const query=`source=${source}&cam_mode=${encodeURIComponent(state.camMode)}&hand_mode=${encodeURIComponent(effectiveHandMode())}&betas=${betas}&fov=${fov}`;
        const bust=`${source==='gt'?'gt':CKPT_TAG}:${renderTag}`;
        const child={id:`${id}_${source}`,panelId:id,kind:P.kind,video,source,
          label:`${methodLabel} ${source.toUpperCase()}`,renderDone:false,
          renderStarted:false,loadRequested:false,failed:false,
          videoUrl:U(`/${endpoint}/${state.eid}?${query}&_=${encodeURIComponent(bust)}`),
          progressUrl:`/api/${endpoint}/progress/${state.eid}?${query}`,
          isActive:()=>state.panels[id]===parent,onStatusChange:updateParent};
        video.controls=false; video.disablePictureInPicture=true; video.tabIndex=-1;
        video.classList.add('synced-follower-video');
        video.setAttribute('controlsList','nodownload nofullscreen noremoteplayback noplaybackrate');
        video.setAttribute('aria-label',`${methodLabel} ${source.toUpperCase()} video synchronized with the overall 2D video`);
        parent.children.push(child); parent.videos.push(video);
        _wireServerVideoPanel(child,badge,{autoStart:false});
        deferredRobotPanels.push(child);
        if(preserveTime&&t>0) video.addEventListener('loadedmetadata',()=>{ try{video.currentTime=Math.min(t,video.duration||t);}catch(e){} },{once:true});
      }
      const preferred=parent.children.find(child=>child.source==='pred')||parent.children[0];
      parent.video=preferred&&preferred.video;
      updateParent();
    } else if(P.kind==='nums'){
      sec.innerHTML = `<div class="bnums"></div>`;
      appendPanel(sec,id);
      state.panels[id] = { id, kind:'nums', frame:P.frame||'world', bnums:sec.querySelector('.bnums') };
    } else if(P.kind==='loss'){
      appendPanel(sec,id);
      if(persist.metricsWrap) sec.appendChild(persist.metricsWrap);  // loss 节点挂进此位置（隐藏时不挂→游离不显示，节点不丢）
      state.panels[id] = { id, kind:'loss' };
    } else if(P.kind==='tool'){
      appendPanel(sec,id);
      const w = persist[P.wrapId];          // 工具面板持久节点挂进此位置,并清掉初始 display:none
      if(w){ w.style.display=''; sec.appendChild(w); }
      state.panels[id] = { id, kind:'tool', wrapId:P.wrapId };
    }
  }
  if(!hasPrimary2D) settlePrimary2D();
  primary2DReady.then(()=>{
    for(const pan of deferredRobotPanels){
      if(!pan.isActive||pan.isActive()) pan.releaseRender();
    }
  });
  if(!cont.children.length){
    cont.innerHTML = `<div class="empty-state"><div><span class="empty-index">READY / WAITING</span>
      <b>选择样本，按需调整参数</b><p>输入与设置只会进入待应用状态；模型仅在点击“开始推理”后运行。</p></div></div>`;
  }
  masterVideo = vids[0] || null;            // 3D/数值/loss 统一跟随首个可见视频的帧
  watchMasterVideo(masterVideo);
  renderOrderBar();
  updateExportButton();
  syncMujocoVideos(true);
  if(loaded) requestDraw();
}

// 当前帧取自主视频；MuJoCo 与 Wuji Hand 只作为整体·2D 的同步跟随画面。
function frameOf(video){ return Math.min(Math.round(((video&&video.currentTime)||0) * state.fps), state.nframes-1); }

function requestDraw(){ _drawPending = true; }

function syncMujocoVideos(force=false){
  if(!masterVideo) return;
  for(const id of ['mujoco_3d','wuji_retarget_3d']){
    const pan=state.panels[id], followers=pan?(pan.videos||[pan.video]):[];
    for(const follower of followers){
      if(!follower||follower.readyState<1) continue;
      const target=masterVideo.currentTime||0;
      if(force||Math.abs((follower.currentTime||0)-target)>.12){
        try{ follower.currentTime=Math.min(target,Number.isFinite(follower.duration)?follower.duration:target); }catch(e){}
      }
      follower.playbackRate=masterVideo.playbackRate||1;
      if(masterVideo.paused||masterVideo.ended) follower.pause();
      else if(follower.paused) follower.play().catch(()=>{});
    }
  }
}

// 只在浏览器真正提交一个新视频帧时重画联动面板；旧浏览器由 RAF 中的帧号变化兜底。
function watchMasterVideo(video){
  if(_masterVideoCleanup){ _masterVideoCleanup(); _masterVideoCleanup=null; }
  _videoFrameDriven=false; _fallbackFrame=-1; requestDraw();
  if(!video) return;
  const events=['loadeddata','loadedmetadata','seeked','play','pause','ended'];
  const onEvent=()=>{ requestDraw(); syncMujocoVideos(true); };
  for(const event of events) video.addEventListener(event,onEvent);
  let stopped=false, callbackId=null;
  if(typeof video.requestVideoFrameCallback==='function'){
    _videoFrameDriven=true;
    const onFrame=()=>{
      if(stopped || masterVideo!==video) return;
      requestDraw(); syncMujocoVideos(false);
      callbackId=video.requestVideoFrameCallback(onFrame);
    };
    callbackId=video.requestVideoFrameCallback(onFrame);
  }
  _masterVideoCleanup=()=>{
    stopped=true;
    for(const event of events) video.removeEventListener(event,onEvent);
    if(callbackId!==null && typeof video.cancelVideoFrameCallback==='function'){
      try{ video.cancelVideoFrameCallback(callbackId); }catch(e){}
    }
  };
}

function _mediaFrameReady(video){
  return Boolean(video&&video.readyState>=2&&video.videoWidth>0&&video.videoHeight>0);
}

function _frameCaptureItem(spec){
  const panel=state.panels[spec.id];
  const canvases=panel&&panel.kind==='scene'
    ? [panel.cOV,panel.cGT,panel.cPred].filter(Boolean) : [];
  const video=panel&&panel.video;
  const ready=canvases.length>0||_mediaFrameReady(video);
  return {...spec,panel,canvases,video,available:Boolean(panel&&ready)};
}

function _setFrameCaptureOpen(open){
  const picker=$('#frameCapturePicker'), button=$('#frameCaptureBtn'), menu=$('#frameCaptureMenu');
  if(!picker||!button||!menu) return;
  const expanded=Boolean(open&&!button.disabled&&!state.capturingFrame);
  menu.hidden=!expanded;
  picker.classList.toggle('open',expanded);
  button.setAttribute('aria-expanded',expanded?'true':'false');
}

function updateFrameCaptureButton(){
  const button=$('#frameCaptureBtn'), menu=$('#frameCaptureMenu');
  if(!button||!menu) return;
  const items=FRAME_CAPTURE_SOURCES.map(_frameCaptureItem);
  const available=items.filter(item=>item.available);
  button.disabled=state.capturingFrame||!state.loaded||!available.length;
  button.textContent=state.capturingFrame?'正在保存…':(state.frameCaptureMessage||'保存当前帧 ▾');
  button.title=!state.loaded?'加载样本后才能保存当前帧'
    :(!available.length?'当前没有可截图的已显示画面':'选择一个画面，保存其当前播放帧为 PNG');
  menu.querySelectorAll('[data-frame-capture-source]').forEach(option=>{
    const item=items.find(candidate=>candidate.id===option.dataset.frameCaptureSource);
    option.disabled=state.capturingFrame||!item||!item.available;
    option.title=!item||!item.panel?'当前面板未显示'
      :(!item.available?'该视频画面尚未渲染完成':'保存当前播放帧');
  });
  if(button.disabled) _setFrameCaptureOpen(false);
}

function wireFrameCapturePicker(){
  const picker=$('#frameCapturePicker'), button=$('#frameCaptureBtn'), menu=$('#frameCaptureMenu');
  if(!picker||!button||!menu) return;
  button.onclick=event=>{
    event.stopPropagation();
    _setFrameCaptureOpen(menu.hidden);
  };
  menu.querySelectorAll('[data-frame-capture-source]').forEach(option=>{
    option.onclick=event=>{
      event.stopPropagation();
      _setFrameCaptureOpen(false);
      saveCurrentFrame(option.dataset.frameCaptureSource);
    };
  });
  picker.addEventListener('keydown',event=>{
    if(event.key==='Escape'){
      event.preventDefault(); _setFrameCaptureOpen(false); button.focus();
    }else if(event.key==='ArrowDown'&&event.target===button){
      event.preventDefault(); _setFrameCaptureOpen(true);
      const first=menu.querySelector('[data-frame-capture-source]:not(:disabled)');
      if(first) first.focus();
    }
  });
  document.addEventListener('click',event=>{
    if(!picker.contains(event.target)) _setFrameCaptureOpen(false);
  });
  updateFrameCaptureButton();
}

function _snapshotVideoFrame(video){
  if(!_mediaFrameReady(video)) throw new Error('当前视频帧尚未就绪');
  const canvas=document.createElement('canvas');
  canvas.width=video.videoWidth; canvas.height=video.videoHeight;
  const context=canvas.getContext('2d');
  if(!context) throw new Error('浏览器无法创建截图画布');
  context.drawImage(video,0,0,canvas.width,canvas.height);
  return canvas;
}

function _snapshotCanvasView(canvases){
  const sources=canvases.filter(canvas=>canvas&&canvas.width>0&&canvas.height>0);
  if(!sources.length) throw new Error('当前 3D 视角尚未绘制');
  const rects=sources.map(canvas=>canvas.getBoundingClientRect());
  const minLeft=Math.min(...rects.map(rect=>rect.left));
  const minTop=Math.min(...rects.map(rect=>rect.top));
  const maxRight=Math.max(...rects.map(rect=>rect.right));
  const maxBottom=Math.max(...rects.map(rect=>rect.bottom));
  const scale=Math.max(1,window.devicePixelRatio||1);
  const output=document.createElement('canvas');
  output.width=Math.max(1,Math.round((maxRight-minLeft)*scale));
  output.height=Math.max(1,Math.round((maxBottom-minTop)*scale));
  const context=output.getContext('2d');
  if(!context) throw new Error('浏览器无法创建截图画布');
  context.fillStyle='#040908'; context.fillRect(0,0,output.width,output.height);
  sources.forEach((canvas,index)=>{
    const rect=rects[index];
    context.drawImage(
      canvas,
      Math.round((rect.left-minLeft)*scale),
      Math.round((rect.top-minTop)*scale),
      Math.round(rect.width*scale),
      Math.round(rect.height*scale),
    );
  });
  return output;
}

function _renderCurrentWorldView(panel,frame){
  const items=sceneItems();
  const fixedVideo={currentTime:frame/Math.max(1,state.fps)};
  if(state.layout==='overlay'){
    if(panel.cOV) renderScene(panel.cOV,items.ov,panel.v.vov,panel.content,fixedVideo);
    return [panel.cOV].filter(Boolean);
  }
  if(panel.cGT) renderScene(panel.cGT,items.gt,panel.v.vgt,panel.content,fixedVideo);
  if(panel.cPred) renderScene(panel.cPred,items.pred,panel.v.vpred,panel.content,fixedVideo);
  return [panel.cGT,panel.cPred].filter(Boolean);
}

function _frameCaptureFilename(spec,frame){
  let sample=state.sourceName||state.vidName||
    (state.epIdx!=null?`episode_${String(state.epIdx).padStart(4,'0')}`:`item_${state.eid}`);
  sample=String(sample).replace(/\.[^.]+$/,'').replace(/[^A-Za-z0-9._-]+/g,'_').replace(/^_+|_+$/g,'');
  if(!sample) sample=`item_${state.eid}`;
  const digits=Math.max(4,String(Math.max(0,state.nframes-1)).length);
  const sourceFile=spec.id==='world_motion_3d'?`${spec.file}_${state.worldCoordMode}`:spec.file;
  return `${sample}_${sourceFile}_frame_${String(frame).padStart(digits,'0')}.png`;
}

function _downloadSnapshot(canvas,filename){
  return new Promise((resolve,reject)=>{
    const startDownload=href=>{
      const anchor=document.createElement('a');
      anchor.href=href; anchor.download=filename;
      document.body.appendChild(anchor); anchor.click(); anchor.remove();
      resolve();
    };
    if(typeof canvas.toBlob!=='function'){
      try{ startDownload(canvas.toDataURL('image/png')); }catch(error){ reject(error); }
      return;
    }
    canvas.toBlob(blob=>{
      if(!blob){ reject(new Error('PNG 编码失败')); return; }
      const url=URL.createObjectURL(blob);
      startDownload(url);
      setTimeout(()=>URL.revokeObjectURL(url),1000);
    },'image/png');
  });
}

async function saveCurrentFrame(sourceId){
  if(state.capturingFrame) return;
  const spec=FRAME_CAPTURE_SOURCES.find(item=>item.id===sourceId);
  const item=spec&&_frameCaptureItem(spec);
  if(!item||!item.available){ info.textContent='当前画面尚未就绪，或对应面板未显示'; return; }
  const frame=item.panel.kind==='scene' ? frameOf(masterVideo) : frameOf(item.video);
  state.capturingFrame=true; state.frameCaptureMessage=''; updateFrameCaptureButton();
  try{
    let snapshot;
    if(item.panel.kind==='scene'){
      const canvases=_renderCurrentWorldView(item.panel,frame);
      snapshot=_snapshotCanvasView(canvases);
      requestDraw();
    }else snapshot=_snapshotVideoFrame(item.video);
    const filename=_frameCaptureFilename(spec,frame);
    await _downloadSnapshot(snapshot,filename);
    state.frameCaptureMessage=`已保存帧 ${frame}`;
    info.textContent=`已保存 ${spec.label} 当前帧 ${frame}：${filename}`;
    setTimeout(()=>{
      if(state.frameCaptureMessage===`已保存帧 ${frame}`){
        state.frameCaptureMessage=''; updateFrameCaptureButton();
      }
    },1800);
  }catch(error){
    state.frameCaptureMessage='保存失败';
    info.textContent='保存当前帧失败：'+error.message;
    setTimeout(()=>{ if(state.frameCaptureMessage==='保存失败'){ state.frameCaptureMessage=''; updateFrameCaptureButton(); } },1800);
  }finally{
    state.capturingFrame=false; updateFrameCaptureButton();
  }
}

function _exportSourceItem(spec){
  const panel=state.panels[spec.id];
  const source=panel&&panel.video;
  const ready=_mediaFrameReady(source);
  const rendered=Boolean(ready||(panel&&panel.renderDone));
  // 四路都是服务端渲染源；机器人面板即使隐藏，导出任务也能直接生成并缓存视频。
  return {...spec,panel,available:Boolean(state.loaded),rendered,ready};
}

function _selectedExportSources(){
  return EXPORT_SOURCES.filter(spec=>state.exportSelection.has(spec.id)).map(_exportSourceItem);
}

function updateExportPicker(){
  const menu=$('#exportSourcesMenu');
  if(!menu) return;
  if(state.exporting) _setExportPickerOpen(false);
  menu.querySelectorAll('[data-export-source]').forEach(input=>{
    const spec=EXPORT_SOURCES.find(item=>item.id===input.dataset.exportSource);
    const item=spec&&_exportSourceItem(spec), label=input.closest('label');
    input.checked=state.exportSelection.has(input.dataset.exportSource);
    input.disabled=state.exporting;
    if(label){
      label.classList.toggle('unavailable',!item||!item.available);
      label.title=!item||!item.available?'加载样本后可导出'
        :(item.rendered?'已渲染，导出时直接复用':'导出时由服务端渲染并缓存');
    }
  });
}

function _setExportPickerOpen(open){
  const picker=$('#exportPicker'), button=$('#exportBtn'), menu=$('#exportSourcesMenu');
  if(!picker||!button||!menu) return;
  const expanded=Boolean(open&&!state.exporting);
  menu.hidden=!expanded;
  picker.classList.toggle('open',expanded);
  button.setAttribute('aria-expanded',expanded?'true':'false');
}

function wireExportPicker(){
  const picker=$('#exportPicker'), button=$('#exportBtn'), menu=$('#exportSourcesMenu');
  if(!picker||!button||!menu) return;
  let closeTimer=null;
  const open=()=>{
    if(closeTimer!==null){ clearTimeout(closeTimer); closeTimer=null; }
    _setExportPickerOpen(true);
  };
  const closeSoon=()=>{
    if(closeTimer!==null) clearTimeout(closeTimer);
    closeTimer=setTimeout(()=>{
      closeTimer=null;
      if(!picker.matches(':hover')&&!picker.contains(document.activeElement)){
        _setExportPickerOpen(false);
      }
    },120);
  };
  picker.addEventListener('mouseenter',open);
  picker.addEventListener('mouseleave',closeSoon);
  picker.addEventListener('focusin',open);
  picker.addEventListener('focusout',closeSoon);
  picker.addEventListener('keydown',event=>{
    if(event.key==='Escape'){
      event.preventDefault(); _setExportPickerOpen(false); button.focus();
    }else if(event.key==='ArrowDown'&&event.target===button){
      event.preventDefault(); open();
      const first=menu.querySelector('[data-export-source]:not(:disabled)');
      if(first) first.focus();
    }
  });
  menu.querySelectorAll('[data-export-source]').forEach(input=>{
    input.onchange=()=>{
      if(input.checked) state.exportSelection.add(input.dataset.exportSource);
      else state.exportSelection.delete(input.dataset.exportSource);
      updateExportButton();
    };
  });
  document.addEventListener('click',event=>{
    if(!picker.contains(event.target)) _setExportPickerOpen(false);
  });
  updateExportPicker();
}

function updateExportButton(){
  const button=$('#exportBtn');
  if(!button) return;
  const selected=_selectedExportSources();
  const missing=selected.filter(item=>!item.available);
  const unrendered=selected.filter(item=>item.available&&!item.rendered);
  button.textContent=state.exporting?`正在导出 ${Math.round(state.exportProgress*100)}%`:`导出选中 ${selected.length} 路`;
  button.classList.toggle('on',state.exporting);
  button.setAttribute('aria-busy',state.exporting?'true':'false');
  button.disabled=state.exporting||!state.loaded||!selected.length||Boolean(missing.length);
  if(!state.loaded) button.title='加载样本后才能导出';
  else if(!selected.length) button.title='请至少选择一个导出画面';
  else if(missing.length) button.title='所选画面当前不可导出：'+missing.map(item=>item.label).join('、');
  else if(unrendered.length) button.title='后台渲染尚未完成；点击后等待并导出：'+unrendered.map(item=>item.label).join('、');
  else button.title='全部已渲染，点击直接复用现有 MP4 导出';
  updateExportPicker();
  updateFrameCaptureButton();
}

function setExportProgress(progress,message,status=''){
  const wrap=$('#exportProgress'), bar=$('#exportProgressBar'), text=$('#exportProgressText');
  state.exportProgress=Math.max(0,Math.min(1,+progress||0));
  if(!wrap||!bar||!text) return;
  wrap.hidden=false; wrap.classList.toggle('done',status==='done'); wrap.classList.toggle('error',status==='error');
  const percent=Math.round(state.exportProgress*100);
  bar.style.width=percent+'%'; text.textContent=`${percent}% · ${message||'正在导出'}`;
  wrap.setAttribute('aria-valuenow',String(percent));
  updateExportButton();
}

function hideExportProgress(){
  const wrap=$('#exportProgress');
  if(wrap){ wrap.hidden=true; wrap.classList.remove('done','error'); }
}

async function exportVideo(){
  if(state.exporting) return;
  const sources=_selectedExportSources();
  if(!sources.length){ info.textContent='请至少选择一个导出画面'; return; }
  const missing=sources.filter(item=>!item.available);
  if(missing.length){ info.textContent='所选画面当前不可导出：'+missing.map(item=>item.label).join('、'); return; }
  const unrendered=sources.filter(item=>!item.rendered);
  state.exporting=true; state.exportProgress=0; hideExportProgress();
  setExportProgress(0,'正在提交导出任务'); updateExportButton();
  _setExportPickerOpen(false);
  info.textContent=unrendered.length
    ? '正在等待后台渲染并导出：'+unrendered.map(item=>item.label).join('、')
    : '正在复用已渲染 MP4 并导出…';
  let completed=false;
  try{
    const response=await fetch(U(`/api/export/${state.eid}`),{
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        sources:sources.map(item=>item.id), mode:state.mode, layout:state.layout,
        content:'both', cam_mode:state.camMode, hand_mode:effectiveHandMode(),
        gt_betas:state.gtBetas, pred_betas:state.predBetas, pred_fov:state.predFov,
        world_views:state.views.world_motion_3d||null,
        world_coord_mode:state.worldCoordMode,
        show_traj:state.showTraj, show_cam_hand:state.showCamHand,
        raw:state.rawOnly,
      }),
    });
    if(!response.ok) throw new Error((await response.text()).replace(/<[^>]+>/g,' ').trim()||`HTTP ${response.status}`);
    const task=await response.json();
    if(!task.progress) throw new Error('服务端未返回导出进度地址');
    let result=null;
    while(true){
      result=await getJSON(task.progress+'?_='+Date.now());
      _syncExportPanelProgress(result.sources);
      setExportProgress(result.progress,result.message||'正在导出',result.stage);
      if(result.stage==='error') throw new Error(result.error||'导出失败');
      if(result.stage==='done') break;
      await new Promise(resolve=>setTimeout(resolve,350));
    }
    if(!result.download) throw new Error('服务端未返回导出文件');
    for(const item of sources){ if(item.panel&&'renderDone' in item.panel) item.panel.renderDone=true; }
    const anchor=document.createElement('a');
    anchor.href=U(result.download); anchor.download=result.filename||'';
    document.body.appendChild(anchor); anchor.click(); anchor.remove();
    completed=true; setExportProgress(1,'导出完成，已开始下载','done');
    info.textContent='导出完成：'+sources.map(item=>item.label).join(' + ');
  }catch(error){
    setExportProgress(state.exportProgress,'导出失败：'+error.message,'error');
    info.textContent='导出失败：'+error.message;
  }finally{
    state.exporting=false; updateExportButton();
    if(completed) setTimeout(()=>{ if(!state.exporting) hideExportProgress(); },1800);
  }
}

function forSampledTrajectory(points, visit){
  if(!points || !points.length) return;
  const stride=Math.max(1,Math.ceil(points.length/MAX_TRAJECTORY_POINTS));
  let last=-1;
  for(let i=0;i<points.length;i+=stride){ visit(points[i],i); last=i; }
  if(last!==points.length-1) visit(points[points.length-1],points.length-1);
}

// 对比内容 key → 坐标系与图层。worldmotion 只在初始化时用首帧相机定义世界系，之后不再跟随。
function contentPlan(key){
  if(key==='hand')     return {frame:'cam',   showHand:true,  showCam:false, showTraj:false};
  if(key==='camworld') return {frame:'world', showHand:false, showCam:true};
  if(key==='worldmotion') return {frame:'world', showHand:true, showCam:true,
                                  anchorFirst:true, fixedView:true};
  return {frame:'world', showHand:true, showCam:true};   // both
}

function cachedWorldExtent(d, modeKey, plan, xf){
  let modes=_sceneExtentCache.get(d);
  if(!modes){ modes=new Map(); _sceneExtentCache.set(d,modes); }
  const key=`${modeKey}:${state.worldCoordMode}:${+plan.showHand}:${+plan.showCam}`;
  if(modes.has(key)) return modes.get(key);
  let mn=null, mx=null;
  const add=p=>{
    if(!p) return;
    const q=xf(p);
    if(!mn){ mn=q.slice(); mx=q.slice(); return; }
    for(let i=0;i<3;i++){ if(q[i]<mn[i]) mn[i]=q[i]; if(q[i]>mx[i]) mx[i]=q[i]; }
  };
  if(plan.showHand){
    for(const side of ['left','right']) for(const p of ((d.traj&&d.traj[side])||[])) add(p);
  }
  if(plan.showCam) for(const p of (d.cam_t||[])) add(p);
  const bounds=mn?{mn,mx}:null;
  modes.set(key,bounds);
  return bounds;
}

// 沿 up 的最低点（地面高度用）。点集与 cachedWorldExtent 一致并含原点，与 Python 的 _extent 同口径。
function cachedGroundLow(d, modeKey, plan, xf, up){
  let modes=_sceneGroundCache.get(d);
  if(!modes){ modes=new Map(); _sceneGroundCache.set(d,modes); }
  // up 随「叠加/并排」下参与的 item 集合略有不同，一并进 key，避免切布局后用到旧高度。
  const key=`${modeKey}:${+plan.showHand}:${+plan.showCam}:${up.map(x=>x.toFixed(3)).join(',')}`;
  if(modes.has(key)) return modes.get(key);
  let low=0;                                          // 原点(首帧相机)沿 up 高度恒为 0
  const add=p=>{ if(!p) return; const h=_dot(xf(p),up); if(h<low) low=h; };
  for(const side of ['left','right']) for(const p of ((d.traj&&d.traj[side])||[])) add(p);
  for(const p of (d.cam_t||[])) add(p);
  modes.set(key,low);
  return low;
}

// 比例尺：正交投影屏幕内长度守恒（proj 只做旋转+等比 scale），故取一段「整数 cm」对应的像素宽
// 画横条+端点+标注，直观读位移多远。scale=像素/cm，随 zoom 变；长度取 1/2/5×10^k 的“整齐”值，
// ≥100cm 自动换算成 m。画右下角，避开左下角的帧信息文字。
function drawScaleBar(g, W, H, scale){
  if(!(scale > 0 && isFinite(scale))) return;
  const target = 90;                                   // 目标像素长度（附近取整齐 cm）
  const raw = target / scale;                          // 对应多少 cm
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  let nice = pow; for(const m of [1,2,5,10]){ if(m*pow <= raw) nice = m*pow; }
  const px = nice * scale;
  const label = nice >= 100 ? (nice/100) + ' m' : nice + ' cm';
  const x1 = W-16, x0 = x1-px, y = H-16;
  g.setLineDash([]); g.strokeStyle='#e6edf3'; g.lineWidth=2;
  g.beginPath(); g.moveTo(x0,y); g.lineTo(x1,y);
  g.moveTo(x0,y-4); g.lineTo(x0,y+4); g.moveTo(x1,y-4); g.lineTo(x1,y+4); g.stroke();
  g.fillStyle='#e6edf3'; g.font='11px system-ui'; g.textAlign='center';
  g.fillText(label, (x0+x1)/2, y-6); g.textAlign='left';
}

// ── 固定世界的观感规格（网格地面 / 尾迹 / 落影 / HUD）──────────────────────────
// 与 render/fixed_world_video.py 逐条对齐：常量、up 估计、格距取值、淡出曲线都必须同口径，
// 否则网页看到的和导出的 mp4 会是两种画面。文案这边用中文，Python 用英文(cv2 无中文字形)。
const GRID_SPAN = 2.3;              // 网格盘半径 = GRID_SPAN × 场景半径
const GRID_TARGET_CELLS = 10;       // 2×场景半径 目标覆盖格数（再取 1/2/5×10^k 整齐步长）
const GRID_MAJOR_EVERY = 5;
const GROUND_CLEARANCE = 0.28;      // 地面放在最低点下方 max(15cm, GROUND_CLEARANCE×半径)
const BG_CENTER = [22,38,43], BG_EDGE = [5,9,11];          // #16262b / #05090b
const BG_MIX = BG_CENTER.map((c,i)=>c*0.45 + BG_EDGE[i]*0.55);
const GRID_MINOR = [61,91,102], GRID_MAJOR = [109,149,163];  // #3d5b66 / #6d95a3
const SHADOW_COL = '#02060a', SHADOW_ALPHA = 0.55;
const _hex2rgb = h => [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];
// 淡色一律「向背景预混」而不是靠 globalAlpha：与 Python 侧同一算法，且省掉大量状态切换。
function mixCol(color, w){
  w=Math.max(0,Math.min(1,w));
  const c=Array.isArray(color)?color:_hex2rgb(color);
  return `rgb(${c.map((x,i)=>Math.round(x*w + BG_MIX[i]*(1-w))).join(',')})`;
}
function niceStep(raw){
  if(!(raw>0)||!isFinite(raw)) return 1;
  const pow=Math.pow(10,Math.floor(Math.log10(raw)));
  let nice=pow; for(const m of [1,2,5,10]) if(m*pow<=raw) nice=m*pow;
  return nice;
}
const _dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const _add = (a,b,s=1) => [a[0]+b[0]*s, a[1]+b[1]*s, a[2]+b[2]*s];
// 点沿 up 压到地面（落影 / 垂线用）
const flattenTo = (p, plane) => _add(p, plane.up, -(_dot(p,plane.up) - _dot(plane.origin,plane.up)));

// OpenCV 模式没有显式重力轴：用全段相机 +Y(下) 平均值取反当「上」。
// cam_R 为 world→cam 行主序，第 1 行(索引 3..5) = 相机 +Y 在原世界中的方向；显示系再左乘首帧旋转。
function upDirection(items, coordOf, mv){
  let acc=[0,0,0], any=false;
  for(const it of items){
    const rows=it.d.cam_R||[]; let mean=[0,0,0], n=0;
    for(const R of rows){ if(!R||R.length<6) continue;
      mean[0]+=R[3]; mean[1]+=R[4]; mean[2]+=R[5]; n++; }
    if(!n) continue;
    mean=mean.map(x=>x/n);
    const c=coordOf(it.d), m=c.rotateFrame ? mv(c.R0, mean) : mean;
    acc=[acc[0]+m[0],acc[1]+m[1],acc[2]+m[2]]; any=true;
  }
  const len=Math.hypot(acc[0],acc[1],acc[2]);
  if(!any||len<1e-6) return [0,-1,0];                       // OpenCV 相机系里 -Y 朝上
  return [-acc[0]/len,-acc[1]/len,-acc[2]/len];
}

// 地面：过「沿 up 的最低点再下移一点」的水平面 + 平面内正交基 + 整齐格距。
function groundPlane(up, low, C, R){
  const level=low - Math.max(15, GROUND_CLEARANCE*R);
  const origin=_add(C, up, level - _dot(C,up));
  let ref = Math.abs(up[0])>0.9 ? [0,0,1] : [1,0,0];
  let e1=_add(ref, up, -_dot(ref,up));
  const n1=Math.hypot(e1[0],e1[1],e1[2])||1; e1=e1.map(x=>x/n1);
  const e2=[up[1]*e1[2]-up[2]*e1[1], up[2]*e1[0]-up[0]*e1[2], up[0]*e1[1]-up[1]*e1[0]];
  return {up, origin, e1, e2, step:niceStep(2*R/GRID_TARGET_CELLS), half:GRID_SPAN*R};
}

// 网格地面：圆盘状、离中心越远越淡，每 GRID_MAJOR_EVERY 格加亮一条。
// 同色同淡度的线段合成一条 path 再 stroke（逐段 stroke 上千次会拖慢逐帧重绘）。
function drawGround(g, plane, proj){
  const {origin,e1,e2,step,half}=plane;
  const count=Math.min(48, Math.max(2, Math.floor(half/step)));
  const buckets=new Map();
  for(const [axis,other] of [[e1,e2],[e2,e1]]){
    for(let index=-count; index<=count; index++){
      const offset=index*step, major=index%GRID_MAJOR_EVERY===0;
      const base=_add(origin, axis, offset);
      for(let cell=-count; cell<count; cell++){
        const s=cell*step, e=(cell+1)*step;
        const dist=Math.hypot(offset,(s+e)/2);
        if(dist>half) continue;
        let fade=Math.pow(1-dist/half,1.15)*(major?1:0.66);
        if(fade<0.03) continue;
        const level=Math.max(1,Math.round(fade*8));          // 量化成 8 档，便于合并 path
        const key=(major?'M':'m')+level;
        let bucket=buckets.get(key);
        if(!bucket){ bucket={major, fade:level/8, path:new Path2D()}; buckets.set(key,bucket); }
        const p0=proj(_add(base,other,s),null), p1=proj(_add(base,other,e),null);
        bucket.path.moveTo(p0[0],p0[1]); bucket.path.lineTo(p1[0],p1[1]);
      }
    }
  }
  g.save(); g.setLineDash([]);
  for(const bucket of buckets.values()){
    g.strokeStyle=mixCol(bucket.major?GRID_MAJOR:GRID_MINOR, bucket.fade);
    g.lineWidth=bucket.major?1.7:1;
    g.stroke(bucket.path);
  }
  // 世界原点(首帧相机)到地面的垂线 + 落点环：一眼读出相机离地多高。
  const O=proj([0,0,0],null), G=proj(flattenTo([0,0,0],plane),null);
  g.setLineDash([4,4]); g.lineWidth=1; g.strokeStyle=mixCol('#e6edf3',0.35);
  g.beginPath(); g.moveTo(O[0],O[1]); g.lineTo(G[0],G[1]); g.stroke();
  g.setLineDash([]); g.strokeStyle=mixCol('#e6edf3',0.5);
  g.beginPath(); g.arc(G[0],G[1],4,0,7); g.stroke();
  g.restore();
}

// 轨迹尾迹：越早的一段越淡（分 8 段批量 stroke，避免逐点改色）。
function drawTrail(g, pts, color, width, dash){
  if(!pts || pts.length<2) return;
  const chunks=8, total=pts.length-1;
  g.save(); g.setLineDash(dash||[]); g.lineWidth=width; g.lineCap='round';
  for(let k=0;k<chunks;k++){
    const from=Math.floor(total*k/chunks), to=Math.min(total, Math.ceil(total*(k+1)/chunks));
    if(to<=from) continue;
    g.strokeStyle=mixCol(color, 0.22 + 0.78*Math.pow((k+1)/chunks,1.4));
    g.beginPath(); let pen=false;
    for(let i=from;i<=to;i++){ const P=pts[i];
      if(!P){ pen=false; continue; }
      if(pen) g.lineTo(P[0],P[1]); else { g.moveTo(P[0],P[1]); pen=true; } }
    g.stroke();
  }
  g.restore();
}

// 骨架：深色描边 + 本色芯 + 圆头端点，腕节点加环（与 Python 的 _draw_hand 同规格）。
function drawHandSkeleton(g, P, conn, color, dash){
  g.save(); g.lineCap='round'; g.lineJoin='round';
  for(const [stroke,width] of [['#05070a',3.4],[color,1.5]]){
    g.setLineDash(dash||[]); g.strokeStyle=stroke; g.lineWidth=width;
    g.beginPath();
    for(const ab of conn||[]){ const a=P[ab[0]], b=P[ab[1]]; if(!a||!b) continue;
      g.moveTo(a[0],a[1]); g.lineTo(b[0],b[1]); }
    g.stroke();
  }
  g.setLineDash([]);
  for(let j=0;j<P.length;j++){
    const r=j===0?3.3:1.45;
    g.fillStyle='#05070a'; g.beginPath(); g.arc(P[j][0],P[j][1],r+.9,0,7); g.fill();
    g.fillStyle=color; g.beginPath(); g.arc(P[j][0],P[j][1], r, 0, 7); g.fill();
    if(j===0){ g.strokeStyle=color; g.lineWidth=.8; g.beginPath(); g.arc(P[j][0],P[j][1],5.4,0,7); g.stroke(); }
  }
  g.restore();
}

// 地面落影：骨架压到地面后画一遍暗色（真 alpha，保留网格透过阴影的层次）。
function drawGroundShadow(g, T, conn, plane, proj){
  const P=T.map(p=>proj(flattenTo(p,plane),null));
  g.save(); g.globalAlpha=SHADOW_ALPHA; g.setLineDash([]);
  g.strokeStyle=SHADOW_COL; g.lineWidth=3; g.lineCap='round';
  g.beginPath();
  for(const ab of conn||[]){ const a=P[ab[0]], b=P[ab[1]]; if(!a||!b) continue;
    g.moveTo(a[0],a[1]); g.lineTo(b[0],b[1]); }
  g.stroke();
  if(P[0]){ g.fillStyle=SHADOW_COL; g.beginPath(); g.arc(P[0][0],P[0][1],5,0,7); g.fill(); }
  g.restore();
}

// 3D 点到地面的垂线（腕 / 光心），提示它离地多高。
function drawDropLine(g, point, color, plane, proj){
  const A=proj(point,null), B=proj(flattenTo(point,plane),null);
  g.save(); g.globalAlpha=.75; g.setLineDash([4,4]); g.lineWidth=1;
  g.strokeStyle=mixCol(color,0.45);
  g.beginPath(); g.moveTo(A[0],A[1]); g.lineTo(B[0],B[1]); g.stroke();
  g.setLineDash([]); g.globalAlpha=SHADOW_ALPHA; g.fillStyle=SHADOW_COL;
  g.beginPath(); g.arc(B[0],B[1],3.5,0,7); g.fill(); g.restore();
}

// 圆角半透明信息卡：rows=[{t:文本, c:颜色, s:字号, b:加粗?}]；align 取 tl/bl/tr/br。
function drawChip(g, rows, x, y, align='tl', swatch=false){
  const pad=8, gap=5, dot=4, sw=swatch?dot*2+6:0;
  g.save();
  const metrics=rows.map(r=>{ g.font=`${r.b?600:400} ${r.s}px system-ui`;
    return {w:g.measureText(r.t).width, h:r.s}; });
  const w=Math.max(...metrics.map(m=>m.w))+pad*2+sw;
  const h=metrics.reduce((a,m)=>a+m.h,0)+gap*(rows.length-1)+pad*2;
  const x0=align.includes('l')?x:x-w, y0=align.includes('t')?y:y-h;
  g.globalAlpha=.78; g.fillStyle='#080d11';
  g.beginPath();
  if(g.roundRect) g.roundRect(x0,y0,w,h,7); else g.rect(x0,y0,w,h);
  g.fill(); g.globalAlpha=1;
  let cursor=y0+pad;
  rows.forEach((r,i)=>{
    const baseline=cursor+metrics[i].h;
    if(swatch){ g.fillStyle=r.c; g.beginPath(); g.arc(x0+pad+dot, baseline-metrics[i].h*0.35, dot, 0, 7); g.fill(); }
    g.font=`${r.b?600:400} ${r.s}px system-ui`; g.fillStyle=r.c;
    g.fillText(r.t, x0+pad+sw, baseline);
    cursor=baseline+gap;
  });
  g.restore();
}

// ── 世界/相机系 3D 场景（正交投影 + orbit + 缩放；可一次画多路 items 叠加）──────────────
// items: [{d: 世界系 payload, dash: 虚线?, tag?: 标签}]；叠加时 GT 实线 + PRED 虚线。
function renderScene(canvas, items, v, modeKey, video){
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if(!(W>0&&H>0)) return;
  const pixelW=Math.max(1,Math.round(W*dpr)), pixelH=Math.max(1,Math.round(H*dpr));
  if(canvas.width!==pixelW || canvas.height!==pixelH){ canvas.width=pixelW; canvas.height=pixelH; }
  const g = canvas.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,W,H);
  // 椭圆径向暗角背景（与 Python 侧 _background 同色）：比纯色/竖向渐变更像一个「舞台」。
  const bg=g.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.hypot(W,H)*0.52);
  bg.addColorStop(0,`rgb(${BG_CENTER.join(',')})`);
  bg.addColorStop(1,`rgb(${BG_EDGE.join(',')})`);
  g.fillStyle=bg; g.fillRect(0,0,W,H);
  items = (items||[]).filter(it => it && it.d);
  if(!items.length){ g.fillStyle='#8b98a8'; g.font='12px system-ui'; g.fillText('（无数据）', 10, 20); return; }
  const cf = frameOf(video), plan = contentPlan(modeKey);
  const zUp = Boolean(plan.anchorFirst&&state.worldCoordMode==='z_up');

  // 每 item 取当前帧相机：cam_R(world→cam 行主序 9)、cam_t(相机世界位置 cm)。
  // camworld 只平移首帧到原点；worldmotion 用首帧完整位姿一次性定义固定世界系。
  const alignCam = modeKey==='camworld';
  const mv = (R,p) => [R[0]*p[0]+R[1]*p[1]+R[2]*p[2],
                       R[3]*p[0]+R[4]*p[1]+R[5]*p[2],
                       R[6]*p[0]+R[7]*p[1]+R[8]*p[2]];
  const cvToZUp = p => [p[0],p[2],-p[1]];           // OpenCV C0 → X右 / Y前 / Z上
  // A*B^T：原世界的 w2c 旋转换到「首帧相机」定义的新世界坐标。
  const mulABt = (A,B) => { const M=[]; for(let i=0;i<3;i++) for(let j=0;j<3;j++){
    let s=0; for(let k=0;k<3;k++) s += A[i*3+k]*B[j*3+k]; M.push(s); } return M; };
  const coordCache = new Map();
  const coordOf = d => {
    if(coordCache.has(d)) return coordCache.get(d);
    let first=0;
    while(first<(d.cam_t||[]).length && (!d.cam_t[first] || !(d.cam_R||[])[first])) first++;
    const t0 = first<(d.cam_t||[]).length ? d.cam_t[first] : [0,0,0];
    const R0 = first<(d.cam_R||[]).length ? d.cam_R[first] : [1,0,0,0,1,0,0,0,1];
    let xf = p => p, rotateFrame = false;
    if(plan.anchorFirst){
      xf = p => {
        const q=mv(R0,[p[0]-t0[0],p[1]-t0[1],p[2]-t0[2]]);
        return zUp?cvToZUp(q):q;
      };
      rotateFrame = true;
    } else if(alignCam){
      xf = p => [p[0]-t0[0], p[1]-t0[1], p[2]-t0[2]];
    }
    const c={xf, R0, rotateFrame, zUp}; coordCache.set(d,c); return c;
  };
  const camOf = d => {
    const c=coordOf(d), t0=d.cam_t ? d.cam_t[Math.min(cf,d.cam_t.length-1)] : null;
    const R0=d.cam_R ? d.cam_R[Math.min(cf,d.cam_R.length-1)] : null;
    let displayR=(R0&&c.rotateFrame) ? mulABt(R0,c.R0) : R0;
    if(displayR&&c.zUp){
      const mapped=[];
      for(let row=0;row<3;row++) mapped.push(displayR[row*3],displayR[row*3+2],-displayR[row*3+1]);
      displayR=mapped;
    }
    return { R:displayR,
             t:t0 ? c.xf(t0) : null, xf:c.xf };
  };
  // 世界点 → 相机系： p_cam = R·(p - t)。
  const toCam = (p, cam) => { const x=p[0]-cam.t[0], y=p[1]-cam.t[1], z=p[2]-cam.t[2], R=cam.R;
    return [R[0]*x+R[1]*y+R[2]*z, R[3]*x+R[4]*y+R[5]*z, R[6]*x+R[7]*y+R[8]*z]; };
  const tx = (p, cam) => (plan.frame==='cam' && cam && cam.R && cam.t) ? toCam(p, cam)
                           : ((plan.frame==='world' && cam && cam.xf) ? cam.xf(p) : p);

  // 包围盒（含变换后要画的点）→ 中心 C、半径 R。
  let mn=[0,0,0], mx=[0,0,0], any=false;
  const ext=p=>{ if(!any){ mn=p.slice(); mx=p.slice(); any=true; }
    else for(let i=0;i<3;i++){ if(p[i]<mn[i])mn[i]=p[i]; if(p[i]>mx[i])mx[i]=p[i]; } };
  ext([0,0,0]);
  for(const it of items){ const d=it.d, cam=camOf(d);
    if(plan.frame==='cam'){
      for(let h=0; h<2; h++){ const J=d.joints[h][Math.min(cf,d.joints[h].length-1)]; if(J) for(const p of J) ext(tx(p,cam)); }
    } else {
      const bounds=cachedWorldExtent(d,modeKey,plan,cam.xf);
      if(bounds){ ext(bounds.mn); ext(bounds.mx); }
    }
  }
  const C=[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2];
  let R=Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2], 1)*0.5;
  R=Math.max(R, Math.hypot(C[0],C[1],C[2]));
  const scale=Math.min(W,H)*0.42*v.zoom/R;

  // 视角对齐：world 帧 + 对齐相机 → 用参考(首个 item=GT)当前帧 cam_R 旋整个场景；cam 帧本身已对齐。
  const refCam = camOf(items[0].d);
  const Rb = (plan.frame==='world' && !plan.fixedView && state.followCam && refCam.R) ? refCam.R : null;
  const ca=Math.cos(v.az),sa=Math.sin(v.az),ce=Math.cos(v.el),se=Math.sin(v.el);
  const proj=(p, cam)=>{
    const q=tx(p, cam);
    let x=q[0]-C[0], y=q[1]-C[1], z=q[2]-C[2];
    if(Rb){ const X=Rb[0]*x+Rb[1]*y+Rb[2]*z, Y=Rb[3]*x+Rb[4]*y+Rb[5]*z, Z=Rb[6]*x+Rb[7]*y+Rb[8]*z;
            x=X; y=Y; z=Z; }
    let x1,y2;
    if(zUp){
      x1=x*ca+y*sa;                                    // yaw 绕世界 +Z
      const depth=-x*sa+y*ca;
      y2=-z*ce-depth*se;                               // +Z 映射到屏幕上方
    }else{
      x1=x*ca+z*sa;
      const depth=-x*sa+z*ca;                          // OpenCV 模式沿 +Y 为屏幕竖直
      y2=y*ce-depth*se;
    }
    return [W/2 + (v.panX||0) + x1*scale, H/2 + (v.panY||0) + y2*scale];
  };

  // 网格参考地面：Z-up 固定使用 XY 平面；OpenCV 模式沿相机 -Y 估计向上方向。
  // 先画地面再画其它图层，落影/垂线随后压在网格上，读出纵深。
  let plane=null;
  if(plan.frame==='world' && plan.anchorFirst){
    const up=zUp?[0,0,1]:upDirection(items, coordOf, mv);
    let low=0;
    for(const it of items){
      const c=camOf(it.d);
      low=Math.min(low, cachedGroundLow(it.d, modeKey, plan, c.xf, up));
    }
    plane=groundPlane(up, low, C, R);
    drawGround(g, plane, proj);
  }

  // 固定世界画世界轴；相机系画当前 OpenCV 相机轴（光心恒为原点）。
  if((plan.frame==='world' && plan.showCam) || plan.frame==='cam'){
    const aL=R*0.6, O=proj([0,0,0], null); g.save(); g.setLineDash([]); g.globalAlpha=.48;
    g.lineCap='round';
    [['X',[aL,0,0],'#ff6b6b'],['Y',[0,aL,0],'#51cf66'],['Z',[0,0,aL],'#5c9dff']].forEach(([nm,e,col])=>{
      const P=proj(e,null);
      g.strokeStyle='#05070a'; g.lineWidth=2.4;
      g.beginPath(); g.moveTo(O[0],O[1]); g.lineTo(P[0],P[1]); g.stroke();
      g.strokeStyle=col; g.lineWidth=1.35;
      g.beginPath(); g.moveTo(O[0],O[1]); g.lineTo(P[0],P[1]); g.stroke();
      g.fillStyle=col; g.font='600 13px system-ui'; g.fillText(nm, P[0]+3, P[1]+3); });
    g.lineCap='butt';
    g.fillStyle='#e6edf3'; g.beginPath(); g.arc(O[0],O[1],3,0,7); g.fill();
    g.fillStyle='#8b98a8'; g.font='11px system-ui'; g.fillText('O', O[0]+5, O[1]-5);
    g.restore();
  }

  const handCol=['#4dd2ff','#ffd34d'];                 // 0=左手(青), 1=右手(金)
  const sides=['left','right'], CAMCOL='#ff8adf';

  // 固定世界块用三个真实相机局部方向表达完整姿态；标签加 c，避免与世界 Z-up 轴混淆。
  // 三轴长度只用于视觉提示，不参与距离计算。
  const drawCameraPose = (cam, isPred) => {
    if(!cam.t || !cam.R) return;
    const markerColor=isPred?'#ff6b6b':'#51cf66', Pc=proj(cam.t,null);
    const poseSegments=[];
    const poseColor=isPred
      ? {x:'#ff922b',y:'#ffd43b',z:'#b197fc'}
      : {x:'#ff595e',y:'#69db7c',z:'#4dabf7'};
    const right=cam.R.slice(0,3), up=cam.R.slice(3,6).map(x=>-x), view=cam.R.slice(6,9);
    const axisLen=Math.max(7,Math.min(R*0.22,22)), viewLen=Math.max(12,Math.min(R*0.50,48));
    const drawPoseArrow=(axis,len,color,text,width,pattern)=>{
      const end=[cam.t[0]+axis[0]*len,cam.t[1]+axis[1]*len,cam.t[2]+axis[2]*len];
      const Pe=proj(end,null), vx=Pe[0]-Pc[0], vy=Pe[1]-Pc[1], vl=Math.max(1,Math.hypot(vx,vy));
      poseSegments.push({Pe,color});
      const ux=vx/vl, uy=vy/vl, ah=width>=4?9:7;
      g.save(); g.globalAlpha=.58; g.lineCap='round'; g.lineJoin='round'; g.setLineDash(pattern||[]);
      g.strokeStyle='#05070a'; g.lineWidth=width+4;
      g.beginPath(); g.moveTo(Pc[0],Pc[1]); g.lineTo(Pe[0],Pe[1]); g.stroke();
      g.strokeStyle=color; g.lineWidth=width; g.shadowColor=color; g.shadowBlur=6;
      g.beginPath(); g.moveTo(Pc[0],Pc[1]); g.lineTo(Pe[0],Pe[1]); g.stroke();
      g.setLineDash([]); g.fillStyle=color; g.beginPath(); g.moveTo(Pe[0],Pe[1]);
      g.lineTo(Pe[0]-ux*ah-uy*ah*.52,Pe[1]-uy*ah+ux*ah*.52);
      g.lineTo(Pe[0]-ux*ah+uy*ah*.52,Pe[1]-uy*ah-ux*ah*.52); g.closePath(); g.fill();
      g.shadowBlur=0; g.font=`bold ${width>=4?11:10}px system-ui`; const tw=g.measureText(text).width;
      let lx=Math.max(3,Math.min(W-tw-7,Pe[0]+4)), ly=Math.max(12,Math.min(H-8,Pe[1]-4));
      g.globalAlpha=.68; g.fillStyle='#05070a'; g.fillRect(lx-3,ly-10,tw+6,14);
      g.globalAlpha=.72; g.fillStyle=color; g.fillText(text,lx,ly); g.restore();
    };
    // 相机姿态轴用低透明实线：方向连续可读，同时不压过手部与渐变轨迹。
    const cameraDash=[];
    drawPoseArrow(right,axisLen,poseColor.x,'右 / +Xc',2.4,cameraDash);
    drawPoseArrow(up,axisLen,poseColor.y,'上 / -Yc',2.4,cameraDash);
    drawPoseArrow(view,viewLen,poseColor.z,'视线 / +Zc',3.2,cameraDash);
    g.save();
    // 光心也使用低透明实线框，不填充，保持“相机为辅助参照”的语义。
    const size=(isPred?16:12)*v.zoom, bx=Pc[0]-size/2, by=Pc[1]-size/2;
    g.globalAlpha=.55; g.setLineDash(cameraDash); g.strokeStyle='#05070a'; g.lineWidth=2.6; g.strokeRect(bx,by,size,size);
    g.strokeStyle=markerColor; g.lineWidth=1.2; g.strokeRect(bx,by,size,size);
    g.restore();

    // 在方块遮住的范围内重绘姿态轴，形成透视效果。
    g.save(); g.beginPath(); g.rect(bx,by,size,size); g.clip(); g.lineCap='round';
    for(const seg of poseSegments){
      g.setLineDash([]); g.globalAlpha=.58;
      g.strokeStyle='#05070a'; g.lineWidth=4;
      g.beginPath(); g.moveTo(Pc[0],Pc[1]); g.lineTo(seg.Pe[0],seg.Pe[1]); g.stroke();
      g.strokeStyle=seg.color; g.lineWidth=2;
      g.beginPath(); g.moveTo(Pc[0],Pc[1]); g.lineTo(seg.Pe[0],seg.Pe[1]); g.stroke();
    }
    g.restore();
  };

  // 相机图标/姿态三轴必须最后绘制，避免被手骨架、轨迹和测距线遮挡。
  const cameraOverlays=[];

  for(const it of items){ const d=it.d, cam=camOf(d), dash = it.dash ? [7,5] : [];
    const cameraTrajDash=[8,5];                   // 相机始终虚线；手仍以 GT 实线 / PRED 虚线区分
    const handTrajDash=it.dash?[8,5]:[];

    // 相机位姿（world 帧 + 相机与世界）：位置轨迹 + 当前帧三轴（w2c 行=相机基向量在世界方向）
    if(plan.frame==='world' && plan.showCam && d.cam_t){
      if(state.showTraj){
        const pts=[]; forSampledTrajectory(d.cam_t,p=>pts.push(p?proj(cam.xf(p),null):null));
        drawTrail(g, pts, CAMCOL, 1.4, cameraTrajDash);
      }
      if(cam.t && cam.R){ const Pc=proj(cam.t,null), Rr=cam.R, L=R*(plan.anchorFirst ? 0.12 : 0.35);
        g.setLineDash([]);
        if(!plan.anchorFirst){
          [[Rr[0],Rr[1],Rr[2],'#ff6b6b'],[Rr[3],Rr[4],Rr[5],'#51cf66'],[Rr[6],Rr[7],Rr[8],'#5c9dff']]
            .forEach(([ax,ay,az,col])=>{ const P=proj([cam.t[0]+ax*L,cam.t[1]+ay*L,cam.t[2]+az*L],null);
              g.strokeStyle=col; g.lineWidth=2; g.beginPath(); g.moveTo(Pc[0],Pc[1]); g.lineTo(P[0],P[1]); g.stroke(); });
        }
        if(plane && state.showCamHand) drawDropLine(g, cam.t, CAMCOL, plane, proj);
        g.fillStyle=CAMCOL; g.beginPath(); g.arc(Pc[0],Pc[1], it.dash?3:4,0,7); g.fill();
        if(!plan.anchorFirst){
          g.fillStyle='#8b98a8'; g.font='11px system-ui'; g.fillText(it.dash?'PRED cam':'cam', Pc[0]+5, Pc[1]-5);
        } else if(state.showCamHand){
          const cameraName=it.tag||(state.no_truth?'PRED':(state.rawOnly?'GT':(it.dash?'PRED':'GT')));
          cameraOverlays.push({cam,isPred:cameraName==='PRED'});
        }
      }
    }

    // 相机光心 → 左/右手腕的当前帧三维测距线。标签值取世界系欧氏距离，视角/缩放不改变它。
    if(plan.anchorFirst && state.showCamHand && cam.t){
      for(let h=0;h<2;h++){
        const J=d.joints[h][Math.min(cf,d.joints[h].length-1)];
        if(!J || !J[0]) continue;
        const wrist=tx(J[0],cam), dx=wrist[0]-cam.t[0], dy=wrist[1]-cam.t[1], dz=wrist[2]-cam.t[2];
        const dist=Math.hypot(dx,dy,dz), Pc=proj(cam.t,null), Ph=proj(wrist,null);
        g.save(); g.globalAlpha=.78; g.strokeStyle=handCol[h]; g.lineWidth=1.4;
        g.setLineDash([]);
        g.beginPath(); g.moveTo(Pc[0],Pc[1]); g.lineTo(Ph[0],Ph[1]); g.stroke();
        const prefix=items.length>1?(it.dash?'PRED ':'GT '):'';
        const label=`${prefix}${h===0?'L':'R'} ${dist.toFixed(1)} cm`;
        g.font='11px system-ui'; const tw=g.measureText(label).width;
        let lx=(Pc[0]+Ph[0])/2+5, ly=(Pc[1]+Ph[1])/2+(it.dash?11:-7);
        lx=Math.max(3,Math.min(W-tw-7,lx)); ly=Math.max(12,Math.min(H-8,ly));
        g.globalAlpha=.86; g.fillStyle='#0b0f15'; g.fillRect(lx-3,ly-10,tw+6,14);
        g.globalAlpha=1; g.fillStyle=handCol[h]; g.fillText(label,lx,ly); g.restore();
      }
    }

    // 手：手腕轨迹 + 当前帧骨架（骨架前先画地面落影与腕垂线，纵深才读得出来）
    if(plan.showHand){
      if(plan.showTraj!==false && state.showTraj) for(let h=0; h<2; h++){ const tr=d.traj[sides[h]]; if(!tr) continue;
        const pts=[]; forSampledTrajectory(tr,p=>pts.push(p?proj(p,cam):null));
        drawTrail(g, pts, handCol[h], 2, handTrajDash);
        const pc=tr[Math.min(cf,tr.length-1)];
        if(pc){ const P=proj(pc,cam); g.setLineDash([]); g.fillStyle=handCol[h]; g.beginPath(); g.arc(P[0],P[1],4,0,7); g.fill(); }
      }
      for(let h=0; h<2; h++){ const J=d.joints[h][Math.min(cf,d.joints[h].length-1)]; if(!J) continue;
        const T=J.map(p=>tx(p,cam));                       // 先转到显示系，落影与骨架共用
        if(plane){ drawGroundShadow(g, T, d.conn, plane, proj); drawDropLine(g, T[0], handCol[h], plane, proj); }
        drawHandSkeleton(g, T.map(p=>proj(p,null)), d.conn, handCol[h], dash);
      }
    }
  }

  for(const cue of cameraOverlays){
    drawCameraPose(cue.cam,cue.isPred);
  }

  // HUD：标题卡 / 图例卡 / 帧号 pill / 右下比例尺（与 Python 导出同布局，文案中文）
  g.setLineDash([]);
  const fname = plan.frame==='cam' ? '当前相机系 OpenCV(X右/Y下/Z前)'
              : (plan.anchorFirst
                ? (zUp?'固定世界系 Z-up(X右/Y前/Z上)':'固定世界系 OpenCV(X右/Y下/Z前)')
                : '世界系');
  const lay = state.no_truth ? 'PRED'
            : (state.layout==='overlay' ? 'GT实线 / PRED虚线' : (items[0].tag||''));
  const gridTxt = plane ? `　格 ${plane.step>=100 ? (plane.step/100)+' m' : plane.step+' cm'}` : '';
  drawChip(g, [{t:fname, c:'#e6edf3', s:13, b:true},
               {t:`${lay}　cm${gridTxt}`, c:'#8b98a8', s:11}], 12, 12, 'tl');
  const legend=[{t:'左手', c:'#4dd2ff', s:11}, {t:'右手', c:'#ffd34d', s:11}];
  if(plan.showCam) legend.push({t:'相机', c:'#ff8adf', s:11});
  drawChip(g, legend, 12, H-44, 'bl', true);
  drawChip(g, [{t:`帧 ${cf} / ${state.nframes-1}`, c:'#e6edf3', s:11}], 12, H-10, 'bl');
  drawScaleBar(g, W, H, scale);                         // 右下角比例尺：读实际位移距离
}

// ── 逐帧 loss 面板(随当前帧联动;按 config「输出头→loss」分组,行由后端 groups 动态生成)──
// 行不再写死:groups 来自 metrics(读 ckpt 自带 config 的 model/loss 段),config 启用了哪些头、
// 每头挂哪些 term,面板就列哪些 —— 数量/计算与训练一致(切窗也同 clip_len)。
let _lastMF = -1;
function _fmt(x){ return (x===null||x===undefined||Number.isNaN(x)) ? '—' : (+x).toFixed(4); }
const _R2D = 180/Math.PI;
// 值列格式:一律显 loss 原单位(弧度);deg 项额外括号补度数(便于观看,不改数值)。
function _fmtv(x, deg){
  if(x===null||x===undefined||Number.isNaN(x)) return '—';
  const s = (+x).toFixed(4);
  return deg ? `${s} (${(+x*_R2D).toFixed(2)}°)` : s;
}
function _pct(x){ return (x===null||x===undefined||Number.isNaN(x)) ? '—' : (+x).toFixed(1)+'%'; }
function _grpBadge(g){                        // 输出头启用/可用状态徽标
  if(!g.enabled) return '未启用';
  if(!g.available) return g.requires_hand ? '本 episode 无手 GT' : '不可用';
  return '';
}
function drawMetrics(){
  const el = document.getElementById('metrics'); if(!el) return;
  const M = state.metrics;
  if(!M){
    const reason = state.metricsError
      ? `逐帧 loss 不可用：${_esc(state.metricsError)}；预测与 2D/3D 可视化不受影响`
      : '该 episode 无逐帧 loss：非预算集或未加载预测 ckpt';
    el.innerHTML = `<div class="mnote">（${reason}）</div>`;
          _lastMF = -1; return; }
  const cf = frameOf(masterVideo); if(cf === _lastMF) return; _lastMF = cf;   // loss 表跟随第一块
  const pf = M.per_frame, mn = M.mean, W = M.weight || {};
  const cpf = (M.contrib && M.contrib.per_frame) || {}, cmn = (M.contrib && M.contrib.mean) || {};
  const groups = M.groups || [];
  const at = (o,k) => (o[k] ? o[k][Math.min(cf, o[k].length-1)] : null);
  const totC = at(cpf, 'total'), totM = cmn['total'];                          // 当前帧/均值 的加权总
  const share = (c,t) => (c==null || t==null || !isFinite(t) || t===0) ? null : 100*c/t;
  // 切窗信息:与训练同 clip_len/stride 才逐字一致(时序项不跨窗)。
  const win = M.clipwin ? `clip_len=${M.clip_len} stride=${M.stride} n_clips=${M.n_clips}（与训练同切窗）`
                        : `整段单窗（该段 ${M.nframes} 帧 &lt; clip_len=${M.clip_len}，非训练切窗）`;
  let h = `<div class="cap">切窗：${win}</div>`
        + `<table class="mt"><thead><tr><th>输出头 / loss 项</th><th>值·当前帧 ${cf}</th><th>值·均值</th>`
        + `<th>权重</th><th>加权·当前</th><th>占比·当前</th><th>占比·均值</th></tr></thead><tbody>`;
  for(const g of groups){
    const badge = _grpBadge(g);
    const head = `${g.label}${g.arch?` · ${g.arch}`:''}${badge?` · ${badge}`:''} · 组权重 ${_fmt(g.weight)}`;
    h += `<tr class="grp"><td colspan="7">${head}</td></tr>`;
    for(const t of (g.terms||[])){
      const key = t.key, nm = t.name + (t.normalized?' (norm)':(t.deg?' (rad)':'')), c = at(cpf,key);
      h += `<tr><td>${nm}</td><td class="v">${_fmtv(at(pf,key),t.deg)}</td><td>${_fmtv(mn[key],t.deg)}</td>`
         + `<td>${_fmt(W[key])}</td><td class="v">${_fmt(c)}</td>`
         + `<td>${_pct(share(c,totC))}</td><td>${_pct(share(cmn[key],totM))}</td></tr>`;
    }
  }
  // 跨组总 loss（加权）：逐帧=各 term 加权贡献之和；均值=训练整批返回（权威）。
  h += `<tr class="tot"><td>总 loss（加权）</td><td class="v">${_fmt(totC)}</td><td>${_fmt(totM)}</td>`
     + `<td>—</td><td class="v">${_fmt(totC)}</td><td>100%</td><td>100%</td></tr>`;
  h += '</tbody></table>';
  el.innerHTML = h;
}

// ── 「整体」块下方综合数字表：相机(pos/欧拉/fov)+左右手(pos/欧拉/betas) 每帧 GT|PRED|Δ + 底部整段平均 ──
function _fmt3(a, p){ return a==null ? '—' : a.map(x=>(+x).toFixed(p)).join(' , '); }
function _delta(g, pd){ return (g&&pd) ? g.map((x,i)=>pd[i]-x) : null; }
function _bfmt(a){ return a ? a.map(x=>(+x).toFixed(2)).join(' ') : '—'; }   // betas 10 值一行
function _entTable(title, rec, cf, withFov){
  if(!rec || (!rec.gt && !rec.pred)) return '';
  const at = o => (o ? o[Math.min(cf, o.length-1)] : null);
  const g = rec.gt || {}, p = rec.pred || {};
  const gpos=at(g.pos), ppos=at(p.pos), geul=at(g.eul), peul=at(p.eul);
  let h = `<table class="nt"><thead><tr><th>${title}</th><th>GT</th><th>PRED</th><th>Δ</th></tr></thead><tbody>`;
  h += `<tr><td>位置 XYZ (cm)</td><td>${_fmt3(gpos,2)}</td><td>${_fmt3(ppos,2)}</td><td class="d">${_fmt3(_delta(gpos,ppos),2)}</td></tr>`;
  h += `<tr><td>欧拉 XYZ (°)</td><td>${_fmt3(geul,1)}</td><td>${_fmt3(peul,1)}</td><td class="d">${_fmt3(_delta(geul,peul),1)}</td></tr>`;
  if(withFov){ const gf=at(g.fov), pf=at(p.fov);
    h += `<tr><td>FoV (°)</td><td>${_fmt3(gf,2)}</td><td>${_fmt3(pf,2)}</td><td class="d">${_fmt3(_delta(gf,pf),2)}</td></tr>`; }
  if(g.betas || p.betas){   // betas 也分 GT/PRED 两列（与位置/欧拉对齐），不再上下叠
    h += `<tr><td>betas(10)</td><td class="betas">${_bfmt(at(g.betas))}</td><td class="betas">${_bfmt(at(p.betas))}</td><td>—</td></tr>`;
  }
  return h + '</tbody></table>';
}
// 整段平均（固定，不随帧）：GT/PRED 的 FoV 与左右手 betas 平均——与上面「每帧」列对比看抖动。
function _meanTable(mean){
  if(!mean) return '';
  let h = `<table class="nt"><thead><tr><th>整段平均</th><th>GT</th><th>PRED</th></tr></thead><tbody>`;
  h += `<tr><td>FoV (°)</td><td>${_fmt3(mean.gt_fov,2)}</td><td>${_fmt3(mean.pred_fov,2)}</td></tr>`;
  const gb=mean.gt_betas||{}, pb=mean.pred_betas||{};
  h += `<tr><td>左手 betas</td><td class="betas">${_bfmt(gb.left)}</td><td class="betas">${_bfmt(pb.left)}</td></tr>`;
  h += `<tr><td>右手 betas</td><td class="betas">${_bfmt(gb.right)}</td><td class="betas">${_bfmt(pb.right)}</td></tr>`;
  return h + '</tbody></table>';
}
// 整段平均（仅 betas，相机系面板用）：betas 与坐标系无关，不含世界系相机 FoV。
function _meanBetasTable(mean){
  if(!mean) return '';
  const gb=mean.gt_betas||{}, pb=mean.pred_betas||{};
  let h = `<table class="nt"><thead><tr><th>整段平均</th><th>GT</th><th>PRED</th></tr></thead><tbody>`;
  h += `<tr><td>左手 betas</td><td class="betas">${_bfmt(gb.left)}</td><td class="betas">${_bfmt(pb.left)}</td></tr>`;
  h += `<tr><td>右手 betas</td><td class="betas">${_bfmt(gb.right)}</td><td class="betas">${_bfmt(pb.right)}</td></tr>`;
  return h + '</tbody></table>';
}
// 世界系(frame='world')：相机+左右手+FoV+平均；相机系(frame='cam')：仅左右手(手腕原始 transl_cam/orient6d)+betas 平均。
function drawIntegratedNums(b){
  const el = b.bnums; if(!el) return;
  const N = state.nums, cf = frameOf(masterVideo);   // 跟随首个可见视频帧
  if(b._lastNumF === cf && b._numsOn === !!N) return;   // 帧未变且数据态未变 → 免重建
  b._lastNumF = cf; b._numsOn = !!N;
  if(!N){ el.innerHTML=''; return; }
  if(b.frame==='cam'){
    const hc = N.hand_cam || {};
    el.innerHTML = `<div class="ncap">逐帧数值（相机系）· 帧 ${cf}</div>`
      + _entTable('左手', hc.left, cf)
      + _entTable('右手', hc.right, cf)
      + _meanBetasTable(N.mean);
  }else{
    el.innerHTML = `<div class="ncap">逐帧数值（世界系）· 帧 ${cf}</div>`
      + _entTable('相机', N.cam, cf, true)
      + _entTable('左手', N.hand && N.hand.left, cf)
      + _entTable('右手', N.hand && N.hand.right, cf)
      + _meanTable(N.mean);
  }
}

// ── 数据集分析：视频、LeRobot 内参与多样性统计独立运行并分别缓存。──
let _analysisDirPath='', _analysisDirParent='', _analysisDirTarget='single', _analysisPolling=false;
let _analysisPageLoading=false, _analysisLoadedKey='', _analysisSearchTimer=null;

function _analysisCleanPath(path){
  const value=String(path||'').trim();
  return value==='/'?value:value.replace(/\/+$/,'');
}

function _analysisAddDiversityPath(path){
  const clean=_analysisCleanPath(path); if(!clean) return false;
  const current=(state.analysis.input_dirs||[]).map(_analysisCleanPath);
  if(!current.includes(clean)) state.analysis.input_dirs=[...current,clean];
  const input=$('#analysisDiversityPathInput'); if(input) input.value='';
  renderAnalysisDiversityPaths();
  return true;
}

function renderAnalysisDiversityPaths(){
  const root=$('#analysisDiversityPathList'); if(!root) return;
  const paths=(state.analysis.input_dirs||[]).map(_analysisCleanPath).filter(Boolean);
  if(!paths.length){
    root.innerHTML='<div class="analysis-diversity-path-empty">尚未添加目录，请从默认路径、目录浏览器或绝对路径添加。</div>';
    return;
  }
  const choices=state.analysis.diversity_choices||[];
  const colors={ego4d:'#58d5d2',egodex:'#ff9e64',epickitchen:'#a1e26e'};
  root.innerHTML=paths.map((path,index)=>{
    const choice=choices.find(item=>_analysisCleanPath(item.path)===path);
    const basename=path.split('/').filter(Boolean).pop()||'/';
    const label=choice?.label||basename;
    const color=colors[choice?.dataset]||'#8da9a0';
    return `<div class="analysis-diversity-path-item" style="--dataset-color:${color}"><i></i><span><b>${_esc(label)}</b><code title="${_esc(path)}">${_esc(path)}</code></span><button type="button" data-diversity-path-remove="${index}" title="移除此目录">移除</button></div>`;
  }).join('');
  root.querySelectorAll('[data-diversity-path-remove]').forEach(button=>button.onclick=()=>{
    const index=+button.dataset.diversityPathRemove;
    state.analysis.input_dirs=paths.filter((_,itemIndex)=>itemIndex!==index);
    renderAnalysisDiversityPaths();
  });
}

function _analysisOpenDirPicker(target,startPath){
  const picker=$('#analysisDirPicker'), choose=$('#analysisDirChoose'); if(!picker) return;
  _analysisDirTarget=target;
  if(choose) choose.textContent=target==='diversity'?'＋ 添加此目录':'✓ 选择此目录';
  picker.hidden=false;
  analysisDirGo(startPath||inCur||'/');
}

const _analysisNumber = value => (value==null || !Number.isFinite(+value)) ? '—' : Math.round(+value).toLocaleString('zh-CN');
function _analysisDuration(seconds){
  if(seconds==null || !Number.isFinite(+seconds)) return '—';
  const value=+seconds;
  if(value>=3600) return (value/3600).toFixed(value>=36000?1:2)+' h';
  if(value>=60) return (value/60).toFixed(value>=600?1:2)+' min';
  return value.toFixed(value<10?2:1)+' s';
}
function _analysisBytes(bytes){
  if(bytes==null || !Number.isFinite(+bytes)) return '—';
  const units=['B','KiB','MiB','GiB','TiB']; let value=+bytes, index=0;
  while(value>=1024 && index<units.length-1){ value/=1024; index++; }
  return value.toFixed(index===0?0:value>=100?0:value>=10?1:2)+' '+units[index];
}
function _analysisBitrate(value){
  if(value==null || !Number.isFinite(+value)) return '—';
  return (+value/1e6).toFixed(+value>=1e7?1:2)+' Mbps';
}
function _analysisPct(value){ return value==null || !Number.isFinite(+value) ? '—' : (+value).toFixed(1)+'%'; }

async function analysisDirGo(path){
  const list=$('#analysisDirList'), pathEl=$('#analysisDirPath'), up=$('#analysisDirUp');
  if(!list || !pathEl || !up) return;
  list.textContent='读取目录…';
  try{
    const data=await getJSON('/api/dirbrowse?path='+encodeURIComponent(path||''));
    _analysisDirPath=data.path||path||'/'; _analysisDirParent=data.parent||_analysisDirPath;
    pathEl.textContent=_analysisDirPath; pathEl.title=_analysisDirPath;
    up.disabled=!_analysisDirParent || _analysisDirParent===_analysisDirPath;
    list.innerHTML='';
    for(const name of (data.dirs||[])){
      const button=document.createElement('button');
      button.type='button'; button.textContent='📁 '+name; button.title=name;
      button.onclick=()=>analysisDirGo((_analysisDirPath==='/'?'':_analysisDirPath)+'/'+name);
      list.appendChild(button);
    }
    if(!(data.dirs||[]).length) list.innerHTML='<span class="bmut">（没有子目录，可直接选择当前目录）</span>';
    if(data.truncated){ const note=document.createElement('span'); note.className='bmut'; note.textContent='仅显示前 2000 个子目录'; list.appendChild(note); }
  }catch(error){ list.textContent='读取失败：'+error.message; }
}

function wireDatasetAnalysis(){
  const button=$('#datasetAnalysisBtn'), launcher=$('#datasetAnalysisLauncher');
  const close=$('#analysisClose'), browse=$('#analysisBrowse');
  if(button) button.onclick=()=>{
    if(!launcher) return;
    launcher.classList.toggle('open');
    button.setAttribute('aria-expanded',launcher.classList.contains('open')?'true':'false');
  };
  document.querySelectorAll('[data-analysis-type]').forEach(option=>option.onclick=()=>{
    selectDatasetAnalysis(option.dataset.analysisType||'video');
    if(launcher) launcher.classList.remove('open');
    option.blur();
    if(button){ button.setAttribute('aria-expanded','false'); button.blur(); }
  });
  if(close) close.onclick=()=>showDatasetAnalysis(false);
  const useCurrent=$('#analysisUseCurrent');
  if(useCurrent) useCurrent.onclick=()=>{
    const input=$('#analysisInput'); if(input) input.value=inCur||'';
    const picker=$('#analysisDirPicker'); if(picker) picker.hidden=true;
  };
  if(browse) browse.onclick=()=>{
    const input=$('#analysisInput');
    _analysisOpenDirPicker('single',(input&&input.value.trim())||inCur||'/');
  };
  const diversityBrowse=$('#analysisDiversityBrowse');
  if(diversityBrowse) diversityBrowse.onclick=()=>{
    const paths=state.analysis.input_dirs||[];
    _analysisOpenDirPicker('diversity',paths[paths.length-1]||state.analysis.diversity_root||inCur||'/');
  };
  const diversityUseCurrent=$('#analysisDiversityUseCurrent');
  if(diversityUseCurrent) diversityUseCurrent.onclick=()=>_analysisAddDiversityPath(inCur||'');
  const diversityPathInput=$('#analysisDiversityPathInput'), diversityPathAdd=$('#analysisDiversityPathAdd');
  const addTypedPath=()=>_analysisAddDiversityPath(diversityPathInput?.value||'');
  if(diversityPathAdd) diversityPathAdd.onclick=addTypedPath;
  if(diversityPathInput) diversityPathInput.onkeydown=event=>{
    if(event.key==='Enter'){ event.preventDefault(); addTypedPath(); }
  };
  const up=$('#analysisDirUp'), choose=$('#analysisDirChoose'), pickerClose=$('#analysisDirClose');
  if(up) up.onclick=()=>analysisDirGo(_analysisDirParent);
  if(choose) choose.onclick=()=>{
    if(_analysisDirTarget==='diversity') _analysisAddDiversityPath(_analysisDirPath);
    else { const input=$('#analysisInput'); if(input) input.value=_analysisDirPath; }
    const picker=$('#analysisDirPicker'); if(picker) picker.hidden=true;
  };
  if(pickerClose) pickerClose.onclick=()=>{ const picker=$('#analysisDirPicker'); if(picker) picker.hidden=true; };
  const run=$('#analysisRun'), cancel=$('#analysisCancel');
  if(run) run.onclick=startDatasetAnalysis;
  if(cancel) cancel.onclick=cancelDatasetAnalysis;
  const filterIds=['analysisAnomaly','analysisCodec','analysisResolution','analysisOrientation','analysisPageSize'];
  for(const id of filterIds){
    const control=$('#'+id); if(!control) continue;
    control.onchange=()=>{
      state.analysis.page=1;
      state.analysis.anomaly=$('#analysisAnomaly').value;
      state.analysis.codec=$('#analysisCodec').value;
      state.analysis.resolution=$('#analysisResolution').value;
      state.analysis.orientation=$('#analysisOrientation').value;
      state.analysis.page_size=+$('#analysisPageSize').value||100;
      loadAnalysisPage();
    };
  }
  const search=$('#analysisSearch');
  if(search) search.oninput=()=>{
    clearTimeout(_analysisSearchTimer);
    _analysisSearchTimer=setTimeout(()=>{
      state.analysis.search=search.value||''; state.analysis.page=1; loadAnalysisPage();
    },260);
  };
  const prev=$('#analysisPrev'), next=$('#analysisNext');
  if(prev) prev.onclick=()=>{ if(state.analysis.page>1){ state.analysis.page--; loadAnalysisPage(); } };
  if(next) next.onclick=()=>{ if(state.analysis.page<state.analysis.pages){ state.analysis.page++; loadAnalysisPage(); } };
  document.querySelectorAll('[data-analysis-sort]').forEach(header=>header.onclick=()=>{
    const sort=header.dataset.analysisSort;
    if(state.analysis.sort===sort) state.analysis.order=state.analysis.order==='asc'?'desc':'asc';
    else { state.analysis.sort=sort; state.analysis.order='asc'; }
    state.analysis.page=1; loadAnalysisPage();
  });
  renderDatasetAnalysis();
}

function selectDatasetAnalysis(analysisType){
  const type=analysisType==='intrinsics'?'intrinsics':analysisType==='diversity'?'diversity':'video';
  if(state.analysis.running && state.analysis.analysis_type!==type) return;
  const changed=state.analysis.analysis_type!==type;
  state.analysis={...state.analysis,analysis_type:type,
    ...(changed?{stage:'idle',phase:'待分析',error:null,cancelled:false,cached:false,
      result_ready:false,summary:null,rows:[],page:1,pages:1,page_total:0,filters:null}: {})};
  const input=$('#analysisInput');
  if(input && type!=='diversity' && !input.value) input.value=inCur||'';
  const workersInput=$('#analysisWorkers');
  if(workersInput && changed) workersInput.value=type==='video'?'32':'8';
  _analysisLoadedKey='';
  showDatasetAnalysis(true);
  renderDatasetAnalysis();
}

function showDatasetAnalysis(on){
  if(on) state.hidden.delete('analysis'); else state.hidden.add('analysis');
  buildPanels(true);
  if(on){
    const section=document.querySelector('#blocks section[data-pid="analysis"]');
    if(section) section.scrollIntoView({behavior:'smooth',block:'start'});
  }
}

async function startDatasetAnalysis(){
  if(state.analysis.running) return;
  const selectedType=state.analysis.analysis_type;
  const analysisType=selectedType==='intrinsics'?'intrinsics':selectedType==='diversity'?'diversity':'video';
  const input=($('#analysisInput').value||'').trim();
  const inputDirs=(state.analysis.input_dirs||[]).map(_analysisCleanPath).filter(Boolean);
  const defaultWorkers=analysisType==='video'?32:8;
  const workers=Math.max(1,Math.min(32,+($('#analysisWorkers').value||defaultWorkers)));
  const sampleFiles=Math.max(1,Math.min(96,+($('#analysisSampleFiles')?.value||24)));
  if(analysisType!=='diversity'&&!input){
    state.analysis={...state.analysis,error:'请选择或输入数据集目录',phase:'未启动'};
    renderDatasetAnalysis(); return;
  }
  if(analysisType==='diversity'&&!inputDirs.length){
    state.analysis={...state.analysis,error:'多样性分析至少添加一个数据集目录',phase:'未启动'};
    renderDatasetAnalysis(); return;
  }
  state.analysis={...state.analysis,running:true,
    stage:analysisType==='intrinsics'?'discovering_intrinsics':analysisType==='diversity'?'discovering_diversity':'scanning',
    phase:'启动中…',input_dir:analysisType==='diversity'?(inputDirs[0]||''):input,
    input_dirs:analysisType==='diversity'?inputDirs:[],analysis_type:analysisType,
    selected_datasets:[],sample_files:sampleFiles,
    workers,discovered:0,total:0,done:0,failed:0,current:'',error:null,cancelled:false,cached:false,
    result_ready:false,summary:null,rows:[],page:1,pages:1,page_total:0,filters:null,
    search:'',anomaly:'',codec:'',resolution:'',orientation:'',sort:'relative_path',order:'asc'};
  if($('#analysisSearch')) $('#analysisSearch').value='';
  for(const id of ['analysisAnomaly','analysisCodec','analysisResolution','analysisOrientation']){
    const control=$('#'+id); if(control) control.value='';
  }
  _analysisLoadedKey=''; renderDatasetAnalysis();
  try{
    const response=await fetch(U('/api/dataset-analysis/start'),{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({input_dir:input,workers,refresh:!!$('#analysisRefresh').checked,
        analysis_type:analysisType,input_dirs:analysisType==='diversity'?inputDirs:undefined,
        sample_files:sampleFiles})});
    const result=await response.json().catch(()=>({}));
    if(!response.ok || !result.ok) throw new Error(result.error||'无法启动分析');
    startDatasetAnalysisPoll();
  }catch(error){
    state.analysis={...state.analysis,running:false,stage:'error',phase:'启动失败',error:error.message};
    renderDatasetAnalysis();
  }
}

async function cancelDatasetAnalysis(){
  try{ await fetch(U('/api/dataset-analysis/cancel'),{method:'POST'}); }catch(error){}
}

async function startDatasetAnalysisPoll(){
  if(_analysisPolling) return;
  _analysisPolling=true;
  while(_analysisPolling){
    try{
      const status=await getJSON('/api/dataset-analysis/status');
      state.analysis={...state.analysis,...status};
      if(status.summary) state.analysis.summary=status.summary;
      renderDatasetAnalysis();
      if(status.running && state.hidden.has('analysis')) showDatasetAnalysis(true);
      if(status.result_ready && status.cache_key!==_analysisLoadedKey){
        _analysisLoadedKey=status.cache_key||'latest';
        state.analysis.page=1;
        if(status.analysis_type==='video') await loadAnalysisPage();
      }
      if(!status.running){ _analysisPolling=false; break; }
    }catch(error){}
    await new Promise(resolve=>setTimeout(resolve,650));
  }
}

async function loadAnalysisPage(){
  if(_analysisPageLoading || !state.analysis.result_ready || state.analysis.analysis_type!=='video') return;
  _analysisPageLoading=true;
  const a=state.analysis;
  const params=new URLSearchParams({page:String(a.page||1),page_size:String(a.page_size||100),
    search:a.search||'',anomaly:a.anomaly||'',codec:a.codec||'',resolution:a.resolution||'',
    orientation:a.orientation||'',sort:a.sort||'relative_path',order:a.order||'asc'});
  try{
    const data=await getJSON('/api/dataset-analysis/result?'+params.toString());
    state.analysis={...state.analysis,rows:data.rows||[],page:data.page||1,pages:data.pages||1,
      page_size:data.page_size||a.page_size,page_total:data.total||0,filters:data.filters||a.filters};
    syncAnalysisFilters();
    renderAnalysisTable();
  }catch(error){
    state.analysis={...state.analysis,error:'读取明细失败：'+error.message};
    renderDatasetAnalysis();
  }finally{ _analysisPageLoading=false; }
}

function _analysisSelectOptions(select, values, emptyLabel){
  if(!select) return;
  const selected=select.value;
  select.innerHTML=`<option value="">${_esc(emptyLabel)}</option>`+(values||[]).map(value=>`<option value="${_esc(value)}">${_esc(value)}</option>`).join('');
  if([...select.options].some(option=>option.value===selected)) select.value=selected;
}
function syncAnalysisFilters(){
  const filters=state.analysis.filters||{};
  _analysisSelectOptions($('#analysisCodec'),filters.codecs,'全部编码');
  _analysisSelectOptions($('#analysisResolution'),filters.resolutions,'全部分辨率');
  _analysisSelectOptions($('#analysisOrientation'),filters.orientations,'全部方向');
  const anomaly=$('#analysisAnomaly');
  if(anomaly){
    const selected=state.analysis.anomaly||anomaly.value;
    anomaly.innerHTML='<option value="">全部质量状态</option><option value="any">存在诊断项</option><option value="clean">无诊断项</option><option value="failed">解析失败</option>'
      +(filters.anomalies||[]).filter(item=>item.code!=='probe_failed').map(item=>`<option value="${_esc(item.code)}">${_esc(item.label)} (${item.count})</option>`).join('');
    anomaly.value=selected;
  }
  if($('#analysisCodec')) $('#analysisCodec').value=state.analysis.codec||'';
  if($('#analysisResolution')) $('#analysisResolution').value=state.analysis.resolution||'';
  if($('#analysisOrientation')) $('#analysisOrientation').value=state.analysis.orientation||'';
}

function _analysisChart(title, items, limit=9){
  const visible=(items||[]).slice(0,limit), max=Math.max(1,...visible.map(item=>+item.percent||0));
  if(!visible.length) return '';
  return `<section class="analysis-chart"><header><b>${_esc(title)}</b><span>${(items||[]).length} 类</span></header><div>`
    +visible.map(item=>`<span class="analysis-bar" title="${_esc(item.label)} · ${item.count} · ${_analysisPct(item.percent)}"><i>${_esc(item.label)}</i>`
      +`<em><u style="width:${Math.max(1,(+item.percent||0)/max*100)}%"></u></em><b>${_analysisNumber(item.count)}</b><small>${_analysisPct(item.percent)}</small></span>`).join('')
    +'</div></section>';
}

function _analysisFixed(value,digits=2,suffix=''){
  return value==null||!Number.isFinite(+value)?'—':(+value).toFixed(digits)+suffix;
}

function renderAnalysisIntrinsics(intrinsic){
  const root=$('#analysisIntrinsics'); if(!root) return;
  intrinsic=intrinsic||{};
  if(!intrinsic.available){
    root.classList.add('unavailable');
    root.innerHTML=`<div class="analysis-intrinsic-empty"><b>没有可统计的相机内参</b><span>${_esc(intrinsic.reason||'MP4/ffprobe 不包含标准相机内参；如为 LeRobot 数据，请从包含 meta/info.json 的数据集目录启动分析。')}</span></div>`;
    return;
  }
  root.classList.remove('unavailable');
  const stats=intrinsic.statistics||{}, dist=intrinsic.distributions||{};
  const fv=stats.fov_vertical_deg||{}, fh=stats.fov_horizontal_deg||{};
  const fx=stats.fx_px||{}, fy=stats.fy_px||{}, ratio=stats.fx_over_fy||{};
  const cards=[
    ['采样策略','episode 首帧',`${_analysisNumber(intrinsic.sampled_frames)} 个 FOV 样本`],
    ['有效 episode',`${_analysisNumber(intrinsic.valid_episodes)} / ${_analysisNumber(intrinsic.total_episodes)}`,`${_analysisPct(intrinsic.valid_episode_percent)} 覆盖`],
    ['垂直 FoV',_analysisFixed(fv.median,2,'°'),`P25 ${_analysisFixed(fv.p25,1,'°')} · P95 ${_analysisFixed(fv.p95,1,'°')}`],
    ['水平 FoV',_analysisFixed(fh.median,2,'°'),`P25 ${_analysisFixed(fh.p25,1,'°')} · P95 ${_analysisFixed(fh.p95,1,'°')}`],
    ['焦距 fx / fy',`${_analysisFixed(fx.median,1)} / ${_analysisFixed(fy.median,1)} px`,'由 FOV + 分辨率反解'],
    ['像素焦距比 fx/fy',_analysisFixed(ratio.median,4),`${_analysisNumber(intrinsic.annotated_parquet_files)} 个标注分片`],
  ];
  const charts=[['垂直 FoV',dist.fov_vertical_deg],['水平 FoV',dist.fov_horizontal_deg],
    ['常见 FoV 组合',dist.fov_pairs],['fx / 图宽',dist.fx_over_width],
    ['fy / 图高',dist.fy_over_height],['fx / fy',dist.fx_over_fy]]
    .map(([title,items])=>_analysisChart(title,items,10)).join('');
  const diagnostics=intrinsic.diagnostics||[];
  const diagnosticHtml=diagnostics.length?diagnostics.map(item=>`<span class="analysis-issue ${_esc(item.severity||'warning')}"><i>${_esc(item.label)}</i><b>${_analysisNumber(item.count)} ${_esc(item.unit||'')}</b><small>${_analysisPct(item.percent)}</small></span>`).join(''):'<span class="analysis-clean">✓ episode 首帧 FOV 有效性和焦距比例未发现异常</span>';
  const assumptions=intrinsic.assumptions||{};
  root.innerHTML=`<div class="analysis-intrinsic-kpis">${cards.map(([label,value,note],index)=>`<article style="--k:${index}"><span>${_esc(label)}</span><b>${_esc(value)}</b><small>${_esc(note)}</small></article>`).join('')}</div>`
    +`<div class="analysis-charts">${charts}</div>`
    +`<div class="analysis-intrinsic-foot"><section><b>内参诊断</b><div class="analysis-quality">${diagnosticHtml}</div></section>`
    +`<section><b>字段口径</b><div class="analysis-assumptions"><span><i>FoV</i><em>每个 episode 仅取 frame_index=0；顺序为垂直、水平</em></span><span><i>cx / cy</i><em>${_esc(assumptions.principal_point||'未提供')}</em></span><span><i>skew</i><em>${_esc(assumptions.skew||'未提供')}</em></span><span><i>畸变</i><em>${_esc(assumptions.distortion||'未提供')}</em></span></div></section></div>`;
}

function _analysisDiversityQ(dataset,key,q='p50',digits=2){
  const value=dataset.motion?.[key]?.[q];
  return value==null||!Number.isFinite(+value)?'—':(+value).toFixed(digits);
}
function _analysisTopCount(values){
  const rows=Object.entries(values||{}).sort((a,b)=>b[1]-a[1]);
  return rows.length?rows[0][0]:'—';
}
function renderAnalysisDiversity(summary){
  const root=$('#analysisDiversity'); if(!root) return;
  const overview=summary.overview||{}, datasets=summary.datasets||[], dist=summary.distributions||{};
  const sampledEpisodes=datasets.reduce((total,item)=>total+(item.motion?.sampled_episodes||0),0);
  const cards=[
    ['所选数据集',_analysisNumber(overview.datasets),datasets.map(item=>item.label).join(' + ')],
    ['总时长',_analysisFixed(overview.hours,1,' h'),`${_analysisNumber(overview.frames)} 帧`],
    ['Episodes',_analysisNumber(overview.episodes),`${_analysisNumber(overview.videos)} 源视频`],
    ['任务 ID',_analysisNumber(overview.task_ids),`${_analysisNumber(overview.normalized_unique)} 集内唯一描述`],
    ['语义场景',_analysisNumber(overview.scene_kinds),'自然语言规则化语义域'],
    ['动作族',_analysisNumber(overview.action_kinds),'一个标签可命中多个动作'],
    ['轨迹样本',_analysisNumber(sampledEpisodes),`${_analysisNumber(summary.sample_files)} parquet / 数据集`],
    ['统计缓存',summary.cache_key?'已建立':'—','源文件变化后自动失效'],
  ];
  const totalFrames=Math.max(1,+overview.frames||0);
  const mix=datasets.map(item=>`<i title="${_esc(item.label)} · ${_analysisPct(item.frames/totalFrames*100)}" `+
    `style="width:${item.frames/totalFrames*100}%;background:${_esc(item.color)}"></i>`).join('');
  const legend=datasets.map(item=>`<span><i class="analysis-diversity-dot" style="background:${_esc(item.color)}"></i>`+
    `<b>${_esc(item.label)}</b>${_analysisPct(item.frames/totalFrames*100)} · ${_analysisFixed(item.hours,1,'h')}</span>`).join('');
  const charts=[['数据构成 · 帧',dist.dataset_frames,12],['任务语义场景',dist.scenes,24],
    ['动作族 · 多标签',dist.actions,21],['相机活动范围',dist.activity,12],['有效手监督',dist.hand_usage,12]]
    .map(([title,items,limit])=>_analysisChart(title,items,limit)).join('');
  const rows=datasets.map(item=>{
    const text=item.text||{}, motion=item.motion||{}, hands=motion.hand_usage_counts||{};
    const handTotal=Object.values(hands).reduce((a,b)=>a+(+b||0),0)||1;
    return `<tr><td><i class="analysis-diversity-dot" style="display:inline-block;margin-right:5px;background:${_esc(item.color)}"></i>${_esc(item.label)}</td>`+
      `<td>${_analysisFixed(item.hours,1)}</td><td>${_analysisNumber(item.episodes)}</td>`+
      `<td>${_analysisNumber(text.task_ids)}</td><td>${_analysisPct((text.normalized_unique_ratio||0)*100)}</td>`+
      `<td>${_esc(_analysisTopCount(text.scene_counts))}</td><td>${_esc(_analysisTopCount(text.action_counts))}</td>`+
      `<td>${_analysisPct((text.action_coverage||0)*100)}</td><td>${_analysisNumber(motion.sampled_episodes)}</td>`+
      `<td>${_analysisDiversityQ(item,'camera_span_m')} / ${_analysisDiversityQ(item,'camera_span_m','p90')}</td>`+
      `<td>${_analysisDiversityQ(item,'net_displacement_m')} / ${_analysisDiversityQ(item,'net_displacement_m','p90')}</td>`+
      `<td>${_analysisDiversityQ(item,'head_sweep_deg','p50',1)} / ${_analysisDiversityQ(item,'head_sweep_deg','p90',1)}</td>`+
      `<td>${_analysisDiversityQ(item,'hand_workspace_m')} / ${_analysisDiversityQ(item,'hand_workspace_m','p90')}</td>`+
      `<td>${_analysisPct((hands['双手']||0)/handTotal*100)}</td></tr>`;
  }).join('');
  const terms=datasets.map(item=>`<div style="--dataset-color:${_esc(item.color)}"><b>${_esc(item.label)}</b>`+
    `${(item.text?.top_entity_terms||[]).map(term=>`<span class="analysis-diversity-term"><b>${_esc(term.term)}</b> ${_analysisNumber(term.count)}</span>`).join('')}</div>`).join('');
  const methodology=summary.methodology||{};
  root.innerHTML=`<div class="analysis-kpis">${cards.map(([label,value,note],index)=>`<article style="--k:${index}"><span>${_esc(label)}</span><b>${_esc(value)}</b><small title="${_esc(note)}">${_esc(note)}</small></article>`).join('')}</div>`+
    `<div class="analysis-diversity-mix">${mix}</div><div class="analysis-diversity-legend">${legend}</div>`+
    `<div class="analysis-charts">${charts}</div>`+
    `<div class="analysis-section-head"><span><b>跨数据集活动范围对比</b><small>数值为中位数 / P90；距离单位 m，转角单位 °</small></span></div>`+
    `<div class="analysis-diversity-table-wrap"><table class="analysis-diversity-table"><thead><tr><th>数据集</th><th>小时</th><th>Episode</th><th>任务 ID</th><th>唯一描述率</th><th>首要场景</th><th>首要动作</th><th>动作覆盖</th><th>轨迹样本</th><th>相机跨度</th><th>净位移</th><th>头部转角</th><th>手工作区</th><th>双手帧</th></tr></thead><tbody>${rows}</tbody></table></div>`+
    `<div class="analysis-section-head"><span><b>高频视觉属性 / 实体词（近似）</b><small>每套最多稳定抽样 20 万条自然语言标签</small></span></div>`+
    `<div class="analysis-diversity-terms">${terms}</div>`+
    `<div class="analysis-diversity-method"><b>统计口径：</b> ${_esc(methodology.scene||'')} ${_esc(methodology.action||'')} ${_esc(methodology.motion||'')} ${_esc(methodology.unique||'')}</div>`;
}

function renderAnalysisSummary(){
  const summary=state.analysis.summary, root=$('#analysisResults');
  if(!root){ return; }
  root.hidden=!summary;
  if(!summary) return;
  const type=summary.analysis_type||state.analysis.analysis_type||'video';
  const videoRoot=$('#analysisVideoResults'), intrinsicRoot=$('#analysisIntrinsicResults');
  const diversityRoot=$('#analysisDiversityResults');
  if(videoRoot) videoRoot.hidden=type==='intrinsics'||type==='diversity';
  if(intrinsicRoot) intrinsicRoot.hidden=type==='video'||type==='diversity';
  if(diversityRoot) diversityRoot.hidden=type!=='diversity';
  if(type==='intrinsics'){
    renderAnalysisIntrinsics(summary.intrinsics||{});
    return;
  }
  if(type==='diversity'){
    renderAnalysisDiversity(summary);
    return;
  }
  const overview=summary.overview||{}, stats=summary.statistics||{}, duration=stats.duration_sec||{};
  const kpis=[
    ['视频',_analysisNumber(overview.total_files),`${_analysisNumber(overview.parsed_ok)} 可解析`],
    ['总时长',_analysisDuration(overview.total_duration_sec),`中位 ${_analysisDuration(overview.median_duration_sec)}`],
    ['总帧数',_analysisNumber(overview.total_frames),`${_analysisPct((summary.metadata_coverage||{}).exact_frame_count_percent)} 精确`],
    ['总容量',_analysisBytes(overview.total_size_bytes),`平均 ${_analysisBytes((stats.file_size_bytes||{}).mean)}`],
    ['FPS',overview.median_fps==null?'—':(+overview.median_fps).toFixed(2),`均值 ${(overview.mean_fps==null?'—':(+overview.mean_fps).toFixed(2))}`],
    ['质量项',_analysisNumber(overview.videos_with_anomalies),`${_analysisNumber(overview.parsed_failed)} 解析失败`],
    ['音频覆盖',_analysisPct((summary.metadata_coverage||{}).audio_percent),`${_analysisNumber(overview.with_audio)} 个含音频`],
    ['主格式覆盖',_analysisPct(overview.dominant_format_coverage),`${_analysisNumber(overview.duplicate_groups)} 组疑似重复`],
  ];
  $('#analysisKpis').innerHTML=kpis.map(([label,value,note],index)=>`<article style="--k:${index}"><span>${_esc(label)}</span><b>${_esc(value)}</b><small>${_esc(note)}</small></article>`).join('');
  const dist=summary.distributions||{};
  $('#analysisCharts').innerHTML=[['时长',dist.duration],['FPS',dist.fps],['分辨率',dist.resolution],
    ['视频编码',dist.codec],['像素格式',dist.pixel_format],['画面方向',dist.orientation],
    ['帧率模式',dist.frame_rate_mode],['容器',dist.container]].map(([title,items])=>_analysisChart(title,items)).join('');
  if(type==='combined') renderAnalysisIntrinsics(summary.intrinsics||{});
  const folders=(dist.folders||[]).slice(0,16), folderMax=Math.max(1,...folders.map(item=>item.files||0));
  $('#analysisFolders').innerHTML=folders.map(item=>`<span title="${_esc(item.label)}"><i>${_esc(item.label)}</i><em><u style="width:${Math.max(1,item.files/folderMax*100)}%"></u></em>`
    +`<b>${_analysisNumber(item.files)}</b><small>${_analysisDuration(item.duration_sec)} · ${_analysisBytes(item.size_bytes)}</small></span>`).join('')||'<span class="bmut">无目录统计</span>';
  const anomalies=dist.anomalies||[], coverage=summary.metadata_coverage||{};
  $('#analysisQuality').innerHTML=(anomalies.length?anomalies.map(item=>`<span class="analysis-issue ${_esc(item.severity||'warning')}"><i>${_esc(item.label)}</i><b>${_analysisNumber(item.count)}</b><small>${_analysisPct(item.percent)}</small></span>`).join(''):'<span class="analysis-clean">✓ 未发现规则可识别的异常</span>')
    +`<div class="analysis-coverage"><span>时长元数据 <b>${_analysisPct(coverage.duration_percent)}</b></span><span>FPS 元数据 <b>${_analysisPct(coverage.fps_percent)}</b></span>`
    +`<span>精确帧数 <b>${_analysisPct(coverage.exact_frame_count_percent)}</b></span><span>码率元数据 <b>${_analysisPct(coverage.bitrate_percent)}</b></span>`
    +`<span>创建时间 <b>${_analysisPct(coverage.creation_time_percent)}</b></span><span>时长 P95 <b>${_analysisDuration(duration.p95)}</b></span></div>`;
  renderAnalysisTable();
}

function renderAnalysisTable(){
  const body=$('#analysisTableBody'), count=$('#analysisTableCount'), pager=$('#analysisPage');
  if(!body) return;
  const labels=Object.fromEntries((((state.analysis.summary||{}).distributions||{}).anomalies||[]).map(item=>[item.code,item.label]));
  body.innerHTML=(state.analysis.rows||[]).map(row=>{
    const issueCodes=row.ok?(row.anomalies||[]):['probe_failed'];
    const issues=issueCodes.length?issueCodes.map(code=>`<span title="${_esc(code)}">${_esc(labels[code]||code)}</span>`).join(''):'<i class="analysis-ok">正常</i>';
    const audio=row.has_audio?`${_esc(row.audio_codec||'yes')}${row.audio_channels?' · '+row.audio_channels+'ch':''}`:'—';
    return `<tr class="${row.ok?'':'analysis-row-failed'}"><td class="analysis-path-cell" title="${_esc(row.path||'')}"><b>${_esc(row.relative_path||row.path||'')}</b><small>${_esc(row.folder||'')}</small></td>`
      +`<td>${_analysisDuration(row.duration_sec)}</td><td>${row.fps==null?'—':(+row.fps).toFixed(3)}${row.frame_rate_mode==='variable'?'<small class="analysis-vfr">VFR</small>':''}</td>`
      +`<td>${_analysisNumber(row.frame_count)}${row.frame_count_estimated?'<small>估算</small>':''}</td><td>${_esc(row.resolution||'—')}<small>${_esc(row.orientation||'')}</small></td>`
      +`<td>${_esc(row.codec||'—')}<small>${_esc(row.pixel_format||'')}</small></td><td>${_analysisBitrate(row.bitrate_bps)}</td><td>${_analysisBytes(row.size_bytes)}</td>`
      +`<td>${audio}</td><td class="analysis-issues">${issues}${row.error?`<small title="${_esc(row.error)}">${_esc(row.error)}</small>`:''}</td></tr>`;
  }).join('')||'<tr><td colspan="10" class="analysis-empty">没有符合当前筛选条件的视频</td></tr>';
  if(count) count.textContent=`筛选后 ${_analysisNumber(state.analysis.page_total)} 条`;
  if(pager) pager.textContent=`第 ${state.analysis.page||1} / ${state.analysis.pages||1} 页`;
  if($('#analysisPrev')) $('#analysisPrev').disabled=(state.analysis.page||1)<=1;
  if($('#analysisNext')) $('#analysisNext').disabled=(state.analysis.page||1)>=(state.analysis.pages||1);
  document.querySelectorAll('[data-analysis-sort]').forEach(header=>{
    header.classList.toggle('on',header.dataset.analysisSort===state.analysis.sort);
    header.dataset.order=header.dataset.analysisSort===state.analysis.sort?state.analysis.order:'';
  });
}

function renderDatasetAnalysis(){
  const a=state.analysis||{}, running=!!a.running;
  const analysisType=a.analysis_type==='intrinsics'?'intrinsics':a.analysis_type==='diversity'?'diversity':'video';
  const button=$('#datasetAnalysisBtn'), run=$('#analysisRun'), cancel=$('#analysisCancel');
  const input=$('#analysisInput');
  if(input && analysisType!=='diversity' && a.input_dir && !input.value) input.value=a.input_dir;
  if(button) button.classList.toggle('on',running||!state.hidden.has('analysis'));
  if(button) button.setAttribute('aria-expanded',$('#datasetAnalysisLauncher')?.classList.contains('open')?'true':'false');
  document.querySelectorAll('[data-analysis-type]').forEach(option=>{
    option.classList.toggle('on',option.dataset.analysisType===analysisType);
    option.disabled=running && option.dataset.analysisType!==analysisType;
  });
  const title=$('#analysisTitle'), subtitle=$('#analysisSubtitle'), workersLabel=$('#analysisWorkersLabel');
  if(title) title.textContent=analysisType==='intrinsics'?'内参分析':analysisType==='diversity'?'多样性分析':'视频分析';
  if(subtitle) subtitle.textContent=analysisType==='intrinsics'
    ?'每个 LeRobot episode 仅采样首帧 FOV，不运行 ffprobe'
    :analysisType==='diversity'
      ?'多选 LeRobot 数据集，统计自然语言场景、动作族、相机与手部活动范围'
      :'只读 ffprobe 媒体属性，不扫描 LeRobot FOV';
  if(workersLabel) workersLabel.textContent=analysisType==='intrinsics'?'Parquet 并发':'ffprobe 并发';
  const pathField=$('#analysisPathField'), workersField=$('#analysisWorkersField');
  const diversityOptions=$('#analysisDiversityOptions');
  if(pathField) pathField.hidden=analysisType==='diversity';
  if(workersField) workersField.hidden=analysisType==='diversity';
  if(diversityOptions) diversityOptions.hidden=analysisType!=='diversity';
  if(analysisType==='diversity'){
    renderAnalysisDiversityPaths();
    const sample=$('#analysisSampleFiles'); if(sample && a.sample_files) sample.value=String(a.sample_files);
  }
  if(run){
    run.textContent=analysisType==='intrinsics'?'▶ 开始内参分析':analysisType==='diversity'?'▶ 开始多样性分析':'▶ 开始视频分析';
    run.disabled=running; run.style.display=running?'none':'';
  }
  if(cancel) cancel.style.display=running?'':'none';
  document.querySelectorAll('#analysisWrap .analysis-form input, #analysisWrap .analysis-form button, #analysisWrap .analysis-form select')
    .forEach(control=>{ control.disabled=running; });
  const picker=$('#analysisDirPicker'); if(picker&&running) picker.hidden=true;
  const phase=$('#analysisPhase');
  if(phase){
    phase.className='step '+(a.error&&!running?'err':running?'busy':a.result_ready?'ok':a.cancelled?'warn':'');
    phase.textContent=a.error&&!running?'✗ '+(a.phase||'分析失败')+'：'+a.error:(a.phase||'待分析');
  }
  const progress=$('#analysisProg'), bar=$('#analysisBar');
  if(progress){
    if(running){
      progress.style.display='inline-block';
      if(a.total>0&&(a.stage==='probing'||a.stage==='intrinsics'||a.stage?.startsWith('diversity_'))){ progress.classList.remove('indet'); if(bar) bar.style.width=Math.round((a.done||0)/a.total*100)+'%'; }
      else progress.classList.add('indet');
    }else if(a.result_ready){ progress.style.display='inline-block'; progress.classList.remove('indet'); if(bar) bar.style.width='100%'; }
    else progress.style.display='none';
  }
  const stats=$('#analysisStats');
  const unit=analysisType==='intrinsics'?'Parquet ':analysisType==='diversity'?'阶段 ':'视频 ';
  if(stats) stats.textContent=a.total?`${unit}${_analysisNumber(a.done)}/${_analysisNumber(a.total)} · 失败 ${_analysisNumber(a.failed)}`
    :(a.discovered?`已发现 ${_analysisNumber(a.discovered)} 个${analysisType==='video'?'视频':' LeRobot 数据集'}`:'');
  const current=$('#analysisCurrent');
  if(current) current.innerHTML=a.current?`<span>${a.stage==='scanning'||a.stage==='discovering_intrinsics'||a.stage==='discovering_diversity'?'扫描':a.stage==='intrinsics'||a.stage==='diversity_motion'?'采样':a.stage==='diversity_labels'?'标签':'处理'}</span><code>${_esc(a.current)}</code>`:'';
  const cache=$('#analysisCacheBadge');
  if(cache){ cache.hidden=!a.result_ready; cache.textContent=a.cached?'命中缓存':'新鲜分析'; cache.classList.toggle('cached',!!a.cached); }
  const exports=$('#analysisExport');
  if(exports){
    exports.hidden=!a.result_ready;
    exports.querySelectorAll('[data-analysis-export]').forEach(link=>{
      link.hidden=analysisType==='intrinsics'&&link.dataset.analysisExport==='csv';
      link.href=U('/api/dataset-analysis/export?format='+link.dataset.analysisExport);
      link.setAttribute('download','');
    });
  }
  renderAnalysisSummary();
}

// ── 批量推理：单独选 Run/Step；启动后服务端切换并加载，单视频复用全部常驻 GPU 副本。──
function _batchDefaultOutput(input){
  const clean = String(input||'').replace(/\/+$/, '');
  return clean ? clean+'_predictions' : '';
}
let _batchModelsLoaded=false, _batchModelsLoading=false;
function _batchSetOptions(select, values, placeholder){
  if(!select) return;
  select.innerHTML='';
  if(!values.length){
    const opt=document.createElement('option'); opt.value=''; opt.textContent=placeholder; select.appendChild(opt);
    return;
  }
  for(const value of values){
    const opt=document.createElement('option'); opt.value=value; opt.textContent=value; select.appendChild(opt);
  }
}
async function loadBatchSteps(run, preferredStep=''){
  const stepSel=$('#batchModelStep');
  if(!stepSel) return;
  if(!run){ _batchSetOptions(stepSel, [], '请先选择 Run'); renderBatch(); return; }
  stepSel.disabled=true;
  try{
    const data=await getJSON('/api/ckpts/'+_enc(run));
    const steps=data.steps||[];
    _batchSetOptions(stepSel, steps, '该 Run 没有 checkpoint');
    stepSel.value=steps.includes(preferredStep) ? preferredStep : (steps[steps.length-1]||'');
  }catch(e){
    _batchSetOptions(stepSel, [], '读取 checkpoint 失败');
    state.batch={...state.batch, error:'读取批量模型失败：'+e.message};
  }finally{
    stepSel.disabled=!!state.batch.running;
    renderBatch();
  }
}
async function loadBatchModels(){
  if(_batchModelsLoaded || _batchModelsLoading) return;
  _batchModelsLoading=true;
  const runSel=$('#batchModelRun');
  if(!runSel){ _batchModelsLoading=false; return; }
  runSel.disabled=true;
  try{
    const data=await getJSON('/api/ckpts');
    const runs=data.runs||[], current=data.current||{};
    _batchSetOptions(runSel, runs, '没有可用模型');
    runSel.value=runs.includes(current.run) ? current.run : (runs[0]||'');
    await loadBatchSteps(runSel.value, current.run===runSel.value ? current.step : '');
    _batchModelsLoaded=true;
  }catch(e){
    _batchSetOptions(runSel, [], '读取模型列表失败');
    state.batch={...state.batch, error:'读取批量模型失败：'+e.message};
  }finally{
    _batchModelsLoading=false;
    runSel.disabled=!!state.batch.running;
    renderBatch();
  }
}
let _batchDirPath='', _batchDirParent='';
function _setBatchInputPath(path){
  const input=$('#batchInput'), output=$('#batchOutput');
  if(!input) return;
  const oldDefault=_batchDefaultOutput(input.value);
  input.value=path||'';
  if(output && (!output.value || output.value===oldDefault)) output.value=_batchDefaultOutput(input.value);
}
async function batchDirGo(path){
  const picker=$('#batchDirPicker'), list=$('#batchDirList'), pathEl=$('#batchDirPath'), up=$('#batchDirUp');
  if(!picker || !list || !pathEl || !up) return;
  list.textContent='读取目录…';
  try{
    const data=await getJSON('/api/dirbrowse?path='+encodeURIComponent(path||''));
    _batchDirPath=data.path||path||'/'; _batchDirParent=data.parent||_batchDirPath;
    pathEl.textContent=_batchDirPath; pathEl.title=_batchDirPath;
    up.disabled=!_batchDirParent || _batchDirParent===_batchDirPath;
    list.innerHTML='';
    for(const name of (data.dirs||[])){
      const button=document.createElement('button');
      button.type='button'; button.textContent='📁 '+name; button.title=name;
      button.onclick=()=>batchDirGo((_batchDirPath==='/'?'':_batchDirPath)+'/'+name);
      list.appendChild(button);
    }
    if(!(data.dirs||[]).length){
      const empty=document.createElement('span'); empty.className='bmut'; empty.textContent='（没有子目录）'; list.appendChild(empty);
    }
    if(data.truncated){
      const note=document.createElement('span'); note.className='bmut'; note.textContent='仅显示前 2000 个子目录'; list.appendChild(note);
    }
  }catch(e){
    list.textContent='读取失败：'+e.message;
  }
}
function wireBatch(){
  const btn=$('#batchBtn'), useCurrent=$('#batchUseCurrent'), browseInput=$('#batchBrowseInput');
  if(btn) btn.onclick = ()=>{
    showBatch(true);
    const input = $('#batchInput'), output = $('#batchOutput');
    if(input && !input.value) input.value = inCur || '';
    if(output && !output.value) output.value = _batchDefaultOutput(input && input.value);
    renderBatch();
  };
  if(useCurrent) useCurrent.onclick = ()=>{
    _setBatchInputPath(inCur||'');
    const picker=$('#batchDirPicker'); if(picker) picker.hidden=true;
  };
  if(browseInput) browseInput.onclick=()=>{
    const picker=$('#batchDirPicker'), input=$('#batchInput');
    if(!picker) return;
    picker.hidden=false;
    batchDirGo((input&&input.value.trim())||inCur||'/');
  };
  const dirUp=$('#batchDirUp'), dirChoose=$('#batchDirChoose'), dirClose=$('#batchDirClose');
  if(dirUp) dirUp.onclick=()=>batchDirGo(_batchDirParent);
  if(dirChoose) dirChoose.onclick=()=>{
    _setBatchInputPath(_batchDirPath);
    const picker=$('#batchDirPicker'); if(picker) picker.hidden=true;
  };
  if(dirClose) dirClose.onclick=()=>{ const picker=$('#batchDirPicker'); if(picker) picker.hidden=true; };
  const run=$('#batchRun'), cancel=$('#batchCancel'), close=$('#batchClose');
  const modelRun=$('#batchModelRun');
  if(modelRun) modelRun.onchange=()=>loadBatchSteps(modelRun.value);
  if(run) run.onclick = startBatch;
  if(cancel) cancel.onclick = cancelBatch;
  if(close) close.onclick = ()=> showBatch(false);
  renderBatch();
}
function showBatch(on){
  if(on) state.hidden.delete('batch'); else state.hidden.add('batch');
  buildPanels(true);
  if(on){
          loadBatchModels();
          const s=document.querySelector('#blocks section[data-pid="batch"]');
          if(s) s.scrollIntoView({behavior:'smooth', block:'start'}); }
}
async function startBatch(){
  if(state.batch.running) return;
  const dirPicker=$('#batchDirPicker'); if(dirPicker) dirPicker.hidden=true;
  const input=($('#batchInput').value||'').trim(), output=($('#batchOutput').value||'').trim();
  const name=($('#batchName').value||'').trim();
  const checkpointRun=($('#batchModelRun').value||'').trim();
  const checkpointStep=($('#batchModelStep').value||'').trim();
  if(!input || !output || !name){
    state.batch={...state.batch, phase:'未启动', error:'请输入输入目录、输出根目录和文件名模板'};
    renderBatch(); return;
  }
  if(!checkpointRun || !checkpointStep){
    state.batch={...state.batch, phase:'未启动', error:'请选择批量模型 Run 和 Checkpoint Step'};
    renderBatch(); return;
  }
  const body={input_dir:input, output_dir:output, name_template:name,
    checkpoint_run:checkpointRun, checkpoint_step:checkpointStep,
    mode:$('#batchMode').value, cam_mode:$('#batchCam').value, hand_mode:$('#batchHand').value,
    pred_betas:$('#batchBetas').value, pred_fov:$('#batchFov').value,
    overwrite:$('#batchOverwrite').checked};
  console.log('[btn] 开始批量推理 model='+checkpointRun+'/'+checkpointStep+' input='+input+' output='+output+' name='+name);
  state.batch={...state.batch, running:true, phase:'启动中…', input_root:input, output_root:output,
    name_template:name, checkpoint_path:checkpointRun+'/'+checkpointStep,
    total:0, done:0, succeeded:0, failed:0, skipped:0, current:null,
    log:[], error:null, manifest:null, cancelled:false, progress:0};
  renderBatch();
  try{
    const r=await fetch(U('/api/batch/start'), {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)});
    if(!r.ok){ const j=await r.json().catch(()=>null);
      state.batch={...state.batch, running:false, phase:'未启动', error:(j&&j.error)||'启动失败'};
      renderBatch(); return; }
    const started=await r.json();
    const modelChanged=!!started.checkpoint && started.checkpoint!==CKPT_TAG;
    state.batch={...state.batch, checkpoint:started.checkpoint||'',
      checkpoint_path:started.checkpoint_path||state.batch.checkpoint_path,
      devices:started.devices||[]};
    CKPT_TAG=started.checkpoint||CKPT_TAG;
    if(modelChanged){
      state.gt=state.pred=state.metrics=state.nums=null; state.metricsError=null;
      state.loaded=false; setRawOnly(false); state.comparisonSnapshot=null; state.inferenceDirty=true;
      buildPanels(false);
    }
    state.modelReady=!!started.model_ready;
    state.modelLoading=!state.modelReady;
    updateLoadBtn();
    if(ckptBrowser && started.checkpoint_path){
      ckptBrowser.go(started.checkpoint_path).then(()=>ckptBrowser.mark(started.checkpoint_path));
    }
    if(!state.modelReady) pollModelReady();
  }catch(e){ state.batch={...state.batch, running:false, phase:'未启动', error:e.message}; renderBatch(); return; }
  startBatchPoll();
}
async function cancelBatch(){
  console.log('[btn] 取消批量推理');
  try{ await fetch(U('/api/batch/cancel'), {method:'POST'}); }catch(e){}
}
let _batchPolling=false;
async function startBatchPoll(){
  if(_batchPolling) return; _batchPolling=true;
  while(_batchPolling){
    try{
      const s=await getJSON('/api/batch/status');
      state.batch={...state.batch, ...s};
      renderBatch();
      if((s.running || s.manifest || s.error) && state.hidden.has('batch')) showBatch(true);
      if(!s.running){ _batchPolling=false; break; }
    }catch(e){}
    await new Promise(r=>setTimeout(r,700));
  }
}
function renderBatch(){
  const b=state.batch||{}, running=!!b.running;
  const btn=$('#batchBtn'), run=$('#batchRun'), cancel=$('#batchCancel'), phase=$('#batchPhase');
  const prog=$('#batchProg'), bar=$('#batchBar'), stats=$('#batchStats');
  const current=$('#batchCurrent'), result=$('#batchResult'), log=$('#batchLog'), devices=$('#batchDevices');
  const modelRun=$('#batchModelRun'), modelStep=$('#batchModelStep');
  if(btn) btn.classList.toggle('on', running);
  if(run){ run.disabled=running || !modelRun || !modelRun.value || !modelStep || !modelStep.value;
    run.style.display=running?'none':'';
    run.title='启动后自动切换并加载所选 Run / Step，再扫描目录推理'; }
  if(cancel) cancel.style.display=running?'':'none';
  document.querySelectorAll('#batchWrap .batch-form input, #batchWrap .batch-form select, #batchUseCurrent, #batchBrowseInput')
    .forEach(el=>{ el.disabled=running; });
  const dirPicker=$('#batchDirPicker'); if(dirPicker && running) dirPicker.hidden=true;
  const activeDevices=(b.devices&&b.devices.length)?b.devices:state.modelDevices;
  if(devices) devices.textContent=activeDevices&&activeDevices.length
    ? `设备 ${activeDevices.length} 张 · ${activeDevices.join(', ')} · 1 路推理 + ${b.workers||2} 路渲染`
    : '启动后自动加载所选模型';
  if(phase){
    phase.className='step '+(b.error&&!running?'err':running?'busy':b.manifest&&!b.cancelled?'ok':b.cancelled?'warn':'');
    phase.textContent=b.error&&!running ? '✗ '+(b.phase||'失败')+'：'+b.error
      : (b.phase || '待运行');
  }
  const total=+b.total||0, done=+b.done||0;
  if(prog){
    if(running && total>0){ prog.style.display='inline-block'; prog.classList.remove('indet'); if(bar) bar.style.width=Math.round(done/total*100)+'%'; }
    else if(running){ prog.style.display='inline-block'; prog.classList.add('indet'); }
    else if(b.manifest){ prog.style.display='inline-block'; prog.classList.remove('indet'); if(bar) bar.style.width=(b.cancelled&&total?Math.round(done/total*100):100)+'%'; }
    else prog.style.display='none';
  }
  if(stats) stats.textContent=total
    ? `${done}/${total} · 成功 ${b.succeeded||0} · 失败 ${b.failed||0} · 跳过 ${b.skipped||0}` : '';
  if(current){
    const active=Array.isArray(b.active)?b.active:[];
    if(active.length){
      current.innerHTML=active.map(item=>{
        const ft=+item.total||0, fd=+item.done||0;
        const fp=ft?` ${fd}/${ft} (${Math.round(fd/ft*100)}%)`:'';
        return `<div><b>${item.index}/${total||'?'}</b> <span class="bmut">${_esc(item.stage_label||item.stage||'处理中')}${fp}</span> <code>${_esc(item.path||'')}</code></div>`;
      }).join('');
    }else if(b.current){
      const ft=+b.file_total||0, fd=+b.file_done||0, fp=ft?` · 当前渲染 ${fd}/${ft} 帧 (${Math.round(fd/ft*100)}%)`:'';
      current.innerHTML=`<b>${b.current_index||done+1}/${total||'?'}</b> <code>${_esc(b.current)}</code>${fp}`;
    }else current.textContent='';
  }
  if(result){
    const model=b.checkpoint_path||b.checkpoint||'';
    result.innerHTML=b.manifest
      ? `<span>模型 <code>${_esc(model)}</code></span><span>输出根目录 <code>${_esc(b.output_root||'')}</code></span><span>任务清单 <code>${_esc(b.manifest)}</code></span>`
      : '<span class="bmut">所选模型会在启动后自动加载；输出保持输入目录的相对层级，每个视频生成同名 .mp4 与 .npz。</span>';
  }
  if(log) log.textContent=(b.log||[]).join('\n');
}

// ── Benchmark：顶部按钮展开面板并加载「功能×数据集」三态选择网格；勾选后 [运行测评] 才开跑。结果落 #benchWrap ──
function wireBench(){
  const btn = $('#benchBtn');
  if(btn) btn.onclick = ()=>{ showBench(true); loadBenchModels(); loadBenchCaps(); loadBenchSizes(); loadBenchGpus(); loadBenchAliyunDefaults(); };   // 顶部按钮：展开面板 + 拉模型/能力/规模/执行资源（不自动开跑）
  const run = $('#bfRun'), cancel = $('#bfCancel'), close = $('#bfClose');
  if(run) run.onclick = ()=> startBench();
  if(cancel) cancel.onclick = ()=> cancelBench();
  if(close) close.onclick = ()=> showBench(false);               // ✕ 仅收起，后台评测继续
  const modelRun=$('#bfModelRun');
  const modelAdd=$('#bfModelAdd'), historyAdd=$('#bfHistoryAdd');
  const historyTier=$('#bfHistoryTier');
  if(modelRun) modelRun.onchange=()=>loadBenchModelSteps(modelRun.value);
  if(modelAdd) modelAdd.onclick=()=>addBenchModel();
  if(historyAdd) historyAdd.onclick=()=>addBenchHistory();
  if(historyTier) historyTier.onchange=()=>{
    state.bench.historyTier=historyTier.value;
    renderBenchHistoryOptions();
    renderBenchModels();
  };
  const suiteSwitch=$('#bfSuiteSwitch');
  if(suiteSwitch) suiteSwitch.querySelectorAll('[data-bench-suite]').forEach(button=>{
    button.onclick=()=>setBenchSuite(button.dataset.benchSuite);
  });
  document.querySelectorAll('[data-bench-backend]').forEach(button=>{
    button.onclick=()=>setBenchBackend(button.dataset.benchBackend);
  });
  for(const id of ['#bfAliNodes','#bfAliGpus']){
    const input=$(id); if(input) input.oninput=()=>{ renderBenchExecutor(); updateBenchCount(); };
  }
  for(const id of ['#bfSeqStart','#bfSeqEnd','#bfFrames']){
    const input=$(id); if(input) input.oninput=()=>updateBenchCount();
  }
  document.querySelectorAll('[data-bench-sample]').forEach(button=>{
    button.onclick=()=>setBenchSamplePreset(button.dataset.benchSample);
  });
  const autoUkf=$('#bfAutoUkf');
  if(autoUkf) autoUkf.onchange=()=>{
    state.bench.autoUkf=autoUkf.checked;
    updateBenchCount();
  };
  wireBenchTableExport();
  renderBench();
}
// benchmark 面板已纳入栏目系统:显隐=切 state.hidden 并重建(与调节栏 👁 同一套);on 时滚动到位。
function showBench(on){
  if(on) state.hidden.delete('bench'); else state.hidden.add('bench');
  buildPanels(true);
  if(on){ const s = document.querySelector('#blocks section[data-pid="bench"]');
          if(s) s.scrollIntoView({behavior:'smooth', block:'start'}); }
}

function _benchSetOptions(select, values, placeholder){
  if(!select) return;
  select.innerHTML=values.length
    ? values.map(value=>`<option value="${_esc(value)}">${_esc(value)}</option>`).join('')
    : `<option value="">${_esc(placeholder)}</option>`;
}

const BENCH_HISTORY_TIER_LABELS={minimum:'最小',quarter:'25%',half:'50%',full:'100%',custom:'自定义'};
function benchHistoryTierLabel(tier){
  const parts=String(tier||'custom').split('+');
  return parts.map(part=>BENCH_HISTORY_TIER_LABELS[part]||part).join(' + ');
}
function benchHistoryTime(value,compact=false){
  const date=new Date(value||'');
  if(!Number.isFinite(date.getTime())) return value||'时间未知';
  return date.toLocaleString('zh-CN',compact
    ? {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}
    : {year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
}
function benchHistoryOptionLabel(record){
  const variant=record.variant==='ukf'?'UKF':'原始';
  const reuse=record.cache_hit?' · 复用缓存':'';
  const selection=record.selection||{};
  const heads=Array.isArray(selection.heads)?selection.heads.join('+'):String(selection.heads||'');
  const datasets=Array.isArray(selection.datasets)?selection.datasets.join('+'):String(selection.datasets||'');
  const scope=[heads,datasets].filter(Boolean).join(' / ');
  return `${record.run} / ${record.step} · ${variant}${reuse} · ${benchHistoryTime(record.recorded_at)} · ${benchHistoryTierLabel(record.sampling_tier)}${scope?' · '+scope:''} · ${record.protocol_revision||'legacy'}`;
}
function benchHistoryOptionKey(record){
  return `${encodeURIComponent(record.run||'')}|${encodeURIComponent(record.record_id||'')}`;
}
function _benchSetHistoryOptions(select,records,placeholder,preferred=''){
  if(!select) return;
  select.innerHTML=records.length
    ? records.map(record=>`<option value="${_esc(benchHistoryOptionKey(record))}">${_esc(benchHistoryOptionLabel(record))}</option>`).join('')
    : `<option value="">${_esc(placeholder)}</option>`;
  if(records.some(record=>benchHistoryOptionKey(record)===preferred)) select.value=preferred;
}

function benchFilteredHistoryOptions(){
  const tier=state.bench.historyTier||'half';
  if(tier==='all') return state.bench.historyOptions||[];
  return (state.bench.historyOptions||[]).filter(record=>
    String(record.sampling_tier||'custom').split('+').includes(tier));
}
function renderBenchHistoryOptions(preferred=''){
  const select=$('#bfModelHistory'), tierSelect=$('#bfHistoryTier');
  if(!select) return;
  if(tierSelect) tierSelect.value=state.bench.historyTier||'half';
  if(state.bench.historyLoading){
    _benchSetHistoryOptions(select,[],'自动扫描默认 logs 目录中…');
    return;
  }
  if(state.bench.historyError){
    _benchSetHistoryOptions(select,[],'历史记录读取失败');
    select.title=state.bench.historyError;
    return;
  }
  const records=benchFilteredHistoryOptions(), tier=state.bench.historyTier||'half';
  const label=tier==='all'?'全部份量':benchHistoryTierLabel(tier);
  select.title=`${label} · 自动扫描 ${state.bench.historyRoot||'output/model_train'}/**/logs 下的 Benchmark 记录`;
  _benchSetHistoryOptions(select,records,
    (state.bench.historyOptions||[]).length?`没有 ${label} 的历史测评`:'默认 logs 目录暂无历史测评',preferred);
}

let _benchHistoryRequest=0;
async function loadBenchHistory(preferred=''){
  const select=$('#bfModelHistory'); if(!select) return;
  const requestId=++_benchHistoryRequest;
  state.bench.historyLoading=true;
  state.bench.historyError=null;
  renderBenchHistoryOptions();
  renderBenchModels();
  try{
    const data=await getJSON('/api/benchmark/history');
    if(requestId!==_benchHistoryRequest) return;
    state.bench.historyOptions=data.records||[];
    state.bench.historyRoot=data.root||'output/model_train';
  }catch(e){
    if(requestId!==_benchHistoryRequest) return;
    state.bench.historyOptions=[];
    state.bench.historyRoot='';
    state.bench.historyError=e.message||String(e);
  }finally{
    if(requestId===_benchHistoryRequest){
      state.bench.historyLoading=false;
      renderBenchHistoryOptions(preferred);
      renderBenchModels();
    }
  }
}

async function loadBenchModelSteps(run, preferredStep=''){
  const select=$('#bfModelStep'); if(!select) return;
  if(!run){ _benchSetOptions(select, [], '请先选择 Run'); renderBenchModels(); return; }
  _benchSetOptions(select, [], '加载中…');
  try{
    const data=await getJSON('/api/ckpts/'+_enc(run)), steps=data.steps||[];
    _benchSetOptions(select, steps, '未找到 checkpoint');
    if(steps.length) select.value=steps.includes(preferredStep)?preferredStep:steps[steps.length-1];
  }catch(e){ _benchSetOptions(select, [], '读取失败'); }
  renderBenchModels();
}

let _benchModelsLoading=false;
async function loadBenchModels(){
  if(_benchModelsLoading) return;
  if(state.bench.modelOptionsLoaded){
    await loadBenchHistory(($('#bfModelHistory')&&$('#bfModelHistory').value)||'');
    renderBenchModels(); return;
  }
  _benchModelsLoading=true;
  const runSelect=$('#bfModelRun');
  try{
    const data=await getJSON('/api/ckpts'), runs=data.runs||[], current=data.current||{};
    state.bench.modelRuns=runs; state.bench.modelOptionsLoaded=true;
    _benchSetOptions(runSelect, runs, '未找到训练 Run');
    const preferredRun=(state.bench.selectedModels[0]||{}).run||current.run||runs[0]||'';
    if(runSelect && preferredRun) runSelect.value=preferredRun;
    if(!state.bench.selectedModels.length && current.path){
      const fallback=String(current.path).replace(/\/+$/,'').split('/').filter(Boolean);
      const step=current.step||fallback.pop()||'checkpoint';
      const run=current.run||fallback.pop()||'path';
      state.bench.selectedModels=[{
        run, step, path:current.run&&current.step?null:current.path,
        label:current.run&&current.step?`${current.run} / ${current.step}`:current.path,
      }];
    }
    await loadBenchModelSteps(preferredRun, (state.bench.selectedModels[0]||{}).step||current.step||'');
  }catch(e){
    state.bench.modelOptionsLoaded=false;
    _benchSetOptions(runSelect, [], '模型列表读取失败');
  }finally{
    await loadBenchHistory(($('#bfModelHistory')&&$('#bfModelHistory').value)||'');
    _benchModelsLoading=false; renderBenchModels(); updateBenchCount();
  }
}

function addBenchModel(){
  if(state.bench.running) return;
  const run=($('#bfModelRun')&&$('#bfModelRun').value)||'';
  const step=($('#bfModelStep')&&$('#bfModelStep').value)||'';
  if(!run || !step) return;
  if(!state.bench.selectedModels.some(model=>model.run===run&&model.step===step)){
    state.bench.selectedModels.push({run, step, label:`${run} / ${step}`});
  }
  renderBenchModels(); updateBenchCount();
}

async function addBenchHistory(){
  if(state.bench.historyLoading) return;
  const selected=($('#bfModelHistory')&&$('#bfModelHistory').value)||'';
  const record=(state.bench.historyOptions||[]).find(item=>benchHistoryOptionKey(item)===selected);
  if(!record) return;
  const run=record.run, recordId=record.record_id;
  if((state.bench.historyResults||[]).some(model=>model.run===run&&model.historical_record_id===recordId)) return;
  state.bench.historyLoading=true;
  renderBenchModels();
  try{
    const query=new URLSearchParams({run,record_id:recordId});
    const data=await getJSON('/api/benchmark/history/result?'+query.toString());
    const model=data.model||{};
    if(!model.report) throw new Error('历史记录缺少可比较结果');
    const variant=model.variant==='ukf'?' · UKF':'';
    const time=benchHistoryTime(model.history_recorded_at,true);
    const tier=benchHistoryTierLabel(model.history_sampling_tier);
    model.label=`${model.run} / ${model.step}${variant} · ${time} · ${tier}`;
    state.bench.historyResults=[...(state.bench.historyResults||[]),model];
    state.bench.error=null;
  }catch(e){
    state.bench.error='读取历史测评失败：'+e.message;
  }finally{
    state.bench.historyLoading=false;
    renderBench(); updateBenchCount();
  }
}

function removeBenchModel(index){
  if(state.bench.running) return;
  state.bench.selectedModels.splice(index, 1);
  renderBenchModels(); updateBenchCount();
}

function removeBenchHistory(index){
  state.bench.historyResults.splice(index,1);
  renderBench();
}

function renderBenchModels(){
  const list=$('#bfModelList'), run=$('#bfModelRun'), step=$('#bfModelStep');
  const history=$('#bfModelHistory'), historyTier=$('#bfHistoryTier');
  const add=$('#bfModelAdd'), historyAdd=$('#bfHistoryAdd');
  const running=!!state.bench.running;
  if(run) run.disabled=running;
  if(step) step.disabled=running;
  // Historical reports are read-only comparison columns and do not alter the active run.
  if(historyTier) historyTier.disabled=state.bench.historyLoading;
  if(history) history.disabled=state.bench.historyLoading;
  if(add) add.disabled=running||!run||!run.value||!step||!step.value;
  if(historyAdd) historyAdd.disabled=state.bench.historyLoading||!history||!history.value;
  if(!list) return;
  const results=new Map((state.bench.modelResults||[])
    .filter(model=>model.variant!=='ukf')
    .map(model=>[`${model.run}\n${model.step}`,model]));
  const histories=state.bench.historyResults||[];
  if(!state.bench.selectedModels.length&&!histories.length){
    list.innerHTML='<span class="bmut">可把 checkpoint 加入待测队列，也可直接加载某次历史记录横向对比。</span>';
    return;
  }
  const statusText={pending:'等待',running:'评测中',completed:'完成',failed:'失败',cancelled:'已取消'};
  const pending=state.bench.selectedModels.map((model,index)=>{
    const result=results.get(`${model.run}\n${model.step}`)||{};
    const status=result.status||'pending';
    return `<span class="bf-model ${_esc(status)}"><span class="bf-model-order">${index+1}</span>`
      + `<code title="${_esc(model.label)}">${_esc(model.run)} / ${_esc(model.step)}</code>`
      + `<span class="bf-model-status">待测 · ${result.cache_hit?'复用缓存':(statusText[status]||status)}${result.benchmark_log?' · 已写时间戳记录':''}</span>`
      + `<button type="button" data-index="${index}" title="移除此模型"${running?' disabled':''}>×</button></span>`;
  }).join('');
  const loaded=histories.map((model,index)=>{
    const variant=model.variant==='ukf'?'UKF · ':'';
    const reuse=model.cache_hit?'复用缓存 · ':'';
    const tier=benchHistoryTierLabel(model.history_sampling_tier);
    const time=benchHistoryTime(model.history_recorded_at,true);
    const title=`${model.label||''}\n协议 ${model.history_protocol_revision||'legacy'}\n${benchSelectionSummary(model.history_selection||{})}`;
    return `<span class="bf-model historical"><span class="bf-model-order">H${index+1}</span>`
      + `<code title="${_esc(title)}">${_esc(model.run)} / ${_esc(model.step)}</code>`
      + `<span class="bf-model-status">历史 · ${_esc(variant+reuse+tier)} · ${_esc(time)}</span>`
      + `<button type="button" data-history-index="${index}" title="移除此历史记录">×</button></span>`;
  }).join('');
  list.innerHTML=pending+loaded;
  list.querySelectorAll('button[data-index]').forEach(button=>{
    button.onclick=()=>removeBenchModel(+button.dataset.index);
  });
  list.querySelectorAll('button[data-history-index]').forEach(button=>{
    button.onclick=()=>removeBenchHistory(+button.dataset.historyIndex);
  });
}

// —— 能力网格：拉 /capabilities，渲染「功能(head)×数据集」三态勾选（可跑/待实现·待模型/不匹配）——
async function loadBenchCaps(){
  try{ state.bench.caps = await getJSON('/api/benchmark/capabilities'); }
  catch(e){ state.bench.caps = null; }
  renderBenchGrid();
}
// 每数据集规模（序列条数/总帧数）：后端不加载模型、后台算一次并缓存；仍在算(computing)则稍后再拉。
async function loadBenchSizes(){
  try{
    const s = await getJSON('/api/benchmark/sizes');
    if(s.sizes){ state.bench.sizes = s.sizes; renderBenchGrid(); }
    if(s.computing) setTimeout(loadBenchSizes, 1500);   // 后台还在枚举，稍后再拉一次
  }catch(e){ /* 规模统计失败不影响网格其余功能 */ }
}
// 可用显卡:拉 /gpus,渲染多选 checkbox(默认只勾第一张=单卡;多卡由用户加勾并行分片)。
async function loadBenchGpus(){
  try{ state.bench.gpus = await getJSON('/api/benchmark/gpus'); }
  catch(e){ state.bench.gpus = null; }
  renderBenchGpus();
}

const BENCH_ALIYUN_FIELDS = {
  region:'#bfAliRegion', workspace_id:'#bfAliWorkspace', resource_id:'#bfAliResource',
  image:'#bfAliImage', cpfs_uri:'#bfAliCpfs', repo_dir:'#bfAliRepo',
  nnodes:'#bfAliNodes', gpus_per_node:'#bfAliGpus', worker_cpu:'#bfAliCpu',
  worker_memory:'#bfAliMemory', worker_shared_memory:'#bfAliSharedMemory',
  conda_env:'#bfAliConda', job_name:'#bfAliJobName'
};
async function loadBenchAliyunDefaults(){
  if(state.bench.aliyunDefaultsLoaded) return;
  try{
    const data=await getJSON('/api/benchmark/aliyun/defaults');
    state.bench.aliyun={...(data.aliyun||{})};
    state.bench.aliyunDefaultsLoaded=true;
    for(const [key,selector] of Object.entries(BENCH_ALIYUN_FIELDS)){
      const input=$(selector); if(input && state.bench.aliyun[key]!=null) input.value=state.bench.aliyun[key];
    }
  }catch(e){
    state.bench.error='读取 Aliyun 默认配置失败：'+e.message;
  }
  renderBenchExecutor();
}
function benchAliyunConfig(){
  const config={};
  for(const [key,selector] of Object.entries(BENCH_ALIYUN_FIELDS)){
    const input=$(selector), raw=input?input.value:'';
    config[key]=['nnodes','gpus_per_node','worker_cpu'].includes(key)?Math.floor(+raw):raw.trim();
  }
  return config;
}
function setBenchBackend(backend){
  if(state.bench.running || !['local','aliyun'].includes(backend)) return;
  state.bench.backend=backend;
  if(backend==='aliyun') loadBenchAliyunDefaults();
  renderBenchExecutor(); updateBenchCount();
}
function renderBenchExecutor(){
  const backend=state.bench.backend||'local', running=!!state.bench.running;
  document.querySelectorAll('[data-bench-backend]').forEach(button=>{
    const active=button.dataset.benchBackend===backend;
    button.classList.toggle('on',active); button.disabled=running;
    button.setAttribute('aria-selected',active?'true':'false');
  });
  const local=$('#bfLocalExecutor'), aliyun=$('#bfAliyunExecutor');
  if(local) local.hidden=backend!=='local';
  if(aliyun) aliyun.hidden=backend!=='aliyun';
  document.querySelectorAll('#bfLocalExecutor input,#bfAliyunExecutor input').forEach(input=>input.disabled=running);
  const nodes=Math.max(0,Math.floor(+($('#bfAliNodes')&&$('#bfAliNodes').value)||0));
  const gpus=Math.max(0,Math.floor(+($('#bfAliGpus')&&$('#bfAliGpus').value)||0));
  const world=$('#bfAliWorld'); if(world) world.innerHTML=`<b>${nodes*gpus}</b><small>全局 shards</small>`;
  const remote=$('#bfRemoteJob');
  if(remote){
    const text=state.bench.job_id ? `${state.bench.job_id} · ${state.bench.job_status||'提交中'}` : '';
    remote.textContent=text; remote.title=text;
  }
}
function renderBenchGpus(){
  const el = $('#bfGpus'); if(!el) return;
  const g = state.bench.gpus;
  if(!g || !(g.gpus||[]).length){ el.innerHTML = '<span class="bmut">显卡: 不可用</span>'; return; }
  let h = '<span class="bflbl">显卡</span>';
  g.gpus.forEach((gpu, i)=>{
    const mem = (gpu.mem_total!=null) ? ` <span class="mut">${Math.round(gpu.mem_used/1024)}/${Math.round(gpu.mem_total/1024)}G</span>` : '';
    const util = (gpu.util!=null) ? ` ${gpu.util}%` : '';
    const tip = `${gpu.name||''}${util}`;
    h += `<label class="bfgpu-l" title="${tip}"><input type="checkbox" class="bfgpu" data-idx="${gpu.index}"${i===0?' checked':''}>#${gpu.index}${mem}</label>`;
  });
  el.innerHTML = h;
  el.querySelectorAll('input.bfgpu').forEach(input=>input.onchange=updateBenchCount);
}
function benchGpuSelection(){   // 勾选的物理 GPU index 列表
  return [...document.querySelectorAll('#bfGpus .bfgpu:checked')].map(c=>parseInt(c.dataset.idx,10));
}
const _subset = (arr, set) => arr.every(x => set.has(x));

function currentBenchSuite(){
  return BENCHMARK_SUITES[state.bench.suite] || BENCHMARK_SUITES.sota;
}
function benchDatasetMeta(name){
  const current=currentBenchSuite();
  for(const suite of [current,...Object.values(BENCHMARK_SUITES).filter(item=>item!==current)]){
    const dataset=(suite.datasets||[]).find(item=>item.name===name);
    if(dataset) return dataset;
  }
  return {name,label:name,note:'',purpose:''};
}
function benchDatasetLabel(name){
  return benchDatasetMeta(name).label||name;
}
function benchSuiteSources(suite=currentBenchSuite()){
  const ids=suite.metricSuiteIds||[];
  return ids.length?ids.map(id=>BENCHMARK_SUITES[id]).filter(Boolean):[suite];
}
function benchSuiteMetrics(suite=currentBenchSuite()){
  const seen=new Set(), metrics=[];
  for(const source of benchSuiteSources(suite)) for(const metric of source.metrics||[]){
    if(!seen.has(metric.id)){ seen.add(metric.id); metrics.push(metric); }
  }
  return metrics;
}
function benchSuiteTables(suite=currentBenchSuite()){
  const ids=suite.tableSuiteIds||[];
  return ids.length?ids.flatMap(id=>(BENCHMARK_SUITES[id]||{}).tables||[]):suite.tables||[];
}

const BENCH_TABLE_EXPORTS = Object.freeze([
  {key:'hot3d-camera', suite:'hot3d_camera', table:'icra-hot3d-camera-comparison', file:'hot3d-camera'},
  {key:'arctic-camera', suite:'arctic_camera', table:'icra-arctic-camera-comparison', file:'arctic-camera'},
  {key:'vidihand-arctic', suite:'vidihand', table:'vidihand-arctic', file:'vidihand-arctic'},
  {key:'vidihand-hot3d', suite:'vidihand', table:'vidihand-hot3d', file:'vidihand-hot3d'},
]);

function benchTableExportDefinition(spec){
  const suite=BENCHMARK_SUITES[spec.suite];
  const table=((suite&&suite.tables)||[]).find(item=>item.id===spec.table);
  return table?{...spec,suiteLabel:suite.label,table}:null;
}

function benchTableExportSelected(){
  return BENCH_TABLE_EXPORTS.filter(spec=>state.bench.tableExportSelection.has(spec.key))
    .map(benchTableExportDefinition).filter(Boolean);
}

function setBenchTableExportStatus(message,kind=''){
  const status=$('#bfTableExportStatus');
  if(!status) return;
  status.className=kind;
  status.textContent=message;
}

function updateBenchTableExportControls(){
  const selected=state.bench.tableExportSelection, exporting=Boolean(state.bench.tableExporting);
  document.querySelectorAll('[data-bench-table-export]').forEach(box=>{
    box.checked=selected.has(box.dataset.benchTableExport);
    box.disabled=exporting;
  });
  const count=$('#bfTableExportCount');
  if(count) count.textContent=`已选择 ${selected.size} / ${BENCH_TABLE_EXPORTS.length}`;
  for(const id of ['#bfTableExportSeparate','#bfTableExportCombined']){
    const button=$(id); if(button) button.disabled=exporting||!selected.size;
  }
  document.querySelectorAll('[data-bench-table-select]').forEach(button=>{ button.disabled=exporting; });
}

function wireBenchTableExport(){
  document.querySelectorAll('[data-bench-table-export]').forEach(box=>box.onchange=()=>{
    const key=box.dataset.benchTableExport;
    if(box.checked) state.bench.tableExportSelection.add(key);
    else state.bench.tableExportSelection.delete(key);
    setBenchTableExportStatus(state.bench.tableExportSelection.size?'就绪 · 图片按完整指标列导出':'请至少选择一张表格');
    updateBenchTableExportControls();
  });
  document.querySelectorAll('[data-bench-table-select]').forEach(button=>button.onclick=()=>{
    state.bench.tableExportSelection=new Set(button.dataset.benchTableSelect==='all'
      ?BENCH_TABLE_EXPORTS.map(spec=>spec.key):[]);
    setBenchTableExportStatus(state.bench.tableExportSelection.size?'就绪 · 图片按完整指标列导出':'请至少选择一张表格');
    updateBenchTableExportControls();
  });
  const separate=$('#bfTableExportSeparate'), combined=$('#bfTableExportCombined');
  if(separate) separate.onclick=()=>exportBenchTablesAsJpg(false);
  if(combined) combined.onclick=()=>exportBenchTablesAsJpg(true);
  updateBenchTableExportControls();
}

function benchTableExportRows(table){
  const rows=(table.rows||[]).map(row=>({...row,values:[...(row.values||[])]}));
  const entries=benchResultEntries(benchDisplayResults(state.bench),state.bench.live_report,state.bench.activeModel);
  const live=benchSotaCurrentRows(entries,table).filter(row=>row.hasValues);
  live.forEach((row,index)=>rows.push({
    section:index===0?'本次模型 · 当前 Viewer 结果':null,
    method:`M${row.modelIndex+1} · ${row.model.variant==='ukf'?'UKF · ':''}${row.model.label||row.model.step||'checkpoint'}`,
    values:row.values,
    comparable:row.comparable,
    live:true,
  }));
  return rows;
}

function benchTableExportTimestamp(){
  const date=new Date(), pad=value=>String(value).padStart(2,'0');
  return `${date.getFullYear()}${pad(date.getMonth()+1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

function benchTableExportWrap(ctx,text,maxWidth){
  const source=String(text||'').trim();
  if(!source) return [];
  const lines=[];
  for(const paragraph of source.split(/\n/)){
    let line='';
    for(const char of paragraph){
      const candidate=line+char;
      if(line&&ctx.measureText(candidate).width>maxWidth){
        lines.push(line.trimEnd()); line=char.trimStart();
      }else line=candidate;
    }
    if(line) lines.push(line.trimEnd());
  }
  return lines;
}

function benchTableExportMetrics(table,rows,ctx){
  ctx.font='700 14px "Noto Sans SC", "Microsoft YaHei", sans-serif';
  const methodWidth=Math.max(360,Math.min(760,Math.ceil(Math.max(
    ctx.measureText('Method').width,...rows.map(row=>ctx.measureText(row.method||'').width))+34)));
  const columns=(table.columns||[]).map((column,index)=>{
    const unit=column.unit||((benchGlobalMetricSpec(column.metric)||{}).unit)||'';
    const second=`${benchMetricDirectionMark(column.direction)}${unit?' · '+unit:''}`;
    const values=rows.map(row=>benchTableValue(table,column,(row.values||[])[index]));
    const content=Math.max(ctx.measureText(column.label).width,ctx.measureText(second).width,
      ...values.map(value=>ctx.measureText(value).width));
    return {...column,index,unit,width:Math.max(116,Math.min(190,Math.ceil(content+30)))};
  });
  return {columns,methodWidth,tableWidth:methodWidth+columns.reduce((sum,column)=>sum+column.width,0)};
}

function benchRenderTableExport(definition){
  const table=definition.table, rows=benchTableExportRows(table), scale=2;
  const measure=document.createElement('canvas').getContext('2d');
  const metrics=benchTableExportMetrics(table,rows,measure), outer=34, inner=26;
  const width=metrics.tableWidth+2*(outer+inner);
  measure.font='600 13px "Noto Sans SC", "Microsoft YaHei", sans-serif';
  const textWidth=metrics.tableWidth;
  const sourceLines=benchTableExportWrap(measure,table.source||'',textWidth);
  measure.font='400 13px "Noto Sans SC", "Microsoft YaHei", sans-serif';
  const noteLines=benchTableExportWrap(measure,table.note||'',textWidth);
  const titleHeight=42+sourceLines.length*20+noteLines.length*21+20;
  const headerHeight=68, sectionHeight=36, rowHeight=48;
  const bodyHeight=rows.reduce((sum,row)=>sum+(row.section?sectionHeight:0)+rowHeight,0);
  const footerHeight=42;
  const height=outer*2+inner*2+titleHeight+headerHeight+bodyHeight+footerHeight;
  const canvas=document.createElement('canvas');
  canvas.width=Math.ceil(width*scale); canvas.height=Math.ceil(height*scale);
  canvas.dataset.logicalWidth=width; canvas.dataset.logicalHeight=height;
  const ctx=canvas.getContext('2d'); ctx.scale(scale,scale);
  ctx.fillStyle='#07100f'; ctx.fillRect(0,0,width,height);
  ctx.strokeStyle='rgba(88,213,210,.06)'; ctx.lineWidth=1;
  for(let x=0;x<width;x+=28){ ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,height);ctx.stroke(); }
  for(let y=0;y<height;y+=28){ ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(width,y);ctx.stroke(); }
  const cardX=outer,cardY=outer,cardW=width-outer*2,cardH=height-outer*2;
  ctx.fillStyle='#0e1917';ctx.strokeStyle='#29413a';ctx.lineWidth=1.5;
  ctx.beginPath();
  if(ctx.roundRect) ctx.roundRect(cardX,cardY,cardW,cardH,16);
  else ctx.rect(cardX,cardY,cardW,cardH);
  ctx.fill();ctx.stroke();
  const x0=cardX+inner, contentY=cardY+inner;
  ctx.fillStyle='#b9f45a';ctx.font='800 11px "JetBrains Mono", monospace';
  ctx.fillText('MINT / BENCHMARK TABLE',x0,contentY+10);
  ctx.fillStyle='#edf5ef';ctx.font='800 27px "Noto Sans SC", "Microsoft YaHei", sans-serif';
  ctx.fillText(`${definition.suiteLabel} · ${table.title}`,x0,contentY+43);
  let y=contentY+66;
  ctx.fillStyle='#58d5d2';ctx.font='600 13px "Noto Sans SC", "Microsoft YaHei", sans-serif';
  sourceLines.forEach(line=>{ctx.fillText(line,x0,y);y+=20;});
  ctx.fillStyle='#9cafaa';ctx.font='400 13px "Noto Sans SC", "Microsoft YaHei", sans-serif';
  noteLines.forEach(line=>{ctx.fillText(line,x0,y);y+=21;});
  y=contentY+titleHeight;
  const columnX=[x0,x0+metrics.methodWidth];
  metrics.columns.forEach(column=>columnX.push(columnX[columnX.length-1]+column.width));
  ctx.fillStyle='#14221f';ctx.fillRect(x0,y,metrics.tableWidth,headerHeight);
  ctx.strokeStyle='#365249';ctx.strokeRect(x0,y,metrics.tableWidth,headerHeight);
  ctx.fillStyle='#8da29a';ctx.font='700 13px "Noto Sans SC", "Microsoft YaHei", sans-serif';
  ctx.textAlign='left';ctx.textBaseline='middle';ctx.fillText('Method',x0+15,y+headerHeight/2);
  metrics.columns.forEach((column,index)=>{
    const left=columnX[index+1],center=left+column.width/2;
    ctx.strokeStyle='#29413a';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left,y+headerHeight);ctx.stroke();
    ctx.textAlign='center';ctx.fillStyle='#edf5ef';ctx.font='700 13px "Noto Sans SC", "Microsoft YaHei", sans-serif';
    ctx.fillText(column.label,center,y+25);
    ctx.fillStyle='#8da29a';ctx.font='500 11px "Noto Sans SC", "Microsoft YaHei", sans-serif';
    ctx.fillText(`${benchMetricDirectionMark(column.direction)}${column.unit?' · '+column.unit:''}`,center,y+46);
  });
  y+=headerHeight;
  const comparableBest=new Map();
  metrics.columns.forEach(column=>{
    const values=rows.filter(row=>row.comparable!==false)
      .map(row=>benchComparableValue(table,column,(row.values||[])[column.index])).filter(Number.isFinite);
    if(values.length) comparableBest.set(column.index,benchBestMetricValue(values,column.direction));
  });
  rows.forEach((row,rowIndex)=>{
    if(row.section){
      ctx.fillStyle='#102522';ctx.fillRect(x0,y,metrics.tableWidth,sectionHeight);
      ctx.strokeStyle='#29413a';ctx.strokeRect(x0,y,metrics.tableWidth,sectionHeight);
      ctx.textAlign='left';ctx.fillStyle='#58d5d2';ctx.font='700 11px "JetBrains Mono", "Noto Sans SC", monospace';
      ctx.fillText(row.section,x0+14,y+sectionHeight/2);y+=sectionHeight;
    }
    ctx.fillStyle=row.live?'#132923':(rowIndex%2?'#0c1715':'#0e1b18');ctx.fillRect(x0,y,metrics.tableWidth,rowHeight);
    ctx.strokeStyle='#203a33';ctx.strokeRect(x0,y,metrics.tableWidth,rowHeight);
    ctx.textAlign='left';ctx.fillStyle=row.live?'#b9f45a':'#d5e2dc';
    ctx.font=`${row.live?'700':'550'} 13px "Noto Sans SC", "Microsoft YaHei", sans-serif`;
    ctx.fillText(row.method||'',x0+14,y+rowHeight/2);
    metrics.columns.forEach((column,index)=>{
      const left=columnX[index+1],center=left+column.width/2,value=(row.values||[])[column.index];
      const comparable=benchComparableValue(table,column,value);
      const best=row.comparable!==false&&Number.isFinite(comparable)&&comparable===comparableBest.get(column.index);
      ctx.strokeStyle='#203a33';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left,y+rowHeight);ctx.stroke();
      if(best){ctx.fillStyle='rgba(255,111,111,.12)';ctx.fillRect(left+1,y+1,column.width-2,rowHeight-2);}
      ctx.textAlign='center';ctx.fillStyle=best?'#ff8b8b':(row.live?'#b9f45a':'#edf5ef');ctx.font=`${best||row.live?'750':'500'} 13px "JetBrains Mono", "Noto Sans SC", monospace`;
      ctx.fillText(benchTableValue(table,column,value),center,y+rowHeight/2);
    });
    y+=rowHeight;
  });
  ctx.textAlign='left';ctx.textBaseline='alphabetic';ctx.fillStyle='#789088';ctx.font='500 10px "JetBrains Mono", monospace';
  ctx.fillText(`MINT Viewer · JPG export · ${new Date().toLocaleString()}`,x0,height-outer-inner+8);
  return canvas;
}

function benchCanvasBlob(canvas){
  return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error('JPG 编码失败')),'image/jpeg',0.95));
}

function benchDownloadBlob(blob,filename){
  const url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download=filename;document.body.appendChild(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),30000);
}

async function exportBenchTablesAsJpg(combined){
  if(state.bench.tableExporting) return;
  const selected=benchTableExportSelected();
  if(!selected.length){setBenchTableExportStatus('请至少选择一张表格','error');return;}
  state.bench.tableExporting=true;updateBenchTableExportControls();
  const stamp=benchTableExportTimestamp();
  try{
    if(document.fonts&&document.fonts.ready) await document.fonts.ready;
    const rendered=[];
    for(const [index,definition] of selected.entries()){
      setBenchTableExportStatus(`正在生成 ${index+1} / ${selected.length} · ${definition.suiteLabel} · ${definition.table.title}`,'busy');
      rendered.push({definition,canvas:benchRenderTableExport(definition)});
      await new Promise(resolve=>requestAnimationFrame(resolve));
    }
    if(combined){
      const scale=2,gap=28,top=126,bottom=44,side=38;
      const widths=rendered.map(item=>+item.canvas.dataset.logicalWidth),heights=rendered.map(item=>+item.canvas.dataset.logicalHeight);
      const width=Math.max(...widths)+side*2,height=top+bottom+heights.reduce((sum,value)=>sum+value,0)+gap*(rendered.length-1);
      const canvas=document.createElement('canvas');canvas.width=Math.ceil(width*scale);canvas.height=Math.ceil(height*scale);
      const ctx=canvas.getContext('2d');ctx.scale(scale,scale);ctx.fillStyle='#040908';ctx.fillRect(0,0,width,height);
      ctx.fillStyle='#b9f45a';ctx.font='800 12px "JetBrains Mono", monospace';ctx.fillText('MINT / SELECTED BENCHMARK EXPORT',side,42);
      ctx.fillStyle='#edf5ef';ctx.font='800 30px "Noto Sans SC", "Microsoft YaHei", sans-serif';ctx.fillText('Benchmark 四表汇总',side,80);
      ctx.fillStyle='#8da29a';ctx.font='500 12px "Noto Sans SC", "Microsoft YaHei", sans-serif';ctx.fillText(`已选择 ${rendered.length} 张表 · ${new Date().toLocaleString()}`,side,105);
      let y=top;
      rendered.forEach((item,index)=>{const w=widths[index],h=heights[index];ctx.drawImage(item.canvas,(width-w)/2,y,w,h);y+=h+gap;});
      benchDownloadBlob(await benchCanvasBlob(canvas),`mint-benchmark-combined-${stamp}.jpg`);
      setBenchTableExportStatus(`已汇总导出 ${rendered.length} 张表格 · mint-benchmark-combined-${stamp}.jpg`,'ok');
    }else{
      for(const [index,item] of rendered.entries()){
        setBenchTableExportStatus(`正在保存 ${index+1} / ${rendered.length} · ${item.definition.table.title}`,'busy');
        benchDownloadBlob(await benchCanvasBlob(item.canvas),`mint-benchmark-${item.definition.file}-${stamp}.jpg`);
      }
      setBenchTableExportStatus(`已分别导出 ${rendered.length} 张 JPG；浏览器可能提示允许多个文件下载`,'ok');
    }
  }catch(error){
    console.error('[benchmark table export]',error);setBenchTableExportStatus(`导出失败：${error&&error.message||error}`,'error');
  }finally{
    state.bench.tableExporting=false;updateBenchTableExportControls();
  }
}
function benchSuiteMetricAlias(metricId,suite=currentBenchSuite()){
  for(const source of benchSuiteSources(suite)){
    const alias=(source.liveMetricAliases||{})[metricId];
    if(alias) return alias;
  }
  return null;
}
function benchSuiteMetricUnit(metricId,suite=currentBenchSuite()){
  for(const source of benchSuiteSources(suite)){
    if(Object.prototype.hasOwnProperty.call(source.liveMetricUnits||{},metricId)) return source.liveMetricUnits[metricId];
  }
  return '';
}
function benchMetricSelection(suite=currentBenchSuite()){
  const stored=state.bench.metricSelections[suite.id];
  if(Array.isArray(stored)) return new Set(stored);
  const all=benchSuiteMetrics(suite).map(metric=>metric.id);
  state.bench.metricSelections[suite.id]=all;
  return new Set(all);
}
function setBenchSuite(suiteId){
  if(state.bench.running || !BENCHMARK_SUITES[suiteId]) return;
  state.bench.suite=suiteId;
  applyBenchSuiteSelection(suiteId==='custom');
  renderBench();
}
function applyBenchSuiteSelection(selectAllCustom=false){
  const grid=$('#bfGrid'), suite=currentBenchSuite();
  if(!grid) return;
  if(suite.id==='custom'){
    if(selectAllCustom) grid.querySelectorAll('.bfck').forEach(box=>{ box.checked=true; });
    updateBenchCount();
    return;
  }
  const combos=new Set((suite.combos||[]).map(([head,dataset])=>`${head}\n${dataset}`));
  grid.querySelectorAll('.bfck').forEach(box=>{
    box.checked=combos.has(`${box.dataset.h}\n${box.dataset.d}`);
  });
  updateBenchCount();
}
function setBenchMetricSelection(metricIds){
  const suite=currentBenchSuite();
  if(suite.id==='custom') return;
  const allowed=new Set(benchSuiteMetrics(suite).map(metric=>metric.id));
  state.bench.metricSelections[suite.id]=metricIds.filter(metric=>allowed.has(metric));
  renderBench();
}
function benchMetricDetailHTML(metric){
  const detail=metric.detail||{};
  const direction=metric.direction==='max'?'越高越好':(metric.direction==='target'?'越接近 1 越好':'越低越好');
  const rows=[
    ['定义',detail.definition||`${metric.label}，${direction}。`],
    ['对齐',detail.alignment||'当前协议没有提供额外对齐说明。'],
    ['论文协议',detail.protocol||'请以对应公开表和官方评测代码为准。'],
    ['本地对应',detail.local||'本地结果只在相同实现、单位和 split 下才可直接比较。'],
  ];
  return `<article class="bf-metric-detail-card"><header><b>${_esc(metric.label)}</b>`
    +`<span>${_esc(detail.name||metric.label)} · ${direction}${metric.unit?' · '+_esc(metric.unit):''}</span></header>`
    +`<dl>${rows.map(([term,text])=>`<dt>${term}</dt><dd>${_esc(text)}</dd>`).join('')}</dl></article>`;
}
function renderBenchSuitePanel(){
  const suite=currentBenchSuite(), metrics=benchSuiteMetrics(suite), selected=benchMetricSelection(suite);
  const switcher=$('#bfSuiteSwitch'), picker=$('#bfMetricPicker');
  if(switcher) switcher.querySelectorAll('[data-bench-suite]').forEach(button=>{
    const active=button.dataset.benchSuite===suite.id;
    button.classList.toggle('on',active);
    button.setAttribute('aria-selected',active?'true':'false');
    button.disabled=state.bench.running;
  });
  if(!picker) return;
  if(suite.id==='custom'){
    picker.innerHTML='<span class="bmut">完整网格不限制结果指标；切换到相机表或 ViDiHand 可使用对应口径的逐项指标筛选。</span>';
    return;
  }
  const openKey=state.bench.metricDetailOpen;
  picker.innerHTML='<span class="bf-metric-label">展示指标</span>'
    +metrics.map(metric=>{
      const key=`${suite.id}:${metric.id}`, open=openKey===key;
      const direction=metric.direction==='max'?'越高越好':(metric.direction==='target'?'越接近 1 越好':'越低越好');
      return `<span class="bf-metric-entry"><label class="bf-metric-option" title="${direction}${metric.unit?' · '+_esc(metric.unit):''}">`
        +`<input type="checkbox" data-bench-metric="${_esc(metric.id)}"${selected.has(metric.id)?' checked':''}>`
        +`<span>${_esc(metric.label)}</span><small>${metric.direction==='max'?'↑':(metric.direction==='target'?'→ 1':'↓')}</small></label>`
        +`<button type="button" class="bf-metric-info" data-bench-metric-info="${_esc(metric.id)}" aria-expanded="${open?'true':'false'}" title="展开指标定义">${open?'▾':'▸'}</button></span>`;
    }).join('')
    +`<span class="bf-metric-tools"><button type="button" data-metric-action="all">全选指标</button>`
    +`<button type="button" data-metric-action="none">清空指标</button>`
    +`<small>${selected.size}/${metrics.length}</small></span>`
    +(metrics.map(metric=>`${suite.id}:${metric.id}`===openKey?benchMetricDetailHTML(metric):'').join(''));
  picker.querySelectorAll('[data-bench-metric]').forEach(box=>{
    box.onchange=()=>setBenchMetricSelection([...picker.querySelectorAll('[data-bench-metric]:checked')].map(item=>item.dataset.benchMetric));
  });
  picker.querySelectorAll('[data-bench-metric-info]').forEach(button=>button.onclick=()=>{
    const key=`${suite.id}:${button.dataset.benchMetricInfo}`;
    state.bench.metricDetailOpen=state.bench.metricDetailOpen===key?null:key;
    renderBenchSuitePanel();
  });
  const all=picker.querySelector('[data-metric-action="all"]'), none=picker.querySelector('[data-metric-action="none"]');
  if(all) all.onclick=()=>setBenchMetricSelection(metrics.map(metric=>metric.id));
  if(none) none.onclick=()=>setBenchMetricSelection([]);
}

function benchCellState(hd, d, modelCap){
  // 三态：'none'=能力不匹配 / 'wait'=匹配但功能|数据集未实现或模型不产 / 'run'=今天可跑
  const dcap = new Set(d.capability);
  if(!_subset(hd.required_gt, dcap)) return {kind:'none'};
  const reasons = [];
  if(!hd.implemented) reasons.push('功能未实现');
  if(!d.implemented) reasons.push('数据集未实现');
  const miss = hd.required_gt.filter(x => !modelCap.has(x));
  if(miss.length) reasons.push('模型不产 '+miss.join('/'));
  return reasons.length ? {kind:'wait', reason:reasons.join('；')} : {kind:'run'};
}
function renderBenchGrid(){
  const g = $('#bfGrid'); if(!g) return;
  const caps = state.bench.caps;
  if(!caps){ g.innerHTML = '<div class="bmut">能力清单加载失败（需先在左侧选一个 ckpt）</div>'; updateBenchCount(); return; }
  const modelCap = new Set(caps.model_capability||[]);
  const heads = caps.heads||[], dsets = caps.datasets||[];
  const sizes = state.bench.sizes || null;
  let h = '<table class="bgrid"><thead><tr><th class="cnr">功能 \\ 数据集</th>';
  for(const d of dsets){
    // 列头附序列条数：sizes 未回来→…；缺数据/计数失败→—(tooltip 说明)；有值→·N条(tooltip 补总帧数)
    const sz = sizes ? sizes[d.name] : undefined;
    let cnt, szt;
    if(!sizes){ cnt = ' <span class="dscnt mut">·…</span>'; szt = '规模统计中…'; }
    else if(sz && sz.n_seqs != null){
      cnt = ` <span class="dscnt">·${sz.n_seqs}条</span>`;
      szt = `序列 ${sz.n_seqs} 条 · 共 ${sz.n_frames} 帧${sz.source_root?' · 数据 '+sz.source_root:''}${sz.note?' · '+sz.note:''}`;
    }
    else { cnt = ' <span class="dscnt mut">·—</span>'; szt = '无法计数：' + ((sz && sz.note) || '缺数据'); }
    h += `<th title="内部名 ${_esc(d.name)} · 提供 ${d.capability.join('/')||'—'}${d.implemented?'':' · 数据集未实现'}｜${szt}">${_esc(benchDatasetLabel(d.name))}${cnt}</th>`;
  }
  h += '</tr></thead><tbody>';
  for(const hd of heads){
    h += `<tr><th class="rowh" title="需要 ${hd.required_gt.join('/')||'—'}${hd.implemented?'':' · 功能未实现'}">${hd.name}</th>`;
    for(const d of dsets){
      const st = benchCellState(hd, d, modelCap);
      if(st.kind==='run')
        h += `<td class="c-run"><input type="checkbox" class="bfck" data-h="${hd.name}" data-d="${d.name}" checked></td>`;
      else if(st.kind==='wait')
        h += `<td class="c-wait" title="${st.reason}">▨</td>`;
      else
        h += `<td class="c-none" title="能力不匹配">·</td>`;
    }
    h += '</tr>';
  }
  h += '</tbody></table>';
  h += '<div class="bgrid-tools"><button type="button" id="bfAll">全选可跑</button>'
     + '<button type="button" id="bfNone">清空</button>'
     + '<span class="bmut">绿=可跑 · ▨=待实现/待模型(悬停看原因) · ·=不匹配</span></div>';
  g.innerHTML = h;
  applyBenchSuiteSelection();
  g.querySelectorAll('.bfck').forEach(c => c.onchange = ()=>{
    if(state.bench.suite!=='custom'){
      state.bench.suite='custom';
      renderBenchSuitePanel();
    }
    updateBenchCount();
  });
  const all = $('#bfAll'), none = $('#bfNone');
  if(all) all.onclick = ()=>{ state.bench.suite='custom'; g.querySelectorAll('.bfck').forEach(c=>c.checked=true); renderBenchSuitePanel(); updateBenchCount(); };
  if(none) none.onclick = ()=>{ state.bench.suite='custom'; g.querySelectorAll('.bfck').forEach(c=>c.checked=false); renderBenchSuitePanel(); updateBenchCount(); };
  updateBenchCount();
}
function benchSelection(){
  const hs = new Set(), ds = new Set();
  const checked = document.querySelectorAll('#bfGrid .bfck:checked');
  checked.forEach(c=>{ hs.add(c.dataset.h); ds.add(c.dataset.d); });
  return {heads:[...hs], datasets:[...ds], n: checked.length};
}
const BENCH_SOTA_SAMPLE_PRESETS = Object.freeze({
  minimum:{label:'最小份量', fixed:{arctic_hand_coverage:24, hot3d_hand_coverage:32}},
  quarter:{label:'25%', ratio:.25},
  half:{label:'50%', ratio:.5},
  full:{label:'100%', ratio:1},
});
const BENCH_SOTA_FULL_COUNTS = Object.freeze({
  camera_hot3d:27, camera_arctic:34,
  arctic_hand_coverage:302, hot3d_hand_coverage:437,
});
const BENCH_SOTA_FIXED_SPLIT_VERSION = 'sota-camera-balanced-v2';
function setBenchSamplePreset(preset){
  if(state.bench.running || !BENCH_SOTA_SAMPLE_PRESETS[preset]) return;
  state.bench.samplePreset=preset;
  renderBenchSampling(); updateBenchCount();
}
function benchDatasetAvailable(dataset){
  const size=(state.bench.sizes||{})[dataset]||{};
  return Number.isFinite(+size.n_seqs) ? Math.max(0,Math.floor(+size.n_seqs))
    : (BENCH_SOTA_FULL_COUNTS[dataset]||0);
}
function benchSotaDatasetSelection(){
  if(currentBenchSuite().id!=='sota') return {};
  const preset=BENCH_SOTA_SAMPLE_PRESETS[state.bench.samplePreset]||BENCH_SOTA_SAMPLE_PRESETS.minimum;
  const selection={};
  for(const dataset of Object.keys(BENCH_SOTA_FULL_COUNTS)){
    const available=benchDatasetAvailable(dataset);
    const fixed={fixed_tier:state.bench.samplePreset,split_version:BENCH_SOTA_FIXED_SPLIT_VERSION};
    // Camera tables use the fixed complete sequence lists.
    if(dataset==='camera_hot3d'||dataset==='camera_arctic'){
      selection[dataset]={sampling:'all',...fixed}; continue;
    }
    if(preset.ratio===1){ selection[dataset]={sampling:'all',...fixed}; continue; }
    const requested=preset.fixed ? preset.fixed[dataset] : Math.ceil(available*preset.ratio);
    selection[dataset]={sampling:'diverse',sample_count:Math.max(1,Math.min(available,requested)),...fixed};
  }
  return selection;
}
function benchUsesSotaSampling(){
  return currentBenchSuite().id==='sota' && benchSelection().datasets.length>1;
}
function benchRunRange(){
  const datasets=benchSelection().datasets;
  if(benchUsesSotaSampling() || datasets.length!==1){
    return {seqStart:0,seqEnd:null,maxFrames:null,datasetSelection:benchSotaDatasetSelection()};
  }
  const start=Math.max(0,Math.floor(+($('#bfSeqStart')&&$('#bfSeqStart').value)||0));
  const rawEnd=$('#bfSeqEnd')&&$('#bfSeqEnd').value;
  const seqEnd=rawEnd===''||rawEnd==null?null:Math.floor(+rawEnd);
  const rawFrames=$('#bfFrames')&&$('#bfFrames').value;
  return {seqStart:start,seqEnd,maxFrames:rawFrames?Math.floor(+rawFrames):null,datasetSelection:{}};
}
function renderBenchSampling(){
  const root=$('#bfSampling'), sota=$('#bfSotaSampling'), single=$('#bfSingleSampling');
  if(!root || !sota || !single) return;
  const datasets=benchSelection().datasets, useSota=benchUsesSotaSampling(), useSingle=!useSota&&datasets.length===1;
  root.hidden=!useSota&&!useSingle; sota.hidden=!useSota; single.hidden=!useSingle;
  root.querySelectorAll('input,button').forEach(element=>{ element.disabled=state.bench.running; });
  document.querySelectorAll('[data-bench-sample]').forEach(button=>{
    const active=button.dataset.benchSample===state.bench.samplePreset;
    button.classList.toggle('on',active); button.setAttribute('aria-checked',active?'true':'false');
  });
  if(useSota){
    const options=benchSotaDatasetSelection(), parts=[];
    for(const dataset of ['camera_hot3d','camera_arctic','arctic_hand_coverage','hot3d_hand_coverage']){
      const meta=benchDatasetMeta(dataset), option=options[dataset]||{}, available=benchDatasetAvailable(dataset);
      const count=option.sampling==='all'?available:option.sample_count;
      parts.push(`${meta.label||dataset} ${count}/${available}${option.max_frames?`，每视频≤${option.max_frames}帧`:''}`);
    }
    const summary=$('#bfSamplingSummary');
    if(summary) summary.textContent=`固定清单 ${BENCH_SOTA_FIXED_SPLIT_VERSION} · `+parts.join(' · ');
  }
  if(useSingle){
    const label=$('#bfSingleDatasetLabel'), meta=benchDatasetMeta(datasets[0]);
    if(label) label.textContent=`${meta.label||datasets[0]} · 单数据集范围`;
  }
}
function benchSelectionSummary(selection){
  const configured=Object.entries((selection||{}).dataset_selection||{});
  if(configured.length){
    const fixed=configured.find(([_dataset,option])=>option.fixed_tier);
    const details=configured.map(([dataset,option])=>{
      const count=option.sampling==='all'?'全部':`${option.sample_count}条`;
      return `${benchDatasetLabel(dataset)} ${count}${option.max_frames?`×≤${option.max_frames}帧`:''}`;
    }).join(' · ');
    return fixed?`固定清单 ${fixed[1].split_version} · ${details}`:details;
  }
  const end=selection&&selection.seq_end==null?'全部':selection.seq_end;
  return `范围 [${(selection&&selection.seq_start)||0}, ${end})${selection&&selection.max_frames?` · 每条≤${selection.max_frames}帧`:''}`;
}
const BENCH_HEAD_LABELS = Object.freeze({
  extrinsics:'相机轨迹', camera_trajectory:'全长相机轨迹', hands_world:'世界系手部运动',
  hands_coverage:'双手检测与 coverage-aware 姿态',
  hands:'相机系手部姿态', depth:'深度', intrinsics:'相机内参', world_points:'世界点云',
});
function benchSelectedWork(){
  const grouped=new Map();
  document.querySelectorAll('#bfGrid .bfck:checked').forEach(box=>{
    if(!grouped.has(box.dataset.d)) grouped.set(box.dataset.d,[]);
    grouped.get(box.dataset.d).push(box.dataset.h);
  });
  const range=benchRunRange(), perDataset=range.datasetSelection||{};
  return [...grouped].map(([dataset,heads])=>{
    const size=(state.bench.sizes||{})[dataset]||{};
    const option=perDataset[dataset]||{};
    const available=Number.isFinite(+size.n_seqs)?Math.max(0,+size.n_seqs)
      : (Object.keys(option).length?benchDatasetAvailable(dataset):null);
    const start=range.seqStart||0, requestedEnd=range.seqEnd;
    const end=available==null?requestedEnd:(requestedEnd==null?available:Math.min(available,requestedEnd));
    const count=option.sample_count!=null ? Math.min(available==null?option.sample_count:available,option.sample_count)
      : (option.sampling==='all'?available:(end==null?null:Math.max(0,end-start)));
    const coverage=dataset.endsWith('_hand_coverage');
    const meta=benchDatasetMeta(dataset);
    return {dataset,heads,label:meta.label||dataset,count,
            maxFrames:option.max_frames||range.maxFrames,
            unit:coverage?'个 81 帧片段':'个源序列'};
  });
}
function renderBenchWorkSummary(){
  const el=$('#bfWorkSummary'); if(!el) return;
  const work=benchSelectedWork();
  const combos=work.reduce((sum,item)=>sum+item.heads.length,0);
  const models=Math.max(state.bench.selectedModels.length,(state.bench.modelResults||[]).length);
  if(!work.length){
    el.innerHTML='<span class="bf-work-empty">当前没有选择任何评测组合。</span>';
    return;
  }
  const items=work.map(item=>{
    const count=item.count==null?'数量统计中':`${item.count} ${item.unit}${item.maxFrames?` · 每条≤${item.maxFrames}帧`:''}`;
    const heads=item.heads.map(head=>BENCH_HEAD_LABELS[head]||head).join(' + ');
    const reuse=item.dataset==='hot3d'&&item.heads.includes('extrinsics')&&item.heads.includes('hands_world')
      ? '；两个功能共用同一次模型预测':'';
    return `<article class="bf-work-item"><span><b>${_esc(item.label)}</b><strong>${_esc(count)}</strong></span>`
      +`<p class="bf-work-heads">实际运行：${_esc(heads+reuse)}</p></article>`;
  }).join('');
  const hasHand=work.some(item=>item.heads.some(head=>['hands','hands_world','hands_coverage'].includes(head)));
  const ukf=Boolean(state.bench.autoUkf&&hasHand);
  const modelText=models
    ? `${models} 个原始模型将依次重复同一套协议${ukf?'，完成后追加 1 次最佳模型 UKF 复测':''}`
    : '添加模型后运行';
  el.innerHTML=`<header><b>当前实际评测</b><span>${work.length} 个协议任务 · ${combos} 个功能组合 · ${_esc(modelText)}</span></header>`
    +`<div class="bf-work-items">${items}</div>`;
}
function updateBenchCount(){
  const el = $('#bfCount'); if(!el) return;
  const s = benchSelection();
  const models=state.bench.selectedModels.length;
  const ukf=Boolean(state.bench.autoUkf&&s.heads.some(head=>['hands','hands_world','hands_coverage'].includes(head)));
  const backend=state.bench.backend||'local';
  let resource='';
  if(backend==='aliyun'){
    const nodes=Math.floor(+($('#bfAliNodes')&&$('#bfAliNodes').value)||0), gpus=Math.floor(+($('#bfAliGpus')&&$('#bfAliGpus').value)||0);
    resource=` · Aliyun ${nodes} node × ${gpus} 卡`;
  }else{
    resource=` · 本地 ${benchGpuSelection().length} 卡`;
  }
  el.textContent = s.n
    ? `将跑 ${models} 个原始模型 × ${s.n} 个组合${ukf?' + 最佳模型 UKF 复测':''}${resource}`
    : '未选组合';
  renderBenchSampling();
  renderBenchWorkSummary();
}

async function startBench(){
  if(state.bench.running) return;
  const models=state.bench.selectedModels.map(model=>({
    run:model.run, step:model.step, path:model.path||null, label:model.label,
  }));
  if(!models.length){ state.bench = {...state.bench, error:'请先添加至少一个 Benchmark 模型', phase:'未选模型'}; renderBench(); return; }
  const sel = benchSelection();
  if(!sel.n){ state.bench = {...state.bench, error:'请先在网格里勾选至少一个「功能×数据集」组合', phase:'未选组合'}; renderBench(); return; }
  const backend=state.bench.backend||'local';
  const devices = backend==='local' ? benchGpuSelection() : [];
  if(backend==='local' && !devices.length){ state.bench = {...state.bench, error:'请至少勾选一张显卡', phase:'未选显卡'}; renderBench(); return; }
  const aliyun=backend==='aliyun'?benchAliyunConfig():null;
  if(backend==='aliyun' && (!(aliyun.nnodes>=1) || !(aliyun.gpus_per_node>=1))){
    state.bench={...state.bench,error:'Aliyun 节点数和每节点 GPU 数必须大于 0',phase:'资源配置无效'}; renderBench(); return;
  }
  const {seqStart,seqEnd,maxFrames,datasetSelection}=benchRunRange();
  if(seqStart<0 || (seqEnd!==null && seqEnd<=seqStart)){
    state.bench={...state.bench, error:'序列范围必须满足 0 <= start < end（end 可留空）', phase:'范围无效'};
    renderBench(); return;
  }
  if(maxFrames!==null && maxFrames<=0){
    state.bench={...state.bench, error:'max-frames 必须大于 0（或留空）', phase:'范围无效'};
    renderBench(); return;
  }
  const reuseCache=Boolean($('#bfReuseCache')&&$('#bfReuseCache').checked);
  const autoUkf=Boolean(state.bench.autoUkf&&sel.heads.some(head=>['hands','hands_world','hands_coverage'].includes(head)));
  console.log('[btn] 运行 benchmark backend='+backend+' models='+models.map(model=>model.label).join(',')+' heads='+sel.heads+' datasets='+sel.datasets+' gpus='+devices);
  // 传勾选的 head 并集 × dataset 并集；非矩形勾选多出的组合由后端能力匹配自动 skip（日志可见）。
  const pendingModels=models.map(model=>({...model,status:'pending'}));
  if(autoUkf) pendingModels.push({run:'',step:'',variant:'ukf',hand_mode:'smooth',status:'pending',
                                 label:'UKF融合 · 等待选择平均质量最佳模型'});
  state.bench = {...state.bench, backend, aliyun:aliyun||state.bench.aliyun, running:true, phase:'启动中…', count:0, report:null, live_report:null,
                 modelResults:pendingModels, activeModel:null,
                 error:null, out:null, log:[], progress:{}, liveUpdatedAt:null, _t0:Date.now()};
  _benchDsOpen.clear();
  _benchModelDetailsOpen = false;
  renderBench();
  try{
    const r = await fetch(U('/api/benchmark/start'), {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({heads:sel.heads.join(','), datasets:sel.datasets.join(','),
                            seq_start:seqStart, seq_end:seqEnd,
                            max_frames:maxFrames, dataset_selection:datasetSelection,
                            devices, models,
                            reuse_cache:reuseCache, auto_ukf_best:autoUkf, backend, aliyun})});
    if(!r.ok){ const j = await r.json().catch(()=>null);
      state.bench = {...state.bench, running:false, phase:'未启动', error:(j&&j.error)||await r.text().catch(()=>'启动失败')};
      renderBench(); return; }
  }catch(e){ state.bench = {...state.bench, running:false, phase:'未启动', error:e.message}; renderBench(); return; }
  startBenchPoll();
}
async function cancelBench(){
  console.log('[btn] 取消 benchmark');
  try{ await fetch(U('/api/benchmark/cancel'), {method:'POST'}); }catch(e){}
}
let _benchPolling = false;
const _benchDsOpen = new Set();   // 展开中的数据集名（每数据集评完可点开看局部结果表；跨轮询保持）
let _benchModelDetailsOpen = false; // 完整报告会被轮询重绘，显式保留原生 details 的展开状态
async function startBenchPoll(){
  if(_benchPolling) return; _benchPolling = true;
  while(_benchPolling){
    try{ const s = await getJSON('/api/benchmark/status');
      state.bench = {...state.bench, running:s.running, phase:s.phase, count:s.count, report:s.report,
                     live_report:s.live_report,
                     liveUpdatedAt:(s.live_report||s.report)?Date.now():state.bench.liveUpdatedAt,
                     modelResults:s.models||[], activeModel:s.active_model||null,
                     error:s.error, out:s.out, log:s.log||[], progress:s.progress||{},
                     backend:s.backend||state.bench.backend, job:s.job||null,
                     job_id:s.job_id||null, job_status:s.job_status||null,
                     aliyun:s.aliyun||state.bench.aliyun,
                     autoUkf:typeof (s.selection||{}).auto_ukf_best==='boolean'
                       ? s.selection.auto_ukf_best : state.bench.autoUkf,
                     selection:s.selection||{}};   // 保留 caps/viz（不被状态覆盖）
      renderBench();
      if((s.running || s.report || s.error) && state.hidden.has('bench')){
        showBench(true);                         // 隐藏时才自动展开(避免每次轮询都重建面板)
        if(!state.bench.caps){ loadBenchCaps(); loadBenchSizes(); loadBenchGpus(); loadBenchAliyunDefaults(); }
      }
      if(!s.running){
        _benchPolling = false;
        loadBenchHistory();
        break;
      }
    }catch(e){}
    await new Promise(r=>setTimeout(r,state.bench.backend==='aliyun'?1800:800));
  }
}
function benchProgressValues(b){
  const pr=b.progress||{};
  const reported=(typeof pr.suite_frac==='number')?pr.suite_frac:pr.frac;
  const terminal=!b.running&&(
    !!b.report||!!b.error||(b.modelResults||[]).some(model=>['completed','failed','cancelled'].includes(model.status))
  );
  const frac=terminal?1:Math.max(0,Math.min(1,Number.isFinite(reported)?reported:0));
  const modelTotal=Math.max(1,Math.floor(+pr.model_total||(b.modelResults||[]).length||(b.selectedModels||[]).length||1));
  const modelIndex=Math.max(1,Math.min(modelTotal,Math.floor(+pr.model_index||1)));
  const currentTotal=Math.max(0,Math.floor(+pr.total||0));
  const currentDone=Math.max(0,Math.floor(+pr.done||0));
  const total=currentTotal?currentTotal*modelTotal:0;
  const done=total?Math.min(total,(modelIndex-1)*currentTotal+currentDone):currentDone;
  const fallbackElapsed=b._t0?Math.max(0,(Date.now()-b._t0)/1000):null;
  const elapsed=pr.elapsed_s!=null&&Number.isFinite(+pr.elapsed_s)?Math.max(0,+pr.elapsed_s):fallbackElapsed;
  let estimated=pr.estimated_total_s!=null&&Number.isFinite(+pr.estimated_total_s)?Math.max(0,+pr.estimated_total_s):null;
  let remaining=pr.remaining_s!=null&&Number.isFinite(+pr.remaining_s)?Math.max(0,+pr.remaining_s):null;
  if(terminal&&elapsed!=null){ estimated=elapsed; remaining=0; }
  else if(elapsed!=null&&frac>=0.005){
    if(estimated==null) estimated=elapsed/frac;
    if(remaining==null) remaining=Math.max(0,estimated-elapsed);
  }
  return {pr,terminal,frac,modelTotal,modelIndex,total,done,elapsed,estimated,remaining};
}
function renderBenchProgress(b){
  const card=$('#bfProgressCard'); if(!card) return;
  const v=benchProgressValues(b);
  const visible=b.running||v.terminal||v.elapsed!=null;
  card.hidden=!visible;
  if(!visible) return;
  const percent=Math.round(v.frac*100);
  const liveStages=(v.pr.gpus||[]).map(gpu=>gpu&&gpu.live&&gpu.live.stage).filter(Boolean);
  const prepareStage=liveStages.find(stage=>stage==='读取 GT / 准备输入')
    ||liveStages.find(stage=>stage&&stage!=='枚举序列')
    ||(liveStages.length?'读取数据集清单':'加载模型 / 启动评测进程');
  const bar=$('#bfProgressBar'), track=card.querySelector('.bf-run-progress-track');
  const status=$('#bfProgressStatus'), percentEl=$('#bfProgressPercent');
  card.classList.toggle('indet',b.running&&!v.total);
  card.classList.toggle('done',v.terminal&&!b.error);
  if(bar) bar.style.width=percent+'%';
  if(track){ track.setAttribute('aria-valuenow',String(percent)); track.setAttribute('aria-valuetext',`${percent}%`); }
  if(percentEl) percentEl.textContent=percent+'%';
  const active=(b.activeModel||{}).label||v.pr.model_label||'准备模型';
  if(status) status.textContent=b.running
    ? `模型 ${v.modelIndex}/${v.modelTotal} · ${active}`
    : (b.error&&!b.report?'任务失败':b.phase||'任务结束');
  const done=$('#bfProgressDone'), elapsed=$('#bfProgressElapsed');
  const totalTime=$('#bfProgressTotalTime'), remaining=$('#bfProgressRemaining'), note=$('#bfProgressNote');
  if(done) done.textContent=v.total?`${v.done} / ${v.total} 评测单元`:`${v.done} / 准备中`;
  if(elapsed) elapsed.textContent=_dur(v.elapsed);
  if(totalTime) totalTime.textContent=_dur(v.estimated);
  if(remaining) remaining.textContent=v.remaining==null?'等待首批完成':_dur(v.remaining);
  if(note){
    if(v.terminal) note.textContent=`任务已经结束；实际墙钟耗时 ${_dur(v.elapsed)}。`;
    else if(!v.total) note.textContent=`当前阶段：${prepareStage}。数据集清单通常只需数秒；读取 GT 和视频输入会在总量确定后逐单元进行。`;
    else if(v.estimated==null) note.textContent='总评测单元已经确定，正在收集首批完成速度。';
    else note.textContent=`时间根据当前吞吐滚动估算，模型加载和数据准备也计入已用时间；${v.modelTotal} 个模型按顺序执行。`;
  }
}

// Polling rebuilds the live tables, so carry each keyed horizontal position forward.
function replaceBenchLiveHTML(root, html){
  const positions=new Map();
  root.querySelectorAll('[data-bench-scroll-key]').forEach(element=>{
    positions.set(element.dataset.benchScrollKey,element.scrollLeft);
  });
  root.innerHTML=html;
  root.querySelectorAll('[data-bench-scroll-key]').forEach(element=>{
    const left=positions.get(element.dataset.benchScrollKey);
    if(Number.isFinite(left)) element.scrollLeft=left;
  });
}

function renderBench(){
  const b = state.bench, pr = b.progress || {}, displayResults=benchDisplayResults(b);
  const btn = $('#benchBtn'), run = $('#bfRun'), cancel = $('#bfCancel');
  const phase = $('#bfPhase'), prog = $('#bfProg'), bar = $('#bfBar');
  const steps = $('#bfSteps'), table = $('#bfTable'), live = $('#bfLive'), liveHint = $('#bfLiveHint'),
        sotaLive=$('#bfSotaLive'), sotaHint=$('#bfSotaLiveHint'), log = $('#bfLog'), grid = $('#bfGrid');
  renderBenchSuitePanel();
  renderBenchReferenceTables();
  renderBenchExecutor();
  if(btn) btn.classList.toggle('on', b.running);   // 运行中高亮；按钮仍可点（重开结果区，不重复启动）
  if(run){ run.disabled = b.running; run.style.display = b.running?'none':''; }
  if(cancel) cancel.style.display = b.running?'':'none';
  if(grid) grid.querySelectorAll('input,button').forEach(el=> el.disabled = b.running);  // 运行中锁网格
  renderBenchModels();
  renderBenchSampling();
  renderBenchWorkSummary();
  renderBenchProgress(b);
  for(const id of ['#bfSeqStart','#bfSeqEnd','#bfFrames','#bfReuseCache','#bfAutoUkf']){
    const el=$(id); if(el) el.disabled=b.running;
  }
  const autoUkf=$('#bfAutoUkf'); if(autoUkf) autoUkf.checked=Boolean(b.autoUkf);
  if(phase){
    phase.className = 'step ' + (b.error && !b.running ? 'err' : b.running ? 'busy' : b.report ? 'ok' : '');
    phase.textContent = (b.error && !b.running) ? ('✗ '+(b.phase||'失败')+'：'+b.error)
                      : (b.phase || (b.report ? '✓ 完成' : '待运行'));
  }
  // 进度条：运行中有分步进度→确定百分比；无 frac（刚启动/等模型）→不确定动画；完成→100%
  const overallFrac=(typeof pr.suite_frac==='number')?pr.suite_frac:pr.frac;
  const hasFrac = b.running && pr && typeof overallFrac === 'number' && (pr.total||0) > 0;
  if(prog){
    if(hasFrac){ prog.style.display='inline-block'; prog.classList.remove('indet'); if(bar) bar.style.width=Math.round(overallFrac*100)+'%'; }
    else if(b.running){ prog.style.display='inline-block'; prog.classList.add('indet'); }
    else if(b.report){ prog.style.display='inline-block'; prog.classList.remove('indet'); if(bar) bar.style.width='100%'; }
    else prog.style.display='none';
  }
  // 总体摘要 + 每卡一条固定状态行；轮询只替换这些行，不追加刷屏。
  if(steps){
    if(b.running){
      const seg = [];
      if(b.backend==='aliyun'){
        const ali=b.aliyun||{}, job=b.job_id?`DLC ${b.job_id}`:'DLC 提交中';
        seg.push(`${job} · ${ali.nnodes||'?'} node × ${ali.gpus_per_node||8} 卡`);
      }
      if(pr.model_total) seg.push(`模型 ${pr.model_index||1}/${pr.model_total} · ${pr.model_label||((b.activeModel||{}).label||'加载中')}`);
      seg.push(`已完成 ${pr.done||0}/${pr.total||0} 序列`);
      const sel=b.selection||{};
      if(sel.seq_start!=null || Object.keys(sel.dataset_selection||{}).length) seg.push(benchSelectionSummary(sel));
      if(hasFrac){
        seg.push(`全部模型 ${Math.round(overallFrac*100)}%`);
        if(overallFrac >= 0.999){                                            // 全部模型已评完，剩收尾
          seg.push('收尾中…');
        }else if(b._t0 && overallFrac > 0.02){                               // 用全部模型进度估算剩余(ETA)
          const el = (Date.now() - b._t0)/1000, eta = el*(1 - overallFrac)/overallFrac;
          seg.push('约剩 ' + (eta < 90 ? Math.round(eta)+'s' : Math.round(eta/60)+'m'));
        }
      }
      const gpuLines = (pr.gpus||[]).map(g=>{
        const e=g.live||{}, gpuLabel=b.backend==='aliyun'?`node${g.node}/gpu${g.local_gpu}`:`gpu${g.index}`;
        const parts=[gpuLabel, `${g.done||0}/${g.total||0}`];
        if(e.ds) parts.push(e.ds);
        if(e.seq_id) parts.push(e.seq_id);
        const stage=e.stage||(e.kind==='seq_done'?'完成':'');
        if(stage) parts.push(stage);
        if((e.win_total||0)>0) parts.push(`${e.win_done||0}/${e.win_total}窗`);
        return `<div class="bfgpu"><span class="bfgpu-dot"></span>${parts.map(_esc).join('<span>·</span>')}</div>`;
      }).join('');
      steps.innerHTML = `<div class="bf-overall">${seg.map(_esc).join(' · ')}</div><div class="bfgpu-lines">${gpuLines}</div>`;
      steps.style.display = '';
    }else{ steps.innerHTML=''; steps.style.display='none'; }
  }
  // 数据集进度与评测结果分区：运行中不混在一起，完成后结果区切到最终报告。
  if(table){
    if(b.running || (pr.ds_order||[]).length && !b.report) table.innerHTML = benchDsPanelHTML(pr);
    else table.innerHTML = '';
    // 绑定每数据集行的展开/收起（仅已完成且有结果的可点）
    table.querySelectorAll('.bfds-h[data-openable="1"]').forEach(el => el.onclick = ()=>{
      const ds = el.dataset.ds;
      if(_benchDsOpen.has(ds)) _benchDsOpen.delete(ds); else _benchDsOpen.add(ds);
      renderBench();
    });
  }
  if(sotaLive){
    const liveTitle=sotaLive.closest('.bf-sota-live')?.querySelector('.bf-live-title b');
    if(liveTitle) liveTitle.textContent=benchSuiteIsLocalBaseline()?'固定表 + 本次模型 · 实时':'公开 SOTA + 本次模型 · 实时';
    replaceBenchLiveHTML(
      sotaLive,benchSotaLiveHTML(displayResults,b.live_report,b.activeModel,b.running));
    if(sotaHint){
      const updated=b.liveUpdatedAt?new Date(b.liveUpdatedAt).toLocaleTimeString('zh-CN',{hour12:false})
        : (b.historyResults||[]).length?'已加载指定历史记录':'尚无当前结果';
      const cadence=b.backend==='aliyun'?'约 1.8 秒':'约 0.8 秒';
      const modelCount=Math.max(displayResults.length,(b.selectedModels||[]).length+(b.historyResults||[]).length);
      const rows=modelCount>1?`M1–M${modelCount} 分行 · `:'';
      sotaHint.textContent=b.running?`${rows}${cadence}刷新 · 最近 ${updated}`:`${rows}最终结果 · ${updated}`;
    }
  }
  if(live){
    const details = live.querySelector('.bf-model-details');
    if(details) _benchModelDetailsOpen = details.open;
    replaceBenchLiveHTML(live,benchComparisonHTML(
      displayResults, b.live_report, b.activeModel, b.running, _benchModelDetailsOpen));
    const groupFilter=live.querySelector('[data-bench-group-filter]');
    if(groupFilter) groupFilter.onchange=()=>{
      state.bench.comparisonGroup=groupFilter.value;
      renderBench();
    };
    if(liveHint) liveHint.textContent = b.running
      ? (b.backend==='aliyun'?'从 CPFS 汇总各节点进度；模型聚合完成后加入对比列':'约 0.8s 更新当前滚动均值；已完成模型保留在对比列中')
      : '按各记录自己的 split 横向展示；确认范围与协议一致后再比较，两个以上模型时红色为当前最优';
  }
  if(log) log.textContent = (b.log||[]).join('\n');
}
// 每数据集进度面板：一行一集，进度条 + 状态；已完成且有结果的可点开内联展开该集局部结果表。
function benchDsPanelHTML(pr){
  const order = pr.ds_order || [], dsets = pr.datasets || {};
  if(!order.length) return '<div class="bmut">评测进行中…（等各卡枚举数据集）</div>';
  let h = '<div class="bfds-wrap">';
  for(const ds of order){
    const d = dsets[ds] || {done:0, total:0, status:'pending', report:null};
    const st = d.status || 'pending';
    const pct = d.total ? Math.round((d.done/d.total)*100) : (st==='done'?100:0);
    const badge = st==='done' ? '✓ 完成' : (st==='running' ? '评测中…' : '待评测');
    // 时间：运行中→预计总时长(已用/已完成×总数)；完成→实际耗时
    let tm = '';
    if(st==='done' && d.elapsed_s!=null) tm = `耗时 ${_dur(d.elapsed_s)}`;
    else if(st==='running'){
      if(d.eta_s!=null) tm = `预计 ${_dur(d.eta_s)}` + (d.elapsed_s!=null?` (已用 ${_dur(d.elapsed_s)})`:'');
      else if(d.elapsed_s!=null) tm = `已用 ${_dur(d.elapsed_s)}`;
    }
    const openable = st==='done' && d.report && Object.keys((d.report.heads)||{}).length;
    const open = _benchDsOpen.has(ds);
    const arrow = openable ? (open?'▾':'▸') : '';
    h += `<div class="bfds">`
       + `<div class="bfds-h ${st}" data-ds="${ds}" data-openable="${openable?1:0}">`
       + `<span class="bfds-arw">${arrow}</span>`
       + `<span class="bfds-name" title="${_esc(ds)}">${_esc(benchDatasetLabel(ds))}</span>`
       + `<span class="bfds-bar"><span class="bfds-fill" style="width:${pct}%"></span></span>`
       + `<span class="bfds-num">${d.done||0}/${d.total||0}</span>`
       + `<span class="bfds-eta">${tm}</span>`
       + `<span class="bfds-badge ${st}">${badge}</span>`
       + `</div>`;
    if(openable && open) h += `<div class="bfds-body">${benchTableHTML(d.report)}</div>`;
    h += `</div>`;
  }
  h += '</div>';
  return h;
}
const _dur = s => (s==null || !isFinite(s)) ? '—' : (s<60 ? Math.round(s)+'s' : (s<3600 ? Math.floor(s/60)+'m'+Math.round(s%60)+'s' : Math.floor(s/3600)+'h'+Math.round(s%3600/60)+'m'));
const _nfmt = v => (v==null || !isFinite(v)) ? '—' : (Number.isInteger(v) ? String(v) : (+v).toFixed(4));
const _tfmt = s => (s==null || !isFinite(s)) ? '—' : (+s).toFixed(2)+'s';   // 秒

function benchSuiteMetricForRow(row, suite=currentBenchSuite()){
  if(suite.id==='custom') return row.metric;
  return benchSuiteMetricAlias(row.metric,suite);
}
function benchSuiteRowVisible(row){
  const suite=currentBenchSuite();
  if(suite.id==='custom') return true;
  if(!(suite.combos||[]).some(([head,dataset])=>head===row.head&&dataset===row.dataset)) return false;
  const metric=benchSuiteMetricForRow(row,suite);
  return metric!==null && benchMetricSelection(suite).has(metric);
}
function benchMetricSpec(metricId){
  return benchSuiteMetrics().find(item=>item.id===metricId)||null;
}
function benchMetricDirection(metricId){
  const metric=benchMetricSpec(metricId);
  if(metric) return metric.direction;
  return ['FAcc','Recall','F1'].includes(metricId)?'max':'min';
}
function benchBestMetricValue(values,direction){
  if(!values.length) return null;
  if(direction==='max') return Math.max(...values);
  if(direction==='target') return values.reduce((best,value)=>Math.abs(value-1)<Math.abs(best-1)?value:best);
  return Math.min(...values);
}
function benchMetricDirectionMark(direction){
  return direction==='max'?'↑':(direction==='target'?'→ 1':'↓');
}
function renderBenchReferenceTable(table, selectedMetrics){
  if(table.liveOnly) return '';
  const columns=(table.columns||[]).map((column,index)=>({...column,index}))
    .filter(column=>selectedMetrics.has(column.metric));
  if(!columns.length) return '';
  const best=new Map();
  for(const column of columns){
    const values=(table.rows||[]).filter(row=>row.comparable!==false)
      .map(row=>benchComparableValue(table,column,row.values[column.index])).filter(Number.isFinite);
    if(values.length) best.set(column.index,benchBestMetricValue(values,column.direction));
  }
  let body='', previousSection=null;
  const localBaseline=table.referenceKind==='local-baseline';
  for(const row of table.rows||[]){
    if(row.section && row.section!==previousSection){
      body+=`<tr class="bf-public-section"><th colspan="${columns.length+1}">${_esc(row.section)}</th></tr>`;
      previousSection=row.section;
    }
    body+=`<tr><th>${_esc(row.method)}</th>`;
    for(const column of columns){
      const value=row.values[column.index], comparable=benchComparableValue(table,column,value);
      const isBest=row.comparable!==false&&Number.isFinite(comparable)&&comparable===best.get(column.index);
      body+=`<td class="bmetrics${isBest?' sota':''}"${isBest?` title="${localBaseline?'该离线基线快照此列最优':'该公开表此列 SOTA'}"`:''}>${_esc(benchTableValue(table,column,value))}</td>`;
    }
    body+='</tr>';
  }
  return `<article class="bf-reference-card"><header><span><b>${_esc(table.title)}</b><small>${_esc(table.source)}</small></span>`
    +`<p>${_esc(table.note||'')}</p></header><div class="bf-public-table"><table class="nt bencht"><thead><tr><th>Method</th>`
    +columns.map(column=>{
      const spec=benchMetricSpec(column.metric), rawUnit=column.unit||(spec&&spec.unit)||'';
      const unit=rawUnit?` · ${rawUnit}`:'';
      return `<th>${_esc(column.label)}<small>${benchMetricDirectionMark(column.direction)}${_esc(unit)}</small></th>`;
    }).join('')
    +`</tr></thead><tbody>${body}</tbody></table></div></article>`;
}
function renderBenchReferenceTables(){
  const root=$('#bfReferenceTables'); if(!root) return;
  const suite=currentBenchSuite();
  if(suite.id==='custom'||suite.id==='sota'){
    root.innerHTML=''; root.hidden=true; return;
  }
  root.hidden=false;
  const selected=benchMetricSelection(suite);
  const tables=benchSuiteTables(suite).map(table=>renderBenchReferenceTable(table,selected)).filter(Boolean);
  const localBaseline=benchSuiteIsLocalBaseline(suite);
  root.innerHTML=`<div class="bf-reference-title"><span><b>${localBaseline?'固定离线基线':'论文公开结果'}</b><small>${localBaseline?'从 ICRA 全量结果快照读取；点击运行不会重跑外部 SLAM 环境':'公开 test 表与本地评测 split 不混算；每张表逐列标记最优值'}</small></span><i>红色 = ${localBaseline?'该快照此列最优':'公开表 SOTA'}</i></div>`
    +(tables.join('')||'<div class="bmut">未选择展示指标。评测不会重跑，重新勾选即可恢复表格。</div>');
}

function benchSuiteIsLocalBaseline(suite=currentBenchSuite()){
  return (suite.tables||[]).some(table=>table.referenceKind==='local-baseline');
}

function benchTableValue(table,column,value){
  if(column.format==='coverage'){
    if(typeof value==='string') return value;
    if(Number.isFinite(value)) return `${Math.round(value)}/${table.expectedSequences||'?'}`;
    return '—';
  }
  if(typeof value==='string') return value;
  if(value==null||!Number.isFinite(+value)) return '—';
  return Number.isInteger(column.digits)?Number(value).toFixed(column.digits):_nfmt(value);
}

function benchRowValue(model,row){
  const node=((((model.report||{}).heads||{})[row.head]||{})[row.dataset]||{});
  const groups=node.mean_by_side||{overall:node.mean||{}};
  return (groups[row.group]||{})[row.metric];
}

function benchDisplayResults(bench=state.bench){
  return [...(bench.modelResults||[]),...(bench.historyResults||[])];
}

function benchModelColumnMeta(model){
  const signature=model.benchmark_signature||{};
  const datasetSelection=(signature.selection||{}).dataset_selection||{};
  const tiers=[...new Set(Object.values(datasetSelection).map(option=>option&&option.fixed_tier).filter(Boolean))];
  const tier=model.historical?model.history_sampling_tier:(tiers.length?tiers.join('+'):'custom');
  const protocol=model.historical?model.history_protocol_revision:signature.protocol_revision;
  const variant=model.variant==='ukf'?'UKF · ':'';
  const time=model.historical?` · ${benchHistoryTime(model.history_recorded_at,true)}`:'';
  return `${model.step||''} · ${variant}${benchHistoryTierLabel(tier)}${time}${protocol?' · '+protocol:''}`;
}

function benchResultEntries(results,liveReport,activeModel){
  const entries=(results||[]).map(model=>({...model}));
  if(activeModel && liveReport){
    const index=entries.findIndex(model=>
      model.run===activeModel.run&&model.step===activeModel.step
      &&(model.variant||'raw')===(activeModel.variant||'raw'));
    const live={...(index>=0?entries[index]:activeModel), status:'running', report:liveReport};
    if(index>=0) entries[index]=live; else entries.push(live);
  }
  return entries;
}

function benchGlobalMetricSpec(metricId){
  for(const suite of Object.values(BENCHMARK_SUITES)){
    const metric=(suite.metrics||[]).find(item=>item.id===metricId);
    if(metric) return metric;
  }
  return null;
}

function benchLiveNode(report,spec){
  return (((report||{}).heads||{})[spec.head]||{})[spec.dataset]||null;
}

function benchLiveRuleValue(report,node,spec,rule){
  const source=(rule.dataset||rule.head)
    ?((((report||{}).heads||{})[rule.head||spec.head]||{})[rule.dataset||spec.dataset]||null)
    :node;
  if(!source) return null;
  const groups=source.mean_by_side||{overall:source.mean||{}};
  const metrics=rule.source==='counts'?(source.counts||{}):(groups[spec.group||'overall']||{});
  const raw=metrics[rule.metric];
  if(!Number.isFinite(raw)) return null;
  let value=Number(raw);
  if(rule.transform==='target-error') value=Math.abs(value-1);
  if(rule.transform==='target-error-pct') value=Math.abs(value-1)*100;
  return value*(Number.isFinite(rule.scale)?rule.scale:1)+(Number.isFinite(rule.offset)?rule.offset:0);
}

function benchLiveTableValue(report,node,spec,column){
  const mapping=(spec.values||{})[column.key];
  if(!mapping) return null;
  if(Array.isArray(mapping.average)){
    const values=mapping.average.map(rule=>benchLiveRuleValue(report,node,spec,rule));
    return values.length&&values.every(Number.isFinite)
      ? values.reduce((sum,value)=>sum+value,0)/values.length : null;
  }
  if(mapping.balancedCameraScore){
    const config=mapping.balancedCameraScore, datasetScores=[];
    for(const dataset of config.datasets||[]){
      const source=((((report||{}).heads||{})[spec.head]||{})[dataset.dataset]||null);
      const coverage=source&&Number(source.counts&&source.counts.sequences)/dataset.expectedSequences;
      if(!source||!Number.isFinite(coverage)||coverage<=0) return null;
      const costs=(config.metrics||[]).map(metric=>benchLiveRuleValue(
        report,source,{...spec,dataset:dataset.dataset},typeof metric==='string'?{metric}:metric));
      if(costs.length!==dataset.bases.length||!costs.every(Number.isFinite)) return null;
      const normalized=costs.map((value,index)=>value/dataset.bases[index]);
      datasetScores.push(100*normalized.reduce((sum,value)=>sum+value,0)/normalized.length/coverage);
    }
    return datasetScores.length&&datasetScores.every(Number.isFinite)
      ? datasetScores.reduce((sum,value)=>sum+value,0)/datasetScores.length : null;
  }
  if(mapping.balancedDatasetScore){
    const config=mapping.balancedDatasetScore, datasetScores=[];
    for(const dataset of config.datasets||[]){
      const source=((((report||{}).heads||{})[spec.head]||{})[dataset.dataset]||null);
      if(!source) return null;
      const values=(config.metrics||[]).map(metric=>benchLiveRuleValue(
        report,source,{...spec,dataset:dataset.dataset},{metric:metric.metric}));
      if(values.length!==dataset.bases.length||!values.every(Number.isFinite)) return null;
      const normalized=values.map((value,index)=>config.metrics[index].direction==='max'
        ? dataset.bases[index]/value : value/dataset.bases[index]);
      datasetScores.push(100*normalized.reduce((sum,value)=>sum+value,0)/normalized.length);
    }
    return datasetScores.length&&datasetScores.every(Number.isFinite)
      ? datasetScores.reduce((sum,value)=>sum+value,0)/datasetScores.length : null;
  }
  const rule=typeof mapping==='string'?{metric:mapping}:mapping;
  return benchLiveRuleValue(report,node,spec,rule);
}

function benchComparableValue(table,column,value){
  if(column.format==='coverage'){
    if(Number.isFinite(value)) return Number(value)/Math.max(1,Number(table.expectedSequences)||1);
    const match=String(value||'').match(/^(\d+)\/(\d+)$/);
    return match&&+match[2]>0 ? +match[1]/+match[2] : null;
  }
  return Number.isFinite(value)?Number(value):null;
}

function benchSotaCurrentRows(entries,table){
  const rows=[];
  entries.forEach((model,modelIndex)=>{
    for(const spec of table.liveRows||[]){
      const node=benchLiveNode(model.report,spec);
      const values=(table.columns||[]).map(column=>benchLiveTableValue(model.report,node,spec,column));
      const sameSplit=Boolean(node&&node.protocol&&node.protocol.reference_same_split);
      const comparable=spec.comparable!==false&&(!spec.requiresSameSplit||sameSplit);
      const status=model.status||'pending';
      rows.push({
        model, modelIndex, spec, values, comparable, status,
        hasValues:values.some(Number.isFinite),
      });
    }
  });
  return rows;
}

function benchSotaTableHTML(suite,table,entries,running){
  const columns=table.columns||[], publicBest=new Map();
  for(const [index,column] of columns.entries()){
    const values=(table.rows||[]).filter(row=>row.comparable!==false)
      .map(row=>benchComparableValue(table,column,row.values[index])).filter(Number.isFinite);
    if(values.length) publicBest.set(index,benchBestMetricValue(values,column.direction));
  }
  let body='', previousSection=null;
  for(const row of table.rows||[]){
    if(row.section && row.section!==previousSection){
      body+=`<tr class="bf-public-section"><th colspan="${columns.length+1}">${_esc(row.section)}</th></tr>`;
      previousSection=row.section;
    }
    body+=`<tr class="bf-sota-public"><th>${_esc(row.method)}</th>`;
    columns.forEach((column,index)=>{
      const value=row.values[index], comparable=benchComparableValue(table,column,value);
      const best=row.comparable!==false&&Number.isFinite(comparable)&&comparable===publicBest.get(index);
      const bestTitle=table.referenceKind==='local-baseline'?'该固定对比表此列最优':'该论文公开表此列最优';
      body+=`<td class="bmetrics${best?' sota':''}"${best?` title="${bestTitle}"`:''}>${_esc(benchTableValue(table,column,value))}</td>`;
    });
    body+='</tr>';
  }
  const currentRows=benchSotaCurrentRows(entries,table);
  for(const row of currentRows){
    const modelLabel=row.model.label||`${row.model.run||''} / ${row.model.step||''}`;
    const variantLabel=row.model.variant==='ukf'?'UKF融合 · ':'';
    const stateLabel=row.hasValues?(row.status==='running'?'滚动均值':(row.model.historical?'指定历史记录':(row.model.cache_hit?'复用缓存':'最终值'))):(row.status==='running'||row.status==='pending'?'等待数据':'本次未评测');
    const compareLabel=table.liveOnly?'本地尺度诊断':(row.comparable?'同 split 可比':'非同 split / 协议参考');
    body+=`<tr class="bf-sota-current ${_esc(row.status)}"><th title="${_esc(modelLabel)}"><b>M${row.modelIndex+1}</b> · ${_esc(variantLabel+row.spec.label)}<span>${_esc(stateLabel)} · ${_esc(compareLabel)}</span></th>`;
    columns.forEach((column,index)=>{
      const value=row.values[index], comparable=benchComparableValue(table,column,value), best=publicBest.get(index);
      const beats=row.comparable&&Number.isFinite(comparable)&&Number.isFinite(best)
        &&(column.direction==='max'?comparable>best:(column.direction==='target'?Math.abs(comparable-1)<Math.abs(best-1):comparable<best));
      body+=`<td class="bmetrics current${beats?' beats-sota':''}"${beats?' title="同协议下数值优于固定表最优值"':''}>${_esc(benchTableValue(table,column,value))}</td>`;
    });
    body+='</tr>';
  }
  if(!entries.length){
    body+=`<tr class="bf-sota-empty"><th>本次模型</th><td colspan="${columns.length}">添加一个或多个模型并运行后，将显示为 M1、M2… 独立结果行</td></tr>`;
  }
  const scrollKey=`sota:${suite.id}:${table.id||table.title}`;
  return `<article class="bf-sota-card"><header><span><b>${_esc(suite.label)} · ${_esc(table.title)}</b><strong>${_esc(table.source)}</strong></span><p>${_esc(table.note||'')}</p></header>`
    +`<div class="bf-public-table" data-bench-scroll-key="${_esc(scrollKey)}"><table class="nt bencht"><thead><tr><th>Method</th>`
    +columns.map(column=>{
      const metric=benchGlobalMetricSpec(column.metric), rawUnit=column.unit||(metric&&metric.unit)||'';
      return `<th>${_esc(column.label)}<span>${benchMetricDirectionMark(column.direction)}${rawUnit?' · '+_esc(rawUnit):''}</span></th>`;
    }).join('')
    +`</tr></thead><tbody>${body}</tbody></table></div></article>`;
}

function benchSotaLiveHTML(results,liveReport,activeModel,running){
  const entries=benchResultEntries(results,liveReport,activeModel);
  const current=currentBenchSuite();
  const dedicated=['camera','hot3d_camera','arctic_camera','vidihand'].includes(current.id);
  const suiteIds=current.tableSuiteIds||(dedicated?[current.id]:['hot3d_camera','arctic_camera','vidihand']);
  const suites=suiteIds.map(id=>BENCHMARK_SUITES[id]).filter(Boolean);
  const cards=[];
  for(const suite of suites){
    for(const table of suite.tables||[]){
      if((table.liveRows||[]).length) cards.push(benchSotaTableHTML(suite,table,entries,running));
    }
  }
  return cards.join('');
}

function benchComparisonHTML(results, liveReport, activeModel, running, modelDetailsOpen=false){
  const entries=benchResultEntries(results,liveReport,activeModel);
  const statusText={pending:'等待',running:'评测中',completed:'完成',failed:'失败',cancelled:'已取消'};
  let h='<div class="bf-compare-status">'+entries.map((model,index)=>{
    const status=model.status||'pending';
    const ranking=model.quality_ranking||{};
    const rankText=model.variant==='ukf'&&Number.isFinite(+ranking.score)
      ? ` · 自动优选损失 ${_nfmt(+ranking.score)}`:'';
    const historyText=model.historical
      ? `历史 · ${model.variant==='ukf'?'UKF · ':''}${model.cache_hit?'复用缓存 · ':''}${benchHistoryTierLabel(model.history_sampling_tier)} · ${benchHistoryTime(model.history_recorded_at,true)}`:'';
    return `<span class="bf-compare-model ${_esc(status)}${model.historical?' historical':''}"><b>M${index+1}</b>`
      + `<code title="${_esc(model.ckpt||model.label||'')}">${_esc(model.label||`${model.run||''} / ${model.step||''}`)}</code>`
      + `<span>${_esc(historyText||(model.cache_hit?'复用缓存':(statusText[status]||status)))}${rankText}${model.benchmark_log?' · 已写时间戳记录':''}</span>${model.error?`<i title="${_esc(model.error)}">!</i>`:''}</span>`;
  }).join('')+'</div>';
  const reports=entries.filter(model=>model.report&&Object.keys(model.report.heads||{}).length);
  if(!reports.length) return h+`<div class="bmut">${running?'等待正在运行的模型完成首条序列…':'尚无可比较结果'}</div>`;

  const rowMap=new Map();
  for(const model of reports){
    for(const [head,datasets] of Object.entries(model.report.heads||{})){
      for(const [dataset,node] of Object.entries(datasets||{})){
        const groups=node.mean_by_side||{overall:node.mean||{}};
        for(const [group,metrics] of Object.entries(groups)){
          for(const metric of Object.keys(metrics||{})){
            const key=JSON.stringify([head,dataset,group,metric]);
            if(!rowMap.has(key)) rowMap.set(key,{head,dataset,group,metric});
          }
        }
      }
    }
  }
  const suiteRows=[...rowMap.values()].filter(benchSuiteRowVisible);
  const groupOrder={overall:0,left:1,right:2};
  const groups=[...new Set(suiteRows.map(row=>row.group))].sort((a,b)=>
    (groupOrder[a]??10)-(groupOrder[b]??10)||a.localeCompare(b));
  const requestedGroup=state.bench.comparisonGroup||'overall';
  const groupFilter=requestedGroup==='all'||groups.includes(requestedGroup)
    ? requestedGroup : (groups.includes('overall')?'overall':'all');
  state.bench.comparisonGroup=groupFilter;
  const rows=suiteRows.filter(row=>groupFilter==='all'||row.group===groupFilter).sort((a,b)=>
    a.head.localeCompare(b.head)||a.dataset.localeCompare(b.dataset)||a.group.localeCompare(b.group)||a.metric.localeCompare(b.metric));
  if(rows.length){
    const groupOptions=[...groups,'all'].map(group=>`<option value="${_esc(group)}"${group===groupFilter?' selected':''}>${_esc(group==='all'?'全部':group)}</option>`).join('');
    h+='<div class="bf-compare-table" data-bench-scroll-key="model-comparison"><table class="nt bencht"><thead><tr><th>head</th><th>dataset</th>'
      + `<th class="bf-compare-group-filter"><label><span>范围</span><select data-bench-group-filter title="筛选评测范围">${groupOptions}</select></label></th><th>metric</th>`
      + entries.map((model,index)=>`<th title="${_esc(model.label||'')}">M${index+1}<small>${_esc(benchModelColumnMeta(model))}</small></th>`).join('')
      + '</tr></thead><tbody>';
    for(const row of rows){
      const metricId=benchSuiteMetricForRow(row)||row.metric;
      const values=entries.map(model=>benchRowValue(model,row));
      const numeric=values.filter(value=>Number.isFinite(value));
      const best=numeric.length>1?benchBestMetricValue(numeric,benchMetricDirection(metricId)):null;
      const unit=benchSuiteMetricUnit(row.metric);
      const metricLabel=(metricId===row.metric?metricId:`${metricId} · ${row.metric}`)+(unit?` [${unit}]`:'');
      h+=`<tr><td>${_esc(row.head)}</td><td title="${_esc(row.dataset)}">${_esc(benchDatasetLabel(row.dataset))}</td><td>${_esc(row.group)}</td><td>${_esc(metricLabel)}</td>`;
      for(const value of values){
        const isBest=best!==null&&Number.isFinite(value)&&value===best;
        h+=`<td class="bmetrics${isBest?' sota-local':''}"${isBest?' title="当前本地 split 的模型间最优值"':''}>${_nfmt(value)}</td>`;
      }
      h+='</tr>';
    }
    h+='</tbody></table></div>';
  }else h+='<div class="bmut">当前协议下尚无已选指标的本地结果。</div>';
  h+=`<details class="bf-model-details"${modelDetailsOpen?' open':''}><summary>查看各模型完整报告</summary>`;
  for(const [index,model] of entries.entries()){
    if(!model.report) continue;
    h+=`<section><h4>M${index+1} · ${_esc(model.label||'')}</h4>${benchTableHTML(model.report)}</section>`;
  }
  return h+'</details>';
}

// 结果按 head 分组卡片：每卡一张小表（dataset 行 × metric 列 + eval 耗时列），未评测的数据集折叠进「未评测」小节。
function benchTableHTML(rep){
  const heads = (rep && rep.heads) || {}, names = Object.keys(heads).sort();
  if(!names.length) return '<div class="bmut">（无结果）</div>';
  const T = (rep && rep.timings) || null;        // 计时：{total_s, datasets:{ds:{total_s,predict_s,n_seqs,heads}}}
  const tds = (T && T.datasets) || {};
  let h = '';
  if(rep.ckpt) h += `<div class="bmut">ckpt: <code>${rep.ckpt}</code></div>`;
  if(rep.selection){
    h += `<div class="bmut">测评范围: <code>${_esc(benchSelectionSummary(rep.selection))}</code></div>`;
  }
  if(T){                                          // 顶部耗时摘要：总耗时 + 每数据集 前向/总
    h += `<div class="btime"><b>耗时</b> 总 ${_tfmt(T.total_s)}`;
    const seg = Object.keys(tds).sort().map(d =>
      `${d} <span class="mut">总${_tfmt(tds[d].total_s)}·前向${_tfmt(tds[d].predict_s)}</span>`).join('　');
    if(seg) h += ` ｜ ${seg}`;
    h += '</div>';
  }
  for(const hd of names){
    const dss = heads[hd], dnames = Object.keys(dss).sort();
    const evald = dnames.filter(d => dss[d].mean);
    const skipped = dnames.filter(d => !dss[d].mean);
    h += `<div class="bcard"><div class="bcard-h">${hd}</div>`;
    const protocolNotes=evald.map(d=>({dataset:d, protocol:dss[d].protocol})).filter(x=>x.protocol);
    for(const item of protocolNotes){
      const p=item.protocol;
      if(p.sequence_mode){
        h += `<div class="bmut">${_esc(benchDatasetLabel(item.dataset))}: ${_esc(p.name||'protocol')} · ${_esc(p.evaluation_split||'full sequence')} · ${_esc(p.alignment||'whole sequence')} · ${_esc(p.inference||'windowed')}</div>`;
      }else{
        const split=p.official_split?'official':'non-official';
        const mode=p.clip_single_forward?'81 帧单次前向':'含分窗前向（与论文 clip inference 不一致）';
        h += `<div class="bmut">${_esc(benchDatasetLabel(item.dataset))}: ${_esc(p.name||'protocol')} · ${split} split · seed=${_esc(p.split_seed)} · ${mode}</div>`;
      }
    }
    if(evald.length){
      const hasSideMeans=evald.some(d=>dss[d].mean_by_side);
      const cols = [...new Set(evald.flatMap(d => Object.values(dss[d].mean_by_side||{overall:dss[d].mean}).flatMap(m=>Object.keys(m||{}))))];
      h += '<table class="nt bencht"><thead><tr><th>dataset</th>'+(hasSideMeans?'<th>范围</th>':'')+'<th>序列</th>'
         + cols.map(c=>`<th>${c}</th>`).join('') + (T ? '<th>eval(s)</th>' : '') + '</tr></thead><tbody>';
      for(const d of evald){
        const node=dss[d], seqIds=Object.keys(node.seqs||{});
        const groups=node.mean_by_side||{overall:node.mean};
        const et = (tds[d] && tds[d].heads) ? tds[d].heads[hd] : null;
        for(const group of ['overall','left','right']){
          if(!groups[group]) continue;
          const ns=group==='overall'?seqIds.length:seqIds.filter(id=>id.endsWith('#'+group)).length;
          h += `<tr><td title="${_esc(d)}">${_esc(benchDatasetLabel(d))}</td>${hasSideMeans?`<td>${group}</td>`:''}<td>${ns}</td>`
             + cols.map(c=>`<td class="bmetrics">${_nfmt(groups[group][c])}</td>`).join('')
             + (T ? `<td class="bmetrics">${_tfmt(et)}</td>` : '') + '</tr>';
        }
        const reference=((node.reference||{}).metrics)||null;
        if(reference){
          h += `<tr><td>${_esc(benchDatasetLabel(d))} · public reference</td>${hasSideMeans?'<td>overall</td>':''}<td>Table 1</td>`
             + cols.map(c=>`<td class="bmetrics">${_nfmt(reference[c])}</td>`).join('')
             + (T?'<td>—</td>':'')+'</tr>';
          const delta=node.delta_vs_reference||{};
          h += `<tr><td>${_esc(benchDatasetLabel(d))} · ours-public</td>${hasSideMeans?'<td>overall</td>':''}<td>delta</td>`
             + cols.map(c=>`<td class="bmetrics">${_nfmt(delta[c])}</td>`).join('')
             + (T?'<td>—</td>':'')+'</tr>';
        }
      }
      h += '</tbody></table>';
    }
    if(skipped.length){
      const items = skipped.map(d=>{
        const sc = dss[d].status_counts||{}, tag = Object.keys(sc).map(k=>`${k}×${sc[k]}`).join(', ');
        return `${d}（${tag}${dss[d].note? ' · '+dss[d].note : ''}）`;
      }).join('；');
      h += `<details class="bskip"><summary>未评测 ${skipped.length} 个数据集</summary><div class="bmut">${items}</div></details>`;
    }
    h += '</div>';
  }
  return h;
}

// ── 配置/代码变更（log_diff）：选两个 run → 后端子进程跑 tools/log_diff.py → 只展示实际变更 ──
function _esc(s){ return String(s==null?'':s).replace(/[&<>"']/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function wireLogDiff(){
  const btn = $('#logDiffBtn');
  if(btn) btn.onclick = ()=>{ showLd(true); loadLdRuns(); };
  const run = $('#ldRun'), close = $('#ldClose');
  if(run) run.onclick = ()=> startLd();
  if(close) close.onclick = ()=> showLd(false);         // ✕ 仅收起，后台对比继续
  const A = $('#ldA'), B = $('#ldB');
  if(A) A.onchange = loadLdScopes;
  if(B) B.onchange = loadLdScopes;
  renderLd();
}
// 配置/代码变更面板同样纳入栏目系统:显隐=切 state.hidden 并重建;on 时滚动到位。
function showLd(on){
  if(on) state.hidden.delete('logdiff'); else state.hidden.add('logdiff');
  buildPanels(true);
  if(on){ const s = document.querySelector('#blocks section[data-pid="logdiff"]');
          if(s) s.scrollIntoView({behavior:'smooth', block:'start'}); }
}
async function loadLdRuns(){
  let data = null;
  try{ data = await getJSON('/api/logdiff/runs'); }catch(e){}
  const runs = (data && data.runs) || [];
  state.logdiff.runs = runs; state.logdiff.loaded = true;
  const A = $('#ldA'), B = $('#ldB');
  if(!A || !B) return;
  if(!runs.length){
    A.innerHTML = B.innerHTML = '<option value="">（未找到含 logs/node*.log 的 run）</option>';
    const scope = $('#ldScope');
    if(scope){ scope.innerHTML = '<option value="">全部代码</option>'; scope.disabled = true; }
    return;
  }
  const opts = runs.map(r=>`<option value="${_esc(r)}">${_esc(r)}</option>`).join('');
  A.innerHTML = opts; B.innerHTML = opts;
  // 默认 A=当前 ckpt 所属 run，B=另一个（不同则取列表里第一个非 A）
  const cur = data && data.current;
  A.value = (cur && runs.includes(cur)) ? cur : runs[0];
  B.value = runs.find(r=> r !== A.value) || runs[Math.min(1, runs.length-1)];
  await loadLdScopes();
}
let _ldScopeToken = 0;
async function loadLdScopes(){
  const token = ++_ldScopeToken;
  const A = $('#ldA'), B = $('#ldB'), scope = $('#ldScope');
  if(!A || !B || !scope) return;
  const previous = scope.value || state.logdiff.scope || '';
  const a = A.value, b = B.value;
  scope.disabled = true;
  scope.innerHTML = '<option value="">读取模块…</option>';
  if(!a || !b){ scope.innerHTML = '<option value="">全部代码</option>'; return; }
  try{
    const data = await getJSON('/api/logdiff/scopes?run_a='+encodeURIComponent(a)+'&run_b='+encodeURIComponent(b));
    if(token !== _ldScopeToken) return;
    const scopes = Array.isArray(data.scopes) ? data.scopes : [];
    state.logdiff.scopes = scopes;
    scope.innerHTML = '<option value="">全部代码</option>'
      + scopes.map(s=>`<option value="${_esc(s)}">${_esc(s)}</option>`).join('');
    scope.value = scopes.includes(previous) ? previous : '';
    state.logdiff.scope = scope.value;
  }catch(e){
    if(token !== _ldScopeToken) return;
    state.logdiff.scopes = [];
    scope.innerHTML = '<option value="">全部代码（模块读取失败）</option>';
    state.logdiff.scope = '';
  }finally{
    if(token === _ldScopeToken) scope.disabled = state.logdiff.running;
  }
}
async function startLd(){
  if(state.logdiff.running) return;
  const a = $('#ldA').value, b = $('#ldB').value, codeScope = $('#ldScope').value || '';
  if(!a || !b){ state.logdiff = {...state.logdiff, error:'请先选择两个 run', phase:'未选'}; renderLd(); return; }
  if(a === b){ state.logdiff = {...state.logdiff, error:'两个 run 相同，无可对比', phase:'未选'}; renderLd(); return; }
  console.log('[btn] 运行 log_diff A='+a+' B='+b+' scope='+(codeScope||'全部代码'));
  state.logdiff = {...state.logdiff, running:true, phase:'启动中…', result:null, error:null,
                   log:[], scope:codeScope};
  renderLd();
  try{
    const r = await fetch(U('/api/logdiff/start'), {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({run_a:a, run_b:b, code_scope:codeScope})});
    if(!r.ok){ const j = await r.json().catch(()=>null);
      state.logdiff = {...state.logdiff, running:false, phase:'未启动', error:(j&&j.error)||'启动失败'};
      renderLd(); return; }
  }catch(e){ state.logdiff = {...state.logdiff, running:false, phase:'未启动', error:e.message}; renderLd(); return; }
  startLdPoll();
}
let _ldPolling = false;
async function startLdPoll(){
  if(_ldPolling) return; _ldPolling = true;
  while(_ldPolling){
    try{ const s = await getJSON('/api/logdiff/status');
      state.logdiff = {...state.logdiff, running:s.running, phase:s.phase, result:s.result,
                       error:s.error, log:s.log||[]};
      renderLd();
      if((s.running || s.result || s.error) && state.hidden.has('logdiff')) showLd(true);   // 隐藏时才自动展开(避免每次轮询都重建面板)
      if(!s.running){ _ldPolling = false; break; }
    }catch(e){}
    await new Promise(r=>setTimeout(r, 800));
  }
}
function renderLd(){
  const b = state.logdiff;
  const run = $('#ldRun'), phase = $('#ldPhase'), prog = $('#ldProg'), bar = $('#ldBar');
  const table = $('#ldTable'), btn = $('#logDiffBtn');
  const A = $('#ldA'), B = $('#ldB'), scope = $('#ldScope');
  if(btn) btn.classList.toggle('on', b.running);
  if(run){ run.disabled = b.running; run.textContent = b.running ? '对比中…' : '▶ 对比'; }
  if(A) A.disabled = b.running; if(B) B.disabled = b.running;
  if(scope) scope.disabled = b.running;
  if(phase){
    phase.className = 'step ' + (b.error && !b.running ? 'err' : b.running ? 'busy' : b.result ? 'ok' : '');
    phase.textContent = (b.error && !b.running) ? ('✗ '+(b.phase||'失败')+'：'+b.error)
                      : (b.phase || (b.result ? '✓ 完成' : '待运行'));
  }
  if(prog){
    if(b.running){ prog.style.display='inline-block'; prog.classList.add('indet'); }
    else if(b.result){ prog.style.display='inline-block'; prog.classList.remove('indet'); if(bar) bar.style.width='100%'; }
    else prog.style.display='none';
  }
  if(table){
    table.innerHTML = b.result ? ldResultHTML(b.result)
      : (b.running ? '<div class="bmut">正在提取配置和代码变更…</div>' : '');
  }
}
// 把 unified diff 文本渲成带色 <pre>，弱化上下文，让实际增删行成为视觉焦点。
function _patchHTML(patch){
  if(!patch) return '<div class="bmut">代码无变化。</div>';
  const groups = [];
  for(const ln of patch.split('\n')){
    let cls = 'ld-ctx';
    if(ln.startsWith('+++') || ln.startsWith('---')) cls = 'ld-fh';
    else if(ln.startsWith('@@')) cls = 'ld-hunk';
    else if(ln.startsWith('+')) cls = 'ld-add';
    else if(ln.startsWith('-')) cls = 'ld-del';
    const last = groups[groups.length-1];
    if(last && last.cls === cls) last.lines.push(ln);
    else groups.push({cls, lines:[ln]});
  }
  const html = groups.map(g=>`<span class="${g.cls}">${_esc(g.lines.join('\n'))}</span>`).join('\n');
  return `<pre class="ld-patch">${html}</pre>`;
}
function ldResultHTML(r){
  const cf = r.config || {}, chg = cf.changed||[], oa = cf.only_a||[], ob = cf.only_b||[];
  const count = chg.length + oa.length + ob.length;
  const missing = '<span class="ld-missing">—</span>';
  let crows = chg.map(c=>`<tr><td><code>${_esc(c.key)}</code></td><td class="ld-old">${_esc(c.a)}</td><td class="ld-new">${_esc(c.b)}</td></tr>`).join('');
  crows += oa.map(c=>`<tr><td><code>${_esc(c.key)}</code></td><td class="ld-old">${_esc(c.value)}</td><td>${missing}</td></tr>`).join('');
  crows += ob.map(c=>`<tr><td><code>${_esc(c.key)}</code></td><td>${missing}</td><td class="ld-new">${_esc(c.value)}</td></tr>`).join('');
  let out = `<div class="ld-card"><div class="ld-h">配置变更 · ${count} 项</div>`;
  if(count) out += `<table class="nt ld-config"><thead><tr><th>配置项</th><th>A（原值）</th><th>B（新值）</th></tr></thead><tbody>${crows}</tbody></table>`;
  else out += '<div class="bmut">配置无变化。</div>';
  out += '</div>';
  out += `<div class="ld-card"><div class="ld-h">代码变更 · ${_esc(r.code_scope || '全部代码')}</div>`;
  if(!r.code){ out += `<div class="bmut">${_esc(r.code_note || '至少一侧无 logs/record/code 快照，代码层跳过')}</div>`; }
  else out += _patchHTML(r.code.patch);
  out += '</div>';
  return out;
}

// 该内容(content)在 overlay/side 下各画哪些 item(GT 实线 / PRED 虚线)。
function sceneItems(){
  if(state.no_truth) return {ov:[{d:state.pred,dash:false}], gt:[{d:state.pred,dash:false,tag:'PRED'}], pred:[]};
  if(state.rawOnly)  return {ov:[{d:state.gt,dash:false}],   gt:[{d:state.gt,dash:false,tag:'GT'}],   pred:[]};
  return {ov:[{d:state.gt,dash:false},{d:state.pred,dash:true}],
          gt:[{d:state.gt,dash:false,tag:'GT'}], pred:[{d:state.pred,dash:false,tag:'PRED'}]};
}
function draw(){
  const it = state.loaded ? sceneItems() : null;
  for(const id of state.order){
    if(state.hidden.has(id)) continue;
    const pan = state.panels[id]; if(!pan) continue;
    if(pan.kind==='scene' && it){
      const c = pan.content;
      if(state.layout==='overlay'){
        if(pan.cOV) renderScene(pan.cOV, it.ov, pan.v.vov, c, masterVideo);
      }else{
        if(pan.cGT) renderScene(pan.cGT, it.gt, pan.v.vgt, c, masterVideo);
        if(pan.cPred) renderScene(pan.cPred, it.pred, pan.v.vpred, c, masterVideo);
      }
    } else if(pan.kind==='nums'){ drawIntegratedNums(pan); }
  }
  if(!state.no_truth && !state.rawOnly && !state.hidden.has('loss')) drawMetrics();
  if(state.loaded){                                // 仅加载后覆盖 info，未加载时保留 pending 提示
    const source = state.sourceName ? state.sourceName + ' · ' : '';
    const name = state.no_truth ? (state.sourceName || state.vidName || `#${state.eid}`)
               : source + (state.epIdx!=null ? `ep ${String(state.epIdx).padStart(4,'0')}` : `#${state.eid}`);
    const idx = state.no_truth ? `视频项 #${state.eid}`
              : `Episode 序号 ${state.epOrdinal!=null ? state.epOrdinal : '?'} / ${state.epTotal!=null ? state.epTotal : '?'}`;
    const tail = state.rawOnly ? '· 仅原始GT（未推理）' : `· ckpt ${CKPT_TAG}`;
    info.textContent = `${name} (${idx}) · ${state.mode} · fps ${state.fps} · ${state.nframes} 帧 ${tail}`;
    info.title = state.sourcePath || '';
  }
}

// 视图交互：左键平移、Ctrl+左键 orbit、滚轮缩放。window 监听只挂一次，避免重建时累积。
let _activeV = null;
window.addEventListener('mousemove', e=>{ const v=_activeV; if(!v || !v.drag) return;
  const dx=e.clientX-v.drag.x, dy=e.clientY-v.drag.y;
  if(v.drag.rotate){
    v.az+=dx*.01; v.el+=dy*.01; v.el=Math.max(-1.5,Math.min(1.5,v.el));
  }else{
    v.panX=(v.panX||0)+dx; v.panY=(v.panY||0)+dy;
  }
  v.drag.x=e.clientX; v.drag.y=e.clientY;
  requestDraw();
});
window.addEventListener('mouseup', ()=>{
  if(_activeV&&_activeV.drag){
    if(_activeV.drag.canvas) _activeV.drag.canvas.classList.remove('rotating');
    _activeV.drag=null;
  }
  _activeV=null;
});
function attachOrbit(canvas, v, onChange){
  canvas.addEventListener('mousedown', e=>{
    if(e.button!==0) return;
    _activeV=v; v.drag={x:e.clientX,y:e.clientY,rotate:e.ctrlKey,canvas};
    canvas.classList.toggle('rotating',e.ctrlKey); e.preventDefault();
  });
  canvas.addEventListener('wheel', e=>{ e.preventDefault();
    v.zoom *= (e.deltaY<0 ? 1.1 : 0.9); v.zoom=Math.max(0.2, Math.min(8, v.zoom));
    if(onChange) onChange(); requestDraw(); }, {passive:false});
  canvas.addEventListener('mouseup',()=>{ if(onChange) onChange(); });
}

// GT/PRED 叠加/并排控件跟随「整体·2D」面板重建，因此每次创建面板时重新绑定。
function wireLayoutControls(root){
  root.querySelectorAll('#layoutSeg button').forEach(button=>button.onclick=()=>{
    if(state.layout===button.dataset.layout) return;
    state.layout=button.dataset.layout;
    state.layoutPref=state.layout;
    console.log('[btn] GT/PRED 对照布局 → '+state.layout);
    if(state.loaded) buildPanels(true);
  });
}

function syncMujocoButton(){
  for(const [selector,id] of [['#mujocoBtn','mujoco_3d'],['#retargetBtn','wuji_retarget_3d']]){
    const button=$(selector);
    if(button){
      const on=!state.hidden.has(id);
      button.classList.toggle('on',on);
      button.setAttribute('aria-pressed',on?'true':'false');
    }
  }
}
function toggleMujocoPanel(){
  if(state.hidden.has('mujoco_3d')) state.hidden.delete('mujoco_3d');
  else state.hidden.add('mujoco_3d');
  console.log('[btn] MuJoCo 仿真 → '+(!state.hidden.has('mujoco_3d')));
  buildPanels(true);
}
function toggleRetargetPanel(){
  if(state.hidden.has('wuji_retarget_3d')) state.hidden.delete('wuji_retarget_3d');
  else state.hidden.add('wuji_retarget_3d');
  console.log('[btn] Wuji Hand Retargeting → '+(!state.hidden.has('wuji_retarget_3d')));
  buildPanels(true);
}
// 右侧顺序调节栏：列 state.order，每项 ☰拖动手柄排序 + 👁 显隐（再点恢复）。
function renderOrderBar(){
  const ul = $('#orderBar'); if(!ul) return;
  ul.innerHTML = state.order.map(id=>{
    const P = PANEL_BY[id], hid = state.hidden.has(id);
    return `<li draggable="true" data-id="${id}" class="oitem${hid?' off':''}">`
      + `<span class="oh">☰</span><span class="onm">${P?P.name:id}</span>`
      + `<button class="oeye" title="显示/隐藏">${hid?'🚫':'👁'}</button></li>`;
  }).join('');
  ul.querySelectorAll('.oeye').forEach(btn => btn.onclick = e=>{
    e.stopPropagation(); const id = btn.closest('li').dataset.id;
    if(state.hidden.has(id)) state.hidden.delete(id); else state.hidden.add(id);
    buildPanels(true);
  });
  syncMujocoButton();
  let dragId = null;
  ul.querySelectorAll('li').forEach(li=>{
    li.addEventListener('dragstart', ()=>{ dragId=li.dataset.id; li.classList.add('drag'); });
    li.addEventListener('dragend',   ()=>{ li.classList.remove('drag'); dragId=null; });
    li.addEventListener('dragover',  e=>{ e.preventDefault(); li.classList.add('over'); });
    li.addEventListener('dragleave', ()=>{ li.classList.remove('over'); });
    li.addEventListener('drop', e=>{ e.preventDefault(); li.classList.remove('over');
      const tgt = li.dataset.id; if(!dragId || dragId===tgt) return;
      const o = state.order.slice(); o.splice(o.indexOf(dragId), 1);   // 摘出被拖项
      o.splice(o.indexOf(tgt), 0, dragId);                             // 插到目标项之前
      state.order = o; buildPanels(true);
    });
  });
}

// 新视频帧或交互使视图失效时才重画；fallback 仅做廉价帧号比较，不再每个 RAF 全量绘制。
function loop(){
  requestAnimationFrame(loop);
  if(masterVideo && !_videoFrameDriven){
    const frame=frameOf(masterVideo);
    if(frame!==_fallbackFrame){ _fallbackFrame=frame; requestDraw(); }
  }
  if(!_drawPending) return;
  _drawPending=false;
  draw();
}
window.addEventListener('resize', requestDraw);
requestAnimationFrame(loop);
init().catch(e=> info.textContent = '加载失败: '+(e && e.message || e));
