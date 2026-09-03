
## 可视化观感（固定世界 3D / Wuji Hand，2026-08-05）
- 固定世界 3D 是**两套并行实现**：网页 `viewer/web/app.js:renderScene` 与导出 `render/fixed_world_video.py`。视角默认值与地面网格规格（`VIEW_AZ0/EL0`、`GRID_SPAN`、`GRID_TARGET_CELLS`、`GRID_MAJOR_EVERY`、`GROUND_CLEARANCE`）由 `test_viewer_frontend_contract.py` 逐条比对，改一端必须同步另一端。该坐标系无天然重力轴，「上」统一取**全段相机 +Y(下) 平均值取反**（Wuji 面板 `_gravity_up` 同口径）。
- 官方左右手 MJCF 可安全合成一个模型：名字都带 `left_/right_` 前缀不撞名，但两侧 `<compiler meshdir>` 冲突，必须把 `<mesh file>` 改写成绝对路径；视觉 geom 是 `group="1"`（各 26 个）、其余是重复碰撞网格。合并后靠 `mjvOption.geomgroup` 逐帧隐藏未检测的手（不出现也不投影），深度合成可整段删掉。
- 固定第三视角下 MuJoCo/Wuji 共用相机与地面口径：取景只包围双手，参考地面位于最低关节点下约 0.08 m；不要重新把完整相机轨迹计入取景包围盒，否则手会缩得过小。
- MuJoCo/Wuji 机器人视频不画连续轨迹，只共用 1 个固定起点框和 1 个随帧移动的实时框。alpha 分别为 0.18 和 0.98，使用 `mjGEOM_LINE` 避免投影；完整轨迹只保留在固定世界 3D。
- 服务端导出的每路画面都使用原视频宽高；带 GT 时按四列两排组织：第一排为 `2D GT/PRED + Fixed World GT/PRED`，第二排为 `MuJoCo GT/PRED + Wuji GT/PRED`。例如 HOT3D 每格保持 512×512，八格合成为 2048×1024，不能压成 960×540 宽屏。
