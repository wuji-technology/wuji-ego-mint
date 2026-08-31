<div align="center">

<!--
  FULL FILM (1 min 41 s) — the promo film opens the README, above the hero
  wall. Source: `out/MINT_full_en.mp4` in the promo project, 1920 × 1080,
  72 MB, kept out of this repository on purpose. GitHub renders an inline
  player only for github.com/user-attachments URLs, and those can only be
  minted through the browser: open any issue or PR comment box in this
  repository, drag the MP4 into it, copy the generated
  https://github.com/user-attachments/assets/...
  link, and paste it on its own line right below this comment. Do not commit
  the MP4 — a <video> tag pointing at a file in the repository is stripped by
  GitHub's markdown sanitizer and renders as nothing.
-->

<img src="assets/readme/scale.webp" width="100%" alt="108 MINT renders playing at once, one tile per clip">

<h2>MINT: Minting a Unified Model for World-Space Camera and Hand Motion<br>from Scalable Egocentric Pipeline Supervision</h2>

Zijie Zhu<sup>1,3,4</sup> &nbsp;·&nbsp; Weiren Cai<sup>3</sup> &nbsp;·&nbsp; Yizhou Wang<sup>1,3</sup> &nbsp;·&nbsp; Zhenjie Yang<sup>4</sup> &nbsp;·&nbsp; Yide Liu<sup>3,5</sup> &nbsp;·&nbsp; Jiahao Chen<sup>3,\*</sup> &nbsp;·&nbsp; Guanqi He<sup>2,3,\*</sup>

<sub><sup>1</sup>ShanghaiTech University &nbsp;&nbsp;<sup>2</sup>Tsinghua University &nbsp;&nbsp;<sup>3</sup>Wuji Technology &nbsp;&nbsp;<sup>4</sup>The University of Hong Kong &nbsp;&nbsp;<sup>5</sup>Zhejiang University &nbsp;&nbsp;<sup>\*</sup>Corresponding authors</sub>

<!-- TODO: paste the arXiv link into the badge below once it is live. -->
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b)]()
[![Model](https://img.shields.io/badge/%F0%9F%A4%97_Model-mint__v1-ff9d00)](https://huggingface.co/ZZJAsher/mint_v1)
[![Model](https://img.shields.io/badge/ModelScope_Model-mint__v1-624aff)](https://www.modelscope.cn/models/AsherZhu/mint_v1)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97_Dataset-1,021_hours-ff9d00)](https://huggingface.co/datasets/ZZJAsher/wuji_ego_mint)
[![Dataset](https://img.shields.io/badge/ModelScope_Dataset-1,021_hours-624aff)](https://www.modelscope.cn/datasets/AsherZhu/wuji_ego_mint)
[![License](https://img.shields.io/badge/License-MIT-3fa03f)](LICENSE)

[中文说明](README_ZH.md)

</div>

<img src="assets/readme/world_space.webp" width="100%" alt="Eight panels: device ground truth beside MINT prediction in the ego view, in world space, and retargeted to the Wuji hand in MuJoCo">

---

### 🤲 Meet MINT — world-space egocentric hand data anyone can afford to produce

Where the person wearing the camera walked, where each hand went and what it did there — in the real space around them, not in pixels on the screen. Getting that today means buying capture hardware with SLAM built in, or chaining calibration → monocular depth → SLAM → hand reconstruction → trajectory cleanup: a weight file, a failure mode and a license restriction per stage. The ability to produce this data ends up with the few labs that can run the chain — while large-scale real human hand data is exactly what embodied AI is shortest of.

MINT replaces the chain. **One ordinary RGB video, one 24 GB GPU, one forward pass** gives you world-space camera and two-hand motion. If you have video, you can produce data.

- **One unified model, not a five-stage chain.** A shared spatiotemporal representation feeds four heads — camera extrinsics, field of view, camera-frame MANO, per-frame hand presence — and explicit differentiable rigid composition turns them into world-space hand motion. No dense 3D intermediate at inference.
- **Structured pipeline amortization.** The multi-stage pipeline stays offline as a supervision generator, and one model learns its final structured state — not logit distillation, but the camera–hand state of an entire non-end-to-end system. Running that system is paid for once, by us, rather than once per user.
- **Open end to end.** Model weights, training and inference code, the EgoPipeline labeling system, and a filtered **1,021-hour** structured egocentric dataset — reproducible, auditable and yours to extend, not a service you query.

<img src="assets/readme/pipeline_vs_mint.webp" width="100%" alt="Same frames through the conventional ego pipeline and through MINT: the pipeline places both hands away from the real hands, MINT keeps them on the hands">

<div align="center"><sub>Same frames, same overlay renderer. Left: the input. Middle: the conventional ego pipeline, whose output is <b>pseudo-label, not ground truth</b>. Right: MINT, one forward pass.</sub></div>

---

## 📑 Table of Contents

<details open>
<summary>Collapse</summary>

- [📋 Release status](#-release-status)
- [🚀 Quick Start: Web Viewer](#-quick-start-web-viewer)
- [🧠 How MINT works](#-how-mint-works)
- [📊 What MINT is measured on](#-what-mint-is-measured-on)
- [📦 What is included](#-what-is-included)
- [🏋️ Training and optional pipeline reconstruction](#️-training-and-optional-pipeline-reconstruction)
- [🗂️ Public Ego pretraining data](#️-public-ego-pretraining-data)
- [🔐 LeRobot sample and privacy](#-lerobot-sample-and-privacy)
- [⚠️ Known limits](#️-known-limits)
- [🧾 Repository layout](#-repository-layout)
- [📚 Documentation](#-documentation)
- [✨ Acknowledgements](#-acknowledgements)
- [📜 License](#-license)
- [📖 Citation](#-citation)

</details>

---

## 📋 Release status

| | Item | Where |
| :-- | :-- | :-- |
| ✅ | MINT checkpoint, 1.131 B parameters | [Hugging Face](https://huggingface.co/ZZJAsher/mint_v1) · [ModelScope](https://www.modelscope.cn/models/AsherZhu/mint_v1) |
| ✅ | Inference, Web Viewer, training code | this repository |
| ✅ | 1,021-hour structured egocentric dataset (non-video portion) | [Hugging Face](https://huggingface.co/datasets/ZZJAsher/wuji_ego_mint) · [ModelScope](https://www.modelscope.cn/datasets/AsherZhu/wuji_ego_mint) |
| ✅ | Benchmark implementation and CLI, integrated into the Viewer | `eval/model_effect/benchmark/` |
| ✅ | EgoPipeline orchestration, cleaning and LeRobot export reference | `ego_pipeline/` |
| ✅ | Video normalisation and hand-filtering scripts | [`ego_pipeline/preprocessing/`](ego_pipeline/preprocessing/) |
| ✅ | Wuji hand URDF/MJCF/STL and retargeting | `eval/simulate/wuji-retargeting/` |
| ✅ | Zero-shot HOT3D / ARCTIC result table (paper Table 1) | [Camera-frame bimanual reconstruction](#camera-frame-bimanual-reconstruction) |
| ⏳ | Scale-corrected camera trajectories — the released dataset's trajectories are scale-enlarged | The current ones are usable for pretraining, not for metric evaluation. Cause in [Public Ego pretraining data](#️-public-ego-pretraining-data) |
| ❌ | License-restricted pipeline adaptations (adapted HaWoR source, weights, MANO) | cannot be redistributed; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) |

## 🚀 Quick Start: Web Viewer

The Web Viewer is the primary MINT entry point. It lets you select a checkpoint, load the MINT model, run inference on a LeRobot episode or an egocentric MP4 video, and inspect GT/prediction overlays, camera trajectories, hand motion, frame metrics, and benchmark results in one interface.

> **Hardware requirement: MINT model inference requires an NVIDIA GPU with at least 24 GB of VRAM.**

Install the inference environment and download the public MINT model. The Viewer also requires the separately licensed MANO hand models. Create an account on the [official MANO website](https://mano.is.tue.mpg.de/), accept the MANO license, and download the MANO release before running the asset check below.

```bash
git clone https://github.com/wuji-technology/wuji-ego-mint.git
cd wuji-ego-mint
bash scripts/create_env.sh inference
conda activate mint-inference
bash scripts/download_assets.sh

# Copy the MANO files from the downloaded release into the project.
mkdir -p assets/mano/mano_right assets/mano/mano_left
cp /path/to/MANO_RIGHT.pkl assets/mano/mano_right/MANO_RIGHT.pkl
cp /path/to/MANO_LEFT.pkl assets/mano/mano_left/MANO_LEFT.pkl

python -m mint doctor --profile inference
python -m mint viewer
```

### Model and asset locations

Place the models and assets used by Quick Start at the paths below. `scripts/download_assets.sh` downloads the public MINT checkpoint automatically; MANO must be downloaded from its official website and copied manually.

| Model or asset | Download source | Path in this repository | Notes |
| --- | --- | --- | --- |
| MINT checkpoint | [ModelScope](https://www.modelscope.cn/models/AsherZhu/mint_v1) or [Hugging Face](https://huggingface.co/ZZJAsher/mint_v1) | `checkpoints/model.safetensors` | The download script places and verifies it automatically; use the same path for a manual download. |
| MANO left- and right-hand models | [MANO website](https://mano.is.tue.mpg.de/) | `assets/mano/mano_right/MANO_RIGHT.pkl`<br>`assets/mano/mano_left/MANO_LEFT.pkl` | Registration and acceptance of the MANO License are required. |
| LingBot-Map pretrained backbone | [LingBot-Map](https://github.com/robbyant/lingbot-map) | `assets/models/lingbot-map.pt` | Optional; download only when required by the selected configuration. |
| Wuji Hand URDF, MJCF, and STL | Included in this repository | `eval/simulate/wuji-retargeting/wuji_retargeting/wuji-description/hand/body/` | No additional download is required. |

### Using the Viewer

The Viewer automatically opens `http://127.0.0.1:8011` in the default browser. Then:

1. In **Model and Sample**, keep `checkpoints/model.safetensors` or select another compatible checkpoint.
2. Click **Load Model** and wait until the model status is ready.
3. Select a LeRobot episode or an egocentric MP4 video, then choose the camera-inference, hand-window, and geometry settings.
4. Click **Start Inference**.
5. Inspect the synchronized GT/Pred 2D view, fixed-world and camera-frame 3D panels, per-frame values, losses, exports, and optional benchmark tools.

![MINT Web Viewer after loading the model and running inference](data/samples/mint-web-viewer.png)

**Testing your own video** needs no configuration change and no command line: browse to the video in the input-directory picker at the bottom right of the Viewer and click it. Supported formats are `.mp4`, `.mov`, `.avi`, `.mkv` and `.webm`. A plain video carries no ground truth, so the Viewer runs in prediction-only mode — GT panels, the GT/Pred side-by-side layout and the loss readout are hidden, and everything shown is prediction.

All visualization operations live in the Viewer panel — there is no separate command-line visualization step. See [Installation](docs/installation.md) for installation, CUDA, MANO, and offline deployment details.

## 🧠 How MINT works

<img src="assets/readme/figure_teaser.webp" width="100%" alt="Overview: large-scale egocentric video on the left, MINT and dataset diversity in the middle, zero-shot world-space outputs on the right">

<div align="center"><sub>Left: the released supervision — <b>1,021 h</b>, <b>560 K</b> episodes of egocentric video, and how its diversity compares with EgoDex, Ego4D and EPIC-KITCHENS. Right: one unseen video through MINT in a single pass — world-space camera and hand trajectory, MANO hands, and the same motion retargeted to a robot hand.</sub></div>

One shared encode per frame, four factorized heads, then explicit rigid composition into world space — so a world-space error updates the camera branch and the hand branch together.

### Model at a glance

| | |
| --- | --- |
| **Input** | monocular egocentric RGB, 378 × 518 frames, patch 14, 999 tokens per frame |
| **Backbone** | LingBot-Map / GCT — 24 alternating frame-wise / global attention pairs |
| **Clip** | T = 32 frames, re-anchored to the window's own first frame |
| **Parameters** | 1.131 B trainable |
| **Head 01** | camera extrinsics, 7-D `[t, q]`, iterative causal refinement, 4 steps |
| **Head 02** | field of view, `f_h, f_w`, independent temporal branch, Softplus |
| **Head 03** | camera-frame hand MANO, 218-D, left + right, component queries, 2 refinements |
| **Head 04** | per-frame hand presence, logits L / R, patch cross-attention |
| **Composition** | `x_c = R x_w + t`, `p_w = Rᵀ(p_c − t)`, `Q_w = Rᵀ Q_c` — explicit and differentiable |
| **At inference** | no depth map, no point cloud, no dense 3D intermediate |
| **Training** | Stage 1 pretrain on 1,021 h of pipeline supervision → Stage 2 correct the camera trajectory on a small high-precision trajectory set, with the geometric encoder and the hand, presence and field-of-view modules frozen |

<img src="assets/readme/figure_egopipeline.webp" width="100%" alt="EgoPipeline stages: hand detection and frame filtering, camera pose estimation through GeoCalib, MoGe-2 and MegaSaM, HaWoR hand reconstruction, then outlier rejection, interpolation, temporal smoothing and the world-coordinate transform">

<div align="center"><sub><b>EgoPipeline</b>, the offline supervision generator. Top: pre-filtering and the per-stage estimators. Bottom: the post-processing that turns raw stage output into usable structured state. Stage by stage in the table below.</sub></div>

### Where the supervision comes from

**EgoPipeline** is our open-source implementation of the conventional multi-stage route, and it stays in the project as the *supervision generator*, not as the deployment path:

| Stage | Component | Produces |
| --- | --- | --- |
| 00 | [preprocessing and filtering](ego_pipeline/preprocessing/) | normalise fps/resolution/duration; cut out stretches with no hands or more than two; validate metadata, split overlapping clips |
| 01 | GeoCalib | camera intrinsics |
| 02 | MoGe-2 | monocular depth |
| 03 | MegaSaM / DROID-SLAM | metric camera track |
| 04 | HaWoR | camera-frame MANO |
| 05 | cleanup | outlier rejection, gap interpolation, temporal filters |

Its output is **pseudo-label, not ground truth** — that distinction is load-bearing everywhere in this repository.

### Why one model instead of the chain

<img src="assets/readme/teaser.webp" width="100%" alt="Three panels from one forward pass: ego view with reprojected MANO, the world-space camera and two-hand trajectory, and that trajectory retargeted to the Wuji hand in MuJoCo">

A staged chain has structural costs that better engineering does not remove: each stage re-encodes the same video, camera and hands meet only in post-processing, every stage is capped by the one before it, and one operator regressing regresses the whole record.

Inference throughput measured on **RTX 4090D**, at 512 × 384 and 30 fps under identical conditions; time is the **marginal cost per frame in steady state**. Speedups are against **VITRA**, which **EgoPipeline**, this project's data pipeline, was built by optimising. The three methods are points on one line of work: VITRA to start, EgoPipeline as the pipeline-level optimisation, MINT replacing the chain with a single model.

<table>
<thead>
<tr>
  <th align="left">Method</th>
  <th align="right">Time ↓<br><sub>ms/frame</sub></th>
  <th align="right">fps ↑</th>
  <th align="right">Speedup ↑<br><sub>vs VITRA</sub></th>
</tr>
</thead>
<tbody>
<tr><th colspan="4" align="left">One GPU</th></tr>
<tr><td align="left">VITRA</td><td align="right">1260.0</td><td align="right">0.8</td><td align="right">—</td></tr>
<tr><td align="left">MINT</td><td align="right"><b>72.4</b></td><td align="right"><b>13.8</b></td><td align="right"><b>17.4×</b></td></tr>
<tr><th colspan="4" align="left">Four GPUs</th></tr>
<tr><td align="left">EgoPipeline</td><td align="right">83.4</td><td align="right">12.0</td><td align="right">3.4×</td></tr>
<tr><td align="left">MINT</td><td align="right"><b>22.7</b></td><td align="right"><b>44.1</b></td><td align="right"><b>12.5×</b></td></tr>
</tbody>
</table>

MINT is **17.4×** faster on one GPU (1260.0 → 72.4 ms/frame) and **12.5×** on four (283.3 → 22.7 ms/frame). The four-GPU block shows where that comes from: optimising the pipeline accounts for 3.4× (283.3 → 83.4), and the unified model takes it to 22.7. Only MINT clears 30 fps real time, at 44.1 fps.

EgoPipeline is itself distributed-optimised (Ray multi-GPU operators, persistent workers, asynchronous CPU stages): rewriting its serial single-GPU execution as a scheduled 4-GPU worker pool took 179.0 s down to 63.9 s per 270 frames, **2.8×** internal to the pipeline. That is whole-clip wall-clock including decode and write-out, a different accounting from the steady-state cost above; the two do not multiply.

## 📊 What MINT is measured on

### Camera-frame bimanual reconstruction

**MINT is zero-shot on both benchmarks** — it never saw either during training. ViDiHand, marked `*`, did train on most of both, so it is listed **for reference only** and is left out when deciding which value is best; bold marks the best of the remaining rows. `MINT + UKF` is the same model with the inference-time filter switched on, nothing else changed.

<table>
<thead>
<tr>
  <th rowspan="2" align="left">Method</th>
  <th colspan="3" align="center">Detection</th>
  <th colspan="2" align="center">3D pose</th>
  <th colspan="2" align="center">Orient. &amp; pos.</th>
  <th colspan="1" align="center">Temporal</th>
</tr>
<tr>
  <th align="right">FAcc ↑</th>
  <th align="right">Recall ↑</th>
  <th align="right">F1 ↑</th>
  <th align="right">MPJPE-p ↓<br><sub>mm</sub></th>
  <th align="right">PA-MPJPE-p ↓<br><sub>mm</sub></th>
  <th align="right">GO-p ↓<br><sub>deg</sub></th>
  <th align="right">CT-p ↓<br><sub>m</sub></th>
  <th align="right">Jitter ↓<br><sub>mm/frame²</sub></th>
</tr>
</thead>
<tbody>
<tr><th colspan="9" align="left">ARCTIC</th></tr>
<tr><td align="left">InterWild</td><td align="right">0.878</td><td align="right">0.943</td><td align="right">0.959</td><td align="right">30.82</td><td align="right">15.95</td><td align="right">25.39</td><td align="right">0.097</td><td align="right">46.58</td></tr>
<tr><td align="left">HaMeR</td><td align="right">0.875</td><td align="right">0.943</td><td align="right">0.957</td><td align="right">29.20</td><td align="right">14.60</td><td align="right">24.91</td><td align="right">0.095</td><td align="right">18.28</td></tr>
<tr><td align="left">Hamba</td><td align="right">0.833</td><td align="right">0.912</td><td align="right">0.941</td><td align="right">31.23</td><td align="right">17.17</td><td align="right">27.82</td><td align="right">0.110</td><td align="right">15.36</td></tr>
<tr><td align="left">WildHands</td><td align="right">0.879</td><td align="right">0.946</td><td align="right">0.960</td><td align="right">25.70</td><td align="right">13.94</td><td align="right">22.32</td><td align="right"><b>0.058</b></td><td align="right">12.97</td></tr>
<tr><td align="left">OmniHands</td><td align="right">0.866</td><td align="right">0.949</td><td align="right">0.954</td><td align="right">29.67</td><td align="right">14.20</td><td align="right">24.58</td><td align="right">0.087</td><td align="right">45.31</td></tr>
<tr><td align="left">WiLoR</td><td align="right"><b>0.919</b></td><td align="right">0.951</td><td align="right">0.974</td><td align="right"><b>22.01</b></td><td align="right"><b>11.87</b></td><td align="right"><b>17.36</b></td><td align="right">0.075</td><td align="right">24.09</td></tr>
<tr><td align="left">Dyn-HaMR</td><td align="right">0.842</td><td align="right">0.918</td><td align="right">0.951</td><td align="right">27.90</td><td align="right">17.02</td><td align="right">25.95</td><td align="right">0.121</td><td align="right">12.84</td></tr>
<tr><td align="left">HaWoR</td><td align="right">0.700</td><td align="right">0.817</td><td align="right">0.895</td><td align="right">45.36</td><td align="right">26.38</td><td align="right">43.33</td><td align="right">0.149</td><td align="right">19.79</td></tr>
<tr><td align="left"><i>ViDiHand*</i></td><td align="right"><i>0.997</i></td><td align="right"><i>0.999</i></td><td align="right"><i>0.999</i></td><td align="right"><i>21.67</i></td><td align="right"><i>9.82</i></td><td align="right"><i>14.64</i></td><td align="right"><i>0.047</i></td><td align="right"><i>3.18</i></td></tr>
<tr><td align="left">MINT</td><td align="right">0.916</td><td align="right"><b>0.957</b></td><td align="right"><b>0.978</b></td><td align="right">51.03</td><td align="right">27.71</td><td align="right">24.19</td><td align="right">0.140</td><td align="right">12.26</td></tr>
<tr><td align="left">MINT + UKF</td><td align="right">0.916</td><td align="right"><b>0.957</b></td><td align="right"><b>0.978</b></td><td align="right">51.09</td><td align="right">27.70</td><td align="right">24.22</td><td align="right">0.140</td><td align="right"><b>2.54</b></td></tr>
<tr><th colspan="9" align="left">HOT3D</th></tr>
<tr><td align="left">InterWild</td><td align="right">0.669</td><td align="right">0.881</td><td align="right">0.868</td><td align="right">77.17</td><td align="right">24.81</td><td align="right">58.50</td><td align="right">0.213</td><td align="right">101.16</td></tr>
<tr><td align="left">HaMeR</td><td align="right">0.692</td><td align="right">0.904</td><td align="right">0.883</td><td align="right">68.31</td><td align="right">21.46</td><td align="right">49.64</td><td align="right">0.102</td><td align="right">23.63</td></tr>
<tr><td align="left">Hamba</td><td align="right">0.632</td><td align="right">0.828</td><td align="right">0.853</td><td align="right">71.73</td><td align="right">29.62</td><td align="right">56.53</td><td align="right">0.128</td><td align="right">18.51</td></tr>
<tr><td align="left">WildHands</td><td align="right">0.655</td><td align="right">0.863</td><td align="right">0.844</td><td align="right">52.79</td><td align="right">28.95</td><td align="right">53.93</td><td align="right">0.157</td><td align="right">22.89</td></tr>
<tr><td align="left">OmniHands</td><td align="right">0.649</td><td align="right">0.895</td><td align="right">0.868</td><td align="right">63.28</td><td align="right">22.68</td><td align="right">49.12</td><td align="right">0.133</td><td align="right">69.51</td></tr>
<tr><td align="left">WiLoR</td><td align="right">0.827</td><td align="right">0.897</td><td align="right">0.937</td><td align="right">30.97</td><td align="right">19.98</td><td align="right">25.75</td><td align="right">0.098</td><td align="right">17.98</td></tr>
<tr><td align="left">Dyn-HaMR</td><td align="right">0.614</td><td align="right">0.811</td><td align="right">0.802</td><td align="right">74.21</td><td align="right">38.20</td><td align="right">43.85</td><td align="right">0.571</td><td align="right">44.94</td></tr>
<tr><td align="left">HaWoR</td><td align="right">0.348</td><td align="right">0.499</td><td align="right">0.654</td><td align="right">71.40</td><td align="right">66.03</td><td align="right">79.35</td><td align="right">0.262</td><td align="right">23.87</td></tr>
<tr><td align="left"><i>ViDiHand*</i></td><td align="right"><i>0.948</i></td><td align="right"><i>0.974</i></td><td align="right"><i>0.983</i></td><td align="right"><i>21.51</i></td><td align="right"><i>11.38</i></td><td align="right"><i>15.83</i></td><td align="right"><i>0.040</i></td><td align="right"><i>3.74</i></td></tr>
<tr><td align="left">MINT</td><td align="right"><b>0.940</b></td><td align="right"><b>0.977</b></td><td align="right"><b>0.950</b></td><td align="right"><b>23.61</b></td><td align="right">10.70</td><td align="right">16.78</td><td align="right"><b>0.073</b></td><td align="right">11.52</td></tr>
<tr><td align="left">MINT + UKF</td><td align="right"><b>0.940</b></td><td align="right"><b>0.977</b></td><td align="right"><b>0.950</b></td><td align="right">23.62</td><td align="right"><b>10.69</b></td><td align="right"><b>16.77</b></td><td align="right"><b>0.073</b></td><td align="right"><b>2.39</b></td></tr>
</tbody>
</table>

On **HOT3D**, setting aside the reference row ViDiHand, MINT has the best numbers in the table on all three detection metrics and on MPJPE-p, PA-MPJPE-p, GO-p and CT-p, and `MINT + UKF` has the lowest jitter of any row. On **ARCTIC**, its camera-frame joint error is clearly behind methods that do nothing but hand reconstruction — because this checkpoint's hand branch trained only on the coarse labels EgoPipeline produces and was never fine-tuned on high-precision hand data, so its hand accuracy tracks the quality of those labels. We report both benchmarks rather than only the favourable one.

**The row that actually tests the claim is HaWoR.** HaWoR is upstream third-party work, and EgoPipeline calls it at stage 04 to produce camera-frame MANO — so it generated every hand label this checkpoint was trained on. It is the teacher here. A student trained on pseudo-labels cannot be expected to beat the process that generated them; matching it is the result being asked for. MINT matches and passes it: on ARCTIC, better global orientation (24.19 vs 43.33 deg) and translation (0.140 vs 0.149 m) with comparable Procrustes-aligned joint error (27.71 vs 26.38 mm); on HOT3D, better on every one of the eight metrics, by wide margins (MPJPE-p 23.61 vs 71.40 mm, F1 0.950 vs 0.654). One unified model, in one forward pass, reproduces the structured output of the cascade it was distilled from.

That is what makes the remaining gap a **data problem rather than an architectural one**. The ceiling on camera-frame hand accuracy is the pseudo-label quality, not the model: the same architecture fine-tuned on high-precision camera-frame hand data should produce a MINT that emits high-precision labels, and that fine-tune is the next step, not a redesign.

### World-frame camera trajectory

Table 2 of the paper: HOT3D (27 sequences, 94,978 frames) and the ARCTIC P2 validation split (34 sequences, 25,883 frames). Sequences are evaluated at **full length, with SE(3)-only alignment and no fitted scale** — so scale error is charged rather than absorbed, and the arc-length ratio is what exposes it. Coverage is how many sequences a method completed: `MegaSaM†` runs without depth refinement and runs out of memory on the three longest HOT3D sequences. `MINT w/o stage 2` never sees metric ground truth. Bold marks the best value in each column; for the arc-length ratio, best means closest to 1.

<table>
<thead>
<tr>
  <th rowspan="2" align="left">Method</th>
  <th rowspan="2" align="right">Cov.<br><sub>seq.</sub></th>
  <th colspan="2" align="center">ATE ↓ (mm)</th>
  <th colspan="2" align="center">RPE-T ↓ (mm)</th>
  <th colspan="2" align="center">RPE-R ↓ (deg)</th>
  <th rowspan="2" align="right">Arc len.<br><sub>ratio → 1</sub></th>
  <th rowspan="2" align="right">ATE ↓<br><sub>%</sub></th>
</tr>
<tr>
  <th align="right">mean</th>
  <th align="right">med.</th>
  <th align="right">mean</th>
  <th align="right">med.</th>
  <th align="right">mean</th>
  <th align="right">med.</th>
</tr>
</thead>
<tbody>
<tr><th colspan="10" align="left">HOT3D</th></tr>
<tr><td align="left">DROID-SLAM</td><td align="right">27/27</td><td align="right"><b>49.1</b></td><td align="right"><b>39.6</b></td><td align="right">5.36</td><td align="right">3.52</td><td align="right">0.227</td><td align="right">0.146</td><td align="right">0.778</td><td align="right"><b>0.36</b></td></tr>
<tr><td align="left">HaWoR</td><td align="right">27/27</td><td align="right">200.3</td><td align="right">179.2</td><td align="right">8.60</td><td align="right">7.33</td><td align="right">1.098</td><td align="right">0.928</td><td align="right"><b>0.950</b></td><td align="right">1.28</td></tr>
<tr><td align="left">InfiniteVGGT</td><td align="right">27/27</td><td align="right">124.8</td><td align="right">100.0</td><td align="right">13.52</td><td align="right">9.67</td><td align="right">1.492</td><td align="right">0.392</td><td align="right">0.556</td><td align="right">0.80</td></tr>
<tr><td align="left">LingBot-Map</td><td align="right">27/27</td><td align="right">85.5</td><td align="right">51.0</td><td align="right">7.56</td><td align="right">6.18</td><td align="right">0.684</td><td align="right">0.253</td><td align="right">0.712</td><td align="right">0.55</td></tr>
<tr><td align="left">MegaSaM†</td><td align="right">24/27</td><td align="right">94.4</td><td align="right">65.3</td><td align="right"><b>3.18</b></td><td align="right"><b>2.13</b></td><td align="right"><b>0.082</b></td><td align="right"><b>0.063</b></td><td align="right">0.716</td><td align="right">0.69</td></tr>
<tr><td align="left">MINT w/o stage 2</td><td align="right">27/27</td><td align="right">524.7</td><td align="right">434.6</td><td align="right">8.75</td><td align="right">8.37</td><td align="right">0.234</td><td align="right">0.229</td><td align="right">0.466</td><td align="right">3.29</td></tr>
<tr><td align="left">MINT</td><td align="right">27/27</td><td align="right">181.7</td><td align="right">155.5</td><td align="right">4.69</td><td align="right">4.78</td><td align="right">0.284</td><td align="right">0.259</td><td align="right">1.094</td><td align="right">1.15</td></tr>
<tr><th colspan="10" align="left">ARCTIC</th></tr>
<tr><td align="left">DROID-SLAM</td><td align="right">34/34</td><td align="right">181.5</td><td align="right">49.6</td><td align="right">33.84</td><td align="right">14.25</td><td align="right">1.006</td><td align="right">0.423</td><td align="right"><b>0.964</b></td><td align="right">8.07</td></tr>
<tr><td align="left">HaWoR</td><td align="right">34/34</td><td align="right">66.2</td><td align="right"><b>29.4</b></td><td align="right">24.08</td><td align="right">4.16</td><td align="right">0.298</td><td align="right"><b>0.151</b></td><td align="right">0.759</td><td align="right">2.87</td></tr>
<tr><td align="left">InfiniteVGGT</td><td align="right">34/34</td><td align="right">79.0</td><td align="right">69.1</td><td align="right">16.21</td><td align="right">12.63</td><td align="right">1.265</td><td align="right">0.655</td><td align="right">0.284</td><td align="right">3.23</td></tr>
<tr><td align="left">LingBot-Map</td><td align="right">34/34</td><td align="right">59.6</td><td align="right">62.0</td><td align="right">9.17</td><td align="right">8.47</td><td align="right">0.980</td><td align="right">0.717</td><td align="right">0.591</td><td align="right">2.46</td></tr>
<tr><td align="left">MegaSaM†</td><td align="right">34/34</td><td align="right"><b>51.4</b></td><td align="right">50.4</td><td align="right">8.73</td><td align="right">5.58</td><td align="right">0.779</td><td align="right">0.725</td><td align="right">1.956</td><td align="right"><b>2.15</b></td></tr>
<tr><td align="left">MINT w/o stage 2</td><td align="right">34/34</td><td align="right">63.7</td><td align="right">59.3</td><td align="right">3.53</td><td align="right">3.62</td><td align="right"><b>0.251</b></td><td align="right">0.243</td><td align="right">0.755</td><td align="right">2.63</td></tr>
<tr><td align="left">MINT</td><td align="right">34/34</td><td align="right">81.9</td><td align="right">83.2</td><td align="right"><b>3.39</b></td><td align="right"><b>3.37</b></td><td align="right">0.256</td><td align="right">0.251</td><td align="right">1.412</td><td align="right">3.40</td></tr>
</tbody>
</table>

**RPE-T and RPE-R are what to read here.** MINT produces training data, and training uses per-frame camera deltas rather than absolute pose in a global frame. Relative pose error is what reaches whatever is trained on the output; ATE measures drift accumulated over a whole sequence, which a delta-based trainer never sees.

On **ARCTIC**, MINT has the lowest RPE-T in the table (3.39 mm mean, against 8.73 mm for the next-best external method), and its RPE-R mean of 0.256 also beats every external method (HaWoR is next at 0.298). On **HOT3D**, RPE-T is second at 4.69 mm behind MegaSaM's 3.18 mm, and MegaSaM finishes 24 of 27 sequences; RPE-R there trails MegaSaM and DROID-SLAM.

**ATE is not where this checkpoint wins**: 181.7 mm on HOT3D against DROID-SLAM's 49.1 mm. The arc-length ratio says why — 1.094 on HOT3D and 1.412 on ARCTIC mean the predicted path is longer than the real one, the scale enlargement noted in [Known limits](#️-known-limits). Stage 2 is what corrects it: without Stage 2 the HOT3D ratio is only 0.466, far too short; Stage 2 brings it to 1.094, and overshoots on ARCTIC.

### Protocol

- **Held out completely.** HOT3D and ARCTIC images, pseudo-labels and ground truth took no part in training, high-precision calibration, hyper-parameter or loss-weight selection, or checkpoint selection, and there is no test-time tuning. Only under that condition is a result zero-shot.
- **Split by video.** Official splits are made by original video ID before clipping; where participant IDs exist, we separate by participant as well.
- **Every method gets the same input.** Inputs, evaluation preprocessing and the evaluation manifest are identical, and sequence-level failures count the same way for everyone. A method that cannot output both camera and hands gets N/A; we never substitute another method's output.
- **The one exception is labelled.** Where a compared method reports results from its own training on HOT3D/ARCTIC rather than zero-shot, we say so rather than averaging it in.

**Benchmark integrity statement.** We commit that every metric reported by this project is an authentic result produced under the stated evaluation protocol; we do not alter the original numeric results. To reproduce a baseline or another method's exact values, use that method's official repository and environment. The metric definitions, alignment rules, aggregation logic, and reporting code used by MINT are available in `eval/model_effect/benchmark/` for inspection.

The complete implementation and tests live in `eval/model_effect/benchmark/` and are integrated into the Viewer's Benchmark panel. Before use, download HOT3D and ARCTIC, organize the benchmark data as required by each adapter, and install the optional runtime. The CLI remains available:

```bash
python eval/model_effect/benchmark/run.py \
  --ckpt /path/to/checkpoint \
  --config configs/training/mint_step2.yaml \
  --data-root /path/to/benchmark-data
```

Set `CAMERA_TRAJECTORY_ROOT` for camera-trajectory exports when needed. Aliyun defaults are placeholders; users must configure the workspace, resource, image, CPFS, credentials, and environment. This project does not provision or maintain benchmark environments.

## 📦 What is included

| Area | Entry point | Purpose |
| --- | --- | --- |
| Infer and view | `python -m mint viewer` | Web UI for LeRobot GT, predictions, 2D/3D trajectories, and frame metrics. |
| Train | `python -m mint train` | Train the camera-and-hand MINT model with Accelerate/DDP. |
| Benchmark | `python eval/model_effect/benchmark/run.py` | Open benchmark CLI with user-provided data and environment. |
| Audit | `python -m mint doctor` | Verify the environment, optional backends, assets, and GPU runtime. |
| Pipeline reference | `ego_pipeline/` | Reuse the open Ray orchestration, interfaces, cleaning, and LeRobot export code after integrating the required upstream backends locally. |

## 🏋️ Training and optional pipeline reconstruction

Training or local pipeline-development work requires the full environment:

```bash
bash scripts/create_env.sh full
conda activate mint
python -m mint doctor --profile full
```

The public release uses the Viewer as the unified entry point for MINT inference, visualization, and artifact export. This project publishes only code that third-party licenses permit us to distribute; license-restricted adaptations and internal integrations are excluded, so **this repository does not contain the complete production data pipeline.**

GeoCalib, MoGe, and Mega-SAM source snapshots are distributed under `third_party/`, but some production adaptations cannot be published under their upstream license terms. In particular, the locally adapted HaWoR source is Git-ignored because CC BY-NC-ND prohibits redistribution of modifications. Weights, MANO files, and other separately licensed assets are also excluded.

If you need to reconstruct the data-generation pipeline, first read [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), obtain and install every required upstream library and asset under its own terms, and implement the necessary compatibility adapters locally. The open code in `ego_pipeline/` documents the orchestration, interfaces, data flow, cleaning, manifests, and LeRobot export contracts. An AI coding assistant may help reconcile upstream API differences, but the resulting integration remains the user's responsibility and must comply with all licenses. Only after completing that local integration should the data-profile doctor and `python -m mint pipeline` be treated as usable entry points.

The public release provides an implementation reference, not a one-command reproduction of the production data generator. Use MINT directly for inference; for pipeline reconstruction, supply the licensed upstream code and assets and complete the local integration described in [Data pipeline](docs/data-pipeline.md).

If you do go down this road, the least painful order is: install each upstream project separately by its own instructions and get it running on its own first, then wire them together against the source in `ego_pipeline/`. That wiring step suits an AI coding assistant well — point it at the upstream APIs and at this repository's calling conventions and have it write the compatibility layer. The fixed in-repository paths for each asset are:

| Data-pipeline asset | Path in this repository |
| --- | --- |
| GeoCalib weights | `model/geocalib/pinhole.tar` |
| MoGe weights | `model/moge2/model.pt` |
| Mega-SAM weights | `model/megasam/megasam_final.pth` |
| HaWoR weights | `model/hawor/hawor.ckpt` |
| HaWoR configuration | `model/hawor/model_config.yaml` |
| HaWoR detector | `model/hawor/detector.pt` |
| DROID-SLAM weights | `third_party/HaWoR/weights/external/droid.pth` |
| Metric3D weights | `third_party/HaWoR/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth` |
| HaWoR right-hand MANO | `third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl` |
| HaWoR left-hand MANO | `third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl` |

### Train

Only the two configurations associated with the selected checkpoints are kept. `step_00019000` is Stage 1; `step_00004500` is the Stage 2 WorldEngine camera-only adaptation initialized from Stage 1 and is the final fine-tuned checkpoint released publicly:

```bash
python -m mint train --config configs/training/mint_step1.yaml

python -m mint train --config configs/training/mint_step2.yaml
```

The Stage 2 `train.init_from` points to the Stage 1 `step_00019000/model.safetensors`.

`mint train` consumes a compatible, separately prepared LeRobot dataset and writes training checkpoints. After training, select the new checkpoint directly in the Viewer panel for interactive inspection.

## 🗂️ Public Ego pretraining data

We provide the non-video portions of the `ego4d`, `egodex`, and `epickitchen` data processed by this repository's Ego data-production pipeline:

- [Hugging Face: ZZJAsher/wuji_ego_mint](https://huggingface.co/datasets/ZZJAsher/wuji_ego_mint)
- [ModelScope: AsherZhu/wuji_ego_mint](https://www.modelscope.cn/datasets/AsherZhu/wuji_ego_mint)

| | |
| --- | --- |
| Structured supervision | camera trajectory, two-hand MANO, per-frame presence, world-space motion |
| Kept after filtering and trajectory cleaning | **1,021.514 h** · 560,649 episodes · 110,323,558 frames |
| Sources | Ego4D 332.874 h · EPIC-KITCHENS-100 55.194 h · EgoDex 633.446 h |
| Yield | 59.1 % of 1,729 raw hours |

That figure is the whole chain accumulated, not any single rule: normalisation, [cutting stretches with no hands or more than two](ego_pipeline/preprocessing/), dropping segments that end up too short, per-stage failures and timeouts, and outlier rejection during trajectory cleaning all reduce what survives. The final step — packing the cleaned results into the training LeRobot dataset — runs in an internal repository and is not part of this release, so only the overall figure is given here; it cannot be broken down per rule.

> **Important: the camera trajectories in this data were generated by this repository's Ego data pipeline and currently exhibit obvious scale enlargement. We recommend using the data only to pretrain this project's Ego hand reconstruction model. Do not use it for real-scale evaluation, precise camera trajectory evaluation, or as real-scale ground truth.**

The public dataset does not include `.mp4` videos. The source videos belong to Ego4D, EgoDex, and EPIC-KITCHENS and are governed by their respective dataset licenses and access terms, so this project cannot redistribute them. Users who need the videos must apply for and download them through the corresponding official dataset channels and verify their own usage and redistribution rights.

## 🔐 LeRobot sample and privacy

`data/samples/lerobot_v3/` is the only bundled sample: eight contrasting centered 15-second clips selected from the sorted Hot3D sequence exports, forming a valid **8-episode, 3,600-frame** LeRobot v3 dataset — 512 × 512 at 30 fps, 21 MB on disk — with synchronized H.264 video, Hot3D camera and two-hand labels, task text, and episode metadata. Participant IDs and original sequence names are not stored; only anonymous source collection indices and centered frame ranges, in `lerobot_v3/sample_manifest.json`.

Rebuild it from local full exports with:

```bash
python scripts/build_sample_lerobot.py \
  --source-root /path/to/hot3d_to_lerobot \
  --output data/samples/lerobot_v3
```

Dataset access does not itself grant redistribution rights. The publisher must still confirm licensing, participant consent, and frame-by-frame privacy review. See [Privacy](docs/privacy.md) for details.

## ⚠️ Known limits

We would rather you read these here than discover them later:

- **Camera-frame hand accuracy is limited by the quality of the labels it was trained on.** The hand branch of the released checkpoint is supervised entirely by the coarse labels EgoPipeline produces automatically. Stage 2 corrects only the camera trajectory — the hand, presence and field-of-view modules are frozen for that step — so high-precision hand data never fine-tunes them. Its hand accuracy therefore sits roughly at the level of the labels it learned from: on HOT3D it is the best row in the table (setting aside ViDiHand, which trained on both benchmarks and is listed only for reference), but on the close-range two-handed manipulation in ARCTIC it is clearly behind methods that do nothing but hand reconstruction ([Table 1](#camera-frame-bimanual-reconstruction)). Improving this needs a fine-tune of the hand branch on high-precision hand data, not a change to the model; that fine-tune is not in this release.
- **The released camera trajectories are scale-enlarged.** The shape of a trajectory is usable but its length runs long, so use them for pretraining rather than for metric evaluation or as real-scale ground truth. The cause is described under [Public Ego pretraining data](#️-public-ego-pretraining-data).
- **Training happens on 32-frame clips, which by itself proves nothing about long videos.** Long-sequence behaviour is reported separately; we do not extrapolate it from clip-level accuracy.
- **Whether a hand is in frame comes from a detector with a fixed threshold.** That label therefore carries the systematic bias of that threshold. MINT sees the whole clip, so it can correct the detector's occasional misses and false positives — but the claim that it does correct them is checked against hand-labelled, blind-reviewed presence, not against the labels it trained on.
- **This repository is not the complete production data pipeline.** The upstream code and weights we modified — adapted HaWoR, for instance — are deliberately left out because their licenses do not permit redistribution. To run the whole pipeline yourself you download those under each upstream project's own terms; see [Training and optional pipeline reconstruction](#️-training-and-optional-pipeline-reconstruction).

## 🧾 Repository layout

```text
mint/
|-- configs/          Two-stage training recipes and inference settings
|-- data/samples/     Approved Hot3D LeRobot v3 sample
|-- eval/model_effect Original visualization, inference adapters, and benchmarks
|-- docs/             Architecture and operational guides
|-- environments/     Full and inference-only dependency specifications
|-- mint/             CLI, inference engine, renderer, and web viewer
|-- model_train/      Training engine, model, losses, and LeRobot loader
|-- ego_pipeline/     Ray scheduling, actors, model backends, trajectory cleanup, manifests, and export
|   `-- preprocessing/  video normalisation and hand filtering, ahead of the pipeline stages
|-- scripts/          Reproducible setup, asset, privacy, and sample tools
`-- third_party/      Redistributable source snapshot; adapted HaWoR is local-only, assets excluded
```

## 📚 Documentation

| | |
| --- | --- |
| [Architecture](docs/architecture.md) | how the model and the repository fit together |
| [Installation](docs/installation.md) | profiles, CUDA, MANO, offline deployment |
| [Data pipeline](docs/data-pipeline.md) | EgoPipeline stages and local reconstruction |
| [Training](docs/training.md) | the two-stage recipe |
| [LeRobot training data format](docs/lerobot-training-data.md) | what a compatible dataset must contain |
| [Inference and viewer](docs/inference.md) | Viewer panels, exports, benchmark tools |
| [Privacy and release checklist](docs/privacy.md) | consent, review, redistribution |
| [Security policy](SECURITY.md) | how to report a vulnerability |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | upstream licenses — read before distributing |

## ✨ Acknowledgements

MINT is made possible by the following research projects, models, and datasets.

- **[VITRA](https://microsoft.github.io/VITRA/)** — MINT's data-processing architecture, egocentric reconstruction workflow, world-space camera/hand annotations, and LeRobot conversion conventions evolved from the VITRA and VITRA-1M data engine.
- **[LingBot-Map](https://github.com/robbyant/lingbot-map)** — provides the core model architecture and the upstream source adapted for MINT camera-and-hand training and inference.
- **[HaWoR](https://github.com/ThunderVVV/HaWoR)** — provides monocular hand motion reconstruction, MANO estimation, tracking, and world-space hand-processing components used by the optional data pipeline. Its use remains subject to the upstream non-commercial, no-derivatives license.
- **Camera, depth, and tracking research** — [GeoCalib](https://github.com/cvg/GeoCalib), [MoGe](https://github.com/microsoft/MoGe), [Mega-SAM](https://github.com/mega-sam/mega-sam), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), [UniDepth](https://github.com/lpiccinelli-eth/UniDepth), [Metric3D](https://github.com/YvanYin/Metric3D), [DeepCalib](https://github.com/alexvbogdan/DeepCalib), [DINOv2](https://github.com/facebookresearch/dinov2), [VGGT](https://github.com/facebookresearch/vggt), InfiniteVGGT, and [PyTorch3D](https://github.com/facebookresearch/pytorch3d).
- **Hand models, simulation, and retargeting** — [MANO](https://mano.is.tue.mpg.de), [SMPL-X](https://smpl-x.is.tue.mpg.de), [MuJoCo](https://mujoco.org), and the Wuji hand description and retargeting components used by the optional Viewer panels.
- **Datasets and benchmarks** — [HOT3D](https://github.com/facebookresearch/hot3d), [ARCTIC](https://arctic.is.tue.mpg.de), [Ego4D](https://ego4d-data.org), [EPIC-KITCHENS](https://epic-kitchens.github.io), and [EgoDex](https://github.com/apple/ml-egodex). Dataset access and redistribution remain governed by each dataset's own terms.

We thank all upstream authors and maintainers. This acknowledgement does not replace their citation or license requirements; see [Third-party notices](THIRD_PARTY_NOTICES.md) before use or distribution.

## 📜 License

wuji-ego-mint's original code is released under the MIT License. Upstream models, datasets, MANO assets, vendored LingBot-Map files, and optional research backends retain their own licenses. Review [Third-party notices](THIRD_PARTY_NOTICES.md) before distribution.

## 📖 Citation

The paper is under review; this entry will be replaced with the published reference.

```bibtex
@misc{zhu2026mint,
  title  = {MINT: Minting a Unified Model for World-Space Camera and Hand Motion
            from Scalable Egocentric Pipeline Supervision},
  author = {Zhu, Zijie and Cai, Weiren and Wang, Yizhou and Yang, Zhenjie and
            Liu, Yide and Chen, Jiahao and He, Guanqi},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## 📮 Contact

We are glad to share what we have learned and to help move egocentric data forward. If you run into anything — installation, data generation, model training, inference, or the Viewer — please contact Zijie Zhu. WeChat: `z3132544408`; email: `3132544408@qq.com`.
