# LeRobot v3 training data contract

This document defines only the LeRobot fields consumed by the two retained MINT
training configurations. It is not a general LeRobot v3 specification. The
contract covers the Ego4D, EgoDex, and EPIC-KITCHENS pretraining data and the
camera-trajectory data required by Stage 2.

The authoritative reader is `model_train/data/lingbotmap/lerobot_v3.py`. Camera
translation statistics are read separately by
`model_train/data/camera_normalization.py` before the dataset is constructed.

## Datasets used by the retained recipes

The datasets use LeRobot `v3.0`, 30 FPS H.264 ego video, the OpenCV camera
coordinate convention, and `hand_frame: camera`.

| Stage | Example dataset root | Stored video shape `[H,W,C]` | Supervision consumed |
| --- | --- | --- | --- |
| Stage 1 | `data/lerobot/ego4d/lerobot_v3` | `[768,1024,3]` | Camera, FoV, per-side presence, and camera-frame MANO |
| Stage 1 | `data/lerobot/egodex/lerobot_v3` | `[384,512,3]` | Camera, FoV, per-side presence, and camera-frame MANO |
| Stage 1 | `data/lerobot/epickitchen/lerobot_v3` | `[384,512,3]` | Camera, FoV, per-side presence, and camera-frame MANO |
| Stage 2 | `data/lerobot/stage2/lerobot_v3` (placeholder) | `[384,512,3]` | Camera translation and rotation only |

Stage 2 files contain additional FoV, MANO, presence, and principal-point data,
but the retained Stage 2 loss trains only the camera head. Its configuration
sets `require_mano_gt: false`, freezes the FoV and hand modules, and does not
consume the stored MANO values as loss targets.

## Required directory layout

```text
<root>/
|-- meta/
|   |-- info.json
|   `-- episodes/
|       `-- chunk-NNN/file-NNN.parquet
|-- data/
|   `-- chunk-NNN/file-NNN.parquet
`-- videos/
    `-- observation.images.ego/
        `-- chunk-NNN/file-NNN.mp4
```

`meta/tasks.parquet` may be present as part of a normal LeRobot export, but the
current training loader does not read it.

## `meta/info.json`

Only the following values are part of the current training contract.

| JSON field | Type | Definition |
| --- | --- | --- |
| `fps` | number | Video and label sampling rate. The retained datasets use `30.0`. It converts an episode video timestamp to a frame index. |
| `features.observation.images.ego.shape` | `[H,W,3]` | Stored ego-video frame shape before training resize. The loader requires this feature entry. |
| `features.observation.images.ego.dtype` | `"video"` | Declares `observation.images.ego` as the video stream. |
| `hand_frame` | `"camera"` | Semantic contract for Stage 1 MANO root translation and orientation. Older Stage 1 exports do not contain `hand_supervision`, so the training config and actual columns remain authoritative. |

Other totals, split descriptions, task counts, and chunk sizes are useful
LeRobot metadata but are not read by the current training loader.

## Episode metadata Parquet

Each row under `meta/episodes/**/*.parquet` describes one episode and maps it to
one data Parquet file and one MP4 file. The loader reads exactly these columns:

| Column | Parquet type | Definition |
| --- | --- | --- |
| `episode_index` | `int64` | Globally unique episode identifier within this root. |
| `length` | `int64` | Episode length in frames. Episodes shorter than `clip_len=32` do not produce training samples. |
| `data/chunk_index` | `int64` | Chunk number of the data Parquet containing this episode. |
| `data/file_index` | `int64` | File number of the data Parquet containing this episode. |
| `videos/observation.images.ego/chunk_index` | `int64` | Chunk number of the MP4 containing the ego frames. |
| `videos/observation.images.ego/file_index` | `int64` | File number of the MP4 containing the ego frames. |
| `videos/observation.images.ego/from_timestamp` | `float64` | Episode start time in seconds inside the referenced MP4. The first video frame is `round(from_timestamp * fps)`. |
| `dataset_from_index` | `int64` | Inclusive global row index of the episode's first frame. The loader subtracts the first global index stored in the referenced data file to obtain the local Parquet row. |

Fields such as `tasks`, `atomic_segments`, `dataset_to_index`, source provenance,
and `to_timestamp` are not consumed by the current training reader.

## Per-frame data Parquet

### Fields required by both stages

| Column | Parquet type and shape | Definition and use |
| --- | --- | --- |
| `episode_index` | `int64` | Episode identifier for each frame. Used by the camera-normalization scan to split rows into episodes. |
| `frame_index` | `int64` | Zero-based frame number within the episode. It must start at zero and increase by one. Used by the camera-normalization scan. |
| `state_mask` | `fixed_size_list<bool>[2]` | Legacy `[left,right]` state-availability flag. The generic loader requires it and returns the value at the first frame of each clip, but the retained Stage 1 and Stage 2 losses use `hand_kept` instead. |
| `hand_kept` | `fixed_size_list<bool>[2]` | Per-frame `[left_valid,right_valid]`. It is the binary target for the hand-presence head and the mask for every MANO loss. `false` means that side must not contribute to a hand loss for that frame. |
| `cam_trans` | `fixed_size_list<float32>[3]` | Translation `t` in the OpenCV world-to-camera transform `[R|t]`, in meters. The training loader rebases it to the first frame of every clip. |
| `cam_quat` | `fixed_size_list<float32>[4]` | World-to-camera rotation quaternion in `[x,y,z,w]` order, with the real component last. It represents the same `R` as the OpenCV extrinsic and is rebased per clip. |
| `cam_fov` | `fixed_size_list<float32>[2]` | `[vertical_fov, horizontal_fov]` in radians. For original image size `(H,W)`, `fov_h = 2*atan((H/2)/fy)` and `fov_w = 2*atan((W/2)/fx)`. |

The OpenCV camera axes are `+x` right, `+y` down, and `+z` forward. Camera
quaternions must be finite, approximately unit length, and continuous enough for
relative-rotation and velocity losses.

### Additional Stage 1 MANO fields

Stage 1 sets `require_mano_gt: true`, so all eight columns below are mandatory.
Each side contributes 109 values in this order:

```text
transl_cam[3] + orient6d[6] + pose6d[90] + betas[10] = 109
```

The in-memory target concatenates the left hand before the right hand:

```text
hand_gt[218] = left[109] + right[109]
```

| Column pattern | Parquet type and shape | Definition |
| --- | --- | --- |
| `{side}_mano_transl_cam` | `fixed_size_list<float32>[3]` | MANO wrist/root translation in meters in the current camera frame. |
| `{side}_mano_orient6d` | `fixed_size_list<float32>[6]` | Global wrist/root orientation in the camera frame. The repository uses row-major rotation 6D: flatten the first two rows of a `3x3` rotation matrix. |
| `{side}_mano_pose6d` | `fixed_size_list<float32>[90]` | Fifteen local MANO joint rotations in official MANO order, each represented by the same row-major 6D convention: `15 * 6 = 90`. |
| `{side}_mano_betas` | `fixed_size_list<float32>[10]` | Ten dimensionless MANO shape coefficients. They are stored per frame even when constant across an episode. |

`side` is exactly `left` or `right`. Invalid rows should still contain finite
placeholders; `hand_kept` controls whether the values contribute to losses.

The current Stage 1 recipe does not use `left_kpt21` or `right_kpt21` because
`require_kpt21_gt: false`.

## Video contract and frame alignment

The referenced stream is `videos/observation.images.ego/**/*.mp4`. Prepare the
videos as H.264, YUV420p, 30 FPS. Decord returns RGB
`uint8` frames to the loader.

For a clip beginning at episode-local offset `off`, frame `k` is read as:

```text
video_frame[k] = round(from_timestamp * fps) + off + k
```

The MP4 must cover the full episode interval. The loader clamps an out-of-range
index to the final frame, but a valid training export must not depend on that
fallback because it would duplicate terminal images against changing labels.

Decoded frames are transformed as follows:

```text
[S,H0,W0,3] uint8 RGB
    -> permute to [S,3,H0,W0]
    -> float32 / 255
    -> bilinear resize to [S,3,378,518]
```

There is no crop, channel reordering, mean subtraction, or standard-deviation
normalization in the current loader.

## Clip construction

Both retained configurations use:

```yaml
clip_len: 32
clip_stride: 32
size_hw: [378, 518]
```

Clips never cross episode boundaries. For an episode of length `L`, starts are
`0, 32, 64, ...` while `start + 32 <= L`; a tail shorter than 32 frames is
dropped.

Camera extrinsics are made relative to the first frame of each 32-frame clip.
For stored world-to-camera rotation `R[s]` and translation `t[s]`:

```text
R_rel[s] = R[s] @ transpose(R[0])
t_rel[s] = t[s] - R_rel[s] @ t[0]
```

The first target frame is therefore identity rotation and zero translation.
`cam_fov` is not rebased. The resulting camera target is:

```text
gt_pose_enc[S,9] = t_rel[3] + quat_xyzw(R_rel)[4] + cam_fov[2]
```

With batch size `B`, the loader emits:

| Batch key | Shape | Source |
| --- | --- | --- |
| `images` | `[B,32,3,378,518]` | Decoded and resized ego video |
| `gt_pose_enc` | `[B,32,9]` | Clip-relative camera translation, quaternion, and FoV |
| `state_mask` | `[B,2]` | `state_mask` from the clip's first row |
| `hand_kept` | `[B,32,2]` | Per-frame left/right validity |
| `hand_gt` | `[B,32,218]` | Stage 1 MANO target; zero placeholder when MANO loading is disabled |
| `mano_gt_valid` | `[B]` | `true` for Stage 1 and `false` for Stage 2 |
| `kpt21_gt` | `[B,32,2,21,3]` | Zero placeholder in both retained recipes |
| `kpt21_gt_valid` | `[B]` | `false` in both retained recipes |

Before dataset iteration, automatic camera normalization scans
`episode_index`, `frame_index`, `cam_trans`, and `cam_quat` from every selected
root. It applies the same 32-frame sampling and first-frame rebasing, then
computes one global three-axis standard deviation for position, per-frame
velocity, and per-frame acceleration. These scales are injected into the batch;
they are not additional source Parquet columns.

## Required invariants

- An episode must be fully contained in one data Parquet file; split episodes are rejected by the normalization scan.
- Within each episode, `frame_index` starts at `0` and increases by exactly `1`.
- The episode metadata row range must fit inside the referenced data file.
- `length`, the data row range, and the timestamped MP4 interval must describe the same number of frames at `fps`.
- All camera values must be finite; FoV values must be positive and expressed in radians.
- Stage 1 roots must provide all left/right MANO columns with `hand_frame: camera` semantics.
- Rotation 6D rows must be finite and non-degenerate so Gram-Schmidt reconstruction is defined.
- `hand_kept` is the sole per-frame hand-loss mask and must match the actual validity of each side.

Columns commonly present in the inspected exports but not used by the retained
training recipes include `index`, `task_index`, `main_type`, per-frame
`timestamp`, task text, atomic segments, Stage 2 `cam_principal_norm`, and
all action fields. No `action` column is required by the MINT trainer.
