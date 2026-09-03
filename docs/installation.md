# Installation

## Supported platform

- Linux x86_64
- Python 3.10
- NVIDIA GPU with a driver compatible with the selected PyTorch CUDA build
- Conda or Mamba
- FFmpeg with H.264 encoding support

The tested model environment uses PyTorch 2.8.0 and torchvision 0.23.0. On
Linux x86_64, these PyPI packages install the CUDA 12.8 runtime dependencies.
CPU import and configuration tests are supported, but practical training and
model inference require a CUDA GPU.

## Inference environment

```bash
bash scripts/create_env.sh inference
conda activate mint-inference
python -m mint doctor --profile inference --strict
```

The inference profile includes PyTorch, OpenCV, Flask, PyArrow, Decord,
numerical dependencies, and FFmpeg. It excludes Ray, W&B, data conversion,
model compilers, and upstream processing backends. Start the default interactive workflow with:

```bash
bash scripts/download_assets.sh
python -m mint viewer
```

## Full data and training environment

```bash
bash scripts/create_env.sh full
conda activate mint
python -m mint doctor --profile full --strict
```

The full profile adds the Ray data-pipeline and training dependencies. Both
profiles use pinned requirement layers and never clone another local
environment. If the selected environment already exists, the script reuses it
and continues package installation.
The initial solve uses the channels already configured on the host. A resolved
configuration containing a known-blocked TUNA or HIT endpoint is skipped. If
the system solve is skipped or fails, the script retries with
`--override-channels` and the official conda-forge channel. The override
prevents defaults and every configured channel from being queried during
fallback. It never edits `.condarc` or global settings.
Set `MINT_CONDA_FALLBACK_CHANNELS` to a space-separated list of trusted channel
URLs when another fallback order is required.
Both profiles install one identical inference foundation before any full-only
packages. NumPy, PyTorch, and the remaining inference requirements use the
USTC PyPI mirror by default. The full-only requirement layers are constrained
by the pinned inference and PyTorch files so they cannot replace those shared
versions.
Set `MINT_TORCH_INDEX_URL` to use another compatible package index, or
`MINT_TORCH_FIND_LINKS` for a trusted flat wheel mirror, before running the
script. `MINT_PYPI_INDEX_URL` overrides the general package source, and
`MINT_NUMPY_INDEX_URL` overrides only NumPy when a separate trusted wheel source
is required.

The environment script streams subprocess output directly. Conda runs in
verbose mode, and pip download/install progress bars are forced on, so long
CUDA wheel downloads no longer appear to hang silently.

If a download is interrupted, run the same command again. The script reuses
the existing environment and resumes package installation instead of requiring
the partially created environment to be deleted. Only activate the environment
and run `mint doctor` after the script prints that the environment is ready.

## Model checkpoint

The asset helper downloads the public MINT checkpoint to
`checkpoints/model.safetensors`. It tries the
[ModelScope release](https://www.modelscope.cn/models/AsherZhu/mint_v1) first,
then the [Hugging Face release](https://huggingface.co/ZZJAsher/mint_v1), and
finally the configured Hugging Face mirror. Set `HF_TOKEN` if Hugging Face
requires authentication. The helper selects only `model.safetensors`; it does
not fetch optimizer or random-state files. You may also place that file
manually or pass another compatible checkpoint to the Viewer with `--ckpt`.

Do not commit checkpoints to Git. The model configuration must match the
checkpoint architecture.

The optional pretrained LingBot-Map backbone belongs at:

```text
assets/models/lingbot-map.pt
```

Obtain it from the official LingBot-Map release and review its model terms.
The public asset helper does not download this optional backbone. It downloads
only the public MINT checkpoint. The URDF, MJCF, and STL robot-description
assets are bundled with the source repository and require no separate download.
MANO remains a separately licensed asset and is not downloaded by the helper.
The checkpoint is checksum-verified before it replaces the destination:

```bash
bash scripts/download_assets.sh
```

Downloads show a live progress bar and use `.part` files for safe resume. For
Hugging Face assets, the helper first tries the official endpoint. If the
connection does not start transferring at least 1 KiB/s within 30 seconds, it
automatically continues the same partial file through `https://hf-mirror.com`.
Run the same command again after any interruption; completed bytes are kept.
Set `MINT_HF_MIRROR` to use a different mirror, or tune the timeout with
`MINT_DOWNLOAD_SPEED_TIME`.

## MANO assets

Mesh and skeleton rendering requires separately licensed MANO files:

```text
assets/mano/mano_right/MANO_RIGHT.pkl
assets/mano/mano_left/MANO_LEFT.pkl
```

Create an account on the [official MANO website](https://mano.is.tue.mpg.de/),
download `mano_v*_*.zip`, and accept the
[MANO license](https://mano.is.tue.mpg.de/license.html). MANO cannot be
downloaded or redistributed by this project.

HaWoR and the data pipeline use the upstream `_DATA` layout:

```text
third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl
third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

The standalone viewer also accepts the public-project layout shown above. It
automatically checks the HaWoR locations when `assets/mano/` is absent.

## Data backends

GeoCalib, MoGe, and Mega-SAM source snapshots are available under
`third_party/`, but they do not make the production data pipeline turnkey. An
authorized HaWoR checkout and any required compatibility adaptations must be
supplied locally at `third_party/HaWoR`; the MINT-adapted infra copy is not part
of source releases because HaWoR prohibits redistribution of modifications.

After reviewing `THIRD_PARTY_NOTICES.md` and completing the local integration,
the installer can acknowledge the restrictions and register the source already
present on that machine:

```bash
bash scripts/install_data_backends.sh
python -m mint doctor --profile data --strict
```

The installer neither downloads missing source nor recreates MINT's internal
third-party adaptations. It also does not download model weights or MANO. Some
native CUDA extensions used by the full processing chain must be compiled
for the installed PyTorch and GPU architecture. Do not copy extensions from a
different Conda environment: extension ABI mismatches can produce silent
errors or crashes.

Place separately obtained backend assets at the paths already used by the
pipeline:

```text
model/geocalib/pinhole.tar
model/moge2/model.pt
model/megasam/megasam_final.pth
model/hawor/hawor.ckpt
model/hawor/model_config.yaml
model/hawor/detector.pt
third_party/HaWoR/weights/external/droid.pth
third_party/HaWoR/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth
```

These files remain Git-ignored. Their licenses are independent from the source
licenses summarized in `THIRD_PARTY_NOTICES.md`.

## Offline installation

For an offline host, populate a wheelhouse on a connected machine with the
same Linux, Python, PyTorch, and CUDA targets. Install with `--no-index` and a
trusted local `--find-links` directory. Keep the wheelhouse outside Git because
CUDA wheels are large and may carry their own redistribution terms.
