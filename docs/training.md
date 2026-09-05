# Training

## Dataset

The loader expects LeRobot v3 datasets. The repository keeps one training
configuration per stage: Stage 1 uses the Ego4D, EgoDex, and EPIC-KITCHENS roots,
and Stage 2 uses a separately prepared high-precision camera-trajectory dataset.

Run the commands from the repository root. The configurations use portable
paths; prepare the corresponding files or update the paths before training:

| Setting | Default path |
| --- | --- |
| Stage 1 `model.pretrained` | `assets/models/lingbot-map.pt` |
| Stage 1 `data.root` | `data/lerobot/{ego4d,egodex,epickitchen}/lerobot_v3` (three roots) |
| Stage 2 `data.root` | `data/lerobot/stage2/lerobot_v3` |
| Stage 2 `train.init_from` | `checkpoints/stage1/model.safetensors` |

The dataset roots and Stage 1 initialization checkpoint are not populated by
the model download script. Set `train.init_from` to the actual `model.safetensors`
produced by your Stage 1 run, or copy that file to the default path.

See [LeRobot v3 training data contract](lerobot-training-data.md) for the exact
directory layout, Parquet columns, coordinate conventions, masks, video-frame
alignment, and clip construction used by these recipes.

## Inspect first

```bash
python -m mint train --config configs/training/mint_step1.yaml --inspect
python -m mint train --config configs/training/mint_step2.yaml --inspect
```

Inspection builds the model on CPU, prints the module and freeze structure, and
skips pretrained weights and dataset loading. It catches registry and shape
configuration errors without allocating training GPUs.

## Train

```bash
python -m mint train --config configs/training/mint_step1.yaml
python -m mint train --config configs/training/mint_step2.yaml
```

Stage 1 pretrains on the three public data sources.
Stage 2 initializes from the Stage 1 model, freezes the aggregator, FoV, and hand
modules, and trains only the camera head on high-precision camera-trajectory
data. Accelerate selects the visible GPU topology. Strictly
deterministic CUDA algorithms remain disabled because some required operations
do not provide deterministic implementations.

Both configurations use offline W&B logging without a credential file. To use
online logging, authenticate with `wandb login` or your own `WANDB_API_KEY`, then
set `train.wandb.mode: online`. Set `train.wandb.enabled: false` to disable W&B.

The optional Stage 1 `aliyun` section contains placeholders for the workspace,
resource, image, and CPFS mount. Configure them for your cloud environment
before submission.

## Resume and initialize

- `--resume <step-directory>` restores model, optimizer, scheduler, random
  state, and global step from an Accelerate checkpoint.
- `--init-from <checkpoint>` loads model parameters only and starts a new
  optimizer and schedule.

These options are intentionally exclusive. Keep the resolved configuration
snapshot beside each run to make later inference reproducible.
