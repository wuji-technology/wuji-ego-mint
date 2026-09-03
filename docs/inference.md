# Inference and viewer

## Web viewer

```bash
python -m mint viewer
```

The viewer is the original `eval/model_effect` interactive interface. It starts
at `data/samples/lerobot_v3` and supports the bundled dataset's
ground-truth/prediction 2D and 3D comparison workflow. Model loading and
inference are separate explicit actions. The default model configuration is
`configs/training/mint_step2.yaml`.

Pass a path only when overriding a default, for example:

```bash
python -m mint viewer --ckpt /path/to/model.safetensors
```

If the default checkpoint is missing, run `bash scripts/download_assets.sh`.
The viewer provides:

- a LeRobot episode browser and raw GT mode;
- GT/prediction overlay and side-by-side playback;
- fixed-world and current-camera interactive 3D views;
- frame values, configured losses, cancellation, and video/frame export;
- optional MuJoCo and Wuji Hand panels when their dependencies/assets exist.

For a LeRobot episode with ground truth, the four selected export categories
(2D, fixed-world 3D, MuJoCo, and Wuji Hand) can be downloaded either as one
four-column grid MP4 or as a ZIP containing eight individual GT/prediction MP4
files. Prediction-only videos export one file per selected category.

The viewer is a resident process. Optional `--compile-mode` and `--fp8-mode`
settings apply to repeated workloads; leave them unset for the default eager
execution path.

Benchmark code is intentionally separate from the Viewer UI. Run
`eval/model_effect/benchmark/run.py` directly after configuring the required
datasets and optional environment.

## Headless command-line inference

`mint infer` is an alternative for automation and artifact export; it does not
need the viewer, and the viewer does not require it to run first.

```bash
python -m mint infer \
  --input /path/to/video.mp4 \
  --checkpoint checkpoints/model.safetensors \
  --output artifacts/example
```

The command decodes a bounded number of frames, runs windowed camera-and-hand
prediction, saves `prediction.npz`, and renders `prediction.mp4`. No annotation
file or ground-truth column is opened. Use `--no-render` when MANO assets are
unavailable or only numeric output is needed.

The one-video CLI keeps acceleration off by default because compiling a fresh
process can cost more than one short inference. Use `--compile-mode auto`,
`--fp8-mode auto`, and `--warmup-passes 2` only for repeated or sufficiently
large headless workloads.

## Browser startup

The Viewer automatically attempts to open its URL in the default browser after
startup. Pass `--no-open` when browser launch is not desired. The built-in
development server is not a public production server.
