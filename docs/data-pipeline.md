# Data pipeline reconstruction reference

The public MINT release does not include a turnkey copy of the production data
generator. It publishes the repository-owned Ray orchestration, backend
interfaces, trajectory cleaning, manifests, and LeRobot v3 export code as an
implementation reference. Some adapted third-party code cannot be redistributed
under its upstream license, and all model weights and separately licensed assets
are excluded.

For the supported public workflow, use the Viewer or `mint infer` to run MINT
inference and export prediction artifacts. Reconstruct the raw-video-to-LeRobot
pipeline only when that production capability is specifically required.

## Local reconstruction requirements

Before attempting a local integration:

1. Read `THIRD_PARTY_NOTICES.md` and verify that the intended use is permitted.
2. Obtain the required upstream libraries, including GeoCalib, MoGe, Mega-SAM,
   and HaWoR, under their original license terms.
3. Obtain the required checkpoints, MANO files, and other licensed assets from
   their official sources.
4. Reconcile the upstream APIs with the contracts in `ego_pipeline/` and keep
   any non-redistributable adaptations local. An AI coding assistant may be used
   to help implement this compatibility layer.
5. Validate the completed local environment before processing any data.

The source snapshots under `third_party/` and
`scripts/install_data_backends.sh` do not by themselves recreate the internal
production environment.

## Conditional local entry point

After every required backend and compatibility adapter has been supplied
locally, validate the integration:

```bash
conda activate mint
python -m mint doctor --profile data --strict
```

Only after that check passes should the local pipeline entry point be used:

```bash
python -m mint pipeline \
  --input /path/to/approved-video.mp4 \
  --output output/processed \
  --num-gpus 1
```

This command is an integration entry point, not a guarantee that a fresh public
checkout can reproduce the production output. Start with one short,
non-sensitive video and verify intermediate trajectories, final Parquet/video
alignment, and available disk space before scheduling a long Ray job.

If the local integration supports manifests, resume the same output instead of
copying partial files:

```bash
python -m mint pipeline --input /path/to/input --resume output/processed/<run-directory>
```

Use `--retry-failed` only after addressing the recorded error. Retrying a
missing backend, adapter, or model asset without fixing it creates noisy
duplicate failures.

## Output contract

The open output contract consists of:

- per-video prediction and cleaned trajectory files;
- sidecar metadata that uses paths relative to the run root;
- LeRobot v3 Parquet and video chunks;
- a resumable manifest with status and failure summaries.

No output should contain source-system mount names, operator usernames, cloud
credentials, or unrestricted source paths. Run the privacy audit before using
an output as a release artifact.
