# Architecture

MINT keeps inference and training modular and exposes the redistributable parts
of the research data pipeline as an implementation reference.

```text
Approved videos
      |
      v
Locally integrated Ray pipeline -----> LeRobot v3 dataset
      |                         |
      |                         v
Optional research backends   Training engine -----> Checkpoint
                                                   |
                                                   v
                                      Prediction engine
                                         |       |
                                         v       v
                                      NPZ data  Web viewer
```

## Boundaries

### Data plane

`ego_pipeline/` contains the repository-owned scheduling, actors, resumable
manifests, monitoring, and LeRobot export code. Its `backends/` package defines
interfaces for optional upstream research models, and its `data_cleaning/`
package contains trajectory filters used before export. A public checkout is
not a complete production data generator until the user supplies and integrates
the required upstream backends locally.

Redistributable backend source snapshots live under `third_party/`, together
with their upstream licenses and provenance manifest. They are not sufficient
to recreate every production adaptation. Model weights, MANO files, generated
extensions, and upstream Git histories remain outside the MINT source release.
The MINT-adapted HaWoR infra copy exists only on the development machine and is
Git-ignored because CC BY-NC-ND 4.0 restricts commercial use and prohibits
redistribution of modifications.

### Training plane

`model_train/` is a configuration-driven PyTorch and Accelerate stack. A small
registry constructs the dataset, model, and loss objects. The public build
includes the primary LingBot-Map student and excludes cloud-vendor submission
logic, private dataset mixtures, credentials, and experimental model variants.

### Inference plane

`mint/inference/` reuses the exact training preprocessing and model constructor.
It supports bounded windowed inference and saves only prediction arrays. The
renderer decodes predicted camera and MANO values without loading annotations.

### Viewer boundary

The original `eval/model_effect` Viewer starts at the configured sample root
and deliberately provides server-side directory browsing, including parent
navigation. Treat it as a trusted local research tool: use SSH forwarding or
an authenticated reverse proxy, and do not expose it directly to an untrusted
network. Its Benchmark panel and the standalone benchmark CLI share the same
evaluation implementation.

## Design principles

1. Portable paths are resolved from the repository, never a developer home.
2. Credentials arrive through the runtime environment, never tracked files.
3. Dataset and checkpoint licenses remain explicit installation boundaries.
4. Long-running work is resumable or bounded by an explicit frame limit.
5. Prediction visualization is independent of ground-truth data.
