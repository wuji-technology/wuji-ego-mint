#!/usr/bin/env python3
"""Launch reproducible external camera baselines in their isolated environments.

The launcher calls one explicit Python interpreter per method. Configure all
external roots through environment variables before launching a run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
EXTERNAL_ROOT = Path(os.environ.get("CAMERA_BASELINE_CODE_ROOT", "external/camera_baselines"))
CONDA_ENVS = Path(os.environ.get("CAMERA_BASELINE_ENVS_ROOT", "external/conda_envs"))
PIPELINE_ROOT = Path(os.environ.get("MINT_REPO_ROOT", str(REPO_ROOT)))
PIPELINE_PYTHON = Path(os.environ.get("MINT_PYTHON", sys.executable))
DEFAULT_RUN_ROOT = Path(os.environ.get("CAMERA_BASELINE_RUN_ROOT", "output/eval/camera_baselines"))
DEFAULT_WORK_ROOT = Path(os.environ.get("CAMERA_BASELINE_WORK_ROOT", "output/cache/camera_baselines"))
DATASETS = ("hot3d_val", "arctic_val")

TRAIN_RUN = Path(os.environ.get(
    "CAMERA_BASELINE_TRAIN_RUN", str(PIPELINE_ROOT / "output/model_train/REPLACE_ME")
))
TRAIN_CONFIG = TRAIN_RUN / "logs/record/config.yaml"


@dataclass(frozen=True)
class Method:
    env: str
    runner: Path
    args: tuple[str, ...]
    python: Path | None = None
    uses_work: bool = False

    @property
    def interpreter(self) -> Path:
        return self.python or CONDA_ENVS / self.env / "bin/python"


METHODS = {
    # Princeton-VL runner and weights, not MegaSaM's no-depth ablation.
    "droid_slam_official": Method(
        "icra_droidslam", EXTERNAL_ROOT / "code/droid_slam/run_droid_slam.py", (),
    ),
    "megasam": Method(
        "icra_megasam", EXTERNAL_ROOT / "code/megasam/run_megasam.py",
        ("--variant", "megasam"), uses_work=True,
    ),
    "hawor": Method(
        "icra_hawor", EXTERNAL_ROOT / "code/hawor/run_hawor.py",
        ("--stage", "native", "--camera-only"), uses_work=True,
    ),
    "infinitevggt": Method(
        "icra_vggt", EXTERNAL_ROOT / "code/infinitevggt/run_infinitevggt.py", (),
    ),
    "lingbotmap": Method(
        "icra_lingbotmap", EXTERNAL_ROOT / "code/common/run_feedforward.py",
        ("--variant", "lingbotmap", "--use-sdpa"),
    ),
    "egopipeline": Method(
        "icra_egopipe", EXTERNAL_ROOT / "code/megasam/run_megasam.py",
        ("--variant", "egopipeline"), uses_work=True,
    ),
    "ours_step4500": Method(
        "mint", EXTERNAL_ROOT / "code/ours/run_student.py",
        ("--ckpt", str(TRAIN_RUN / "step_00004500"),
         "--config", str(TRAIN_CONFIG), "--method", "ours_step4500"),
        python=PIPELINE_PYTHON,
    ),
    "ours_step19000": Method(
        "mint", EXTERNAL_ROOT / "code/ours/run_student.py",
        ("--ckpt", str(TRAIN_RUN / "step_00019000"),
         "--config", str(TRAIN_CONFIG), "--method", "ours_step19000"),
        python=PIPELINE_PYTHON,
    ),
}

DEFAULT_METHODS = (
    "droid_slam_official", "megasam", "hawor", "infinitevggt",
    "lingbotmap", "egopipeline", "ours_step4500", "ours_step19000",
)


def _sequences(data_root: Path, datasets: tuple[str, ...]) -> list[dict]:
    records = []
    for dataset in datasets:
        directory = data_root / dataset
        if not directory.is_dir():
            raise FileNotFoundError(f"dataset is missing: {directory}")
        for sequence in sorted(directory.iterdir()):
            images = sequence / "images"
            if not (sequence / "gt.npz").is_file() or not images.is_dir():
                continue
            frames = sum(1 for _ in images.glob("*.jpg"))
            if frames < 2:
                raise ValueError(f"sequence has fewer than two frames: {sequence}")
            records.append({"dataset": dataset, "sequence": sequence.name,
                            "path": str(sequence.resolve()), "frames": frames})
    if not records:
        raise FileNotFoundError(f"no camera sequences found under {data_root}")
    return records


def balanced_shards(records: list[dict], count: int) -> list[list[dict]]:
    """Longest-processing-time assignment keeps frame loads close across GPUs."""
    if count < 1:
        raise ValueError("shard count must be positive")
    shards = [[] for _ in range(count)]
    loads = [0] * count
    for record in sorted(records, key=lambda item: (-item["frames"], item["path"])):
        target = min(range(count), key=lambda index: (loads[index], index))
        shards[target].append(record)
        loads[target] += record["frames"]
    for shard in shards:
        shard.sort(key=lambda item: (item["dataset"], item["sequence"]))
    return shards


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _validate_methods(names: tuple[str, ...]) -> None:
    unknown = sorted(set(names) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")
    for name in names:
        method = METHODS[name]
        if not method.interpreter.is_file():
            raise FileNotFoundError(f"{name} Python is missing: {method.interpreter}")
        if not method.runner.is_file():
            raise FileNotFoundError(f"{name} runner is missing: {method.runner}")


def method_command(
    name: str, sequences: list[dict], pred_root: Path, work_root: Path,
    overwrite: bool = False,
) -> list[str]:
    method = METHODS[name]
    command = [str(method.interpreter), "-u", str(method.runner), *method.args,
               "--input", *(record["path"] for record in sequences),
               "--out", str(pred_root)]
    if method.uses_work:
        command.extend(("--work", str(work_root)))
    if overwrite:
        command.append("--overwrite")
    return command


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def audit_shard_outputs(name: str, shard: list[dict], pred_root: Path) -> list[dict]:
    issues = []
    for record in shard:
        path = (pred_root / name / "camera_pose" / record["dataset"]
                / f"{record['sequence']}.npz")
        if not path.is_file():
            issues.append({"sequence": record["sequence"], "kind": "missing_output"})
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                if "c2w" not in archive.files:
                    raise ValueError("missing c2w")
                c2w = np.asarray(archive["c2w"])
                if c2w.shape != (record["frames"], 4, 4):
                    raise ValueError(
                        f"c2w shape={c2w.shape}, expected=({record['frames']}, 4, 4)"
                    )
                if not np.isfinite(c2w).all():
                    raise ValueError("c2w contains NaN/Inf")
                declared = int(archive["frames"]) if "frames" in archive.files else len(c2w)
                if declared != record["frames"]:
                    raise ValueError(
                        f"declared frames={declared}, expected={record['frames']}"
                    )
        except (OSError, ValueError) as exc:
            issues.append({"sequence": record["sequence"], "kind": "invalid_output",
                           "detail": str(exc)})
    return issues


def recover_cached_moge_timings(
    pred_root: Path, log_root: Path, methods: tuple[str, ...],
) -> dict:
    """Restore first-pass MoGe time omitted by cache-backed SLAM retries."""
    sequence_pattern = re.compile(
        r"^\[\d+/\d+\]\s+([^/\s]+)/([^\s]+)\s+\d+\s+"
    )
    timing_pattern = re.compile(
        r"^\[moge\]\s+([^\s]+).*\(([0-9]+(?:\.[0-9]+)?)s\)\s*$"
    )
    summary = {"repaired": [], "missing": []}
    for method in methods:
        timings = {}
        for log_path in sorted(log_root.glob(f"{method}_gpu*.log")):
            current = None
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                sequence_match = sequence_pattern.match(line)
                if sequence_match:
                    current = sequence_match.groups()
                    continue
                timing_match = timing_pattern.match(line)
                if current and timing_match and timing_match.group(1) == current[1]:
                    elapsed = float(timing_match.group(2))
                    if elapsed > 0:
                        timings.setdefault(current, elapsed)

        prediction_root = pred_root / method / "camera_pose"
        if not prediction_root.is_dir():
            continue
        for path in sorted(prediction_root.glob("*/*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                payload = {key: np.array(archive[key]) for key in archive.files}
            variant = str(payload.get("variant", np.asarray("")).item())
            if "moge2_depth" not in variant:
                continue
            raw_stages = payload.get("stage_seconds", np.asarray("{}"))
            try:
                stages = json.loads(str(raw_stages.item()))
            except (ValueError, TypeError, json.JSONDecodeError):
                stages = {}
            if float(stages.get("moge2", 0.0) or 0.0) > 0:
                continue
            key = (path.parent.name, path.stem)
            elapsed = timings.get(key)
            if not elapsed:
                summary["missing"].append(f"{method}/{key[0]}/{key[1]}")
                continue
            stages["moge2"] = round(elapsed, 3)
            payload["seconds"] = np.float64(float(payload["seconds"].item()) + elapsed)
            payload["stage_seconds"] = np.asarray(json.dumps(stages))
            payload["timing_recovered_from"] = np.asarray("first_pass_worker_log")
            temporary = path.with_suffix(".tmp.npz")
            np.savez(temporary, **payload)
            temporary.replace(path)
            summary["repaired"].append({
                "method": method, "dataset": key[0], "sequence": key[1],
                "moge2_seconds": elapsed,
            })
    return summary


def run_worker(args) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    shard = manifest["shards"][args.shard]
    methods = tuple(manifest["methods"])
    pred_root = Path(manifest["pred_root"])
    work_root = Path(manifest["work_root"])
    log_root = Path(manifest["log_root"])
    status_path = log_root / f"status_gpu{args.gpu}.json"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["TORCH_HOME"] = str(EXTERNAL_ROOT / ".cache/torch")
    failures = []

    for index, name in enumerate(methods, 1):
        command = method_command(name, shard, pred_root, work_root, args.overwrite)
        log_path = log_root / f"{name}_gpu{args.gpu}.log"
        state = {
            "gpu": args.gpu, "shard": args.shard, "method": name,
            "method_index": index, "method_count": len(methods),
            "sequence_count": len(shard),
            "frame_count": sum(item["frames"] for item in shard),
            "status": "running", "command": command,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(status_path, state)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {state['updated_at']} fresh CPFS rerun =====\n")
            log.write(shlex.join(command) + "\n")
            log.flush()
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=environment, check=False)
        output_issues = audit_shard_outputs(name, shard, pred_root)
        state["status"] = (
            "completed" if result.returncode == 0 and not output_issues else "failed"
        )
        state["returncode"] = result.returncode
        state["output_issues"] = output_issues
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(status_path, state)
        if result.returncode or output_issues:
            failures.append({"method": name, "returncode": result.returncode,
                             "output_issues": output_issues})

    _atomic_json(status_path, {
        "gpu": args.gpu, "shard": args.shard,
        "status": "completed" if not failures else "completed_with_failures",
        "failures": failures, "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    return 1 if failures else 0


def build_manifest(args) -> tuple[dict, Path]:
    methods = _parse_csv(args.methods)
    datasets = _parse_csv(args.datasets)
    gpus = tuple(int(item) for item in _parse_csv(args.gpus))
    if not gpus:
        raise ValueError("at least one GPU is required")
    _validate_methods(methods)
    records = _sequences(Path(args.data_root), datasets)
    shards = balanced_shards(records, len(gpus))
    run_root = Path(args.run_root).resolve()
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "icra_full_sequence_v1",
        "data_root": str(Path(args.data_root).resolve()),
        "pred_root": str(run_root / "pred"),
        "work_root": str(Path(args.work_root).resolve()),
        "log_root": str(run_root / "logs"),
        "methods": list(methods), "datasets": list(datasets), "gpus": list(gpus),
        "shards": shards,
        "environments": {
            name: {"env": METHODS[name].env,
                   "python": str(METHODS[name].interpreter)}
            for name in methods
        },
    }
    path = run_root / "manifest.json"
    return manifest, path


def launch(args) -> int:
    manifest, manifest_path = build_manifest(args)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    _atomic_json(manifest_path, manifest)
    session = args.session
    if subprocess.run(("tmux", "has-session", "-t", session),
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session}")

    log_root = Path(manifest["log_root"])
    for shard, gpu in enumerate(manifest["gpus"]):
        _atomic_json(log_root / f"status_gpu{gpu}.json", {
            "gpu": gpu, "shard": shard, "status": "launching",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    script = Path(__file__).resolve()
    for shard, gpu in enumerate(manifest["gpus"]):
        worker = [sys.executable, str(script), "worker", "--manifest", str(manifest_path),
                  "--shard", str(shard), "--gpu", str(gpu)]
        if args.overwrite:
            worker.append("--overwrite")
        tmux_command = ["tmux", "new-session" if shard == 0 else "new-window", "-d"]
        if shard == 0:
            tmux_command.extend(("-s", session, "-n", f"gpu{gpu}"))
        else:
            tmux_command.extend(("-t", f"{session}:", "-n", f"gpu{gpu}"))
        tmux_command.append(shlex.join(worker))
        subprocess.run(tmux_command, check=True)

    finalizer = [sys.executable, str(script), "finalize", "--manifest", str(manifest_path)]
    finalize_log = log_root / "finalize.log"
    finalize_shell = f"{shlex.join(finalizer)} > {shlex.quote(str(finalize_log))} 2>&1"
    subprocess.run(("tmux", "new-window", "-d", "-t", f"{session}:",
                    "-n", "finalize", finalize_shell), check=True)

    loads = [sum(item["frames"] for item in shard) for shard in manifest["shards"]]
    print(f"launched tmux session {session}: GPUs={manifest['gpus']} frame_loads={loads}")
    print(f"manifest: {manifest_path}")
    print(f"logs: {manifest['log_root']}")
    print("finalizer: waits for all workers, audits coverage, then publishes panel rows")
    return 0


def finalize(args) -> int:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log_root = Path(manifest["log_root"])
    terminal = {"completed", "completed_with_failures"}
    while True:
        statuses = []
        for gpu in manifest["gpus"]:
            path = log_root / f"status_gpu{gpu}.json"
            if not path.is_file():
                statuses.append("missing")
                continue
            try:
                statuses.append(json.loads(path.read_text(encoding="utf-8")).get("status"))
            except (OSError, json.JSONDecodeError):
                statuses.append("invalid")
        print(f"[finalize] worker statuses: {statuses}", flush=True)
        if len(statuses) == len(manifest["gpus"]) and all(item in terminal for item in statuses):
            break
        time.sleep(max(5, args.poll_seconds))

    timing_recovery = recover_cached_moge_timings(
        Path(manifest["pred_root"]), log_root, tuple(manifest["methods"]),
    )
    print(f"[finalize] timing recovery: {timing_recovery}", flush=True)
    importer = Path(__file__).with_name("import_external_results.py")
    panel_js = (
        PIPELINE_ROOT / "eval/model_effect/visualization/viewer/web/camera_baselines.js"
    )
    output = Path(manifest["pred_root"]).parent / "metrics/baselines.json"
    command = [
        str(PIPELINE_PYTHON), str(importer),
        "--pred-root", manifest["pred_root"],
        "--data-root", manifest["data_root"],
        "--methods", ",".join(manifest["methods"]),
        "--datasets", ",".join(manifest["datasets"]),
        "--output", str(output), "--panel-js", str(panel_js),
    ]
    print(f"[finalize] {shlex.join(command)}", flush=True)
    result = subprocess.run(command, check=False)
    summary = {
        "status": "completed" if result.returncode == 0 else "audit_failed",
        "returncode": result.returncode,
        "worker_statuses": statuses,
        "timing_recovery": timing_recovery,
        "metrics": str(output), "panel_js": str(panel_js),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(log_root / "finalize_status.json", summary)
    return result.returncode


def show_status(args) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    log_root = Path(manifest["log_root"])
    pred_root = Path(manifest["pred_root"])
    all_records = [record for shard in manifest["shards"] for record in shard]
    print("Workers")
    for gpu in manifest["gpus"]:
        path = log_root / f"status_gpu{gpu}.json"
        if not path.is_file():
            print(f"  GPU {gpu}: waiting for status")
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  GPU {gpu}: invalid status ({exc})")
            continue
        method = value.get("method", "-")
        progress = (
            f" {value.get('method_index')}/{value.get('method_count')}"
            if value.get("method_index") else ""
        )
        print(f"  GPU {gpu}: {value.get('status', 'unknown')} {method}{progress}")

    print("\nOutputs")
    for method in manifest["methods"]:
        pieces = []
        for dataset in manifest["datasets"]:
            records = [record for record in all_records if record["dataset"] == dataset]
            complete = 0
            frames = 0
            for record in records:
                path = (pred_root / method / "camera_pose" / dataset
                        / f"{record['sequence']}.npz")
                if not path.is_file():
                    continue
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        c2w = np.asarray(archive["c2w"])
                    if c2w.shape == (record["frames"], 4, 4) and np.isfinite(c2w).all():
                        complete += 1
                        frames += record["frames"]
                except (OSError, ValueError, KeyError):
                    pass
            expected_frames = sum(record["frames"] for record in records)
            pieces.append(
                f"{dataset} {complete}/{len(records)} seq, {frames}/{expected_frames} frames"
            )
        print(f"  {method:<26} " + " | ".join(pieces))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("launch", help="create one sequential worker per GPU")
    start.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    start.add_argument("--datasets", default=",".join(DATASETS))
    start.add_argument("--gpus", default="0,1,2,3")
    start.add_argument("--data-root", default=str(EXTERNAL_ROOT / "data"))
    start.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    start.add_argument("--work-root", default=str(DEFAULT_WORK_ROOT))
    start.add_argument("--session", default="mint-camera-benchmark")
    start.add_argument("--overwrite", action="store_true")
    start.add_argument("--dry-run", action="store_true")
    start.set_defaults(func=launch)

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--manifest", required=True)
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--gpu", type=int, required=True)
    worker.add_argument("--overwrite", action="store_true")
    worker.set_defaults(func=run_worker)

    finish = subparsers.add_parser("finalize", help="wait for workers and publish audited rows")
    finish.add_argument("--manifest", required=True)
    finish.add_argument("--poll-seconds", type=int, default=30)
    finish.set_defaults(func=finalize)

    status = subparsers.add_parser("status", help="show audited per-method output progress")
    status.add_argument("--manifest", default=str(DEFAULT_RUN_ROOT / "manifest.json"))
    status.set_defaults(func=show_status)
    args = parser.parse_args()
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
