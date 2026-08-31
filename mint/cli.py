"""Stable command surface for the data, training, inference, and viewer tools."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _run_script(relative: str, args: list[str]) -> int:
    return subprocess.call([sys.executable, str(PROJECT_DIR / relative), *args])


def _normalize_pipeline_args(args: list[str]) -> list[str]:
    aliases = {
        "--num-gpus": "--num_gpus",
        "--retry-failed": "--retry_failed",
        "--long-video-threshold-s": "--long_video_threshold_s",
    }
    return [aliases.get(item, item) for item in args]


def main(argv: list[str] | None = None) -> int:
    import argparse

    raw = list(sys.argv[1:] if argv is None else argv)
    commands = ("pipeline", "train", "infer", "viewer", "doctor")
    parser = argparse.ArgumentParser(
        prog="python -m mint",
        description="MINT egocentric video toolkit",
        epilog="Use 'python -m mint <command> --help' for command-specific options.",
    )
    parser.add_argument("command", nargs="?", choices=commands)
    if not raw or raw[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if raw[0] not in commands:
        parser.error(f"unknown command: {raw[0]}")
    command, remainder = raw[0], raw[1:]

    if command == "pipeline":
        return _run_script("ego_pipeline/run_batch.py", _normalize_pipeline_args(remainder))
    if command == "train":
        return _run_script("model_train/train.py", remainder)
    if command == "infer":
        from .inference.runner import inference_main

        return inference_main(remainder)
    if command == "viewer":
        return _run_script("eval/model_effect/visualization/viewer_web.py", remainder)
    from .doctor import doctor_main

    return doctor_main(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
