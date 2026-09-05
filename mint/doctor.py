"""Environment and asset diagnostics with privacy-safe output."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _module(name: str, required: bool = True) -> Check:
    found = importlib.util.find_spec(name) is not None
    version = "not installed"
    if found:
        distribution = {
            "cv2": "opencv-python-headless",
            "pinocchio": "pin",
            "yaml": "PyYAML",
        }.get(name, name)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "available"
    return Check(name, found, version, required)


def _model_stack() -> Check:
    """Import the same model modules used when the Viewer loads a checkpoint."""
    model_train = PROJECT_DIR / "model_train"
    for path in (model_train, model_train / "_vendor"):
        value = str(path)
        if path.is_dir() and value not in sys.path:
            sys.path.insert(0, value)
    try:
        importlib.import_module("models")
        importlib.import_module("lingbot_map.models.gct_stream")
    except Exception as exc:
        return Check("MINT model stack", False, f"{type(exc).__name__}: {exc}")
    return Check("MINT model stack", True, "importable")


def run_checks(profile: str) -> list[Check]:
    checks = [
        Check("python", sys.version_info[:2] == (3, 10), sys.version.split()[0]),
        Check("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not found"),
        _module("torch"),
        _module("numpy"),
        _module("cv2"),
        _module("yaml"),
    ]
    if profile in {"inference", "full"}:
        checks.extend([
            _module("flask"),
            _module("scipy"),
            _module("smplx"),
            _module("torchao"),
            _module("decord"),
            _module("einops"),
            _module("huggingface_hub"),
            _module("mujoco"),
            _module("nlopt"),
            _module("pinocchio"),
            _module("pyarrow"),
            _module("safetensors"),
            _module("tqdm"),
            _model_stack(),
        ])
        right = PROJECT_DIR / "assets" / "mano" / "mano_right" / "MANO_RIGHT.pkl"
        left = PROJECT_DIR / "assets" / "mano" / "mano_left" / "MANO_LEFT.pkl"
        hawor_data = PROJECT_DIR / "third_party" / "HaWoR" / "_DATA"
        right = right if right.is_file() else hawor_data / "data" / "mano" / "MANO_RIGHT.pkl"
        left = left if left.is_file() else hawor_data / "data_left" / "mano_left" / "MANO_LEFT.pkl"
        checks.append(Check("MANO assets", right.is_file() and left.is_file(), "licensed files", False))
        wuji_body = (
            PROJECT_DIR / "eval" / "simulate" / "wuji-retargeting" / "wuji_retargeting"
            / "wuji-description" / "hand" / "body"
        )
        wuji_core = [
            wuji_body / kind / f"{side}.{extension}"
            for kind, extension in (("urdf", "urdf"), ("mjcf", "xml"))
            for side in ("left", "right")
        ]
        wuji_meshes = list((wuji_body / "meshes").glob("*/*.STL"))
        checks.append(Check(
            "Wuji retargeting assets",
            all(path.is_file() for path in wuji_core) and len(wuji_meshes) == 52,
            f"{len(wuji_meshes)} STL meshes",
        ))
    if profile in {"train", "full"}:
        checks.extend([_module("accelerate"), _module("decord"), _module("pyarrow")])
    if profile in {"data", "full"}:
        checks.extend([_module("ray"), _module("joblib"), _module("ultralytics")])
        for name in ("GeoCalib", "MoGe", "mega-sam"):
            path = PROJECT_DIR / "third_party" / name
            checks.append(Check(
                f"backend:{name}", path.is_dir(),
                "source available" if path.is_dir() else "source missing",
            ))
        assets = {
            "GeoCalib weights": PROJECT_DIR / "model" / "geocalib" / "pinhole.tar",
            "MoGe weights": PROJECT_DIR / "model" / "moge2" / "model.pt",
            "Mega-SAM weights": PROJECT_DIR / "model" / "megasam" / "megasam_final.pth",
        }
        for name, path in assets.items():
            detail = str(path.relative_to(PROJECT_DIR))
            checks.append(Check(name, path.is_file(), detail, False))
    return checks


def doctor_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a MINT runtime without exposing host details.")
    parser.add_argument("--profile", choices=("inference", "train", "data", "full"), default="full")
    parser.add_argument("--strict", action="store_true", help="Return a non-zero status for required failures.")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    checks = run_checks(args.profile)

    if args.as_json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        width = max(len(check.name) for check in checks)
        for check in checks:
            marker = "PASS" if check.ok else ("FAIL" if check.required else "NOTE")
            print(f"[{marker:4}] {check.name:<{width}}  {check.detail}")

    failed = [check for check in checks if check.required and not check.ok]
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(doctor_main())
