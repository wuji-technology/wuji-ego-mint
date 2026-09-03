import sys
from pathlib import Path

import numpy as np
import pytest


MODEL_EFFECT = Path(__file__).resolve().parents[2]
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))

from visualization.reproj_core import mano  # noqa: E402
from visualization.viewer import ckpts  # noqa: E402
from visualization.viewer import store as viewer_store  # noqa: E402
from visualization.viewer.const import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_HAND_MODE,
    DEFAULT_UKF_PARAMS,
)


def test_viewer_finds_repo_mano_assets():
    mano.ensure_mano_weights()
    assert (mano._MANO_RIGHT_DIR / "MANO_RIGHT.pkl").is_file()
    assert (mano._MANO_LEFT_DIR / "MANO_LEFT.pkl").is_file()
    pytest.importorskip("scipy")
    data = mano._load_mano_data(mano._MANO_RIGHT_DIR, is_right=True)
    assert isinstance(data["shapedirs"], np.ndarray)
    assert data["shapedirs"].shape == (778, 3, 10)


def test_viewer_defaults_to_ukf_smoothing():
    assert DEFAULT_HAND_MODE == "smooth"
    assert DEFAULT_UKF_PARAMS == {"q": 0.7, "r": 0.5, "beta": 0.3}


def test_viewer_normalizes_parameterized_ukf_mode():
    pytest.importorskip("flask")
    from visualization.viewer.routes import _normalize_hand_mode

    assert _normalize_hand_mode("hard") == "hard"
    assert _normalize_hand_mode("smooth") == "smooth@0.6,0.6,2"
    assert _normalize_hand_mode("smooth@0.7000,0.500,1.600") == "smooth@0.7,0.5,1.6"
    with pytest.raises(ValueError):
        _normalize_hand_mode("smooth@0.7,9,1.6")


def test_viewer_defaults_to_stage2_checkpoint_config():
    assert DEFAULT_CONFIG.name == "mint_step2.yaml"


def test_viewer_enters_directly_nested_lerobot_dataset(tmp_path):
    from visualization.viewer.routes import _default_input_path

    dataset = tmp_path / "lerobot_v3"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "info.json").write_text("{}", encoding="utf-8")

    assert _default_input_path(tmp_path) == dataset.resolve()
    assert _default_input_path(dataset) == dataset.resolve()


def test_viewer_keeps_parent_when_multiple_nonstandard_datasets_exist(tmp_path):
    from visualization.viewer.routes import _default_input_path

    for name in ("dataset_a", "dataset_b"):
        meta = tmp_path / name / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text("{}", encoding="utf-8")

    assert _default_input_path(tmp_path) == tmp_path.resolve()


def test_public_checkpoint_is_discoverable(monkeypatch, tmp_path):
    model_train = tmp_path / "output" / "model_train"
    public_checkpoint = tmp_path / "checkpoints" / "model.safetensors"
    public_checkpoint.parent.mkdir(parents=True)
    public_checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(ckpts, "MODEL_TRAIN_ROOT", model_train)
    monkeypatch.setattr(ckpts, "DEFAULT_CHECKPOINT", public_checkpoint)

    assert ckpts.auto_pick_ckpt() == str(public_checkpoint)
    assert ckpts.DEFAULT_CHECKPOINT_RUN in ckpts.list_runs()
    assert ckpts.list_steps(ckpts.DEFAULT_CHECKPOINT_RUN) == ["model.safetensors"]
    assert ckpts.browse("")["dirs"] == [ckpts.DEFAULT_CHECKPOINT_RUN]
    assert ckpts.browse(ckpts.DEFAULT_CHECKPOINT_RUN)["steps"] == ["model.safetensors"]
    assert ckpts.resolve_ckpt(
        ckpts.DEFAULT_CHECKPOINT_RUN, "model.safetensors"
    ) == public_checkpoint.resolve()


def test_latest_training_checkpoint_precedes_public_checkpoint(monkeypatch, tmp_path):
    model_train = tmp_path / "output" / "model_train"
    training_checkpoint = model_train / "run" / "step_00000002"
    training_checkpoint.mkdir(parents=True)
    public_checkpoint = tmp_path / "checkpoints" / "model.safetensors"
    public_checkpoint.parent.mkdir(parents=True)
    public_checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(ckpts, "MODEL_TRAIN_ROOT", model_train)
    monkeypatch.setattr(ckpts, "DEFAULT_CHECKPOINT", public_checkpoint)

    assert ckpts.auto_pick_ckpt() == str(training_checkpoint)


def test_arbitrary_checkpoint_file_and_directory_are_resolved(tmp_path):
    checkpoint_dir = tmp_path / "external" / "step_custom"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_file = checkpoint_dir / "weights.safetensors"
    checkpoint_file.write_bytes(b"weights")

    assert ckpts.resolve_checkpoint_path(checkpoint_dir) == checkpoint_dir.resolve()
    assert ckpts.resolve_checkpoint_path(checkpoint_file) == checkpoint_file.resolve()
    assert ckpts.resolve_checkpoint_path(tmp_path / "missing") is None


def test_optional_frame_metrics_failure_does_not_break_visualization(monkeypatch):
    from visualization.render import metrics

    def fail_metrics(*args, **kwargs):
        raise RuntimeError("normalization metadata missing")

    monkeypatch.setattr(metrics, "frame_metrics", fail_metrics)
    result, error = viewer_store._safe_frame_metrics({}, {}, {}, None, None)

    assert result is None
    assert error == "RuntimeError: normalization metadata missing"


def test_checkpoint_browser_and_path_selection_api(tmp_path):
    pytest.importorskip("flask")
    from visualization.viewer.routes import create_app

    checkpoint_dir = tmp_path / "models"
    checkpoint_dir.mkdir()
    checkpoint_file = checkpoint_dir / "custom.safetensors"
    checkpoint_file.write_bytes(b"weights")

    class FakeStore:
        root = tmp_path
        default_mode = "mesh_skel"
        ckpt_path = str(checkpoint_file)
        ckpt_tag = "custom"
        benchmark_args = None

        def swap_ckpt(self, path):
            self.ckpt_path = path
            return {"ckpt": path, "tag": "selected", "reload": False}

        def start_benchmark(self, **kwargs):
            self.benchmark_args = kwargs
            return {"ok": True}

    store = FakeStore()
    app = create_app(store)
    client = app.test_client()
    response = client.get("/api/ckpt/browse", query_string={"path": checkpoint_dir})
    assert response.status_code == 200
    assert response.get_json()["files"] == ["custom.safetensors"]

    response = client.post("/api/ckpt", json={"path": str(checkpoint_file)})
    assert response.status_code == 200
    assert response.get_json()["ckpt"] == str(checkpoint_file.resolve())

    response = client.post("/api/benchmark/start", json={
        "models": [{"path": str(checkpoint_file), "label": "custom"}],
    })
    assert response.status_code == 200
    assert store.benchmark_args["checkpoints"][0]["ckpt"] == str(checkpoint_file.resolve())
