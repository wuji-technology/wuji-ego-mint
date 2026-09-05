from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
TRAINING_CONFIG_DIR = REPO_ROOT / "configs" / "training"
STAGE1_NAME = "mint_step1.yaml"
STAGE2_NAME = "mint_step2.yaml"


def _load(name: str) -> dict:
    return yaml.safe_load((TRAINING_CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_only_selected_stage_configs_are_kept():
    assert sorted(path.name for path in TRAINING_CONFIG_DIR.glob("*.yaml")) == [
        STAGE1_NAME,
        STAGE2_NAME,
    ]


def test_stage1_uses_public_pretraining_recipe():
    config = _load(STAGE1_NAME)

    assert config["model"]["pretrained_exclude"] == ["camera_head*"]
    assert "freeze" not in config["model"]
    assert len(config["data"]["root"]) == 3
    assert config["data"]["require_mano_gt"] is True
    assert set(config["loss"]) == {
        "hand_presence", "image_hand", "camera", "fov", "mano_param"
    }
    assert config["train"]["grad_accum"] == 1
    assert config["optim"]["lr"] == 1.0e-4


def test_stage2_matches_step_4500_camera_only_recipe():
    config = _load(STAGE2_NAME)

    assert config["model"]["pretrained"] is None
    assert config["model"]["freeze"] == [
        "backbone.aggregator*",
        "backbone.fov_head*",
        "hand_head*",
        "hand_presence_head*",
    ]
    assert config["data"]["require_mano_gt"] is False
    assert set(config["loss"]) == {"camera"}
    assert [term["name"] for term in config["loss"]["camera"]["terms"]] == [
        "trans_l1", "rot_geo", "trans_vel_l1", "rot_vel_geo"
    ]
    assert config["train"]["grad_accum"] == 2
    assert config["train"]["init_from"] == "checkpoints/stage1/model.safetensors"
    assert config["optim"]["param_groups"] == [{
        "match": "backbone.camera_head",
        "lr": 5.0e-5,
        "grad_value_clip": 1.0,
    }]
