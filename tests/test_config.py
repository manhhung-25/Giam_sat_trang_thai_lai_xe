from pathlib import Path

import pytest

from driver_safety.config import load_config


def test_load_default_config() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.vision.provider == "auto"
    assert config.thresholds.eye_aspect_ratio > 0


def test_invalid_config_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("vision:\n  process_every_n_frames: 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_camera_config_loaded() -> None:
    config = load_config(Path("configs/raspi5-realtime.yaml"))
    assert config.camera.width == 320
    assert config.camera.height == 240
    assert config.camera.fps == 30
    assert config.camera.buffer_size == 1
    assert config.actuators.enabled is True
    assert config.actuators.buzzer_gpio == 18
    assert config.actuators.led_gpio == 23


def test_invalid_fatigue_policy_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad-fatigue.yaml"
    path.write_text(
        "fatigue_policy:\n  rest_recommendation_seconds: 40000\n  mandatory_rest_seconds: 36000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(path)
