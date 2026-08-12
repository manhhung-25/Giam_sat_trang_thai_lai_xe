from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ThresholdConfig:
    eye_aspect_ratio: float = 0.22
    mouth_aspect_ratio: float = 0.5
    head_offset: float = 0.42
    phone_confidence: float = 0.45
    phone_use_frames: int = 2
    phone_hold_frames: int = 12
    missing_face_frames: int = 8
    eye_closed_frames: int = 8
    drowsy_frames: int = 36
    yawn_frames: int = 6
    distracted_frames: int = 12


@dataclass(slots=True)
class VisionConfig:
    provider: str = "auto"
    face_landmarker_model: str = "models/face_landmarker.task"
    fallback_to_haar: bool = True
    process_every_n_frames: int = 1
    draw_landmarks: bool = True


@dataclass(slots=True)
class ObjectDetectorConfig:
    enabled: bool = False
    provider: str = "none"
    model_path: str = "models/driver-objects.onnx"
    labels_path: str = "models/driver-objects.labels"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    process_interval_seconds: float = 0.0
    skin_hand_scan_width: int = 320
    skin_hand_min_ratio: float = 0.18
    phone_labels: list[str] = field(default_factory=lambda: ["cell phone", "phone", "mobile"])


@dataclass(slots=True)
class RuntimeConfig:
    output_fps: float | None = None
    max_frames: int | None = None
    display: bool = False
    write_video: bool = True
    debug_frames: bool = False
    minimal_overlay: bool = False
    alert_cooldown_seconds: float = 2.0


@dataclass(slots=True)
class CameraConfig:
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    buffer_size: int | None = 1
    fourcc: str | None = None
    threaded: bool = False


@dataclass(slots=True)
class DHT11Config:
    enabled: bool = True
    csv_path: str | None = None
    simulate_when_missing: bool = True
    comfort_temp_min_c: float = 20.0
    comfort_temp_max_c: float = 30.0
    comfort_humidity_min_pct: float = 35.0
    comfort_humidity_max_pct: float = 70.0


@dataclass(slots=True)
class FatiguePolicyConfig:
    rest_recommendation_seconds: float = 4 * 60 * 60
    mandatory_rest_seconds: float = 10 * 60 * 60
    initial_driving_seconds_today: float = 0.0


@dataclass(slots=True)
class CabinSafetyConfig:
    driver_absence_alert_seconds: float = 60 * 60
    occupant_labels: list[str] = field(
        default_factory=lambda: ["person", "child", "baby", "passenger"]
    )


@dataclass(slots=True)
class ReportConfig:
    unsafe_threshold: float = 0.45
    timeline_stride_frames: int = 3


@dataclass(slots=True)
class DriverSafetyConfig:
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    object_detector: ObjectDetectorConfig = field(default_factory=ObjectDetectorConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    dht11: DHT11Config = field(default_factory=DHT11Config)
    fatigue_policy: FatiguePolicyConfig = field(default_factory=FatiguePolicyConfig)
    cabin_safety: CabinSafetyConfig = field(default_factory=CabinSafetyConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    signal_weights: dict[str, float] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> DriverSafetyConfig:
    config = DriverSafetyConfig()
    if path is None:
        default_path = Path("configs/default.yaml")
        path = default_path if default_path.exists() else None
    if path is None:
        return config

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    merged = _to_nested_dict(config)
    _deep_update(merged, data)
    parsed = _from_nested_dict(merged)
    _validate_config(parsed)
    return parsed


def _to_nested_dict(config: DriverSafetyConfig) -> dict[str, Any]:
    return asdict(config)


def _from_nested_dict(data: dict[str, Any]) -> DriverSafetyConfig:
    return DriverSafetyConfig(
        thresholds=ThresholdConfig(**data.get("thresholds", {})),
        vision=VisionConfig(**data.get("vision", {})),
        object_detector=ObjectDetectorConfig(**data.get("object_detector", {})),
        runtime=RuntimeConfig(**data.get("runtime", {})),
        camera=CameraConfig(**data.get("camera", {})),
        dht11=DHT11Config(**data.get("dht11", {})),
        fatigue_policy=FatiguePolicyConfig(**data.get("fatigue_policy", {})),
        cabin_safety=CabinSafetyConfig(**data.get("cabin_safety", {})),
        report=ReportConfig(**data.get("report", {})),
        signal_weights=data.get("signal_weights", {}) or {},
    )


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _validate_config(config: DriverSafetyConfig) -> None:
    if config.vision.process_every_n_frames < 1:
        raise ValueError("vision.process_every_n_frames must be >= 1")
    for field_name in ("eye_aspect_ratio", "mouth_aspect_ratio", "head_offset"):
        value = getattr(config.thresholds, field_name)
        if not 0 < value < 1.5:
            raise ValueError(f"thresholds.{field_name} must be within a sensible range")
    if config.runtime.max_frames is not None and config.runtime.max_frames < 1:
        raise ValueError("runtime.max_frames must be >= 1 when set")
    if config.camera.width is not None and config.camera.width < 1:
        raise ValueError("camera.width must be >= 1 when set")
    if config.camera.height is not None and config.camera.height < 1:
        raise ValueError("camera.height must be >= 1 when set")
    if config.camera.fps is not None and config.camera.fps <= 0:
        raise ValueError("camera.fps must be > 0 when set")
    if config.camera.buffer_size is not None and config.camera.buffer_size < 1:
        raise ValueError("camera.buffer_size must be >= 1 when set")
    if config.camera.fourcc is not None and len(config.camera.fourcc) != 4:
        raise ValueError("camera.fourcc must contain exactly 4 characters when set")
    if config.thresholds.phone_use_frames < 1:
        raise ValueError("thresholds.phone_use_frames must be >= 1")
    if config.thresholds.phone_hold_frames < 0:
        raise ValueError("thresholds.phone_hold_frames must be >= 0")
    if not 0 < config.object_detector.confidence_threshold < 1:
        raise ValueError("object_detector.confidence_threshold must be between 0 and 1")
    if not 0 < config.object_detector.iou_threshold < 1:
        raise ValueError("object_detector.iou_threshold must be between 0 and 1")
    if config.object_detector.process_interval_seconds < 0:
        raise ValueError("object_detector.process_interval_seconds must be >= 0")
    if config.object_detector.skin_hand_scan_width < 64:
        raise ValueError("object_detector.skin_hand_scan_width must be >= 64")
    if not 0 < config.object_detector.skin_hand_min_ratio < 1:
        raise ValueError("object_detector.skin_hand_min_ratio must be between 0 and 1")
    if config.fatigue_policy.rest_recommendation_seconds <= 0:
        raise ValueError("fatigue_policy.rest_recommendation_seconds must be > 0")
    if config.fatigue_policy.mandatory_rest_seconds <= 0:
        raise ValueError("fatigue_policy.mandatory_rest_seconds must be > 0")
    if (
        config.fatigue_policy.rest_recommendation_seconds
        >= config.fatigue_policy.mandatory_rest_seconds
    ):
        raise ValueError(
            "fatigue_policy.rest_recommendation_seconds must be lower than mandatory_rest_seconds"
        )
    if config.fatigue_policy.initial_driving_seconds_today < 0:
        raise ValueError("fatigue_policy.initial_driving_seconds_today must be >= 0")
    if config.cabin_safety.driver_absence_alert_seconds <= 0:
        raise ValueError("cabin_safety.driver_absence_alert_seconds must be > 0")
