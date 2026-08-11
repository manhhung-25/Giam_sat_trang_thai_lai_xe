from pathlib import Path

import cv2
import numpy as np

from driver_safety.config import load_config
from driver_safety.core.models import DriverState, FramePacket
from driver_safety.runtime.runner import analyze_video
from driver_safety.vision.landmarks import FaceObservation
from driver_safety.vision.object_detector import ObjectObservation
from driver_safety.vision.pipeline import DriverSafetyPipeline


def test_analyze_video_writes_artifacts(tmp_path: Path) -> None:
    video = tmp_path / "demo.mp4"
    _write_video(video)
    config = load_config(Path("configs/default.yaml"))
    config.runtime.max_frames = 40
    out = tmp_path / "run"
    artifacts = analyze_video(video, out, config)
    assert artifacts["events"].exists()
    assert artifacts["summary"].exists()
    assert artifacts["html"].exists()
    assert artifacts["csv"].exists()
    assert artifacts["annotated_video"].exists()
    html = artifacts["html"].read_text(encoding="utf-8")
    assert "Drowsiness Monitoring" in html
    assert "DHT11 Temperature & Humidity" in html
    assert "Cabin Safety Monitoring" in html


def test_missing_driver_view_is_labeled_distracted() -> None:
    config = load_config(Path("configs/default.yaml"))
    config.thresholds.missing_face_frames = 1
    pipeline = DriverSafetyPipeline(config, face_detector=_NoFaceDetector())
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    result = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.0, frame_index=0))

    assert result.state == DriverState.DISTRACTED
    assert result.events[0].signal == "distracted"
    assert "attention not trackable" in result.events[0].message


def test_phone_use_is_temporally_gated() -> None:
    config = load_config(Path("configs/default.yaml"))
    config.thresholds.missing_face_frames = 999
    config.thresholds.phone_confidence = 0.5
    config.thresholds.phone_use_frames = 2
    pipeline = DriverSafetyPipeline(
        config,
        face_detector=_NoFaceDetector(),
        object_detector=_PhoneObjectDetector(),
    )
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    first = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.0, frame_index=0))
    second = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.1, frame_index=1))

    assert not [event for event in first.events if event.signal == "phone_use"]
    assert [event for event in second.events if event.signal == "phone_use"]


def test_phone_use_persists_through_short_detector_dropouts() -> None:
    config = load_config(Path("configs/default.yaml"))
    config.thresholds.missing_face_frames = 999
    config.thresholds.phone_confidence = 0.5
    config.thresholds.phone_use_frames = 1
    config.thresholds.phone_hold_frames = 2
    pipeline = DriverSafetyPipeline(
        config,
        face_detector=_NoFaceDetector(),
        object_detector=_OneFramePhoneObjectDetector(),
    )
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    first = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.0, frame_index=0))
    second = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.1, frame_index=1))
    third = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.2, frame_index=2))
    fourth = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.3, frame_index=3))

    assert [event for event in first.events if event.signal == "phone_use"]
    assert [event for event in second.events if event.signal == "phone_use"]
    assert [event for event in third.events if event.signal == "phone_use"]
    assert not [event for event in fourth.events if event.signal == "phone_use"]


def test_cabin_left_behind_alert_when_driver_absent_with_occupant() -> None:
    config = load_config(Path("configs/default.yaml"))
    config.thresholds.missing_face_frames = 1
    config.cabin_safety.driver_absence_alert_seconds = 3600
    pipeline = DriverSafetyPipeline(
        config,
        face_detector=_NoFaceDetector(),
        object_detector=_PassengerObjectDetector(),
    )
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    first = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.0, frame_index=0))
    second = pipeline.process_frame(FramePacket(frame=frame, timestamp=3601.0, frame_index=1))

    assert not [event for event in first.events if event.signal == "cabin_left_behind"]
    assert [event for event in second.events if event.signal == "cabin_left_behind"]


def test_driving_time_alerts_emit_at_4h_and_10h() -> None:
    config = load_config(Path("configs/default.yaml"))
    config.thresholds.missing_face_frames = 999
    pipeline = DriverSafetyPipeline(config, face_detector=_NoFaceDetector())
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    _ = pipeline.process_frame(FramePacket(frame=frame, timestamp=0.0, frame_index=0))
    four_hours = pipeline.process_frame(
        FramePacket(frame=frame, timestamp=4 * 3600 + 1, frame_index=1)
    )
    ten_hours = pipeline.process_frame(
        FramePacket(frame=frame, timestamp=10 * 3600 + 2, frame_index=2)
    )

    assert [event for event in four_hours.events if event.signal == "rest_recommended_4h"]
    assert [event for event in ten_hours.events if event.signal == "mandatory_rest_10h"]


def _write_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 180))
    assert writer.isOpened()
    frame = np.full((180, 320, 3), 28, dtype=np.uint8)
    for _ in range(48):
        writer.write(frame)
    writer.release()


class _NoFaceDetector:
    provider = "test"

    def detect(self, packet: FramePacket) -> list[FaceObservation]:
        return []


class _PhoneObjectDetector:
    provider = "test"

    def detect(self, packet: FramePacket) -> list[ObjectObservation]:
        return [
            ObjectObservation(
                label="cell phone",
                confidence=0.8,
                bbox=(120, 90, 32, 42),
                provider=self.provider,
            )
        ]


class _OneFramePhoneObjectDetector:
    provider = "test"

    def __init__(self) -> None:
        self._calls = 0

    def detect(self, packet: FramePacket) -> list[ObjectObservation]:
        self._calls += 1
        if self._calls > 1:
            return []
        return [
            ObjectObservation(
                label="cell phone",
                confidence=0.8,
                bbox=(120, 90, 32, 42),
                provider=self.provider,
            )
        ]


class _PassengerObjectDetector:
    provider = "test"

    def detect(self, packet: FramePacket) -> list[ObjectObservation]:
        return [
            ObjectObservation(
                label="person",
                confidence=0.91,
                bbox=(20, 30, 70, 120),
                provider=self.provider,
            )
        ]
