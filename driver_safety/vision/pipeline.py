from __future__ import annotations

from time import perf_counter

import cv2

from driver_safety.config import DriverSafetyConfig
from driver_safety.core.alerts import AlertPolicy
from driver_safety.core.models import (
    DetectionEvent,
    DriverState,
    FramePacket,
    ProcessedFrame,
    Severity,
)
from driver_safety.core.scoring import RiskScorer
from driver_safety.core.smoothing import SignalSmoother
from driver_safety.vision.landmarks import FaceLandmarkDetector, create_face_detector
from driver_safety.vision.metrics import (
    eye_aspect_ratio,
    horizontal_head_offset,
    mouth_aspect_ratio,
)
from driver_safety.vision.object_detector import (
    ObjectDetector,
    ObjectObservation,
    create_object_detector,
)


class DriverSafetyPipeline:
    def __init__(
        self,
        config: DriverSafetyConfig,
        *,
        face_detector: FaceLandmarkDetector | None = None,
        object_detector: ObjectDetector | None = None,
    ) -> None:
        self.config = config
        self.face_detector = face_detector or create_face_detector(config)
        self.object_detector = object_detector or create_object_detector(config)
        self.smoother = SignalSmoother(window_size=5)
        self.scorer = RiskScorer(config.signal_weights or None)
        self.alert_policy = AlertPolicy(config.runtime.alert_cooldown_seconds)
        self._closed_counter = 0
        self._yawn_counter = 0
        self._distracted_counter = 0
        self._missing_face_counter = 0
        self._phone_counter = 0
        self._phone_hold_counter = 0
        self._last_phone: ObjectObservation | None = None
        self._driver_absent_started_at: float | None = None
        self._child_left_behind_triggered = False
        self._rest_recommendation_issued = False
        self._mandatory_rest_issued = False
        self._last_timestamp: float | None = None
        self._last_object_detection_at: float | None = None
        self._cached_object_observations: list[ObjectObservation] = []
        self._driving_seconds_today = config.fatigue_policy.initial_driving_seconds_today
        self._last_climate_alert_at: float | None = None

    def process_frame(self, packet: FramePacket) -> ProcessedFrame:
        started = perf_counter()
        self._update_driving_clock(packet.timestamp)
        raw_signals = {
            "eyes_closed": 0.0,
            "drowsy": 0.0,
            "yawning": 0.0,
            "distracted": 0.0,
            "phone_use": 0.0,
            "driving_fatigue": min(
                1.0,
                self._driving_seconds_today
                / max(1.0, self.config.fatigue_policy.mandatory_rest_seconds),
            ),
            "cabin_occupant_risk": 0.0,
            "driver_absent_seconds": 0.0,
            "driving_hours_today": self._driving_seconds_today / 3600.0,
            "cabin_occupancy": 0.0,
            "dht11_temperature_c": 0.0,
            "dht11_humidity_pct": 0.0,
        }
        events: list[DetectionEvent] = []
        face_bbox: tuple[int, int, int, int] | None = None
        landmarks: list[tuple[float, float]] = []
        face_observations = self.face_detector.detect(packet)

        if not face_observations:
            self._missing_face_counter += 1
            if self._missing_face_counter >= self.config.thresholds.missing_face_frames:
                raw_signals["distracted"] = 1.0
                events.append(
                    self._event(
                        packet,
                        "distracted",
                        DriverState.DISTRACTED,
                        1.0,
                        Severity.WARNING,
                        "Distracted: driver attention not trackable",
                    )
                )
        else:
            self._missing_face_counter = 0
            observation = face_observations[0]
            face_bbox = observation.bbox
            landmarks = observation.all_points
            raw_signals.update(self._face_signals(packet, observation.bbox, observation.landmarks))
            events.extend(self._events_from_signals(packet, raw_signals, face_bbox, landmarks))

        object_observations = self._object_observations(packet, face_bbox)
        phone_labels = {label.lower() for label in self.config.object_detector.phone_labels}
        occupant_labels = {label.lower() for label in self.config.cabin_safety.occupant_labels}
        best_phone = None
        display_objects: list[ObjectObservation] = []
        occupant_observations: list[ObjectObservation] = []
        for obj in object_observations:
            if (
                obj.label.lower() in phone_labels
                and obj.confidence >= self.config.thresholds.phone_confidence
                and (best_phone is None or obj.confidence > best_phone.confidence)
            ):
                best_phone = obj
            if obj.label.lower() in occupant_labels:
                occupant_observations.append(obj)

        if best_phone is None:
            if self._phone_hold_counter > 0 and self._last_phone is not None:
                self._phone_hold_counter -= 1
                best_phone = self._last_phone
            else:
                self._phone_counter = 0
                self._last_phone = None
        else:
            self._last_phone = best_phone
            self._phone_hold_counter = self.config.thresholds.phone_hold_frames

        if best_phone is not None:
            self._phone_counter += 1
            phone_gate = min(
                1.0, self._phone_counter / max(1, self.config.thresholds.phone_use_frames)
            )
            raw_signals["phone_use"] = best_phone.confidence * phone_gate
            if self._phone_counter >= self.config.thresholds.phone_use_frames:
                events.append(
                    self._event(
                        packet,
                        "phone_use",
                        DriverState.PHONE_USE,
                        raw_signals["phone_use"],
                        Severity.CRITICAL,
                        "Phone use detected while driving",
                        bbox=best_phone.bbox,
                        metadata={
                            "label": best_phone.label,
                            "provider": best_phone.provider,
                            "tracking": "raw" if best_phone in object_observations else "held",
                        },
                    )
                )
                display_objects.append(best_phone)

        occupant_present = bool(occupant_observations) or len(face_observations) > 1
        raw_signals["cabin_occupancy"] = 1.0 if occupant_present else 0.0
        if not face_observations:
            if self._driver_absent_started_at is None:
                self._driver_absent_started_at = packet.timestamp
            driver_absent_seconds = max(0.0, packet.timestamp - self._driver_absent_started_at)
            raw_signals["driver_absent_seconds"] = driver_absent_seconds
            if (
                occupant_present
                and driver_absent_seconds >= self.config.cabin_safety.driver_absence_alert_seconds
            ):
                raw_signals["cabin_occupant_risk"] = 1.0
                if not self._child_left_behind_triggered:
                    events.append(
                        self._event(
                            packet,
                            "cabin_left_behind",
                            DriverState.DISTRACTED,
                            1.0,
                            Severity.CRITICAL,
                            "Critical: driver absent >60 minutes while person/child remains in cabin",
                            metadata={
                                "driver_absent_seconds": round(driver_absent_seconds, 2),
                                "occupant_count": float(len(occupant_observations)),
                            },
                        )
                    )
                    self._child_left_behind_triggered = True
        else:
            self._driver_absent_started_at = None
            self._child_left_behind_triggered = False

        events.extend(self._driving_time_events(packet))
        events.extend(self._climate_events(packet, raw_signals))
        if not face_observations:
            display_objects.extend(occupant_observations[:2])

        signals = self.smoother.update(raw_signals)
        risk_score = self.scorer.score(signals)
        state = self.scorer.state_from_events(events, risk_score)
        latency_ms = (perf_counter() - started) * 1000
        # infer driver_position from face bbox (left/center/right) for overlay and downstream rules
        driver_position: str | None = None
        if face_bbox is not None:
            fx, fy, fbw, fbh = face_bbox
            center_x = fx + fbw / 2.0
            frame_w = float(packet.frame.shape[1])
            if center_x < frame_w / 3.0:
                driver_position = "left"
            elif center_x > 2.0 * frame_w / 3.0:
                driver_position = "right"
            else:
                driver_position = "center"

        return ProcessedFrame(
            packet=packet,
            state=state,
            risk_score=risk_score,
            signals=signals,
            events=events,
            latency_ms=latency_ms,
            face_bbox=face_bbox,
            landmarks=landmarks,
            objects=[obj.to_dict() for obj in display_objects],
            driver_position=driver_position,
        )

    def _object_observations(
        self,
        packet: FramePacket,
        face_bbox: tuple[int, int, int, int] | None,
    ) -> list[ObjectObservation]:
        interval = self.config.object_detector.process_interval_seconds
        should_detect = (
            interval <= 0
            or self._last_object_detection_at is None
            or packet.timestamp - self._last_object_detection_at >= interval
        )
        if should_detect:
            if self.config.object_detector.provider == "skin_hand":
                self._cached_object_observations = self._skin_hand_phone_observations(
                    packet,
                    face_bbox,
                )
            else:
                self._cached_object_observations = self.object_detector.detect(packet)
            self._last_object_detection_at = packet.timestamp
        return self._cached_object_observations

    def _skin_hand_phone_observations(
        self,
        packet: FramePacket,
        face_bbox: tuple[int, int, int, int] | None,
    ) -> list[ObjectObservation]:
        if not self.config.object_detector.enabled or face_bbox is None:
            return []

        frame = packet.frame
        frame_h, frame_w = frame.shape[:2]
        fx, fy, fw, fh = face_bbox
        if fw <= 0 or fh <= 0:
            return []
        observations: list[ObjectObservation] = []
        occluding_phone = _phone_occlusion_observation(frame, face_bbox)
        if occluding_phone is not None:
            observations.append(occluding_phone)

        zone_w = max(12, int(fw * 0.42))
        y1 = max(0, fy + int(fh * 0.08))
        y2 = min(frame_h, fy + int(fh * 0.62))
        zones = [
            (max(0, fx - zone_w), y1, max(0, fx - zone_w) + zone_w, y2),
            (min(frame_w, fx + fw), y1, min(frame_w, fx + fw + zone_w), y2),
        ]

        scale = min(1.0, self.config.object_detector.skin_hand_scan_width / max(1, frame_w))
        small = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

        best: tuple[float, tuple[int, int, int, int]] | None = None
        for x1, zy1, x2, zy2 in zones:
            if x2 <= x1 or zy2 <= zy1:
                continue
            sx1, sy1, sx2, sy2 = [int(value * scale) for value in (x1, zy1, x2, zy2)]
            roi = mask[sy1:sy2, sx1:sx2]
            if roi.size == 0:
                continue
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            ratio = area / float(roi.size)
            rx, ry, rw, rh = cv2.boundingRect(contour)
            aspect = rw / max(1.0, float(rh))
            if ratio > 0.55 or not 0.25 <= aspect <= 1.8:
                continue
            if best is None or ratio > best[0]:
                best = (ratio, (x1, zy1, x2 - x1, zy2 - zy1))

        if best is None:
            return observations
        ratio, bbox = best
        if ratio < self.config.object_detector.skin_hand_min_ratio:
            return observations
        confidence = min(
            0.68,
            0.45 + 0.12 * ratio / max(self.config.object_detector.skin_hand_min_ratio, 1e-6),
        )
        observations.append(
            ObjectObservation(
                label="phone",
                confidence=confidence,
                bbox=bbox,
                provider="skin_hand",
            )
        )
        return observations

    def _update_driving_clock(self, timestamp: float) -> None:
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            return
        delta = max(0.0, timestamp - self._last_timestamp)
        self._driving_seconds_today += delta
        self._last_timestamp = timestamp

    def _driving_time_events(self, packet: FramePacket) -> list[DetectionEvent]:
        events: list[DetectionEvent] = []
        fatigue = self.config.fatigue_policy
        if (
            self._driving_seconds_today >= fatigue.rest_recommendation_seconds
            and not self._rest_recommendation_issued
        ):
            events.append(
                self._event(
                    packet,
                    "rest_recommended_4h",
                    DriverState.DROWSY,
                    min(1.0, self._driving_seconds_today / fatigue.mandatory_rest_seconds),
                    Severity.WARNING,
                    "Driving time exceeded 4 hours: recommend rest break",
                    metadata={"driving_hours_today": round(self._driving_seconds_today / 3600.0, 3)},
                )
            )
            self._rest_recommendation_issued = True
        if self._driving_seconds_today >= fatigue.mandatory_rest_seconds and not self._mandatory_rest_issued:
            events.append(
                self._event(
                    packet,
                    "mandatory_rest_10h",
                    DriverState.DROWSY,
                    1.0,
                    Severity.CRITICAL,
                    "Critical: driving time exceeded 10 hours/day. Driver must rest now",
                    metadata={"driving_hours_today": round(self._driving_seconds_today / 3600.0, 3)},
                )
            )
            self._mandatory_rest_issued = True
        return events

    def _climate_events(
        self,
        packet: FramePacket,
        signals: dict[str, float],
    ) -> list[DetectionEvent]:
        events: list[DetectionEvent] = []
        temperature = packet.telemetry.get("dht11_temperature_c")
        humidity = packet.telemetry.get("dht11_humidity_pct")
        if temperature is None or humidity is None:
            return events
        signals["dht11_temperature_c"] = float(temperature)
        signals["dht11_humidity_pct"] = float(humidity)
        dht11 = self.config.dht11
        out_of_comfort = (
            temperature < dht11.comfort_temp_min_c
            or temperature > dht11.comfort_temp_max_c
            or humidity < dht11.comfort_humidity_min_pct
            or humidity > dht11.comfort_humidity_max_pct
        )
        if not out_of_comfort:
            return events
        if self._last_climate_alert_at is not None and packet.timestamp - self._last_climate_alert_at < 300:
            return events
        self._last_climate_alert_at = packet.timestamp
        events.append(
            self._event(
                packet,
                "cabin_climate_warning",
                DriverState.DISTRACTED,
                0.7,
                Severity.WARNING,
                "Cabin climate outside comfort band (DHT11)",
                metadata={"temperature_c": float(temperature), "humidity_pct": float(humidity)},
            )
        )
        return events

    def _face_signals(
        self,
        packet: FramePacket,
        bbox: tuple[int, int, int, int],
        landmarks: dict[str, list[tuple[float, float]]],
    ) -> dict[str, float]:
        thresholds = self.config.thresholds
        left_ear = eye_aspect_ratio(landmarks["left_eye"])
        right_ear = eye_aspect_ratio(landmarks["right_eye"])
        ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(landmarks["mouth"])
        head_offset = horizontal_head_offset(bbox, packet.frame.shape[1])
        head_pose_offset = _head_pose_offset(bbox, landmarks.get("pose", []))
        looking_away = (
            head_offset > thresholds.head_offset
            or head_pose_offset > thresholds.head_pose_offset
        )

        self._closed_counter = self._closed_counter + 1 if ear < thresholds.eye_aspect_ratio else 0
        self._yawn_counter = self._yawn_counter + 1 if mar > thresholds.mouth_aspect_ratio else 0
        self._distracted_counter = (
            self._distracted_counter + 1 if looking_away else 0
        )

        return {
            "eyes_closed": min(1.0, self._closed_counter / max(1, thresholds.eye_closed_frames)),
            "drowsy": min(1.0, self._closed_counter / max(1, thresholds.drowsy_frames)),
            "yawning": min(1.0, self._yawn_counter / max(1, thresholds.yawn_frames)),
            "distracted": min(1.0, self._distracted_counter / max(1, thresholds.distracted_frames)),
            "phone_use": 0.0,
            "ear": round(ear, 4),
            "mar": round(mar, 4),
            "head_offset": round(head_offset, 4),
            "head_pose_offset": round(head_pose_offset, 4),
        }

    def _events_from_signals(
        self,
        packet: FramePacket,
        signals: dict[str, float],
        bbox: tuple[int, int, int, int],
        landmarks: list[tuple[float, float]],
    ) -> list[DetectionEvent]:
        events: list[DetectionEvent] = []
        thresholds = self.config.thresholds
        if self._closed_counter >= thresholds.eye_closed_frames:
            events.append(
                self._event(
                    packet,
                    "eyes_closed",
                    DriverState.EYES_CLOSED,
                    signals["eyes_closed"],
                    Severity.WARNING,
                    "Eyes closed beyond configured threshold",
                    bbox=bbox,
                    landmarks=landmarks,
                )
            )
        if self._closed_counter >= thresholds.drowsy_frames:
            events.append(
                self._event(
                    packet,
                    "drowsy",
                    DriverState.DROWSY,
                    signals["drowsy"],
                    Severity.CRITICAL,
                    "Sustained eye closure indicates drowsiness",
                    bbox=bbox,
                    landmarks=landmarks,
                )
            )
        if self._yawn_counter >= thresholds.yawn_frames:
            events.append(
                self._event(
                    packet,
                    "yawning",
                    DriverState.YAWNING,
                    signals["yawning"],
                    Severity.WARNING,
                    "Yawn detected from mouth landmarks",
                    bbox=bbox,
                    landmarks=landmarks,
                )
            )
        if self._distracted_counter >= thresholds.distracted_frames:
            events.append(
                self._event(
                    packet,
                    "distracted",
                    DriverState.DISTRACTED,
                    signals["distracted"],
                    Severity.WARNING,
                    "Head pose indicates driver is looking away",
                    bbox=bbox,
                    landmarks=landmarks,
                )
            )
        return events

    def _event(
        self,
        packet: FramePacket,
        signal: str,
        state: DriverState,
        score: float,
        severity: Severity,
        message: str,
        *,
        bbox: tuple[int, int, int, int] | None = None,
        landmarks: list[tuple[float, float]] | None = None,
        metadata: dict[str, float | str] | None = None,
    ) -> DetectionEvent:
        return DetectionEvent(
            timestamp=packet.timestamp,
            frame_index=packet.frame_index,
            signal=signal,
            state=state,
            score=round(float(score), 4),
            severity=severity,
            message=message,
            bbox=bbox,
            landmarks=landmarks or [],
            metadata=metadata or {},
        )


def create_pipeline(config: DriverSafetyConfig | None = None) -> DriverSafetyPipeline:
    return DriverSafetyPipeline(config or DriverSafetyConfig())


def _head_pose_offset(
    face_bbox: tuple[int, int, int, int],
    pose_landmarks: list[tuple[float, float]],
) -> float:
    if not pose_landmarks:
        return 0.0
    fx, fy, fw, fh = face_bbox
    if fw <= 0 or fh <= 0:
        return 0.0
    nose_x, nose_y = pose_landmarks[0]
    x_offset = abs((nose_x - (fx + fw * 0.5)) / fw)
    y_offset = abs((nose_y - (fy + fh * 0.52)) / fh)
    return float(max(x_offset, y_offset))


def _phone_occlusion_observation(
    frame,
    face_bbox: tuple[int, int, int, int],
) -> ObjectObservation | None:
    frame_h, frame_w = frame.shape[:2]
    fx, fy, fw, fh = face_bbox
    x1 = max(0, fx - int(fw * 0.25))
    x2 = min(frame_w, fx + fw + int(fw * 0.25))
    y1 = max(0, fy + int(fh * 0.34))
    y2 = min(frame_h, fy + int(fh * 1.05))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    dark_mask = cv2.inRange(gray, 0, 82)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    face_area = float(fw * fh)
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue
        cx, cy, cw, ch = cv2.boundingRect(contour)
        bx, by = x1 + cx, y1 + cy
        area_ratio = area / max(1.0, face_area)
        if not 0.025 <= area_ratio <= 0.42:
            continue
        aspect = cw / max(1.0, float(ch))
        if not 0.25 <= aspect <= 2.4:
            continue
        if cw < fw * 0.14 and ch < fh * 0.22:
            continue

        ix1 = max(bx, fx)
        iy1 = max(by, fy)
        ix2 = min(bx + cw, fx + fw)
        iy2 = min(by + ch, fy + fh)
        intersects_face = ix2 > ix1 and iy2 > iy1
        center_x = bx + cw * 0.5
        center_y = by + ch * 0.5
        lower_face = center_y > fy + fh * 0.38
        side_or_face_overlap = (
            intersects_face
            or center_x < fx + fw * 0.20
            or center_x > fx + fw * 0.80
        )
        if not lower_face or not side_or_face_overlap:
            continue

        score = min(0.99, 0.72 + min(0.24, area_ratio) / 0.24 * 0.27)
        if best is None or score > best[0]:
            best = (score, (bx, by, cw, ch))

    if best is None:
        return None
    confidence, bbox = best
    return ObjectObservation(
        label="phone",
        confidence=confidence,
        bbox=bbox,
        provider="face_occlusion",
    )
