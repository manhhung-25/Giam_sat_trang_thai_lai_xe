from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from driver_safety.core.models import DriverState, ProcessedFrame

Array = NDArray[Any]

STATE_COLORS = {
    DriverState.ATTENTIVE: (76, 190, 118),
    DriverState.EYES_CLOSED: (36, 174, 222),
    DriverState.DROWSY: (38, 64, 230),
    DriverState.YAWNING: (36, 174, 222),
    DriverState.DISTRACTED: (42, 42, 238),
    DriverState.PHONE_USE: (42, 42, 238),
}

class AnnotatedVideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {self.path}")

    def write(self, frame: Array) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def draw_overlay(processed: ProcessedFrame) -> Array:
    frame = processed.packet.frame.copy()
    h, w = frame.shape[:2]
    state_color = STATE_COLORS.get(processed.state, (255, 255, 255))
    _draw_status_panel(frame, processed, state_color)
    _draw_face_mask(frame, processed, state_color)
    for obj in processed.objects:
        x, y, bw, bh = obj["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (55, 65, 235), 2)
        cv2.putText(
            frame,
            str(obj["label"]),
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (55, 65, 235),
            2,
            cv2.LINE_AA,
        )
    _draw_dashboard(frame, processed, w, h)
    for idx, event in enumerate(processed.events[:3]):
        y = h - 90 + idx * 24
        cv2.putText(
            frame,
            event.message,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return cast(Array, frame)


def _draw_panel(frame: Array, x: int, y: int, w: int, h: int, *, alpha: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 22, 23), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (82, 92, 94), 1, cv2.LINE_AA)


def _draw_status_panel(frame: Array, processed: ProcessedFrame, state_color: tuple[int, int, int]) -> None:
    panel_w = min(360, max(230, frame.shape[1] // 2))
    _draw_panel(frame, 14, 14, panel_w, 86, alpha=0.66)
    _draw_brand(frame, 28, 25)
    state = processed.state.value.replace("_", " ").upper()
    cv2.putText(
        frame,
        state,
        (28, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        state_color,
        2,
        cv2.LINE_AA,
    )
    risk = min(1.0, max(0.0, processed.risk_score))
    cv2.putText(frame, "RISK", (28, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (208, 216, 216), 1, cv2.LINE_AA)
    _bar(frame, 72, 68, panel_w - 176, 10, risk, _risk_color(risk))
    cv2.putText(
        frame,
        f"{risk:.2f}",
        (panel_w - 88, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (236, 242, 242),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{processed.latency_ms:.1f} ms",
        (panel_w - 46, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (176, 190, 190),
        1,
        cv2.LINE_AA,
    )


def _draw_brand(frame: Array, x: int, y: int) -> None:
    # Branding removed per user request. No overlay logo or text will be drawn.
    return


def _draw_face_mask(frame: Array, processed: ProcessedFrame, state_color: tuple[int, int, int]) -> None:
    if not processed.face_bbox:
        return
    x, y, bw, bh = processed.face_bbox
    overlay = frame.copy()
    if len(processed.landmarks) >= 8:
        pts = np.array([(int(px), int(py)) for px, py in processed.landmarks], dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(overlay, hull, state_color)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, [hull], True, state_color, 1, cv2.LINE_AA)
        step = max(1, len(processed.landmarks) // 54)
        for px, py in processed.landmarks[::step]:
            cv2.circle(frame, (int(px), int(py)), 1, (135, 232, 255), -1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), state_color, 1, cv2.LINE_AA)
    corner = max(12, min(bw, bh) // 5)
    thickness = 2
    for sx, sy in ((x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)):
        dx = corner if sx == x else -corner
        dy = corner if sy == y else -corner
        cv2.line(frame, (sx, sy), (sx + dx, sy), state_color, thickness, cv2.LINE_AA)
        cv2.line(frame, (sx, sy), (sx, sy + dy), state_color, thickness, cv2.LINE_AA)


def _draw_dashboard(frame: Array, processed: ProcessedFrame, width: int, height: int) -> None:
    panel_w = min(282, max(214, int(width * 0.34)))
    start_x = max(12, width - panel_w - 18)
    start_y = 16
    panel_h = min(height - 32, 430)
    _draw_panel(frame, start_x, start_y, panel_w, panel_h, alpha=0.66)
    inner_x = start_x + 16
    right_x = start_x + panel_w - 16
    y = start_y + 26
    cv2.putText(
        frame,
        "DRIVER MONITOR",
        (inner_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (236, 242, 242),
        1,
        cv2.LINE_AA,
    )
    y += 20
    labels = ["eyes_closed", "drowsy", "yawning", "distracted", "phone_use"]
    for label in labels:
        value = min(1.0, max(0.0, processed.signals.get(label, 0.0)))
        readable = label.replace("_", " ")
        cv2.putText(
            frame,
            readable,
            (inner_x, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (216, 222, 222),
            1,
            cv2.LINE_AA,
        )
        bar_x = inner_x + 94
        _bar(
            frame,
            bar_x,
            y,
            max(56, right_x - bar_x - 36),
            10,
            value,
            _signal_color(label, value),
        )
        cv2.putText(
            frame,
            f"{int(value * 100):02d}",
            (right_x - 28, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (190, 202, 202),
            1,
            cv2.LINE_AA,
        )
        y += 25
    y += 12
    _section_rule(frame, inner_x, right_x, y)
    y += 22
    _draw_climate_rows(frame, processed, inner_x, right_x, y)
    y += 98
    _section_rule(frame, inner_x, right_x, y)
    y += 20
    _draw_cabin_rows(frame, processed, inner_x, y)


def _draw_climate_rows(frame: Array, processed: ProcessedFrame, x: int, right_x: int, y: int) -> None:
    temp = processed.signals.get("dht11_temperature_c", 0.0)
    humidity = processed.signals.get("dht11_humidity_pct", 0.0)
    cv2.putText(
        frame,
        "CABIN CLIMATE",
        (x, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (236, 242, 242),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{temp:.1f} C",
        (x, y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (112, 218, 252),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"{humidity:.1f} %",
        (right_x - 92, y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (122, 236, 189),
        2,
        cv2.LINE_AA,
    )
    _meter(frame, x, y + 52, "temp", min(1.0, max(0.0, temp / 45.0)))
    _meter(frame, x, y + 72, "humid", min(1.0, max(0.0, humidity / 100.0)))


def _draw_cabin_rows(frame: Array, processed: ProcessedFrame, x: int, y: int) -> None:
    absent_minutes = max(0.0, processed.signals.get("driver_absent_seconds", 0.0)) / 60.0
    occupied = processed.signals.get("cabin_occupancy", 0.0) >= 0.5
    drive_hours = max(0.0, processed.signals.get("driving_hours_today", 0.0))
    cv2.putText(
        frame,
        "CABIN SAFETY",
        (x, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (236, 242, 242),
        1,
        cv2.LINE_AA,
    )
    rows = [
        (f"Driver absent  {absent_minutes:.1f} min", (242, 210, 116)),
        (
            f"Occupant       {'YES' if occupied else 'NO'}",
            (103, 232, 168) if occupied else (180, 193, 193),
        ),
        (f"Driving today  {drive_hours:.2f} h", (231, 231, 231)),
    ]
    for idx, (text, color) in enumerate(rows):
        cv2.putText(
            frame,
            text,
            (x, y + 26 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _section_rule(frame: Array, x: int, right_x: int, y: int) -> None:
    cv2.line(frame, (x, y), (right_x, y), (82, 92, 94), 1, cv2.LINE_AA)


def _bar(
    frame: Array,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (52, 58, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * value), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (88, 98, 100), 1, cv2.LINE_AA)


def _risk_color(value: float) -> tuple[int, int, int]:
    if value < 0.35:
        return (62, 197, 124)
    if value < 0.7:
        return (46, 167, 235)
    return (42, 42, 238)


def _meter(frame: Array, x: int, y: int, label: str, value: float) -> None:
    cv2.putText(
        frame,
        label,
        (x, y + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (208, 216, 216),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (x + 72, y), (x + 190, y + 10), (60, 66, 68), -1)
    cv2.rectangle(
        frame,
        (x + 72, y),
        (x + 72 + int(118 * value), y + 10),
        (74, 188, 220),
        -1,
    )


def _signal_color(label: str, value: float) -> tuple[int, int, int]:
    if value < 0.45:
        return (62, 197, 124)
    if label == "distracted":
        return (42, 42, 238)
    return (46, 167, 235)
