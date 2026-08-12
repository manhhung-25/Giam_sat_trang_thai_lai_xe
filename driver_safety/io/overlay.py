from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from driver_safety.core.models import DriverState, ProcessedFrame
from driver_safety.runtime.event_logger import LocalEventLogRow

Array = NDArray[Any]

HUD_BG = (18, 22, 26)
HUD_PANEL = (26, 32, 37)
HUD_BORDER = (70, 82, 90)
TEXT = (242, 245, 247)
MUTED = (188, 198, 204)
TRACK = (51, 58, 63)
MINT = (56, 217, 150)
CYAN = (76, 201, 240)
AMBER = (255, 183, 3)
ORANGE = (255, 122, 61)
ROSE = (255, 59, 92)

STATE_COLORS = {
    DriverState.ATTENTIVE: MINT,
    DriverState.EYES_CLOSED: CYAN,
    DriverState.DROWSY: AMBER,
    DriverState.YAWNING: AMBER,
    DriverState.DISTRACTED: ORANGE,
    DriverState.PHONE_USE: ROSE,
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
    state_color = STATE_COLORS.get(processed.state, TEXT)
    _draw_face_layer(frame, processed, state_color)
    _draw_objects(frame, processed)
    _draw_top_hud(frame, processed, state_color, w)
    _draw_signal_strip(frame, processed, w, h)
    _draw_alert_banner(frame, processed, w, h)
    return cast(Array, frame)


def draw_overlay_with_status_panel(
    processed: ProcessedFrame,
    *,
    recent_events: list[LocalEventLogRow] | None = None,
    panel_width: int = 340,
) -> Array:
    overlay = draw_overlay(processed)
    h, w = overlay.shape[:2]
    panel_width = max(300, panel_width)
    canvas = np.full((h, w + panel_width, 3), HUD_BG, dtype=overlay.dtype)
    canvas[:, :w] = overlay
    _draw_status_panel(canvas, processed, recent_events or [], w, panel_width, h)
    return cast(Array, canvas)


def draw_minimal_overlay(processed: ProcessedFrame) -> Array:
    return draw_minimal_overlay_on_frame(processed.packet.frame, processed)


def draw_minimal_overlay_on_frame(frame: Array, processed: ProcessedFrame) -> Array:
    state_color = STATE_COLORS.get(processed.state, TEXT)
    fps = processed.packet.telemetry.get("display_fps")
    fps_text = f" FPS {fps:.1f}" if fps is not None else ""
    label = f"{_state_label(processed.state)}  RUI RO {processed.risk_score:.2f}{fps_text}"
    _text(frame, label, 10, 24, 0.5, state_color, 2)
    if processed.face_bbox:
        x, y, bw, bh = processed.face_bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), state_color, 1, cv2.LINE_AA)
    for obj in processed.objects[:1]:
        x, y, bw, bh = obj["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), ROSE, 1, cv2.LINE_AA)
    return cast(Array, frame)


def _draw_status_panel(
    frame: Array,
    processed: ProcessedFrame,
    recent_events: list[LocalEventLogRow],
    x0: int,
    width: int,
    height: int,
) -> None:
    cv2.rectangle(frame, (x0, 0), (x0 + width, height), HUD_BG, -1)
    cv2.line(frame, (x0, 0), (x0, height), HUD_BORDER, 1, cv2.LINE_AA)
    pad = 18
    x = x0 + pad
    right = x0 + width - pad
    y = 26

    _text(frame, "HE THONG", x, y, 0.54, TEXT, 2)
    y += 20
    _text(frame, "Du lieu that tu Pi 5", x, y, 0.36, MUTED, 1)
    y += 26

    telemetry = processed.packet.telemetry
    y = _compact_metric_row(frame, "CPU", _fmt_percent(telemetry.get("cpu_percent")), x, right, y)
    y = _compact_metric_row(frame, "RAM", _fmt_ram(telemetry), x, right, y)
    y = _compact_metric_row(
        frame,
        "Chip",
        _fmt_temp(telemetry.get("chip_temperature_c")),
        x,
        right,
        y,
    )
    y += 8

    _text(frame, "HIEN THI", x, y, 0.42, TEXT, 2)
    y += 24
    y = _compact_metric_pair_row(
        frame,
        "Cam",
        _fmt_number(telemetry.get("camera_fps"), "fps"),
        "AI",
        _fmt_number(telemetry.get("analysis_fps"), "fps"),
        x,
        right,
        y,
    )
    y = _compact_metric_pair_row(
        frame,
        "Display",
        _fmt_number(telemetry.get("display_fps"), "fps"),
        "Latency",
        f"{processed.latency_ms:.1f} ms",
        x,
        right,
        y,
    )
    y = _compact_metric_row(
        frame,
        "Alert resp",
        _fmt_ms(telemetry.get("alert_response_ms")),
        x,
        right,
        y,
    )
    y += 8

    _text(frame, "NHAT KY", x, y, 0.42, TEXT, 2)
    y += 24
    if not recent_events:
        _text(frame, "Chua co canh bao", x, min(y, height - 14), 0.36, MUTED, 1)
        return

    row = recent_events[0]
    color = ROSE if row.severity == "critical" else AMBER
    time_label = row.wall_time.split("T")[-1][:8]
    label = _event_signal_label(row.signal)
    y = min(y, height - 20)
    cv2.rectangle(frame, (x, y - 14), (right, min(height - 4, y + 18)), HUD_PANEL, -1)
    cv2.line(frame, (x, y - 14), (x, min(height - 4, y + 18)), color, 3, cv2.LINE_AA)
    _text(frame, f"{time_label} {label} risk {row.risk_score:.2f}", x + 10, y + 5, 0.34, TEXT, 1)


def _compact_metric_row(frame: Array, label: str, value: str, x: int, right: int, y: int) -> int:
    cv2.rectangle(frame, (x, y - 15), (right, y + 9), HUD_PANEL, -1)
    _text(frame, label, x + 10, y, 0.38, MUTED, 1)
    size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
    _text(frame, value, max(x + 114, right - size[0] - 10), y, 0.38, TEXT, 1)
    return y + 30


def _compact_metric_pair_row(
    frame: Array,
    left_label: str,
    left_value: str,
    right_label: str,
    right_value: str,
    x: int,
    right: int,
    y: int,
) -> int:
    mid = x + (right - x) // 2
    cv2.rectangle(frame, (x, y - 15), (right, y + 9), HUD_PANEL, -1)
    _text(frame, left_label, x + 10, y, 0.35, MUTED, 1)
    _text(frame, left_value, x + 78, y, 0.35, TEXT, 1)
    _text(frame, right_label, mid + 8, y, 0.35, MUTED, 1)
    size = cv2.getTextSize(right_value, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    _text(frame, right_value, right - size[0] - 10, y, 0.35, TEXT, 1)
    return y + 30


def _draw_top_hud(
    frame: Array,
    processed: ProcessedFrame,
    state_color: tuple[int, int, int],
    width: int,
) -> None:
    bar_h = 42
    cv2.rectangle(frame, (0, 0), (width, bar_h), HUD_BG, -1)
    cv2.line(frame, (0, bar_h), (width, bar_h), HUD_BORDER, 1, cv2.LINE_AA)
    state = _state_label(processed.state)
    _text(frame, state, 8, 18, 0.46, state_color, 2)

    risk = _clamp(processed.risk_score)
    risk_x = 8
    _text(frame, "rui ro", risk_x, 34, 0.30, MUTED, 1)
    _bar(frame, risk_x + 44, 27, max(50, width - 166), 7, risk, _risk_color(risk))
    _text(frame, f"{risk:.2f}", width - 110, 34, 0.30, TEXT, 1)

    fps = processed.packet.telemetry.get("display_fps")
    fps_text = f"FPS {fps:.1f}" if fps is not None else "FPS --"
    latency = f"{processed.latency_ms:.0f} ms"
    right = f"{fps_text}  {latency}"
    size = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)[0]
    _text(frame, right, max(8, width - size[0] - 8), 18, 0.30, MUTED, 1)


def _draw_signal_strip(frame: Array, processed: ProcessedFrame, width: int, height: int) -> None:
    strip_h = 54
    y0 = max(0, height - strip_h)
    cv2.rectangle(frame, (0, y0), (width, height), HUD_BG, -1)
    cv2.line(frame, (0, y0), (width, y0), HUD_BORDER, 1, cv2.LINE_AA)

    labels = [
        ("mat", "eyes_closed", False),
        ("ngu", "drowsy", False),
        ("ngap", "yawning", False),
        ("tap trung", "distracted", True),
    ]
    col_w = max(1, width // len(labels))
    for idx, (label, signal, invert) in enumerate(labels):
        x = idx * col_w + 6
        raw_value = _clamp(processed.signals.get(signal, 0.0))
        value = 1.0 - raw_value if invert else raw_value
        color = _attention_color(value) if invert else _signal_color(signal, value)
        label_scale = 0.27 if len(label) > 6 else 0.32
        _text(frame, label, x, y0 + 17, label_scale, MUTED, 1)
        _bar(frame, x, y0 + 24, max(22, col_w - 12), 7, value, color)
        if col_w >= 58:
            _text(frame, f"{int(value * 100):02d}", x + col_w - 24, y0 + 17, 0.28, TEXT, 1)


def _draw_face_layer(
    frame: Array,
    processed: ProcessedFrame,
    state_color: tuple[int, int, int],
) -> None:
    if not processed.face_bbox:
        return
    x, y, bw, bh = processed.face_bbox
    if len(processed.landmarks) >= 8:
        pts = np.array([(int(px), int(py)) for px, py in processed.landmarks], dtype=np.int32)
        hull = cv2.convexHull(pts)
        hx, hy, hw, hh = cv2.boundingRect(hull)
        hx = max(0, hx)
        hy = max(0, hy)
        hw = min(frame.shape[1] - hx, hw)
        hh = min(frame.shape[0] - hy, hh)
        if hw > 0 and hh > 0:
            roi = frame[hy : hy + hh, hx : hx + hw]
            overlay = roi.copy()
            cv2.fillConvexPoly(overlay, hull - (hx, hy), state_color)
            cv2.addWeighted(overlay, 0.10, roi, 0.90, 0, roi)
        cv2.polylines(frame, [hull], True, state_color, 1, cv2.LINE_AA)
        step = max(1, len(processed.landmarks) // 68)
        for px, py in processed.landmarks[::step]:
            cv2.circle(frame, (int(px), int(py)), 1, CYAN, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), state_color, 1, cv2.LINE_AA)
    corner = max(10, min(bw, bh) // 5)
    for sx, sy in ((x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)):
        dx = corner if sx == x else -corner
        dy = corner if sy == y else -corner
        cv2.line(frame, (sx, sy), (sx + dx, sy), state_color, 2, cv2.LINE_AA)
        cv2.line(frame, (sx, sy), (sx, sy + dy), state_color, 2, cv2.LINE_AA)


def _draw_objects(frame: Array, processed: ProcessedFrame) -> None:
    for obj in processed.objects[:2]:
        x, y, bw, bh = obj["bbox"]
        label = _object_label(str(obj["label"]))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), ROSE, 2, cv2.LINE_AA)
        _tag(frame, label, x, max(14, y - 6), ROSE)


def _draw_alert_banner(frame: Array, processed: ProcessedFrame, width: int, height: int) -> None:
    critical = next((event for event in processed.events if event.severity.value == "critical"), None)
    event = critical or (processed.events[0] if processed.events else None)
    if event is None:
        return
    banner_h = 24
    y = max(34, height - 54 - banner_h)
    color = ROSE if event.severity.value == "critical" else AMBER
    cv2.rectangle(frame, (8, y), (width - 8, y + banner_h), color, -1)
    _text(frame, _event_message(event.message)[:64], 16, y + 17, 0.36, (18, 22, 26), 1)


def _tag(frame: Array, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    cv2.rectangle(frame, (x, y - 13), (x + size[0] + 8, y + 4), color, -1)
    _text(frame, text, x + 4, y, 0.35, (18, 22, 26), 1)


def _bar(
    frame: Array,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), TRACK, -1)
    cv2.rectangle(frame, (x, y), (x + int(w * _clamp(value)), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), HUD_BORDER, 1, cv2.LINE_AA)


def _text(
    frame: Array,
    text: str,
    x: int,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _state_label(state: DriverState) -> str:
    labels = {
        DriverState.ATTENTIVE: "TAP TRUNG",
        DriverState.EYES_CLOSED: "NHAM MAT",
        DriverState.DROWSY: "BUON NGU",
        DriverState.YAWNING: "NGAP",
        DriverState.DISTRACTED: "MAT TAP TRUNG",
        DriverState.PHONE_USE: "DUNG DIEN THOAI",
    }
    return labels.get(state, state.value.replace("_", " ").upper())


def _object_label(label: str) -> str:
    labels = {
        "phone": "DIEN THOAI",
        "cell phone": "DIEN THOAI",
        "mobile": "DIEN THOAI",
    }
    return labels.get(label.lower(), label.upper())


def _event_message(message: str) -> str:
    replacements = {
        "Phone use detected while driving": "Phat hien dung dien thoai khi lai xe",
        "Distracted: driver attention not trackable": "Mat tap trung: khong theo doi duoc tai xe",
        "Eyes closed beyond configured threshold": "Mat nham qua nguong",
        "Sustained eye closure indicates drowsiness": "Dau hieu buon ngu",
        "Yawn detected from mouth landmarks": "Phat hien ngap",
        "Head pose indicates driver is looking away": "Tai xe nhin lech huong",
        "Cabin climate outside comfort band (DHT11)": "Khoang lai ngoai nguong thoai mai",
    }
    return replacements.get(message, message)


def _event_signal_label(signal: str) -> str:
    labels = {
        "eyes_closed": "NHAM MAT",
        "drowsy": "BUON NGU",
        "yawning": "NGAP",
        "distracted": "MAT TAP TRUNG",
        "phone_use": "DIEN THOAI",
        "cabin_climate_warning": "KHI HAU",
    }
    return labels.get(signal, signal.replace("_", " ").upper())


def _fmt_percent(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f}%"


def _fmt_number(value: float | None, suffix: str) -> str:
    return "--" if value is None else f"{value:.1f} {suffix}"


def _fmt_ms(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f} ms"


def _fmt_temp(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f} C"


def _fmt_ram(telemetry: dict[str, float]) -> str:
    used = telemetry.get("ram_used_mb")
    total = telemetry.get("ram_total_mb")
    percent = telemetry.get("ram_percent")
    if used is None or total is None or percent is None:
        return "--"
    return f"{used:.0f}/{total:.0f} MB {percent:.0f}%"


def _risk_color(value: float) -> tuple[int, int, int]:
    if value < 0.35:
        return MINT
    if value < 0.7:
        return AMBER
    return ROSE


def _signal_color(label: str, value: float) -> tuple[int, int, int]:
    if label == "phone_use" and value >= 0.45:
        return ROSE
    if label == "distracted" and value >= 0.45:
        return ORANGE
    if label in {"drowsy", "yawning"} and value >= 0.45:
        return AMBER
    if label == "eyes_closed" and value >= 0.45:
        return CYAN
    return MINT if value < 0.45 else AMBER


def _attention_color(value: float) -> tuple[int, int, int]:
    if value < 0.35:
        return ROSE
    if value < 0.65:
        return AMBER
    return MINT


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
