from __future__ import annotations

import argparse
import sys
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter, sleep

import numpy as np

from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver_safety.config import load_config
from driver_safety.core.models import DriverState, FramePacket, ProcessedFrame
from driver_safety.io.overlay import draw_minimal_overlay_on_frame, draw_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Raspberry Pi 5 realtime scheduling.")
    parser.add_argument("--config", default="configs/raspi5-realtime.yaml")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--analysis-ms", type=float, default=80.0)
    parser.add_argument("--min-display-fps", type=float, default=29.0)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    source_fps = float(config.camera.fps or 30.0)
    display_fps = float(config.runtime.output_fps or source_fps)
    display_interval = 1.0 / display_fps
    analysis_interval = config.vision.process_every_n_frames / source_fps
    analysis_seconds = args.analysis_ms / 1000.0
    width = int(config.camera.width or 320)
    height = int(config.camera.height or 240)
    base_frame = np.zeros((height, width, 3), dtype=np.uint8)
    face_bbox = (width // 3, height // 4, width // 4, height // 3)
    landmarks = [
        (face_bbox[0] + (idx % 18) * max(1, face_bbox[2] // 18), face_bbox[1] + (idx // 18) * 4)
        for idx in range(108)
    ]
    processed = ProcessedFrame(
        packet=FramePacket(frame=base_frame, timestamp=0.0, frame_index=0),
        state=DriverState.ATTENTIVE,
        risk_score=0.12,
        signals={
            "eyes_closed": 0.0,
            "drowsy": 0.0,
            "yawning": 0.0,
            "distracted": 0.0,
            "phone_use": 0.0,
            "dht11_temperature_c": 0.0,
            "dht11_humidity_pct": 0.0,
            "driver_absent_seconds": 0.0,
            "cabin_occupancy": 0.0,
            "driving_hours_today": 0.0,
        },
        events=[],
        latency_ms=args.analysis_ms,
        face_bbox=face_bbox,
        landmarks=landmarks,
    )

    lock = Lock()
    pending_timestamp: float | None = None
    processed_count = 0
    submitted_count = 0
    stop = False

    def worker() -> None:
        nonlocal pending_timestamp, processed_count
        while not stop:
            with lock:
                timestamp = pending_timestamp
                pending_timestamp = None
            if timestamp is None:
                sleep(0.001)
                continue
            sleep(analysis_seconds)
            processed_count += 1

    thread = Thread(target=worker, daemon=True)
    thread.start()

    started = perf_counter()
    last_submitted_at = -analysis_interval
    displayed_count = 0
    while perf_counter() - started < args.duration:
        frame_started = perf_counter()
        elapsed = frame_started - started
        if elapsed - last_submitted_at >= analysis_interval:
            with lock:
                pending_timestamp = elapsed
            submitted_count += 1
            last_submitted_at = elapsed
        frame = base_frame.copy()
        if config.runtime.minimal_overlay:
            draw_minimal_overlay_on_frame(frame, processed)
        else:
            draw_overlay(replace(processed, packet=FramePacket(frame=frame, timestamp=elapsed, frame_index=displayed_count)))
        displayed_count += 1
        sleep(max(0.0, display_interval - (perf_counter() - frame_started)))

    stop = True
    thread.join(timeout=0.5)
    actual_duration = perf_counter() - started
    measured_display_fps = displayed_count / actual_duration
    measured_analysis_fps = processed_count / actual_duration

    print(f"config={args.config}")
    print(f"target_display_fps={display_fps:.2f}")
    print(f"displayed_frames={displayed_count}")
    print(f"processed_frames={processed_count}")
    print(f"submitted_frames={submitted_count}")
    print(f"measured_display_fps={measured_display_fps:.2f}")
    print(f"measured_analysis_fps={measured_analysis_fps:.2f}")
    print(f"simulated_analysis_latency_ms={args.analysis_ms:.1f}")

    if measured_display_fps < args.min_display_fps:
        raise SystemExit(
            f"display FPS {measured_display_fps:.2f} is below required {args.min_display_fps:.2f}"
        )


if __name__ == "__main__":
    main()
