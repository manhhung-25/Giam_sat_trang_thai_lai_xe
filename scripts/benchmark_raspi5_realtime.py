from __future__ import annotations

import argparse
import sys
from pathlib import Path
from threading import Lock, Thread
from time import perf_counter, sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver_safety.config import load_config


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
