"""Benchmark FPS and per-stage latency of the driver safety pipeline.

Usage:
    python scripts/benchmark_fps.py [--config configs/raspi5-realtime.yaml] [--video docs/demo/real-human-demo.mp4]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from driver_safety.config import load_config
from driver_safety.core.models import FramePacket
from driver_safety.vision.pipeline import DriverSafetyPipeline


def benchmark(config_path: str, video_path: str, num_frames: int = 200) -> None:
    config = load_config(config_path)
    pipeline = DriverSafetyPipeline(config)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 24.0

    # Warmup
    for _ in range(5):
        ok, frame = cap.read()
        if not ok:
            break
        packet = FramePacket(frame=frame, timestamp=0.0, frame_index=0, source_id="bench", fps=fps)
        pipeline.process_frame(packet)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_start = time.perf_counter()
    processed = 0
    latencies = []
    frame_idx = 0
    while processed < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % config.vision.process_every_n_frames != 0:
            frame_idx += 1
            continue
        packet = FramePacket(
            frame=frame,
            timestamp=frame_idx / fps,
            frame_index=frame_idx,
            source_id="bench",
            fps=fps,
        )
        start = time.perf_counter()
        pipeline.process_frame(packet)
        latencies.append((time.perf_counter() - start) * 1000)
        processed += 1
        frame_idx += 1

    cap.release()
    if not latencies:
        print("No frames processed")
        return

    avg = sum(latencies) / len(latencies)
    latencies_sorted = sorted(latencies)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95) - 1]
    print(f"Config: {config_path}")
    print(f"Video:  {video_path}")
    print(f"Frames processed: {processed}")
    print(f"Avg latency: {avg:.2f} ms")
    print(f"P95 latency: {p95:.2f} ms")
    print(f"Estimated FPS (1/avg): {1000.0 / avg:.2f}")
    print(f"Face provider: {pipeline.face_detector.provider}")
    print(f"Object provider: {pipeline.object_detector.provider}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/raspi5-realtime.yaml")
    parser.add_argument("--video", default="docs/demo/real-human-demo.mp4")
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()
    benchmark(args.config, args.video, args.frames)


if __name__ == "__main__":
    main()