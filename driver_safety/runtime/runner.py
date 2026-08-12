from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, quantiles

import cv2

from driver_safety.config import DriverSafetyConfig
from driver_safety.core.scoring import RiskScorer
from driver_safety.io.overlay import AnnotatedVideoWriter, draw_minimal_overlay, draw_overlay
from driver_safety.io.sources import VideoFrameSource, WebcamFrameSource
from driver_safety.reporting.exports import export_run_artifacts
from driver_safety.reporting.recorder import SessionRecorder
from driver_safety.runtime.audio_alerts import AudioAlertPlayer
from driver_safety.runtime.dht11 import DHT11Reader
from driver_safety.vision.pipeline import DriverSafetyPipeline


def analyze_video(
    video_path: str | Path,
    output_dir: str | Path,
    config: DriverSafetyConfig,
) -> dict[str, Path]:
    source = VideoFrameSource(video_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pipeline = DriverSafetyPipeline(config)
    recorder = SessionRecorder(output)
    dht11_reader = DHT11Reader(config.dht11)
    writer: AnnotatedVideoWriter | None = None
    if config.runtime.write_video:
        output_fps = config.runtime.output_fps or source.fps
        writer = AnnotatedVideoWriter(
            output / "annotated.mp4", output_fps, (source.width, source.height)
        )
    processed_frames = 0
    last_timestamp = 0.0
    try:
        for packet in source:
            if config.runtime.max_frames and processed_frames >= config.runtime.max_frames:
                break
            if packet.frame_index % config.vision.process_every_n_frames != 0:
                continue
            packet.telemetry.update(dht11_reader.read(packet.timestamp))
            processed = pipeline.process_frame(packet)
            last_timestamp = packet.timestamp
            processed_frames += 1
            recorder.write_frame_score(packet.timestamp, processed.risk_score, processed.latency_ms)
            if {"dht11_temperature_c", "dht11_humidity_pct"}.issubset(packet.telemetry):
                recorder.write_dht11(
                    packet.timestamp,
                    packet.telemetry["dht11_temperature_c"],
                    packet.telemetry["dht11_humidity_pct"],
                )
            for event in processed.events:
                recorder.write_event(event)
            if writer:
                writer.write(draw_overlay(processed))
    finally:
        source.close()
        if writer:
            writer.close()

    summary = RiskScorer(config.signal_weights or None).summarize(
        session_id=_session_id(video_path),
        source=str(video_path),
        duration_seconds=last_timestamp,
        processed_frames=processed_frames,
        events=recorder.events,
        frame_scores=recorder.frame_scores,
        metrics=_metrics(source.fps, recorder.latencies_ms, pipeline),
        dht11_timeline=recorder.dht11_timeline,
    )
    artifacts = export_run_artifacts(output, events=recorder.events, summary=summary)
    if writer:
        artifacts["annotated_video"] = output / "annotated.mp4"
    return artifacts


def run_webcam(config: DriverSafetyConfig, index: int = 0) -> None:
    source = WebcamFrameSource(
        index,
        width=config.camera.width,
        height=config.camera.height,
        fps=config.camera.fps,
        buffer_size=config.camera.buffer_size,
        fourcc=config.camera.fourcc,
        threaded=config.camera.threaded,
    )
    pipeline = DriverSafetyPipeline(config)
    dht11_reader = DHT11Reader(config.dht11)
    audio_player = AudioAlertPlayer()
    min_process_interval = (
        config.vision.process_every_n_frames / max(1.0, float(config.camera.fps or source.fps or 30.0))
    )
    last_processed_at = -min_process_interval
    try:
        while True:
<<<<<<< HEAD
            packet = next(source)
            if packet.frame_index % config.vision.process_every_n_frames != 0:
=======
            packet = source.latest()
            if packet.timestamp - last_processed_at < min_process_interval:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
>>>>>>> 6002e911f6036a4aabb6727467e1230de6a5a07a
                continue
            last_processed_at = packet.timestamp
            packet.telemetry.update(dht11_reader.read(packet.timestamp))
            processed = pipeline.process_frame(packet)
            audio_player.handle_events(processed.events, now=packet.timestamp)
            if config.runtime.debug_frames:
                try:
                    events_signals = [e.signal for e in processed.events]
                    objects = [
                        (o["label"], round(float(o["confidence"]), 3), tuple(o["bbox"]))
                        for o in processed.objects
                    ]
                    phone_val = processed.signals.get("phone_use", 0.0)
                    distracted_val = processed.signals.get("distracted", 0.0)
                    print(
                        f"[frame {packet.frame_index}] ts={packet.timestamp:.3f} "
                        f"pos={processed.driver_position} risk={processed.risk_score:.2f} "
                        f"events={events_signals} objects={objects} phone={phone_val:.3f} "
                        f"distracted={distracted_val:.3f}",
                        flush=True,
                    )
                except Exception:
                    pass

            frame = draw_minimal_overlay(processed) if config.runtime.minimal_overlay else draw_overlay(processed)
            cv2.imshow("AI Driver Safety", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        audio_player.close()
        source.close()
        cv2.destroyAllWindows()


def _session_id(video_path: str | Path) -> str:
    stem = Path(video_path).stem
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stem}-{timestamp}"


def _metrics(
    source_fps: float,
    latencies: list[float],
    pipeline: DriverSafetyPipeline,
) -> dict[str, float | int | str]:
    if not latencies:
        return {
            "source_fps": source_fps,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "face_provider": pipeline.face_detector.provider,
            "object_provider": pipeline.object_detector.provider,
        }
    p95 = quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    avg_latency = mean(latencies)
    return {
        "source_fps": round(source_fps, 2),
        "avg_latency_ms": round(avg_latency, 3),
        "p95_latency_ms": round(p95, 3),
        "estimated_runtime_fps": round(1000.0 / avg_latency, 2) if avg_latency else 0.0,
        "face_provider": pipeline.face_detector.provider,
        "object_provider": pipeline.object_detector.provider,
    }
