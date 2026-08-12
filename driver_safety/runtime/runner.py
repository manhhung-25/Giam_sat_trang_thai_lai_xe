from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, quantiles
from threading import Lock, Thread
from time import perf_counter, sleep

import cv2

from driver_safety.config import DriverSafetyConfig
from driver_safety.core.models import FramePacket, ProcessedFrame
from driver_safety.core.scoring import RiskScorer
from driver_safety.io.overlay import (
    AnnotatedVideoWriter,
    draw_minimal_overlay_on_frame,
    draw_overlay,
    draw_overlay_with_status_panel,
)
from driver_safety.io.sources import VideoFrameSource, WebcamFrameSource
from driver_safety.reporting.exports import export_run_artifacts
from driver_safety.reporting.recorder import SessionRecorder
from driver_safety.runtime.audio_alerts import AudioAlertPlayer
from driver_safety.runtime.dht11 import DHT11Reader
from driver_safety.runtime.event_logger import LocalEventLogger
from driver_safety.runtime.gpio_actuators import GpioAlertActuator
from driver_safety.runtime.system_metrics import SystemMetricsReader
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
    metrics_reader = SystemMetricsReader()
    audio_player = AudioAlertPlayer()
    gpio_actuator = GpioAlertActuator(config.actuators, verbose=config.actuators.enabled)
    run_dir = Path("runs/realtime") / datetime.now().strftime("%Y%m%d-%H%M%S")
    event_logger = LocalEventLogger(run_dir)
    source_fps = float(config.camera.fps or source.fps or 30.0)
    target_display_fps = float(config.runtime.output_fps or source_fps or 30.0)
    display_interval = 1.0 / max(1.0, target_display_fps)
    analysis_interval = config.vision.process_every_n_frames / max(1.0, source_fps)
    display_scale = float(config.runtime.display_scale)
    status_panel_width = 260
    window_name = "AI Driver Safety"
    if display_scale != 1.0:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if config.camera.width and config.camera.height:
            extra_width = 0 if config.runtime.minimal_overlay else status_panel_width
            cv2.resizeWindow(
                window_name,
                int((config.camera.width + extra_width) * display_scale),
                int(config.camera.height * display_scale),
            )

    lock = Lock()
    pending_packet: FramePacket | None = None
    latest_processed: ProcessedFrame | None = None
    stop_worker = False

    def submit_for_analysis(packet: FramePacket) -> None:
        nonlocal pending_packet
        packet.telemetry["analysis_submitted_perf_counter"] = perf_counter()
        with lock:
            pending_packet = packet

    def read_latest_processed() -> ProcessedFrame | None:
        with lock:
            return latest_processed

    def analysis_worker() -> None:
        nonlocal latest_processed, pending_packet
        while not stop_worker:
            with lock:
                packet = pending_packet
                pending_packet = None
            if packet is None:
                sleep(0.001)
                continue
            packet.telemetry.update(dht11_reader.read(packet.timestamp))
            packet.telemetry.update(metrics_reader.read().to_telemetry())
            processed = pipeline.process_frame(packet)
            alert_response_ms = None
            if processed.events:
                submitted_at = packet.telemetry.get("analysis_submitted_perf_counter")
                if submitted_at is not None:
                    alert_response_ms = max(0.0, (perf_counter() - submitted_at) * 1000.0)
                if alert_response_ms is not None:
                    processed.packet.telemetry["alert_response_ms"] = alert_response_ms
                event_logger.write_events(
                    processed.events,
                    risk_score=processed.risk_score,
                    alert_response_ms=alert_response_ms,
                )
            audio_player.handle_events(processed.events, now=packet.timestamp)
            gpio_actuator.handle_events(processed.events, now=packet.timestamp)
            if config.runtime.debug_frames:
                _print_debug_frame(processed)
            with lock:
                latest_processed = processed

    worker = Thread(target=analysis_worker, daemon=True)
    worker.start()
    last_submitted_at = -analysis_interval
    display_fps_estimate = 0.0
    last_display_at = perf_counter()
    try:
        while True:
            frame_started = perf_counter()
            display_delta = frame_started - last_display_at
            last_display_at = frame_started
            if display_delta > 0:
                instant_fps = 1.0 / display_delta
                display_fps_estimate = (
                    instant_fps
                    if display_fps_estimate <= 0
                    else display_fps_estimate * 0.9 + instant_fps * 0.1
                )
            packet = source.latest()
            packet.telemetry.update(metrics_reader.read().to_telemetry())
            if packet.timestamp - last_submitted_at >= analysis_interval:
                submit_for_analysis(packet)
                last_submitted_at = packet.timestamp

            processed = read_latest_processed()
            packet.telemetry["display_fps"] = display_fps_estimate
            if config.runtime.minimal_overlay:
                frame = packet.frame.copy()
                if processed is not None:
                    frame = draw_minimal_overlay_on_frame(frame, processed)
            elif processed is not None:
                if "alert_response_ms" in processed.packet.telemetry:
                    packet.telemetry["alert_response_ms"] = processed.packet.telemetry[
                        "alert_response_ms"
                    ]
                frame = draw_overlay_with_status_panel(
                    replace(processed, packet=packet),
                    recent_events=event_logger.recent,
                    panel_width=status_panel_width,
                )
            else:
                frame = packet.frame

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            sleep(max(0.0, display_interval - (perf_counter() - frame_started)))
    finally:
        stop_worker = True
        worker.join(timeout=0.5)
        gpio_actuator.close()
        event_logger.close()
        audio_player.close()
        source.close()
        cv2.destroyAllWindows()


def _print_debug_frame(processed: ProcessedFrame) -> None:
    try:
        events_signals = [event.signal for event in processed.events]
        objects = [
            (obj["label"], round(float(obj["confidence"]), 3), tuple(obj["bbox"]))
            for obj in processed.objects
        ]
        phone_val = processed.signals.get("phone_use", 0.0)
        distracted_val = processed.signals.get("distracted", 0.0)
        print(
            f"[frame {processed.packet.frame_index}] ts={processed.packet.timestamp:.3f} "
            f"pos={processed.driver_position} risk={processed.risk_score:.2f} "
            f"events={events_signals} objects={objects} phone={phone_val:.3f} "
            f"distracted={distracted_val:.3f}",
            flush=True,
        )
    except Exception:
        pass


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
