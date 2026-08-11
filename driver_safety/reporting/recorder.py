from __future__ import annotations

from pathlib import Path

from driver_safety.core.models import DetectionEvent


class SessionRecorder:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[DetectionEvent] = []
        self.frame_scores: list[tuple[float, float]] = []
        self.latencies_ms: list[float] = []
        self.dht11_timeline: list[dict[str, float]] = []

    def write_event(self, event: DetectionEvent) -> None:
        self.events.append(event)

    def write_frame_score(self, timestamp: float, score: float, latency_ms: float) -> None:
        self.frame_scores.append((timestamp, score))
        self.latencies_ms.append(latency_ms)

    def write_dht11(self, timestamp: float, temperature_c: float, humidity_pct: float) -> None:
        self.dht11_timeline.append(
            {
                "timestamp": round(timestamp, 3),
                "temperature_c": round(temperature_c, 3),
                "humidity_pct": round(humidity_pct, 3),
            }
        )
