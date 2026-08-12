from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from driver_safety.core.models import DetectionEvent


@dataclass(frozen=True, slots=True)
class LocalEventLogRow:
    wall_time: str
    timestamp: float
    frame_index: int
    signal: str
    severity: str
    state: str
    score: float
    message: str
    risk_score: float
    alert_response_ms: float | None


class LocalEventLogger:
    def __init__(self, output_dir: str | Path, *, recent_limit: int = 8) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "events_realtime.jsonl"
        self.csv_path = self.output_dir / "events_realtime.csv"
        self._recent: deque[LocalEventLogRow] = deque(maxlen=recent_limit)
        self._csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self._jsonl_file = self.jsonl_path.open("a", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "wall_time",
                "timestamp",
                "frame_index",
                "signal",
                "severity",
                "state",
                "score",
                "message",
                "risk_score",
                "alert_response_ms",
            ],
        )
        if self.csv_path.stat().st_size == 0:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    @property
    def recent(self) -> list[LocalEventLogRow]:
        return list(self._recent)

    def write_events(
        self,
        events: list[DetectionEvent],
        *,
        risk_score: float,
        alert_response_ms: float | None,
    ) -> list[LocalEventLogRow]:
        rows: list[LocalEventLogRow] = []
        for event in events:
            row = LocalEventLogRow(
                wall_time=datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
                timestamp=round(event.timestamp, 3),
                frame_index=event.frame_index,
                signal=event.signal,
                severity=event.severity.value,
                state=event.state.value,
                score=event.score,
                message=event.message,
                risk_score=round(risk_score, 4),
                alert_response_ms=(
                    round(alert_response_ms, 3) if alert_response_ms is not None else None
                ),
            )
            rows.append(row)
            self._recent.appendleft(row)
            data = asdict(row)
            self._csv_writer.writerow(data)
            self._jsonl_file.write(json.dumps(data, ensure_ascii=True) + "\n")
        if rows:
            self._csv_file.flush()
            self._jsonl_file.flush()
        return rows

    def close(self) -> None:
        self._csv_file.close()
        self._jsonl_file.close()
