from __future__ import annotations

import csv
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from driver_safety.config import DHT11Config


@dataclass(slots=True)
class DHT11Sample:
    timestamp: float
    temperature_c: float
    humidity_pct: float


class DHT11Reader:
    def __init__(self, config: DHT11Config) -> None:
        self.config = config
        self._samples = _load_samples(config.csv_path) if config.csv_path else []
        self._timestamps = [sample.timestamp for sample in self._samples]
        # Raspberry Pi reader is optional; attempt to import when available
        try:
            from driver_safety.runtime.raspi_dht11 import RaspiDHT11Reader

            self._raspi_reader = RaspiDHT11Reader()
        except Exception:
            self._raspi_reader = None

    def read(self, timestamp: float) -> dict[str, float]:
        if not self.config.enabled:
            return {}
        # Prefer live RasPi DHT11 if available
        if self._raspi_reader is not None:
            try:
                return self._raspi_reader.read()
            except Exception:
                # fall back to CSV or synthetic
                pass
        if self._samples:
            index = bisect_right(self._timestamps, timestamp) - 1
            if index < 0:
                index = 0
            sample = self._samples[index]
            return {
                "dht11_temperature_c": sample.temperature_c,
                "dht11_humidity_pct": sample.humidity_pct,
            }
        if not self.config.simulate_when_missing:
            return {}
        return _synthetic_sample(timestamp)


def _load_samples(path: str | Path) -> list[DHT11Sample]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"DHT11 CSV not found: {csv_path}")
    samples: list[DHT11Sample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "temperature_c", "humidity_pct"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "DHT11 CSV must include headers: timestamp,temperature_c,humidity_pct"
            )
        for row in reader:
            samples.append(
                DHT11Sample(
                    timestamp=float(row["timestamp"]),
                    temperature_c=float(row["temperature_c"]),
                    humidity_pct=float(row["humidity_pct"]),
                )
            )
    if not samples:
        raise ValueError(f"DHT11 CSV has no samples: {csv_path}")
    samples.sort(key=lambda item: item.timestamp)
    return samples


def _synthetic_sample(timestamp: float) -> dict[str, float]:
    # Demo fallback so the DHT11 panel is still visible without a physical sensor stream.
    wave = math.sin(timestamp / 24.0)
    wave2 = math.cos(timestamp / 27.0)
    return {
        "dht11_temperature_c": 27.0 + wave * 3.2,
        "dht11_humidity_pct": 58.0 + wave2 * 12.0,
    }
