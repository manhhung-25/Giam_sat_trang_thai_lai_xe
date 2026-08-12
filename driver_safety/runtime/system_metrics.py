from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic


@dataclass(slots=True)
class SystemMetricsSnapshot:
    cpu_percent: float | None
    ram_used_mb: float | None
    ram_total_mb: float | None
    ram_percent: float | None
    chip_temperature_c: float | None

    def to_telemetry(self) -> dict[str, float]:
        telemetry: dict[str, float] = {}
        if self.cpu_percent is not None:
            telemetry["cpu_percent"] = self.cpu_percent
        if self.ram_used_mb is not None:
            telemetry["ram_used_mb"] = self.ram_used_mb
        if self.ram_total_mb is not None:
            telemetry["ram_total_mb"] = self.ram_total_mb
        if self.ram_percent is not None:
            telemetry["ram_percent"] = self.ram_percent
        if self.chip_temperature_c is not None:
            telemetry["chip_temperature_c"] = self.chip_temperature_c
        return telemetry


class SystemMetricsReader:
    def __init__(self, *, sample_interval_seconds: float = 1.0) -> None:
        self._sample_interval_seconds = sample_interval_seconds
        self._last_sample_at = -sample_interval_seconds
        self._last_cpu_times: tuple[int, int] | None = None
        self._last_snapshot = SystemMetricsSnapshot(None, None, None, None, None)

    def read(self) -> SystemMetricsSnapshot:
        now = monotonic()
        if now - self._last_sample_at < self._sample_interval_seconds:
            return self._last_snapshot
        self._last_sample_at = now
        ram = self._read_ram()
        ram_used_mb = ram[0] if ram is not None else None
        ram_total_mb = ram[1] if ram is not None else None
        ram_percent = ram[2] if ram is not None else None
        self._last_snapshot = SystemMetricsSnapshot(
            cpu_percent=self._read_cpu_percent(),
            ram_used_mb=ram_used_mb,
            ram_total_mb=ram_total_mb,
            ram_percent=ram_percent,
            chip_temperature_c=self._read_chip_temperature_c(),
        )
        return self._last_snapshot

    def _read_cpu_percent(self) -> float | None:
        try:
            parts = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
            values = [int(value) for value in parts[1:]]
        except Exception:
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        current = (idle, total)
        if self._last_cpu_times is None:
            self._last_cpu_times = current
            return 0.0
        last_idle, last_total = self._last_cpu_times
        self._last_cpu_times = current
        total_delta = total - last_total
        idle_delta = idle - last_idle
        if total_delta <= 0:
            return None
        return round(max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta))), 1)

    def _read_ram(self) -> tuple[float, float, float] | None:
        try:
            lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
        except Exception:
            return None
        values: dict[str, int] = {}
        for line in lines:
            key, raw_value = line.split(":", 1)
            values[key] = int(raw_value.strip().split()[0])
        total_kb = values.get("MemTotal")
        available_kb = values.get("MemAvailable")
        if total_kb is None or available_kb is None or total_kb <= 0:
            return None
        used_kb = total_kb - available_kb
        used_mb = used_kb / 1024.0
        total_mb = total_kb / 1024.0
        percent = used_kb / total_kb * 100.0
        return round(used_mb, 1), round(total_mb, 1), round(percent, 1)

    def _read_chip_temperature_c(self) -> float | None:
        candidates = [
            Path("/sys/class/thermal/thermal_zone0/temp"),
            Path("/sys/devices/virtual/thermal/thermal_zone0/temp"),
        ]
        for path in candidates:
            try:
                raw = path.read_text(encoding="utf-8").strip()
                return round(float(raw) / 1000.0, 1)
            except Exception:
                continue
        return None
