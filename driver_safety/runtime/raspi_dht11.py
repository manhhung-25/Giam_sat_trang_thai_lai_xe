from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

try:
    import RPi.GPIO as GPIO
    import dht11
except Exception:  # pragma: no cover - optional on non-RPi
    GPIO = None  # type: ignore
    dht11 = None  # type: ignore


@dataclass(slots=True)
class RaspiDHT11Reader:
    pin: int = 4

    def __post_init__(self) -> None:
        if GPIO is None or dht11 is None:
            raise RuntimeError("RPi GPIO or dht11 library not available on this system")

    def read(self) -> dict[str, float]:
        # Returns latest temperature and humidity from the DHT11 sensor on the given pin
        instance = dht11.DHT11(pin=self.pin)
        result = instance.read()
        if not result.is_valid():
            return {}
        return {"dht11_temperature_c": float(result.temperature), "dht11_humidity_pct": float(result.humidity)}
