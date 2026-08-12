from __future__ import annotations

from threading import Event, Lock, Thread
from time import monotonic, sleep

from driver_safety.config import ActuatorConfig
from driver_safety.core.models import DetectionEvent, Severity


class GpioAlertActuator:
    def __init__(self, config: ActuatorConfig) -> None:
        self._config = config
        self._gpio = _load_gpio() if config.enabled else None
        self._stop = Event()
        self._wake = Event()
        self._lock = Lock()
        self._worker: Thread | None = None
        self._active_until = 0.0
        self._on_value = 1 if config.active_high else 0
        self._off_value = 0 if config.active_high else 1

        if self._gpio is None:
            return

        self._gpio.setmode(self._gpio.BCM)
        self._gpio.setup(config.buzzer_gpio, self._gpio.OUT, initial=self._off_value)
        self._gpio.setup(config.led_gpio, self._gpio.OUT, initial=self._off_value)
        self._worker = Thread(target=self._run, name="gpio-alert-actuator", daemon=True)
        self._worker.start()

    @property
    def available(self) -> bool:
        return self._gpio is not None

    def handle_events(self, events: list[DetectionEvent], *, now: float | None = None) -> None:
        if self._gpio is None or not any(self._should_alert(event) for event in events):
            return

        current = monotonic()
        hold_seconds = max(self._config.cooldown_seconds, self._config.pulse_seconds * 2.0)
        with self._lock:
            self._active_until = current + hold_seconds
        self._wake.set()

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
        if self._gpio is not None:
            self._write_outputs(self._off_value)
            self._gpio.cleanup((self._config.buzzer_gpio, self._config.led_gpio))

    def _should_alert(self, event: DetectionEvent) -> bool:
        if event.severity == Severity.CRITICAL:
            return self._config.critical_enabled
        if event.severity == Severity.WARNING:
            return self._config.warnings_enabled
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if not self._is_active():
                    self._write_outputs(self._off_value)
                    self._wake.wait(timeout=0.2)
                    self._wake.clear()
                    continue

                self._write_outputs(self._on_value)
                self._sleep_or_stop(self._config.pulse_seconds)
                self._write_outputs(self._off_value)
                self._sleep_or_stop(self._config.pulse_seconds)
            except Exception:
                sleep(0.05)

    def _is_active(self) -> bool:
        with self._lock:
            return monotonic() < self._active_until

    def _sleep_or_stop(self, seconds: float) -> None:
        deadline = monotonic() + seconds
        while not self._stop.is_set() and monotonic() < deadline:
            sleep(0.02)

    def _write_outputs(self, value: int) -> None:
        assert self._gpio is not None
        self._gpio.output(self._config.buzzer_gpio, value)
        self._gpio.output(self._config.led_gpio, value)


def _load_gpio():
    try:
        import RPi.GPIO as GPIO
    except Exception:
        return None
    return GPIO
