from __future__ import annotations

from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Protocol

from driver_safety.config import ActuatorConfig
from driver_safety.core.models import DetectionEvent, Severity


class GpioBackend(Protocol):
    name: str

    def setup(self) -> None: ...

    def output(self, pin: int, value: int) -> None: ...

    def cleanup(self, pins: tuple[int, int]) -> None: ...


class GpioAlertActuator:
    def __init__(self, config: ActuatorConfig, *, verbose: bool = False) -> None:
        self._config = config
        self._gpio = _create_gpio_backend(config) if config.enabled else None
        self._stop = Event()
        self._wake = Event()
        self._lock = Lock()
        self._worker: Thread | None = None
        self._active_until = 0.0
        self._pending_activation_started_at: float | None = None
        self._last_activation_latency_ms: float | None = None
        self._on_value = 1 if config.active_high else 0
        self._off_value = 0 if config.active_high else 1

        if self._gpio is None:
            if config.enabled and verbose:
                print("GPIO actuator disabled: install gpiozero/lgpio or RPi.GPIO on Raspberry Pi.")
            return

        self._gpio.setup()
        if verbose:
            print(
                f"GPIO actuator ready via {self._gpio.name}: "
                f"buzzer=GPIO{config.buzzer_gpio}, led=GPIO{config.led_gpio}"
            )
        self._worker = Thread(target=self._run, name="gpio-alert-actuator", daemon=True)
        self._worker.start()

    @property
    def available(self) -> bool:
        return self._gpio is not None

    @property
    def last_activation_latency_ms(self) -> float | None:
        with self._lock:
            return self._last_activation_latency_ms

    def handle_events(self, events: list[DetectionEvent], *, now: float | None = None) -> None:
        if self._gpio is None or not any(self._should_alert(event) for event in events):
            return

        current = monotonic()
        hold_seconds = max(self._config.cooldown_seconds, self._config.pulse_seconds * 2.0)
        with self._lock:
            if current >= self._active_until:
                self._pending_activation_started_at = current
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
                self._record_activation_latency()
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

    def _record_activation_latency(self) -> None:
        with self._lock:
            started_at = self._pending_activation_started_at
            self._pending_activation_started_at = None
            if started_at is not None:
                self._last_activation_latency_ms = round((monotonic() - started_at) * 1000.0, 3)

    def _write_outputs(self, value: int) -> None:
        assert self._gpio is not None
        self._gpio.output(self._config.buzzer_gpio, value)
        self._gpio.output(self._config.led_gpio, value)


class RpiGpioBackend:
    name = "RPi.GPIO"

    def __init__(self, config: ActuatorConfig, off_value: int) -> None:
        import RPi.GPIO as GPIO

        self._gpio = GPIO
        self._config = config
        self._off_value = off_value

    def setup(self) -> None:
        self._gpio.setmode(self._gpio.BCM)
        self._gpio.setup(self._config.buzzer_gpio, self._gpio.OUT, initial=self._off_value)
        self._gpio.setup(self._config.led_gpio, self._gpio.OUT, initial=self._off_value)

    def output(self, pin: int, value: int) -> None:
        self._gpio.output(pin, value)

    def cleanup(self, pins: tuple[int, int]) -> None:
        self._gpio.cleanup(pins)


class GpioZeroBackend:
    name = "gpiozero"

    def __init__(self, config: ActuatorConfig, off_value: int) -> None:
        from gpiozero import DigitalOutputDevice

        self._device_class = DigitalOutputDevice
        self._config = config
        self._off_value = off_value
        self._devices: dict[int, object] = {}

    def setup(self) -> None:
        initial_value = bool(self._off_value)
        self._devices[self._config.buzzer_gpio] = self._device_class(
            self._config.buzzer_gpio,
            active_high=True,
            initial_value=initial_value,
        )
        self._devices[self._config.led_gpio] = self._device_class(
            self._config.led_gpio,
            active_high=True,
            initial_value=initial_value,
        )

    def output(self, pin: int, value: int) -> None:
        device = self._devices[pin]
        if value:
            device.on()
        else:
            device.off()

    def cleanup(self, pins: tuple[int, int]) -> None:
        for pin in pins:
            device = self._devices.get(pin)
            if device is not None:
                device.close()


def _create_gpio_backend(config: ActuatorConfig) -> GpioBackend | None:
    off_value = 0 if config.active_high else 1
    try:
        return GpioZeroBackend(config, off_value)
    except Exception:
        pass
    try:
        return RpiGpioBackend(config, off_value)
    except Exception:
        return None
