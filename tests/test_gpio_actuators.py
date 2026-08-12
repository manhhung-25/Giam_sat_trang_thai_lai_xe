from time import sleep

from driver_safety.config import ActuatorConfig
from driver_safety.core.models import DetectionEvent, DriverState, Severity
from driver_safety.runtime import gpio_actuators
from driver_safety.runtime.gpio_actuators import GpioAlertActuator


class FakeGPIO:
    BCM = "BCM"
    OUT = "OUT"

    def __init__(self) -> None:
        self.mode = None
        self.setups = []
        self.outputs = []
        self.cleaned = None

    def setmode(self, mode: str) -> None:
        self.mode = mode

    def setup(self, pin: int, direction: str, initial: int) -> None:
        self.setups.append((pin, direction, initial))

    def output(self, pin: int, value: int) -> None:
        self.outputs.append((pin, value))

    def cleanup(self, pins: tuple[int, int]) -> None:
        self.cleaned = pins


def test_gpio_alert_actuator_blinks_while_alert_is_active(monkeypatch) -> None:
    fake_gpio = FakeGPIO()
    monkeypatch.setattr(gpio_actuators, "_load_gpio", lambda: fake_gpio)
    config = ActuatorConfig(enabled=True, pulse_seconds=0.01, cooldown_seconds=0.04)
    actuator = GpioAlertActuator(config)
    event = DetectionEvent(
        timestamp=1.0,
        frame_index=1,
        signal="drowsy",
        state=DriverState.DROWSY,
        score=1.0,
        severity=Severity.WARNING,
        message="Drowsy",
    )

    actuator.handle_events([event], now=1.0)
    sleep(0.025)
    actuator.handle_events([event], now=1.03)
    sleep(0.08)
    actuator.close()

    assert fake_gpio.mode == fake_gpio.BCM
    assert (18, fake_gpio.OUT, 0) in fake_gpio.setups
    assert (23, fake_gpio.OUT, 0) in fake_gpio.setups
    assert fake_gpio.outputs.count((18, 1)) >= 2
    assert fake_gpio.outputs.count((23, 1)) >= 2
    assert fake_gpio.outputs[-2:] == [(18, 0), (23, 0)]
    assert fake_gpio.cleaned == (18, 23)
