from time import sleep

from driver_safety.config import ActuatorConfig
from driver_safety.core.models import DetectionEvent, DriverState, Severity
from driver_safety.runtime import gpio_actuators
from driver_safety.runtime.gpio_actuators import GpioAlertActuator


class FakeGPIO:
    name = "fake"

    def __init__(self) -> None:
        self.setups = []
        self.outputs = []
        self.cleaned = None

    def setup(self) -> None:
        self.setups.append("setup")

    def output(self, pin: int, value: int) -> None:
        self.outputs.append((pin, value))

    def cleanup(self, pins: tuple[int, int]) -> None:
        self.cleaned = pins


def test_gpio_alert_actuator_blinks_while_alert_is_active(monkeypatch) -> None:
    fake_gpio = FakeGPIO()
    monkeypatch.setattr(gpio_actuators, "_create_gpio_backend", lambda config: fake_gpio)
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

    assert fake_gpio.setups == ["setup"]
    assert fake_gpio.outputs.count((18, 1)) >= 2
    assert fake_gpio.outputs.count((23, 1)) >= 2
    assert fake_gpio.outputs[-2:] == [(18, 0), (23, 0)]
    assert fake_gpio.cleaned == (18, 23)
