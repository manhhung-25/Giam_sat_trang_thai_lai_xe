from __future__ import annotations

from argparse import ArgumentParser
from time import sleep

from driver_safety.config import ActuatorConfig
from driver_safety.core.models import DetectionEvent, DriverState, Severity
from driver_safety.runtime.gpio_actuators import GpioAlertActuator


def main() -> None:
    parser = ArgumentParser(description="Blink Raspberry Pi buzzer and LED alert GPIO pins.")
    parser.add_argument("--buzzer-gpio", type=int, default=18)
    parser.add_argument("--led-gpio", type=int, default=23)
    parser.add_argument("--pulse-seconds", type=float, default=0.25)
    parser.add_argument("--duration-seconds", type=float, default=6.0)
    parser.add_argument("--active-low", action="store_true")
    args = parser.parse_args()

    config = ActuatorConfig(
        enabled=True,
        buzzer_gpio=args.buzzer_gpio,
        led_gpio=args.led_gpio,
        active_high=not args.active_low,
        pulse_seconds=args.pulse_seconds,
        cooldown_seconds=0.5,
    )
    actuator = GpioAlertActuator(config, verbose=True)
    if not actuator.available:
        raise SystemExit(
            "GPIO backend not available. On Raspberry Pi, run: "
            "sudo apt install -y python3-gpiozero python3-lgpio"
        )

    event = DetectionEvent(
        timestamp=0.0,
        frame_index=0,
        signal="test",
        state=DriverState.DROWSY,
        score=1.0,
        severity=Severity.WARNING,
        message="GPIO test alert",
    )

    print(
        f"Testing buzzer GPIO{args.buzzer_gpio} and LED GPIO{args.led_gpio} "
        f"for {args.duration_seconds:.1f}s. Press Ctrl+C to stop."
    )
    try:
        deadline = args.duration_seconds
        elapsed = 0.0
        while elapsed < deadline:
            actuator.handle_events([event])
            sleep(0.2)
            elapsed += 0.2
    finally:
        actuator.close()
        print("GPIO test finished.")


if __name__ == "__main__":
    main()
