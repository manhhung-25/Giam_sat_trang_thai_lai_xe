from driver_safety.core.models import DetectionEvent, DriverState, Severity
from driver_safety.runtime.audio_alerts import AlertMessageGate, alert_message_for_signal


def test_alert_message_mapping_for_requested_signals() -> None:
    assert alert_message_for_signal("yawning") == "Phat hien tai xe ngap"
    assert alert_message_for_signal("eyes_closed") == "Phat hien ngu gat, nguy hiem, nguy hiem"
    assert alert_message_for_signal("drowsy") == "Canh bao dau hieu buon ngu, hay nghi ngoi"
    expected = "Phat hien sao nhang khi lai xe, hay tap trung"
    assert alert_message_for_signal("phone_use") == expected
    assert alert_message_for_signal("distracted") == expected


def test_alert_message_gate_respects_cooldown() -> None:
    gate = AlertMessageGate()
    event = DetectionEvent(
        timestamp=1.0,
        frame_index=1,
        signal="yawning",
        state=DriverState.YAWNING,
        score=1.0,
        severity=Severity.WARNING,
        message="Yawn detected",
    )
    first = gate.collect([event], now=100.0)
    second = gate.collect([event], now=101.0)
    third = gate.collect([event], now=109.0)
    assert len(first) == 1
    assert len(second) == 0
    assert len(third) == 1
