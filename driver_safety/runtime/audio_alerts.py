from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic
from typing import Protocol

from driver_safety.core.models import DetectionEvent


def alert_message_for_signal(signal: str) -> str | None:
    messages = {
        "yawning": "Phat hien tai xe ngap",
        "eyes_closed": "Phat hien ngu gat, nguy hiem, nguy hiem",
        "drowsy": "Canh bao dau hieu buon ngu, hay nghi ngoi",
        "phone_use": "Phat hien sao nhang khi lai xe, hay tap trung",
        "distracted": "Phat hien sao nhang khi lai xe, hay tap trung",
    }
    return messages.get(signal)


def alert_cooldown_for_signal(signal: str) -> float:
    cooldowns = {
        "yawning": 8.0,
        "eyes_closed": 5.0,
        "drowsy": 10.0,
        "phone_use": 6.0,
        "distracted": 6.0,
    }
    return cooldowns.get(signal, 6.0)


@dataclass(slots=True)
class AlertMessageGate:
    last_announced_at: dict[str, float] = field(default_factory=dict)

    def collect(self, events: list[DetectionEvent], *, now: float) -> list[str]:
        messages: list[str] = []
        for event in events:
            message = alert_message_for_signal(event.signal)
            if not message:
                continue
            cooldown = alert_cooldown_for_signal(event.signal)
            previous = self.last_announced_at.get(event.signal)
            if previous is not None and now - previous < cooldown:
                continue
            self.last_announced_at[event.signal] = now
            messages.append(message)
        return messages


class SpeechBackend(Protocol):
    def speak(self, text: str) -> None: ...


class TtsSpeechBackend:
    def __init__(self) -> None:
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 170)

    def speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()


class BeepSpeechBackend:
    def __init__(self) -> None:
        import winsound

        self._winsound = winsound

    def speak(self, text: str) -> None:
        # Fallback audio cue when TTS backend is unavailable.
        for _ in range(2):
            self._winsound.MessageBeep(self._winsound.MB_ICONEXCLAMATION)


def create_speech_backend() -> SpeechBackend | None:
    try:
        return TtsSpeechBackend()
    except Exception:
        pass
    try:
        return BeepSpeechBackend()
    except Exception:
        return None


class AudioAlertPlayer:
    def __init__(self) -> None:
        self._gate = AlertMessageGate()
        self._queue: Queue[str] = Queue()
        self._stop = Event()
        self._backend = create_speech_backend()
        self._worker: Thread | None = None
        if self._backend is not None:
            self._worker = Thread(target=self._run, name="audio-alert-player", daemon=True)
            self._worker.start()

    def handle_events(self, events: list[DetectionEvent], *, now: float | None = None) -> None:
        if self._backend is None:
            return
        current = monotonic() if now is None else now
        for message in self._gate.collect(events, now=current):
            self._queue.put(message)

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)

    def _run(self) -> None:
        assert self._backend is not None
        while not self._stop.is_set():
            try:
                message = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._backend.speak(message)
            except Exception:
                # Keep webcam loop alive even if audio backend fails.
                continue
