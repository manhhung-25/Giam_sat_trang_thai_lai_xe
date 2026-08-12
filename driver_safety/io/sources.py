from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock, Thread
from time import monotonic, sleep

import cv2

from driver_safety.core.models import FramePacket


class VideoFrameSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video: {self.path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0) or 24.0
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def __iter__(self) -> VideoFrameSource:
        return self

    def __next__(self) -> FramePacket:
        ok, frame = self.capture.read()
        if not ok:
            self.close()
            raise StopIteration
        frame_index = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp = frame_index / self.fps
        return FramePacket(
            frame=frame,
            timestamp=timestamp,
            frame_index=frame_index,
            source_id=str(self.path),
            fps=self.fps,
        )

    def latest(self, *, max_grabs: int = 4) -> FramePacket:
        for _ in range(max(0, max_grabs - 1)):
            if not self.capture.grab():
                break
        return next(self)

    def close(self) -> None:
        self.capture.release()


class WebcamFrameSource:
    def __init__(
        self,
        index: int = 0,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        buffer_size: int | None = 1,
        fourcc: str | None = None,
        threaded: bool = False,
    ) -> None:
        self.index = index
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else 0
        self.capture = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open webcam index {index}")
        if buffer_size is not None:
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        if fourcc:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.started = monotonic()
        self.frame_index = -1
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0) or 24.0
        self._threaded = threaded
        self._closed = False
        self._lock = Lock()
        self._latest_frame = None
        self._latest_index = -1
        self._latest_timestamp = 0.0
        self._thread: Thread | None = None
        if self._threaded:
            self._thread = Thread(target=self._reader_loop, daemon=True)
            self._thread.start()

    def __iter__(self) -> WebcamFrameSource:
        return self

    def __next__(self) -> FramePacket:
        if self._threaded:
            return self.latest()
        ok, frame = self.capture.read()
        if not ok:
            self.close()
            raise StopIteration
        self.frame_index += 1
        return FramePacket(
            frame=frame,
            timestamp=monotonic() - self.started,
            frame_index=self.frame_index,
            source_id=f"webcam:{self.index}",
            fps=self.fps,
        )

    def latest(self) -> FramePacket:
        if not self._threaded:
            return next(self)
        while not self._closed:
            with self._lock:
                if self._latest_frame is not None:
                    return FramePacket(
                        frame=self._latest_frame.copy(),
                        timestamp=self._latest_timestamp,
                        frame_index=self._latest_index,
                        source_id=f"webcam:{self.index}",
                        fps=self.fps,
                    )
            sleep(0.001)
        raise StopIteration

    def _reader_loop(self) -> None:
        while not self._closed:
            ok, frame = self.capture.read()
            if not ok:
                sleep(0.005)
                continue
            timestamp = monotonic() - self.started
            with self._lock:
                self.frame_index += 1
                self._latest_frame = frame
                self._latest_index = self.frame_index
                self._latest_timestamp = timestamp

    def close(self) -> None:
        self._closed = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self.capture.release()
