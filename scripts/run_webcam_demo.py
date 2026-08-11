from pathlib import Path
import cv2

from driver_safety.config import load_config
from driver_safety.io.sources import WebcamFrameSource
from driver_safety.vision.pipeline import DriverSafetyPipeline
from driver_safety.io.overlay import draw_overlay


def main():
    cfg = load_config(Path("configs/default.yaml"))
    # keep demo short
    max_frames = 150

    source = WebcamFrameSource(0)
    class _NoFace:
        provider = "test"
        def detect(self, packet):
            return []
    pipeline = DriverSafetyPipeline(cfg, face_detector=_NoFace())
    frames = 0
    try:
        for packet in source:
            processed = pipeline.process_frame(packet)
            frame = draw_overlay(processed)
            cv2.imshow("AI Driver Safety (demo)", frame)
            frames += 1
            if frames >= max_frames:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
