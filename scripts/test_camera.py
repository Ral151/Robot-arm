"""Quick camera test script – streams live video and prints FPS."""

import sys
import time
from pathlib import Path

# Allow running from the scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import yaml

from camera.camera_stream import CameraStream
from utils.logger import get_logger

logger = get_logger("test_camera")


def main() -> None:
    with open("configs/camera.yaml") as f:
        config = yaml.safe_load(f)

    camera = CameraStream(config)
    camera.start()
    logger.info("Camera started. Press 'q' to quit.")

    frame_count = 0
    start = time.time()

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            frame_count += 1
            elapsed = time.time() - start
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        logger.info(f"Captured {frame_count} frames in {elapsed:.1f}s ({fps:.1f} FPS)")


if __name__ == "__main__":
    main()
