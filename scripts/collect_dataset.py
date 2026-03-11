"""Collect and save raw training images from the camera."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import cv2
import yaml

from camera.camera_stream import CameraStream
from utils.logger import get_logger

logger = get_logger("collect_dataset")


def main(args: argparse.Namespace) -> None:
    with open("configs/camera.yaml") as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = CameraStream(config)
    camera.start()
    logger.info(f"Saving frames to {output_dir}. Press 's' to save, 'q' to quit.")

    saved = 0
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            display = frame.copy()
            cv2.putText(
                display,
                f"Saved: {saved} | Press 's' to capture, 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Collect Dataset", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                filename = output_dir / f"frame_{saved:05d}.jpg"
                cv2.imwrite(str(filename), frame)
                saved += 1
                logger.info(f"Saved {filename}")
            elif key == ord("q"):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        logger.info(f"Dataset collection complete. {saved} images saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect raw training images")
    parser.add_argument(
        "--output-dir",
        default="data/raw_images",
        help="Directory to save captured frames",
    )
    main(parser.parse_args())
