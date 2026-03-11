"""Test the camera-to-robot coordinate transform."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import yaml

from calibration.transforms import load_calibration
from utils.logger import get_logger

logger = get_logger("test_transform")


def main() -> None:
    with open("configs/calibration.yaml") as f:
        config = yaml.safe_load(f)

    transform = load_calibration(config)

    test_centroids = [
        (320.0, 240.0),
        (640.0, 360.0),
        (100.0, 100.0),
    ]

    logger.info("Testing image-to-robot projection:")
    for centroid in test_centroids:
        robot_coords = transform.image_to_robot(centroid)
        logger.info(f"  Pixel {centroid} → Robot {robot_coords}")

    # Test 3-D camera-to-robot transform
    test_3d = np.array([0.1, 0.05, 0.5])
    result = transform.camera_to_robot(test_3d)
    logger.info(f"3-D transform: camera {test_3d} → robot {result} mm")
    logger.info("Transform test complete.")


if __name__ == "__main__":
    main()
