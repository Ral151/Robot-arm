"""Quick robot connection and movement test script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from robot.dobot_controller import DobotController
from utils.logger import get_logger

logger = get_logger("test_robot")


def main() -> None:
    with open("configs/robot.yaml") as f:
        config = yaml.safe_load(f)

    robot = DobotController(config)
    logger.info("Moving to home position …")
    robot.home()

    test_positions = [
        {"x": 220.0, "y": 0.0, "z": 50.0, "r": 0.0},
        {"x": 220.0, "y": 50.0, "z": 50.0, "r": 0.0},
        {"x": 220.0, "y": -50.0, "z": 50.0, "r": 0.0},
    ]

    for pos in test_positions:
        logger.info(f"Moving to {pos} …")
        robot.move_to(pos["x"], pos["y"], pos["z"], pos["r"])

    logger.info("Returning home …")
    robot.home()
    robot.disconnect()
    logger.info("Robot test complete.")


if __name__ == "__main__":
    main()
