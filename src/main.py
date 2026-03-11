"""
Main entry point for the robot-arm pick-and-place pipeline.

Workflow:
1. Load configuration files.
2. Initialise camera stream.
3. Initialise Dobot controller.
4. Run detection + pick-and-place loop.
"""

import argparse

import yaml

from camera.camera_stream import CameraStream
from calibration.transforms import load_calibration
from robot.dobot_controller import DobotController
from vision.detector import Detector
from vision.target_selector import TargetSelector
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(args: argparse.Namespace) -> None:
    logger.info("Loading configurations …")
    robot_cfg = load_config(args.robot_config)
    camera_cfg = load_config(args.camera_config)
    calib_cfg = load_config(args.calibration_config)
    classes_cfg = load_config(args.classes_config)

    logger.info("Initialising camera …")
    camera = CameraStream(camera_cfg)
    camera.start()

    logger.info("Initialising robot …")
    robot = DobotController(robot_cfg)
    robot.home()

    logger.info("Loading calibration …")
    transform = load_calibration(calib_cfg)

    logger.info("Loading detector …")
    detector = Detector(args.model, classes_cfg)
    selector = TargetSelector(classes_cfg)

    logger.info("Starting pick-and-place loop. Press Ctrl+C to stop.")
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            detections = detector.detect(frame)
            target = selector.select(detections)

            if target is None:
                continue

            robot_coords = transform.image_to_robot(target.centroid)
            logger.info(f"Target at image {target.centroid} → robot {robot_coords}")
            robot.pick(robot_coords)
            robot.place(robot_cfg["robot"]["home_position"])

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        camera.stop()
        robot.disconnect()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot-arm pick-and-place demo")
    parser.add_argument("--robot-config", default="configs/robot.yaml")
    parser.add_argument("--camera-config", default="configs/camera.yaml")
    parser.add_argument("--calibration-config", default="configs/calibration.yaml")
    parser.add_argument("--classes-config", default="configs/classes.yaml")
    parser.add_argument("--model", default="models/yolo/best.pt")
    main(parser.parse_args())
