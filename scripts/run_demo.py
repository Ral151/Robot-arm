"""End-to-end demo: detect objects and pick-and-place using the robot arm."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import cv2
import yaml

from camera.camera_stream import CameraStream
from calibration.transforms import load_calibration
from robot.dobot_controller import DobotController
from vision.detector import Detector
from vision.target_selector import TargetSelector
from utils.helpers import draw_detections, save_frame
from utils.logger import get_logger

logger = get_logger("run_demo")


def main(args: argparse.Namespace) -> None:
    def _load(p):
        with open(p) as f:
            return yaml.safe_load(f)

    robot_cfg = _load(args.robot_config)
    camera_cfg = _load(args.camera_config)
    calib_cfg = _load(args.calibration_config)
    classes_cfg = _load(args.classes_config)

    camera = CameraStream(camera_cfg)
    camera.start()

    robot = DobotController(robot_cfg)
    robot.home()

    transform = load_calibration(calib_cfg)
    detector = Detector(args.model, classes_cfg)
    selector = TargetSelector(classes_cfg)

    debug_dir = Path("outputs/debug_frames")
    debug_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    logger.info("Demo running. Press 'q' to stop.")

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue

            detections = detector.detect(frame)
            target = selector.select(detections)

            annotated = draw_detections(frame, detections)

            if target is not None:
                cx, cy = target.centroid
                cv2.circle(annotated, (int(cx), int(cy)), 8, (0, 0, 255), -1)
                robot_coords = transform.image_to_robot(target.centroid)
                logger.info(f"Target → robot {robot_coords}")
                robot.pick(robot_coords)
                robot.place(robot_cfg["robot"]["home_position"])

            if args.save_debug:
                save_frame(annotated, str(debug_dir), f"frame_{frame_idx:06d}.jpg")

            cv2.imshow("Robot-Arm Demo", annotated)
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        camera.stop()
        robot.disconnect()
        cv2.destroyAllWindows()
        logger.info("Demo finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot-arm pick-and-place demo")
    parser.add_argument("--robot-config", default="configs/robot.yaml")
    parser.add_argument("--camera-config", default="configs/camera.yaml")
    parser.add_argument("--calibration-config", default="configs/calibration.yaml")
    parser.add_argument("--classes-config", default="configs/classes.yaml")
    parser.add_argument("--model", default="models/yolo/best.pt")
    parser.add_argument("--save-debug", action="store_true", help="Save annotated debug frames")
    main(parser.parse_args())
