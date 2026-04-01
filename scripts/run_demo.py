"""End-to-end demo: detect and sort nuts, bolts, screws using the robot arm.

Robot Arm Challenge Demo with visualization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import signal
from typing import Dict

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

# Global flag for graceful shutdown
_running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global _running
    logger.info("Shutdown signal received...")
    _running = False


def main(args: argparse.Namespace) -> None:
    global _running
    
    signal.signal(signal.SIGINT, signal_handler)
    
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
    
    # Get sorting bin locations
    sorting_bins = robot_cfg["robot"]["sorting_bins"]

    debug_dir = Path("outputs/debug_frames")
    debug_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    stats: Dict[str, int] = {"nut": 0, "bolt": 0, "screw": 0, "total": 0, "failed": 0}
    
    logger.info("="*50)
    logger.info("Robot Arm Sorting Challenge Demo")
    logger.info("Press 'q' to stop. Window shows live detection.")
    logger.info("="*50)

    try:
        while _running:
            frame = camera.read()
            if frame is None:
                continue

            detections = detector.detect(frame)
            target = selector.select(detections)

            annotated = draw_detections(frame, detections)
            
            # Draw statistics overlay
            cv2.putText(annotated, f"Total: {stats['total']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated, f"Nuts: {stats['nut']} | Bolts: {stats['bolt']} | Screws: {stats['screw']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if target is not None:
                cx, cy = target.centroid()
                cv2.circle(annotated, (int(cx), int(cy)), 10, (0, 0, 255), -1)
                cv2.putText(annotated, f"Target: {target.label}", (int(cx)+15, int(cy)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                robot_coords = transform.image_to_robot(target.centroid())
                target_label = target.label.lower()
                
                if target_label in sorting_bins:
                    target_bin = sorting_bins[target_label]
                    logger.info(f"Sorting {target_label} → bin")
                    
                    try:
                        robot.pick(robot_coords)
                        robot.place(target_bin)
                        stats[target_label] += 1
                        stats["total"] += 1
                        logger.info(f"✓ Success! Total: {stats['total']}")
                    except Exception as e:
                        logger.error(f"✗ Failed: {e}")
                        stats["failed"] += 1
                        robot.home()
                else:
                    logger.warning(f"Unknown class: {target_label}")
                    stats["failed"] += 1

            if args.save_debug:
                save_frame(annotated, str(debug_dir), f"frame_{frame_idx:06d}.jpg")

            cv2.imshow("Robot-Arm Sorting Demo", annotated)
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                _running = False
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        camera.stop()
        robot.home()
        robot.disconnect()
        cv2.destroyAllWindows()
        
        # Print final statistics
        logger.info("="*50)
        logger.info("=== FINAL RESULTS ===")
        logger.info(f"Total sorted: {stats['total']}")
        logger.info(f"  Nuts: {stats['nut']} | Bolts: {stats['bolt']} | Screws: {stats['screw']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info("="*50)
        logger.info("Demo finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Arm Sorting Challenge Demo")
    parser.add_argument("--robot-config", default="configs/robot.yaml")
    parser.add_argument("--camera-config", default="configs/camera.yaml")
    parser.add_argument("--calibration-config", default="configs/calibration.yaml")
    parser.add_argument("--classes-config", default="configs/classes.yaml")
    parser.add_argument("--model", default="models/yolo/best.pt")
    parser.add_argument("--save-debug", action="store_true", help="Save annotated debug frames")
    main(parser.parse_args())
