"""
Main entry point for the robot-arm sorting challenge.

Robot Arm Challenge Workflow:
1. Load configuration files.
2. Initialise camera stream (fixed above robot).
3. Initialise Dobot controller with gripper.
4. Run continuous detection + sorting loop until time expires.
5. Sort nuts, bolts, and screws into designated bins.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time
import signal
import sys
from typing import Dict

import yaml
import numpy as np
import cv2

from camera.camera_stream import CameraStream
from calibration.transforms import calc_calibration, update_calib_yaml
from calibration.apriltag_detection import get_apriltag_object
from Dobot.Dobot_movement import DobotController
from vision.detector import Detector
from vision.target_selector import TargetSelector
from utils.logger import get_logger
from camera.rs_demo.realsense_pixel_to_3d import pixel_to_homogeneous_point
from camera.rs_demo.realsense_utils import get_aligned_frames


logger = get_logger(__name__)

# Global flag for graceful shutdown
_running = True

def signal_handler(sig, frame):
    # Handle Ctrl+C for graceful shutdown
    global _running
    logger.info("Shutdown signal received. Finishing current operation...")
    _running = False

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(args: argparse.Namespace) -> None:
    global _running
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    #====================================================
    # 1) LOAD ALL CONFIGURATIONS
    #====================================================
    logger.info("=== Robot Arm Sorting Challenge ===")
    logger.info("Loading configurations …")
    robot_cfg = load_config(args.robot_config) #Robot settings
    camera_cfg = load_config(args.camera_config) #Camera Settings
    classes_cfg = load_config(args.classes_config)

    #=====================================================
    # 2) START & PREPARE DEVICES
    #=====================================================
    #Start Camera
    logger.info("Initialising camera …")
    camera = CameraStream(camera_cfg)
    camera.start()
    intrinsics = camera.get_intrinsics()
    pipeline = camera.get_pipeline()
    align = camera.get_align()

    #Start Robot
    logger.info("Initialising robot …")
    robot = DobotController(robot_cfg) # By this point, we will have a Dobot device "robot.device" with DobotController object "robot"
    robot.home_to_position()  

    # Get apriltag
    apriltag = get_apriltag_object(pipeline, align, intrinsics)
    # Get calibration matrix
    base_T_cam = calc_calibration(robot._device,apriltag)

    #Preparing YOLO AI Model
    logger.info("Loading YOLO detector …")
    detector = Detector(args.model, classes_cfg)
    selector = TargetSelector(classes_cfg)
    
    # Get sorting bin locations
    sorting_bins = robot_cfg["robot"]["sorting_bins"]
    logger.info(f"Sorting bins configured: {list(sorting_bins.keys())}")
    
    # Statistics tracking
    stats: Dict[str, int] = {"nut": 0, "long_bolt": 0, "short_bolt": 0, "total": 0, "failed": 0}
    
    #=====================================================
    # 3) START LOOP
    #=====================================================
    logger.info("="*50)
    logger.info("Starting continuous sorting loop.")
    logger.info("System will run until Ctrl+C or time limit.")
    logger.info("="*50)
    window_name = "RealSense View"
    roi_window_name = "ROI View"
    
    try:
        cycle_count = 0
        while _running:
            robot.home_to_position()  # Ensure we start from a known position each cycle
            frame = camera.read()
            frame_roi = camera.read_roi()
            if frame is None or frame_roi is None:
                continue

            # Detect objects in the ROI frame
            detections_roi = detector.detect(frame_roi)
            roi = camera.get_roi()
            roi_x1, roi_y1, _, _ = roi
            display_frame = frame.copy()
            display_roi = frame_roi.copy()

            for det in detections_roi:
                x1, y1, x2, y2 = det.bbox
                pt1 = (roi_x1 + x1, roi_y1 + y1)
                pt2 = (roi_x1 + x2, roi_y1 + y2)
                cv2.rectangle(display_frame, pt1, pt2, (0, 255, 0), 2)
                cv2.rectangle(display_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det.label} {det.confidence:.2f}"
                cv2.putText(display_frame, label, (pt1[0], max(pt1[1] - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display_roi, label, (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw center pixel on full frame and show its coordinates.
                cx, cy = det.centroid(roi)
                center_pt = (int(round(cx)), int(round(cy)))
                cv2.circle(display_frame, center_pt, 4, (0, 255, 255), -1)
                coord_text = f"({center_pt[0]}, {center_pt[1]})"
                cv2.putText(
                    display_frame,
                    coord_text,
                    (center_pt[0] + 6, max(center_pt[1] - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                )

            cv2.imshow(window_name, display_frame)
            cv2.imshow(roi_window_name, display_roi)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                _running = False
                break

            target = selector.select(detections_roi)

            if target is None:
                # No valid target found, continue to next frame
                continue

            cycle_count += 1
            logger.info(f"[Cycle {cycle_count}] Detected: {target.label} (conf={target.confidence:.2f})")
            
            # Convert pixel coordinates to robot coordinates using real frame coordinates
            target_centroid = target.centroid(roi)
            # Get aligned frames
            color_frame, depth_frame = get_aligned_frames(pipeline,align)
            # Convert center target_centroid to 3D realsense coordinates using depth information before do the transformation to robot coordinates
            px = int(round(target_centroid[0]))
            py = int(round(target_centroid[1]))
            P_camera = pixel_to_homogeneous_point(intrinsics, px, py, depth_frame)
            if P_camera is None:
                continue
            X, Y, Z = P_camera[0,0], P_camera[1,0], P_camera[2,0]

            # Convert to millimeters for easier reading
            X_mm, Y_mm, Z_mm = X * 1000, Y * 1000, Z * 1000
            P_camera = np.array([[X_mm],[Y_mm],[Z_mm],[1.0]], dtype=np.float64)
            
            # Calculate P_dobot
            P_dobot = base_T_cam @ P_camera
            robot_coords = {
                "x": float(P_dobot[0][0]),
                "y": float(P_dobot[1][0]),
                "z": float(P_dobot[2][0]),
            }
            
            # Determine target bin based on object class
            target_label = target.label.lower()
            if target_label in sorting_bins:
                target_bin = sorting_bins[target_label]
                
                # Execute pick and place
                try:
                    """Need to consider the real frame bounding boxes to determine the size of the object and adjust the gripper
                    accordingly. Ie. if the bounding boxes create a 90 degree angle, the gripper should rotate to match that angle."""
                    robot.pick(robot_coords)
                    time.sleep(1)
                    robot.place(target_bin)
                    
                    # Update statistics
                    stats[target_label] += 1
                    stats["total"] += 1
                    logger.info(f"  ✓ Sorted {target_label} to bin #{stats[target_label]}")
                    # logger.info(f"  Total sorted: {stats['total']} (nuts={stats['nut']}, bolts={stats['long_bolt']}, screws={stats['short_bolt']})")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to sort {target_label}: {e}")
                    stats["failed"] += 1
                    robot.home_to_position()  # Return to safe position
            else:
                logger.warning(f"  Unknown object class '{target_label}' - skipping")
                stats["failed"] += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        cv2.destroyAllWindows()
        camera.stop()
        robot.home_to_position()  # Return to home_to_position before disconnecting
        robot.disconnect()
        
        # Print final statistics
        logger.info("="*50)
        logger.info("=== FINAL RESULTS ===")
        logger.info(f"Total items sorted: {stats['total']}")
        # logger.info(f"  Nuts:   {stats['nut']}")
        # logger.info(f"  Bolts:  {stats['long_bolt']}")
        # logger.info(f"  Screws: {stats['short_bolt']}")
        logger.info(f"Failed attempts: {stats['failed']}")
        logger.info(f"Success rate: {(stats['total']/(stats['total']+stats['failed'])*100) if (stats['total']+stats['failed']) > 0 else 0:.1f}%")
        logger.info("="*50)
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Arm Sorting Challenge")
    parser.add_argument("--robot-config", default="configs/robot.yaml")
    parser.add_argument("--camera-config", default="configs/camera.yaml")
    parser.add_argument("--calibration-config", default="configs/calibration.yaml")
    parser.add_argument("--classes-config", default="configs/classes.yaml")
    parser.add_argument("--model", default="bestV2.pt")
    main(parser.parse_args())
