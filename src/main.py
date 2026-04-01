"""
Main entry point for the robot-arm sorting challenge.

Robot Arm Challenge Workflow:
1. Load configuration files.
2. Initialise camera stream (fixed above robot).
3. Initialise Dobot controller with gripper.
4. Run continuous detection + sorting loop until time expires.
5. Sort nuts, bolts, and screws into designated bins.
"""

import argparse
import signal
import sys
from typing import Dict

import yaml

from camera.camera_stream import CameraStream
import calibration.calibration_matrices
from calibration.transforms import load_calibration,calc_calibration,update_calib_yaml
from Dobot.Dobot_movement import DobotController
from Dobot.ports import check_port,get_dobot_port
import camera.rs_demo as rs
from pydobotplus import Dobot
from vision.detector import Detector
from vision.target_selector import TargetSelector
from utils.logger import get_logger

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
    
    #Get calibration matrix
    base_T_cam = calc_calibration()
    #Upload Calibration data to yaml FILE
    update_calib_yaml(base_T_cam)
    calib_cfg = load_config(args.calibration_config) # Calibration Data
    

    #=====================================================
    # 2) START & PREPARE DEVICES
    #=====================================================
    #Start Camera
    logger.info("Initialising camera …")
    camera = CameraStream(camera_cfg)
    camera.start()

    #Start Robot
    logger.info("Initialising robot …")
    device_port = check_port()
    device = Dobot(port=device_port)
    device.home()

    #Preparing Calibration Data
    logger.info("Loading calibration …")
    transform = load_calibration(calib_cfg)

    #Preparing YOLO AI Model
    logger.info("Loading YOLO detector …")
    detector = Detector(args.model, classes_cfg)
    selector = TargetSelector(classes_cfg)
    
    # Get sorting bin locations
    sorting_bins = robot_cfg["robot"]["sorting_bins"]
    logger.info(f"Sorting bins configured: {list(sorting_bins.keys())}")
    
    # Statistics tracking
    stats: Dict[str, int] = {"nut": 0, "bolt": 0, "screw": 0, "total": 0, "failed": 0}
    
    #=====================================================
    # 3) START LOOP
    #=====================================================
    logger.info("="*50)
    logger.info("Starting continuous sorting loop.")
    logger.info("System will run until Ctrl+C or time limit.")
    logger.info("="*50)
    
    try:
        cycle_count = 0
        while _running:
            frame = camera.read()
            frame_roi = camera.read_roi()
            if frame_roi is None:
                continue

            # Detect objects in the frame
            detections_roi = detector.detect(frame_roi)
            # Need to get full frame bounding boxes 
            target = selector.select(detections_roi) 

            if target is None:
                # No valid target found, continue to next frame
                continue

            cycle_count += 1
            logger.info(f"[Cycle {cycle_count}] Detected: {target.label} (conf={target.confidence:.2f})")
            
            # Convert pixel coordinates to robot coordinates
            robot_coords = transform.image_to_robot(target.centroid)
            logger.info(f"  Image {target.centroid} → Robot {robot_coords}")
            
            # Determine target bin based on object class
            target_label = target.label.lower()
            if target_label in sorting_bins:
                target_bin = sorting_bins[target_label]
                
                # Execute pick and place
                try:
                    robot.pick(robot_coords)
                    robot.place(target_bin)
                    
                    # Update statistics
                    stats[target_label] += 1
                    stats["total"] += 1
                    logger.info(f"  ✓ Sorted {target_label} to bin #{stats[target_label]}")
                    logger.info(f"  Total sorted: {stats['total']} (nuts={stats['nut']}, bolts={stats['bolt']}, screws={stats['screw']})")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to sort {target_label}: {e}")
                    stats["failed"] += 1
                    robot.home()  # Return to safe position
            else:
                logger.warning(f"  Unknown object class '{target_label}' - skipping")
                stats["failed"] += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        camera.stop()
        robot.home()  # Return to home before disconnecting
        robot.disconnect()
        
        # Print final statistics
        logger.info("="*50)
        logger.info("=== FINAL RESULTS ===")
        logger.info(f"Total items sorted: {stats['total']}")
        logger.info(f"  Nuts:   {stats['nut']}")
        logger.info(f"  Bolts:  {stats['bolt']}")
        logger.info(f"  Screws: {stats['screw']}")
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
    parser.add_argument("--model", default="models/yolo/best.pt")
    main(parser.parse_args())
