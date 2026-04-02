"""Coordinate-frame transforms between camera space and robot space."""
import os
import yaml
import numpy as np
import yaml,sys
import matplotlib.pyplot as plt
import cv2
import pyrealsense2 as rs

from utils.logger import get_logger
from utils.get_port import get_dobot_port
from utils.camera_functions import initialize_pipeline
from utils.camera_functions import get_camera_intrinsics
from .calibration_matrices import get_dobot_to_gripper_matrix, get_gripper_to_tag_matrix, get_tag_to_camera_matrix

from typing import Dict, Tuple
from pydobotplus import Dobot
from pupil_apriltags import Detector
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R

def calc_calibration(device: Dobot,apriltag):
    dobot_T_gripper = get_dobot_to_gripper_matrix(device.get_pose())
    gripper_T_tag = get_gripper_to_tag_matrix()
    tag_T_camera = np.linalg.inv(get_tag_to_camera_matrix(apriltag))
    base_T_cam = dobot_T_gripper @ gripper_T_tag @ tag_T_camera
    return base_T_cam
        
def update_calib_yaml(base_T_cam, config_path: str = "configs/calibration.yaml"):
    translation = base_T_cam[:3, 3]
    rotation = base_T_cam[:3, :3]
    translation_list = translation.tolist()
    rotation_list = rotation.tolist()
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    data.setdefault("calibration", {}).setdefault("camera_to_robot", {})
    data["calibration"]["camera_to_robot"]["translation"] = translation_list
    data["calibration"]["camera_to_robot"]["rotation_matrix"] = rotation_list
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
        
def load_calib_yaml():
    with open("../../configs/calibration.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    translation = np.array(data["calibration"]["camera_to_robot"]["translation"], dtype=float)
    rotation = np.array(data["calibration"]["camera_to_robot"]["rotation_matrix"], dtype=float)

    base_T_cam = np.eye(4, dtype=float)
    base_T_cam[:3, :3] = rotation
    base_T_cam[:3, 3] = translation
    return base_T_cam

def get_target_coords(base_T_cam,P_camera):
    rotation = base_T_cam[:3,:3]
    translation = base_T_cam[:3,3]
    new_coords = rotation @ P_camera + translation
    return new_coords[:3]