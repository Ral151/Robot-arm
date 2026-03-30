#Calculates the Transformation matrices needed for calibration

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R
from pydobotplus import Dobot

def get_robot_arm_matrix(pose):
    """
    Build a 4x4 transformation matrix (will transform point under gripper frame to base frame) in mm from a robot pose object.
    Assumes planar robot (rotation about Z only).
    """
    x = pose.position.x
    y = pose.position.y
    z = pose.position.z
    theta = np.arctan2(y,x)
    base_T_gripper = np.array([[np.cos(theta),np.sin(-theta),0,x],
                               [np.sin(theta),np.cos(theta),0,y],
                               [0,0,1,z],
                               [0,0,0,1]
                               ], dtype=np.float32)
    return base_T_gripper
def get_tag_to_gripper_matrix():
    """
    Build a 4x4 transformation matrix (will transform points under tag frame to gripper frame) in mm.
    Tag is 153mm below gripper, 30mm forward along gripper X.
    X-axes aligned, but Z-axis flipped.
    """
    gripper_T_tag = np.array([
            [-1,0, 0,  30],
            [ 0,1, 0,   0],
            [ 0,0,-1, 153],
            [ 0,0, 0,   1]
        ], dtype=np.float32)
    return gripper_T_tag
def get_tag_to_camera_matrix(tag):
    """
    Build a 4x4 transformation matrix (will transform points under tag frame to camera frame) in mm from a detected tag object.
    """
    cam_T_tag = np.eye(4)
    cam_T_tag[:3,:3] = tag.pose_R
    cam_T_tag[:3,3] = tag.pose_t.flatten() * 1000
    cam_T_tag[3][0] = -cam_T_tag[3][0]
    cam_T_tag[3][1] = -cam_T_tag[3][1]
    cam_T_tag[3][2] = -cam_T_tag[3][2]
    
    return cam_T_tag
