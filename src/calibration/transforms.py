"""Coordinate-frame transforms between camera space and robot space."""

from typing import Dict, Tuple

import numpy as np
import yaml

from utils.logger import get_logger

logger = get_logger(__name__)


class CameraToRobotTransform:
    """Applies a rigid-body transform from camera frame to robot base frame."""

    def __init__(self, rotation_matrix: np.ndarray, translation: np.ndarray) -> None:
        self._R = rotation_matrix          # 3×3
        self._t = translation.reshape(3)   # 3-vector (metres → converted to mm)

    def camera_to_robot(self, point_camera: np.ndarray) -> np.ndarray:
        """Transform a 3-D point from camera frame to robot base frame (mm).

        Args:
            point_camera: 3-D point in camera frame (metres).

        Returns:
            3-D point in robot base frame (mm).
        """
        point_m = self._R @ point_camera.reshape(3) + self._t
        return point_m * 1000.0  # convert metres → mm

    def image_to_robot(self, centroid: Tuple[float, float], z_camera: float = 0.5) -> Dict[str, float]:
        """Project a 2-D image centroid to a robot-frame pick coordinate.

        This method uses the stored rotation and translation to transform a
        camera-frame 3-D point to robot-base-frame coordinates.  The 3-D
        camera-frame point is obtained by back-projecting the pixel coordinate
        through the assumed depth ``z_camera`` using a simplified normalised
        approach.  For accurate results, replace this with a proper
        back-projection using the camera intrinsics
        (``point_cam = K_inv @ [u, v, 1] * depth``).

        Args:
            centroid: (u, v) pixel coordinates.
            z_camera: Assumed depth in the camera frame (metres).

        Returns:
            Dict with keys ``x``, ``y``, ``z`` in robot frame (mm).
        """
        # NOTE: Replace with proper intrinsic back-projection when camera
        # calibration parameters are available.  The normalised-coordinate
        # representation below is intentionally simple and should be updated.
        u, v = centroid
        point_cam = np.array([u / 1000.0, v / 1000.0, z_camera])
        robot_xyz = self.camera_to_robot(point_cam)
        return {"x": float(robot_xyz[0]), "y": float(robot_xyz[1]), "z": float(robot_xyz[2])}


def load_calibration(config: dict) -> CameraToRobotTransform:
    """Create a :class:`CameraToRobotTransform` from a calibration config dict.

    Args:
        config: Parsed contents of ``calibration.yaml``.

    Returns:
        A ready-to-use :class:`CameraToRobotTransform`.
    """
    calib = config.get("calibration", {})
    cam_to_robot = calib.get("camera_to_robot", {})
    translation = np.array(cam_to_robot.get("translation", [0.0, 0.0, 0.0]))
    rotation_matrix = np.array(cam_to_robot.get("rotation_matrix", np.eye(3).tolist()))
    logger.info("Calibration transform loaded.")
    return CameraToRobotTransform(rotation_matrix, translation)
