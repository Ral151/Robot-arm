"""AprilTag pose estimation for hand-eye calibration."""

from typing import Dict, Optional, Tuple

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import pupil_apriltags as apriltag  # type: ignore
    _APRILTAG_AVAILABLE = True
except ImportError:
    _APRILTAG_AVAILABLE = False
    logger.warning("pupil-apriltags not installed – AprilTag detection unavailable.")


class AprilTagPoseEstimator:
    """Detect AprilTags in a frame and estimate their 6-DoF pose."""

    def __init__(self, config: dict, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        calib_cfg = config.get("calibration", {})
        tag_cfg = calib_cfg.get("apriltag", {})
        self._family: str = tag_cfg.get("family", "tag36h11")
        self._tag_size: float = tag_cfg.get("tag_size", 0.05)
        self._tag_id: int = tag_cfg.get("tag_id", 0)
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs

        if _APRILTAG_AVAILABLE:
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            self._detector = apriltag.Detector(
                families=self._family,
                quad_decimate=1.0,
            )
            self._camera_params = (fx, fy, cx, cy)
        else:
            self._detector = None
            self._camera_params = None

    def detect(self, gray_frame: np.ndarray) -> Optional[Dict]:
        """Detect the reference AprilTag and return its pose.

        Args:
            gray_frame: Greyscale image as a numpy array.

        Returns:
            A dict with ``pose_R`` (3×3 rotation matrix) and ``pose_t`` (3-vector
            translation in metres), or ``None`` if the tag was not found.
        """
        if self._detector is None:
            return None

        results = self._detector.detect(
            gray_frame,
            estimate_tag_pose=True,
            camera_params=self._camera_params,
            tag_size=self._tag_size,
        )

        for tag in results:
            if tag.tag_id == self._tag_id:
                return {
                    "pose_R": tag.pose_R,
                    "pose_t": tag.pose_t,
                    "corners": tag.corners,
                    "center": tag.center,
                }
        return None
