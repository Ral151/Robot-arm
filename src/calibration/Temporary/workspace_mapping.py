"""Map pixel coordinates to robot workspace coordinates using a planar homography."""

from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class WorkspaceMapper:
    """Computes and applies a planar homography between image and robot workspace.

    The mapper is initialised with four image-space and robot-space point
    correspondences, from which it computes an OpenCV homography matrix.
    """

    def __init__(
        self,
        image_points: np.ndarray,
        robot_points: np.ndarray,
    ) -> None:
        """
        Args:
            image_points: (4, 2) array of pixel coordinates.
            robot_points: (4, 2) array of robot XY coordinates (mm).
        """
        self._H, mask = cv2.findHomography(image_points, robot_points)
        if self._H is None:
            raise ValueError("Homography computation failed – check your point correspondences.")
        logger.info("Workspace homography computed.")

    def image_to_robot(self, pixel: Tuple[float, float]) -> Tuple[float, float]:
        """Map a single image pixel to a robot XY coordinate.

        Args:
            pixel: (u, v) pixel coordinate.

        Returns:
            (robot_x, robot_y) in mm.
        """
        src = np.array([[[pixel[0], pixel[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self._H)
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])

    @classmethod
    def from_config(cls, config: dict) -> Optional["WorkspaceMapper"]:
        """Construct a WorkspaceMapper from the calibration config section.

        Returns ``None`` if the config does not contain mapping point pairs.
        """
        mapping = config.get("calibration", {}).get("workspace_mapping", None)
        if mapping is None:
            return None
        image_pts = np.array(mapping["image_points"], dtype=np.float32)
        robot_pts = np.array(mapping["robot_points"], dtype=np.float32)
        return cls(image_pts, robot_pts)
