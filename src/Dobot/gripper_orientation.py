import numpy as np

def get_gripper_orientation(self) -> float | None:
    """Returns current gripper orientation in degrees, or None if disconnected."""
    if self._device is None:
        logger.warning("get_gripper_orientation called but Dobot is not connected.")
        return None

    pose = self._device.get_pose()
    theta = np.arctan2(pose.position.y, pose.position.x)
    
    logger.debug(f"Gripper orientation: {np.degrees(theta):.2f}°")
    return np.degrees(theta)