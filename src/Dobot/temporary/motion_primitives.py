"""Reusable motion primitives built on top of DobotController."""

from typing import Dict, List

from src.Dobot.Dobot_movement import DobotController
from utils.logger import get_logger

logger = get_logger(__name__)


def linear_move_sequence(
    robot: DobotController, waypoints: List[Dict[str, float]]
) -> None:
    """Execute a sequence of linear moves through a list of waypoints.

    Args:
        robot: An initialised DobotController instance.
        waypoints: List of dicts with keys ``x``, ``y``, ``z``, and optional ``r``.
    """
    for idx, wp in enumerate(waypoints):
        logger.debug(f"Waypoint {idx}: {wp}")
        robot.move_to(wp["x"], wp["y"], wp["z"], wp.get("r", 0.0))


def pick_and_place(
    robot: DobotController,
    pick_coords: Dict[str, float],
    place_coords: Dict[str, float],
) -> None:
    """Convenience wrapper that calls pick then place.

    Args:
        robot: An initialised DobotController instance.
        pick_coords: Target pick position (x, y, z[, r]) in robot frame (mm).
        place_coords: Target place position (x, y, z[, r]) in robot frame (mm).
    """
    robot.pick(pick_coords)
    robot.place(place_coords)
    logger.info("pick_and_place sequence complete.")


def sweep_workspace(
    robot: DobotController,
    x_range: tuple,
    y_range: tuple,
    z: float,
    steps: int = 5,
) -> None:
    """Sweep the arm across the workspace in a raster pattern (for inspection).

    Args:
        robot: An initialised DobotController instance.
        x_range: (x_min, x_max) in mm.
        y_range: (y_min, y_max) in mm.
        z: Height at which to sweep (mm).
        steps: Number of steps along each axis.
    """
    import numpy as np

    xs = np.linspace(x_range[0], x_range[1], steps)
    ys = np.linspace(y_range[0], y_range[1], steps)

    for i, x in enumerate(xs):
        row_ys = list(ys) if i % 2 == 0 else list(reversed(ys))
        for y in row_ys:
            robot.move_to(float(x), float(y), z)
    logger.info("Workspace sweep complete.")
