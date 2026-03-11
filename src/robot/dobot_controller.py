"""Dobot Magician controller wrapper."""

from typing import Dict

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from pydobot import Dobot  # type: ignore
    _DOBOT_AVAILABLE = True
except ImportError:
    _DOBOT_AVAILABLE = False
    logger.warning("pydobot not installed – running in simulation mode.")


class DobotController:
    """High-level interface for the Dobot Magician robot arm."""

    def __init__(self, config: dict) -> None:
        robot_cfg = config.get("robot", {})
        self._port: str = robot_cfg.get("port", "/dev/ttyUSB0")
        self._speed: int = robot_cfg.get("speed", 50)
        self._acceleration: int = robot_cfg.get("acceleration", 50)
        self._home: Dict[str, float] = robot_cfg.get(
            "home_position", {"x": 200.0, "y": 0.0, "z": 50.0, "r": 0.0}
        )
        self._gripper_close_delay: float = robot_cfg.get("gripper", {}).get("close_delay", 0.5)
        self._gripper_open_delay: float = robot_cfg.get("gripper", {}).get("open_delay", 0.3)
        self._gripper_close_value: int = robot_cfg.get("gripper", {}).get("close_value", 480)
        self._gripper_open_value: int = robot_cfg.get("gripper", {}).get("open_value", 200)
        self._device = None
        self._connect()

    def _connect(self) -> None:
        if _DOBOT_AVAILABLE:
            try:
                self._device = Dobot(port=self._port)
                logger.info(f"Connected to Dobot on {self._port}")
            except Exception as exc:
                logger.error(f"Failed to connect to Dobot: {exc}")
                self._device = None
        else:
            logger.info("Simulation mode: Dobot not connected.")

    def home(self) -> None:
        """Move the arm to the configured home position."""
        h = self._home
        self.move_to(h["x"], h["y"], h["z"], h.get("r", 0.0))
        logger.info("Moved to home position.")

    def move_to(self, x: float, y: float, z: float, r: float = 0.0) -> None:
        """Move the end-effector to (x, y, z, r) in robot coordinates (mm)."""
        logger.debug(f"move_to({x:.1f}, {y:.1f}, {z:.1f}, {r:.1f})")
        if self._device is not None:
            self._device.move_to(x, y, z, r, wait=True)

    def set_gripper(self, close: bool) -> None:
        """Open or close the gripper.
        
        Args:
            close: True to close/grip, False to open/release
        """
        if self._device is not None:
            # Dobot gripper control: higher value = more closed
            value = self._gripper_close_value if close else self._gripper_open_value
            self._device.grip(value)
            logger.debug(f"Gripper {'closed' if close else 'opened'} (value={value})")

    def pick(self, coords: Dict[str, float]) -> None:
        """Lower to the target, close gripper to grab, then lift."""
        import time
        x, y, z = coords["x"], coords["y"], coords["z"]
        r = coords.get("r", 0.0)
        # Open gripper before approaching
        self.set_gripper(False)
        time.sleep(self._gripper_open_delay)
        # Move above target
        self.move_to(x, y, z + 30, r)
        # Lower to target
        self.move_to(x, y, z, r)
        # Close gripper to grab
        self.set_gripper(True)
        time.sleep(self._gripper_close_delay)
        # Lift
        self.move_to(x, y, z + 30, r)
        logger.info(f"Picked object at ({x:.1f}, {y:.1f}, {z:.1f})")

    def place(self, coords: Dict[str, float]) -> None:
        """Move to the drop position and open gripper to release."""
        import time
        x, y, z = coords["x"], coords["y"], coords["z"]
        r = coords.get("r", 0.0)
        self.move_to(x, y, z + 30, r)
        self.move_to(x, y, z, r)
        # Open gripper to release
        self.set_gripper(False)
        time.sleep(self._gripper_open_delay)
        self.move_to(x, y, z + 30, r)
        logger.info(f"Placed object at ({x:.1f}, {y:.1f}, {z:.1f})")

    def disconnect(self) -> None:
        if self._device is not None:
            self._device.close()
            logger.info("Dobot disconnected.")
