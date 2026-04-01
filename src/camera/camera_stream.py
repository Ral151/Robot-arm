import threading
from typing import Optional
import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False


class CameraStream:
    """Thread-safe RealSense camera frame grabber."""

    def __init__(self, config: dict) -> None:
        cam_cfg = config.get("camera", {})
        self._width: int = cam_cfg.get("width", 640)
        self._height: int = cam_cfg.get("height", 480)
        self._fps: int = cam_cfg.get("fps", 30)
        self._flip: bool = cam_cfg.get("flip", False)
        self._camera_serial: Optional[str] = cam_cfg.get("serial", 0)
        roi_cfg = cam_cfg.get("roi", {})
        self._roi = [
            roi_cfg.get("x1", 325),
            roi_cfg.get("y1", 201),
            roi_cfg.get("x2", 556),
            roi_cfg.get("y2", 413),
        ]

        # RealSense pipeline
        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._color_frame: Optional[np.ndarray] = None
        self._depth_frame: Optional[np.ndarray] = None
        self._intrinsics: Optional[rs.intrinsics] = None
        self._roi_frame = None

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Open the RealSense camera and start the background capture thread."""
        if not HAS_REALSENSE:
            raise RuntimeError("pyrealsense2 not installed. Install via: pip install pyrealsense2")

        self._pipeline = rs.pipeline()
        config = rs.config()

        if self._camera_serial:
            config.enable_device(self._camera_serial)

        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
        config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)

        profile = self._pipeline.start(config)

        # Align depth to color frame
        self._align = rs.align(rs.stream.color)

        # Extract color intrinsics
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._intrinsics = color_stream.get_intrinsics()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        """Background thread that continuously grabs frames from RealSense."""
        roi_x1, roi_y1, roi_x2, roi_y2 = self._roi

        while self._running:
            frames = self._pipeline.wait_for_frames()
            aligned_frames = self._align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if color_frame and depth_frame:
                color_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data())
                roi = color_img[roi_y1:roi_y2, roi_x1:roi_x2]
                display_roi = roi.copy()
                color_full_roi = cv2.rectangle(color_img.copy(),(roi_x1, roi_y1),(roi_x2, roi_y2),(255, 0, 0),2,)

                if self._flip:
                    color_img = np.fliplr(color_img)
                    depth_img = np.fliplr(depth_img)

                with self._lock:
                    self._color_frame = color_full_roi
                    self._depth_frame = depth_img
                    self._roi_frame = display_roi
    def read(self) -> Optional[np.ndarray]:
        """Return the latest color frame (or None if not yet available)."""
        with self._lock:
            return self._color_frame.copy() if self._color_frame is not None else None

    def read_depth(self) -> Optional[np.ndarray]:
        """Return the latest depth frame."""
        with self._lock:
            return self._depth_frame.copy() if self._depth_frame is not None else None
    
    def read_roi(self) -> Optional[np.ndarray]:
        """Return the latest ROI color frame."""
        with self._lock:
            return self._roi_frame.copy() if self._roi_frame is not None else None

    def get_roi(self) -> list:
        """Return ROI bounds as [x1, y1, x2, y2]."""
        return list(self._roi)

    def get_intrinsics(self) -> Optional[rs.intrinsics]:
        """Return camera intrinsics (fx, fy, cx, cy, coeffs)."""
        return self._intrinsics

    def stop(self) -> None:
        """Stop the capture thread and release the camera resource."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pipeline is not None:
            self._pipeline.stop()
