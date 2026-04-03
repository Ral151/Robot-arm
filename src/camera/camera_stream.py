import threading
from typing import Optional
import cv2
import numpy as np

from camera.rs_demo.realsense_utils import (
    initialize_pipeline,
    get_camera_intrinsics,
    get_aligned_frames,
    frames_to_numpy,
    pixel_to_3d,
    print_camera_info
)

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
        self._camera_serial: Optional[str] = cam_cfg.get("serial", 0)
        roi_cfg = cam_cfg.get("roi", {})
        self._roi = [
            roi_cfg.get("x1", 348),
            roi_cfg.get("y1", 217),
            roi_cfg.get("x2", 545),
            roi_cfg.get("y2", 423),
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

        self._pipeline, profile, _ = initialize_pipeline(
            width=self._width,
            height=self._height,
            fps=self._fps,
        )
        self._intrinsics = get_camera_intrinsics(profile)
        self._align = rs.align(rs.stream.color)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        """Background thread that continuously grabs frames from RealSense."""
        roi_x1, roi_y1, roi_x2, roi_y2 = self._roi

        while self._running:
            color_frame, depth_frame = get_aligned_frames(self._pipeline, self._align)

            if color_frame is not None and depth_frame is not None:
                color_img, depth_img = frames_to_numpy(color_frame, depth_frame)
                roi = color_img[roi_y1:roi_y2, roi_x1:roi_x2]
                depth_roi = depth_img[roi_y1:roi_y2, roi_x1:roi_x2]
                display_roi = roi.copy()
                color_full_roi = cv2.rectangle(color_img.copy(),(roi_x1, roi_y1),(roi_x2, roi_y2),(255, 0, 0),2,)

                with self._lock:
                    self._depth_roi = depth_roi
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
    
    def read_depth_roi(self) -> Optional[np.ndarray]:
        """Return the latest depth frame in ROI."""
        with self._lock:
            return self._depth_roi.copy() if self._depth_roi is not None else None
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
    
    def get_pipeline(self):
        return self._pipeline
    
    def get_align(self):
        return self._align

    def stop(self) -> None:
        """Stop the capture thread and release the camera resource."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pipeline is not None:
            self._pipeline.stop()
