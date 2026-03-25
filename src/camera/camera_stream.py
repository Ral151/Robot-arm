# import threading
# from typing import Optional

# import cv2
# import numpy as np


# class CameraStream:
#     """Thread-safe camera frame grabber."""

#     def __init__(self, config: dict) -> None:
#         cam_cfg = config.get("camera", {})
#         self._device_id: int = cam_cfg.get("device_id", 0)
#         self._width: int = cam_cfg.get("width", 1280)
#         self._height: int = cam_cfg.get("height", 720)
#         self._fps: int = cam_cfg.get("fps", 30)
#         self._flip: bool = cam_cfg.get("flip", False)

#         self._cap: Optional[cv2.VideoCapture] = None
#         self._frame: Optional[np.ndarray] = None
#         self._lock = threading.Lock()
#         self._running = False
#         self._thread: Optional[threading.Thread] = None

#     def start(self) -> None:
#         """Open the camera and start the background capture thread."""
#         self._cap = cv2.VideoCapture(self._device_id)
#         self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
#         self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
#         self._cap.set(cv2.CAP_PROP_FPS, self._fps)
#         if not self._cap.isOpened():
#             raise RuntimeError(f"Cannot open camera device {self._device_id}")
#         self._running = True
#         self._thread = threading.Thread(target=self._capture_loop, daemon=True)
#         self._thread.start()

#     def _capture_loop(self) -> None:
#         while self._running:
#             ret, frame = self._cap.read()
#             if ret:
#                 if self._flip:
#                     frame = cv2.flip(frame, 1)
#                 with self._lock:
#                     self._frame = frame

#     def read(self) -> Optional[np.ndarray]:
#         """Return the latest captured frame (or None if not yet available)."""
#         with self._lock:
#             return self._frame.copy() if self._frame is not None else None

#     def stop(self) -> None:
#         """Stop the capture thread and release the camera resource."""
#         self._running = False
#         if self._thread is not None:
#             self._thread.join(timeout=2.0)
#         if self._cap is not None:
#             self._cap.release()
