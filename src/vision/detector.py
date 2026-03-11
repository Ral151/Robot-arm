"""YOLO-based object detector wrapper."""

from pathlib import Path
from typing import List

import numpy as np

from vision.postprocess import Detection
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from ultralytics import YOLO  # type: ignore
    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False
    logger.warning("ultralytics not installed – detector running in stub mode.")


class Detector:
    """Wraps an Ultralytics YOLO model for object detection."""

    def __init__(self, model_path: str, classes_config: dict) -> None:
        self._classes: dict = classes_config.get("classes", {})
        self._thresholds: dict = classes_config.get("thresholds", {})
        self._default_threshold: float = self._thresholds.get("default", 0.5)
        self._model = None

        if _ULTRALYTICS_AVAILABLE and Path(model_path).exists():
            self._model = YOLO(model_path)
            logger.info(f"YOLO model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path} – using stub detector.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a BGR frame and return a list of :class:`Detection` objects.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            List of :class:`~vision.postprocess.Detection` instances.
        """
        if self._model is None:
            return []

        results = self._model(frame, verbose=False)
        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self._classes.get(cls_id, str(cls_id))
                threshold = self._thresholds.get(label, self._default_threshold)
                if conf < threshold:
                    continue
                xyxy = box.xyxy[0].cpu().numpy()
                detections.append(
                    Detection(
                        label=label,
                        confidence=conf,
                        bbox=tuple(xyxy.tolist()),
                    )
                )
        return detections
