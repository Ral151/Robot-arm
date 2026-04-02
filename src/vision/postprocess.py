"""Post-processing helpers for detection results."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """Represents a single object detection result.

    Attributes:
        label: Class name string.
        confidence: Detection confidence in [0, 1].
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
    """

    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]

    def centroid(self, roi: Optional[List[int]] = None) -> Tuple[float, float]:
        """Return the centre pixel of the bounding box and convert it into full frame coordinates.
           roi = [ROI_x1, ROI_y1, ROI_x2, ROI_y2]
        """
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if roi is not None:
            cx += roi[0]
            cy += roi[1]
        return (cx, cy)

    @property
    def area(self) -> float:
        """Return the bounding-box area in pixels²."""
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def non_max_suppression(detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
    """Apply NMS to remove overlapping detections.

    Args:
        detections: List of :class:`Detection` objects.
        iou_threshold: IoU above which the lower-confidence box is suppressed.

    Returns:
        Filtered list of :class:`Detection` objects.
    """
    if not detections:
        return []

    boxes = np.array([d.bbox for d in detections], dtype=float)
    scores = np.array([d.confidence for d in detections], dtype=float)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        inter_x1 = np.maximum(x1[i], x1[order[1:]])
        inter_y1 = np.maximum(y1[i], y1[order[1:]])
        inter_x2 = np.minimum(x2[i], x2[order[1:]])
        inter_y2 = np.minimum(y2[i], y2[order[1:]])
        inter_area = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
        iou = inter_area / (areas[i] + areas[order[1:]] - inter_area)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return [detections[idx] for idx in keep]
