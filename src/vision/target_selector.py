"""Select the best target from a list of detections."""

from typing import List, Optional

from vision.postprocess import Detection, non_max_suppression
from utils.logger import get_logger

logger = get_logger(__name__)


class TargetSelector:
    """Choose the single best detection to pick up.

    Selection priority (highest first):
    1. Highest confidence score.
    2. Largest bounding-box area (tiebreaker).
    """

    def __init__(self, classes_config: dict) -> None:
        priority = classes_config.get("priority", [])
        if priority:
            self._priority_classes = list(priority)
        else:
            self._priority_classes = list(classes_config.get("classes", {}).values())

    def select(self, detections: List[Detection]) -> Optional[Detection]:
        """Return the best target from a list of detections.

        Args:
            detections: Raw detections from the :class:`~vision.detector.Detector`.

        Returns:
            The best :class:`~vision.postprocess.Detection`, or ``None`` if the
            list is empty after NMS.
        """
        filtered = non_max_suppression(detections)
        if not filtered:
            return None

        # Sort by (priority index ascending, confidence descending, area descending)
        def sort_key(d: Detection):
            try:
                priority = self._priority_classes.index(d.label)
            except ValueError:
                priority = len(self._priority_classes)
            return (priority, -d.confidence, -d.area)

        filtered.sort(key=sort_key)
        target = filtered[0]
        logger.debug(f"Selected target: {target.label} (conf={target.confidence:.2f})")
        return target
