"""General-purpose helper utilities."""

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], path: str) -> None:
    """Serialise a dictionary to a YAML file.

    Args:
        data: Dictionary to serialise.
        path: Destination file path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels onto a copy of *frame*.

    Args:
        frame: BGR image as a numpy array.
        detections: List of :class:`~vision.postprocess.Detection` objects.

    Returns:
        Annotated BGR image.
    """
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def save_frame(frame: np.ndarray, output_dir: str, filename: str) -> Path:
    """Save a frame to disk.

    Args:
        frame: BGR image as a numpy array.
        output_dir: Directory to write the image file.
        filename: File name (including extension).

    Returns:
        The resolved :class:`~pathlib.Path` of the saved file.
    """
    out_path = Path(output_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return out_path
