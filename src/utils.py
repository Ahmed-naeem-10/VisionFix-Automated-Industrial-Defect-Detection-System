"""
Utility Functions
==================
Shared helpers used across preprocessing, detection, and dashboard.
"""

import cv2
import json
import yaml
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# FILE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return YAML config as dict."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: dict) -> None:
    """Create all project directories defined in config."""
    for key, path in cfg["paths"].items():
        Path(path).mkdir(parents=True, exist_ok=True)
    log.info("All project directories ensured.")


def list_images(directory: str, extensions: set = None) -> list:
    """Return sorted list of image file paths in a directory."""
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return sorted(
        [str(f) for f in Path(directory).iterdir() if f.suffix.lower() in extensions]
    )


def save_json(data: dict | list, path: str) -> None:
    """Save data to a JSON file, creating parent dirs as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict | list:
    """Load and return JSON file content."""
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """Resize image to (width, height) using INTER_LINEAR."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def pad_to_square(image: np.ndarray, fill: int = 114) -> np.ndarray:
    """
    Pad image to square (letterbox style).
    Used before feeding to fixed-input-size models.
    """
    h, w = image.shape[:2]
    size = max(h, w)
    padded = np.full((size, size, 3), fill, dtype=np.uint8)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    padded[y_off:y_off + h, x_off:x_off + w] = image
    return padded


def draw_text_with_bg(
    image: np.ndarray, text: str, position: tuple,
    font_scale: float = 0.6, thickness: int = 1,
    text_color=(255, 255, 255), bg_color=(0, 0, 0)
) -> np.ndarray:
    """Draw text with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(image, (x, y - th - baseline - 4), (x + tw + 4, y + baseline), bg_color, -1)
    cv2.putText(image, text, (x + 2, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return image


# ══════════════════════════════════════════════════════════════════════════════
# METRICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_iou(box1: list, box2: list) -> float:
    """
    Compute Intersection over Union between two boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return round(intersection / union, 4) if union > 0 else 0.0


def detection_summary(results: list) -> dict:
    """
    Compute aggregate stats from a list of detection result dicts.
    Works with output from detect.py.
    """
    if not results:
        return {}

    total      = len(results)
    defective  = sum(1 for r in results if r.get("status") == "defective")
    all_dets   = [d for r in results for d in r.get("detections", [])]
    defect_dets = [d for d in all_dets if d.get("is_defect")]

    from collections import Counter
    class_counts = Counter(d["class_name"] for d in all_dets)
    avg_conf     = round(np.mean([d["confidence"] for d in all_dets]), 4) if all_dets else 0.0
    avg_inf      = round(np.mean([r.get("inference_ms", 0) for r in results]), 1)

    return {
        "total_images":      total,
        "defective_images":  defective,
        "good_images":       total - defective,
        "defect_rate_pct":   round(defective / total * 100, 2),
        "total_detections":  len(all_dets),
        "total_defects":     len(defect_dets),
        "class_counts":      dict(class_counts),
        "avg_confidence":    avg_conf,
        "avg_inference_ms":  avg_inf,
        "generated_at":      datetime.now().isoformat()
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """Configure root logger with file + console handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"visionfix_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
