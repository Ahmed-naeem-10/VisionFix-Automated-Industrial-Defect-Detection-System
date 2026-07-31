"""
Data Augmentation Pipeline
============================
Augments labeled factory images to expand the dataset.
Applies factory-relevant transforms: flips, rotations, brightness,
blur simulation, and noise injection.

Input : images + YOLO format labels
Output: augmented copies in output_dir with updated labels
"""

import cv2
import numpy as np
import random
import os
import logging
from pathlib import Path
from tqdm import tqdm

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# YOLO LABEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_yolo_labels(label_path: str) -> list:
    """Load YOLO format labels. Returns list of [cls, cx, cy, w, h]."""
    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return labels


def save_yolo_labels(labels: list, path: str) -> None:
    """Save YOLO format labels to file."""
    with open(path, "w") as f:
        for label in labels:
            f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


def flip_labels_h(labels: list) -> list:
    """Adjust YOLO cx for horizontal flip."""
    return [[l[0], 1.0 - l[1], l[2], l[3], l[4]] for l in labels]


def flip_labels_v(labels: list) -> list:
    """Adjust YOLO cy for vertical flip."""
    return [[l[0], l[1], 1.0 - l[2], l[3], l[4]] for l in labels]


def rotate_labels_90(labels: list, direction: str = "cw") -> list:
    """
    Rotate YOLO bounding boxes 90 degrees.
    direction: "cw" (clockwise) or "ccw" (counter-clockwise)
    """
    rotated = []
    for l in labels:
        cls, cx, cy, w, h = l
        if direction == "cw":
            rotated.append([cls, 1.0 - cy, cx, h, w])
        else:
            rotated.append([cls, cy, 1.0 - cx, h, w])
    return rotated


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

def augment_flip_horizontal(image: np.ndarray, labels: list):
    return cv2.flip(image, 1), flip_labels_h(labels)


def augment_flip_vertical(image: np.ndarray, labels: list):
    return cv2.flip(image, 0), flip_labels_v(labels)


def augment_rotate_90(image: np.ndarray, labels: list, direction: str = "cw"):
    if direction == "cw":
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated, rotate_labels_90(labels, direction)


def augment_brightness(image: np.ndarray, labels: list, factor: float = None):
    """Random brightness adjustment — simulates factory lighting variation."""
    if factor is None:
        factor = random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), labels


def augment_gaussian_noise(image: np.ndarray, labels: list, std: int = None):
    """Add Gaussian noise — simulates camera sensor noise."""
    if std is None:
        std = random.randint(5, 25)
    noise = np.random.normal(0, std, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy, labels


def augment_blur(image: np.ndarray, labels: list, kernel: int = None):
    """Apply motion/defocus blur — simulates camera movement."""
    if kernel is None:
        kernel = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (kernel, kernel), 0), labels


def augment_contrast(image: np.ndarray, labels: list, alpha: float = None):
    """Adjust contrast: alpha > 1 increases, < 1 decreases."""
    if alpha is None:
        alpha = random.uniform(0.7, 1.4)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0), labels


def augment_hsv_shift(image: np.ndarray, labels: list):
    """Random hue/saturation shift — simulates color variation."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-30, 30), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), labels


# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

# Each transform: (function, probability, suffix)
TRANSFORMS = [
    (augment_flip_horizontal, 0.5,  "flip_h"),
    (augment_flip_vertical,   0.3,  "flip_v"),
    (augment_brightness,      0.6,  "bright"),
    (augment_gaussian_noise,  0.4,  "noise"),
    (augment_blur,            0.3,  "blur"),
    (augment_contrast,        0.5,  "contrast"),
    (augment_hsv_shift,       0.4,  "hsv"),
    (lambda i, l: augment_rotate_90(i, l, "cw"),  0.2, "rot90cw"),
    (lambda i, l: augment_rotate_90(i, l, "ccw"), 0.2, "rot90ccw"),
]


def augment_image(
    image_path: str,
    label_path: str,
    output_images_dir: str,
    output_labels_dir: str,
    copies: int = 3
) -> int:
    """
    Apply random combination of transforms to one image.
    Generates `copies` augmented versions.
    Returns number of new images created.
    """
    image  = cv2.imread(image_path)
    labels = load_yolo_labels(label_path)

    if image is None:
        log.warning(f"Cannot read: {image_path}")
        return 0

    stem = Path(image_path).stem
    ext  = Path(image_path).suffix

    created = 0
    for i in range(copies):
        aug_image  = image.copy()
        aug_labels = [l.copy() for l in labels]
        applied    = []

        for transform, prob, suffix in TRANSFORMS:
            if random.random() < prob:
                try:
                    aug_image, aug_labels = transform(aug_image, aug_labels)
                    applied.append(suffix)
                except Exception as e:
                    log.debug(f"Transform failed ({suffix}): {e}")

        suffix_str = "_".join(applied) if applied else "orig"
        out_name   = f"{stem}_aug{i}_{suffix_str}"

        out_img_path   = Path(output_images_dir) / f"{out_name}{ext}"
        out_label_path = Path(output_labels_dir) / f"{out_name}.txt"

        cv2.imwrite(str(out_img_path), aug_image)
        save_yolo_labels(aug_labels, str(out_label_path))
        created += 1

    return created


def augment_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    copies_per_image: int = 3
) -> None:
    """
    Augment entire dataset.
    Copies originals + generates augmented versions.

    Args:
        images_dir       : path to original images
        labels_dir       : path to YOLO label files
        output_dir       : base output directory
        copies_per_image : augmented copies per original image
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_imgs   = Path(output_dir) / "images"
    out_lbls   = Path(output_dir) / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in extensions]

    if not image_files:
        log.error(f"No images found in {images_dir}")
        return

    log.info(f"Augmenting {len(image_files)} images × {copies_per_image} copies...")

    total_new = 0
    for img_path in tqdm(image_files, desc="Augmenting"):
        label_path = labels_dir / img_path.with_suffix(".txt").name

        # Copy original
        import shutil
        shutil.copy2(img_path, out_imgs / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, out_lbls / label_path.name)

        # Generate augmented versions
        if label_path.exists():
            n = augment_image(
                str(img_path), str(label_path),
                str(out_imgs), str(out_lbls),
                copies=copies_per_image
            )
            total_new += n

    original = len(image_files)
    log.info(f"Done. Original: {original} | New: {total_new} | Total: {original + total_new}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    augment_dataset(
        images_dir="data/processed",
        labels_dir="data/annotations",
        output_dir="data/augmented",
        copies_per_image=3
    )
