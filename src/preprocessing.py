"""
Factory Image Preprocessing Pipeline
=====================================
Handles: noise removal, lighting correction, sharpening, normalization
Input  : raw blurry/noisy factory images
Output : clean, model-ready images + quality log
"""

import cv2
import numpy as np
import os
import json
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — IMAGE QUALITY ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════

def compute_blur_score(image: np.ndarray) -> float:
    """
    Laplacian variance — higher = sharper.
    Score < 100 means image is blurry and needs sharpening.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_noise_level(image: np.ndarray) -> float:
    """
    Estimate noise using high-frequency residuals.
    Higher value = more noise present.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Subtract a median-filtered version to isolate noise
    median = cv2.medianBlur(gray, 5)
    noise = np.std(gray.astype(np.float32) - median.astype(np.float32))
    return float(noise)


def assess_quality(image: np.ndarray) -> dict:
    """Returns quality metrics before processing."""
    return {
        "blur_score": compute_blur_score(image),
        "noise_level": compute_noise_level(image),
        "mean_brightness": float(np.mean(image)),
        "std_brightness": float(np.std(image)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — NOISE REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

def remove_noise(image: np.ndarray, noise_level: float) -> np.ndarray:
    """
    - Non-Local Means Denoising  : preserves edges & texture (critical for defects)
    - Median Filter on top       : removes salt-and-pepper noise from sensors
    
    Filter strength adapts to measured noise level.
    """
    # Adaptive strength: more noise → stronger filter (capped to avoid over-smoothing)
    h_strength = int(np.clip(noise_level * 1.5, 5, 20))

    # Non-Local Means (color image)
    if len(image.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=h_strength,          # luminance filter strength
            hColor=h_strength,     # color filter strength
            templateWindowSize=7,
            searchWindowSize=21
        )
    else:
        denoised = cv2.fastNlMeansDenoising(image, None, h=h_strength)

    # Median filter to remove remaining salt-and-pepper
    denoised = cv2.medianBlur(denoised, 3)

    return denoised


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — LIGHTING / ILLUMINATION CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def correct_lighting(image: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) on L-channel.
    Works in LAB color space so color is not distorted.
    
    Handles uneven lighting across factory floor (shadows, overexposure).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # clipLimit=2.0 prevents over-amplification of noise
    # tileGridSize=(8,8) divides image into local regions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_corrected = clahe.apply(l_channel)

    merged = cv2.merge([l_corrected, a, b])
    corrected = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return corrected


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — SHARPENING (Blur Fix)
# ══════════════════════════════════════════════════════════════════════════════

def sharpen_image(image: np.ndarray, blur_score: float) -> np.ndarray:
    """
    Unsharp Masking — recovers edge detail lost during denoising.
    Strength is adaptive: more blurry → stronger sharpening.
    
    Only applied if blur_score < 150 (image actually needs it).
    """
    if blur_score >= 150:
        return image  # Already sharp enough, skip

    # Adaptive sigma: blurrier images need wider blur radius
    sigma = np.clip(1.5 + (100 - min(blur_score, 100)) / 100, 1.0, 3.0)
    amount = np.clip(1.5 - blur_score / 200, 0.5, 2.0)

    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)

    return sharpened


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Pixel values scaled to [0, 255] uint8 range (for saving).
    For model input, use normalize_for_model() separately.
    """
    norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def normalize_for_model(image: np.ndarray) -> np.ndarray:
    """
    Converts to float32 [0.0, 1.0] for neural network input.
    Uses ImageNet mean/std (standard for pretrained models).
    """
    img = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img = (img - mean) / std
    return img


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — Single Image
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_path: str, output_dir: str) -> dict:
    """
    Full pipeline for one image:
    Assess → Denoise → Fix Lighting → Sharpen → Normalize → Save → Log
    
    Returns quality metrics (before & after) for audit logging.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        log.warning(f"Could not read: {image_path.name}")
        return {"file": image_path.name, "status": "failed", "reason": "unreadable"}

    # ── Step 1: Assess quality before processing ──
    before_metrics = assess_quality(image)

    # ── Step 2: Remove noise ──
    cleaned = remove_noise(image, before_metrics["noise_level"])

    # ── Step 3: Fix lighting ──
    cleaned = correct_lighting(cleaned)

    # ── Step 4: Sharpen ──
    cleaned = sharpen_image(cleaned, before_metrics["blur_score"])

    # ── Step 5: Normalize ──
    cleaned = normalize_image(cleaned)

    # ── Save processed image ──
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), cleaned)

    # ── Step 6: Assess quality after processing ──
    after_metrics = assess_quality(cleaned)

    return {
        "file": image_path.name,
        "status": "success",
        "before": before_metrics,
        "after": after_metrics,
        "improvement": {
            "blur_delta": round(after_metrics["blur_score"] - before_metrics["blur_score"], 2),
            "noise_delta": round(before_metrics["noise_level"] - after_metrics["noise_level"], 2),
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING — 5,000 Images
# ══════════════════════════════════════════════════════════════════════════════

def _worker(args):
    """Multiprocessing worker — wraps preprocess_image."""
    return preprocess_image(*args)


def preprocess_batch(input_dir: str, output_dir: str, workers: int = None) -> None:
    """
    Processes all images in input_dir using parallel workers.
    
    - Auto-detects CPU count for optimal speed
    - Progress bar via tqdm
    - Saves full quality audit log to JSON
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    # Collect all image files
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

    if not image_files:
        log.error(f"No images found in {input_dir}")
        return

    log.info(f"Found {len(image_files)} images — starting batch preprocessing...")

    # Use all available CPUs minus 1 (leave one for OS)
    num_workers = workers or max(1, cpu_count() - 1)
    log.info(f"Using {num_workers} parallel workers")

    args = [(str(f), str(output_dir)) for f in image_files]

    results = []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(_worker, args), total=len(args), desc="Preprocessing"):
            results.append(result)

    # ── Save audit log ──
    log_path = output_dir / "preprocessing_log.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Summary ──
    success = sum(1 for r in results if r["status"] == "success")
    failed  = len(results) - success

    log.info(f"Done. Success: {success} | Failed: {failed}")
    log.info(f"Quality log saved -> {log_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    preprocess_batch(
        input_dir="data/raw",
        output_dir="data/processed"
    )
