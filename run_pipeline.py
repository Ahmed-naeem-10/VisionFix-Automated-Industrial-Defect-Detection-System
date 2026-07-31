"""
VisionFix Pipeline — Master Runner
=====================================
Runs the complete pipeline end-to-end:
  1. Preprocess raw images
  2. Augment dataset
  3. Train model
  4. Run batch detection
  5. Launch dashboard

Usage:
    python run_pipeline.py --steps all
    python run_pipeline.py --steps preprocess detect
    python run_pipeline.py --steps dashboard
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# ─── Setup ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import setup_logging, ensure_dirs, load_config

setup_logging("logs")
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STEPS
# ══════════════════════════════════════════════════════════════════════════════

def step_preprocess(cfg: dict):
    log.info("=" * 50)
    log.info("STEP 1 — IMAGE PREPROCESSING")
    log.info("=" * 50)
    from preprocessing import preprocess_batch
    preprocess_batch(
        input_dir=cfg["paths"]["raw_data"],
        output_dir=cfg["paths"]["processed_data"],
        workers=cfg["preprocessing"]["workers"] or None
    )


def step_augment(cfg: dict):
    log.info("=" * 50)
    log.info("STEP 2 — DATA AUGMENTATION")
    log.info("=" * 50)
    from data_augmentation import augment_dataset
    augment_dataset(
        images_dir=cfg["paths"]["processed_data"],
        labels_dir=cfg["paths"]["annotations"],
        output_dir="data/augmented",
        copies_per_image=3
    )


def step_train(cfg: dict):
    log.info("=" * 50)
    log.info("STEP 3 — MODEL TRAINING")
    log.info("=" * 50)
    from train import train_model, evaluate_model
    best_weights = train_model("config.yaml")
    evaluate_model(best_weights, "data/yolo_dataset/dataset.yaml")


def step_detect(cfg: dict):
    log.info("=" * 50)
    log.info("STEP 4 — BATCH DETECTION")
    log.info("=" * 50)
    from detect import DefectDetector
    detector = DefectDetector("config.yaml")
    results  = detector.detect_batch(cfg["paths"]["processed_data"])

    total     = len(results)
    defective = sum(1 for r in results if r.get("status") == "defective")
    log.info(f"Detection complete | Total: {total} | Defective: {defective} | Good: {total - defective}")


def step_dashboard(cfg: dict):
    log.info("=" * 50)
    log.info("STEP 5 — LAUNCHING DASHBOARD")
    log.info("=" * 50)
    import subprocess
    host = cfg["dashboard"]["host"]
    port = cfg["dashboard"]["port"]
    log.info(f"Dashboard → http://{host}:{port}")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard.py",
        "--server.address", host,
        "--server.port", str(port)
    ])


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

STEP_MAP = {
    "preprocess": step_preprocess,
    "augment":    step_augment,
    "train":      step_train,
    "detect":     step_detect,
    "dashboard":  step_dashboard,
}

ALL_STEPS = ["preprocess", "augment", "train", "detect", "dashboard"]


def main():
    parser = argparse.ArgumentParser(
        description="VisionFix Pipeline — Factory Defect Detection"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS + ["all"],
        default=["all"],
        help="Steps to run (default: all)"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    steps = ALL_STEPS if "all" in args.steps else args.steps

    log.info(f"VisionFix Pipeline starting | Steps: {steps}")

    for step in steps:
        try:
            STEP_MAP[step](cfg)
        except Exception as e:
            log.error(f"Step '{step}' failed: {e}", exc_info=True)
            sys.exit(1)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
