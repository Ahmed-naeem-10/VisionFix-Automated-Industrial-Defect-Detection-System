"""
YOLOv8 Model Training
======================
Trains defect detection model on preprocessed factory images.
Auto-resumes if training was interrupted (laptop shutdown etc).
"""

import os
import yaml
import shutil
import logging
import random
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

CHECKPOINT = Path("runs/detect/models/defect_detector/weights/last.pt")
BEST_PT    = Path("runs/detect/models/defect_detector/weights/best.pt")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_yolo_dataset(images_dir, labels_dir, output_dir, val_split=0.2, classes=None):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    if classes is None:
        classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in extensions
        and (labels_dir / f.with_suffix(".txt").name).exists()
    ]

    if not all_images:
        raise FileNotFoundError(f"No labeled images found in {images_dir}")

    log.info(f"Found {len(all_images)} labeled images")

    random.shuffle(all_images)
    split_idx    = int(len(all_images) * (1 - val_split))
    train_images = all_images[:split_idx]
    val_images   = all_images[split_idx:]
    log.info(f"Train: {len(train_images)} | Val: {len(val_images)}")

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split_name, split_files in [("train", train_images), ("val", val_images)]:
        for img_path in split_files:
            lbl = labels_dir / img_path.with_suffix(".txt").name
            shutil.copy2(img_path, output_dir / "images" / split_name / img_path.name)
            shutil.copy2(lbl,      output_dir / "labels" / split_name / lbl.name)

    dataset_cfg = {
        "path":  str(output_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(classes),
        "names": classes
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False)

    log.info(f"Dataset YAML -> {yaml_path}")
    return str(yaml_path)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(config_path: str = "config.yaml") -> str:
    """
    Train YOLOv8. Auto-resumes from last.pt if interrupted.
    Returns path to best.pt weights.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    Path("logs").mkdir(parents=True, exist_ok=True)

    # ── Resume if checkpoint exists (e.g. laptop shutdown)
    if CHECKPOINT.exists():
        log.info(f"Checkpoint found -> resuming from {CHECKPOINT}")
        model = YOLO(str(CHECKPOINT))
        model.train(resume=True)
        log.info(f"Training complete. Best -> {BEST_PT}")
        return str(BEST_PT)

    # ── Fresh training
    dataset_yaml = prepare_yolo_dataset(
        images_dir=cfg["paths"]["processed_data"],
        labels_dir=cfg["paths"]["annotations"],
        output_dir="data/yolo_dataset",
        val_split=cfg["training"]["val_split"],
        classes=cfg["classes"]
    )

    log.info(f"Loading base model: {cfg['model']['pretrained_weights']}")
    model = YOLO(cfg["model"]["pretrained_weights"])

    log.info(f"Training for {cfg['training']['epochs']} epochs...")
    model.train(
        data=dataset_yaml,
        epochs=cfg["training"]["epochs"],
        batch=cfg["training"]["batch_size"],
        imgsz=cfg["training"]["image_size"],
        lr0=cfg["training"]["learning_rate"],
        patience=cfg["training"]["patience"],
        device=cfg["model"]["device"],
        project="runs/detect/models",
        name="defect_detector",
        save=True,
        plots=True,
        verbose=True
    )

    log.info(f"Training complete. Best -> {BEST_PT}")
    return str(BEST_PT)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(weights_path: str, dataset_yaml: str, config_path: str = "config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model   = YOLO(weights_path)
    metrics = model.val(
        data=dataset_yaml,
        imgsz=cfg["training"]["image_size"],
        device=cfg["model"]["device"],
        conf=cfg["model"]["confidence"],
        iou=cfg["model"]["iou_threshold"]
    )

    results = {
        "mAP50":     round(float(metrics.box.map50), 4),
        "mAP50_95":  round(float(metrics.box.map),   4),
        "precision": round(float(metrics.box.mp),    4),
        "recall":    round(float(metrics.box.mr),    4),
    }
    log.info(f"Evaluation: {results}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    best = train_model("config.yaml")
    if Path("data/yolo_dataset/dataset.yaml").exists():
        evaluate_model(best, "data/yolo_dataset/dataset.yaml")
