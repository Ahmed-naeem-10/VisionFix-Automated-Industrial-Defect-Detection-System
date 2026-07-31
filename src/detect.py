"""
Real-Time Defect Detection Engine
===================================
Runs YOLOv8 inference on:
  - Single images
  - Folders (batch)
  - Live webcam / RTSP camera stream
Outputs annotated images + structured JSON results.
"""

import cv2
import yaml
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from preprocessing import preprocess_image, normalize_for_model

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/detection.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DEFECT DETECTOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

class DefectDetector:
    """
    Main detection engine wrapping YOLOv8.
    Handles preprocessing + inference + result formatting.
    """

    # Color map per class (BGR for OpenCV) — NEU-DET classes
    CLASS_COLORS = {
        "crazing":          (0, 165, 255),    # orange
        "inclusion":        (0, 0, 255),      # red
        "patches":          (255, 0, 255),    # magenta
        "pitted_surface":   (0, 0, 180),      # dark red
        "rolled-in_scale":  (255, 0, 0),      # blue
        "scratches":        (0, 200, 255),    # yellow
    }

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        weights = self.cfg["model"]["weights"]
        if not Path(weights).exists():
            log.warning(f"Trained weights not found at {weights}. Using pretrained base.")
            weights = self.cfg["model"]["pretrained_weights"]

        self.model      = YOLO(weights)
        self.conf       = self.cfg["model"]["confidence"]
        self.iou        = self.cfg["model"]["iou_threshold"]
        self.device     = self.cfg["model"]["device"]
        self.imgsz      = self.cfg["model"]["input_size"]
        self.classes    = self.cfg["classes"]
        self.results_dir = Path(self.cfg["paths"]["results"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"DefectDetector ready | Model: {weights} | Device: {self.device}")


    # ── Preprocessing wrapper ──────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to a raw image array.
        (noise removal → lighting fix → sharpening → normalize)
        """
        from preprocessing import (
            assess_quality, remove_noise,
            correct_lighting, sharpen_image, normalize_image
        )
        metrics  = assess_quality(image)
        image    = remove_noise(image, metrics["noise_level"])
        image    = correct_lighting(image)
        image    = sharpen_image(image, metrics["blur_score"])
        image    = normalize_image(image)
        return image


    # ── Core inference ─────────────────────────────────────────────────────────

    def _run_inference(self, image: np.ndarray) -> list:
        """
        Runs YOLOv8 on a single BGR image.
        Returns list of detection dicts.
        """
        results = self.model(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id  = int(box.cls[0])
                cls_name = self.classes[cls_id] if cls_id < len(self.classes) else f"class_{cls_id}"
                conf     = round(float(box.conf[0]), 4)

                detections.append({
                    "class_id":   cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox":       [x1, y1, x2, y2],
                    "is_defect":  True   # NEU-DET: all classes are defects
                })

        return detections


    # ── Annotation ─────────────────────────────────────────────────────────────

    def _annotate(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draws bounding boxes and labels on image."""
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label  = f"{det['class_name']} {det['confidence']:.2f}"
            color  = self.CLASS_COLORS.get(det["class_name"], (200, 200, 200))

            # Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

            # Label text
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Status banner
        defects = [d for d in detections if d["is_defect"]]
        status  = f"DEFECTS: {len(defects)}" if defects else "OK"
        color   = (0, 0, 220) if defects else (0, 180, 0)
        cv2.putText(annotated, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        return annotated


    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def detect_image(self, image_path: str, save: bool = True) -> dict:
        """
        Run detection on a single image file.
        Returns structured result dict.
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))

        if image is None:
            log.error(f"Cannot read image: {image_path}")
            return {"status": "error", "file": str(image_path)}

        start = time.time()
        preprocessed = self._preprocess(image)
        detections   = self._run_inference(preprocessed)
        elapsed_ms   = round((time.time() - start) * 1000, 1)

        annotated = self._annotate(preprocessed, detections)

        result = {
            "timestamp":    datetime.now().isoformat(),
            "file":         image_path.name,
            "inference_ms": elapsed_ms,
            "total_detections": len(detections),
            "defect_count": sum(1 for d in detections if d["is_defect"]),
            "detections":   detections,
            "status":       "defective" if any(d["is_defect"] for d in detections) else "ok"
        }

        if save:
            out_img  = self.results_dir / f"annotated_{image_path.name}"
            out_json = self.results_dir / f"{image_path.stem}_result.json"
            cv2.imwrite(str(out_img), annotated)
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)

        log.info(f"{image_path.name} | {result['status'].upper()} | "
                 f"defects={result['defect_count']} | {elapsed_ms}ms")

        return result


    def detect_batch(self, input_dir: str) -> list:
        """Run detection on all images in a folder."""
        input_dir = Path(input_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        if not images:
            log.warning(f"No images in {input_dir}")
            return []

        log.info(f"Batch detection: {len(images)} images")
        all_results = [self.detect_image(str(img)) for img in images]

        # Save combined results
        summary_path = self.results_dir / "batch_results.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)

        total    = len(all_results)
        defective = sum(1 for r in all_results if r.get("status") == "defective")
        log.info(f"Batch done | Total: {total} | Defective: {defective} | Good: {total - defective}")

        return all_results


    def detect_stream(self, source=0, display: bool = True, results_queue=None):
        """
        Real-time detection from webcam (source=0) or RTSP stream URL.

        Args:
            source       : 0 for webcam, or RTSP/HTTP URL string
            display      : show OpenCV window if True
            results_queue: multiprocessing.Queue to push results to dashboard
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            log.error(f"Cannot open video source: {source}")
            return

        log.info(f"Stream started — source: {source}")
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Frame grab failed, retrying...")
                    time.sleep(0.1)
                    continue

                frame_count += 1

                # Process every frame (reduce to every 2nd frame for slow hardware)
                start        = time.time()
                preprocessed = self._preprocess(frame)
                detections   = self._run_inference(preprocessed)
                elapsed_ms   = round((time.time() - start) * 1000, 1)

                annotated = self._annotate(preprocessed, detections)

                result = {
                    "timestamp":        datetime.now().isoformat(),
                    "frame":            frame_count,
                    "inference_ms":     elapsed_ms,
                    "defect_count":     sum(1 for d in detections if d["is_defect"]),
                    "detections":       detections,
                    "status":           "defective" if any(d["is_defect"] for d in detections) else "ok"
                }

                # Push to dashboard queue if provided
                if results_queue is not None:
                    try:
                        results_queue.put_nowait(result)
                    except Exception:
                        pass  # Queue full — skip frame result

                if display:
                    cv2.imshow("VisionFix — Real-Time Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        log.info("Stream stopped by user.")
                        break

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VisionFix Defect Detector")
    parser.add_argument("--mode",   choices=["image", "batch", "stream"], default="image")
    parser.add_argument("--source", default="data/processed",
                        help="Image path, folder path, or camera index / RTSP URL")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    detector = DefectDetector(config_path=args.config)

    if args.mode == "image":
        result = detector.detect_image(args.source)
        print(json.dumps(result, indent=2))

    elif args.mode == "batch":
        results = detector.detect_batch(args.source)
        defective = sum(1 for r in results if r.get("status") == "defective")
        print(f"\nSummary: {len(results)} images | {defective} defective")

    elif args.mode == "stream":
        source = int(args.source) if args.source.isdigit() else args.source
        detector.detect_stream(source=source, display=True)
