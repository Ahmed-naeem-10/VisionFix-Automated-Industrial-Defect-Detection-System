---
title: VisionFix Pipeline
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.39.0"
app_file: src/dashboard.py
pinned: false
license: mit
---

# 🔍 VisionFix Pipeline — Automated Industrial Defect Detection

End-to-end factory defect detection system:
**Raw blurry images → Clean preprocessing → YOLOv8 detection → Real-time dashboard**

---

## 📁 Project Structure

```
factory-defect-detection/
├── data/
│   ├── raw/              ← Put your 5,000 raw images here
│   ├── processed/        ← Auto-generated cleaned images
│   ├── annotations/      ← YOLO format labels (.txt files)
│   ├── augmented/        ← Auto-generated augmented dataset
│   └── yolo_dataset/     ← Auto-generated train/val split
├── src/
│   ├── preprocessing.py  ← Image cleaning pipeline
│   ├── train.py          ← YOLOv8 model training
│   ├── detect.py         ← Inference engine (image/batch/stream)
│   ├── dashboard.py      ← Streamlit real-time dashboard
│   ├── data_augmentation.py  ← Dataset augmentation
│   └── utils.py          ← Shared utilities
├── models/               ← Saved model weights
├── results/              ← Detection outputs + logs
├── logs/                 ← Pipeline logs
├── config.yaml           ← All settings
├── run_pipeline.py       ← Master runner
└── requirements.txt
```

---

## ⚙️ Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your images
#    → Put raw factory images in:  data/raw/
#    → Put YOLO labels in:         data/annotations/

# 3. Edit config.yaml if needed (classes, model size, etc.)
```

---

## 🚀 Usage

### Run Full Pipeline (all steps)
```bash
python run_pipeline.py --steps all
```

### Run Individual Steps
```bash
# Preprocess only
python run_pipeline.py --steps preprocess

# Preprocess + Augment + Train
python run_pipeline.py --steps preprocess augment train

# Detect on processed images
python run_pipeline.py --steps detect

# Launch dashboard only
python run_pipeline.py --steps dashboard
```

### Direct Script Usage
```bash
# Preprocess 5000 images
python src/preprocessing.py

# Train model
python src/train.py

# Detect — single image
python src/detect.py --mode image --source data/processed/part_001.jpg

# Detect — batch
python src/detect.py --mode batch --source data/processed/

# Detect — live webcam
python src/detect.py --mode stream --source 0

# Detect — RTSP camera
python src/detect.py --mode stream --source rtsp://192.168.1.100/stream

# Launch dashboard
streamlit run src/dashboard.py
```

---

## � Live Demo & Deployment

The Streamlit dashboard can be hosted for free as a public live demo.

### Recommended — Hugging Face Spaces (free, ML-friendly, 16 GB RAM)

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Space** → choose **SDK: Streamlit**, **Hardware: CPU basic**, **Visibility: Public**.
3. Push this repo to the Space (or clone the Space and copy files):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/visionfix-pipeline
   git push space main
   ```
4. In the Space **Files** tab, edit `README.md` and add this YAML front-matter at the top:
   ```yaml
   ---
   title: VisionFix Pipeline
   emoji: 🔍
   colorFrom: indigo
   colorTo: blue
   sdk: streamlit
   sdk_version: "1.39.0"
   app_file: src/dashboard.py
   pinned: false
   license: mit
   ---
   ```
5. The Space installs `requirements.txt` and starts `streamlit run src/dashboard.py`.
   Your public demo URL will be: `https://<your-username>-visionfix-pipeline.hf.space`

### Alternative — Streamlit Community Cloud (deploys straight from GitHub)

1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Sign in with GitHub → **New app** → select this repo → set **Main file path** to `src/dashboard.py`.
3. Click **Deploy**. Public URL: `https://visionfix-pipeline.streamlit.app` (name is customizable).

> **Live demo:** _(paste your deployed URL here once live)_

**Notes for hosting:**
- The trained weights live at `models/best.pt` (committed) so the dashboard detects real defects out of the box.
- Inference runs on CPU (free tiers have no GPU). Expect ~0.7 s per image.
- `results/batch_results.json` ships pre-populated so charts render immediately; upload your own image for on-demand detection.

---

## �🏷️ Annotation Format (YOLO)

Each image needs a `.txt` label file with the same base name:

```
# Format: class_id cx cy width height (all normalized 0-1)
0 0.512 0.430 0.120 0.085    ← good_part
1 0.231 0.612 0.045 0.032    ← scratch
2 0.788 0.310 0.067 0.054    ← dent
```

**Class IDs (from config.yaml):**
| ID | Class |
|----|-------|
| 0  | good_part |
| 1  | scratch |
| 2  | dent |
| 3  | crack |
| 4  | discoloration |
| 5  | missing_component |

---

## 📊 Dashboard Features

- Live KPI metrics (defect rate, total inspected, avg inference time)
- Defect rate trend chart (rolling 10-image window)
- Class distribution bar chart
- Good vs Defective donut chart
- Inference speed histogram
- Recent annotated image gallery
- Detection log table with CSV download
- Auto-refresh every 2 seconds
- Upload image for on-demand detection
- Alert banner when defect rate exceeds threshold

---

## 🧠 Preprocessing Steps

1. **Quality Assessment** — blur score + noise level measurement
2. **Noise Removal** — Non-Local Means + Median filter (adaptive strength)
3. **Lighting Correction** — CLAHE in LAB color space
4. **Sharpening** — Unsharp masking (adaptive, only if needed)
5. **Normalization** — [0-255] for saving, [0-1] float32 for model

---

## 📈 Training Tips

- Start with `yolov8n` (nano) for speed; upgrade to `yolov8s/m` for accuracy
- Minimum ~100 labeled images per class recommended
- Use `data_augmentation.py` to expand small datasets (3× default)
- Monitor training in `models/defect_detector/` (auto-generated plots)

---

## 🔧 Config Reference (`config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `model.architecture` | yolov8n | Model size |
| `model.confidence` | 0.45 | Detection confidence threshold |
| `training.epochs` | 100 | Max training epochs |
| `training.batch_size` | 16 | Batch size |
| `preprocessing.blur_threshold` | 150 | Below = apply sharpening |
| `dashboard.refresh_interval` | 15 | Dashboard refresh (seconds) |
