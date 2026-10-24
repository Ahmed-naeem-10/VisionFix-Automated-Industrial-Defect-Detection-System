"""
VisionFix — Real-Time Dashboard
=================================
Streamlit dashboard showing:
  - Live detection feed
  - Defect rate over time
  - Class distribution
  - Per-image results table
  - Alert system for high defect rates

Run with:  streamlit run src/dashboard.py
"""

import streamlit as st
import cv2
import yaml
import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
from PIL import Image
import threading
import queue

# ─── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="VisionFix Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Config ───────────────────────────────────────────────────────────────
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

CLASSES       = CFG["classes"]
RESULTS_DIR   = Path(CFG["paths"]["results"])
REFRESH_SEC   = CFG["dashboard"]["refresh_interval"]
MAX_HISTORY   = CFG["dashboard"]["max_history"]

# ── Defect classes (everything except good_part)
DEFECT_CLASSES = [c for c in CLASSES if c != "good_part"]


# ── Cached model loader (loads YOLO once, reused across reruns/uploads) ───────
@st.cache_resource
def get_detector():
    """Instantiate DefectDetector once and cache it for the session."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from detect import DefectDetector
    return DefectDetector(CONFIG_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# CSS Styling
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .alert-box {
        background: #ff4b4b22;
        border-left: 4px solid #ff4b4b;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .ok-box {
        background: #21c35422;
        border-left: 4px solid #21c354;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .stMetric label { font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=REFRESH_SEC)
def load_batch_results() -> pd.DataFrame:
    """Load batch_results.json into a DataFrame."""
    batch_file = RESULTS_DIR / "batch_results.json"
    if not batch_file.exists():
        return pd.DataFrame()

    with open(batch_file) as f:
        data = json.load(f)

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rows.append({
            "timestamp":    item.get("timestamp", ""),
            "file":         item.get("file", ""),
            "status":       item.get("status", "unknown"),
            "defect_count": item.get("defect_count", 0),
            "inference_ms": item.get("inference_ms", 0),
            "detections":   item.get("detections", [])
        })

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns and not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def get_latest_annotated_images(n: int = 6) -> list:
    """Return paths of the n most recently created annotated images."""
    images = sorted(
        RESULTS_DIR.glob("annotated_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return images[:n]


def compute_summary(df: pd.DataFrame) -> dict:
    """Compute top-level KPI metrics from results DataFrame."""
    if df.empty:
        return {
            "total": 0, "defective": 0, "good": 0,
            "defect_rate": 0.0, "avg_inference_ms": 0.0,
            "top_defect": "N/A"
        }

    total     = len(df)
    defective = int((df["status"] == "defective").sum())
    good      = total - defective

    # Flatten all detections to count defect types
    all_dets  = [d for row in df["detections"] for d in row if d.get("is_defect")]
    if all_dets:
        from collections import Counter
        counts    = Counter(d["class_name"] for d in all_dets)
        top_defect = counts.most_common(1)[0][0]
    else:
        top_defect = "None"

    return {
        "total":           total,
        "defective":       defective,
        "good":            good,
        "defect_rate":     round(defective / total * 100, 1) if total else 0.0,
        "avg_inference_ms": round(df["inference_ms"].mean(), 1) if not df.empty else 0.0,
        "top_defect":      top_defect
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def chart_defect_rate_over_time(df: pd.DataFrame):
    """Line chart of defect rate over time (rolling window)."""
    if df.empty or "timestamp" not in df.columns:
        st.info("No data yet.")
        return

    df_sorted = df.sort_values("timestamp").tail(MAX_HISTORY).copy()
    df_sorted["is_defective"] = (df_sorted["status"] == "defective").astype(int)
    df_sorted["rolling_rate"] = df_sorted["is_defective"].rolling(10, min_periods=1).mean() * 100

    fig = px.line(
        df_sorted,
        x="timestamp",
        y="rolling_rate",
        title="Defect Rate Over Time (Rolling 10)",
        labels={"rolling_rate": "Defect Rate (%)", "timestamp": "Time"},
        color_discrete_sequence=["#e74c3c"]
    )
    fig.add_hline(y=20, line_dash="dash", line_color="orange",
                  annotation_text="Alert Threshold (20%)")
    fig.update_layout(height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def chart_class_distribution(df: pd.DataFrame):
    """Bar chart of detected defect types."""
    if df.empty:
        st.info("No data yet.")
        return

    all_dets = [d for row in df["detections"] for d in row]
    if not all_dets:
        st.info("No detections found.")
        return

    from collections import Counter
    counts = Counter(d["class_name"] for d in all_dets)
    chart_df = pd.DataFrame(counts.items(), columns=["Class", "Count"]).sort_values("Count", ascending=False)

    colors = ["#e74c3c" if c != "good_part" else "#2ecc71" for c in chart_df["Class"]]
    fig = px.bar(
        chart_df, x="Class", y="Count",
        title="Detection Class Distribution",
        color="Class",
        color_discrete_map={
            "good_part": "#2ecc71",
            "scratch": "#e67e22",
            "dent": "#e74c3c",
            "crack": "#c0392b",
            "discoloration": "#9b59b6",
            "missing_component": "#2980b9"
        }
    )
    fig.update_layout(height=300, margin=dict(t=40, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def chart_inference_speed(df: pd.DataFrame):
    """Histogram of inference times."""
    if df.empty or "inference_ms" not in df.columns:
        st.info("No data yet.")
        return

    fig = px.histogram(
        df, x="inference_ms",
        title="Inference Speed Distribution (ms)",
        nbins=30,
        color_discrete_sequence=["#3498db"]
    )
    fig.update_layout(height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def chart_good_vs_defective(summary: dict):
    """Donut chart good vs defective."""
    fig = go.Figure(data=[go.Pie(
        labels=["Good Parts", "Defective"],
        values=[summary["good"], summary["defective"]],
        hole=0.55,
        marker_colors=["#2ecc71", "#e74c3c"]
    )])
    fig.update_layout(
        title="Good vs Defective",
        height=280,
        margin=dict(t=40, b=10, l=10, r=10),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Sidebar: upload image for on-demand detection."""
    st.sidebar.title("🔧 Controls")

    st.sidebar.subheader("📁 Upload & Detect")
    uploaded = st.sidebar.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded is not None:
        # Save temp file
        tmp_path = Path("data/temp_upload") / uploaded.name
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.read())

        if st.sidebar.button("▶ Run Detection"):
            with st.spinner("Running detection..."):
                try:
                    detector = get_detector()
                    result   = detector.detect_image(str(tmp_path), save=True)

                    st.sidebar.success(f"Status: **{result['status'].upper()}**")
                    st.sidebar.write(f"Defects found: {result['defect_count']}")
                    st.sidebar.write(f"Inference: {result['inference_ms']} ms")

                    # Show annotated image
                    ann_path = RESULTS_DIR / f"annotated_{uploaded.name}"
                    if ann_path.exists():
                        st.sidebar.image(str(ann_path), caption="Detection Result", use_column_width=True)

                except Exception as e:
                    st.sidebar.error(f"Detection failed: {e}")

    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Settings")
    alert_threshold = st.sidebar.slider("Alert Threshold (%)", 5, 50, 20)
    st.sidebar.caption(f"Alert fires when defect rate > {alert_threshold}%")

    return alert_threshold


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown('<div class="main-header">🔍 VisionFix Pipeline — Defect Detection Dashboard</div>',
                unsafe_allow_html=True)
    st.caption(f"Auto-refreshes every {REFRESH_SEC}s | Last updated: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()

    alert_threshold = render_sidebar()

    # Load data
    df      = load_batch_results()
    summary = compute_summary(df)

    # ── Alert Banner ──────────────────────────────────────────────────────────
    if summary["defect_rate"] > alert_threshold:
        st.markdown(
            f'<div class="alert-box">⚠️ <b>ALERT:</b> Defect rate is '
            f'<b>{summary["defect_rate"]}%</b> — exceeds threshold of {alert_threshold}%!</div>',
            unsafe_allow_html=True
        )
    elif summary["total"] > 0:
        st.markdown(
            f'<div class="ok-box">✅ <b>All Clear:</b> Defect rate '
            f'<b>{summary["defect_rate"]}%</b> is within acceptable range.</div>',
            unsafe_allow_html=True
        )

    # ── KPI Metrics Row ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Inspected",   summary["total"])
    c2.metric("✅ Good Parts",      summary["good"])
    c3.metric("❌ Defective",       summary["defective"])
    c4.metric("Defect Rate",       f"{summary['defect_rate']}%",
              delta=None if summary["defect_rate"] < alert_threshold else "⚠ HIGH",
              delta_color="inverse")
    c5.metric("Avg Inference",     f"{summary['avg_inference_ms']} ms")
    c6.metric("Top Defect",        summary["top_defect"])

    st.divider()

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])
    with col_left:
        chart_defect_rate_over_time(df)
    with col_right:
        chart_good_vs_defective(summary)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        chart_class_distribution(df)
    with col_b:
        chart_inference_speed(df)

    st.divider()

    # ── Recent Annotated Images ───────────────────────────────────────────────
    st.subheader("📸 Recent Detection Results")
    recent_images = get_latest_annotated_images(n=6)

    if recent_images:
        img_cols = st.columns(3)
        for idx, img_path in enumerate(recent_images):
            with img_cols[idx % 3]:
                img = Image.open(img_path)
                # Find matching result for status label
                stem   = img_path.name.replace("annotated_", "").rsplit(".", 1)[0]
                status = "unknown"
                if not df.empty and "file" in df.columns:
                    match = df[df["file"].str.startswith(stem)]
                    if not match.empty:
                        status = match.iloc[-1]["status"]

                caption_color = "🔴" if status == "defective" else "🟢"
                st.image(img, caption=f"{caption_color} {img_path.name}", use_column_width=True)
    else:
        st.info("No annotated images yet. Run batch detection first.")

    st.divider()

    # ── Results Table ─────────────────────────────────────────────────────────
    st.subheader("📋 Detection Log")

    if not df.empty:
        display_df = df[["timestamp", "file", "status", "defect_count", "inference_ms"]].copy()
        display_df = display_df.sort_values("timestamp", ascending=False).head(50)
        display_df.columns = ["Timestamp", "File", "Status", "Defects", "Inference (ms)"]

        # Color status column
        def color_status(val):
            if val == "defective":
                return "background-color: #ffe0e0"
            elif val == "ok":
                return "background-color: #e0ffe0"
            return ""

        styled = display_df.style.applymap(color_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, height=350)

        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="⬇ Download Results CSV",
            data=csv,
            file_name=f"defect_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No results found. Run `python src/detect.py --mode batch` first.")

    # ── Auto Refresh ──────────────────────────────────────────────────────────
    time.sleep(REFRESH_SEC)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
