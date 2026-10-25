"""
VisionFix — Real-Time Dashboard (Emerald & Mint Edition)
=========================================================
Professional dark-theme dashboard with Emerald & Mint color palette.
Run with:  streamlit run src/dashboard.py
"""

import streamlit as st
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

# ── Page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="VisionFix Pipeline",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

CLASSES      = CFG["classes"]
RESULTS_DIR  = Path(CFG["paths"]["results"])
REFRESH_SEC  = CFG["dashboard"]["refresh_interval"]
MAX_HISTORY  = CFG["dashboard"]["max_history"]

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "emerald":      "#10b981",   # primary
    "emerald_dark": "#059669",
    "emerald_dim":  "#10b98120",
    "mint":         "#6ee7b7",
    "mint_dim":     "#6ee7b715",
    "teal":         "#14b8a6",
    "danger":       "#f43f5e",
    "danger_dim":   "#f43f5e18",
    "warning":      "#f59e0b",
    "warning_dim":  "#f59e0b18",
    "bg":           "#0a0f0d",   # near-black green-tinted
    "surface":      "#111917",
    "card":         "#162420",
    "border":       "#1e3330",
    "text":         "#e2e8f0",
    "muted":        "#94a3b8",
}

# ── Class colour map (charts) ──────────────────────────────────────────────────
CLASS_COLORS = {
    "patches":          C["emerald"],
    "scratches":        C["mint"],
    "rolled-in_scale":  C["teal"],
    "inclusion":        "#818cf8",
    "pitted_surface":   C["warning"],
    "crazing":          "#fb7185",
}

# ══════════════════════════════════════════════════════════════════════════════
# CSS — full professional theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* ── Global reset ── */
  html, body, [class*="css"] {{
      font-family: 'Inter', sans-serif !important;
      background-color: {C["bg"]} !important;
      color: {C["text"]} !important;
  }}

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 1.5rem 2rem 2rem 2rem !important; max-width: 1400px; }}

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
      background: {C["surface"]} !important;
      border-right: 1px solid {C["border"]} !important;
  }}
  section[data-testid="stSidebar"] * {{ color: {C["text"]} !important; }}
  section[data-testid="stSidebar"] .stButton > button {{
      background: linear-gradient(135deg, {C["emerald_dark"]}, {C["teal"]}) !important;
      color: #fff !important;
      border: none !important;
      border-radius: 8px !important;
      font-weight: 600 !important;
      width: 100%;
      padding: 0.5rem 1rem;
      transition: opacity .2s;
  }}
  section[data-testid="stSidebar"] .stButton > button:hover {{ opacity: .85; }}

  /* ── Top header bar ── */
  .vf-header {{
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.5rem 0 1.2rem 0;
      border-bottom: 1px solid {C["border"]};
      margin-bottom: 1.2rem;
  }}
  .vf-logo {{
      font-size: 2rem;
      line-height: 1;
  }}
  .vf-title {{
      font-size: 1.65rem;
      font-weight: 700;
      background: linear-gradient(90deg, {C["emerald"]}, {C["mint"]});
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      letter-spacing: -0.5px;
  }}
  .vf-subtitle {{
      font-size: 0.78rem;
      color: {C["muted"]};
      margin-top: 0.15rem;
  }}
  .vf-badge {{
      margin-left: auto;
      background: {C["emerald_dim"]};
      border: 1px solid {C["emerald"]};
      color: {C["emerald"]};
      font-size: 0.72rem;
      font-weight: 600;
      padding: 0.25rem 0.7rem;
      border-radius: 20px;
      letter-spacing: 0.5px;
  }}

  /* ── KPI cards ── */
  .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 0.75rem;
      margin-bottom: 1.25rem;
  }}
  .kpi-card {{
      background: {C["card"]};
      border: 1px solid {C["border"]};
      border-radius: 12px;
      padding: 1rem 1.1rem;
      position: relative;
      overflow: hidden;
      transition: border-color .2s;
  }}
  .kpi-card:hover {{ border-color: {C["emerald"]}55; }}
  .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg, {C["emerald"]}, {C["mint"]});
      border-radius: 12px 12px 0 0;
  }}
  .kpi-label {{
      font-size: 0.72rem;
      font-weight: 500;
      color: {C["muted"]};
      text-transform: uppercase;
      letter-spacing: 0.6px;
      margin-bottom: 0.4rem;
  }}
  .kpi-value {{
      font-size: 1.7rem;
      font-weight: 700;
      color: {C["text"]};
      line-height: 1.1;
  }}
  .kpi-value.danger {{ color: {C["danger"]}; }}
  .kpi-value.good   {{ color: {C["emerald"]}; }}
  .kpi-delta {{
      font-size: 0.72rem;
      margin-top: 0.25rem;
      color: {C["muted"]};
  }}
  .kpi-delta.high {{ color: {C["danger"]}; font-weight: 600; }}

  /* ── Alert / status banners ── */
  .banner {{
      border-radius: 10px;
      padding: 0.85rem 1.1rem;
      margin-bottom: 1.1rem;
      font-size: 0.9rem;
      font-weight: 500;
      display: flex;
      align-items: center;
      gap: 0.6rem;
  }}
  .banner.alert {{
      background: {C["danger_dim"]};
      border: 1px solid {C["danger"]}55;
      color: {C["danger"]};
  }}
  .banner.ok {{
      background: {C["emerald_dim"]};
      border: 1px solid {C["emerald"]}55;
      color: {C["emerald"]};
  }}
  .banner.warn {{
      background: {C["warning_dim"]};
      border: 1px solid {C["warning"]}55;
      color: {C["warning"]};
  }}

  /* ── Section headers ── */
  .section-header {{
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: {C["muted"]};
      margin: 1.4rem 0 0.7rem 0;
      display: flex;
      align-items: center;
      gap: 0.5rem;
  }}
  .section-header::after {{
      content: '';
      flex: 1;
      height: 1px;
      background: {C["border"]};
  }}

  /* ── Chart cards ── */
  .chart-card {{
      background: {C["card"]};
      border: 1px solid {C["border"]};
      border-radius: 12px;
      padding: 1rem;
  }}

  /* ── Image gallery ── */
  .gallery-img-wrap {{
      background: {C["card"]};
      border: 1px solid {C["border"]};
      border-radius: 10px;
      overflow: hidden;
      transition: border-color .2s;
  }}
  .gallery-img-wrap:hover {{ border-color: {C["emerald"]}77; }}
  .gallery-caption {{
      padding: 0.4rem 0.7rem;
      font-size: 0.72rem;
      color: {C["muted"]};
      background: {C["surface"]};
  }}
  .gallery-caption.defective {{ color: {C["danger"]}; }}
  .gallery-caption.ok        {{ color: {C["emerald"]}; }}

  /* ── Table ── */
  .stDataFrame {{ border-radius: 10px; overflow: hidden; }}
  .stDataFrame thead tr th {{
      background: {C["surface"]} !important;
      color: {C["muted"]} !important;
      font-size: 0.75rem !important;
      font-weight: 600 !important;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 1px solid {C["border"]} !important;
  }}
  .stDataFrame tbody tr td {{
      background: {C["card"]} !important;
      color: {C["text"]} !important;
      font-size: 0.82rem !important;
      border-bottom: 1px solid {C["border"]} !important;
  }}
  .stDataFrame tbody tr:hover td {{ background: {C["surface"]} !important; }}

  /* ── Download button ── */
  .stDownloadButton > button {{
      background: transparent !important;
      border: 1px solid {C["emerald"]}88 !important;
      color: {C["emerald"]} !important;
      border-radius: 8px !important;
      font-size: 0.82rem !important;
      font-weight: 500 !important;
      padding: 0.4rem 1rem !important;
      transition: all .2s;
  }}
  .stDownloadButton > button:hover {{
      background: {C["emerald_dim"]} !important;
      border-color: {C["emerald"]} !important;
  }}

  /* ── Slider ── */
  .stSlider [data-baseweb="slider"] div[role="slider"] {{
      background: {C["emerald"]} !important;
  }}

  /* ── Divider ── */
  hr {{ border-color: {C["border"]} !important; }}

  /* ── Plotly chart bg ── */
  .js-plotly-plot .plotly .bg {{ fill: transparent !important; }}
</style>
""", unsafe_allow_html=True)


# ── Plotly base layout shared by all charts ────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=C["muted"], size=11),
    margin=dict(t=36, b=24, l=8, r=8),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
    title_font=dict(size=13, color=C["text"], family="Inter"),
)

# Reusable axis style (applied per-chart to avoid duplicate-key conflict)
AXIS_STYLE = dict(gridcolor=C["border"], linecolor=C["border"], zerolinecolor=C["border"])


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTS AUTO-DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS_PATH = Path("models/best.pt")
WEIGHTS_URL  = (
    "https://github.com/Ahmed-naeem-10/"
    "VisionFix-Automated-Industrial-Defect-Detection-System/"
    "releases/download/v1.0/best.pt"
)

def _ensure_weights():
    if WEIGHTS_PATH.exists():
        return
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    with st.spinner("⬇ Downloading model weights (~24 MB) — first run only…"):
        try:
            urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        except Exception as e:
            st.error(f"Could not download weights: {e}\n\nFalling back to base YOLOv8n.")

_ensure_weights()


# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource
def get_detector():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from detect import DefectDetector
    return DefectDetector(CONFIG_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=REFRESH_SEC)
def load_batch_results() -> pd.DataFrame:
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
    return sorted(
        RESULTS_DIR.glob("annotated_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:n]


def compute_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total": 0, "defective": 0, "good": 0,
                "defect_rate": 0.0, "avg_inference_ms": 0.0, "top_defect": "N/A"}
    total     = len(df)
    defective = int((df["status"] == "defective").sum())
    good      = total - defective
    all_dets  = [d for row in df["detections"] for d in row if d.get("is_defect")]
    if all_dets:
        from collections import Counter
        top_defect = Counter(d["class_name"] for d in all_dets).most_common(1)[0][0]
    else:
        top_defect = "None"
    return {
        "total":            total,
        "defective":        defective,
        "good":             good,
        "defect_rate":      round(defective / total * 100, 1) if total else 0.0,
        "avg_inference_ms": round(df["inference_ms"].mean(), 1) if not df.empty else 0.0,
        "top_defect":       top_defect,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def chart_defect_rate_over_time(df: pd.DataFrame):
    if df.empty:
        st.info("No data yet.")
        return
    df_s = df.sort_values("timestamp").tail(MAX_HISTORY).copy()
    df_s["is_defective"]  = (df_s["status"] == "defective").astype(int)
    df_s["rolling_rate"]  = df_s["is_defective"].rolling(10, min_periods=1).mean() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_s["timestamp"], y=df_s["rolling_rate"],
        mode="lines",
        line=dict(color=C["emerald"], width=2.5),
        fill="tozeroy",
        fillcolor=C["emerald_dim"],
        name="Defect Rate %",
        hovertemplate="%{y:.1f}%<extra></extra>"
    ))
    fig.add_hline(
        y=20, line_dash="dot", line_color=C["warning"], line_width=1.5,
        annotation_text="Threshold 20%",
        annotation_font_color=C["warning"],
        annotation_font_size=10
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Defect Rate Over Time (Rolling 10)",
        height=290,
        showlegend=False,
        xaxis=AXIS_STYLE,
        yaxis=dict(**AXIS_STYLE, ticksuffix="%")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chart_class_distribution(df: pd.DataFrame):
    if df.empty:
        st.info("No data yet.")
        return
    all_dets = [d for row in df["detections"] for d in row]
    if not all_dets:
        st.info("No detections found.")
        return
    from collections import Counter
    counts   = Counter(d["class_name"] for d in all_dets)
    chart_df = pd.DataFrame(counts.items(), columns=["Class", "Count"]).sort_values("Count", ascending=True)
    colors   = [CLASS_COLORS.get(c, C["mint"]) for c in chart_df["Class"]]

    fig = go.Figure(go.Bar(
        x=chart_df["Count"],
        y=chart_df["Class"],
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}: %{x} detections<extra></extra>"
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Defect Class Distribution",
        height=290,
        xaxis=dict(**AXIS_STYLE, title="Detections"),
        yaxis=dict(**AXIS_STYLE, title="")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chart_inference_speed(df: pd.DataFrame):
    if df.empty:
        st.info("No data yet.")
        return
    fig = go.Figure(go.Histogram(
        x=df["inference_ms"],
        nbinsx=30,
        marker_color=C["teal"],
        opacity=0.85,
        hovertemplate="%{x:.0f} ms — %{y} images<extra></extra>"
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Inference Speed Distribution",
        height=265,
        xaxis=dict(**AXIS_STYLE, title="ms per image"),
        yaxis=dict(**AXIS_STYLE, title="Count")
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def chart_good_vs_defective(summary: dict):
    fig = go.Figure(go.Pie(
        labels=["Good Parts", "Defective"],
        values=[summary["good"], summary["defective"]],
        hole=0.62,
        marker=dict(
            colors=[C["emerald"], C["danger"]],
            line=dict(color=C["card"], width=2)
        ),
        textfont=dict(color=C["text"], size=12),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"
    ))
    fig.add_annotation(
        text=f"<b>{summary['defect_rate']}%</b><br><span style='font-size:10px'>defect rate</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=C["text"], family="Inter")
    )
    fig.update_layout(
        **PLOT_LAYOUT,
        title="Quality Overview",
        height=265,
        showlegend=True,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.05,
            bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])
        )
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.markdown(f"""
    <div style="padding:0.5rem 0 1rem 0; border-bottom:1px solid {C['border']}; margin-bottom:1rem;">
      <div style="font-size:1.1rem;font-weight:700;
                  background:linear-gradient(90deg,{C['emerald']},{C['mint']});
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        🔍 VisionFix
      </div>
      <div style="font-size:0.72rem;color:{C['muted']};margin-top:0.2rem;">
        Industrial Defect Detection
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown(f"<div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;"
                        f"letter-spacing:0.8px;color:{C['muted']};margin-bottom:0.5rem;'>"
                        f"📁 Upload & Detect</div>", unsafe_allow_html=True)

    uploaded = st.sidebar.file_uploader(
        "Drop an image here", type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded is not None:
        tmp_path = Path("data/temp_upload") / uploaded.name
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.getvalue())

        st.sidebar.image(str(tmp_path), use_column_width=True,
                         caption="Uploaded image")

        if st.sidebar.button("▶  Run Detection", use_container_width=True):
            with st.spinner("Running inference…"):
                try:
                    detector = get_detector()
                    result   = detector.detect_image(str(tmp_path), save=True)
                    status   = result["status"].upper()
                    color    = C["danger"] if status == "DEFECTIVE" else C["emerald"]
                    st.sidebar.markdown(
                        f"<div style='background:{color}20;border:1px solid {color}55;"
                        f"border-radius:8px;padding:0.7rem;margin-top:0.5rem;'>"
                        f"<b style='color:{color}'>{status}</b><br>"
                        f"<span style='color:{C['muted']};font-size:0.78rem;'>"
                        f"Defects: {result['defect_count']} &nbsp;·&nbsp; "
                        f"{result['inference_ms']} ms</span></div>",
                        unsafe_allow_html=True
                    )
                    ann = RESULTS_DIR / f"annotated_{uploaded.name}"
                    if ann.exists():
                        st.sidebar.image(str(ann), use_column_width=True,
                                         caption="Detection result")
                except Exception as e:
                    st.sidebar.error(f"Detection failed: {e}")

    st.sidebar.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;"
                        f"letter-spacing:0.8px;color:{C['muted']};margin-bottom:0.5rem;'>"
                        f"⚙️ Settings</div>", unsafe_allow_html=True)
    alert_threshold = st.sidebar.slider("Alert Threshold (%)", 5, 50, 20,
                                        label_visibility="collapsed")
    st.sidebar.caption(f"Alert fires when defect rate > **{alert_threshold}%**")

    st.sidebar.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<div style='font-size:0.7rem;color:{C['muted']};text-align:center;'>"
        f"VisionFix Pipeline v1.0<br>YOLOv8n · NEU-DET · 6 classes</div>",
        unsafe_allow_html=True
    )
    return alert_threshold


# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS (pure HTML — no st.metric)
# ══════════════════════════════════════════════════════════════════════════════

def render_kpis(summary: dict, alert_threshold: float):
    rate_class  = "danger" if summary["defect_rate"] > alert_threshold else "good"
    rate_delta  = (f'<div class="kpi-delta high">▲ ABOVE THRESHOLD</div>'
                   if summary["defect_rate"] > alert_threshold
                   else f'<div class="kpi-delta">Within threshold</div>')

    st.markdown(f"""
    <div class="kpi-grid">

      <div class="kpi-card">
        <div class="kpi-label">Total Inspected</div>
        <div class="kpi-value">{summary['total']:,}</div>
        <div class="kpi-delta">images processed</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Good Parts</div>
        <div class="kpi-value good">{summary['good']:,}</div>
        <div class="kpi-delta">{round(summary['good']/summary['total']*100,1) if summary['total'] else 0}% pass rate</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Defective</div>
        <div class="kpi-value danger">{summary['defective']:,}</div>
        <div class="kpi-delta">flagged for review</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Defect Rate</div>
        <div class="kpi-value {rate_class}">{summary['defect_rate']}%</div>
        {rate_delta}
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Avg Inference</div>
        <div class="kpi-value">{summary['avg_inference_ms']}</div>
        <div class="kpi-delta">ms per image</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label">Top Defect</div>
        <div class="kpi-value" style="font-size:1.15rem;margin-top:0.2rem;">{summary['top_defect']}</div>
        <div class="kpi-delta">most frequent</div>
      </div>

    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():

    alert_threshold = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="vf-header">
      <div class="vf-logo">🔍</div>
      <div>
        <div class="vf-title">VisionFix Pipeline — Defect Detection Dashboard</div>
        <div class="vf-subtitle">
          Auto-refreshes every {REFRESH_SEC}s &nbsp;·&nbsp;
          Last updated: {datetime.now().strftime('%H:%M:%S')} &nbsp;·&nbsp;
          NEU-DET Steel Surface Dataset
        </div>
      </div>
      <div class="vf-badge">🟢 LIVE</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df      = load_batch_results()
    summary = compute_summary(df)

    # ── Status banner ─────────────────────────────────────────────────────────
    if summary["defect_rate"] > alert_threshold:
        st.markdown(
            f'<div class="banner alert">⚠ ALERT — Defect rate is '
            f'<b>{summary["defect_rate"]}%</b>, exceeding the {alert_threshold}% threshold. '
            f'Immediate inspection recommended.</div>',
            unsafe_allow_html=True
        )
    elif summary["total"] > 0:
        st.markdown(
            f'<div class="banner ok">✓ System Nominal — Defect rate '
            f'<b>{summary["defect_rate"]}%</b> is within acceptable range.</div>',
            unsafe_allow_html=True
        )

    # ── KPI row ───────────────────────────────────────────────────────────────
    render_kpis(summary, alert_threshold)

    # ── Charts row 1 ──────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">📈 Analytics</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2], gap="medium")
    with col_l:
        chart_defect_rate_over_time(df)
    with col_r:
        chart_good_vs_defective(summary)

    # ── Charts row 2 ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        chart_class_distribution(df)
    with col_b:
        chart_inference_speed(df)

    # ── Gallery ───────────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">📸 Recent Detection Results</div>',
                unsafe_allow_html=True)
    recent = get_latest_annotated_images(n=6)
    if recent:
        cols = st.columns(3, gap="small")
        for idx, img_path in enumerate(recent):
            with cols[idx % 3]:
                stem   = img_path.name.replace("annotated_", "").rsplit(".", 1)[0]
                status = "unknown"
                if not df.empty and "file" in df.columns:
                    match = df[df["file"].str.startswith(stem)]
                    if not match.empty:
                        status = match.iloc[-1]["status"]
                icon   = "🔴" if status == "defective" else "🟢"
                cap_cls = "defective" if status == "defective" else "ok"
                st.markdown(f'<div class="gallery-img-wrap">', unsafe_allow_html=True)
                st.image(Image.open(img_path), use_column_width=True)
                st.markdown(
                    f'<div class="gallery-caption {cap_cls}">'
                    f'{icon} {img_path.stem.replace("annotated_","")}</div></div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("No annotated images yet.")

    # ── Detection log table ───────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">📋 Detection Log</div>', unsafe_allow_html=True)
    if not df.empty:
        display_df = df[["timestamp", "file", "status", "defect_count", "inference_ms"]].copy()
        display_df = display_df.sort_values("timestamp", ascending=False).head(50)
        display_df.columns = ["Timestamp", "File", "Status", "Defects", "Inference (ms)"]

        def color_status(val):
            if val == "defective": return f"color: {C['danger']}; font-weight:600"
            if val == "ok":        return f"color: {C['emerald']}; font-weight:600"
            return ""

        styled = display_df.style.map(color_status, subset=["Status"])
        st.dataframe(styled, use_container_width=True, height=320)

        csv = display_df.to_csv(index=False)
        st.download_button(
            "⬇  Export CSV",
            data=csv,
            file_name=f"visionfix_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No results found. Run batch detection first.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid {C['border']};
                display:flex;justify-content:space-between;align-items:center;
                font-size:0.72rem;color:{C['muted']};">
      <span>VisionFix Pipeline v1.0 &nbsp;·&nbsp; YOLOv8n &nbsp;·&nbsp; NEU-DET Dataset</span>
      <span>Built with Streamlit &amp; Ultralytics &nbsp;·&nbsp;
            <a href="https://github.com/Ahmed-naeem-10/VisionFix-Automated-Industrial-Defect-Detection-System"
               style="color:{C['emerald']};text-decoration:none;">GitHub ↗</a></span>
    </div>
    """, unsafe_allow_html=True)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    time.sleep(REFRESH_SEC)
    st.rerun()


if __name__ == "__main__":
    main()
