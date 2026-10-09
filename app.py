"""
Industrial Multi-Agent Vision & Diagnostic RAG Platform — Streamlit Cloud Dashboard
=====================================================================================
Autonomous industrial inspection platform integrating:
  • Task 1: Virtual Camera Sensor & Synthetic Surface Defect Generator
  • Task 2: OpenCV LAB-CLAHE Preprocessing, Contour Segmentation & 21-D Feature Extraction
  • Task 3: Scikit-Learn Random Forest Defect Classifier & Severity Scoring
  • Task 4: Self-RAG Vector Retrieval & OSHA CMMS Maintenance Work Order Synthesis
"""
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress sklearn unpickle version warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import IndustrialVisionPipeline
from src.task1_synthetic.generator import SyntheticIndustrialGenerator, DefectType

# ──────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Industrial Vision & Diagnostic RAG",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS for Industrial Control Room Aesthetic
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.2rem !important;
    }
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 0.9rem 1.4rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: white;
        border: 1px solid #334155;
    }
    .header-banner h1 { margin: 0; font-size: 1.4rem; }
    .header-banner p { margin: 0.25rem 0 0; opacity: 0.8; font-size: 0.85rem; }
    
    .status-badge-pass     { background: #065f46; color: #6ee7b7; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem; }
    .status-badge-low      { background: #854d0e; color: #fef08a; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem; }
    .status-badge-medium   { background: #9a3412; color: #fdba74; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem; }
    .status-badge-critical { background: #991b1b; color: #fca5a5; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem; }
    
    .stMetric {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Pipeline Cache Loader
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Initializing Industrial Vision & RAG Pipeline...")
def get_pipeline():
    return IndustrialVisionPipeline()

@st.cache_resource
def get_generator():
    return SyntheticIndustrialGenerator(img_size=(640, 640))

pipeline = get_pipeline()
generator = get_generator()

# ──────────────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────────────
if "frame_index" not in st.session_state:
    st.session_state["frame_index"] = 101
if "human_signed_off" not in st.session_state:
    st.session_state["human_signed_off"] = False

# ──────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div style="font-weight:700; font-size:14px; color:#e2e8f0; margin-bottom:4px;">🏭 Inspection Controls</div>', unsafe_allow_html=True)

defect_map = {
    "crack": ("⚡ Branching Crack", DefectType.CRACK),
    "scratch": ("🔍 Surface Scratch", DefectType.SCRATCH),
    "corrosion": ("🧪 Oxidation Corrosion", DefectType.CORROSION),
    "dimensional": ("📐 Dimensional Flaw", DefectType.DIMENSIONAL),
    "normal": ("✅ Normal (Clean Component)", DefectType.NORMAL),
    "upload": ("📤 Custom Photo Upload", None),
}

selected_defect_key = st.sidebar.selectbox(
    "Target Defect / Ingestion Mode",
    list(defect_map.keys()),
    index=0,
    format_func=lambda k: defect_map[k][0],
    key="defect_select",
)

uploaded_file = None
if selected_defect_key == "upload":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Component Image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key="custom_upload",
    )

severity = st.sidebar.slider(
    "Defect Severity Level",
    min_value=1.0,
    max_value=10.0,
    value=8.0,
    step=0.5,
    key="severity_slider",
    disabled=(selected_defect_key in ("normal", "upload")),
)

col_f1, col_f2 = st.sidebar.columns([1.2, 0.8])
with col_f1:
    frame_idx = st.number_input(
        "Frame Index",
        min_value=1,
        value=st.session_state["frame_index"],
        step=1,
        key="frame_input",
    )
    st.session_state["frame_index"] = frame_idx
with col_f2:
    if st.button("Next Frame ⏭️", use_container_width=True):
        st.session_state["frame_index"] += 1
        st.session_state["human_signed_off"] = False
        st.rerun()

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Inspect Frame & Run Pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Architecture:**
- **Task 1**: Virtual Camera Ingestion
- **Task 2**: OpenCV 21-D Optics
- **Task 3**: Random Forest ML Classifier
- **Task 4**: Diagnostic RAG & OSHA CMMS
""")

# ──────────────────────────────────────────────────────────────────────
# Main Header
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🏭 Industrial Multi-Agent Vision & Diagnostic RAG Platform</h1>
    <p>Autonomous Optical Defect Inspection • Real-Time Feature Telemetry • Self-RAG Maintenance Tickets</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Image Acquisition
# ──────────────────────────────────────────────────────────────────────
frame = None
if selected_defect_key == "upload":
    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        frame = cv2.resize(np.array(pil_img), (640, 640))
    else:
        st.info("👈 Please upload an image in the sidebar to run inspection.")
        st.stop()
else:
    target_defect_enum = defect_map[selected_defect_key][1]
    generator.set_seed(int(frame_idx))
    frame, true_meta, _ = generator.generate_sample(
        defect_type=target_defect_enum,
        severity=float(severity),
    )

# ──────────────────────────────────────────────────────────────────────
# Run Pipeline Execution
# ──────────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
results = pipeline.process_frame(frame, int(frame_idx))
latency_ms = (time.perf_counter() - t0) * 1000

alert = results["alert"]
work_order = results["work_order"]
mask = results["mask"]
features = results["features"]

pred_defect = alert.get("defect_type", "unknown").upper()
sev_score = alert.get("severity_score", 0.0)
sev_level = alert.get("severity_level", "PASS")

# Map severity to color and CSS class
level_classes = {
    "PASS": ("status-badge-pass", "#10b981"),
    "LOW": ("status-badge-low", "#eab308"),
    "MEDIUM": ("status-badge-medium", "#f97316"),
    "CRITICAL": ("status-badge-critical", "#ef4444"),
}
badge_class, border_color = level_classes.get(sev_level, ("status-badge-pass", "#10b981"))

# ──────────────────────────────────────────────────────────────────────
# Top Telemetry KPI Bar
# ──────────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("🔍 Predicted Defect", pred_defect, f"Frame #{frame_idx}")
with kpi2:
    st.metric("⚡ Dynamic Severity Score", f"{sev_score:.1f} / 10.0", sev_level)
with kpi3:
    wo_label = f"ID: {work_order.get('id')}" if work_order else "PASS / NO TICKET"
    st.metric("🛠️ OSHA CMMS Status", sev_level, wo_label)
with kpi4:
    st.metric("⏱️ Inference Latency", f"{latency_ms:.0f} ms", "Random Forest + RAG")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────
# Dual Camera Viewport (Raw Feed & Highlighted Defect Mask)
# ──────────────────────────────────────────────────────────────────────
st.markdown(f"### 📷 Dual Camera Viewport — Frame #{frame_idx} [{pred_defect}]")
col_cam1, col_cam2 = st.columns(2, gap="medium")

with col_cam1:
    st.markdown("**Optical Sensor Camera Feed (Raw Ingestion)**")
    # Draw border colored by severity
    bordered_frame = frame.copy()
    b_color = (0, 255, 0) if sev_level == "PASS" else ((0, 255, 255) if sev_level == "LOW" else ((0, 165, 255) if sev_level == "MEDIUM" else (0, 0, 255)))
    cv2.rectangle(bordered_frame, (0, 0), (bordered_frame.shape[1]-1, bordered_frame.shape[0]-1), b_color, 6)
    st.image(bordered_frame, channels="BGR", use_container_width=True)

with col_cam2:
    st.markdown("**Defect Segmentation Heatmap & Boundary Highlight**")
    # Blend red overlay on defects
    highlight_frame = frame.copy()
    if mask is not None and np.any(mask > 0):
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = (0, 0, 255) # Red highlight for defect
        highlight_frame = cv2.addWeighted(frame, 0.65, colored_mask, 0.35, 0)
        # Find contours and draw clean green boundary
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlight_frame, contours, -1, (0, 255, 128), 2)
    st.image(highlight_frame, channels="BGR", use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────
# Operational Intelligence Tabs
# ──────────────────────────────────────────────────────────────────────
tab_wo, tab_features, tab_arch = st.tabs([
    "🛠️ OSHA CMMS Maintenance Ticket",
    "🔬 21-D Optical Feature Telemetry",
    "🧠 Multi-Task Pipeline Architecture",
])

# ── Tab 1: OSHA CMMS Work Order (Task 4 RAG) ──
with tab_wo:
    if work_order:
        wo_id = work_order.get("id", f"WO-{frame_idx}")
        priority = work_order.get("priority", sev_level)
        status = "APPROVED ✅" if st.session_state["human_signed_off"] else work_order.get("status", "PENDING_APPROVAL")

        st.markdown(f"""
        <div style="background:#0f172a; border:1px solid #334155; border-radius:10px; padding:14px 18px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:18px; font-weight:700; color:#f8fafc;">Work Order: {wo_id}</span>
                <span style="margin-left:12px;" class="{badge_class}">PRIORITY: {priority}</span>
            </div>
            <div>
                <span style="color:#94a3b8; font-size:13px; font-weight:600;">Status: <strong>{status}</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        wcol1, wcol2 = st.columns([1, 1.8], gap="medium")
        with wcol1:
            st.markdown("#### 🛡️ Mandatory Safety Directive")
            st.error(f"**Directives:** {work_order.get('safety_directive', 'OSHA LOTO Required')}")

            sources = work_order.get("sources", [])
            if sources:
                st.markdown("#### 📚 Cited SOP Technical Manuals")
                for s in sources:
                    st.caption(f"• `{os.path.basename(s)}`")

            # Human-in-the-Loop Sign-off
            st.markdown("#### ✍️ Engineering Gate Sign-Off")
            if not st.session_state["human_signed_off"]:
                if st.button("Authorize & Release Ticket", type="primary", key="signoff_btn"):
                    st.session_state["human_signed_off"] = True
                    st.success("Ticket Authorized and Dispatched to CMMS.")
                    st.rerun()
            else:
                st.success("✅ Signed off by Certified Maintenance Engineer.")
                if st.button("Reset Sign-Off", key="reset_signoff"):
                    st.session_state["human_signed_off"] = False
                    st.rerun()

        with wcol2:
            st.markdown("#### 📋 Step-by-Step Remediation Protocol")
            steps = work_order.get("repair_steps", [])
            if steps:
                for idx, step in enumerate(steps, 1):
                    # Clean step markdown formatting
                    clean_step = step.strip().replace("#", "").strip()
                    st.markdown(f"**Step {idx}:** {clean_step}")
            else:
                st.info("Follow general standard maintenance operating procedures.")

    else:
        st.success(
            f"✨ **Component Passed Inspection (Status: {sev_level})**\n\n"
            f"Zero actionable structural defects detected in Frame #{frame_idx}. "
            "No maintenance work order synthesis required."
        )

# ── Tab 2: 21-D Optical Features (Task 2 OpenCV) ──
with tab_features:
    st.subheader("🔬 21-D Optical & Defect Feature Telemetry")
    if features:
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            st.metric("Total Defect Area", f"{features.get('total_area', 0.0):,.0f} px²")
            st.metric("Contour Count", f"{features.get('contour_count', 0)}")
        with fcol2:
            ratio = features.get("total_area", 0) / (640 * 640) * 100
            st.metric("Defect Area Ratio", f"{ratio:.2f} %")
            st.metric("Mean Gray Intensity", f"{features.get('mean_intensity', 0.0):.1f}")
        with fcol3:
            st.metric("Aspect Ratio (W/H)", f"{features.get('aspect_ratio', 0.0):.2f}")
            st.metric("Perimeter Length", f"{features.get('total_perimeter', 0.0):,.1f} px")

        st.markdown("---")
        with st.expander("📊 Complete Feature Vector (JSON)", expanded=True):
            st.json(features)
    else:
        st.info("No optical features extracted.")

# ── Tab 3: Architecture & Workflow Graph ──
with tab_arch:
    st.subheader("🧠 4-Task Industrial Vision RAG Architecture")

    st.markdown("""
```mermaid
graph LR
    T1["📷 Task 1<br/>Virtual Camera / Ingestion"] --> T2["🔬 Task 2<br/>OpenCV LAB-CLAHE & 21-D Features"]
    T2 --> T3["🎯 Task 3<br/>Random Forest ML Classifier"]
    T3 -->|Severity > 4.0| T4["🛠️ Task 4<br/>LangGraph RAG Agent"]
    T3 -->|Pass / Clean| PASS["✅ Component Released"]
    T4 --> SOP["📚 SOP Vector Database"]
    SOP --> T4
    T4 --> GATE["✍️ Human Engineering Signoff"]
    GATE --> WO["📋 OSHA CMMS Ticket Dispatched"]
```
    """)

    st.markdown("""
    ### Pipeline Specifications:
    - **Task 1 (Virtual Sensor)**: Simulates physical assembly line camera capture with configurable illumination, scratch textures, cracking mechanics, and corrosion pitting.
    - **Task 2 (OpenCV Vision)**: LAB-color space CLAHE dynamic range compression, adaptive thresholding contour segmentation, and extraction of 21 geometric, photometric, and texture descriptors.
    - **Task 3 (Scikit-Learn ML)**: 100-tree Random Forest classifier predicting 5 classes (`normal`, `scratch`, `crack`, `corrosion`, `dimensional`) with dynamic severity scaling ($0.0 - 10.0$).
    - **Task 4 (Self-RAG Agent)**: LangGraph state machine with automated query formulation, vector database similarity search over technical SOP manuals, self-RAG relevance grading, and CMMS ticket synthesis.
    """)

# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; opacity:0.5; font-size:0.8rem;">'
    '🏭 Industrial Multi-Agent Vision & Diagnostic RAG Platform • '
    'Powered by OpenCV • Scikit-Learn • LangGraph • Streamlit Cloud'
    '</div>',
    unsafe_allow_html=True,
)
