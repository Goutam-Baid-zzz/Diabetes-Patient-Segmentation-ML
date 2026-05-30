"""
MedSegment — Patient Segmentation & Healthcare Intelligence
UCI Diabetes 130-US Hospitals Dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MedSegment | Patient Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Dark Clinical Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ════════════════════════════════════
   ROOT VARIABLES
════════════════════════════════════ */
:root {
    --bg-deep:        #040d1c;
    --bg-card:        #071428;
    --bg-card-hover:  #0c1e3a;
    --bg-surface:     #0a1628;
    --border-dim:     rgba(0, 212, 255, 0.10);
    --border-bright:  rgba(0, 212, 255, 0.28);
    --cyan:           #00d4ff;
    --cyan-dim:       rgba(0, 212, 255, 0.15);
    --cyan-glow:      0 0 20px rgba(0, 212, 255, 0.18);
    --amber:          #f59e0b;
    --amber-dim:      rgba(245, 158, 11, 0.12);
    --amber-glow:     0 0 20px rgba(245, 158, 11, 0.15);
    --red:            #ef4444;
    --red-dim:        rgba(239, 68, 68, 0.12);
    --red-glow:       0 0 20px rgba(239, 68, 68, 0.15);
    --green:          #10b981;
    --green-dim:      rgba(16, 185, 129, 0.12);
    --green-glow:     0 0 20px rgba(16, 185, 129, 0.15);
    --text-primary:   #e8f4ff;
    --text-secondary: #7a9cc4;
    --text-muted:     #3d5a7a;
    --font-display:   'Syne', sans-serif;
    --font-body:      'IBM Plex Sans', sans-serif;
    --font-mono:      'IBM Plex Mono', monospace;
}

/* ════════════════════════════════════
   BASE RESETS
════════════════════════════════════ */
html, body, [class*="css"], .stApp {
    font-family: var(--font-body) !important;
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
}

/* ════════════════════════════════════
   ECG BACKGROUND PATTERN
════════════════════════════════════ */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(0,212,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 90% 80%, rgba(0,100,200,0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

.main .block-container {
    position: relative;
    z-index: 1;
    padding-top: 28px !important;
    padding-bottom: 40px !important;
    max-width: 1300px;
}

/* ════════════════════════════════════
   SIDEBAR
════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #02080f !important;
    border-right: 1px solid var(--border-dim) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 24px;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
[data-testid="stSidebarNav"] { display: none; }

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border-dim) !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 9px 14px !important;
    text-align: left !important;
    transition: all 0.2s ease !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--cyan-dim) !important;
    border-color: var(--border-bright) !important;
    color: var(--cyan) !important;
    transform: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: var(--cyan-dim) !important;
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    font-weight: 500 !important;
}

/* ════════════════════════════════════
   TYPOGRAPHY
════════════════════════════════════ */
h1, h2, h3 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.3px;
}
h1 { font-size: 30px !important; font-weight: 700 !important; }
h2 { font-size: 20px !important; font-weight: 600 !important; }
h3 { font-size: 15px !important; font-weight: 600 !important; }

p, span, div, td, th, li {
    color: var(--text-primary);
}

/* ════════════════════════════════════
   METRIC CARDS
════════════════════════════════════ */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
    box-shadow: var(--cyan-glow) !important;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: var(--border-bright) !important;
}
[data-testid="stMetricLabel"] > div {
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stMetricValue"] > div {
    font-family: var(--font-display) !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    color: var(--cyan) !important;
}

/* ════════════════════════════════════
   MAIN CTA BUTTON
════════════════════════════════════ */
.main .stButton > button {
    background: linear-gradient(135deg, #004d6b 0%, #006d8f 100%) !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    border-radius: 10px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 11px 22px !important;
    letter-spacing: 0.3px !important;
    box-shadow: var(--cyan-glow) !important;
    transition: all 0.2s ease !important;
}
.main .stButton > button:hover {
    background: linear-gradient(135deg, #005f80 0%, #0089b5 100%) !important;
    box-shadow: 0 0 30px rgba(0,212,255,0.35) !important;
    transform: translateY(-2px) !important;
}
.main .stButton > button:active {
    transform: translateY(0) !important;
}

/* Download button */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-secondary) !important;
    border-radius: 10px !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    background: var(--cyan-dim) !important;
}

/* ════════════════════════════════════
   DATAFRAME / TABLE
════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-dim) !important;
    background: var(--bg-card) !important;
}
.dvn-scroller { background: var(--bg-card) !important; }

/* ════════════════════════════════════
   PLOTLY CHARTS
════════════════════════════════════ */
[data-testid="stPlotlyChart"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-dim) !important;
    background: var(--bg-card) !important;
}

/* ════════════════════════════════════
   DIVIDER
════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid var(--border-dim) !important;
    margin: 28px 0 !important;
}

/* ════════════════════════════════════
   CUSTOM COMPONENTS
════════════════════════════════════ */

/* Page header */
.page-eyebrow {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--cyan);
    margin: 0 0 6px 0;
}
.page-title {
    font-family: var(--font-display);
    font-size: 28px;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.page-subtitle {
    font-family: var(--font-body);
    font-size: 14px;
    color: var(--text-secondary);
    margin: 0 0 28px 0;
}

/* Section label */
.sec-label {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 28px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-dim);
}

/* Generic card */
.ms-card {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.ms-card:hover { border-color: var(--border-bright); }

/* Cluster cards with accent glow */
.ms-card-amber {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(245,158,11,0.05) 100%);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: var(--amber-glow);
}
.ms-card-green {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(16,185,129,0.05) 100%);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: var(--green-glow);
}
.ms-card-red {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(239,68,68,0.05) 100%);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: var(--red-glow);
}
.ms-card-cyan {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(0,212,255,0.04) 100%);
    border: 1px solid var(--border-bright);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: var(--cyan-glow);
}

/* Badge pill */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 999px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.badge-red    { background: rgba(239,68,68,0.15);   color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-amber  { background: rgba(245,158,11,0.15);  color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-green  { background: rgba(16,185,129,0.15);  color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-cyan   { background: rgba(0,212,255,0.12);   color: var(--cyan); border: 1px solid rgba(0,212,255,0.3); }

/* Cluster assignment banner */
.cluster-banner {
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
}
.cluster-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    filter: blur(60px);
    opacity: 0.15;
}

/* Why-table */
.why-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-family: var(--font-body);
}
.why-table th {
    text-align: left;
    padding: 10px 14px;
    background: rgba(0,212,255,0.04);
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border-dim);
}
.why-table td {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border-dim);
    color: var(--text-secondary);
    vertical-align: middle;
}
.why-table tr:last-child td { border-bottom: none; }
.why-table tr:hover td { background: rgba(0,212,255,0.03); }
.wt-metric { color: var(--text-primary); font-weight: 500; }
.wt-val    { color: var(--cyan); font-family: var(--font-mono); font-weight: 500; font-size: 14px; }
.wt-above  { color: #f87171; font-size: 12px; font-family: var(--font-mono); }
.wt-below  { color: #34d399; font-size: 12px; font-family: var(--font-mono); }
.wt-avg    { color: var(--text-muted); font-size: 12px; font-family: var(--font-mono); }
.wt-neutral{ color: var(--text-muted); font-size: 12px; font-family: var(--font-mono); }

/* Recommendation items */
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 11px 0;
    border-bottom: 1px solid var(--border-dim);
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.55;
}
.rec-item:last-child { border-bottom: none; }
.rec-num {
    min-width: 22px;
    height: 22px;
    border-radius: 6px;
    background: rgba(0,212,255,0.12);
    border: 1px solid rgba(0,212,255,0.25);
    color: var(--cyan);
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 2px;
    flex-shrink: 0;
}

/* Char item */
.char-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid var(--border-dim);
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
}
.char-item:last-child { border-bottom: none; }

/* Stat rows (vertical stack inside cluster cards) */
.stat-rows {
    margin-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.stat-row-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.stat-row-item:last-child { border-bottom: none; }
.stat-row-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0;
}
.stat-row-value {
    font-family: var(--font-display);
    font-size: 15px;
    font-weight: 700;
    margin: 0;
    white-space: nowrap;
}

/* Sidebar brand */
.sidebar-brand {
    font-family: var(--font-display) !important;
    font-size: 20px !important;
    font-weight: 800 !important;
    color: var(--cyan) !important;
    letter-spacing: -0.5px;
    margin: 0 0 2px 0;
}
.sidebar-mono {
    font-family: var(--font-mono) !important;
    font-size: 10px !important;
    color: var(--text-muted) !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Pulse dot */
.pulse-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--cyan);
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 6px var(--cyan);
    animation: pulse-anim 2s infinite;
}
@keyframes pulse-anim {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px var(--cyan); }
    50% { opacity: 0.5; box-shadow: 0 0 12px var(--cyan); }
}

/* Outcome tile */
.outcome-tile {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-radius: 12px;
    padding: 16px 20px;
}
.outcome-label {
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0 0 6px 0;
}
.outcome-value {
    font-family: var(--font-body);
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    margin: 0;
}

/* No patient state */
.empty-state {
    text-align: center;
    padding: 80px 20px;
    border: 1px dashed var(--border-dim);
    border-radius: 16px;
    margin-top: 20px;
}
.empty-icon {
    font-size: 48px;
    margin-bottom: 12px;
    filter: grayscale(0.3);
}

/* Streamlit info/warning/error override */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border-color: var(--border-bright) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* Code blocks */
code {
    font-family: var(--font-mono) !important;
    background: rgba(0,212,255,0.08) !important;
    color: var(--cyan) !important;
    padding: 2px 8px !important;
    border-radius: 5px !important;
    font-size: 13px !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models_and_data():
    try:
        kmeans = joblib.load('kmeans_cluster_k3.pkl')
        scaler = joblib.load('scaler.pkl')
        df = pd.read_csv('diabetic_data_with_clusters.csv')
        return kmeans, scaler, df
    except FileNotFoundError:
        st.error("Model files not found. Place these in the working directory:")
        st.code("kmeans_cluster_k3.pkl\nscaler.pkl\ndiabetic_data_with_clusters.csv")
        st.stop()

kmeans, scaler, df = load_models_and_data()


# ─────────────────────────────────────────────
# CLUSTER DEFINITIONS
# ─────────────────────────────────────────────
CLUSTER_INFO = {
    0: {
        "name": "Moderate Complexity — Renal Focus",
        "short": "Renal Focus",
        "accent":       "#f59e0b",
        "card_class":   "ms-card-amber",
        "badge_class":  "badge-amber",
        "banner_bg":    "linear-gradient(135deg, #0c1628 0%, #1a1200 100%)",
        "banner_glow":  "#f59e0b",
        "readmission_rate": 47.17,
        "population_pct":   34.4,
        "risk_label":   "Moderate Risk",
        "avg_complexity":   33.0,
        "avg_medications":  23.0,
        "avg_inpatient":    0.45,
        "avg_emergency":    0.11,
        "insulin_pct":      71.7,
        "renal_pct":        11.7,
        "characteristics": [
            "High medication burden — avg 23 medications per patient",
            "71.7% require insulin therapy",
            "11.7% have documented renal disease",
            "97% show low healthcare engagement pattern",
            "Moderate recent healthcare utilisation",
        ],
        "care_recommendations": [
            "Regular nephrology monitoring and renal function assessment",
            "Medication adherence programs and regimen simplification",
            "Structured outpatient visits — quarterly minimum",
            "Diabetes education focused on kidney disease prevention",
            "Blood pressure control targets <130/80 mmHg for CKD",
            "Avoid nephrotoxic medications; review NSAID use",
        ],
        "intervention_priority": "High",
        "expected_reduction": "47% → 40% with targeted interventions",
        "monitoring_freq": "Quarterly check-ups",
    },
    1: {
        "name": "Low Complexity — Stable & Well-Controlled",
        "short": "Stable",
        "accent":       "#10b981",
        "card_class":   "ms-card-green",
        "badge_class":  "badge-green",
        "banner_bg":    "linear-gradient(135deg, #0c1628 0%, #001a10 100%)",
        "banner_glow":  "#10b981",
        "readmission_rate": 41.53,
        "population_pct":   44.3,
        "risk_label":   "Low Risk",
        "avg_complexity":   18.8,
        "avg_medications":  12.2,
        "avg_inpatient":    0.34,
        "avg_emergency":    0.09,
        "insulin_pct":      42.5,
        "renal_pct":        5.0,
        "characteristics": [
            "Lowest disease complexity score — 18.8",
            "Fewest medications — avg 12.2 per patient",
            "Only 42.5% require insulin therapy",
            "Lowest renal disease prevalence at 5%",
            "Minimal hospital visits and shortest stays",
        ],
        "care_recommendations": [
            "Annual diabetes check-ups with preventive focus",
            "Lifestyle and exercise guidance reinforcement",
            "Annual foot care and complication screening",
            "Telemedicine for routine follow-ups (cost-effective)",
            "Group education classes for continued engagement",
            "Maintain current stability — avoid over-intervention",
        ],
        "intervention_priority": "Low (Preventive)",
        "expected_reduction": "41.5% → 35% with prevention focus",
        "monitoring_freq": "Annual check-ups",
    },
    2: {
        "name": "High-Risk — Frequent Hospitalizations",
        "short": "High-Risk / Acute",
        "accent":       "#ef4444",
        "card_class":   "ms-card-red",
        "badge_class":  "badge-red",
        "banner_bg":    "linear-gradient(135deg, #0c1628 0%, #1a0000 100%)",
        "banner_glow":  "#ef4444",
        "readmission_rate": 72.61,
        "population_pct":   20.7,
        "risk_label":   "Critical Risk",
        "avg_complexity":   23.9,
        "avg_medications":  17.1,
        "avg_inpatient":    3.24,
        "avg_emergency":    1.23,
        "insulin_pct":      62.9,
        "renal_pct":        15.8,
        "characteristics": [
            "70% show acute / high-risk engagement pattern",
            "1.23 emergency visits avg — 11× higher than Cluster 1",
            "3.24 inpatient visits avg — 10× higher than Cluster 1",
            "62.9% require insulin therapy",
            "15.8% have documented renal disease — highest of all clusters",
        ],
        "care_recommendations": [
            "Intensive case management — dedicated coordinator per patient",
            "Weekly phone contact during identified high-risk periods",
            "Home visits for comprehensive social and medical assessment",
            "ED avoidance strategies and diversion protocols",
            "Post-discharge follow-up within 24–48 hours mandatory",
            "Mental health screening — depression and anxiety assessment",
            "Social determinants assessment — food, housing, transport",
            "Monthly nephrology monitoring for patients with renal disease",
        ],
        "intervention_priority": "Critical — Highest ROI",
        "expected_reduction": "72.6% → 60% (2,500+ readmissions prevented)",
        "monitoring_freq": "Monthly check-ups",
    },
}

CLUSTER_AVERAGES = {
    "patient_complexity":  {0: 33.0,  1: 18.8, 2: 23.9},
    "num_medications":     {0: 23.0,  1: 12.2, 2: 17.1},
    "number_inpatient":    {0: 0.45,  1: 0.34, 2: 3.24},
    "number_emergency":    {0: 0.11,  1: 0.09, 2: 1.23},
    "number_outpatient":   {0: 0.21,  1: 0.20, 2: 2.07},
    "time_in_hospital":    {0: 4.5,   1: 3.1,  2: 5.8},
    "num_lab_procedures":  {0: 44.2,  1: 38.1, 2: 47.3},
}

FEATURE_LABELS = {
    "patient_complexity":  "Complexity Score",
    "num_medications":     "Medications",
    "number_inpatient":    "Inpatient Visits",
    "number_emergency":    "Emergency Visits",
    "number_outpatient":   "Outpatient Visits",
    "time_in_hospital":    "Days in Hospital",
    "num_lab_procedures":  "Lab Procedures",
}

FEATURES_TO_SCALE = [
    'number_inpatient', 'total_recent_visits', 'number_diagnoses',
    'number_emergency', 'number_outpatient', 'time_in_hospital',
    'patient_complexity', 'num_medications', 'insulin_user',
    'medication_changed', 'num_procedures', 'has_renal_disease',
    'num_lab_procedures'
]

CLUSTER_COLORS = ["#f59e0b", "#10b981", "#ef4444"]
CLUSTER_LABELS = ["Cluster 0 — Renal Focus", "Cluster 1 — Stable", "Cluster 2 — High-Risk"]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def predict_patient_cluster(patient_data):
    features  = pd.DataFrame([patient_data[FEATURES_TO_SCALE]])
    scaled    = scaler.transform(features)
    cluster   = kmeans.predict(scaled)[0]
    distances = kmeans.transform(scaled)[0]
    return cluster, distances


@st.cache_resource
def compute_pca_components():
    """Compute PCA for all patients on clustering features"""
    from sklearn.decomposition import PCA
    X_scaled = scaler.transform(df[FEATURES_TO_SCALE])
    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)
    comp_2d = pca_2d.fit_transform(X_scaled)
    comp_3d = pca_3d.fit_transform(X_scaled)
    return pca_2d, pca_3d, comp_2d, comp_3d

pca_2d, pca_3d, components_2d, components_3d = compute_pca_components()


def plotly_dark(title="", height=340, show_legend=False):
    layout = dict(
        title=dict(
            text=title,
            font=dict(size=13, family="IBM Plex Mono", color="#7a9cc4"),
            x=0, xanchor="left",
        ),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color="#7a9cc4", size=12),
        margin=dict(l=14, r=14, t=44 if title else 20, b=14),
        showlegend=show_legend,
        legend=dict(
            font=dict(size=11, family="IBM Plex Sans", color="#7a9cc4"),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,212,255,0.15)",
            borderwidth=1,
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=11, color="#3d5a7a"),
            linecolor="#1e3050",
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(0,212,255,0.06)", zeroline=False,
            tickfont=dict(size=11, color="#3d5a7a"),
            linecolor="#1e3050",
        ),
    )
    return layout


def delta_row(patient_val, cluster_avg, higher_is_bad=True):
    if cluster_avg == 0:
        return f"<td class='wt-val'>{patient_val:.1f}</td><td class='wt-neutral'>—</td>"
    pct = (patient_val - cluster_avg) / cluster_avg * 100
    if abs(pct) < 5:
        delta_html = "<span class='wt-neutral'>≈ avg</span>"
    elif (pct > 0) == higher_is_bad:
        delta_html = f"<span class='wt-above'>▲ {abs(pct):.0f}%</span>"
    else:
        delta_html = f"<span class='wt-below'>▼ {abs(pct):.0f}%</span>"
    return f"<td class='wt-val'>{patient_val:.1f}</td><td>{delta_html}</td>"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-brand">MedSegment</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-mono">Patient Intelligence · K=3</p>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    pages = {
        "Home":             "🏠  Overview",
        "Patient Analysis": "🫀  Patient Analysis",
        "Cluster Stats":    "📊  Cluster Statistics",
        "Dashboard":        "📈  Dashboard",
    }
    for key, label in pages.items():
        active = st.session_state.page == key
        if st.sidebar.button(label, use_container_width=True, key=f"nav_{key}",
                             type="primary" if active else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:12px; line-height:2.2;">
        <span style="color:#3d5a7a; font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:1px; text-transform:uppercase;">Dataset</span><br>
        <span style="color:#7a9cc4;">UCI 130-US Hospitals</span><br><br>
        <span style="color:#3d5a7a; font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:1px; text-transform:uppercase;">Total Encounters</span><br>
        <span style="color:#00d4ff; font-family:'IBM Plex Mono',monospace; font-weight:500;">{len(df):,}</span><br><br>
        <span style="color:#3d5a7a; font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:1px; text-transform:uppercase;">Features</span><br>
        <span style="color:#7a9cc4;">{len(FEATURES_TO_SCALE)} clustering inputs</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:11px; color:#3d5a7a; line-height:1.6;">'
        '⚠️ Analytical insights only.<br>Not a substitute for clinical judgment.</p>',
        unsafe_allow_html=True)

page = st.session_state.page


# ═══════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <p class="page-eyebrow">🫀 Patient Risk Stratification System</p>
    <p class="page-title">Patient Segmentation &<br>Healthcare Intelligence</p>
    <p class="page-subtitle">Machine-learning powered risk stratification for diabetic patients — UCI 130-US Hospitals dataset</p>
    """, unsafe_allow_html=True)

    # ── Cluster cards ──
    c0, c1, c2 = st.columns(3)
    for col, cid in [(c0, 0), (c1, 1), (c2, 2)]:
        info = CLUSTER_INFO[cid]
        with col:
            st.markdown(f"""
            <div class="{info['card_class']}">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:10px;
                                 letter-spacing:1.5px; text-transform:uppercase; color:{info['accent']};">
                        Cluster {cid}
                    </span>
                    <span class="badge {info['badge_class']}">{info['risk_label']}</span>
                </div>
                <p style="font-family:'Syne',sans-serif; font-size:16px; font-weight:700;
                           color:#e8f4ff; margin:0 0 4px 0;">{info['short']}</p>
                <p style="font-size:12px; color:#3d5a7a; margin:0 0 14px 0; font-family:'IBM Plex Mono',monospace;">
                    {info['population_pct']:.1f}% of patients
                </p>
                <div class="stat-rows">
                    <div class="stat-row-item">
                        <span class="stat-row-label">Readmission</span>
                        <span class="stat-row-value" style="color:{info['accent']};">{info['readmission_rate']:.1f}%</span>
                    </div>
                    <div class="stat-row-item">
                        <span class="stat-row-label">Avg Medications</span>
                        <span class="stat-row-value" style="color:#7a9cc4;">{info['avg_medications']:.0f}</span>
                    </div>
                    <div class="stat-row-item">
                        <span class="stat-row-label">Renal Disease</span>
                        <span class="stat-row-value" style="color:#7a9cc4;">{info['renal_pct']:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── How-to + dataset glance ──
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.markdown('<p class="sec-label">How to use this system</p>', unsafe_allow_html=True)
        steps = [
            ("01", "Patient Analysis",
             "Load a random patient — the model instantly assigns them to a risk cluster."),
            ("02", "Understand Why",
             "Compare their clinical metrics against cluster averages to see what drove the assignment."),
            ("03", "Care Plan",
             "Review evidence-based, priority-ranked care recommendations for that cluster."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex; gap:16px; align-items:flex-start; margin-bottom:20px;">
                <div style="min-width:34px; height:34px; border-radius:8px;
                            background:rgba(0,212,255,0.08); border:1px solid rgba(0,212,255,0.2);
                            color:#00d4ff; font-family:'IBM Plex Mono',monospace; font-size:11px;
                            font-weight:600; display:flex; align-items:center; justify-content:center;
                            flex-shrink:0;">{num}</div>
                <div>
                    <p style="margin:0 0 2px 0; font-weight:600; font-size:14px;
                               color:#e8f4ff; font-family:'Syne',sans-serif;">{title}</p>
                    <p style="margin:0; font-size:13px; color:#7a9cc4; line-height:1.5;">{desc}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<p class="sec-label">Dataset at a glance</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ms-card-cyan">
            <table style="width:100%; font-size:13px; border-collapse:collapse;">
                <tr>
                    <td style="color:#3d5a7a; padding:8px 0; border-bottom:1px solid rgba(0,212,255,0.08);
                               font-family:'IBM Plex Mono',monospace; font-size:11px;">Total encounters</td>
                    <td style="text-align:right; font-weight:600; color:#00d4ff;
                               font-family:'IBM Plex Mono',monospace; border-bottom:1px solid rgba(0,212,255,0.08);">
                        {len(df):,}</td>
                </tr>
                <tr>
                    <td style="color:#3d5a7a; padding:8px 0; border-bottom:1px solid rgba(0,212,255,0.08);
                               font-family:'IBM Plex Mono',monospace; font-size:11px;">Cluster 0 — Renal Focus</td>
                    <td style="text-align:right; font-weight:600; color:#f59e0b;
                               font-family:'IBM Plex Mono',monospace; border-bottom:1px solid rgba(0,212,255,0.08);">
                        34.4%</td>
                </tr>
                <tr>
                    <td style="color:#3d5a7a; padding:8px 0; border-bottom:1px solid rgba(0,212,255,0.08);
                               font-family:'IBM Plex Mono',monospace; font-size:11px;">Cluster 1 — Stable</td>
                    <td style="text-align:right; font-weight:600; color:#10b981;
                               font-family:'IBM Plex Mono',monospace; border-bottom:1px solid rgba(0,212,255,0.08);">
                        44.3%</td>
                </tr>
                <tr>
                    <td style="color:#3d5a7a; padding:8px 0; border-bottom:1px solid rgba(0,212,255,0.08);
                               font-family:'IBM Plex Mono',monospace; font-size:11px;">Cluster 2 — High-Risk</td>
                    <td style="text-align:right; font-weight:600; color:#ef4444;
                               font-family:'IBM Plex Mono',monospace; border-bottom:1px solid rgba(0,212,255,0.08);">
                        20.7%</td>
                </tr>
                <tr>
                    <td style="color:#3d5a7a; padding:8px 0;
                               font-family:'IBM Plex Mono',monospace; font-size:11px;">Clustering features</td>
                    <td style="text-align:right; font-weight:600; color:#7a9cc4;
                               font-family:'IBM Plex Mono',monospace;">
                        {len(FEATURES_TO_SCALE)}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:12px; color:#3d5a7a;">⚠️ This tool provides analytical insights only and is NOT a '
        'substitute for clinical judgment. All recommendations must be reviewed by qualified healthcare professionals.</p>',
        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 2 — PATIENT ANALYSIS
# ═══════════════════════════════════════════════════════
elif page == "Patient Analysis":
    st.markdown("""
    <p class="page-eyebrow">🫀 Individual Patient Report</p>
    <p class="page-title">Patient Analysis</p>
    <p class="page-subtitle">Load a random patient to predict their risk cluster and generate a personalised care report.</p>
    """, unsafe_allow_html=True)

    btn_col, spacer, info_col = st.columns([1.4, 0.3, 4])
    with btn_col:
        if st.button("⟳  Load Random Patient", use_container_width=True):
            st.session_state.patient_idx = int(np.random.randint(0, len(df)))

    with info_col:
        if 'patient_idx' in st.session_state:
            st.markdown(
                f'<p style="font-size:14px; color:#3d5a7a; margin-top:12px; font-family:\'IBM Plex Mono\',monospace;">'
                f'Loaded encounter at index '
                f'<code>#{st.session_state.patient_idx}</code>'
                f'</p>',
                unsafe_allow_html=True)

    # ── Empty state ──
    if 'patient_idx' not in st.session_state:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🫀</div>
            <p style="font-size:16px; font-weight:600; color:#7a9cc4; margin:0 0 6px 0;
                       font-family:'Syne',sans-serif;">No patient loaded</p>
            <p style="font-size:13px; color:#3d5a7a;">
                Click <strong style="color:#00d4ff;">Load Random Patient</strong> to begin analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Patient data ──
    patient_idx  = st.session_state.patient_idx
    patient_data = df.iloc[patient_idx]
    cluster_id, distances = predict_patient_cluster(patient_data)
    info = CLUSTER_INFO[cluster_id]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── SECTION A: Cluster assignment banner ──
    glow_color = info['banner_glow']
    st.markdown(f"""
    <div style="background:{info['banner_bg']};
                border:1px solid rgba({','.join([str(int(glow_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4)])},0.35);
                border-radius:16px; padding:28px 32px; margin-bottom:6px;
                box-shadow: 0 0 40px rgba({','.join([str(int(glow_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4)])},0.10);">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
            <div>
                <p style="font-family:'IBM Plex Mono',monospace; font-size:10px; font-weight:500;
                           letter-spacing:2px; text-transform:uppercase; color:{info['accent']};
                           margin:0 0 8px 0;">▸ Cluster Assignment</p>
                <p style="font-family:'Syne',sans-serif; font-size:22px; font-weight:800;
                           color:#e8f4ff; margin:0 0 10px 0; line-height:1.2;">
                    Cluster {cluster_id} — {info['name']}</p>
                <span class="badge {info['badge_class']}">
                    <span class="pulse-dot" style="background:{info['accent']}; box-shadow:0 0 6px {info['accent']};"></span>
                    {info['risk_label']}
                </span>
            </div>
            <div style="text-align:right;">
                <p style="font-family:'IBM Plex Mono',monospace; font-size:10px; color:#3d5a7a;
                           margin:0 0 4px 0; letter-spacing:1px; text-transform:uppercase;">Cluster Readmission Rate</p>
                <p style="font-family:'Syne',sans-serif; font-size:40px; font-weight:800;
                           color:{info['accent']}; margin:0; line-height:1;">{info['readmission_rate']:.1f}%</p>
                <p style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#3d5a7a; margin:4px 0 0 0;">
                    {info['population_pct']:.1f}% of total population</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION B: Clinical snapshot metrics ──
    st.markdown('<p class="sec-label">Clinical Snapshot</p>', unsafe_allow_html=True)
    m0, m1, m2, m3, m4, m5 = st.columns(6)
    snap = [
        (m0, "Complexity Score", f"{patient_data.get('patient_complexity', 0):.1f}"),
        (m1, "Medications",      f"{int(patient_data.get('num_medications', 0))}"),
        (m2, "Days in Hospital", f"{patient_data.get('time_in_hospital', 0):.0f}"),
        (m3, "Emergency Visits", f"{patient_data.get('number_emergency', 0):.0f}"),
        (m4, "Inpatient Visits", f"{patient_data.get('number_inpatient', 0):.0f}"),
        (m5, "On Insulin",       "Yes ✓" if patient_data.get('insulin_user', 0) == 1 else "No"),
    ]
    for col, label, val in snap:
        with col:
            st.metric(label, val)

    # ── SECTION C: Why this cluster? ──
    st.markdown('<p class="sec-label">Why this cluster?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:13px; color:#3d5a7a; margin:-8px 0 14px 0; font-family:\'IBM Plex Mono\',monospace;">'
        'This patient\'s key metrics vs. cluster averages — showing what drove the assignment.</p>',
        unsafe_allow_html=True)

    col_why, col_radar = st.columns([1.2, 1])

    with col_why:
        rows = ""
        for feat, label in FEATURE_LABELS.items():
            try:
                p_val = float(patient_data.get(feat, 0))
            except Exception:
                p_val = 0.0
            c_avg = CLUSTER_AVERAGES[feat][cluster_id]
            delta = delta_row(p_val, c_avg, higher_is_bad=(feat != "num_lab_procedures"))
            rows += (
                f"<tr>"
                f"<td class='wt-metric'>{label}</td>"
                f"{delta}"
                f"<td class='wt-avg'>{c_avg:.1f}</td>"
                f"</tr>"
            )
        st.markdown(f"""
        <div class="ms-card" style="padding:0; overflow:hidden;">
            <table class="why-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>This Patient</th>
                        <th>vs Avg</th>
                        <th>Cluster Avg</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_radar:
        cats  = list(FEATURE_LABELS.values())
        feats = list(FEATURE_LABELS.keys())

        def norm(feat, val):
            # Scale is fixed to min/max of the 3 cluster centroids only.
            # This guarantees the cluster-avg trace is IDENTICAL for every
            # patient in the same cluster regardless of patient values.
            cluster_vals = list(CLUSTER_AVERAGES[feat].values())
            lo, hi = min(cluster_vals), max(cluster_vals)
            if hi == lo:
                return 0.5
            # Patient value is NOT clamped so spikes beyond centroid range are visible
            return (val - lo) / (hi - lo)

        p_vals = [norm(f, float(patient_data.get(f, 0))) for f in feats]
        c_vals = [norm(f, CLUSTER_AVERAGES[f][cluster_id]) for f in feats]

        # Radial range: always 0 to max(1.2, max patient value + 0.1)
        # so the cluster-avg ring stays at a fixed position and patient
        # outliers are shown beyond it rather than distorting the scale.
        r_max = max(1.2, max(p_vals) + 0.1) if p_vals else 1.2

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=c_vals + [c_vals[0]], theta=cats + [cats[0]],
            fill='toself',
            fillcolor="rgba(122,156,196,0.08)",
            line=dict(color="#3d5a7a", width=1.5, dash="dot"),
            name="Cluster avg",
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=p_vals + [p_vals[0]], theta=cats + [cats[0]],
            fill='toself',
            fillcolor=f"rgba({','.join([str(int(info['accent'].lstrip('#')[i:i+2],16)) for i in (0,2,4)])},0.12)",
            line=dict(color=info['accent'], width=2.5),
            name="This patient",
        ))
        fig_r.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=False, range=[0, r_max]),
                angularaxis=dict(
                    tickfont=dict(size=10, family="IBM Plex Mono", color="#3d5a7a"),
                    linecolor="#1e3050",
                    gridcolor="rgba(0,212,255,0.06)",
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(
                font=dict(size=11, family="IBM Plex Sans", color="#7a9cc4"),
                bgcolor="rgba(0,0,0,0)",
                orientation="h", y=-0.1,
            ),
            margin=dict(l=20, r=20, t=20, b=40),
            height=340,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # ── SECTION D: Cluster characteristics ──
    st.markdown('<p class="sec-label">Cluster Characteristics</p>', unsafe_allow_html=True)
    char_items = "".join([
        f'<div class="char-item">'
        f'<span style="color:{info["accent"]}; font-size:14px; flex-shrink:0;">◆</span>'
        f'<span>{c}</span></div>'
        for c in info['characteristics']
    ])
    st.markdown(f'<div class="ms-card" style="padding:14px 20px;">{char_items}</div>', unsafe_allow_html=True)

    # ── SECTION E: Care recommendations ──
    st.markdown('<p class="sec-label">Recommended Care Plan</p>', unsafe_allow_html=True)
    rec_items = "".join([
        f'<div class="rec-item">'
        f'<div class="rec-num">{i+1:02d}</div>'
        f'<span>{r}</span></div>'
        for i, r in enumerate(info['care_recommendations'])
    ])
    priority_badge_map = {0: "badge-amber", 1: "badge-green", 2: "badge-red"}
    st.markdown(f"""
    <div class="ms-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
            <span style="font-family:'Syne',sans-serif; font-size:14px; font-weight:600;
                          color:#e8f4ff;">Care Interventions</span>
            <span class="badge {priority_badge_map[cluster_id]}">Priority: {info['intervention_priority']}</span>
        </div>
        {rec_items}
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION F: Expected outcomes ──
    st.markdown('<p class="sec-label">Expected Outcomes</p>', unsafe_allow_html=True)
    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown(f"""
        <div class="outcome-tile">
            <p class="outcome-label">Expected readmission reduction</p>
            <p class="outcome-value">{info['expected_reduction']}</p>
        </div>
        """, unsafe_allow_html=True)
    with oc2:
        st.markdown(f"""
        <div class="outcome-tile">
            <p class="outcome-label">Recommended monitoring frequency</p>
            <p class="outcome-value">{info['monitoring_freq']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION G: Download ──
    st.markdown("<hr>", unsafe_allow_html=True)
    report_text = f"""
MEDSEGMENT — PATIENT HEALTHCARE INTERPRETATION REPORT
Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Patient Idx: #{patient_idx}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLUSTER ASSIGNMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cluster           : {cluster_id} — {info['name']}
Risk Level        : {info['risk_label']}
Readmission Rate  : {info['readmission_rate']:.2f}%
Population Share  : {info['population_pct']:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLINICAL SNAPSHOT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complexity Score  : {patient_data.get('patient_complexity', 0):.1f}
Medications       : {int(patient_data.get('num_medications', 0))}
Days in Hospital  : {patient_data.get('time_in_hospital', 0):.0f}
Emergency Visits  : {patient_data.get('number_emergency', 0):.0f}
Inpatient Visits  : {patient_data.get('number_inpatient', 0):.0f}
On Insulin        : {'Yes' if patient_data.get('insulin_user', 0) == 1 else 'No'}
Renal Disease     : {'Yes' if patient_data.get('has_renal_disease', 0) == 1 else 'No'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLUSTER CHARACTERISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{"".join([f"  • {c}\n" for c in info['characteristics']])}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CARE RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{"".join([f"  {i+1:02d}. {r}\n" for i, r in enumerate(info['care_recommendations'])])}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED OUTCOMES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Readmission reduction : {info['expected_reduction']}
Monitoring frequency  : {info['monitoring_freq']}
Intervention priority : {info['intervention_priority']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This report provides analytical insights based on healthcare utilisation patterns.
This is NOT a medical diagnosis or clinical judgment.
All recommendations must be reviewed by qualified healthcare professionals.
Individual patient circumstances must always take precedence.
    """.strip()

    dl_col, _ = st.columns([1.6, 4])
    with dl_col:
        st.download_button(
            label="📥  Download Report (.txt)",
            data=report_text,
            file_name=f"medsegment_patient_{patient_idx}_cluster_{cluster_id}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    st.markdown(
        '<p style="font-size:11px; color:#3d5a7a; margin-top:8px; font-family:\'IBM Plex Mono\',monospace;">'
        '⚠️ Analytical insights only — not a substitute for clinical judgment.</p>',
        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 3 — CLUSTER STATISTICS
# ═══════════════════════════════════════════════════════
elif page == "Cluster Stats":
    st.markdown("""
    <p class="page-eyebrow">📊 Comparative Analysis</p>
    <p class="page-title">Cluster Statistics</p>
    <p class="page-subtitle">Clinical and utilisation characteristics compared across the three patient clusters.</p>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sec-label">Key Metrics by Cluster</p>', unsafe_allow_html=True)

    # Custom HTML table — dark theme, no invisible-text issues
    _cols   = ["Metric", "Cluster 0 — Renal Focus", "Cluster 1 — Stable", "Cluster 2 — High-Risk"]
    _accent = ["#f59e0b", "#10b981", "#ef4444"]
    _rows = [
        ("Readmission Rate (%)",  "47.17%",  "41.53%",  "72.61%"),
        ("Avg Complexity Score",  "33.0",    "18.8",    "23.9"),
        ("Avg Medications",       "23.0",    "12.2",    "17.1"),
        ("Insulin Users (%)",     "71.7%",   "42.5%",   "62.9%"),
        ("Renal Disease (%)",     "11.7%",   "5.0%",    "15.8%"),
        ("Avg Emergency Visits",  "0.11",    "0.09",    "1.23"),
        ("Avg Inpatient Visits",  "0.45",    "0.34",    "3.24"),
        ("Population Share (%)",  "34.4%",   "44.3%",   "20.7%"),
    ]
    _header_cells = "".join([
        f'<th style="text-align:left; padding:11px 16px; background:rgba(0,212,255,0.05); '
        f'color:#3d5a7a; font-family:\'IBM Plex Mono\',monospace; font-size:10px; '
        f'font-weight:500; letter-spacing:1px; text-transform:uppercase; '
        f'border-bottom:1px solid rgba(0,212,255,0.12);">{c}</th>'
        for c in _cols
    ])
    _body_rows = ""
    for i, row in enumerate(_rows):
        metric = row[0]
        vals   = row[1:]
        _td_metric = (
            f'<td style="padding:11px 16px; color:#7a9cc4; font-size:13px; '
            f'font-family:\'IBM Plex Sans\',sans-serif; '
            f'border-bottom:1px solid rgba(0,212,255,0.06);">{metric}</td>'
        )
        _td_vals = "".join([
            f'<td style="padding:11px 16px; color:{_accent[vi]}; font-family:\'IBM Plex Mono\',monospace; '
            f'font-size:13px; font-weight:500; '
            f'border-bottom:1px solid rgba(0,212,255,0.06);">{v}</td>'
            for vi, v in enumerate(vals)
        ])
        _body_rows += f'<tr style="background:{"rgba(0,212,255,0.02)" if i%2==0 else "transparent"};">{_td_metric}{_td_vals}</tr>'

    st.markdown(f"""
    <div style="background:#071428; border:1px solid rgba(0,212,255,0.12); border-radius:14px;
                overflow:hidden; margin-bottom:8px;">
        <table style="width:100%; border-collapse:collapse;">
            <thead><tr>{_header_cells}</tr></thead>
            <tbody>{_body_rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure(go.Bar(
            x=CLUSTER_LABELS, y=[47.17, 41.53, 72.61],
            marker=dict(color=CLUSTER_COLORS,
                        line=dict(color="rgba(0,212,255,0.2)", width=1)),
            text=["47.17%", "41.53%", "72.61%"],
            textposition="outside",
            textfont=dict(size=12, family="IBM Plex Mono", color="#7a9cc4"),
        ))
        l = plotly_dark("Readmission rate by cluster", 320)
        fig.update_layout(**l)
        fig.update_yaxes(title_text="Rate (%)", ticksuffix="%", range=[0, 85])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Bar(
            x=CLUSTER_LABELS, y=[23.0, 12.2, 17.1],
            marker=dict(color=CLUSTER_COLORS,
                        line=dict(color="rgba(0,212,255,0.2)", width=1)),
            text=["23", "12.2", "17.1"],
            textposition="outside",
            textfont=dict(size=12, family="IBM Plex Mono", color="#7a9cc4"),
        ))
        l = plotly_dark("Avg medications per patient", 320)
        fig.update_layout(**l)
        fig.update_yaxes(title_text="Count", range=[0, 30])
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=CLUSTER_LABELS, y=[33.0, 18.8, 23.9],
            mode='lines+markers+text',
            line=dict(color="rgba(0,212,255,0.25)", width=2, dash='dot'),
            marker=dict(size=16, color=CLUSTER_COLORS,
                        line=dict(color="#040d1c", width=2)),
            text=["33.0", "18.8", "23.9"],
            textposition="top center",
            textfont=dict(size=12, family="IBM Plex Mono", color="#7a9cc4"),
        ))
        l = plotly_dark("Patient complexity score by cluster", 320)
        fig.update_layout(**l)
        fig.update_yaxes(title_text="Score", range=[10, 42])
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        util_df = pd.DataFrame({
            "Cluster":    CLUSTER_LABELS * 3,
            "Visit Type": (["Emergency"] * 3 + ["Inpatient"] * 3 + ["Outpatient"] * 3),
            "Avg Visits": [0.11, 0.09, 1.23, 0.45, 0.34, 3.24, 0.21, 0.20, 2.07],
        })
        fig = px.line(
            util_df, x="Visit Type", y="Avg Visits", color="Cluster", markers=True,
            color_discrete_map={
                "Cluster 0 — Renal Focus": "#f59e0b",
                "Cluster 1 — Stable": "#10b981",
                "Cluster 2 — High-Risk": "#ef4444",
            },
        )
        fig.update_traces(marker=dict(size=10, line=dict(color="#040d1c", width=2)))
        l = plotly_dark("Healthcare utilisation by cluster", 320, show_legend=True)
        l["legend"]["orientation"] = "h"
        l["legend"]["y"] = -0.22
        fig.update_layout(**l)
        fig.update_yaxes(title_text="Avg visits")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sec-label">Insulin & Renal Disease Prevalence</p>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)

    with col5:
        fig = go.Figure(go.Pie(
            labels=CLUSTER_LABELS,
            values=[71.7, 42.5, 62.9],
            marker=dict(
                colors=CLUSTER_COLORS,
                line=dict(color="#040d1c", width=3)
            ),
            hole=0.5,
            textinfo="percent",
            textfont=dict(size=12, family="IBM Plex Mono", color="#040d1c"),
        ))
        l = plotly_dark("Insulin users (%)", 310, show_legend=True)
        l["legend"]["orientation"] = "v"
        l["legend"]["x"] = 1.0
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        fig = go.Figure(go.Pie(
            labels=CLUSTER_LABELS,
            values=[11.7, 5.0, 15.8],
            marker=dict(
                colors=CLUSTER_COLORS,
                line=dict(color="#040d1c", width=3)
            ),
            hole=0.5,
            textinfo="percent",
            textfont=dict(size=12, family="IBM Plex Mono", color="#040d1c"),
        ))
        l = plotly_dark("Renal disease prevalence (%)", 310, show_legend=True)
        l["legend"]["orientation"] = "v"
        l["legend"]["x"] = 1.0
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sec-label">Cluster Separation in 2D PCA Space</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:13px; color:#3d5a7a; margin:-8px 0 14px 0;">'
        f'Patient distribution across principal component space '
        f'({pca_2d.explained_variance_ratio_.sum():.1%} variance explained)</p>',
        unsafe_allow_html=True)
    
    # Create 2D PCA plot
    pca_df = pd.DataFrame({
        'PC1': components_2d[:, 0],
        'PC2': components_2d[:, 1],
        'cluster': df['cluster'].astype(str),
        'complexity': df['patient_complexity'].round(1),
        'medications': df['num_medications'].astype(int),
    })
    
    fig_2d = px.scatter(
        pca_df,
        x='PC1', y='PC2',
        color='cluster',
        color_discrete_map={
            '0': '#f59e0b',
            '1': '#10b981',
            '2': '#ef4444',
        },
        hover_data={
            'complexity': ':.1f',
            'medications': True,
            'PC1': ':.2f',
            'PC2': ':.2f',
        },
        labels={
            'PC1': f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})',
            'PC2': f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})',
            'cluster': 'Cluster',
        },
        title=f'All {len(df):,} Patients in 2D PCA Space',
        opacity=0.6,
    )
    fig_2d.update_traces(
        marker=dict(size=5, line=dict(width=0)),
        selector=dict(mode='markers')
    )
    l2d = plotly_dark("", 420, show_legend=True)
    l2d['showlegend'] = True
    l2d['legend'] = dict(
        title_text='Patient Cluster',
        font=dict(size=11, family="IBM Plex Sans", color="#7a9cc4"),
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,212,255,0.15)",
        borderwidth=1,
        orientation="v", x=1.02, y=1.0
    )
    fig_2d.update_layout(**l2d)
    fig_2d.update_xaxes(
        title_font=dict(size=11, family="IBM Plex Mono", color="#7a9cc4"),
        tickfont=dict(size=10, family="IBM Plex Mono", color="#3d5a7a"),
        gridcolor="rgba(0,212,255,0.06)",
    )
    fig_2d.update_yaxes(
        title_font=dict(size=11, family="IBM Plex Mono", color="#7a9cc4"),
        tickfont=dict(size=10, family="IBM Plex Mono", color="#3d5a7a"),
        gridcolor="rgba(0,212,255,0.06)",
    )
    st.plotly_chart(fig_2d, use_container_width=True)
elif page == "Dashboard":
    st.markdown("""
    <p class="page-eyebrow">📈 Segmentation Overview</p>
    <p class="page-title">Dashboard</p>
    <p class="page-subtitle">High-level overview of patient segmentation results and key system insights.</p>
    """, unsafe_allow_html=True)

    avg_readmit = (df['readmitted'] == 'Yes').mean() * 100 if 'readmitted' in df.columns else 52.3
    k0, k1, k2, k3 = st.columns(4)
    with k0: st.metric("Total Patients",       f"{len(df):,}")
    with k1: st.metric("Risk Clusters",         "3")
    with k2: st.metric("Avg Readmission Rate",  f"{avg_readmit:.1f}%")
    with k3: st.metric("Clustering Features",   f"{len(FEATURES_TO_SCALE)}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 3D PCA Visualization ──
    st.markdown('<p class="sec-label">3D Cluster Visualization · Principal Component Space</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:13px; color:#3d5a7a; margin:-8px 0 16px 0;">'
        f'Interactive 3D projection of all {len(df):,} patients · {pca_3d.explained_variance_ratio_.sum():.1%} total variance captured</p>',
        unsafe_allow_html=True)
    
    # Prepare 3D PCA data
    pca_3d_df = pd.DataFrame({
        'PC1': components_3d[:, 0],
        'PC2': components_3d[:, 1],
        'PC3': components_3d[:, 2],
        'cluster': df['cluster'].astype(str),
        'complexity': df['patient_complexity'].round(1),
        'medications': df['num_medications'].astype(int),
        'visits': (df['number_inpatient'] + df['number_emergency'] + df['number_outpatient']).round(1),
    })
    
    fig_3d = px.scatter_3d(
        pca_3d_df,
        x='PC1', y='PC2', z='PC3',
        color='cluster',
        color_discrete_map={
            '0': '#f59e0b',
            '1': '#10b981',
            '2': '#ef4444',
        },
        hover_data={
            'complexity': ':.1f',
            'medications': True,
            'visits': ':.1f',
            'PC1': False, 'PC2': False, 'PC3': False,
        },
        labels={
            'PC1': f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
            'PC2': f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
            'PC3': f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})',
            'cluster': 'Cluster',
        },
        title=f'3D PCA: Patient Cluster Topology',
        opacity=0.7,
    )
    fig_3d.update_traces(
        marker=dict(size=4, line=dict(width=0)),
        selector=dict(mode='markers')
    )
    l3d = plotly_dark("", 500, show_legend=True)
    l3d['showlegend'] = True
    l3d['legend'] = dict(
        title_text='Patient Cluster',
        font=dict(size=11, family="IBM Plex Sans", color="#7a9cc4"),
        bgcolor="rgba(0,0,0,0.3)",
        bordercolor="rgba(0,212,255,0.15)",
        borderwidth=1,
        orientation="v", x=0.02, y=0.98
    )
    l3d['scene'] = dict(
        xaxis=dict(
            backgroundcolor="rgba(0,0,0,0.2)",
            gridcolor="rgba(0,212,255,0.06)",
            showbackground=True,
            zeroline=False,
            tickfont=dict(size=10, family="IBM Plex Mono", color="#3d5a7a"),
        ),
        yaxis=dict(
            backgroundcolor="rgba(0,0,0,0.2)",
            gridcolor="rgba(0,212,255,0.06)",
            showbackground=True,
            zeroline=False,
            tickfont=dict(size=10, family="IBM Plex Mono", color="#3d5a7a"),
        ),
        zaxis=dict(
            backgroundcolor="rgba(0,0,0,0.2)",
            gridcolor="rgba(0,212,255,0.06)",
            showbackground=True,
            zeroline=False,
            tickfont=dict(size=10, family="IBM Plex Mono", color="#3d5a7a"),
        ),
    )
    fig_3d.update_layout(**l3d)
    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cluster_counts = df['cluster'].value_counts().sort_index() \
            if 'cluster' in df.columns else pd.Series([34.4, 44.3, 20.7])
        fig = go.Figure(go.Pie(
            labels=CLUSTER_LABELS,
            values=cluster_counts.values,
            marker=dict(colors=CLUSTER_COLORS, line=dict(color="#040d1c", width=3)),
            hole=0.48,
            textinfo="percent",
            textfont=dict(size=12, family="IBM Plex Mono"),
        ))
        l = plotly_dark("Patient distribution across clusters", 360, show_legend=True)
        l["legend"]["orientation"] = "v"
        l["legend"]["x"] = 1.0
        fig.update_layout(**l)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'readmitted' in df.columns and 'cluster' in df.columns:
            data_rows = []
            for cid in range(3):
                cdf  = df[df['cluster'] == cid]
                rate = cdf['readmitted'].eq('Yes').mean() * 100 if len(cdf) > 0 \
                       else CLUSTER_INFO[cid]['readmission_rate']
                data_rows.append({
                    "Cluster": f"Cluster {cid}",
                    "Readmitted": rate,
                    "Not Readmitted": 100 - rate
                })
        else:
            data_rows = [
                {"Cluster": "Cluster 0", "Readmitted": 47.17, "Not Readmitted": 52.83},
                {"Cluster": "Cluster 1", "Readmitted": 41.53, "Not Readmitted": 58.47},
                {"Cluster": "Cluster 2", "Readmitted": 72.61, "Not Readmitted": 27.39},
            ]
        rdf = pd.DataFrame(data_rows)
        fig = px.bar(
            rdf, x="Cluster", y=["Readmitted", "Not Readmitted"], barmode="stack",
            color_discrete_map={
                "Readmitted":     "#ef4444",
                "Not Readmitted": "#10b981",
            },
        )
        l = plotly_dark("Readmission breakdown by cluster", 360, show_legend=True)
        l["legend"]["orientation"] = "h"
        l["legend"]["y"] = -0.18
        fig.update_layout(**l)
        fig.update_traces(marker_line_color="#040d1c", marker_line_width=1)
        fig.update_yaxes(title_text="Patients (%)", ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sec-label">Mean Feature Values by Cluster</p>', unsafe_allow_html=True)
    if 'cluster' in df.columns:
        feat_stats = df.groupby('cluster')[FEATURES_TO_SCALE].mean().round(2)
        feat_stats.index = ["Cluster 0 — Renal Focus", "Cluster 1 — Stable", "Cluster 2 — High-Risk"]
        _cluster_accents = ["#f59e0b", "#10b981", "#ef4444"]
        _feat_headers = ["Feature"] + list(feat_stats.index)
        _fh_cells = "".join([
            f'<th style="text-align:{"left" if i==0 else "right"}; padding:11px 14px; '
            f'background:rgba(0,212,255,0.04); color:{"#3d5a7a" if i==0 else _cluster_accents[i-1]}; '
            f'font-family:\'IBM Plex Mono\',monospace; font-size:10px; font-weight:500; '
            f'letter-spacing:0.8px; text-transform:uppercase; '
            f'border-bottom:1px solid rgba(0,212,255,0.12); white-space:nowrap;">{h}</th>'
            for i, h in enumerate(_feat_headers)
        ])
        _feat_body = ""
        for fi, feat in enumerate(feat_stats.columns):
            vals = feat_stats[feat].values
            _td_feat = (
                f'<td style="padding:10px 14px; color:#7a9cc4; font-size:12px; '
                f'font-family:\'IBM Plex Sans\',sans-serif; white-space:nowrap; '
                f'border-bottom:1px solid rgba(0,212,255,0.06);">{feat.replace("_"," ").title()}</td>'
            )
            _td_vals = "".join([
                f'<td style="padding:10px 14px; text-align:right; color:{_cluster_accents[vi]}; '
                f'font-family:\'IBM Plex Mono\',monospace; font-size:12px; font-weight:500; '
                f'border-bottom:1px solid rgba(0,212,255,0.06);">{v}</td>'
                for vi, v in enumerate(vals)
            ])
            _bg = "rgba(0,212,255,0.02)" if fi % 2 == 0 else "transparent"
            _feat_body += f'<tr style="background:{_bg};">{_td_feat}{_td_vals}</tr>'

        st.markdown(f"""
        <div style="background:#071428; border:1px solid rgba(0,212,255,0.12); border-radius:14px;
                    overflow:auto; margin-bottom:8px;">
            <table style="width:100%; border-collapse:collapse; min-width:700px;">
                <thead><tr>{_fh_cells}</tr></thead>
                <tbody>{_feat_body}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="color:#3d5a7a; font-size:13px;">Cluster column not found in dataset.</p>',
            unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sec-label">Key Insights</p>', unsafe_allow_html=True)
    i0, i1, i2 = st.columns(3)
    with i0:
        st.markdown(f"""
        <div class="ms-card-green">
            <p style="font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:1.5px;
                       text-transform:uppercase; color:#10b981; margin:0 0 8px 0;">Lowest Risk · Cluster 1</p>
            <p style="font-size:13px; color:#7a9cc4; margin:0; line-height:1.6;">
                Stable, well-controlled patients with the lowest readmission rate of
                <strong style="color:#e8f4ff;">{CLUSTER_INFO[1]['readmission_rate']:.1f}%</strong>.
                Preventive care and telemedicine are most cost-effective here.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with i1:
        st.markdown(f"""
        <div class="ms-card-amber">
            <p style="font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:1.5px;
                       text-transform:uppercase; color:#f59e0b; margin:0 0 8px 0;">Moderate Risk · Cluster 0</p>
            <p style="font-size:13px; color:#7a9cc4; margin:0; line-height:1.6;">
                High medication burden and renal disease prevalence at
                <strong style="color:#e8f4ff;">{CLUSTER_INFO[0]['renal_pct']:.1f}%</strong>.
                Structured quarterly monitoring can reduce readmissions meaningfully.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown(f"""
        <div class="ms-card-red">
            <p style="font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:1.5px;
                       text-transform:uppercase; color:#ef4444; margin:0 0 8px 0;">Critical Risk · Cluster 2</p>
            <p style="font-size:13px; color:#7a9cc4; margin:0; line-height:1.6;">
                Readmission rate of <strong style="color:#e8f4ff;">{CLUSTER_INFO[2]['readmission_rate']:.1f}%</strong>
                driven by 10× more inpatient visits.
                Intensive case management offers the highest ROI for intervention.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:8px 0 20px 0;">
    <span style="font-family:'Syne',sans-serif; font-size:14px; font-weight:700; color:#00d4ff;">MedSegment</span>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#3d5a7a; margin:0 12px;">·</span>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#3d5a7a;">UCI Diabetes 130-US Hospitals Dataset</span>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#3d5a7a; margin:0 12px;">·</span>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#3d5a7a;">KMeans (k=3)</span>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#3d5a7a; margin:0 12px;">·</span>
    <span style="font-family:'IBM Plex Mono',monospace; font-size:11px; color:#3d5a7a; font-style:italic;">Not a substitute for clinical judgment</span>
</div>
""", unsafe_allow_html=True)