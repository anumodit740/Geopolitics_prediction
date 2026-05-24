"""
app.py — Streamlit Dashboard for Global Country Stability Intelligence
========================================================================

A production-grade, recruiter-friendly SaaS-style dashboard that lets
users adjust country indicators and receive real-time ML predictions
for country stability risk classification.

Usage:
    streamlit run app.py

Author : Anumol
Project: Global Country Stability Intelligence System
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --- Project imports ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ============================================================================
# PAGE CONFIG (must be FIRST Streamlit call)
# ============================================================================

st.set_page_config(
    page_title="Country Stability Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS — Modern SaaS Dashboard Theme
# ============================================================================

def inject_custom_css() -> None:
    """Inject premium dark-theme CSS to override default Streamlit styling."""
    st.markdown("""
    <style>
    /* ===== Global ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #c9d1d9 !important;
    }

    /* ===== Hero Section ===== */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f97316 100%);
        border-radius: 20px;
        padding: 48px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.25);
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
        line-height: 1.15;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: rgba(255, 255, 255, 0.85);
        margin: 0 0 20px 0;
        font-weight: 400;
        line-height: 1.5;
        position: relative;
        z-index: 1;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        position: relative;
        z-index: 1;
    }

    .hero-badge {
        background: rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(10px);
        color: #ffffff;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* ===== KPI Cards ===== */
    .kpi-card {
        background: linear-gradient(145deg, #1e1e2f, #252540);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }

    .kpi-label {
        font-size: 0.78rem;
        color: #8b8fa3;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 4px 0;
        line-height: 1.1;
    }

    .kpi-sub {
        font-size: 0.8rem;
        color: #6b7280;
        font-weight: 400;
    }

    /* ===== Prediction Card ===== */
    .prediction-card {
        border-radius: 20px;
        padding: 36px 32px;
        text-align: center;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .prediction-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
    }

    .pred-stable {
        background: linear-gradient(145deg, #0d3320, #064e3b);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pred-stable::after { background: linear-gradient(90deg, #10b981, #34d399); }

    .pred-watch {
        background: linear-gradient(145deg, #422006, #78350f);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .pred-watch::after { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

    .pred-risk {
        background: linear-gradient(145deg, #3b0d0d, #7f1d1d);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .pred-risk::after { background: linear-gradient(90deg, #ef4444, #f87171); }

    .pred-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .pred-class {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 8px 0;
    }

    .pred-confidence {
        font-size: 1rem;
        opacity: 0.85;
        font-weight: 400;
    }

    /* ===== Section Headers ===== */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0e0e0;
        margin: 32px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        letter-spacing: -0.3px;
    }

    /* ===== Insight Card ===== */
    .insight-card {
        background: linear-gradient(145deg, #1e1e2f, #252540);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 16px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
        line-height: 1.7;
        color: #c9d1d9;
        font-size: 0.92rem;
    }

    .insight-card strong {
        color: #a5b4fc;
    }

    /* ===== Footer ===== */
    .footer {
        text-align: center;
        padding: 32px 0 16px 0;
        margin-top: 48px;
        border-top: 1px solid rgba(99, 102, 241, 0.15);
        color: #6b7280;
        font-size: 0.82rem;
    }

    .footer a {
        color: #818cf8;
        text-decoration: none;
    }

    /* ===== Hide default Streamlit branding ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ===== Slider styling ===== */
    .stSlider > div > div {
        color: #c9d1d9 !important;
    }

    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model() -> Optional[Any]:
    """
    Load the trained model pipeline from disk.

    Uses st.cache_resource so the model is loaded ONCE and shared across
    all user sessions — not reloaded on every interaction.
    """
    model_path = config.BEST_MODEL_PATH
    if not model_path.exists():
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception:
        return None


@st.cache_resource
def load_feature_list() -> List[str]:
    """Load the saved feature list to ensure input columns match training."""
    try:
        return joblib.load(config.FEATURE_LIST_PATH)
    except Exception:
        return []


@st.cache_data
def load_feature_importance() -> Optional[pd.DataFrame]:
    """Load metrics summary CSV for feature importance display."""
    try:
        return pd.read_csv(config.REPORTS_DIR / "metrics" / "metrics_summary.csv")
    except Exception:
        return None


# ============================================================================
# FEATURE ENGINEERING HELPERS (mirror feature_engineering.py logic)
# ============================================================================

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Division that returns default on zero or NaN denominator."""
    if b == 0 or pd.isna(b):
        return default
    return a / b


def compute_engineered_features(raw: Dict[str, float]) -> Dict[str, float]:
    """
    Replicate the feature engineering logic from feature_engineering.py
    for a single country's raw indicators.

    This ensures the one-row prediction DataFrame has the same engineered
    columns the model was trained on.
    """
    eng = {}

    # --- Economic ---
    eng["GDP_per_Capita"] = safe_divide(raw["GDP"], raw["Population"])
    eng["Log_GDP"] = np.log1p(max(raw["GDP"], 0))
    eng["Log_Population"] = np.log1p(max(raw["Population"], 0))

    # Economic Stress Index (simplified z-score substitute: use raw values)
    eng["Economic_Stress_Index"] = (
        raw["Unemployment_Rate_pct"] * 0.02
        + raw["CPI_Change_pct"] * 0.01
        + raw["Total_Tax_Rate_pct"] * 0.005
        - eng["GDP_per_Capita"] * 1e-6
        - raw["Labor_Participation_pct"] * 0.005
    )
    eng["Inflation_Unemployment_Pressure"] = (
        raw["CPI_Change_pct"] * 0.01 + raw["Unemployment_Rate_pct"] * 0.02
    ) / 2
    eng["Tax_Burden_Index"] = safe_divide(
        raw["Total_Tax_Rate_pct"], raw["Tax_Revenue_pct"]
    )

    # --- Healthcare ---
    eng["Healthcare_Access_Gap"] = safe_divide(
        raw["OOP_Health_Expenditure_pct"], raw.get("Physicians_per_1000", 1.5)
    )

    # --- Education / Workforce ---
    primary_edu = raw.get("Primary_Education_pct", 100.0)
    tertiary_edu = raw.get("Tertiary_Education_pct", 50.0)

    eng["Human_Capital_Index"] = (
        primary_edu * 0.005
        + tertiary_edu * 0.005
        + raw["Labor_Participation_pct"] * 0.005
    ) / 3
    eng["Workforce_Stability_Score"] = (
        raw["Labor_Participation_pct"] * 0.005
        + tertiary_edu * 0.005
        - raw["Unemployment_Rate_pct"] * 0.01
        - raw["Birth_Rate"] * 0.005
    ) / 4
    eng["Education_Pipeline"] = safe_divide(tertiary_edu, primary_edu)
    eng["Education_Gap_Index"] = max(primary_edu - tertiary_edu, 0)

    # --- Demographic ---
    eng["Urbanization_Rate"] = safe_divide(
        raw["Urban_Population"], raw["Population"]
    )
    eng["Population_Pressure_Index"] = (
        raw["Density_per_km2"] * 0.0001
        + raw["Birth_Rate"] * 0.01
        + raw["Fertility_Rate"] * 0.05
    ) / 3
    eng["Dependency_Ratio_Proxy"] = safe_divide(
        raw["Birth_Rate"], raw["Labor_Participation_pct"]
    )
    eng["Military_Burden"] = safe_divide(
        raw["Armed_Forces"], raw["Population"]
    )
    eng["Abs_Latitude"] = abs(raw["Latitude"])

    # --- Environment ---
    eng["CO2_per_Capita"] = safe_divide(raw["CO2_Emissions"], raw["Population"])
    eng["CO2_per_GDP"] = safe_divide(raw["CO2_Emissions"], raw["GDP"])
    eng["Environmental_Pressure_Index"] = (
        eng["CO2_per_Capita"] * 0.01
        - raw["Forested_Area_pct"] * 0.005
        + raw["Agricultural_Land_pct"] * 0.003
    ) / 3
    eng["Resource_Land_Balance"] = safe_divide(
        raw["Forested_Area_pct"], raw["Agricultural_Land_pct"]
    )
    eng["Fuel_Cost_Pressure"] = safe_divide(
        raw["Gasoline_Price"], raw["Minimum_Wage"]
    )

    return eng


# ============================================================================
# INPUT BUILDER
# ============================================================================

def build_input_dataframe(
    raw_inputs: Dict[str, float],
    feature_list: List[str],
) -> pd.DataFrame:
    """
    Build a single-row DataFrame that exactly matches the model's
    expected feature order and columns.

    The model was trained on a specific set of 47 features in a specific
    order.  This function ensures the prediction input matches exactly,
    filling any missing columns with safe defaults (0.0).
    """
    # Compute engineered features
    engineered = compute_engineered_features(raw_inputs)

    # Merge raw + engineered
    all_features = {**raw_inputs, **engineered}

    # Missingness flags (user provides all values, so no missing)
    all_features["Minimum_Wage_is_missing"] = 0
    all_features["Tax_Revenue_pct_is_missing"] = 0
    all_features["Armed_Forces_is_missing"] = 0
    all_features["Gasoline_Price_is_missing"] = 0

    # Build DataFrame with exact feature order
    row = {}
    for feat in feature_list:
        row[feat] = all_features.get(feat, 0.0)

    return pd.DataFrame([row])


# ============================================================================
# PREDICTION
# ============================================================================

def predict_risk(
    model: Any,
    input_df: pd.DataFrame,
) -> Tuple[int, str, float, Optional[np.ndarray]]:
    """
    Run the model on a single-row input and return class + probabilities.

    Returns:
        (class_id, class_label, confidence, probabilities)
    """
    pred = model.predict(input_df)[0]
    pred = int(pred)
    label = config.CLASS_LABELS.get(pred, "Unknown")

    probas = None
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(input_df)[0]
        confidence = float(probas.max())

    return pred, label, confidence, probas


# ============================================================================
# RENDERING FUNCTIONS
# ============================================================================

def render_hero() -> None:
    """Render the gradient hero banner at the top of the page."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🌍 Global Country Stability Intelligence</div>
        <div class="hero-subtitle">
            ML-powered country risk assessment using socioeconomic,
            healthcare, and demographic indicators
        </div>
        <div class="hero-badges">
            <span class="hero-badge">Machine Learning</span>
            <span class="hero-badge">Risk Analytics</span>
            <span class="hero-badge">Country Intelligence</span>
            <span class="hero-badge">XGBoost</span>
            <span class="hero-badge">Streamlit Dashboard</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_card(
    pred_class: int,
    label: str,
    confidence: float,
) -> None:
    """Render the main prediction result as a styled card."""
    style_map = {
        0: ("pred-stable", "#10b981", "This country shows strong stability indicators."),
        1: ("pred-watch", "#f59e0b", "This country has moderate risk signals that warrant monitoring."),
        2: ("pred-risk", "#ef4444", "This country exhibits significant instability risk factors."),
    }
    css_class, color, interpretation = style_map.get(
        pred_class, ("pred-watch", "#f59e0b", "Classification uncertain.")
    )

    conf_pct = confidence * 100
    icon_map = {0: "✅", 1: "⚠️", 2: "🚨"}
    icon = icon_map.get(pred_class, "❓")

    st.markdown(f"""
    <div class="prediction-card {css_class}">
        <div class="pred-label" style="color: {color};">Predicted Risk Category</div>
        <div class="pred-class" style="color: {color};">{icon} {label}</div>
        <div class="pred-confidence" style="color: rgba(255,255,255,0.8);">
            Model Confidence: <strong style="color:{color};">{conf_pct:.1f}%</strong>
        </div>
        <div style="margin-top:14px; font-size:0.88rem; color:rgba(255,255,255,0.65); line-height:1.6;">
            {interpretation}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, sub: str, color: str) -> None:
    """Render a single KPI metric card."""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color: {color};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def render_probability_chart(probas: np.ndarray) -> None:
    """Render a horizontal bar chart showing class probabilities."""
    if probas is None:
        return

    labels = [config.CLASS_LABELS.get(i, f"Class {i}") for i in range(len(probas))]
    colors = ["#10b981", "#f59e0b", "#ef4444"]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, probas, color=colors[:len(probas)], height=0.55, edgecolor="none")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, color="#c9d1d9", fontweight="600")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability", fontsize=11, color="#8b8fa3", labelpad=8)
    ax.invert_yaxis()

    # Value labels on bars
    for bar, val in zip(bars, probas):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontsize=12, color="#e0e0e0", fontweight="700",
        )

    ax.tick_params(axis="x", colors="#6b7280", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2a2a4a")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#2a2a4a", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_feature_importance_chart(input_df: pd.DataFrame) -> None:
    """
    Render feature importance. Try to load from saved reports first;
    fall back to showing the user's input contribution as a bar chart.
    """
    # Try loading the XGBoost feature importance image
    xgb_img = config.FIGURES_DIR / "feature_importance_xgboost.png"
    if xgb_img.exists():
        st.image(str(xgb_img), use_container_width=True)
        return

    # Fallback: show top user input values as a bar chart
    if not input_df.empty:
        values = input_df.iloc[0].sort_values(ascending=False).head(12)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0f0f1a")
        ax.set_facecolor("#0f0f1a")

        palette = sns.color_palette("viridis", n_colors=len(values))
        ax.barh(values.index[::-1], values.values[::-1], color=palette[::-1], edgecolor="none")
        ax.set_xlabel("Feature Value", fontsize=11, color="#8b8fa3")
        ax.tick_params(axis="y", labelsize=10, colors="#c9d1d9")
        ax.tick_params(axis="x", colors="#6b7280", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#2a2a4a")
        ax.spines["left"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_business_interpretation(
    pred_class: int,
    label: str,
    raw_inputs: Dict[str, float],
) -> None:
    """Render a business-oriented interpretation of the prediction."""
    # Identify key risk drivers
    drivers = []
    if raw_inputs.get("Unemployment_Rate_pct", 0) > 12:
        drivers.append("High unemployment rate signals economic distress")
    if raw_inputs.get("Birth_Rate", 0) > 30:
        drivers.append("High birth rate indicates demographic pressure")
    if raw_inputs.get("CPI_Change_pct", 0) > 10:
        drivers.append("Elevated inflation erodes purchasing power")
    if raw_inputs.get("Total_Tax_Rate_pct", 0) > 60:
        drivers.append("Heavy tax burden may limit economic growth")
    if raw_inputs.get("OOP_Health_Expenditure_pct", 0) > 50:
        drivers.append("High out-of-pocket health costs signal weak public healthcare")
    if raw_inputs.get("Density_per_km2", 0) > 500:
        drivers.append("High population density increases resource strain")
    if raw_inputs.get("Labor_Participation_pct", 0) < 45:
        drivers.append("Low labor participation suggests workforce challenges")

    if not drivers:
        if pred_class == 0:
            drivers.append("Indicators are within healthy ranges across dimensions")
        elif pred_class == 1:
            drivers.append("Some indicators are near threshold levels")
        else:
            drivers.append("Multiple indicators signal elevated risk")

    driver_html = "".join([f"<li>{d}</li>" for d in drivers[:4]])

    context_map = {
        0: "The country demonstrates <strong>strong fundamentals</strong> across economic, healthcare, and demographic dimensions. Investment and partnership opportunities are favorable.",
        1: "The country shows <strong>mixed signals</strong>. While some indicators are healthy, others warrant closer monitoring. Analysts should track quarterly changes.",
        2: "The country exhibits <strong>significant vulnerability</strong> across multiple dimensions. Risk mitigation strategies and deeper due diligence are recommended.",
    }

    st.markdown(f"""
    <div class="insight-card">
        <strong>Key Risk Drivers:</strong>
        <ul style="margin: 8px 0 12px 0; padding-left: 20px;">
            {driver_html}
        </ul>
        <strong>Business Context:</strong><br>
        {context_map.get(pred_class, "")}
        <br><br>
        <strong>Why This Matters:</strong><br>
        This system helps <strong>analytics consultants</strong>, <strong>investment analysts</strong>,
        and <strong>policy researchers</strong> rapidly screen country-level risk using
        transparent, explainable ML models trained on real socioeconomic data.
    </div>
    """, unsafe_allow_html=True)


def render_footer() -> None:
    """Render the footer with project credits."""
    st.markdown("""
    <div class="footer">
        Built as an end-to-end ML project using
        <strong>Python</strong> &middot; <strong>Scikit-learn</strong> &middot;
        <strong>XGBoost</strong> &middot; <strong>Streamlit</strong><br>
        Global Country Stability Intelligence System &copy; 2024
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar() -> Dict[str, float]:
    """
    Render the sidebar with input sliders and return the raw input
    values as a dictionary.
    """
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 12px 0 20px 0;">
        <div style="font-size:1.3rem; font-weight:700; color:#e0e0e0;">
            🎛️ Country Indicators
        </div>
        <div style="font-size:0.78rem; color:#6b7280; margin-top:4px;">
            Adjust sliders to simulate a country profile
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # --- Economic ---
    st.sidebar.markdown("**📊 Economic**")
    gdp = st.sidebar.slider(
        "GDP (USD)", min_value=1_000_000, max_value=5_000_000_000_000,
        value=50_000_000_000, step=1_000_000_000, format="$%d",
    )
    cpi = st.sidebar.slider(
        "CPI", min_value=80.0, max_value=500.0, value=125.0, step=1.0,
    )
    cpi_change = st.sidebar.slider(
        "CPI Change (%)", min_value=-5.0, max_value=50.0, value=2.5, step=0.5,
    )
    unemployment = st.sidebar.slider(
        "Unemployment Rate (%)", min_value=0.0, max_value=30.0, value=5.5, step=0.5,
    )
    tax_revenue = st.sidebar.slider(
        "Tax Revenue (% GDP)", min_value=0.0, max_value=40.0, value=16.0, step=0.5,
    )
    total_tax = st.sidebar.slider(
        "Total Tax Rate (%)", min_value=5.0, max_value=100.0, value=37.0, step=1.0,
    )
    min_wage = st.sidebar.slider(
        "Minimum Wage (USD/hr)", min_value=0.01, max_value=15.0, value=1.5, step=0.1,
    )
    gasoline = st.sidebar.slider(
        "Gasoline Price (USD/L)", min_value=0.0, max_value=2.5, value=1.0, step=0.05,
    )

    st.sidebar.markdown("---")

    # --- Demographic ---
    st.sidebar.markdown("**👥 Demographic**")
    population = st.sidebar.slider(
        "Population", min_value=10_000, max_value=1_400_000_000,
        value=10_000_000, step=1_000_000, format="%d",
    )
    urban_pop = st.sidebar.slider(
        "Urban Population", min_value=5_000, max_value=900_000_000,
        value=5_000_000, step=500_000, format="%d",
    )
    density = st.sidebar.slider(
        "Density (people/km2)", min_value=1.0, max_value=1000.0, value=87.0, step=5.0,
    )
    birth_rate = st.sidebar.slider(
        "Birth Rate (per 1000)", min_value=5.0, max_value=50.0, value=18.0, step=0.5,
    )
    fertility = st.sidebar.slider(
        "Fertility Rate", min_value=0.9, max_value=7.0, value=2.3, step=0.1,
    )
    labor_part = st.sidebar.slider(
        "Labor Participation (%)", min_value=35.0, max_value=90.0, value=62.0, step=1.0,
    )

    st.sidebar.markdown("---")

    # --- Geographic & Resources ---
    st.sidebar.markdown("**🌿 Geography & Resources**")
    land_area = st.sidebar.slider(
        "Land Area (km2)", min_value=100, max_value=17_000_000,
        value=500_000, step=50_000, format="%d",
    )
    ag_land = st.sidebar.slider(
        "Agricultural Land (%)", min_value=0.0, max_value=85.0, value=40.0, step=1.0,
    )
    forest = st.sidebar.slider(
        "Forested Area (%)", min_value=0.0, max_value=99.0, value=32.0, step=1.0,
    )
    co2 = st.sidebar.slider(
        "CO2 Emissions (kt)", min_value=10.0, max_value=1_000_000.0,
        value=12_000.0, step=1000.0, format="%.0f",
    )
    armed = st.sidebar.slider(
        "Armed Forces", min_value=0, max_value=3_000_000,
        value=30_000, step=5_000, format="%d",
    )
    oop_health = st.sidebar.slider(
        "Out-of-Pocket Health (%)", min_value=0.0, max_value=85.0, value=30.0, step=1.0,
    )
    latitude = st.sidebar.slider(
        "Latitude", min_value=-45.0, max_value=65.0, value=17.0, step=1.0,
    )
    longitude = st.sidebar.slider(
        "Longitude", min_value=-175.0, max_value=180.0, value=20.0, step=1.0,
    )

    return {
        "GDP": float(gdp),
        "CPI": float(cpi),
        "CPI_Change_pct": float(cpi_change),
        "Unemployment_Rate_pct": float(unemployment),
        "Tax_Revenue_pct": float(tax_revenue),
        "Total_Tax_Rate_pct": float(total_tax),
        "Minimum_Wage": float(min_wage),
        "Gasoline_Price": float(gasoline),
        "Population": float(population),
        "Urban_Population": float(urban_pop),
        "Density_per_km2": float(density),
        "Birth_Rate": float(birth_rate),
        "Fertility_Rate": float(fertility),
        "Labor_Participation_pct": float(labor_part),
        "Land_Area_km2": float(land_area),
        "Agricultural_Land_pct": float(ag_land),
        "Forested_Area_pct": float(forest),
        "CO2_Emissions": float(co2),
        "Armed_Forces": float(armed),
        "OOP_Health_Expenditure_pct": float(oop_health),
        "Latitude": float(latitude),
        "Longitude": float(longitude),
    }


# ============================================================================
# MAIN APP
# ============================================================================

def main() -> None:
    """Main Streamlit application entry point."""

    inject_custom_css()

    # --- Load model ---
    model = load_model()
    feature_list = load_feature_list()

    # --- Hero ---
    render_hero()

    # --- Check model availability ---
    if model is None:
        st.error(
            "**Model artifact not found.**  \n"
            "Please run `python main.py` first to train and save the model.  \n"
            f"Expected path: `{config.BEST_MODEL_PATH}`"
        )
        render_footer()
        return

    # --- Sidebar inputs ---
    raw_inputs = render_sidebar()

    # --- Build prediction input ---
    input_df = build_input_dataframe(raw_inputs, feature_list)

    # --- Predict ---
    try:
        pred_class, label, confidence, probas = predict_risk(model, input_df)
    except Exception as e:
        st.error(
            f"**Prediction failed.**  \n"
            f"Input columns may not match training features.  \n"
            f"Error: `{e}`"
        )
        render_footer()
        return

    # --- Layout: Prediction + Probabilities ---
    col_pred, col_prob = st.columns([1, 1], gap="large")

    with col_pred:
        render_prediction_card(pred_class, label, confidence)

    with col_prob:
        st.markdown('<div class="section-header">Class Probabilities</div>',
                    unsafe_allow_html=True)
        render_probability_chart(probas)

    # --- KPI Cards ---
    st.markdown('<div class="section-header">Key Indicators</div>',
                unsafe_allow_html=True)

    engineered = compute_engineered_features(raw_inputs)
    k1, k2, k3, k4 = st.columns(4, gap="medium")

    with k1:
        conf_color = "#10b981" if confidence > 0.7 else "#f59e0b" if confidence > 0.5 else "#ef4444"
        render_kpi_card(
            "Model Confidence",
            f"{confidence:.0%}",
            "Prediction certainty",
            conf_color,
        )

    with k2:
        esi = engineered.get("Economic_Stress_Index", 0)
        esi_color = "#ef4444" if esi > 0.1 else "#f59e0b" if esi > -0.1 else "#10b981"
        render_kpi_card(
            "Economic Stress",
            f"{esi:.3f}",
            "Higher = more stress",
            esi_color,
        )

    with k3:
        hag = engineered.get("Healthcare_Access_Gap", 0)
        hag_color = "#ef4444" if hag > 30 else "#f59e0b" if hag > 15 else "#10b981"
        render_kpi_card(
            "Healthcare Gap",
            f"{hag:.1f}",
            "OOP cost / physicians",
            hag_color,
        )

    with k4:
        hci = engineered.get("Human_Capital_Index", 0)
        hci_color = "#10b981" if hci > 0.3 else "#f59e0b" if hci > 0.15 else "#ef4444"
        render_kpi_card(
            "Human Capital",
            f"{hci:.3f}",
            "Education + labor score",
            hci_color,
        )

    # --- Feature Importance ---
    st.markdown('<div class="section-header">Feature Importance (XGBoost)</div>',
                unsafe_allow_html=True)
    render_feature_importance_chart(input_df)

    # --- Business Interpretation ---
    st.markdown('<div class="section-header">Business Interpretation</div>',
                unsafe_allow_html=True)
    render_business_interpretation(pred_class, label, raw_inputs)

    # --- Model Performance Summary ---
    metrics_df = load_feature_importance()
    if metrics_df is not None and not metrics_df.empty:
        st.markdown('<div class="section-header">Model Performance (Test Set)</div>',
                    unsafe_allow_html=True)

        # Show comparison chart
        cm_img = config.FIGURES_DIR / "model_comparison.png"
        if cm_img.exists():
            st.image(str(cm_img), use_container_width=True)
        else:
            st.dataframe(
                metrics_df.style.format(precision=4).set_properties(
                    **{"background-color": "#1e1e2f", "color": "#c9d1d9"}
                ),
                use_container_width=True,
            )

    # --- Footer ---
    render_footer()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
