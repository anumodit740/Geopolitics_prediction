"""
config.py — Central Configuration for the Global Country Stability Intelligence System
=======================================================================================

This module centralizes every tunable parameter, file path, column definition,
and model hyperparameter used across the project. No other module should contain
hardcoded values — everything flows from here.

Design Principles:
    1. Single Source of Truth — change a value here, it propagates everywhere.
    2. Recruiter-Readable — grouped by concern with clear docstrings.
    3. No Hardcoded Absolute Paths — all paths are relative to PROJECT_ROOT.
    4. Type-Safe Constants — uppercase naming, immutable intent.

Author : Anumol
Project: Global Country Stability Intelligence System
"""

import os
import logging
from pathlib import Path


# ============================================================================
# 1. PROJECT PATHS
# ============================================================================
# All paths are built relative to this file's location so the project works
# on any machine without editing. pathlib.Path handles OS-specific separators.

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Data Directories ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- Data Files ---
RAW_DATA_FILE = RAW_DATA_DIR / "world-data-2023.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_data.csv"
FEATURE_ENGINEERED_FILE = PROCESSED_DATA_DIR / "feature_engineered_data.csv"
FINAL_DATASET_FILE = PROCESSED_DATA_DIR / "final_model_ready.csv"

# --- Model Artifacts ---
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_LIST_PATH = MODELS_DIR / "feature_list.joblib"

# --- Reports & Figures ---
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"
SHAP_SUMMARY_PATH = FIGURES_DIR / "shap_summary.png"
CLASS_DISTRIBUTION_PATH = FIGURES_DIR / "class_distribution.png"
CORRELATION_HEATMAP_PATH = FIGURES_DIR / "correlation_heatmap.png"
MODEL_COMPARISON_PATH = FIGURES_DIR / "model_comparison.png"

# --- Logs ---
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "pipeline.log"


# ============================================================================
# 2. RANDOM SEED
# ============================================================================
# A fixed seed guarantees reproducible results across every run.
# Used in: train/test splits, model initialization, KNN imputation.

RANDOM_STATE = 42


# ============================================================================
# 3. TRAIN / TEST / VALIDATION SETTINGS
# ============================================================================
# With only ~190 usable rows, an 80/20 split gives ~152 train / 38 test.
# Stratification ensures each stability class is proportionally represented
# in both sets — critical with only ~63 samples per class.

TEST_SIZE = 0.20          # 20% held out for final evaluation
VALIDATION_SIZE = 0.15    # 15% of training set carved out for early stopping
N_CV_FOLDS = 5            # Stratified K-Fold for cross-validation
STRATIFY = True           # Always stratify splits by target class
SHUFFLE = True            # Shuffle before splitting


# ============================================================================
# 4. TARGET VARIABLE SETTINGS
# ============================================================================
# No pre-existing label exists in the raw data. We engineer a Country
# Stability Index (CSI) from health/development outcomes and bin it into
# 3 risk tiers. The model then predicts these tiers from economic/policy
# features — keeping target-definition columns OUT of the feature set
# to prevent data leakage.

TARGET_COLUMN = "stability_label"

# Number of quantile-based bins for the stability index
N_STABILITY_CLASSES = 3

# Human-readable class labels (mapped to 0, 1, 2 after binning)
CLASS_LABELS = {
    0: "Stable",       # Top tertile — high development, low mortality
    1: "Watch",        # Middle tertile — transitional / mixed signals
    2: "At-Risk",      # Bottom tertile — high mortality, weak economy
}

# Reverse mapping for display purposes (label → integer)
LABEL_TO_INT = {v: k for k, v in CLASS_LABELS.items()}

# Colors used for visualizations and the Streamlit dashboard
CLASS_COLORS = {
    "Stable":  "#2ecc71",   # Green
    "Watch":   "#f39c12",   # Amber
    "At-Risk": "#e74c3c",   # Red
}

# --- Columns used to BUILD the target (Stability Index) ---
# These capture the *outcome* of stability: health and human development.
# They are used ONLY for label creation, then DROPPED from features.
TARGET_POSITIVE_SIGNALS = [
    "Life expectancy",
    "Physicians per thousand",
    "Gross primary education enrollment (%)",
    "Gross tertiary education enrollment (%)",
]

TARGET_NEGATIVE_SIGNALS = [
    "Infant mortality",
    "Maternal mortality ratio",
]


# ============================================================================
# 5. COLUMN DEFINITIONS (from actual dataset inspection)
# ============================================================================
# The raw CSV has 35 columns. Most "numeric" columns are stored as strings
# with $, %, commas, and trailing whitespace. These lists drive the cleaning
# pipeline in data_preprocessing.py.

# --- Columns to DROP before any processing (IDs, no predictive value) ---
COLUMNS_TO_DROP = [
    "Abbreviation",
    "Calling Code",
    "Capital/Major City",
    "Currency-Code",
    "Largest city",
    "Official language",     # 76 unique values — too sparse for 190 rows
]

# --- Countries to DROP (near-empty rows with 20+ null columns) ---
COUNTRIES_TO_DROP = [
    "Palestinian National Authority",
    "Vatican City",
    "Nauru",
    "North Macedonia",
    "Eswatini",
]

# --- Columns stored as "XX.XX%" strings → need % stripped, cast to float ---
PERCENTAGE_COLUMNS = [
    "Agricultural Land( %)",
    "CPI Change (%)",
    "Forested Area (%)",
    "Gross primary education enrollment (%)",
    "Gross tertiary education enrollment (%)",
    "Out of pocket health expenditure",
    "Population: Labor force participation (%)",
    "Tax revenue (%)",
    "Total tax rate",
    "Unemployment rate",
]

# --- Columns stored as "$X,XXX" strings → need $, commas, spaces stripped ---
CURRENCY_COLUMNS = [
    "GDP",
    "Gasoline Price",
    "Minimum wage",
]

# --- Columns stored as "X,XXX" comma-separated integers → strip commas ---
COMMA_NUMERIC_COLUMNS = [
    "Density\n(P/Km2)",
    "Land Area(Km2)",
    "Armed Forces size",
    "Co2-Emissions",
    "Population",
    "Urban_population",
]

# --- Columns that are already numeric (float64) — no cleaning needed ---
ALREADY_NUMERIC_COLUMNS = [
    "Birth Rate",
    "Fertility Rate",
    "Infant mortality",
    "Life expectancy",
    "Maternal mortality ratio",
    "Physicians per thousand",
    "Latitude",
    "Longitude",
]

# --- CPI is mostly numeric but has commas in large values (e.g. "4,583.71") ---
CPI_COLUMNS = [
    "CPI",
]

# --- Friendly column rename mapping (applied after cleaning) ---
COLUMN_RENAME_MAP = {
    "Density\n(P/Km2)": "Density_per_km2",
    "Agricultural Land( %)": "Agricultural_Land_pct",
    "Land Area(Km2)": "Land_Area_km2",
    "Armed Forces size": "Armed_Forces",
    "Co2-Emissions": "CO2_Emissions",
    "CPI Change (%)": "CPI_Change_pct",
    "Fertility Rate": "Fertility_Rate",
    "Forested Area (%)": "Forested_Area_pct",
    "Gasoline Price": "Gasoline_Price",
    "Gross primary education enrollment (%)": "Primary_Education_pct",
    "Gross tertiary education enrollment (%)": "Tertiary_Education_pct",
    "Infant mortality": "Infant_Mortality",
    "Life expectancy": "Life_Expectancy",
    "Maternal mortality ratio": "Maternal_Mortality",
    "Minimum wage": "Minimum_Wage",
    "Out of pocket health expenditure": "OOP_Health_Expenditure_pct",
    "Physicians per thousand": "Physicians_per_1000",
    "Population: Labor force participation (%)": "Labor_Participation_pct",
    "Tax revenue (%)": "Tax_Revenue_pct",
    "Total tax rate": "Total_Tax_Rate_pct",
    "Unemployment rate": "Unemployment_Rate_pct",
    "Urban_population": "Urban_Population",
    "Birth Rate": "Birth_Rate",
    "GDP": "GDP",
    "CPI": "CPI",
    "Population": "Population",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
}


# ============================================================================
# 6. FEATURE ENGINEERING SETTINGS
# ============================================================================
# Composite features are built from z-scored base columns. Each index
# captures a business-relevant dimension of country stability.

# --- Engineered feature names (created in feature_engineering.py) ---
ENGINEERED_FEATURES = [
    "GDP_per_Capita",
    "Urbanization_Rate",
    "CO2_per_Capita",
    "Military_Burden",
    "CO2_per_GDP",
    "Dependency_Ratio_Proxy",
    "Healthcare_Access_Gap",
    "Education_Pipeline",
    "Log_GDP",
    "Log_Population",
    "Abs_Latitude",
    "Economic_Stress_Index",
    "Workforce_Stability_Score",
    "Healthcare_Stability_Score",
    "Human_Development_Proxy",
    "Country_Risk_Score",
]

# --- Features that the MODEL uses for prediction ---
# These are the non-leaky columns: economic, demographic, and policy inputs.
# Health-outcome columns (Life expectancy, Infant mortality, etc.) are
# excluded because they were used to define the target.
PREDICTION_FEATURES_BASE = [
    "GDP",
    "CPI",
    "CPI_Change_pct",
    "Gasoline_Price",
    "Minimum_Wage",
    "Unemployment_Rate_pct",
    "Labor_Participation_pct",
    "Tax_Revenue_pct",
    "Total_Tax_Rate_pct",
    "Birth_Rate",
    "Fertility_Rate",
    "Density_per_km2",
    "Agricultural_Land_pct",
    "Land_Area_km2",
    "Armed_Forces",
    "CO2_Emissions",
    "Forested_Area_pct",
    "Primary_Education_pct",
    "Tertiary_Education_pct",
    "OOP_Health_Expenditure_pct",
    "Population",
    "Urban_Population",
    "Latitude",
    "Longitude",
]

# --- Columns that need log-transformation (heavy right-skew) ---
LOG_TRANSFORM_COLUMNS = [
    "GDP",
    "Population",
    "CO2_Emissions",
    "Land_Area_km2",
    "Urban_Population",
    "Armed_Forces",
]

# --- Columns with >10% missing → create binary missingness indicator ---
HIGH_MISSING_COLUMNS = [
    "Minimum_Wage",       # 23.1% missing
    "Tax_Revenue_pct",    # 13.3% missing
    "Armed_Forces",       # 12.3% missing
    "Gasoline_Price",     # 10.3% missing
]

# --- KNN Imputation settings ---
KNN_IMPUTE_NEIGHBORS = 5


# ============================================================================
# 7. MODEL HYPERPARAMETERS
# ============================================================================
# Baseline hyperparameters for each model. These are conservative defaults
# tuned for a small dataset (~190 rows). Grid/random search can refine them
# further during model_training.py.

# --- Logistic Regression ---
# Why: Fully interpretable coefficients, serves as explainability benchmark.
# C=1.0 is the default regularization; max_iter raised because convergence
# can be slow with many features on small data.
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "l1_ratio": 0,             # Equivalent to penalty="l2" (sklearn >= 1.8)
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",     # Handles any residual class imbalance
}

# --- Random Forest ---
# Why: Robust ensemble baseline with built-in feature importance.
# n_estimators=200 is a safe default; max_depth=10 prevents overfitting
# on 190 rows; min_samples_leaf=5 ensures each leaf has ~2.5% of data.
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
    "n_jobs": -1,
}

# --- XGBoost ---
# Why: Primary model — best balance of accuracy + SHAP explainability.
# learning_rate=0.05 with 300 trees = slow learning for small data.
# max_depth=4 keeps trees shallow to prevent overfitting.
# subsample + colsample_bytree add stochasticity as regularization.
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,            # L1 regularization
    "reg_lambda": 1.0,           # L2 regularization
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

# --- Model Selection ---
# The pipeline trains all models and picks the best by MACRO_F1.
MODELS_TO_TRAIN = ["LogisticRegression", "RandomForest", "XGBoost"]
PRIMARY_METRIC = "macro_f1"


# ============================================================================
# 8. EVALUATION SETTINGS
# ============================================================================
# Metrics and visualization parameters for model evaluation.

EVALUATION_METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "cohen_kappa",
    "roc_auc_ovr",
]

# Confusion matrix display settings
CONFUSION_MATRIX_FIGSIZE = (8, 6)
CONFUSION_MATRIX_CMAP = "Blues"

# SHAP settings
SHAP_MAX_DISPLAY = 15        # Top N features in SHAP summary plot
SHAP_PLOT_TYPE = "bar"       # Options: "bar", "dot", "violin"


# ============================================================================
# 9. LOGGING SETTINGS
# ============================================================================
# Structured logging tracks every pipeline step for debugging and auditing.
# Logs are written to both console (INFO) and file (DEBUG for full detail).

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO
LOG_FILE_LEVEL = logging.DEBUG


def setup_logging() -> logging.Logger:
    """
    Configure project-wide logging with console + file handlers.

    Returns:
        logging.Logger: Configured root logger for the project.

    Usage:
        from config import setup_logging
        logger = setup_logging()
        logger.info("Pipeline started")
    """
    # Ensure log directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("StabilityIntelligence")
    logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # --- Console handler (INFO and above) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(
        logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )

    # --- File handler (DEBUG and above — full detail) ---
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(LOG_FILE_LEVEL)
    file_handler.setFormatter(
        logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ============================================================================
# 10. STREAMLIT DASHBOARD SETTINGS
# ============================================================================
# Configuration for the interactive Streamlit app (app.py).

APP_TITLE = "🌍 Global Country Stability Intelligence System"
APP_SUBTITLE = "ML-Powered Country Risk Classification & Analysis"
APP_ICON = "🌍"
APP_LAYOUT = "wide"

# Sidebar filter options
STREAMLIT_THEME = {
    "primary_color": "#1a73e8",
    "background_color": "#0e1117",
    "secondary_background": "#1e2130",
    "text_color": "#fafafa",
    "font": "Inter",
}

# Prediction display labels (used in dashboard cards)
PREDICTION_LABELS = {
    0: {"label": "🟢 Stable",  "description": "Low risk — strong institutions and economy"},
    1: {"label": "🟡 Watch",   "description": "Moderate risk — mixed signals, monitor closely"},
    2: {"label": "🔴 At-Risk", "description": "High risk — significant instability indicators"},
}


# ============================================================================
# 11. DIRECTORY INITIALIZATION
# ============================================================================
# Ensures all required directories exist when config is first imported.
# This prevents FileNotFoundError during pipeline execution.

def create_directories() -> None:
    """
    Create all project directories if they do not already exist.
    Called automatically on import to guarantee directory structure.
    """
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Auto-create directories on first import
create_directories()
