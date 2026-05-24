"""
src/evaluation.py — Model Evaluation, Comparison & Reporting
==============================================================

This module evaluates every trained model on the held-out test set,
compares them across business-relevant metrics, generates visualizations,
and produces human-readable interpretation reports.

Evaluation Philosophy:
    ~~~~~~~~~~~~~~~~~~~~~~
    This is NOT a Kaggle competition where accuracy wins.  This is a
    BUSINESS system where the cost of errors is asymmetric:

        - Missing a High-Risk country (false negative on class 2)
          could mean an investor enters an unstable market or an NGO
          fails to allocate aid.  This is EXPENSIVE.

        - Mislabeling a Stable country as Watch (false positive)
          causes unnecessary caution — annoying but not dangerous.

    Therefore our metric priority is:
        1. Macro F1       — ensures ALL classes are predicted well
        2. At-Risk Recall — catches every genuinely unstable country
        3. Weighted F1    — overall balanced performance
        4. Accuracy       — secondary sanity check only

Outputs:
    reports/
    ├── figures/
    │   ├── confusion_matrix_{model}.png
    │   ├── feature_importance_{model}.png
    │   └── model_comparison.png
    └── metrics/
        ├── classification_report_{model}.txt
        ├── metrics_summary.csv
        └── model_interpretation.txt

Author : Anumol
Project: Global Country Stability Intelligence System
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no GUI windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --- Project imports ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# ============================================================================
# MODULE-LEVEL LOGGER & SETTINGS
# ============================================================================

logger = config.setup_logging()

# Suppress matplotlib font warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# --- Plot style defaults ---
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("notebook", font_scale=1.1)

# --- Ensure output directories exist ---
METRICS_DIR = config.REPORTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- Class label mapping for display ---
DISPLAY_LABELS = [
    config.CLASS_LABELS.get(i, f"Class {i}")
    for i in range(config.N_STABILITY_CLASSES)
]


# ============================================================================
# 1. SINGLE MODEL EVALUATION
# ============================================================================

def evaluate_single_model(
    model_name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate one trained model/pipeline on the test set and return all
    business-relevant metrics.

    Metrics Computed:
        - accuracy:       overall correctness (secondary metric)
        - macro_precision: unweighted mean precision across all classes
        - macro_recall:   unweighted mean recall across all classes
        - macro_f1:       unweighted mean F1 (PRIMARY metric)
        - weighted_f1:    class-size-weighted F1 (secondary)
        - at_risk_recall: recall specifically for class 2 (At-Risk)
                          — the most important business metric

    Why at_risk_recall separately?
        In a consulting presentation, the client will ask: "Of all the
        countries that ARE genuinely at-risk, how many did your model
        catch?"  That's recall for class 2.  A model with 90% accuracy
        but 50% at-risk recall is DANGEROUS for real-world use.

    Args:
        model_name: Human-readable model name for logging.
        model: Trained sklearn Pipeline or classifier.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        Dict of {metric_name: score} for this model.
    """
    logger.info(f"Evaluating {model_name} on test set ({len(y_test)} samples)...")

    # --- Generate predictions ---
    y_pred = model.predict(X_test)

    # --- Compute metrics ---
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
    }

    # --- At-Risk recall (class 2) ---
    # This is the single most important business metric.
    # We compute per-class recall and extract the At-Risk class.
    per_class_recall = recall_score(
        y_test, y_pred, average=None, zero_division=0
    )
    at_risk_class_id = 2  # As defined in config.CLASS_LABELS
    if at_risk_class_id < len(per_class_recall):
        metrics["at_risk_recall"] = per_class_recall[at_risk_class_id]
    else:
        metrics["at_risk_recall"] = 0.0
        logger.warning(
            f"At-Risk class ({at_risk_class_id}) not found in predictions. "
            f"Setting at_risk_recall to 0.0."
        )

    # --- Log results ---
    logger.info(f"  {model_name} Test Results:")
    for metric, value in metrics.items():
        if metric != "model":
            logger.info(f"    {metric:20s}: {value:.4f}")

    return metrics


# ============================================================================
# 2. EVALUATE ALL MODELS
# ============================================================================

def evaluate_all_models(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate every trained model on the test set and return a comparison
    DataFrame sorted by the primary metric (Macro F1).

    Args:
        trained_models: Dict of {model_name: trained_pipeline}.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        pd.DataFrame: One row per model, columns are metric names,
                      sorted by macro_f1 descending.
    """
    logger.info("=" * 60)
    logger.info("EVALUATING ALL MODELS ON TEST SET")
    logger.info("=" * 60)

    all_metrics = []

    for model_name, model in trained_models.items():
        try:
            metrics = evaluate_single_model(model_name, model, X_test, y_test)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Evaluation failed for {model_name}: {e}")
            continue

    if not all_metrics:
        logger.error("No models were evaluated successfully.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_metrics)
    results_df = results_df.sort_values("macro_f1", ascending=False)
    results_df = results_df.reset_index(drop=True)

    logger.info(f"\nModel Comparison:\n{results_df.to_string(index=False)}")

    return results_df


# ============================================================================
# 3. CLASSIFICATION REPORT
# ============================================================================

def generate_classification_report(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> Dict:
    """
    Generate a detailed per-class classification report and save it as
    a text file for documentation and interview reference.

    The report includes precision, recall, F1, and support for each of
    the 3 stability classes — exactly what a recruiter or interviewer
    would expect to see in a portfolio project.

    Args:
        model: Trained pipeline.
        X_test: Test features.
        y_test: True labels.
        model_name: Name for file naming and headers.

    Returns:
        Dict: sklearn classification_report as a dictionary.
    """
    y_pred = model.predict(X_test)

    # --- Text report ---
    target_names = DISPLAY_LABELS
    report_text = classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0,
    )

    # --- Dict report (for programmatic access) ---
    report_dict = classification_report(
        y_test, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # --- Save to file ---
    report_path = METRICS_DIR / f"classification_report_{model_name.lower()}.txt"
    header = (
        f"Classification Report: {model_name}\n"
        f"{'=' * 55}\n"
        f"Project: Global Country Stability Intelligence System\n"
        f"Test samples: {len(y_test)}\n"
        f"{'=' * 55}\n\n"
    )

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(report_text)
            f.write(f"\n\nNote: 'Stable' = low risk, 'At-Risk' = high risk.\n")
            f.write(f"Macro F1 is the primary evaluation metric.\n")
        logger.info(f"Classification report saved: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save classification report: {e}")

    return report_dict


# ============================================================================
# 4. CONFUSION MATRIX PLOT
# ============================================================================

def plot_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> np.ndarray:
    """
    Generate and save a publication-quality confusion matrix heatmap.

    The confusion matrix is the MOST intuitive evaluation visualization:
    - Diagonal = correct predictions (want these HIGH)
    - Off-diagonal = misclassifications (want these LOW)

    A recruiter glancing at this plot instantly sees whether the model
    confuses Watch with At-Risk (a near-miss) or Stable with At-Risk
    (a dangerous error).

    Args:
        model: Trained pipeline.
        X_test: Test features.
        y_test: True labels.
        model_name: Name for plot title and file naming.

    Returns:
        np.ndarray: The raw confusion matrix.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=config.CONFUSION_MATRIX_FIGSIZE)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=config.CONFUSION_MATRIX_CMAP,
        xticklabels=DISPLAY_LABELS,
        yticklabels=DISPLAY_LABELS,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Confusion Matrix: {model_name}\n"
        f"(Test Set, n={len(y_test)})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout()

    # --- Save ---
    save_path = config.FIGURES_DIR / f"confusion_matrix_{model_name.lower()}.png"
    try:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix: {e}")
    finally:
        plt.close(fig)

    return cm


# ============================================================================
# 5. FEATURE IMPORTANCE EXTRACTION
# ============================================================================

def _get_feature_names_from_pipeline(
    model_pipeline: Any,
    fallback_names: Optional[List[str]] = None,
) -> List[str]:
    """
    Attempt to extract transformed feature names from a fitted Pipeline.

    The preprocessor (ColumnTransformer) may rename features after
    one-hot encoding or scaling.  This function tries multiple strategies
    to recover meaningful names.

    Args:
        model_pipeline: Fitted sklearn Pipeline.
        fallback_names: Feature names to use if extraction fails.

    Returns:
        List of feature name strings.
    """
    try:
        preprocessor = model_pipeline.named_steps.get("preprocessor")
        if preprocessor is not None:
            # get_feature_names_out() works on fitted ColumnTransformers
            names = preprocessor.get_feature_names_out()
            # Clean up sklearn prefixes like "numerical__" or "remainder__"
            cleaned = []
            for name in names:
                # Remove transformer name prefix
                if "__" in name:
                    name = name.split("__", 1)[-1]
                cleaned.append(name)
            return cleaned
    except Exception as e:
        logger.debug(f"Could not extract feature names from pipeline: {e}")

    if fallback_names is not None:
        return fallback_names

    # Last resort: generic feature indices
    try:
        classifier = model_pipeline.named_steps.get("classifier")
        if hasattr(classifier, "n_features_in_"):
            return [f"Feature_{i}" for i in range(classifier.n_features_in_)]
    except Exception:
        pass

    return [f"Feature_{i}" for i in range(20)]  # Arbitrary fallback


def extract_feature_importance(
    model_pipeline: Any,
    feature_names: Optional[List[str]] = None,
    model_name: str = "Model",
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extract feature importance from a trained Pipeline's classifier.

    Supports three types of classifiers:
        - Random Forest / XGBoost: .feature_importances_ (impurity/gain)
        - Logistic Regression: .coef_ (absolute mean across classes)

    For multi-class Logistic Regression, we take the mean of absolute
    coefficient values across all 3 classes — giving a single importance
    score per feature.

    Args:
        model_pipeline: Fitted sklearn Pipeline.
        feature_names: Optional list of feature names.  If None, attempts
                       to extract from the pipeline's preprocessor.
        model_name: Name for logging and file naming.
        top_n: Number of top features to return.

    Returns:
        pd.DataFrame: Columns ['feature', 'importance'], sorted descending.
        Returns empty DataFrame if extraction fails.
    """
    logger.info(f"Extracting feature importance for {model_name}...")

    # --- Get the classifier from the pipeline ---
    try:
        classifier = model_pipeline.named_steps.get("classifier")
        if classifier is None:
            # Maybe it's not a Pipeline — try using the model directly
            classifier = model_pipeline
    except AttributeError:
        classifier = model_pipeline

    # --- Get feature names ---
    resolved_names = _get_feature_names_from_pipeline(
        model_pipeline, fallback_names=feature_names
    )

    # --- Extract importances based on classifier type ---
    importances = None

    # Strategy 1: Tree-based models (RandomForest, XGBoost, LightGBM)
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        logger.debug(f"  Extracted feature_importances_ ({len(importances)} features)")

    # Strategy 2: Linear models (Logistic Regression)
    elif hasattr(classifier, "coef_"):
        # coef_ shape: (n_classes, n_features) for multi-class
        # Take mean of absolute values across classes
        coef = np.abs(classifier.coef_)
        if coef.ndim > 1:
            importances = coef.mean(axis=0)
        else:
            importances = np.abs(coef)
        logger.debug(f"  Extracted coef_ ({len(importances)} features)")

    else:
        logger.warning(
            f"  {model_name}: classifier type '{type(classifier).__name__}' "
            f"does not expose feature_importances_ or coef_. "
            f"Skipping feature importance."
        )
        return pd.DataFrame(columns=["feature", "importance"])

    # --- Align feature names with importances ---
    if len(resolved_names) != len(importances):
        logger.warning(
            f"  Feature name count ({len(resolved_names)}) does not match "
            f"importance count ({len(importances)}). Using generic names."
        )
        resolved_names = [f"Feature_{i}" for i in range(len(importances))]

    # --- Build DataFrame and sort ---
    importance_df = pd.DataFrame({
        "feature": resolved_names,
        "importance": importances,
    })
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df = importance_df.head(top_n).reset_index(drop=True)

    logger.info(
        f"  Top {top_n} features for {model_name}:\n"
        f"{importance_df.to_string(index=False)}"
    )

    return importance_df


# ============================================================================
# 6. FEATURE IMPORTANCE PLOT
# ============================================================================

def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    model_name: str = "Model",
    top_n: int = 15,
) -> None:
    """
    Save a horizontal bar chart of the top N most important features.

    A clean feature importance chart is one of the most impactful visuals
    in an ML portfolio.  Recruiters immediately see what drives the model's
    decisions — and whether those drivers make business sense.

    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' cols.
        model_name: Name for plot title and file naming.
        top_n: Max features to display.
    """
    if feature_importance_df.empty:
        logger.warning(f"No feature importance data for {model_name}. Skipping plot.")
        return

    plot_df = feature_importance_df.head(top_n).copy()
    # Reverse order so most important is at the top of horizontal bar chart
    plot_df = plot_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.45)))

    bars = ax.barh(
        plot_df["feature"],
        plot_df["importance"],
        color=sns.color_palette("viridis", n_colors=len(plot_df)),
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Top {len(plot_df)} Feature Importances: {model_name}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.tick_params(axis="y", labelsize=10)

    # Add value labels on bars
    for bar, val in zip(bars, plot_df["importance"]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
            color="#333333",
        )

    plt.tight_layout()

    # --- Save ---
    save_path = config.FIGURES_DIR / f"feature_importance_{model_name.lower()}.png"
    try:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance plot saved: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save feature importance plot: {e}")
    finally:
        plt.close(fig)


# ============================================================================
# 7. MODEL COMPARISON PLOT
# ============================================================================

def plot_model_comparison(results_df: pd.DataFrame) -> None:
    """
    Save a grouped bar chart comparing all models across key metrics.

    This single plot gives a complete picture for interviews: it shows
    which model wins on which metric, and whether the differences are
    marginal or substantial.

    Args:
        results_df: Output of evaluate_all_models().
    """
    if results_df.empty:
        logger.warning("No results to plot. Skipping model comparison chart.")
        return

    # Metrics to display (exclude 'model' column)
    metric_cols = [
        col for col in results_df.columns
        if col != "model" and results_df[col].dtype in [np.float64, np.float32]
    ]

    if not metric_cols:
        logger.warning("No numeric metrics found for comparison plot.")
        return

    plot_df = results_df.melt(
        id_vars="model",
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="model",
        palette="viridis",
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model Comparison Across All Metrics",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", title_fontsize=11, fontsize=10, loc="lower right")
    ax.tick_params(axis="x", rotation=25, labelsize=10)

    # Add value labels on each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()

    # --- Save ---
    try:
        fig.savefig(config.MODEL_COMPARISON_PATH, dpi=150, bbox_inches="tight")
        logger.info(f"Model comparison plot saved: {config.MODEL_COMPARISON_PATH}")
    except Exception as e:
        logger.error(f"Failed to save model comparison plot: {e}")
    finally:
        plt.close(fig)


# ============================================================================
# 8. BEST MODEL SELECTION
# ============================================================================

def select_best_model(
    results_df: pd.DataFrame,
) -> str:
    """
    Select the best model from the evaluation results using a business-
    aligned priority system.

    Selection Priority:
        1. Macro F1-Score        — PRIMARY: treats all classes equally
        2. At-Risk Recall        — CRITICAL: don't miss unstable countries
        3. Weighted F1-Score     — SECONDARY: overall balance
        4. Accuracy              — TERTIARY: sanity check only

    Why this order?
        A model with 95% accuracy but 20% At-Risk recall is useless for
        the core business question ("which countries are at risk?").
        Macro F1 ensures ALL classes are handled, and At-Risk recall
        is the tiebreaker that aligns with stakeholder priorities.

    Args:
        results_df: Output of evaluate_all_models().

    Returns:
        str: Name of the best model.
    """
    if results_df.empty:
        logger.error("Cannot select best model from empty results.")
        return "Unknown"

    # Sort by priority cascade: macro_f1 > at_risk_recall > weighted_f1 > accuracy
    sort_columns = []
    for col in ["macro_f1", "at_risk_recall", "weighted_f1", "accuracy"]:
        if col in results_df.columns:
            sort_columns.append(col)

    if sort_columns:
        sorted_df = results_df.sort_values(
            sort_columns, ascending=False
        ).reset_index(drop=True)
    else:
        sorted_df = results_df

    best_model_name = sorted_df.iloc[0]["model"]
    best_macro_f1 = sorted_df.iloc[0].get("macro_f1", 0)
    best_at_risk = sorted_df.iloc[0].get("at_risk_recall", 0)

    logger.info(
        f"Best model selected: {best_model_name} "
        f"(Macro F1={best_macro_f1:.4f}, At-Risk Recall={best_at_risk:.4f})"
    )

    return best_model_name


# ============================================================================
# 9. SAVE METRICS SUMMARY
# ============================================================================

def save_metrics_summary(results_df: pd.DataFrame) -> Path:
    """
    Save the model comparison table as a CSV for future reference
    and automated reporting.

    Args:
        results_df: Output of evaluate_all_models().

    Returns:
        Path: Location of the saved CSV file.
    """
    save_path = METRICS_DIR / "metrics_summary.csv"

    try:
        results_df.to_csv(save_path, index=False)
        logger.info(f"Metrics summary saved: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save metrics summary: {e}")
        return save_path


# ============================================================================
# 10. BUSINESS INTERPRETATION
# ============================================================================

def interpret_model_results(results_df: pd.DataFrame) -> str:
    """
    Generate a plain-English interpretation of the model comparison
    results, suitable for a non-technical stakeholder or interview.

    This is the "so what?" of the entire ML pipeline — translating
    numbers into actionable business insight.

    Args:
        results_df: Output of evaluate_all_models().

    Returns:
        str: Multi-paragraph interpretation text.
    """
    if results_df.empty:
        return "No model results available for interpretation."

    best_row = results_df.iloc[0]
    best_name = best_row["model"]
    best_f1 = best_row.get("macro_f1", 0)
    best_acc = best_row.get("accuracy", 0)
    best_at_risk = best_row.get("at_risk_recall", 0)
    best_precision = best_row.get("macro_precision", 0)

    n_models = len(results_df)
    worst_row = results_df.iloc[-1]
    worst_name = worst_row["model"]
    worst_f1 = worst_row.get("macro_f1", 0)

    # --- Build interpretation ---
    lines = [
        "=" * 65,
        "MODEL EVALUATION INTERPRETATION",
        "Global Country Stability Intelligence System",
        "=" * 65,
        "",
        "BEST MODEL SELECTED",
        "-" * 40,
        f"  Model       : {best_name}",
        f"  Macro F1    : {best_f1:.4f}",
        f"  Accuracy    : {best_acc:.4f}",
        f"  At-Risk Recall: {best_at_risk:.4f}",
        f"  Macro Precision: {best_precision:.4f}",
        "",
        "WHY THIS MODEL?",
        "-" * 40,
        f"  {best_name} achieved the highest Macro F1 score ({best_f1:.4f})",
        f"  among {n_models} evaluated models.",
        "",
        f"  The gap between {best_name} ({best_f1:.4f}) and the weakest",
        f"  model {worst_name} ({worst_f1:.4f}) is {best_f1 - worst_f1:.4f}.",
        "",
        "WHY MACRO F1 MATTERS",
        "-" * 40,
        "  Macro F1 is the unweighted average F1 across ALL risk classes.",
        "  Unlike accuracy, it ensures the model doesn't just predict the",
        "  majority class well while ignoring minority classes.",
        "",
        "  For a 3-class problem (Stable / Watch / At-Risk), Macro F1",
        "  penalizes a model equally for poor performance on ANY class.",
        "",
        "WHY AT-RISK RECALL MATTERS",
        "-" * 40,
        f"  The At-Risk recall is {best_at_risk:.4f}, meaning the model",
        f"  correctly identifies {best_at_risk * 100:.1f}% of countries that",
        "  are genuinely at risk of instability.",
        "",
        "  Missing an at-risk country (false negative) is far more costly",
        "  than a false alarm:",
        "    - Investors could lose capital in an unstable market",
        "    - Aid organizations could miss the countries that need help most",
        "    - Insurance companies could underprice sovereign risk",
        "",
        "PRECISION vs RECALL TRADEOFF",
        "-" * 40,
        f"  Macro Precision: {best_precision:.4f} (of predicted at-risk, how many truly are)",
        f"  Macro Recall   : {best_row.get('macro_recall', 0):.4f} (of truly at-risk, how many we caught)",
        "",
        "  In this application, RECALL is more important than precision.",
        "  A false alarm (predicting Watch instead of Stable) causes",
        "  extra caution.  A missed risk (predicting Stable instead of",
        "  At-Risk) causes real damage.",
        "",
        "RECOMMENDATION",
        "-" * 40,
        f"  Deploy {best_name} as the production model.",
        f"  Monitor At-Risk recall as the primary business KPI.",
        f"  Retrain quarterly as new country data becomes available.",
        "",
        "=" * 65,
    ]

    interpretation = "\n".join(lines)

    # --- Save to file ---
    save_path = METRICS_DIR / "model_interpretation.txt"
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(interpretation)
        logger.info(f"Model interpretation saved: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save interpretation: {e}")

    return interpretation


# ============================================================================
# 11. FULL EVALUATION ORCHESTRATOR
# ============================================================================

def run_evaluation_pipeline(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    End-to-end evaluation orchestrator: evaluate all models, generate all
    plots and reports, select the best model.

    Pipeline Flow:
        1. Evaluate all models on test set
        2. Generate classification reports for each model
        3. Plot confusion matrices for each model
        4. Extract and plot feature importances
        5. Plot model comparison chart
        6. Select the best model
        7. Save metrics summary CSV
        8. Generate business interpretation

    Args:
        trained_models: Dict of {model_name: trained_pipeline}.
        X_test: Test features.
        y_test: True test labels.
        feature_names: Optional feature name list for importance extraction.

    Returns:
        Dict with keys:
            - 'results_df':        Model comparison DataFrame
            - 'best_model_name':   Name of the best model
            - 'classification_reports': Dict of per-model reports
            - 'confusion_matrices':    Dict of per-model CM arrays
            - 'feature_importances':   Dict of per-model importance DataFrames
            - 'interpretation':    Business interpretation text
    """
    logger.info("=" * 60)
    logger.info("STARTING FULL EVALUATION PIPELINE")
    logger.info("=" * 60)

    # --- Step 1: Evaluate all models ---
    results_df = evaluate_all_models(trained_models, X_test, y_test)

    # --- Step 2-4: Per-model reports, plots, and feature importance ---
    classification_reports = {}
    confusion_matrices = {}
    feature_importances = {}

    for model_name, model in trained_models.items():
        try:
            # Classification report
            report = generate_classification_report(
                model, X_test, y_test, model_name
            )
            classification_reports[model_name] = report

            # Confusion matrix
            cm = plot_confusion_matrix(model, X_test, y_test, model_name)
            confusion_matrices[model_name] = cm

            # Feature importance
            importance_df = extract_feature_importance(
                model, feature_names=feature_names, model_name=model_name
            )
            feature_importances[model_name] = importance_df
            if not importance_df.empty:
                plot_feature_importance(importance_df, model_name)

        except Exception as e:
            logger.error(f"Report generation failed for {model_name}: {e}")
            continue

    # --- Step 5: Model comparison chart ---
    plot_model_comparison(results_df)

    # --- Step 6: Select best model ---
    best_model_name = select_best_model(results_df)

    # --- Step 7: Save metrics summary ---
    save_metrics_summary(results_df)

    # --- Step 8: Business interpretation ---
    interpretation = interpret_model_results(results_df)

    logger.info("=" * 60)
    logger.info("EVALUATION PIPELINE COMPLETE")
    logger.info("=" * 60)

    return {
        "results_df": results_df,
        "best_model_name": best_model_name,
        "classification_reports": classification_reports,
        "confusion_matrices": confusion_matrices,
        "feature_importances": feature_importances,
        "interpretation": interpretation,
    }


# ============================================================================
# 12. STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run this file directly to test the evaluation pipeline:
        python src/evaluation.py

    Requires:
        - data/processed/final_model_ready.csv
        - models/*.joblib (from model_training.py)
    """
    print("\n" + "=" * 60)
    print("  EVALUATION -- Standalone Test Run")
    print("=" * 60 + "\n")

    # --- Load data ---
    try:
        df = pd.read_csv(config.FINAL_DATASET_FILE)
        print(f"Loaded data: {df.shape}")
    except FileNotFoundError:
        print(f"Data not found. Run feature_engineering.py first.")
        sys.exit(1)

    # --- Separate features and target ---
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    # --- Train/test split (same seed as training) ---
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"Test set: {X_test.shape}")

    # --- Load trained models ---
    from src.model_training import load_model

    trained_models = {}
    for model_name in config.MODELS_TO_TRAIN:
        model_file = config.MODELS_DIR / f"{model_name.lower()}_pipeline.joblib"
        if model_file.exists():
            trained_models[model_name] = load_model(model_file)
            print(f"Loaded: {model_name}")
        else:
            print(f"Model file not found: {model_file}")

    if not trained_models:
        print("No models found. Run model_training.py first.")
        sys.exit(1)

    # --- Feature names ---
    feature_names = X.columns.tolist()

    # --- Run evaluation ---
    results = run_evaluation_pipeline(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )

    # --- Print summary ---
    print(f"\n{'-' * 50}")
    print(f"Best model: {results['best_model_name']}")
    print(f"{'-' * 50}")
    print("\nMetrics:")
    print(results["results_df"].to_string(index=False))
    print(f"\nReports saved to: {METRICS_DIR}")
    print(f"Figures saved to: {config.FIGURES_DIR}")
    print("\n[OK] Evaluation test complete.\n")
