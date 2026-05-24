"""
main.py — End-to-End ML Pipeline Orchestrator
===============================================

This is the single entry point that runs the entire Global Country
Stability Intelligence System pipeline from raw CSV to saved model
and evaluation reports.

Usage:
    python main.py

Pipeline Flow:
    ┌──────────────────────────────────────────────────────────────┐
    │  1. Setup         Create output directories                  │
    │  2. Load          Read raw world-data-2023.csv               │
    │  3. Clean         Standardize columns, fix numeric strings   │
    │  4. Engineer      Build 24 domain features across 5 pillars  │
    │  5. Target        Create stability_score → stability_label   │
    │  6. Anti-Leak     Drop target-construction columns           │
    │  7. Split         80/20 stratified train/test                │
    │  8. Train         LR + RF + XGBoost (inside sklearn Pipes)   │
    │  9. Evaluate      Metrics, confusion matrices, importance    │
    │ 10. Save          Best model, reports, interpretation        │
    └──────────────────────────────────────────────────────────────┘

Data Leakage Prevention:
    ~~~~~~~~~~~~~~~~~~~~~~~~
    - stability_score is derived from Life_Expectancy, Infant_Mortality,
      Maternal_Mortality, and Physicians_per_1000.
    - These columns + stability_score are DROPPED before modeling.
    - The preprocessing pipeline is built UNFITTED and only fitted
      inside the sklearn Pipeline on X_train during model training.
    - Test data is NEVER seen during fit.

Author : Anumol
Project: Global Country Stability Intelligence System
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
import time
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# --- Project modules ---------------------------------------------------------
import config
from src.data_preprocessing import (
    load_raw_data,
    drop_unusable_data,
    clean_column_names,
    clean_numeric_values,
    create_missing_flags,
    analyze_missing_values,
    identify_feature_types,
    create_train_test_split,
)
from src.feature_engineering import (
    engineer_features,
    get_leaky_columns,
)
from src.model_training import (
    run_training_pipeline,
    save_model,
)
from src.evaluation import (
    run_evaluation_pipeline,
)

# ============================================================================
# LOGGER
# ============================================================================

logger = config.setup_logging()


# ============================================================================
# 1. DIRECTORY SETUP
# ============================================================================

def setup_directories() -> None:
    """
    Create all required output directories if they don't already exist.

    Why upfront?
        If we wait until a module tries to write and the directory is
        missing, we get a cryptic FileNotFoundError deep in the pipeline.
        Creating everything upfront is a production best practice.
    """
    directories = [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.FIGURES_DIR,
        config.REPORTS_DIR / "metrics",
        config.LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directories verified ({len(directories)} dirs)")


# ============================================================================
# 2. FULL PIPELINE ORCHESTRATOR
# ============================================================================

def run_pipeline() -> dict:
    """
    Execute the complete ML pipeline end-to-end.

    Returns a dictionary containing all pipeline outputs so that
    downstream consumers (app.py, notebooks) can access any artifact
    without re-running the pipeline.

    Returns:
        dict with keys:
            - 'cleaned_df':       Cleaned DataFrame before feature engineering
            - 'featured_df':      DataFrame after feature engineering
            - 'model_ready_df':   Leakage-free DataFrame used for modeling
            - 'X_train', 'X_test', 'y_train', 'y_test': Split data
            - 'training_results': Output of run_training_pipeline()
            - 'eval_results':     Output of run_evaluation_pipeline()
            - 'best_model_name':  Name of the winning model
    """

    pipeline_start = time.time()

    # ================================================================
    # STEP 1: LOAD RAW DATASET
    # ================================================================
    logger.info("=" * 65)
    logger.info("  STEP 1/10: LOADING RAW DATASET")
    logger.info("=" * 65)

    df_raw = load_raw_data()
    logger.info(f"Raw dataset: {df_raw.shape[0]} countries, {df_raw.shape[1]} columns")

    # ================================================================
    # STEP 2: CLEAN RAW DATA
    # ================================================================
    # Drop identifier columns (Abbreviation, Currency-Code, etc.) and
    # near-empty country rows (Vatican City, etc.) that would add noise.
    logger.info("=" * 65)
    logger.info("  STEP 2/10: CLEANING RAW DATA")
    logger.info("=" * 65)

    df = drop_unusable_data(df_raw)
    df = clean_column_names(df)
    df = clean_numeric_values(df)

    # Create binary flags for high-missing columns BEFORE imputation.
    # These flags capture "was this value originally missing?" — a signal
    # that imputation would destroy.
    df = create_missing_flags(df)

    # Diagnostic only — does not modify data
    missing_summary = analyze_missing_values(df)

    logger.info(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Save cleaned intermediate data for debugging / EDA
    df.to_csv(config.CLEANED_DATA_FILE, index=False)
    logger.info(f"Cleaned data saved: {config.CLEANED_DATA_FILE}")

    # ================================================================
    # STEP 3: FEATURE ENGINEERING
    # ================================================================
    # Build 24 domain-specific features across 5 pillars:
    #   Economic, Healthcare, Education/Workforce, Demographic, Environment
    logger.info("=" * 65)
    logger.info("  STEP 3/10: ENGINEERING DOMAIN FEATURES")
    logger.info("=" * 65)

    # engineer_features with drop_leaky=False so we can inspect the full
    # DataFrame before removing target-construction columns.
    df_featured = engineer_features(df, create_target=True, drop_leaky=False)

    logger.info(f"Featured dataset: {df_featured.shape[0]} rows, {df_featured.shape[1]} columns")

    # Save the full featured dataset (for EDA and dashboards)
    df_featured.to_csv(config.FEATURE_ENGINEERED_FILE, index=False)
    logger.info(f"Featured data saved: {config.FEATURE_ENGINEERED_FILE}")

    # ================================================================
    # STEP 4: VERIFY TARGET COLUMN
    # ================================================================
    logger.info("=" * 65)
    logger.info("  STEP 4/10: VERIFYING TARGET COLUMN")
    logger.info("=" * 65)

    if config.TARGET_COLUMN not in df_featured.columns:
        raise ValueError(
            f"Target column '{config.TARGET_COLUMN}' was not created. "
            f"Check feature_engineering.py -> create_risk_target()."
        )

    target_dist = df_featured[config.TARGET_COLUMN].value_counts().sort_index()
    logger.info(f"Target column '{config.TARGET_COLUMN}' distribution:")
    for cls_id, count in target_dist.items():
        label = config.CLASS_LABELS.get(cls_id, f"Class {cls_id}")
        pct = count / len(df_featured) * 100
        logger.info(f"  {cls_id} ({label}): {count} countries ({pct:.1f}%)")

    # ================================================================
    # STEP 5: REMOVE LEAKAGE COLUMNS
    # ================================================================
    # stability_score was used to CREATE stability_label.
    # If the model sees stability_score as a feature, it's essentially
    # cheating — doing a threshold lookup instead of learning patterns.
    #
    # Similarly, Life_Expectancy, Infant_Mortality, Maternal_Mortality,
    # and Physicians_per_1000 are the raw inputs to stability_score.
    # They must also be excluded, or the model can reconstruct the score.
    #
    # Healthcare composite features (Healthcare_Stability_Score, etc.)
    # derived from these columns are also leaky.
    #
    # The "Country" column is an identifier, not a feature.
    logger.info("=" * 65)
    logger.info("  STEP 5/10: REMOVING LEAKAGE COLUMNS")
    logger.info("=" * 65)

    leaky_cols = get_leaky_columns()
    additional_drops = ["Country"]
    all_drops = leaky_cols + additional_drops

    cols_to_drop = [col for col in all_drops if col in df_featured.columns]
    df_model_ready = df_featured.drop(columns=cols_to_drop)

    logger.info(f"Dropped {len(cols_to_drop)} leaky/non-feature columns:")
    for col in cols_to_drop:
        logger.info(f"  - {col}")
    logger.info(f"Model-ready dataset: {df_model_ready.shape}")

    # Save model-ready dataset
    df_model_ready.to_csv(config.FINAL_DATASET_FILE, index=False)
    logger.info(f"Model-ready data saved: {config.FINAL_DATASET_FILE}")

    # ================================================================
    # STEP 6: TRAIN/TEST SPLIT
    # ================================================================
    # Stratified split ensures each risk class is proportionally
    # represented in both train and test sets.
    #
    # DATA LEAKAGE RULE: This split happens BEFORE any preprocessing
    # fitting. The test set must remain completely unseen.
    logger.info("=" * 65)
    logger.info("  STEP 6/10: SPLITTING TRAIN / TEST")
    logger.info("=" * 65)

    X_train, X_test, y_train, y_test = create_train_test_split(
        df_model_ready, target_column=config.TARGET_COLUMN
    )

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ================================================================
    # STEP 7: IDENTIFY FEATURE TYPES
    # ================================================================
    logger.info("=" * 65)
    logger.info("  STEP 7/10: IDENTIFYING FEATURE TYPES")
    logger.info("=" * 65)

    # Drop target from X before identifying types
    # (create_train_test_split already excludes it from X)
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    logger.info(f"Numerical features  : {len(numerical_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")

    # ================================================================
    # STEP 8: TRAIN ALL MODELS
    # ================================================================
    # Each model is wrapped in an sklearn Pipeline:
    #   Pipeline([preprocessor, classifier])
    #
    # The preprocessor is built UNFITTED and is fitted inside the
    # Pipeline on X_train only. This guarantees:
    #   - Imputation medians come from training data
    #   - Scaling mean/std come from training data
    #   - One-hot categories come from training data
    #   - ZERO data leakage
    logger.info("=" * 65)
    logger.info("  STEP 8/10: TRAINING ALL MODELS")
    logger.info("=" * 65)

    training_results = run_training_pipeline(
        X_train=X_train,
        y_train=y_train,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    trained_models = training_results["trained_models"]
    logger.info(f"Models trained: {list(trained_models.keys())}")
    logger.info(f"Best model (CV): {training_results['best_model_name']}")

    # ================================================================
    # STEP 9: EVALUATE ALL MODELS ON TEST SET
    # ================================================================
    logger.info("=" * 65)
    logger.info("  STEP 9/10: EVALUATING ON HELD-OUT TEST SET")
    logger.info("=" * 65)

    eval_results = run_evaluation_pipeline(
        trained_models=trained_models,
        X_test=X_test,
        y_test=y_test,
        feature_names=numerical_features + categorical_features,
    )

    logger.info(f"Best model (test): {eval_results['best_model_name']}")

    # ================================================================
    # STEP 10: FINAL SUMMARY
    # ================================================================
    logger.info("=" * 65)
    logger.info("  STEP 10/10: PIPELINE COMPLETE")
    logger.info("=" * 65)

    total_time = time.time() - pipeline_start

    # --- Print final summary ---
    results_df = eval_results["results_df"]
    best_name = eval_results["best_model_name"]
    best_row = results_df[results_df["model"] == best_name].iloc[0]

    logger.info(f"")
    logger.info(f"  Pipeline completed in {total_time:.2f} seconds")
    logger.info(f"  Dataset        : {df_raw.shape[0]} countries -> {df_model_ready.shape[0]} (after cleaning)")
    logger.info(f"  Features       : {len(numerical_features)} numerical + {len(categorical_features)} categorical")
    logger.info(f"  Models trained : {len(trained_models)}")
    logger.info(f"  Best model     : {best_name}")
    logger.info(f"  Macro F1       : {best_row.get('macro_f1', 0):.4f}")
    logger.info(f"  At-Risk Recall : {best_row.get('at_risk_recall', 0):.4f}")
    logger.info(f"  Accuracy       : {best_row.get('accuracy', 0):.4f}")
    logger.info(f"")
    logger.info(f"  Saved artifacts:")
    logger.info(f"    Best model     : {config.BEST_MODEL_PATH}")
    logger.info(f"    Feature list   : {config.FEATURE_LIST_PATH}")
    logger.info(f"    Metrics CSV    : {config.REPORTS_DIR / 'metrics' / 'metrics_summary.csv'}")
    logger.info(f"    Interpretation : {config.REPORTS_DIR / 'metrics' / 'model_interpretation.txt'}")
    logger.info(f"    Figures        : {config.FIGURES_DIR}")
    logger.info(f"")
    logger.info("=" * 65)

    return {
        "cleaned_df": df,
        "featured_df": df_featured,
        "model_ready_df": df_model_ready,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "training_results": training_results,
        "eval_results": eval_results,
        "best_model_name": eval_results["best_model_name"],
    }


# ============================================================================
# 3. MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Application entry point.

    Sets up directories, runs the full pipeline, and handles any
    unrecoverable errors with full traceback logging.
    """
    print("\n" + "=" * 65)
    print("  Global Country Stability Intelligence System")
    print("  Full ML Pipeline")
    print("=" * 65 + "\n")

    try:
        # --- Create output directories ---
        setup_directories()

        # --- Run the full pipeline ---
        results = run_pipeline()

        # --- Final console output ---
        print("\n" + "=" * 65)
        print("  PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 65)
        print(f"  Best Model     : {results['best_model_name']}")

        best_row = results["eval_results"]["results_df"]
        best_row = best_row[best_row["model"] == results["best_model_name"]].iloc[0]
        print(f"  Macro F1       : {best_row.get('macro_f1', 0):.4f}")
        print(f"  At-Risk Recall : {best_row.get('at_risk_recall', 0):.4f}")
        print(f"  Accuracy       : {best_row.get('accuracy', 0):.4f}")
        print(f"  Model saved to : {config.BEST_MODEL_PATH}")
        print("=" * 65 + "\n")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error(
            "Hint: Ensure 'world-data-2023.csv' is in data/raw/ "
            "before running the pipeline."
        )
        raise

    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        logger.error("Hint: Check that all required columns exist in the dataset.")
        raise

    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        logger.error(traceback.format_exc())
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
