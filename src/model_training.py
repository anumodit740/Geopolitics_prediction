"""
src/model_training.py — Model Definition, Training & Persistence
=================================================================

This module handles the full model lifecycle: initialization, pipeline
construction, training, cross-validation, comparison, and artifact saving.

Architecture:
    preprocessor (from data_preprocessing.py)
        |
        v
    sklearn.Pipeline([preprocessor, classifier])
        |
        v
    .fit(X_train, y_train)   <-- fitting happens HERE, not in preprocessing
        |
        v
    save to models/*.joblib

Why sklearn Pipelines?
    ~~~~~~~~~~~~~~~~~~~~~~
    Wrapping the preprocessor + classifier into a single Pipeline object
    guarantees that:
        1. Imputation medians are computed ONLY from X_train
        2. Scaling mean/std are computed ONLY from X_train
        3. The same transformations are applied consistently to X_test
        4. No data leakage is possible — the Pipeline enforces the order

Class Imbalance Strategy:
    ~~~~~~~~~~~~~~~~~~~~~~~~
    We use COST-SENSITIVE LEARNING (class_weight="balanced") rather than
    resampling techniques like SMOTE because:
        1. Our dataset is small (~189 rows).  SMOTE creates synthetic
           samples that can introduce noise and overfitting artifacts.
        2. Cost-sensitive learning adjusts the loss function to penalize
           misclassification of minority classes more heavily — no fake
           data is generated.
        3. This approach is more interpretable for consulting/interview
           settings: "we told the model that all classes matter equally."
        4. Our classes are actually balanced (63/63/63), so class_weight
           is a safety net rather than a necessity.

Model Selection Philosophy:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    We do NOT select the best model by accuracy alone.  Priority order:
        1. Macro F1-Score — treats all 3 risk classes equally important
        2. Weighted F1    — accounts for class size (secondary)
        3. High-Risk Recall — missing an At-Risk country is worse than
                              mislabeling a Stable one
        4. Accuracy        — least important; can be misleading with
                             imbalanced classes

Author : Anumol
Project: Global Country Stability Intelligence System
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

# --- XGBoost (required) ---
from xgboost import XGBClassifier

# --- LightGBM (optional — gracefully skip if not installed) ---
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# --- Project imports ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.data_preprocessing import build_preprocessing_pipeline  # noqa: E402

# ============================================================================
# MODULE-LEVEL LOGGER
# ============================================================================

logger = config.setup_logging()


# ============================================================================
# 1. CLASS WEIGHT COMPUTATION
# ============================================================================

def get_class_weights(y_train: pd.Series) -> Dict[int, float]:
    """
    Compute balanced class weights inversely proportional to class frequency.

    When classes are imbalanced, the model tends to favor the majority class.
    Balanced weights assign HIGHER importance to under-represented classes so
    the model pays equal attention to all risk tiers.

    Formula (sklearn's 'balanced'):
        weight_i = n_samples / (n_classes * n_samples_i)

    For our balanced dataset (63/63/63), all weights will be ~1.0.
    But this function adapts automatically if the distribution changes.

    Args:
        y_train: Training set target labels.

    Returns:
        Dict mapping class label (int) → weight (float).
        Example: {0: 1.0, 1: 1.0, 2: 1.0}
    """
    classes = np.array(sorted(y_train.unique()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    weight_dict = dict(zip(classes.astype(int), weights))

    logger.info(f"Computed class weights: {weight_dict}")
    return weight_dict


def get_sample_weights(y_train: pd.Series) -> np.ndarray:
    """
    Compute per-sample weights for models that don't accept class_weight
    (e.g., XGBoost's fit method via sample_weight parameter).

    Each sample receives the weight of its class.  This is functionally
    equivalent to class_weight="balanced" but applied at the sample level.

    Args:
        y_train: Training set target labels.

    Returns:
        np.ndarray: Weight for each training sample.
    """
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )
    logger.debug(
        f"Sample weights computed: min={sample_weights.min():.3f}, "
        f"max={sample_weights.max():.3f}, shape={sample_weights.shape}"
    )
    return sample_weights


# ============================================================================
# 2. MODEL INITIALIZATION
# ============================================================================

def initialize_models(y_train: pd.Series) -> Dict[str, Any]:
    """
    Create all classifier instances with production-ready hyperparameters
    from config.py.

    Each model is initialized but NOT trained yet.  Training happens inside
    the sklearn Pipeline to ensure proper data flow.

    Why these three models?
        - Logistic Regression: interpretable baseline with coefficients.
        - Random Forest: nonlinear ensemble that's robust to noise.
        - XGBoost: state-of-the-art gradient boosting for tabular data.

    Args:
        y_train: Training labels (used for class weight computation).

    Returns:
        Dict of {model_name: classifier_instance}.
    """
    models = {}

    # --- 1. Logistic Regression ---
    # Fully interpretable: each coefficient shows how a feature affects
    # the log-odds of each class.  The "glass box" benchmark.
    if "LogisticRegression" in config.MODELS_TO_TRAIN:
        models["LogisticRegression"] = LogisticRegression(
            **config.LOGISTIC_REGRESSION_PARAMS
        )
        logger.info("Initialized LogisticRegression (interpretable baseline)")

    # --- 2. Random Forest ---
    # Ensemble of decision trees with bootstrap aggregation.
    # Built-in feature importance via impurity decrease.
    if "RandomForest" in config.MODELS_TO_TRAIN:
        models["RandomForest"] = RandomForestClassifier(
            **config.RANDOM_FOREST_PARAMS
        )
        logger.info("Initialized RandomForestClassifier (nonlinear benchmark)")

    # --- 3. XGBoost ---
    # Gradient-boosted trees.  Our primary model for production.
    # XGBoost doesn't natively support class_weight dict; instead we
    # pass sample_weight during fit (handled in train_single_model).
    if "XGBoost" in config.MODELS_TO_TRAIN:
        models["XGBoost"] = XGBClassifier(**config.XGBOOST_PARAMS)
        logger.info("Initialized XGBClassifier (primary production model)")

    # --- 4. LightGBM (optional) ---
    # Only included if the library is installed and listed in config.
    if (
        LIGHTGBM_AVAILABLE
        and "LightGBM" in getattr(config, "MODELS_TO_TRAIN", [])
    ):
        lgbm_params = getattr(config, "LIGHTGBM_PARAMS", {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": config.RANDOM_STATE,
            "class_weight": "balanced",
            "verbose": -1,
            "n_jobs": -1,
        })
        models["LightGBM"] = LGBMClassifier(**lgbm_params)
        logger.info("Initialized LGBMClassifier (optional fast boosting model)")
    elif "LightGBM" in getattr(config, "MODELS_TO_TRAIN", []):
        logger.warning(
            "LightGBM listed in MODELS_TO_TRAIN but lightgbm package "
            "is not installed. Skipping."
        )

    logger.info(f"Total models initialized: {len(models)} -> {list(models.keys())}")
    return models


# ============================================================================
# 3. PIPELINE CONSTRUCTION
# ============================================================================

def build_model_pipeline(
    preprocessor: ColumnTransformer,
    model: Any,
) -> Pipeline:
    """
    Wrap a preprocessor and classifier into a single sklearn Pipeline.

    This is the CORE anti-leakage mechanism.  When pipeline.fit(X_train)
    is called, the preprocessor learns statistics (median, mean, std,
    category mappings) from X_train ONLY, and the classifier trains on
    the transformed features.  When pipeline.predict(X_test) is called,
    the preprocessor applies the SAME learned transformations — no test
    data leaks into the preprocessing.

    Pipeline flow:
        X_train -> [Imputer -> Scaler -> Encoder] -> [Classifier] -> y_pred
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^
                   preprocessor (fit on train)       model (fit on train)

    Args:
        preprocessor: Unfitted ColumnTransformer from data_preprocessing.py.
        model: Unfitted classifier instance (LogisticRegression, etc.).

    Returns:
        sklearn.Pipeline: Unfitted pipeline ready for .fit().
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    model_name = type(model).__name__
    logger.debug(f"Built pipeline: preprocessor -> {model_name}")

    return pipeline


# ============================================================================
# 4. SINGLE MODEL TRAINING
# ============================================================================

def train_single_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train a single model pipeline on the training data.

    Special handling for XGBoost:
        XGBClassifier doesn't accept class_weight as a constructor parameter
        the way sklearn models do.  Instead, we compute per-sample weights
        and pass them via the `classifier__sample_weight` argument.  The
        'classifier__' prefix routes the parameter to the correct pipeline
        step.

    Args:
        model_name: Human-readable name for logging.
        pipeline: Unfitted sklearn Pipeline (preprocessor + classifier).
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Pipeline: FITTED pipeline (preprocessor + classifier both trained).

    Raises:
        Exception: Logged and re-raised if training fails.
    """
    logger.info(f"Training {model_name}...")
    start_time = time.time()

    try:
        # XGBoost and LightGBM: pass sample weights for class balance
        # sklearn models already have class_weight="balanced" set
        if model_name in ("XGBoost", "LightGBM"):
            sample_weights = get_sample_weights(y_train)
            pipeline.fit(
                X_train,
                y_train,
                classifier__sample_weight=sample_weights,
            )
        else:
            pipeline.fit(X_train, y_train)

        elapsed = time.time() - start_time
        logger.info(
            f"  {model_name} trained successfully in {elapsed:.2f}s"
        )
        return pipeline

    except Exception as e:
        logger.error(f"  {model_name} training FAILED: {e}")
        raise


# ============================================================================
# 5. TRAIN ALL MODELS
# ============================================================================

def train_all_models(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Pipeline]:
    """
    Initialize, build, and train all configured models.

    This is the main entry point for the training loop.  It returns a
    dictionary of trained pipelines, ready for evaluation.

    Args:
        preprocessor: Unfitted ColumnTransformer.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Dict of {model_name: trained_pipeline}.
        Models that fail training are excluded with a warning.
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # --- Initialize all classifiers ---
    models = initialize_models(y_train)

    # --- Build and train each pipeline ---
    trained_models = {}

    for model_name, model in models.items():
        try:
            # IMPORTANT: Each model gets its OWN copy of the preprocessor.
            # If they shared the same object, fitting model A would mutate
            # the preprocessor that model B also depends on.  clone() is not
            # needed because Pipeline.fit() refits from scratch.
            from sklearn.base import clone
            fresh_preprocessor = clone(preprocessor)

            pipeline = build_model_pipeline(fresh_preprocessor, model)
            trained_pipeline = train_single_model(
                model_name, pipeline, X_train, y_train
            )
            trained_models[model_name] = trained_pipeline

        except Exception as e:
            logger.warning(
                f"Skipping {model_name} due to training error: {e}"
            )
            continue

    logger.info(
        f"Training complete: {len(trained_models)}/{len(models)} "
        f"models trained successfully"
    )
    logger.info("=" * 60)

    return trained_models


# ============================================================================
# 6. CROSS-VALIDATION
# ============================================================================

def perform_cross_validation(
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int = config.N_CV_FOLDS,
    model_name: str = "Model",
) -> Dict[str, np.ndarray]:
    """
    Perform Stratified K-Fold cross-validation and return scores.

    Why Stratified K-Fold?
        Standard K-Fold can create folds where an entire class is missing
        (likely with only ~63 samples per class).  Stratified K-Fold
        maintains the class distribution in every fold — each fold has
        roughly 33% Stable, 33% Watch, 33% At-Risk.

    Metrics scored:
        - accuracy:      overall correctness (secondary metric)
        - f1_macro:      unweighted mean F1 across classes (PRIMARY)
        - f1_weighted:   class-size-weighted F1 (secondary)
        - recall_macro:  unweighted mean recall (important for At-Risk)

    Why macro F1 is primary:
        Macro F1 treats all 3 classes equally.  If the model is great at
        predicting Stable but terrible at At-Risk, macro F1 will drop.
        Accuracy wouldn't catch this if Stable is the majority class.

    Args:
        model_pipeline: Trained or untrained Pipeline (will be re-fitted
                        in each fold).
        X_train: Full training features.
        y_train: Full training labels.
        n_folds: Number of cross-validation folds.
        model_name: Name for logging purposes.

    Returns:
        Dict of {metric_name: array of per-fold scores}.
    """
    logger.info(f"Cross-validating {model_name} with {n_folds}-fold stratified CV...")

    # --- Define CV strategy ---
    cv_strategy = StratifiedKFold(
        n_splits=n_folds,
        shuffle=config.SHUFFLE,
        random_state=config.RANDOM_STATE,
    )

    # --- Scoring metrics ---
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "recall_macro": "recall_macro",
    }

    # --- Handle sample weights for XGBoost/LightGBM during CV ---
    # cross_validate re-fits the pipeline in each fold, so we need to
    # pass fit_params if the classifier needs sample_weight.
    classifier = model_pipeline.named_steps.get("classifier")
    classifier_name = type(classifier).__name__ if classifier else ""

    fit_params = {}
    if classifier_name in ("XGBClassifier", "LGBMClassifier"):
        fit_params["classifier__sample_weight"] = get_sample_weights(y_train)

    # --- Run cross-validation ---
    try:
        # sklearn >= 1.8 renamed 'fit_params' to 'params'.
        # We try the new API first, then fall back to the old one.
        import inspect
        cv_sig = inspect.signature(cross_validate)
        cv_kwargs = {
            "estimator": model_pipeline,
            "X": X_train,
            "y": y_train,
            "cv": cv_strategy,
            "scoring": scoring,
            "return_train_score": False,
            "n_jobs": -1,
            "error_score": "raise",
        }

        if "params" in cv_sig.parameters:
            # sklearn >= 1.8: use 'params'
            cv_kwargs["params"] = fit_params if fit_params else None
        elif "fit_params" in cv_sig.parameters:
            # sklearn < 1.8: use 'fit_params'
            cv_kwargs["fit_params"] = fit_params if fit_params else None
        # else: neither param exists — skip extra params entirely

        cv_results = cross_validate(**cv_kwargs)

        # --- Log results ---
        logger.info(f"  {model_name} CV Results ({n_folds}-fold):")
        for metric_name in scoring:
            key = f"test_{metric_name}"
            scores = cv_results[key]
            mean_score = scores.mean()
            std_score = scores.std()
            logger.info(
                f"    {metric_name:20s}: {mean_score:.4f} (+/- {std_score:.4f})"
            )

        return cv_results

    except Exception as e:
        logger.error(f"  Cross-validation failed for {model_name}: {e}")
        raise


def cross_validate_all_models(
    trained_models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run cross-validation on all trained model pipelines and return results.

    Args:
        trained_models: Dict of {model_name: trained_pipeline}.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Dict of {model_name: {metric_name: scores_array}}.
    """
    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION FOR ALL MODELS")
    logger.info("=" * 60)

    all_cv_results = {}

    for model_name, pipeline in trained_models.items():
        try:
            cv_results = perform_cross_validation(
                model_pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                model_name=model_name,
            )
            all_cv_results[model_name] = cv_results
        except Exception as e:
            logger.warning(f"CV failed for {model_name}: {e}")
            continue

    return all_cv_results


# ============================================================================
# 7. MODEL COMPARISON & SELECTION
# ============================================================================

def compare_models(
    cv_results: Dict[str, Dict[str, np.ndarray]],
) -> pd.DataFrame:
    """
    Build a comparison table of all models across CV metrics.

    The table is sorted by the primary metric (config.PRIMARY_METRIC)
    so the best model appears at the top.

    Args:
        cv_results: Output of cross_validate_all_models().

    Returns:
        pd.DataFrame: Comparison table with columns per metric.
    """
    rows = []

    for model_name, results in cv_results.items():
        row = {"model": model_name}
        for metric_key in ["test_accuracy", "test_f1_macro", "test_f1_weighted", "test_recall_macro"]:
            if metric_key in results:
                scores = results[metric_key]
                metric_clean = metric_key.replace("test_", "")
                row[f"{metric_clean}_mean"] = scores.mean()
                row[f"{metric_clean}_std"] = scores.std()
        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    # --- Sort by primary metric (macro F1 by default) ---
    sort_col = f"{config.PRIMARY_METRIC.replace('macro_', 'f1_macro_')}_mean"
    if sort_col not in comparison_df.columns:
        # Fallback: try direct match
        sort_col = "f1_macro_mean"

    if sort_col in comparison_df.columns:
        comparison_df = comparison_df.sort_values(sort_col, ascending=False)
    comparison_df = comparison_df.reset_index(drop=True)

    logger.info(f"Model comparison table:\n{comparison_df.to_string(index=False)}")

    return comparison_df


def select_best_model(
    trained_models: Dict[str, Pipeline],
    cv_results: Dict[str, Dict[str, np.ndarray]],
) -> Tuple[str, Pipeline]:
    """
    Select the best model based on cross-validation macro F1-score.

    Selection Logic:
        1. Primary:   Highest mean Macro F1 across CV folds
        2. Tiebreak:  Highest mean Weighted F1
        3. Tiebreak:  Lowest standard deviation (most stable model)

    Why NOT accuracy?
        Accuracy counts all correct predictions equally.  If 60% of
        countries are Stable and the model predicts Stable for everyone,
        accuracy = 60% — but we've missed ALL Watch and At-Risk countries.
        Macro F1 catches this because recall for those classes would be 0%.

    Args:
        trained_models: Dict of {model_name: trained_pipeline}.
        cv_results: Dict of {model_name: {metric: scores}}.

    Returns:
        Tuple of (best_model_name, best_trained_pipeline).
    """
    best_name = None
    best_score = -1.0
    best_weighted_f1 = -1.0
    best_std = float("inf")

    for model_name, results in cv_results.items():
        if "test_f1_macro" not in results:
            continue

        macro_f1_mean = results["test_f1_macro"].mean()
        macro_f1_std = results["test_f1_macro"].std()
        weighted_f1_mean = results.get("test_f1_weighted", np.array([0])).mean()

        # Selection: best macro F1, then weighted F1, then lowest variance
        is_better = (
            macro_f1_mean > best_score
            or (macro_f1_mean == best_score and weighted_f1_mean > best_weighted_f1)
            or (macro_f1_mean == best_score and weighted_f1_mean == best_weighted_f1 and macro_f1_std < best_std)
        )

        if is_better:
            best_name = model_name
            best_score = macro_f1_mean
            best_weighted_f1 = weighted_f1_mean
            best_std = macro_f1_std

    if best_name is None:
        logger.error("No valid CV results found. Cannot select best model.")
        # Fallback: return the first trained model
        best_name = list(trained_models.keys())[0]
        logger.warning(f"Falling back to first model: {best_name}")

    logger.info(
        f"Best model selected: {best_name} "
        f"(Macro F1 = {best_score:.4f} +/- {best_std:.4f})"
    )

    return best_name, trained_models[best_name]


# ============================================================================
# 8. MODEL PERSISTENCE (SAVE / LOAD)
# ============================================================================

def save_model(
    model: Any,
    model_path: Optional[Path] = None,
    model_name: str = "model",
) -> Path:
    """
    Serialize a trained model (or pipeline) to disk using joblib.

    Why joblib over pickle?
        joblib is optimized for objects containing large numpy arrays
        (which sklearn models do).  It's faster and produces smaller
        files than standard pickle.

    Args:
        model: Trained sklearn Pipeline or model object.
        model_path: Full path for the saved file.  Defaults to
                    config.BEST_MODEL_PATH.
        model_name: Name for logging.

    Returns:
        Path: The path where the model was saved.
    """
    if model_path is None:
        model_path = config.BEST_MODEL_PATH

    model_path = Path(model_path)

    # Ensure the directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(model, model_path)
        logger.info(f"Model '{model_name}' saved to: {model_path}")
        return model_path

    except Exception as e:
        logger.error(f"Failed to save model '{model_name}': {e}")
        raise


def load_model(model_path: Optional[Path] = None) -> Any:
    """
    Load a previously saved model from disk.

    Args:
        model_path: Path to the .joblib file.  Defaults to
                    config.BEST_MODEL_PATH.

    Returns:
        The deserialized model/pipeline object.

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    if model_path is None:
        model_path = config.BEST_MODEL_PATH

    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(
            f"No saved model at: {model_path}\n"
            f"Run the training pipeline first."
        )

    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from: {model_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise


def save_training_artifacts(
    trained_models: Dict[str, Pipeline],
    best_model_name: str,
    feature_list: List[str],
) -> None:
    """
    Save all training artifacts: best model, all models, and feature list.

    Saving the feature list ensures that inference uses the exact same
    columns in the exact same order as training — preventing silent
    feature mismatch bugs.

    Args:
        trained_models: Dict of all trained pipelines.
        best_model_name: Key of the best model in the dict.
        feature_list: Ordered list of feature column names used for training.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Save the best model as the primary artifact ---
    save_model(
        trained_models[best_model_name],
        config.BEST_MODEL_PATH,
        model_name=best_model_name,
    )

    # --- Save all trained models individually ---
    for name, pipeline in trained_models.items():
        individual_path = config.MODELS_DIR / f"{name.lower()}_pipeline.joblib"
        save_model(pipeline, individual_path, model_name=name)

    # --- Save the feature list for inference validation ---
    joblib.dump(feature_list, config.FEATURE_LIST_PATH)
    logger.info(f"Feature list ({len(feature_list)} features) saved to: {config.FEATURE_LIST_PATH}")


# ============================================================================
# 9. FULL TRAINING ORCHESTRATOR
# ============================================================================

def run_training_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numerical_features: List[str],
    categorical_features: List[str],
) -> Dict[str, Any]:
    """
    End-to-end training orchestrator: build preprocessor, train all models,
    cross-validate, select the best, and save artifacts.

    This is the main entry point called by main.py.

    Pipeline Flow:
        1. Build unfitted preprocessor (ColumnTransformer)
        2. Initialize and train all models inside sklearn Pipelines
        3. Cross-validate every model with Stratified K-Fold
        4. Compare models on Macro F1 score
        5. Select the best model
        6. Save all artifacts to models/

    Args:
        X_train: Training features (pd.DataFrame).
        y_train: Training labels (pd.Series).
        numerical_features: List of numeric column names for the preprocessor.
        categorical_features: List of categorical column names.

    Returns:
        Dict with keys:
            - 'trained_models':  Dict of all trained pipelines
            - 'best_model_name': Name of the best model
            - 'best_model':      Best trained pipeline
            - 'cv_results':      Cross-validation results for all models
            - 'comparison_df':   Model comparison DataFrame
    """
    logger.info("=" * 60)
    logger.info("FULL TRAINING PIPELINE")
    logger.info("=" * 60)
    overall_start = time.time()

    # --- Step 1: Build preprocessor ---
    preprocessor = build_preprocessing_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # --- Step 2: Train all models ---
    trained_models = train_all_models(preprocessor, X_train, y_train)

    if not trained_models:
        logger.error("No models were trained successfully. Aborting.")
        raise RuntimeError("All model training attempts failed.")

    # --- Step 3: Cross-validate all models ---
    cv_results = cross_validate_all_models(trained_models, X_train, y_train)

    # --- Step 4: Compare models ---
    comparison_df = compare_models(cv_results)

    # --- Step 5: Select the best model ---
    best_model_name, best_model = select_best_model(trained_models, cv_results)

    # --- Step 6: Save all artifacts ---
    feature_list = numerical_features + categorical_features
    save_training_artifacts(trained_models, best_model_name, feature_list)

    total_time = time.time() - overall_start
    logger.info(f"Full training pipeline completed in {total_time:.2f}s")
    logger.info("=" * 60)

    return {
        "trained_models": trained_models,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "cv_results": cv_results,
        "comparison_df": comparison_df,
    }


# ============================================================================
# 10. STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run this file directly to test the training pipeline:
        python src/model_training.py

    Requires data/processed/final_model_ready.csv to exist
    (generated by feature_engineering.py).
    """
    print("\n" + "=" * 60)
    print("  MODEL TRAINING -- Standalone Test Run")
    print("=" * 60 + "\n")

    # --- Load model-ready data ---
    try:
        df = pd.read_csv(config.FINAL_DATASET_FILE)
        print(f"Loaded model-ready data: {df.shape}")
    except FileNotFoundError:
        print(
            f"Model-ready data not found at {config.FINAL_DATASET_FILE}.\n"
            f"Run 'python src/feature_engineering.py' first."
        )
        sys.exit(1)

    # --- Validate target column ---
    if config.TARGET_COLUMN not in df.columns:
        print(f"Target column '{config.TARGET_COLUMN}' not in data. Aborting.")
        sys.exit(1)

    # --- Separate features and target ---
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Target distribution: {y.value_counts().sort_index().to_dict()}")

    # --- Identify feature types ---
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Numerical features : {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # --- Train/test split for standalone run ---
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    # --- Run full training pipeline ---
    results = run_training_pipeline(
        X_train=X_train,
        y_train=y_train,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    # --- Print summary ---
    print(f"\n{'-' * 50}")
    print(f"Models trained: {list(results['trained_models'].keys())}")
    print(f"Best model    : {results['best_model_name']}")
    print(f"{'-' * 50}")

    print("\nModel Comparison:")
    print(results["comparison_df"].to_string(index=False))

    # --- Quick test prediction on test set ---
    best_pipeline = results["best_model"]
    test_preds = best_pipeline.predict(X_test)
    from sklearn.metrics import f1_score, accuracy_score
    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    print(f"\nTest set (holdout) performance of {results['best_model_name']}:")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  Macro F1 : {test_f1:.4f}")

    print(f"\nAll models saved to: {config.MODELS_DIR}")
    print("\n[OK] Model training test complete.\n")
