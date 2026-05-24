"""
src/data_preprocessing.py — Data Loading, Cleaning & Pipeline Construction
===========================================================================

This module is the FIRST step in the ML pipeline. It takes the raw, messy
CSV and produces clean numeric DataFrames ready for feature engineering and
model training.

Responsibilities:
    1. Load raw CSV with validation
    2. Drop useless columns and near-empty country rows
    3. Convert string-encoded numerics ($, %, commas) → float
    4. Rename columns to clean, consistent names
    5. Identify numerical vs categorical feature types
    6. Create a stratified train/test split
    7. Build an UNFITTED sklearn preprocessing pipeline
    8. (Optional) Cap outliers via IQR

Data Leakage Prevention:
    ─────────────────────
    The preprocessing pipeline returned by this module is NEVER fitted here.
    Fitting must happen downstream in model_training.py using ONLY training
    data. This guarantees that no information from the test set leaks into
    imputation statistics (median), scaling parameters (mean/std), or
    encoding mappings.

Author : Anumol
Project: Global Country Stability Intelligence System
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Project imports (config.py lives one level up) -------------------------
# Add project root to path so `config` is importable from any working dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402  (import not at top — intentional after path fix)

# ============================================================================
# MODULE-LEVEL LOGGER
# ============================================================================
# Using config's setup_logging ensures logs go to both console and file.

logger = config.setup_logging()


# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================

def load_raw_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Read the raw CSV dataset into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file. Defaults to config.RAW_DATA_FILE
                   so callers don't need to know the path.

    Returns:
        pd.DataFrame: Raw, unprocessed DataFrame exactly as stored on disk.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the given path.
        pd.errors.EmptyDataError: If the file exists but contains no data.
        Exception: Any other unexpected read error (logged before re-raise).

    Example:
        >>> df = load_raw_data()
        >>> df.shape
        (195, 35)
    """
    if file_path is None:
        file_path = config.RAW_DATA_FILE

    file_path = Path(file_path)

    # --- Validate file exists before attempting read ---
    if not file_path.exists():
        logger.error(f"Dataset not found at: {file_path}")
        raise FileNotFoundError(
            f"Dataset not found at: {file_path}\n"
            f"Expected location: {config.RAW_DATA_FILE}\n"
            f"Please place 'world-data-2023.csv' inside data/raw/"
        )

    try:
        df = pd.read_csv(file_path)

        logger.info(f"Dataset loaded successfully from: {file_path}")
        logger.info(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        logger.debug(f"  Columns: {df.columns.tolist()}")

        # Sanity check: warn if dataset is suspiciously small
        if df.shape[0] < 10:
            logger.warning(
                f"Dataset has only {df.shape[0]} rows — this may be too "
                f"small for reliable ML modeling."
            )

        return df

    except pd.errors.EmptyDataError:
        logger.error(f"File exists but is empty: {file_path}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise


# ============================================================================
# 2. DROP UNUSABLE ROWS AND COLUMNS
# ============================================================================

def drop_unusable_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns with no predictive value and countries with too many
    missing values to be useful.

    This runs BEFORE numeric cleaning because the columns we're dropping
    (e.g., 'Abbreviation', 'Currency-Code') are identifiers that would
    break numeric conversion if left in.

    Args:
        df: Raw DataFrame from load_raw_data().

    Returns:
        pd.DataFrame: DataFrame with junk columns and near-empty rows removed.
    """
    df = df.copy()
    initial_shape = df.shape

    # --- Drop columns that are IDs / have no predictive value ---
    cols_to_drop = [
        col for col in config.COLUMNS_TO_DROP
        if col in df.columns  # Defensive: only drop if column actually exists
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(
            f"Dropped {len(cols_to_drop)} ID/non-predictive columns: "
            f"{cols_to_drop}"
        )

    # --- Drop countries with 20+ null columns (Vatican City, etc.) ---
    if "Country" in df.columns:
        countries_present = [
            c for c in config.COUNTRIES_TO_DROP
            if c in df["Country"].values
        ]
        if countries_present:
            df = df[~df["Country"].isin(countries_present)]
            logger.info(
                f"Dropped {len(countries_present)} near-empty country rows: "
                f"{countries_present}"
            )

    logger.info(
        f"Shape after cleanup: {df.shape} "
        f"(removed {initial_shape[0] - df.shape[0]} rows, "
        f"{initial_shape[1] - df.shape[1]} columns)"
    )

    return df.reset_index(drop=True)


# ============================================================================
# 3. CLEAN COLUMN NAMES
# ============================================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to a consistent, code-friendly format.

    Strategy:
        1. First apply config.COLUMN_RENAME_MAP for domain-specific names
           (e.g., 'Density\\n(P/Km2)' → 'Density_per_km2').
        2. For any remaining columns NOT in the rename map, apply generic
           cleaning: strip whitespace, replace special chars with '_',
           collapse duplicate underscores, strip leading/trailing '_'.

    Why two steps?
        The rename map gives us *meaningful* names chosen by a domain expert.
        The fallback ensures no column is left with spaces or newlines even
        if the dataset schema changes unexpectedly.

    Args:
        df: DataFrame with raw column names.

    Returns:
        pd.DataFrame: DataFrame with clean, standardized column names.
    """
    df = df.copy()

    # --- Step 1: Apply the curated rename map ---
    # Only rename columns that actually exist in the DataFrame
    rename_map = {
        old: new
        for old, new in config.COLUMN_RENAME_MAP.items()
        if old in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed {len(rename_map)} columns via config map")
        logger.debug(f"  Renames applied: {rename_map}")

    # --- Step 2: Generic cleanup for any remaining unhandled columns ---
    cleaned_names = {}
    for col in df.columns:
        new_name = col
        # Replace newlines, tabs, and special chars with underscores
        for char in ["\n", "\t", "(", ")", "/", "-", ":", " "]:
            new_name = new_name.replace(char, "_")
        # Remove percentage and dollar signs
        new_name = new_name.replace("%", "pct").replace("$", "")
        # Collapse multiple consecutive underscores → single
        while "__" in new_name:
            new_name = new_name.replace("__", "_")
        # Strip leading/trailing underscores
        new_name = new_name.strip("_")

        if new_name != col:
            cleaned_names[col] = new_name

    if cleaned_names:
        df = df.rename(columns=cleaned_names)
        logger.debug(f"  Generic cleanup renames: {cleaned_names}")

    logger.info(f"Final column names ({len(df.columns)}): {df.columns.tolist()}")

    return df


# ============================================================================
# 4. CLEAN NUMERIC VALUES
# ============================================================================

def _safe_numeric_convert(value: object) -> object:
    """
    Attempt to convert a single value from a messy string to a float.

    Handles: '$19,101,353,833 ', '58.10%', '323,000', '', None, NaN.
    Returns the original value unchanged if conversion fails (e.g., 'Kabul').

    This is intentionally conservative — we'd rather leave a value as-is
    than corrupt a legitimate string like a country name.
    """
    # Already numeric — no conversion needed
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value

    # Not a string — leave it (covers None, NaN, etc.)
    if not isinstance(value, str):
        return value

    # Strip whitespace first
    cleaned = value.strip()

    # Empty string → NaN (missing data)
    if cleaned == "" or cleaned == "-":
        return np.nan

    # Remove currency symbol and percentage sign
    cleaned = cleaned.replace("$", "").replace("%", "")

    # Remove thousands-separator commas (but preserve decimal dots)
    cleaned = cleaned.replace(",", "")

    # Final whitespace strip after removals
    cleaned = cleaned.strip()

    # Attempt float conversion
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        # Conversion failed → this is probably a genuine string (country, city)
        return value


def clean_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string-encoded numeric columns to actual float values.

    The raw dataset stores most numbers as strings with formatting:
        - GDP:           '$19,101,353,833 '  (dollar sign, commas, spaces)
        - Percentages:   '58.10%'            (percentage sign)
        - Populations:   '38,041,754'        (commas as thousands separators)
        - CPI:           '4,583.71'          (commas + decimal)

    This function applies _safe_numeric_convert to every cell in the
    targeted columns. Columns that are already numeric or genuinely
    categorical (like 'Country') are left untouched.

    Why per-column groups?
        Different column types need different validation after cleaning.
        Processing them in known groups lets us log specific warnings.

    Args:
        df: DataFrame with raw string values.

    Returns:
        pd.DataFrame: DataFrame with numeric columns properly typed.
    """
    df = df.copy()
    converted_count = 0

    # --- Group 1: Percentage columns (stored as "XX.XX%") ---
    for col in config.PERCENTAGE_COLUMNS:
        # Check against both original and renamed column names
        col_renamed = config.COLUMN_RENAME_MAP.get(col, col)
        target_col = col_renamed if col_renamed in df.columns else col
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_safe_numeric_convert)
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            converted_count += 1
            logger.debug(f"  Cleaned percentage column: {target_col}")

    # --- Group 2: Currency columns (stored as "$X,XXX") ---
    for col in config.CURRENCY_COLUMNS:
        col_renamed = config.COLUMN_RENAME_MAP.get(col, col)
        target_col = col_renamed if col_renamed in df.columns else col
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_safe_numeric_convert)
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            converted_count += 1
            logger.debug(f"  Cleaned currency column: {target_col}")

    # --- Group 3: Comma-formatted integers (stored as "X,XXX") ---
    for col in config.COMMA_NUMERIC_COLUMNS:
        col_renamed = config.COLUMN_RENAME_MAP.get(col, col)
        target_col = col_renamed if col_renamed in df.columns else col
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_safe_numeric_convert)
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            converted_count += 1
            logger.debug(f"  Cleaned comma-numeric column: {target_col}")

    # --- Group 4: CPI column (has commas in large values like "4,583.71") ---
    for col in config.CPI_COLUMNS:
        col_renamed = config.COLUMN_RENAME_MAP.get(col, col)
        target_col = col_renamed if col_renamed in df.columns else col
        if target_col in df.columns:
            df[target_col] = df[target_col].apply(_safe_numeric_convert)
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            converted_count += 1
            logger.debug(f"  Cleaned CPI column: {target_col}")

    logger.info(f"Converted {converted_count} columns from string → numeric")

    # --- Final pass: coerce any remaining object columns that look numeric ---
    # This catches edge cases where a column wasn't in any group but is
    # still stored as a parseable numeric string.
    for col in df.columns:
        if df[col].dtype == "object" and col != "Country":
            # Sample a few non-null values to check if they look numeric
            sample_values = df[col].dropna().head(10)
            numeric_looking = 0
            for val in sample_values:
                try:
                    _safe_numeric_convert(val)
                    if isinstance(_safe_numeric_convert(val), float):
                        numeric_looking += 1
                except Exception:
                    pass
            # If >70% of sampled values are numeric, convert the column
            if len(sample_values) > 0 and numeric_looking / len(sample_values) > 0.7:
                df[col] = df[col].apply(_safe_numeric_convert)
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logger.debug(f"  Auto-detected and cleaned numeric column: {col}")

    return df


# ============================================================================
# 5. MISSING VALUE ANALYSIS (Diagnostic — does NOT impute)
# ============================================================================

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary report of missing values in the dataset.

    This is a DIAGNOSTIC function only — it does not modify the data.
    Imputation happens later inside the sklearn pipeline to prevent leakage.

    Args:
        df: Cleaned DataFrame.

    Returns:
        pd.DataFrame: Summary table with columns:
            - column: column name
            - missing_count: number of NaN values
            - missing_pct: percentage of missing values
            - dtype: current data type
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    dtypes = df.dtypes

    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_pct": missing_pct.values,
        "dtype": dtypes.values,
    })

    # Sort by missing percentage descending — worst columns first
    summary = summary.sort_values("missing_pct", ascending=False)
    summary = summary.reset_index(drop=True)

    # Log high-missing columns
    high_missing = summary[summary["missing_pct"] > 10]
    if not high_missing.empty:
        logger.warning(
            f"Columns with >10% missing values:\n"
            f"{high_missing[['column', 'missing_pct']].to_string(index=False)}"
        )
    else:
        logger.info("No columns exceed 10% missing values")

    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    logger.info(
        f"Overall missing: {total_missing}/{total_cells} cells "
        f"({total_missing / total_cells * 100:.2f}%)"
    )

    return summary


# ============================================================================
# 6. CREATE MISSINGNESS INDICATOR FLAGS
# ============================================================================

def create_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    For columns with high missingness (>10%), create binary indicator features
    that capture WHETHER a value was missing, before imputation fills it in.

    Why this matters:
        Missingness itself can be informative. A country with no reported
        minimum wage may be structurally different from one with a $0 wage.
        These flags preserve that signal even after imputation replaces
        NaN with a median.

    Args:
        df: DataFrame (before imputation).

    Returns:
        pd.DataFrame: Original DataFrame plus new binary columns named
                      '{original_col}_is_missing'.
    """
    df = df.copy()
    flags_created = []

    for col in config.HIGH_MISSING_COLUMNS:
        if col in df.columns:
            flag_name = f"{col}_is_missing"
            df[flag_name] = df[col].isnull().astype(int)
            flags_created.append(flag_name)

    if flags_created:
        logger.info(f"Created {len(flags_created)} missingness flags: {flags_created}")

    return df


# ============================================================================
# 7. IDENTIFY FEATURE TYPES
# ============================================================================

def identify_feature_types(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Classify each column as numerical or categorical based on its dtype.

    This classification drives the ColumnTransformer: numerical columns
    get median imputation + scaling, categorical columns get mode
    imputation + one-hot encoding.

    Args:
        df: Cleaned DataFrame.
        target_column: Name of the target column to exclude from features.
        exclude_columns: Additional columns to exclude (e.g., 'Country').

    Returns:
        Tuple of (numerical_features, categorical_features):
            - numerical_features: List of column names with numeric dtypes.
            - categorical_features: List of column names with object/category dtypes.
    """
    if exclude_columns is None:
        exclude_columns = []

    # Columns to exclude from features entirely
    non_feature_cols = set(
        [target_column, "Country"] + exclude_columns
    )

    # Only consider columns that are actually in the DataFrame
    feature_cols = [
        col for col in df.columns
        if col not in non_feature_cols
    ]

    numerical_features = [
        col for col in feature_cols
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    categorical_features = [
        col for col in feature_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    logger.info(
        f"Feature type identification:\n"
        f"  Numerical features  : {len(numerical_features)}\n"
        f"  Categorical features: {len(categorical_features)}\n"
        f"  Excluded columns    : {non_feature_cols}"
    )
    logger.debug(f"  Numerical: {numerical_features}")
    logger.debug(f"  Categorical: {categorical_features}")

    return numerical_features, categorical_features


# ============================================================================
# 8. TRAIN / TEST SPLIT
# ============================================================================

def create_train_test_split(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets with stratification.

    Stratification ensures each stability class (Stable / Watch / At-Risk)
    is proportionally represented in both train and test sets. Without it,
    random chance on ~190 rows could produce a test set missing an entire
    class — making evaluation meaningless.

    DATA LEAKAGE NOTE:
        This split MUST happen BEFORE any fitting of imputers, scalers,
        or encoders. The test set must remain completely unseen during
        all preprocessing decisions.

    Args:
        df: DataFrame containing both features and the target column.
        target_column: Name of the target column.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If target_column is not found in the DataFrame.
    """
    # --- Validate target column exists ---
    if target_column not in df.columns:
        available = df.columns.tolist()
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame.\n"
            f"Available columns: {available}\n"
            f"Hint: Has the target been engineered yet? The target is "
            f"created in feature_engineering.py, not in raw data."
        )

    # --- Separate features and target ---
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- Determine stratification ---
    # Stratify only for categorical / discrete targets (our stability labels)
    stratify_arg = None
    if config.STRATIFY and y.nunique() <= config.N_STABILITY_CLASSES * 2:
        stratify_arg = y
        logger.info(f"Stratified split enabled (target has {y.nunique()} classes)")
    else:
        logger.info("Non-stratified split (target is continuous or has many values)")

    # --- Perform the split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        shuffle=config.SHUFFLE,
        stratify=stratify_arg,
    )

    logger.info(
        f"Train/Test split completed:\n"
        f"  X_train: {X_train.shape}  |  y_train: {y_train.shape}\n"
        f"  X_test : {X_test.shape}   |  y_test : {y_test.shape}"
    )

    # --- Log class distribution in both sets ---
    if stratify_arg is not None:
        train_dist = y_train.value_counts(normalize=True).round(3).to_dict()
        test_dist = y_test.value_counts(normalize=True).round(3).to_dict()
        logger.info(f"  Train class distribution: {train_dist}")
        logger.info(f"  Test  class distribution: {test_dist}")

    return X_train, X_test, y_train, y_test


# ============================================================================
# 9. BUILD PREPROCESSING PIPELINE (UNFITTED)
# ============================================================================

def build_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Construct an sklearn ColumnTransformer that handles both numerical and
    categorical features in a single, composable pipeline object.

    Architecture:
        ┌─────────────────────────────────────────────┐
        │            ColumnTransformer                 │
        │                                             │
        │  ┌─────────────── numerical ──────────────┐ │
        │  │  SimpleImputer(strategy='median')       │ │
        │  │  StandardScaler()                       │ │
        │  └────────────────────────────────────────-┘ │
        │                                             │
        │  ┌────────────── categorical ─────────────┐ │
        │  │  SimpleImputer(strategy='most_frequent')│ │
        │  │  OneHotEncoder(handle_unknown='ignore') │ │
        │  └─────────────────────────────────────────┘ │
        └─────────────────────────────────────────────┘

    DATA LEAKAGE PREVENTION:
        ════════════════════
        This function ONLY builds the pipeline structure.
        It does NOT call .fit() or .fit_transform().

        The pipeline must be fitted ONLY on X_train inside
        model_training.py. This ensures that:
          • Median values come only from training data
          • Scaling mean/std come only from training data
          • One-hot categories come only from training data

        Fitting on the full dataset before splitting would leak
        test-set statistics into the model, inflating metrics.

    Args:
        numerical_features: Column names to route through the numeric pipeline.
        categorical_features: Column names to route through the categorical pipeline.

    Returns:
        ColumnTransformer: Unfitted preprocessor ready for .fit(X_train).
    """
    # --- Numerical sub-pipeline ---
    # 1. Median imputation: robust to outliers (unlike mean), works well for
    #    skewed distributions like GDP and population.
    # 2. StandardScaler: centers features to mean=0, std=1. Required for
    #    Logistic Regression; tree models don't need it but it doesn't hurt.
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # --- Categorical sub-pipeline ---
    # 1. Mode imputation: fills missing categories with the most common value.
    # 2. OneHotEncoder: converts categories to binary columns.
    #    handle_unknown='ignore' prevents errors if test set has unseen values.
    #    sparse_output=False returns dense arrays (easier to work with).
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop=None,  # Keep all categories — dataset is small
            )),
        ]
    )

    # --- Combine into a single ColumnTransformer ---
    # 'remainder=passthrough' keeps any columns not listed (e.g., missingness
    # flags) instead of silently dropping them.
    transformers = []

    if numerical_features:
        transformers.append(("numerical", numerical_pipeline, numerical_features))

    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",     # Keep unlisted columns intact
        verbose_feature_names_out=True,  # Prefix output names with transformer name
    )

    logger.info(
        f"Preprocessing pipeline built (UNFITTED):\n"
        f"  Numerical features  ({len(numerical_features)}): "
        f"{numerical_features[:5]}{'...' if len(numerical_features) > 5 else ''}\n"
        f"  Categorical features ({len(categorical_features)}): "
        f"{categorical_features[:5]}{'...' if len(categorical_features) > 5 else ''}"
    )

    return preprocessor


# ============================================================================
# 10. OUTLIER HANDLING (OPTIONAL — IQR CAPPING)
# ============================================================================

def handle_outliers_iqr(
    df: pd.DataFrame,
    numerical_columns: Optional[List[str]] = None,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Cap extreme values using the Interquartile Range (IQR) method.

    Why CAPPING instead of DELETION:
        ───────────────────────────────
        • With only ~190 rows, deleting outliers means losing entire countries.
          Monaco (density 26,337) is an outlier — but it's a real country.
        • Capping preserves the row and its other valid columns while pulling
          the extreme value to the boundary (Q1 - 1.5*IQR or Q3 + 1.5*IQR).
        • This is standard practice for small datasets in production ML.

    When to use:
        Apply ONLY on the TRAINING set. The test set should be capped using
        the same bounds computed from training data (to prevent leakage).
        This function is provided as a utility — the caller is responsible
        for applying it at the right stage.

    Args:
        df: DataFrame to process.
        numerical_columns: Columns to check for outliers. If None, uses all
                           numeric columns in the DataFrame.
        iqr_multiplier: Multiplier for IQR to define outlier bounds.
                        Default is 1.5 (standard). Use 3.0 for a more
                        lenient definition.

    Returns:
        pd.DataFrame: DataFrame with outliers capped (not removed).
    """
    df = df.copy()

    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    total_capped = 0

    for col in numerical_columns:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        # Count values outside bounds before capping
        outliers_low = (df[col] < lower_bound).sum()
        outliers_high = (df[col] > upper_bound).sum()
        col_capped = outliers_low + outliers_high

        if col_capped > 0:
            # Cap values at the bounds (winsorization)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            logger.debug(
                f"  {col}: capped {col_capped} outliers "
                f"(low: {outliers_low}, high: {outliers_high}) "
                f"[bounds: {lower_bound:.2f} – {upper_bound:.2f}]"
            )
            total_capped += col_capped

    logger.info(
        f"IQR outlier capping complete: {total_capped} values capped "
        f"across {len(numerical_columns)} columns "
        f"(multiplier={iqr_multiplier})"
    )

    return df


# ============================================================================
# 11. MASTER ORCHESTRATOR — preprocess_data()
# ============================================================================

def preprocess_data(
    df: Optional[pd.DataFrame] = None,
    target_column: str = config.TARGET_COLUMN,
) -> Dict:
    """
    End-to-end preprocessing orchestrator. Chains all cleaning, splitting,
    and pipeline-building steps in the correct order.

    Pipeline Order:
        1. Load raw data (if not provided)
        2. Drop unusable rows and columns
        3. Clean column names → friendly format
        4. Clean numeric values → string to float
        5. Create missingness indicator flags
        6. Analyze missing values (diagnostic log only)
        7. Identify numerical vs categorical feature types
        8. Split into train/test (stratified)
        9. Build UNFITTED preprocessing pipeline

    Why return a dict?
        Returning a dict with named keys is clearer than a tuple with 7+
        positional values. Callers can access what they need by name:
            result = preprocess_data(df)
            X_train = result['X_train']
            pipeline = result['preprocessor']

    DATA LEAKAGE SAFEGUARD:
        The preprocessor is returned UNFITTED. The caller (model_training.py)
        must call preprocessor.fit_transform(X_train) and then
        preprocessor.transform(X_test) — never fit on the full dataset.

    Args:
        df: Optional pre-loaded DataFrame. If None, loads from config path.
        target_column: Name of the target column in the DataFrame.

    Returns:
        dict with keys:
            - 'X_train': Training features (pd.DataFrame)
            - 'X_test': Testing features (pd.DataFrame)
            - 'y_train': Training labels (pd.Series)
            - 'y_test': Testing labels (pd.Series)
            - 'preprocessor': Unfitted ColumnTransformer
            - 'numerical_features': List of numeric column names
            - 'categorical_features': List of categorical column names
            - 'missing_summary': DataFrame with missing value analysis
            - 'cleaned_df': Full cleaned DataFrame (before split)
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # --- Step 1: Load raw data if not provided ---
    if df is None:
        df = load_raw_data()

    # --- Step 2: Drop unusable columns and near-empty countries ---
    df = drop_unusable_data(df)

    # --- Step 3: Clean column names ---
    df = clean_column_names(df)

    # --- Step 4: Convert string values to numeric ---
    df = clean_numeric_values(df)

    # --- Step 5: Create missingness flags BEFORE imputation ---
    # These binary flags capture whether a value was originally missing.
    # Must happen before any imputation fills in the NaNs.
    df = create_missing_flags(df)

    # --- Step 6: Diagnostic analysis of missing values ---
    missing_summary = analyze_missing_values(df)

    # --- Step 7: Check if target column exists ---
    # The target is engineered in feature_engineering.py, so it may not
    # exist yet during early pipeline runs. In that case, we return the
    # cleaned DataFrame without splitting.
    if target_column not in df.columns:
        logger.warning(
            f"Target column '{target_column}' not found — returning "
            f"cleaned DataFrame without train/test split. "
            f"Run feature_engineering.py first to create the target."
        )
        numerical_features, categorical_features = identify_feature_types(
            df, target_column=target_column
        )
        return {
            "X_train": None,
            "X_test": None,
            "y_train": None,
            "y_test": None,
            "preprocessor": build_preprocessing_pipeline(
                numerical_features, categorical_features
            ),
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "missing_summary": missing_summary,
            "cleaned_df": df,
        }

    # --- Step 8: Identify feature types ---
    numerical_features, categorical_features = identify_feature_types(
        df, target_column=target_column
    )

    # --- Step 9: Train/Test split (stratified) ---
    X_train, X_test, y_train, y_test = create_train_test_split(
        df, target_column=target_column
    )

    # --- Step 10: Build UNFITTED preprocessing pipeline ---
    preprocessor = build_preprocessing_pipeline(
        numerical_features, categorical_features
    )

    logger.info("=" * 60)
    logger.info("DATA PREPROCESSING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "missing_summary": missing_summary,
        "cleaned_df": df,
    }


# ============================================================================
# 12. STANDALONE EXECUTION (for testing / debugging)
# ============================================================================

if __name__ == "__main__":
    """
    Run this file directly to test the preprocessing pipeline:
        python src/data_preprocessing.py

    This will:
        1. Load the raw dataset
        2. Clean and process it
        3. Print a summary of the results
        4. Save the cleaned data to data/processed/cleaned_data.csv
    """
    print("\n" + "=" * 60)
    print("  DATA PREPROCESSING -- Standalone Test Run")
    print("=" * 60 + "\n")

    # Run the full pipeline
    result = preprocess_data()

    # --- Print summary ---
    cleaned_df = result["cleaned_df"]
    print(f"\n{'-' * 40}")
    print(f"Cleaned DataFrame shape : {cleaned_df.shape}")
    print(f"Numerical features      : {len(result['numerical_features'])}")
    print(f"Categorical features    : {len(result['categorical_features'])}")
    print(f"{'-' * 40}")

    print(f"\nColumn dtypes after cleaning:")
    print(cleaned_df.dtypes.value_counts().to_string())

    print(f"\nTop 10 columns by missing values:")
    top_missing = result["missing_summary"].head(10)
    print(top_missing[["column", "missing_count", "missing_pct"]].to_string(index=False))

    # --- Save cleaned data ---
    try:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(config.CLEANED_DATA_FILE, index=False)
        print(f"\nCleaned data saved to: {config.CLEANED_DATA_FILE}")
    except Exception as e:
        print(f"\nWarning: Could not save cleaned data: {e}")

    if result["X_train"] is not None:
        print(f"\nTrain set: {result['X_train'].shape}")
        print(f"Test set : {result['X_test'].shape}")
    else:
        print(
            f"\nNo train/test split performed — target column "
            f"'{config.TARGET_COLUMN}' not found yet."
        )

    print(f"\nPreprocessor type: {type(result['preprocessor']).__name__}")
    print("\n[OK] Preprocessing pipeline test complete.\n")
