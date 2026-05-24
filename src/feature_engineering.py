"""
src/feature_engineering.py — Domain-Driven Feature Engineering
===============================================================

This module transforms raw socioeconomic indicators into meaningful,
business-interpretable features that capture the multi-dimensional nature
of country stability.

Architecture:
    Raw cleaned columns (from data_preprocessing.py)
        |
        v
    +-- add_economic_features()          --> Economic Stress Index, etc.
    +-- add_healthcare_features()        --> Healthcare Stability Score, etc.
    +-- add_education_workforce_features()--> Human Capital Index, etc.
    +-- add_demographic_pressure_features()--> Population Pressure, etc.
    +-- add_environment_resource_features()--> Environmental Pressure, etc.
        |
        v
    create_stability_score()             --> Composite Stability Score
        |
        v
    create_risk_target()                 --> stability_label (0 / 1 / 2)

Target Engineering Philosophy:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This dataset has NO ground-truth conflict/stability labels.
    We engineer a PROXY target from health and development outcomes
    (life expectancy, mortality ratios, physician access).  The ML
    model then predicts this proxy from INDEPENDENT economic and
    demographic features --- ensuring no data leakage.

    The stability_label is a SYNTHETIC label.  It is NOT a real-world
    conflict classification.  Always disclose this in interviews.

Data Leakage Warning:
    ~~~~~~~~~~~~~~~~~~~~
    - stability_score is derived from Life_Expectancy, Infant_Mortality,
      Maternal_Mortality, and Physicians_per_1000.
    - These columns (and stability_score itself) must be EXCLUDED from
      the feature set used for model training.
    - The engineered composite features (ESI, WSS, etc.) that do NOT
      use target-definition columns are safe to use as model inputs.

Author : Anumol
Project: Global Country Stability Intelligence System
"""

# ============================================================================
# IMPORTS
# ============================================================================

import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Project imports ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# ============================================================================
# MODULE-LEVEL LOGGER
# ============================================================================

logger = config.setup_logging()


# ============================================================================
# HELPER UTILITIES
# ============================================================================

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill_value: float = 0.0,
) -> pd.Series:
    """
    Perform element-wise division that gracefully handles zeros and NaNs.

    In country-level data, zero or missing denominators are common:
    a country with Population = 0 (Vatican City) or NaN should not crash
    the pipeline or produce infinity.

    Args:
        numerator: Dividend series.
        denominator: Divisor series.
        fill_value: Value to substitute where division is undefined.
                    Default 0.0 (neutral in most aggregations).

    Returns:
        pd.Series: Result of numerator / denominator, with Inf and NaN
                   replaced by fill_value.

    Example:
        >>> safe_divide(pd.Series([100, 200]), pd.Series([0, 50]))
        0     0.0
        1     4.0
        dtype: float64
    """
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def _col_exists(df: pd.DataFrame, col: str) -> bool:
    """Check if a column exists in the DataFrame (avoids verbose repetition)."""
    return col in df.columns


def _cols_exist(df: pd.DataFrame, cols: List[str]) -> bool:
    """Check if ALL columns in a list exist in the DataFrame."""
    return all(col in df.columns for col in cols)


def _available_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Return the subset of columns that actually exist in the DataFrame."""
    return [col for col in cols if col in df.columns]


def _safe_zscore(series: pd.Series) -> pd.Series:
    """
    Compute z-score of a series, handling zero-variance columns.

    Z-scoring normalizes features to mean=0, std=1 so they can be
    combined into composite indices without one feature dominating
    due to scale differences (e.g., GDP in trillions vs. birth rate ~20).

    If std is zero (constant column), returns all zeros to avoid NaN.
    """
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


# ============================================================================
# 1. ECONOMIC FEATURES
# ============================================================================

def add_economic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features that capture a country's economic health and pressure.

    Features Created:
        - GDP_per_Capita:   Wealth per person (economic power normalized by size)
        - Log_GDP:          Log-scaled GDP (reduces extreme right-skew)
        - Economic_Stress_Index (ESI):
              Composite of unemployment + inflation + tax burden, minus
              economic buffers (GDP per capita, labor participation).
              Higher ESI = country under more economic pressure.
        - Inflation_Unemployment_Pressure:
              Direct combination of the two most immediate economic distress
              signals.  High values flag stagflation risk.
        - Tax_Burden_Index:
              Ratio of total tax rate to tax revenue efficiency.
              High tax burden + low revenue = inefficient system.

    Business Interpretation:
        A consulting analyst would use ESI to rank countries by economic
        fragility.  An investment team would use GDP_per_Capita and
        Inflation_Unemployment_Pressure to price sovereign risk.

    Args:
        df: Cleaned DataFrame from data_preprocessing.py.

    Returns:
        pd.DataFrame: Input DataFrame with new economic columns appended.
    """
    df = df.copy()
    features_created = []

    # --- GDP per Capita: wealth normalized by population size ---
    # Why: Raw GDP conflates country size with wealth.  Luxembourg ($71B GDP)
    #      is far wealthier per person than India ($2.6T GDP).
    if _cols_exist(df, ["GDP", "Population"]):
        df["GDP_per_Capita"] = safe_divide(df["GDP"], df["Population"])
        features_created.append("GDP_per_Capita")

    # --- Log GDP: tames extreme skew (USA=$21T vs. Tuvalu=$47M) ---
    # Why: Tree models can handle raw GDP, but log-GDP improves
    #      linear models and makes distributions more normal.
    if _col_exists(df, "GDP"):
        df["Log_GDP"] = np.log1p(df["GDP"].clip(lower=0))
        features_created.append("Log_GDP")

    # --- Log Population: same rationale as Log GDP ---
    if _col_exists(df, "Population"):
        df["Log_Population"] = np.log1p(df["Population"].clip(lower=0))
        features_created.append("Log_Population")

    # --- Economic Stress Index (ESI) ---
    # Formula: ESI = mean(z_unemployment, z_inflation, z_tax_burden)
    #              - mean(z_gdp_per_capita, z_labor_participation)
    # Interpretation: High ESI = population is under economic pressure.
    stress_components = []
    buffer_components = []

    if _col_exists(df, "Unemployment_Rate_pct"):
        stress_components.append(_safe_zscore(df["Unemployment_Rate_pct"]))
    if _col_exists(df, "CPI_Change_pct"):
        stress_components.append(_safe_zscore(df["CPI_Change_pct"]))
    if _col_exists(df, "Total_Tax_Rate_pct"):
        stress_components.append(_safe_zscore(df["Total_Tax_Rate_pct"]))

    if _col_exists(df, "GDP_per_Capita"):
        buffer_components.append(_safe_zscore(df["GDP_per_Capita"]))
    if _col_exists(df, "Labor_Participation_pct"):
        buffer_components.append(_safe_zscore(df["Labor_Participation_pct"]))

    if stress_components:
        stress_mean = pd.concat(stress_components, axis=1).mean(axis=1)
        buffer_mean = (
            pd.concat(buffer_components, axis=1).mean(axis=1)
            if buffer_components
            else 0
        )
        df["Economic_Stress_Index"] = stress_mean - buffer_mean
        features_created.append("Economic_Stress_Index")

    # --- Inflation-Unemployment Pressure ---
    # Why: These two together signal stagflation --- the worst macro scenario.
    if _cols_exist(df, ["CPI_Change_pct", "Unemployment_Rate_pct"]):
        df["Inflation_Unemployment_Pressure"] = (
            _safe_zscore(df["CPI_Change_pct"])
            + _safe_zscore(df["Unemployment_Rate_pct"])
        ) / 2
        features_created.append("Inflation_Unemployment_Pressure")

    # --- Tax Burden Index ---
    # High total tax rate relative to collected revenue = inefficiency.
    if _cols_exist(df, ["Total_Tax_Rate_pct", "Tax_Revenue_pct"]):
        df["Tax_Burden_Index"] = safe_divide(
            df["Total_Tax_Rate_pct"], df["Tax_Revenue_pct"]
        )
        features_created.append("Tax_Burden_Index")

    logger.info(f"Economic features created ({len(features_created)}): {features_created}")
    return df


# ============================================================================
# 2. HEALTHCARE FEATURES
# ============================================================================

def add_healthcare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features that capture the strength of a country's healthcare
    system and the resulting population health outcomes.

    Features Created:
        - Healthcare_Stability_Score (HSS):
              Composite of life expectancy + physician access MINUS
              infant mortality + maternal mortality + out-of-pocket burden.
              Higher HSS = stronger, more accessible healthcare.
        - Mortality_Pressure_Index:
              Combined infant and maternal mortality signal.
              High values indicate healthcare system failure.
        - Physician_Access_Score:
              Physicians per 1000 normalized by out-of-pocket cost.
              Many doctors + low personal cost = strong public health.
        - Healthcare_Access_Gap:
              Out-of-pocket expenditure * inverse of physician density.
              High gap = citizens exposed to health costs with few doctors.

    DATA LEAKAGE NOTE:
        Life_Expectancy, Infant_Mortality, Maternal_Mortality, and
        Physicians_per_1000 are used to DEFINE the stability target.
        Therefore, Healthcare_Stability_Score and its sub-components
        MUST NOT be used as model input features.  They exist here for
        target construction and dashboard visualization only.

    Args:
        df: DataFrame with cleaned columns.

    Returns:
        pd.DataFrame: DataFrame with healthcare features appended.
    """
    df = df.copy()
    features_created = []

    # --- Healthcare Stability Score (HSS) ---
    # Formula: HSS = mean(z_life_exp, z_physicians)
    #              - mean(z_infant_mort, z_maternal_mort, z_oop_health)
    positive = []
    negative = []

    if _col_exists(df, "Life_Expectancy"):
        positive.append(_safe_zscore(df["Life_Expectancy"]))
    if _col_exists(df, "Physicians_per_1000"):
        positive.append(_safe_zscore(df["Physicians_per_1000"]))

    if _col_exists(df, "Infant_Mortality"):
        negative.append(_safe_zscore(df["Infant_Mortality"]))
    if _col_exists(df, "Maternal_Mortality"):
        negative.append(_safe_zscore(df["Maternal_Mortality"]))
    if _col_exists(df, "OOP_Health_Expenditure_pct"):
        negative.append(_safe_zscore(df["OOP_Health_Expenditure_pct"]))

    if positive or negative:
        pos_mean = (
            pd.concat(positive, axis=1).mean(axis=1) if positive else 0
        )
        neg_mean = (
            pd.concat(negative, axis=1).mean(axis=1) if negative else 0
        )
        df["Healthcare_Stability_Score"] = pos_mean - neg_mean
        features_created.append("Healthcare_Stability_Score")

    # --- Mortality Pressure Index ---
    # Combines infant + maternal mortality into a single vulnerability signal.
    mort_components = []
    if _col_exists(df, "Infant_Mortality"):
        mort_components.append(_safe_zscore(df["Infant_Mortality"]))
    if _col_exists(df, "Maternal_Mortality"):
        mort_components.append(_safe_zscore(df["Maternal_Mortality"]))

    if mort_components:
        df["Mortality_Pressure_Index"] = (
            pd.concat(mort_components, axis=1).mean(axis=1)
        )
        features_created.append("Mortality_Pressure_Index")

    # --- Physician Access Score ---
    # More doctors + lower personal cost = better public health system.
    if _cols_exist(df, ["Physicians_per_1000", "OOP_Health_Expenditure_pct"]):
        # Invert OOP so that low out-of-pocket cost adds to the score
        inverted_oop = 100.0 - df["OOP_Health_Expenditure_pct"].fillna(50)
        df["Physician_Access_Score"] = (
            _safe_zscore(df["Physicians_per_1000"])
            + _safe_zscore(inverted_oop)
        ) / 2
        features_created.append("Physician_Access_Score")

    # --- Healthcare Access Gap ---
    # High out-of-pocket + few doctors = citizens exposed and underserved.
    if _cols_exist(df, ["OOP_Health_Expenditure_pct", "Physicians_per_1000"]):
        df["Healthcare_Access_Gap"] = safe_divide(
            df["OOP_Health_Expenditure_pct"],
            df["Physicians_per_1000"],
        )
        features_created.append("Healthcare_Access_Gap")

    logger.info(
        f"Healthcare features created ({len(features_created)}): {features_created}"
    )
    return df


# ============================================================================
# 3. EDUCATION & WORKFORCE FEATURES
# ============================================================================

def add_education_workforce_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features that capture human capital strength and labor market
    resilience.

    Features Created:
        - Human_Capital_Index:
              Education enrollment (primary + tertiary) + labor participation.
              Higher = better-educated, more economically active population.
        - Workforce_Stability_Score (WSS):
              Labor participation + tertiary education MINUS unemployment
              and birth rate (demographic pressure on labor market).
        - Education_Pipeline:
              Ratio of tertiary to primary enrollment.  Low ratio = students
              drop out before university = weak knowledge economy pipeline.
        - Education_Gap_Index:
              Difference between primary and tertiary enrollment.
              Large gap = many start school, few finish higher education.

    Args:
        df: DataFrame with cleaned columns.

    Returns:
        pd.DataFrame: DataFrame with education/workforce features appended.
    """
    df = df.copy()
    features_created = []

    # --- Human Capital Index ---
    # Combines education coverage and workforce engagement.
    hci_components = []
    if _col_exists(df, "Primary_Education_pct"):
        hci_components.append(_safe_zscore(df["Primary_Education_pct"]))
    if _col_exists(df, "Tertiary_Education_pct"):
        hci_components.append(_safe_zscore(df["Tertiary_Education_pct"]))
    if _col_exists(df, "Labor_Participation_pct"):
        hci_components.append(_safe_zscore(df["Labor_Participation_pct"]))

    if hci_components:
        df["Human_Capital_Index"] = (
            pd.concat(hci_components, axis=1).mean(axis=1)
        )
        features_created.append("Human_Capital_Index")

    # --- Workforce Stability Score (WSS) ---
    # Formula: WSS = mean(z_labor_part, z_tertiary_edu)
    #              - mean(z_unemployment, z_birth_rate)
    ws_positive = []
    ws_negative = []

    if _col_exists(df, "Labor_Participation_pct"):
        ws_positive.append(_safe_zscore(df["Labor_Participation_pct"]))
    if _col_exists(df, "Tertiary_Education_pct"):
        ws_positive.append(_safe_zscore(df["Tertiary_Education_pct"]))

    if _col_exists(df, "Unemployment_Rate_pct"):
        ws_negative.append(_safe_zscore(df["Unemployment_Rate_pct"]))
    if _col_exists(df, "Birth_Rate"):
        ws_negative.append(_safe_zscore(df["Birth_Rate"]))

    if ws_positive or ws_negative:
        pos_mean = (
            pd.concat(ws_positive, axis=1).mean(axis=1) if ws_positive else 0
        )
        neg_mean = (
            pd.concat(ws_negative, axis=1).mean(axis=1) if ws_negative else 0
        )
        df["Workforce_Stability_Score"] = pos_mean - neg_mean
        features_created.append("Workforce_Stability_Score")

    # --- Education Pipeline ---
    # What fraction of primary-enrolled students reach tertiary education?
    # Low ratio = brain drain or systemic education barriers.
    if _cols_exist(df, ["Tertiary_Education_pct", "Primary_Education_pct"]):
        df["Education_Pipeline"] = safe_divide(
            df["Tertiary_Education_pct"],
            df["Primary_Education_pct"],
        )
        features_created.append("Education_Pipeline")

    # --- Education Gap Index ---
    # Absolute gap between primary and tertiary enrollment.
    # Large gap = many children start school but few pursue higher education.
    if _cols_exist(df, ["Primary_Education_pct", "Tertiary_Education_pct"]):
        df["Education_Gap_Index"] = (
            df["Primary_Education_pct"] - df["Tertiary_Education_pct"]
        ).clip(lower=0)
        features_created.append("Education_Gap_Index")

    logger.info(
        f"Education/Workforce features created ({len(features_created)}): "
        f"{features_created}"
    )
    return df


# ============================================================================
# 4. DEMOGRAPHIC PRESSURE FEATURES
# ============================================================================

def add_demographic_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features that capture population structure and urbanization
    pressures that contribute to instability.

    Features Created:
        - Urbanization_Rate:
              Fraction of population living in cities.  Rapid urbanization
              without infrastructure = social strain.
        - Population_Pressure_Index:
              Combines density, birth rate, and fertility rate.
              High pressure = land and resource strain.
        - Dependency_Ratio_Proxy:
              Birth rate relative to labor participation.  High birth rate
              + low labor force = large dependent population.
        - Military_Burden:
              Armed forces as a fraction of total population.
              Very high values may signal militarized states or conflict.
        - Abs_Latitude:
              Distance from equator.  Historically correlated with
              development levels (a geographic control variable).

    Args:
        df: DataFrame with cleaned columns.

    Returns:
        pd.DataFrame: DataFrame with demographic features appended.
    """
    df = df.copy()
    features_created = []

    # --- Urbanization Rate ---
    if _cols_exist(df, ["Urban_Population", "Population"]):
        df["Urbanization_Rate"] = safe_divide(
            df["Urban_Population"], df["Population"]
        )
        features_created.append("Urbanization_Rate")

    # --- Population Pressure Index ---
    # Combines density + birth rate + fertility into a single pressure metric.
    pp_components = []
    if _col_exists(df, "Density_per_km2"):
        pp_components.append(_safe_zscore(df["Density_per_km2"]))
    if _col_exists(df, "Birth_Rate"):
        pp_components.append(_safe_zscore(df["Birth_Rate"]))
    if _col_exists(df, "Fertility_Rate"):
        pp_components.append(_safe_zscore(df["Fertility_Rate"]))

    if pp_components:
        df["Population_Pressure_Index"] = (
            pd.concat(pp_components, axis=1).mean(axis=1)
        )
        features_created.append("Population_Pressure_Index")

    # --- Dependency Ratio Proxy ---
    # High birth rate relative to labor participation = many dependents.
    if _cols_exist(df, ["Birth_Rate", "Labor_Participation_pct"]):
        df["Dependency_Ratio_Proxy"] = safe_divide(
            df["Birth_Rate"], df["Labor_Participation_pct"]
        )
        features_created.append("Dependency_Ratio_Proxy")

    # --- Military Burden ---
    # Armed forces as % of population.  Very high = potential conflict zone.
    if _cols_exist(df, ["Armed_Forces", "Population"]):
        df["Military_Burden"] = safe_divide(
            df["Armed_Forces"], df["Population"]
        )
        features_created.append("Military_Burden")

    # --- Absolute Latitude ---
    # Distance from equator: a well-documented geographic development proxy.
    # Not causal, but useful as a control variable in stability models.
    if _col_exists(df, "Latitude"):
        df["Abs_Latitude"] = df["Latitude"].abs()
        features_created.append("Abs_Latitude")

    logger.info(
        f"Demographic features created ({len(features_created)}): "
        f"{features_created}"
    )
    return df


# ============================================================================
# 5. ENVIRONMENT & RESOURCE FEATURES
# ============================================================================

def add_environment_resource_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features that capture environmental sustainability and
    resource constraints.

    Features Created:
        - CO2_per_Capita:
              Emissions normalized by population.  Proxy for
              industrialization intensity.
        - CO2_per_GDP:
              Emissions per unit of economic output.  High values =
              carbon-inefficient economy.
        - Environmental_Pressure_Index:
              Composite of CO2 intensity + low forest cover + high
              agricultural land use.
        - Resource_Land_Balance:
              Forested area relative to agricultural land.  Low ratio =
              land overexploited for farming at the cost of ecosystems.
        - Fuel_Cost_Pressure:
              Gasoline price relative to minimum wage.  High ratio =
              fuel is unaffordable for workers = economic drag.

    Args:
        df: DataFrame with cleaned columns.

    Returns:
        pd.DataFrame: DataFrame with environment features appended.
    """
    df = df.copy()
    features_created = []

    # --- CO2 per Capita ---
    if _cols_exist(df, ["CO2_Emissions", "Population"]):
        df["CO2_per_Capita"] = safe_divide(
            df["CO2_Emissions"], df["Population"]
        )
        features_created.append("CO2_per_Capita")

    # --- CO2 per GDP (carbon intensity of economy) ---
    if _cols_exist(df, ["CO2_Emissions", "GDP"]):
        df["CO2_per_GDP"] = safe_divide(df["CO2_Emissions"], df["GDP"])
        features_created.append("CO2_per_GDP")

    # --- Environmental Pressure Index ---
    # High CO2 + low forest + high agricultural land = environmental strain.
    env_components = []
    if _col_exists(df, "CO2_per_Capita"):
        env_components.append(_safe_zscore(df["CO2_per_Capita"]))
    elif _col_exists(df, "CO2_Emissions"):
        env_components.append(_safe_zscore(df["CO2_Emissions"]))

    if _col_exists(df, "Forested_Area_pct"):
        # Invert: LOW forest = HIGH pressure
        env_components.append(-_safe_zscore(df["Forested_Area_pct"]))
    if _col_exists(df, "Agricultural_Land_pct"):
        env_components.append(_safe_zscore(df["Agricultural_Land_pct"]))

    if env_components:
        df["Environmental_Pressure_Index"] = (
            pd.concat(env_components, axis=1).mean(axis=1)
        )
        features_created.append("Environmental_Pressure_Index")

    # --- Resource-Land Balance ---
    # Forest / Agricultural land.  Low = overexploited landscape.
    if _cols_exist(df, ["Forested_Area_pct", "Agricultural_Land_pct"]):
        df["Resource_Land_Balance"] = safe_divide(
            df["Forested_Area_pct"], df["Agricultural_Land_pct"]
        )
        features_created.append("Resource_Land_Balance")

    # --- Fuel Cost Pressure ---
    # Can a minimum-wage worker afford fuel?  High ratio = NO.
    if _cols_exist(df, ["Gasoline_Price", "Minimum_Wage"]):
        df["Fuel_Cost_Pressure"] = safe_divide(
            df["Gasoline_Price"], df["Minimum_Wage"]
        )
        features_created.append("Fuel_Cost_Pressure")

    logger.info(
        f"Environment features created ({len(features_created)}): "
        f"{features_created}"
    )
    return df


# ============================================================================
# 6. COMPOSITE STABILITY SCORE
# ============================================================================

def create_stability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single Country Stability Index (CSI) that summarizes overall
    country health across multiple dimensions.

    The CSI is the FOUNDATION of the target variable.  It aggregates
    positive signals (life expectancy, education, economic output) and
    negative signals (mortality, unemployment, inflation) into one number.

    Methodology:
        1. Z-score each component so they're on comparable scales.
        2. Average the positive z-scores and negative z-scores separately.
        3. CSI = positive_mean - negative_mean.
        4. Higher CSI = more stable country.

    Why z-scores?
        Life expectancy ranges 52-85, GDP per capita 100-100,000, and
        infant mortality 1-85.  Without normalization, GDP would dominate
        the composite simply due to scale.

    The function is ADAPTIVE: it uses whatever target signal columns are
    actually present, rather than failing if one is missing.

    DATA LEAKAGE WARNING:
        =====================
        stability_score is DERIVED from columns like Life_Expectancy and
        Infant_Mortality.  It must NEVER be used as a model input feature.
        It exists solely to create the target label (stability_label).

    Args:
        df: DataFrame with cleaned + engineered features.

    Returns:
        pd.DataFrame: DataFrame with 'stability_score' column appended.
    """
    df = df.copy()

    # --- Map config's original column names to cleaned names ---
    # config.TARGET_POSITIVE_SIGNALS uses raw names like "Life expectancy".
    # We need the cleaned equivalents like "Life_Expectancy".
    positive_cols_cleaned = [
        config.COLUMN_RENAME_MAP.get(col, col)
        for col in config.TARGET_POSITIVE_SIGNALS
    ]
    negative_cols_cleaned = [
        config.COLUMN_RENAME_MAP.get(col, col)
        for col in config.TARGET_NEGATIVE_SIGNALS
    ]

    # Also include GDP_per_Capita as a positive signal if available
    # (it's engineered, not in the original config list)
    if _col_exists(df, "GDP_per_Capita"):
        positive_cols_cleaned.append("GDP_per_Capita")

    # --- Filter to columns that actually exist ---
    available_positive = _available_cols(df, positive_cols_cleaned)
    available_negative = _available_cols(df, negative_cols_cleaned)

    logger.info(
        f"Stability score components:\n"
        f"  Positive signals ({len(available_positive)}): {available_positive}\n"
        f"  Negative signals ({len(available_negative)}): {available_negative}"
    )

    if not available_positive and not available_negative:
        logger.error(
            "Cannot create stability score: no target signal columns found. "
            "Check that data_preprocessing.py has run and columns exist."
        )
        df["stability_score"] = np.nan
        return df

    # --- Compute z-scores and aggregate ---
    positive_z = []
    for col in available_positive:
        z = _safe_zscore(df[col])
        positive_z.append(z)
        logger.debug(f"  + {col}: mean={df[col].mean():.2f}, z-range=[{z.min():.2f}, {z.max():.2f}]")

    negative_z = []
    for col in available_negative:
        z = _safe_zscore(df[col])
        negative_z.append(z)
        logger.debug(f"  - {col}: mean={df[col].mean():.2f}, z-range=[{z.min():.2f}, {z.max():.2f}]")

    positive_mean = (
        pd.concat(positive_z, axis=1).mean(axis=1) if positive_z else 0
    )
    negative_mean = (
        pd.concat(negative_z, axis=1).mean(axis=1) if negative_z else 0
    )

    df["stability_score"] = positive_mean - negative_mean

    logger.info(
        f"Stability score created: "
        f"min={df['stability_score'].min():.3f}, "
        f"max={df['stability_score'].max():.3f}, "
        f"mean={df['stability_score'].mean():.3f}"
    )

    return df


# ============================================================================
# 7. RISK TARGET CREATION
# ============================================================================

def create_risk_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the continuous stability_score into discrete risk categories.

    This creates the TARGET COLUMN for the classification model.

    Binning Strategy:
        We use QUANTILE-BASED thresholds (pd.qcut) to create 3 classes
        with approximately equal size.

        Why quantiles and not fixed thresholds?
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        1. No official stability labels exist --- we don't know the "right"
           cutoff between Stable and At-Risk.
        2. Quantiles guarantee balanced classes (~63 countries each for
           3 classes on 190 rows).  This avoids class imbalance problems
           that would degrade classifier performance.
        3. The boundaries adapt to the data distribution, making the
           system robust to dataset updates.

    Class Mapping (from config.py):
        0 = "Stable"   (top 33% of stability_score)
        1 = "Watch"    (middle 33%)
        2 = "At-Risk"  (bottom 33%)

    Args:
        df: DataFrame containing 'stability_score' column.

    Returns:
        pd.DataFrame: DataFrame with config.TARGET_COLUMN appended.

    Raises:
        ValueError: If 'stability_score' column is not present.
    """
    df = df.copy()

    if "stability_score" not in df.columns:
        raise ValueError(
            "Column 'stability_score' not found. "
            "Run create_stability_score() first."
        )

    # --- Drop rows where stability_score is NaN (can't classify them) ---
    valid_mask = df["stability_score"].notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.warning(
            f"Dropping {n_dropped} rows with NaN stability_score "
            f"before creating target labels."
        )

    # --- Quantile-based binning ---
    # pd.qcut splits into equal-frequency bins.
    # labels=[2, 1, 0] because:
    #   - Lowest stability_score quantile → 2 (At-Risk)
    #   - Middle quantile                 → 1 (Watch)
    #   - Highest quantile               → 0 (Stable)
    try:
        df[config.TARGET_COLUMN] = pd.qcut(
            df["stability_score"],
            q=config.N_STABILITY_CLASSES,
            labels=[2, 1, 0],       # Lowest score = At-Risk (2)
            duplicates="drop",      # Handle edge case of tied values
        ).astype(float)             # float to handle any residual NaN

        # Convert to integer where possible
        df[config.TARGET_COLUMN] = (
            df[config.TARGET_COLUMN]
            .fillna(-1)             # Temporary marker for NaN
            .astype(int)
        )

        # Remove rows that couldn't be labeled
        df = df[df[config.TARGET_COLUMN] != -1].reset_index(drop=True)

    except ValueError as e:
        logger.error(f"Quantile binning failed: {e}")
        logger.info("Falling back to manual tertile boundaries.")
        # Fallback: compute tertile boundaries manually
        score = df.loc[valid_mask, "stability_score"]
        t1 = score.quantile(1 / 3)
        t2 = score.quantile(2 / 3)

        def _assign_label(s):
            if pd.isna(s):
                return -1
            if s <= t1:
                return 2   # At-Risk
            elif s <= t2:
                return 1   # Watch
            else:
                return 0   # Stable

        df[config.TARGET_COLUMN] = df["stability_score"].apply(_assign_label)
        df = df[df[config.TARGET_COLUMN] != -1].reset_index(drop=True)

    # --- Log class distribution ---
    class_counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    logger.info(f"Target column '{config.TARGET_COLUMN}' created:")
    for cls_id, count in class_counts.items():
        label = config.CLASS_LABELS.get(cls_id, "Unknown")
        pct = count / len(df) * 100
        logger.info(f"  Class {cls_id} ({label}): {count} countries ({pct:.1f}%)")

    return df


# ============================================================================
# 8. IDENTIFY TARGET-LEAKY COLUMNS
# ============================================================================

def get_leaky_columns() -> List[str]:
    """
    Return the list of columns that were used to construct the target
    variable and therefore MUST be excluded from model training features.

    These columns are the direct inputs to stability_score.  If the model
    sees them, it's essentially doing a lookup rather than learning
    meaningful patterns from independent features.

    Returns:
        List[str]: Column names that are leaky (cleaned names).
    """
    # Map original config names to cleaned names
    leaky = []
    for col in config.TARGET_POSITIVE_SIGNALS + config.TARGET_NEGATIVE_SIGNALS:
        cleaned = config.COLUMN_RENAME_MAP.get(col, col)
        leaky.append(cleaned)

    # The score itself and healthcare features derived from target columns
    leaky.extend([
        "stability_score",
        "Healthcare_Stability_Score",
        "Mortality_Pressure_Index",
        "Physician_Access_Score",
    ])

    logger.debug(f"Leaky columns (to exclude from features): {leaky}")
    return leaky


def drop_leaky_columns(
    df: pd.DataFrame,
    additional_drops: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Remove target-construction columns from the feature set.

    This is the CRITICAL anti-leakage step.  After calling this function,
    the DataFrame contains ONLY legitimate prediction features and the
    target label.

    Args:
        df: DataFrame with all features + target column.
        additional_drops: Extra columns to remove (e.g., 'Country' before modeling).

    Returns:
        pd.DataFrame: Leakage-free DataFrame ready for modeling.
    """
    df = df.copy()
    leaky = get_leaky_columns()

    if additional_drops:
        leaky.extend(additional_drops)

    cols_to_drop = [col for col in leaky if col in df.columns]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(
            f"Dropped {len(cols_to_drop)} leaky/non-feature columns: "
            f"{cols_to_drop}"
        )

    return df


# ============================================================================
# 9. MASTER ORCHESTRATOR
# ============================================================================

def engineer_features(
    df: pd.DataFrame,
    create_target: bool = True,
    drop_leaky: bool = True,
) -> pd.DataFrame:
    """
    End-to-end feature engineering orchestrator.

    Pipeline Order:
        1. Economic features      (ESI, GDP per capita, tax burden, ...)
        2. Healthcare features    (HSS, mortality pressure, ...)
        3. Education & workforce  (HCI, WSS, education pipeline, ...)
        4. Demographic pressure   (urbanization, military burden, ...)
        5. Environment & resource (CO2 intensity, fuel cost, ...)
        6. Stability score        (composite CSI from target signals)
        7. Risk target            (quantile binning into 3 classes)
        8. Drop leaky columns     (remove target-construction columns)

    Args:
        df: Cleaned DataFrame from data_preprocessing.py.
        create_target: Whether to create stability_score and risk labels.
                       Set False if you only want engineered features
                       (e.g., for inference on new data).
        drop_leaky: Whether to remove columns used in target construction.
                    Set False if you need them for analysis/visualization.

    Returns:
        pd.DataFrame: Feature-engineered DataFrame, optionally with target
                      and leaky columns removed.
    """
    logger.info("=" * 60)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    initial_cols = set(df.columns)

    # --- Step 1-5: Domain-specific feature groups ---
    df = add_economic_features(df)
    df = add_healthcare_features(df)
    df = add_education_workforce_features(df)
    df = add_demographic_pressure_features(df)
    df = add_environment_resource_features(df)

    new_features = set(df.columns) - initial_cols
    logger.info(
        f"Total engineered features created: {len(new_features)}\n"
        f"  {sorted(new_features)}"
    )

    # --- Step 6-7: Target creation ---
    if create_target:
        df = create_stability_score(df)
        df = create_risk_target(df)
        logger.info(f"Target column '{config.TARGET_COLUMN}' is ready.")

    # --- Step 8: Remove leaky columns ---
    if drop_leaky and create_target:
        df = drop_leaky_columns(df, additional_drops=["Country"])

    logger.info(f"Final DataFrame shape: {df.shape}")
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return df


# ============================================================================
# 10. STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run this file directly to test feature engineering:
        python src/feature_engineering.py

    This will:
        1. Load the cleaned dataset from data/processed/cleaned_data.csv
        2. Engineer all features + create target
        3. Save the result to data/processed/feature_engineered_data.csv
        4. Print a summary
    """
    print("\n" + "=" * 60)
    print("  FEATURE ENGINEERING -- Standalone Test Run")
    print("=" * 60 + "\n")

    # --- Load cleaned data ---
    try:
        df = pd.read_csv(config.CLEANED_DATA_FILE)
        print(f"Loaded cleaned data: {df.shape}")
    except FileNotFoundError:
        print(
            f"Cleaned data not found at {config.CLEANED_DATA_FILE}.\n"
            f"Run 'python src/data_preprocessing.py' first."
        )
        sys.exit(1)

    # --- Run full feature engineering (with target, without dropping leaky) ---
    # Keep leaky columns here so we can inspect the full picture
    df_full = engineer_features(df, create_target=True, drop_leaky=False)

    print(f"\n{'-' * 50}")
    print(f"Full DataFrame shape   : {df_full.shape}")
    print(f"Columns: {df_full.columns.tolist()}")
    print(f"{'-' * 50}")

    # --- Show target distribution ---
    if config.TARGET_COLUMN in df_full.columns:
        print(f"\nTarget distribution ({config.TARGET_COLUMN}):")
        dist = df_full[config.TARGET_COLUMN].value_counts().sort_index()
        for cls_id, count in dist.items():
            label = config.CLASS_LABELS.get(cls_id, "Unknown")
            pct = count / len(df_full) * 100
            print(f"  {cls_id} ({label}): {count} ({pct:.1f}%)")

    # --- Show stability score distribution ---
    if "stability_score" in df_full.columns:
        score = df_full["stability_score"]
        print(f"\nStability score stats:")
        print(f"  Min : {score.min():.3f}")
        print(f"  Mean: {score.mean():.3f}")
        print(f"  Max : {score.max():.3f}")
        print(f"  Std : {score.std():.3f}")

    # --- Now create the model-ready version (with leaky columns dropped) ---
    df_model = engineer_features(
        pd.read_csv(config.CLEANED_DATA_FILE),
        create_target=True,
        drop_leaky=True,
    )

    print(f"\nModel-ready DataFrame shape: {df_model.shape}")
    print(f"Model-ready columns: {df_model.columns.tolist()}")

    # --- Save both versions ---
    try:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        df_full.to_csv(config.FEATURE_ENGINEERED_FILE, index=False)
        print(f"\nFull featured data saved to: {config.FEATURE_ENGINEERED_FILE}")

        df_model.to_csv(config.FINAL_DATASET_FILE, index=False)
        print(f"Model-ready data saved to : {config.FINAL_DATASET_FILE}")

    except Exception as e:
        print(f"\nWarning: Could not save data: {e}")

    # --- Show leaky columns that were removed ---
    leaky = get_leaky_columns()
    print(f"\nLeaky columns removed for modeling: {leaky}")

    print("\n[OK] Feature engineering test complete.\n")
