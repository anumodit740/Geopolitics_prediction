# 🌍 Global Country Stability Intelligence System

**Predicting country-level stability risk using real-world socioeconomic data.**

> A fully modular, production-grade ML system that quantifies country stability through proven socioeconomic indicators. Built with the rigor of a consulting firm. Designed to be understood, maintained, and extended.

---

## 🎯 The Problem

During analytics work, a persistent question emerged: *Can we actually quantify how "stable" a country is using publicly available data?*

The answer is **yes** — but it requires thoughtful feature engineering, the right models, and a pipeline designed to prevent leakage and overfitting.

This isn't a notebook. It's a complete, production-ready system that:
- ✅ Takes raw country indicators (GDP, healthcare, education, demographics)
- ✅ Classifies each nation into one of **3 risk categories**
- ✅ Includes an interactive Streamlit dashboard for scenario analysis
- ✅ Maintains explainability at every step

---

## 🎬 Live Demo

<div align="center">

### 🚀 **Try the Interactive Dashboard**

[![Streamlit App](https://img.shields.io/badge/🎨%20Streamlit%20App-Live%20Demo-brightgreen?style=for-the-badge&logo=streamlit)](https://geopoliticsprediction-4n86qou6o6gfg2hcdtyt44.streamlit.app/)

**Explore real-time country stability predictions with an interactive interface:**

- 🎚️ **Adjust Indicators** — Use sliders to modify GDP, birth rate, healthcare spending, education enrollment
- 📊 **Instant Predictions** — See stability risk category update in real time
- 📈 **Feature Impact Visualization** — Understand which indicators drive predictions
- 💾 **Export Results** — Download predictions as CSV for further analysis
- 🌐 **Scenario Analysis** — Test hypothetical country profiles

**✨ No installation required — click above to launch!**

</div>

---

## 📊 What It Does

The system transforms raw socioeconomic data into actionable stability predictions:

```
Raw Input (189 countries, 35 indicators)
           ↓
    [Feature Engineering] → 24 composite features
           ↓
    [Classification Models] → Logistic Regression, Random Forest, XGBoost
           ↓
Output (Stability Risk Category)
```

### Prediction Classes

| Class | Label | Definition | Business Context |
|:---:|:---:|---|---|
| 0 | 🟢 **Stable** | Strong fundamentals across all dimensions | Low risk exposure; safe for long-term strategy |
| 1 | 🟡 **Watch** | Mixed signals requiring monitoring | Emerging risks; requires scenario planning |
| 2 | 🔴 **At-Risk** | Significant vulnerability indicators | High instability; urgent intervention needed |

---

## 🏆 Results

**XGBoost delivered the best performance on the held-out test set (38 countries):**

| Metric | Score |
|--------|-------|
| **Macro F1** | **0.804** 🥇 |
| **Accuracy** | **81.6%** |
| **At-Risk Recall** | **92.3%** |

> **Why Macro F1 matters:** With 3 balanced classes, accuracy alone is misleading. The model correctly flags **12 of 13** genuinely at-risk countries—the metric that drives real decision-making.

### Model Comparison

```
XGBoost      ████████████████████ 0.804
Random Forest ███████████████████ 0.780
Log. Regression ███████████████████ 0.780
```

### 🔍 Top 5 Features Driving Predictions

These findings are *intuitive and explainable*—not noise:

1. **Birth Rate** — High fertility in low-income settings signals demographic pressure
2. **Education Pipeline Ratio** (tertiary/primary enrollment) — Human capital determines future prosperity
3. **GDP per Capita** — Wealth correlates strongly with institutional stability
4. **Population Pressure Index** — Density relative to arable land creates resource stress
5. **Healthcare Access Gap** — Out-of-pocket costs signal fragile healthcare systems

---

## 🗂️ Project Architecture

```
Geopolitics_prediction/
│
├── 📋 config.py                         # Single source of truth: paths, params, column maps
├── 🚀 main.py                           # Full pipeline orchestrator (run this first)
├── 🎨 app.py                            # Interactive Streamlit dashboard
│
├── 📁 src/
│   ├── data_preprocessing.py            # Load, clean, split → sklearn pipelines
│   ├── feature_engineering.py           # 24 engineered features + target creation
│   ├── model_training.py                # 3 models, cross-validation, checkpoint best
│   └── evaluation.py                    # Metrics, confusion matrices, feature importance
│
├── 📊 data/
│   ├── raw/                             # world-data-2023.csv (195 countries, 35 cols)
│   └── processed/                       # Cleaned, engineered, model-ready CSVs
│
├── 🤖 models/                           # Saved preprocessing + model pipelines (.joblib)
├── 📈 reports/
│   ├── figures/                         # Confusion matrices, feature importance plots
│   └── metrics/                         # Classification reports, model comparison
│
├── 📝 logs/                             # Execution logs + pipeline traces
├── requirements.txt                     # All dependencies pinned
└── README.md
```

**Design philosophy:** Each module is self-contained and testable. The pipeline is declarative—no magic, no hidden dependencies.

---

## 🚀 Quick Start

### Step 1: Clone & Install

```bash
git clone https://github.com/anumodit740/Geopolitics_prediction.git
cd Geopolitics_prediction
pip install -r requirements.txt
```

### Step 2: Run the Pipeline

```bash
python main.py
```

This executes the **entire workflow**:
- ✅ Load raw CSV (195 countries, 35 columns)
- ✅ Clean & validate (handle missing values, format inconsistencies)
- ✅ Engineer 24 features (economic, healthcare, education, demographic, environmental domains)
- ✅ Create stratified target variable
- ✅ Train 3 models with cross-validation
- ✅ Evaluate on held-out test set
- ✅ Save best model + all metrics & visualizations
- ✅ Log execution trace

**Total runtime:** ~2-3 minutes. Output saved to `reports/` and `models/`.

### Step 3: Launch Interactive Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

**Live prediction playground:**
- Adjust GDP, birth rate, healthcare spending, education enrollment with sliders
- See stability prediction update in real time
- Visualize how each feature influences the outcome
- Export predictions as CSV

---

## 🔄 Pipeline Deep Dive

### **Step 1-2: Load & Clean**
- Read raw CSV with 195 countries × 35 socioeconomic indicators
- Drop non-predictive columns (currency codes, city names)
- Standardize number formatting (remove commas, dollar signs, percentage signs)
- Handle missing values with median imputation (fit on train, apply to test)

### **Step 3: Feature Engineering** ⭐ *The Heart of the Work*

24 engineered features across 5 domains:

**💰 Economic (5 features)**
- GDP per Capita
- Economic Stress Index (unemployment × inflation proxy)
- Tax Burden Ratio
- Debt Service Ratio
- Trade Openness

**🏥 Healthcare (4 features)**
- Healthcare Access Gap (out-of-pocket costs / physician availability)
- Maternal Mortality Rate
- Infant Mortality Rate
- Life Expectancy

**📚 Education (4 features)**
- Human Capital Index (literacy × enrollment composite)
- Education Pipeline Ratio (tertiary/primary)
- Workforce Stability (adults with secondary education)
- University Enrollment Rate

**👥 Demographic (5 features)**
- Urbanization Rate
- Population Pressure Index (density relative to arable land)
- Dependency Ratio (children + elderly / working-age)
- Birth Rate
- Growth Rate

**🌱 Environmental (2 features)**
- CO2 per Capita
- Resource-Land Balance (renewable land / total land)

### **Step 4: Target Creation**

Composite stability score from 5 health indicators:
- Life expectancy (weight: 0.3)
- Infant mortality (weight: 0.3)
- Maternal mortality (weight: 0.2)
- Physician access (weight: 0.1)
- GDP per capita (weight: 0.1)

Binned into 3 equal-frequency classes (stratified) to prevent class imbalance.

### **Step 5-7: Train & Validate**

- **80/20 stratified split** by stability class
- **Preprocessing pipeline:** Median imputation → Standard scaling
- **3-fold cross-validation** with macro F1 scoring
- **Each model wrapped as:** `Pipeline([preprocessor, classifier])`
  - Prevents leakage (preprocessor fit only on training data)
  - Ensures consistency at prediction time

### **Step 8-9: Evaluate & Save**

For each model:
- Confusion matrix (normalized)
- Per-class precision, recall, F1
- Feature importance chart
- Cross-validation score distribution

All artifacts persisted to `reports/` and `models/`.

---

## 📦 Dataset

**Source:** [World Data 2023](https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023) (Kaggle)

- **195 countries** | **35 indicators** | **Single year (2023)**
- Coverage: Population, economics, healthcare, education, environment

**Why this dataset is challenging:**
- Only ~63 samples per class → every feature engineering decision matters
- Single-year snapshot → can't track trends
- Missing data in some columns → requires thoughtful imputation

**This constraint is a feature, not a bug.** It forced disciplined feature engineering and regularized models instead of overfitting.

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.12 |
| **ML Framework** | scikit-learn | 1.8 |
| **Boosting** | XGBoost | 3.2 |
| **Data** | pandas / numpy | Latest |
| **Viz** | matplotlib / seaborn | Latest |
| **Dashboard** | Streamlit | Latest |
| **Serialization** | joblib | Latest |

**Intentional choices:**
- ❌ No deep learning (tabular data doesn't need neural networks)
- ❌ No uninterpretable black boxes (all models have clear decision logic)
- ✅ Regularized models (prevent overfitting on small dataset)
- ✅ Production patterns (pipelines, logging, artifact management)

---

## 🎓 Design Decisions Explained

### Why No Neural Networks?

189 samples × 24 features. A neural net would memorize the training set. Logistic Regression with L2 regularization + ensemble methods (Random Forest, XGBoost) are the right tools here.

### Why Feature Engineering Over Complexity?

Raw features are noisy. Composite features (Education Pipeline, Population Pressure Index, Healthcare Access Gap) encode domain knowledge and improve interpretability. I spent 60% of time here, 40% on modeling.

### Why Stratified Splits & Cross-Validation?

With ~63 samples per class, random splits create misleading performance estimates. Stratification ensures each fold has balanced class representation. Cross-validation catches overfitting early.

### Why This Structure?

Designed for **teams**. Each module is testable independently. New data? Drop it in `data/raw/`. Want to add features? Edit `feature_engineering.py`. Need a different model? Modify `model_training.py`. No gatekeeping.

---

## 🔮 Future Improvements

With more time, I'd tackle:

- **Time-series data** (World Bank API, IMF indicators) → Track stability *trends*, not just current snapshots
- **SHAP values** → Per-prediction explanations ("this country is at-risk because of birth rate + low education")
- **Hyperparameter tuning** → Optuna/Bayesian search to squeeze out 2-3% more F1
- **5-tier classification** (Critical / High Risk / Elevated / Watch / Stable) → More nuance for advisory work
- **Fairness audits** → Ensure predictions aren't biased by region or development level
- **API endpoint** → REST service for real-time predictions

---
