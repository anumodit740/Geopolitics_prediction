# 🌍 Global Country Stability Intelligence System

**Predicting country-level stability risk using real-world socioeconomic data.**

I built this project to answer a question I kept running into during my analytics work — *can we actually quantify how "stable" a country is using publicly available indicators?* Turns out, with the right feature engineering and some well-tuned tree models, we can get surprisingly close.

This isn't a toy notebook. It's a fully modular, pipeline-driven ML system designed the way you'd actually build things at a consulting firm or analytics startup. Every module has a clear job, the code is readable, and the whole thing runs end-to-end with a single command.

---

## What does it do?

Takes raw country-level data (GDP, birth rates, healthcare spending, education enrollment, CO2 emissions, etc.) and classifies each country into one of three risk categories:

| Class | Label | What it means |
|-------|-------|---------------|
| 0 | **Stable** | Strong fundamentals across the board |
| 1 | **Watch** | Mixed signals — needs monitoring |
| 2 | **At-Risk** | Significant vulnerability indicators |

The system also ships with a Streamlit dashboard where you can tweak country indicators with sliders and see predictions update in real time.

---

## Why I built it this way

A few things I was intentional about:

- **No deep learning.** This is tabular data with 189 rows. Neural nets would memorize it. I stuck with Logistic Regression, Random Forest, and XGBoost — models you can actually explain in a stakeholder meeting.

- **Feature engineering matters more than model complexity.** I spent more time building meaningful composite features (Healthcare Access Gap, Population Pressure Index, Education Pipeline ratio) than tuning hyperparameters. That's where the real signal lives.

- **Anti-leakage by design.** The stability score that generates the target labels is dropped before training. So are the raw columns used to compute it. The preprocessing pipeline is fitted inside `sklearn.Pipeline` — test data never touches the fit step. I've seen too many Kaggle notebooks leak data through careless preprocessing.

- **Modular, not monolithic.** Each `.py` file does one thing. You can run any module standalone to test it. The pipeline orchestrator (`main.py`) just wires them together.

---

## Results

Here's what the models scored on the held-out test set (38 countries, stratified split):

| Model | Macro F1 | Accuracy | At-Risk Recall |
|-------|----------|----------|----------------|
| **XGBoost** | **0.804** | **0.816** | **92.3%** |
| Random Forest | 0.780 | 0.789 | 92.3% |
| Logistic Regression | 0.780 | 0.789 | 92.3% |

XGBoost won on Macro F1 (which is the primary metric — accuracy alone doesn't cut it when you have 3 balanced classes and the cost of missing an at-risk country is high).

The 92.3% at-risk recall means the model correctly flags 12 out of 13 genuinely at-risk countries. That's the number that matters most from a business perspective.

### Top features driving predictions (XGBoost)

1. Birth Rate
2. Education Pipeline (tertiary/primary enrollment ratio)
3. GDP per Capita
4. Population Pressure Index
5. Healthcare Access Gap

These make intuitive sense — countries with high birth rates, poor education pipelines, and low GDP per capita tend to be less stable. The model isn't picking up on noise.

---

## Project structure

```
├── config.py                       # All paths, params, column maps in one place
├── main.py                         # Run this → full pipeline end-to-end
├── app.py                          # Streamlit dashboard
│
├── src/
│   ├── data_preprocessing.py       # Load, clean, split, build sklearn pipelines
│   ├── feature_engineering.py      # 24 engineered features + proxy target creation
│   ├── model_training.py           # Train 3 models, cross-validate, save best
│   └── evaluation.py               # Metrics, confusion matrices, feature importance
│
├── data/
│   ├── raw/                        # world-data-2023.csv (195 countries, 35 cols)
│   └── processed/                  # Cleaned + engineered + model-ready CSVs
│
├── models/                         # Saved pipelines (.joblib)
├── reports/
│   ├── figures/                    # Confusion matrices, feature importance plots
│   └── metrics/                    # Classification reports, comparison CSV
│
├── logs/                           # Pipeline execution logs
└── requirements.txt
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/Geopolitics_prediction.git
cd Geopolitics_prediction
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

This does everything — loads the raw CSV, cleans it, engineers 24 features, creates the target variable, trains 3 models with cross-validation, evaluates on held-out test set, saves the best model, and generates all reports. Takes about 20 seconds on my machine.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. You'll see a dark-themed SaaS-style dashboard where you can adjust sliders for GDP, birth rate, unemployment, etc. and get live predictions.

---

## How the pipeline works

I'll keep this brief since the code is heavily commented, but here's the flow:

**Step 1-2: Load & Clean** → Read the raw CSV, drop ID columns (currency codes, city names — not useful for ML), fix messy number formatting (some columns have commas and dollar signs baked in), create missingness flags for columns with >10% missing.

**Step 3: Feature Engineering** → This is where the real work happens. I create 24 features across 5 domains:
- *Economic*: GDP per capita, economic stress index, tax burden ratio
- *Healthcare*: healthcare access gap (out-of-pocket cost / physicians)
- *Education*: human capital index, education pipeline, workforce stability
- *Demographic*: urbanization rate, population pressure, dependency ratio
- *Environment*: CO2 per capita, resource-land balance, fuel cost pressure

**Step 4: Target Creation** → Build a composite stability score from life expectancy, infant/maternal mortality, physician access, education, and GDP per capita. Bin into 3 equal-frequency classes. Then *drop the score and its raw inputs* before training — this is critical for preventing leakage.

**Step 5-7: Split & Train** → 80/20 stratified split. Each model gets wrapped in an `sklearn.Pipeline([preprocessor, classifier])` so the preprocessor (median imputation + standard scaling) is fitted only on training data. XGBoost gets sample weights for class balance since it doesn't natively support `class_weight`.

**Step 8-9: Evaluate & Save** → Full classification reports, confusion matrices, feature importance charts, a model comparison bar chart, and a plain-English business interpretation. Everything gets saved to `reports/`.

---

## Dataset

[World Data 2023](https://www.kaggle.com/datasets/nelgiriyewithana/countries-of-the-world-2023) from Kaggle — 195 countries, 35 columns covering population, economics, healthcare, education, and environmental indicators. After cleaning (dropping near-empty rows like Vatican City), we're left with 189 countries.

It's not a huge dataset, which is actually part of the challenge. With ~63 samples per class, every engineering decision matters and overfitting is a real risk. That's why I went with regularized models, cross-validation, and kept the feature set interpretable.

---

## Tech stack

- **Python 3.12**
- **scikit-learn 1.8** — preprocessing pipelines, logistic regression, random forest, cross-validation
- **XGBoost 3.2** — primary production model
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — all visualizations
- **Streamlit** — interactive dashboard
- **joblib** — model serialization

No deep learning. No unnecessary complexity.

---

## What I'd improve with more time

A few things I'd tackle if I were extending this:

- **More data sources.** This uses a single-year snapshot. Adding time-series data (World Bank API, IMF indicators) would let us track stability *trends*, not just current state.
- **SHAP values.** Right now I use feature importances and coefficients. SHAP would give per-prediction explanations — "this country is predicted at-risk primarily because of its birth rate and low GDP per capita."
- **Hyperparameter tuning.** I used reasonable defaults from experience, but a proper Optuna/Bayesian search could squeeze out another couple points of F1.
- **More classes.** The 3-class setup works, but a 5-tier system (Critical / High Risk / Elevated / Watch / Stable) would be more nuanced for real advisory work.

---

## Who this is for

I designed this with analytics consulting firms in mind — the kind of work you'd do at Fractal, Tiger Analytics, Tredence, or similar shops. The emphasis is on:

- Clean, modular code that a team could maintain
- Business-relevant metrics (not just accuracy)
- Explainable models (try explaining a transformer to a VP of Risk)
- Production patterns (pipelines, artifact persistence, logging)

If you're reviewing this for a DS/ML role — the best place to start is `main.py` (shows the full flow) and then `src/feature_engineering.py` (shows domain thinking).

---

*Built by Anumol — feedback and questions welcome.*
#   G e o p o l i t i c s _ p r e d i c t i o n  
 