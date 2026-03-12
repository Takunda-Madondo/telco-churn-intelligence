# Telco Customer Churn Intelligence

A full end-to-end data science portfolio project covering ETL pipeline design, exploratory data analysis, churn prediction modelling, and an interactive marketing dashboard — built entirely with free, open-source tools and deployed on Streamlit Community Cloud.

---

## Business Problem

Telecommunications companies lose significant revenue to customer churn. Acquiring a new customer costs five to ten times more than retaining an existing one, making early identification of at-risk customers a high-value problem. This project builds a repeatable, production-style pipeline that:

- Identifies high-risk customers before they leave
- Quantifies the drivers of churn using interpretable ML (SHAP)
- Surfaces actionable customer segments for marketing teams through an interactive dashboard

---

## Dataset

**Telco Customer Churn Dataset**  
Source: [IBM Sample Data — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset covers **7,043 customers** across **21 features** including customer demographics, subscribed services, contract and billing details, and a binary churn label.

### Dataset Inconsistencies & Cleaning Decisions

During EDA and the ETL pipeline, two data quality issues were identified and handled explicitly.

**1. TotalCharges loaded as object type**

The `TotalCharges` column is numeric in nature but is read as `object` (string) by pandas. The cause is a small number of rows containing blank whitespace strings (`" "`) instead of a numeric value. Running `pd.to_numeric()` without stripping whitespace first causes the entire column to silently remain as a string, which would corrupt any downstream feature engineering or modelling that uses this column.

Fix applied in `src/etl/clean.py`:
```python
df['TotalCharges'] = df['TotalCharges'].str.strip()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
```

**2. TotalCharges nulls — 11 rows with tenure = 0**

After the coercion above, 11 rows produce `NaN` for `TotalCharges`. Inspecting these rows reveals they all have `tenure = 0`, meaning these are brand-new customers who have not yet received a bill. Dropping them would lose valid customer records. Imputing with the global mean would be misleading since their actual expected charge is their first month's fee.

Fix applied in `src/etl/clean.py`:
```python
null_mask = df['TotalCharges'].isnull()
df.loc[null_mask, 'TotalCharges'] = df.loc[null_mask, 'MonthlyCharges']
```

**3. Service columns with three-value encoding**

Several service columns (`OnlineSecurity`, `TechSupport`, `StreamingTV`, etc.) use three possible values: `"Yes"`, `"No"`, and `"No internet service"` (or `"No phone service"`). The third value is semantically equivalent to `"No"` — the customer simply does not have the prerequisite service. Treating it as a separate category would create a spurious third class that carries no additional signal.

Fix applied in `src/etl/clean.py`:
```python
df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
```

**4. Class imbalance**

The dataset has a churn rate of approximately **26.5%**, meaning roughly 1 in 4 customers churned. This is a moderately imbalanced binary classification problem. A naive model that always predicts "no churn" would achieve ~73.5% accuracy while being entirely useless. This is handled in modelling using `class_weight='balanced'` for scikit-learn models and `scale_pos_weight` for XGBoost, and evaluated using ROC-AUC and F1 on the churn class rather than accuracy.

---

## Project Structure

```
telco-churn-intelligence/
│
├── data/
│   ├── raw/                        # Original CSV — never modified, not committed to git
│   └── processed/                  # Pipeline checkpoints (01_ingested, 02_cleaned, 03_features)
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── etl/
│   │   ├── ingest.py               # Load raw data, validate schema
│   │   ├── clean.py                # Handle nulls, type fixes, encoding
│   │   └── features.py             # RFM proxies, engagement scores, risk segments
│   ├── models/
│   │   ├── train.py                # Train, evaluate, and save all models
│   │   └── predict.py              # ChurnPredictor class for scoring
│   └── utils.py                    # Shared paths, logger, schema validation
│
├── models/                         # Saved model artifacts (.pkl) — not committed to git
├── app/
│   └── streamlit_app.py            # Interactive dashboard
│
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/telco-churn-intelligence.git
cd telco-churn-intelligence

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# Place WA_Fn-UseC_-Telco-Customer-Churn.csv into data/raw/

# 5. Run the ETL pipeline (each step writes a checkpoint to data/processed/)
python src/etl/ingest.py
python src/etl/clean.py
python src/etl/features.py

# 6. Train models
python src/models/train.py

# 7. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_EDA.ipynb` | Univariate distributions, churn correlations, bivariate analysis, correlation matrix. Every finding is documented with markdown explanations. |
| `02_feature_engineering.ipynb` | Walks through each engineered feature with before/after distributions and validation against actual churn rates. |
| `03_modeling.ipynb` | Full modelling pipeline: baseline → Logistic Regression → Random Forest → XGBoost. Includes ROC curves, confusion matrices, cross-validation, and SHAP analysis. |

---

## Feature Engineering

All features are engineered in `src/etl/features.py` and imported by the notebooks — no logic is duplicated.

| Feature | Source Columns | Description |
|---|---|---|
| `rfm_recency` | `tenure` | Inverted, normalised tenure. High value = newer customer = higher churn risk. |
| `rfm_frequency` | service columns | Count of active services subscribed. Deeper engagement = lower risk. |
| `rfm_monetary` | `TotalCharges` | Lifetime value proxy. |
| `engagement_score` | 6 optional services | Proportion of optional services adopted (0–1). Low score correlates with churn. |
| `has_streaming` | `StreamingTV`, `StreamingMovies` | Binary: customer has any streaming service. |
| `has_support_services` | `OnlineSecurity`, `TechSupport` | Binary: customer has any protection/support service. |
| `contract_risk` | `Contract` | Ordinal risk weight: Month-to-month=3, One year=2, Two year=1. |
| `payment_risk` | `PaymentMethod` | Ordinal risk weight: Electronic check=3, Mailed check=2, Automatic=1. |
| `charges_per_month_ratio` | `MonthlyCharges`, `TotalCharges` | Ratio of monthly to total charges. High = early-stage customer. |
| `tenure_band` | `tenure` | Lifecycle stage: New (<12m), Developing (12–36m), Established (>36m). |
| `risk_segment` | `contract_risk`, `payment_risk`, `engagement_score` | Rule-based segment: High / Medium / Low. Used in dashboard. |

---

## Modelling

### Training Approach

Three models were trained in order of increasing complexity, following a deliberate baseline-first strategy. The dataset was split into three stratified sets — **60% train, 20% validation** (threshold optimisation only), **20% test** (final evaluation only) — ensuring no data leakage between the tuning and evaluation steps.

Class imbalance (~26.5% churn) was addressed using `class_weight='balanced'` for scikit-learn models and `scale_pos_weight` for XGBoost. Accuracy was excluded as a primary metric — a model predicting "no churn" for every customer would score 73.5% accuracy while being entirely useless. Instead, the following metrics were used:

- **ROC-AUC** — how well the model separates churners from non-churners across all thresholds
- **Precision** — of all customers flagged as churning, how many actually did
- **Recall** — of all customers who actually churned, how many were caught
- **F1 Score** — harmonic mean of precision and recall; the primary selection metric

### Models Trained

**1. Logistic Regression (baseline)**

Trained on scaled features using `StandardScaler`. Serves as the interpretable baseline. Notably achieved slightly higher recall and ROC-AUC than the other models, but lower precision — meaning it catches more churners but also generates more false positives, which is costly in a campaign context.

**2. Random Forest**

An ensemble of 200 decision trees with `max_depth=10`. Robust to the multicollinearity between `tenure` and `TotalCharges` (r ≈ 0.83), which affects Logistic Regression more. Provides built-in feature importance as a secondary interpretability tool.

**3. XGBoost — with four improvement steps**

Gradient boosted trees with `scale_pos_weight` for class imbalance, taken through the following steps:

**a. Hyperparameter tuning via `RandomizedSearchCV`** — 50 parameter combinations sampled from continuous distributions covering `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, and `reg_lambda`. Optimised for ROC-AUC using 5-fold inner cross-validation. Random search was chosen over grid search because the XGBoost parameter space is too large for exhaustive enumeration at practical compute cost.

**b. Feature importance filtering** — features with normalised gain below `0.005` were dropped and the model was refit on the reduced set, reducing noise without meaningful loss of signal.

**c. Classification threshold adjustment** — the default threshold of 0.5 was replaced with an optimised threshold of **0.614**, found by searching the precision-recall curve on the validation set for the F1-maximising cutoff. Raising the threshold reduced false positives — improving precision — while still identifying a strong portion of churners. The threshold is saved to `models/optimal_threshold.pkl` and applied automatically at inference time.

**d. Nested cross-validation** — 5-fold stratified CV confirmed the model generalised consistently across folds, with low variance indicating stable, non-overfit performance.

### Results

| Model | ROC-AUC | F1 (Churn) | Precision | Recall | Threshold |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | 0.500 |
| Random Forest | — | — | — | — | 0.500 |
| XGBoost (default threshold) | — | — | — | — | 0.500 |
| **XGBoost (optimised threshold)** | — | — | — | — | **0.614** |

**XGBoost nested CV:** ROC-AUC `0.8475 ± 0.0127`  |  F1 `0.6307 ± 0.0170`

*ROC-AUC, Precision, Recall, and F1 figures for individual models populate after running `python src/models/train.py`. CV results above reflect actual training runs.*

### Final Model Selection

The **tuned XGBoost model at threshold 0.614** was selected as the final model because it achieved:

- The **highest F1 score** — best overall balance between identifying churners and avoiding false alarms
- The **highest precision** — churn predictions are more reliable, making retention campaigns more cost-effective
- A **strong ROC-AUC** demonstrating good overall discrimination ability
- **Stable cross-validation performance** (ROC-AUC 0.8475 ± 0.0127, F1 0.6307 ± 0.0170) confirming generalisation to unseen data

While Logistic Regression achieved slightly higher recall and ROC-AUC, XGBoost provided the best overall balance between catching churners and minimising false positives — making it the most suitable model for practical business use where campaign budgets are finite and targeting accuracy matters.

### Top SHAP Feature Drivers (XGBoost)

| Rank | Feature | Direction |
|---|---|---|
| 1 | `contract_risk` | Month-to-month contract = strong positive churn signal |
| 2 | `tenure` / `rfm_recency` | Newer customers churn at significantly higher rates |
| 3 | `payment_risk` | Electronic check payment = elevated churn probability |
| 4 | `engagement_score` | Low service adoption = easy to leave |
| 5 | `MonthlyCharges` | Higher monthly cost amplifies churn risk |

**Business interpretation:** The highest-risk customer profile is a new customer on a month-to-month contract, paying by electronic check, with few optional services and a high monthly bill. This segment should be the primary target for contract upgrade campaigns and service bundle promotions.

---

## Dashboard

Built with Streamlit. Three views:

- **Overview** — Churn rate KPIs, risk segment distribution, churn by contract type and tenure band
- **Customer Explorer** — Filter by risk label, contract, tenure band, and minimum churn probability. Scatter plot and full customer table with CSV download.
- **Model Insights** — Model comparison table, churn probability distributions by actual status, SHAP feature importance ranking, and marketing action recommendations.

Live demo: *[Add Streamlit Community Cloud link after deployment]*

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data processing | pandas, numpy |
| Machine learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Hosting | Streamlit Community Cloud (free tier) |
| Version control | Git + GitHub |

---

## Author

**Taku (Takunda Madondo)** — Data Scientist & Software Engineer  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
