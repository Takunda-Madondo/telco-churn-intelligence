"""Train churn models and record their results.

Fits a basic logistic model, a random forest, then tunes an XGBoost
classifier. Outputs models, selected features, thresholds and metrics
to the models/ folder.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import get_logger, PROCESSED_DIR, MODELS_DIR, ensure_dirs

from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, precision_recall_curve
)
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

logger = get_logger("models.train")

# which columns we feed into models
FEATURE_COLS = [
    "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "rfm_recency", "rfm_frequency", "rfm_monetary",
    "engagement_score", "has_streaming", "has_support_services",
    "contract_risk", "payment_risk", "charges_per_month_ratio",
]
TARGET_COL = "Churn"
CATEGORICAL_COLS = ["InternetService", "Contract", "PaymentMethod"]

# drop features whose normalised importance is below this
IMPORTANCE_THRESHOLD = 0.005



def load_features() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "03_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run features.py first.")
    return pd.read_csv(path)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the handful of text columns into numeric codes."""
    df = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def prepare_data(df: pd.DataFrame):
    """Encode cats and split into X, y, feature list."""
    df = encode_categoricals(df)
    all_features = FEATURE_COLS + CATEGORICAL_COLS
    X = df[all_features]
    y = df[TARGET_COL]
    return X, y, all_features



def evaluate_model(model, X_test, y_test, model_name: str,
                   threshold: float = 0.5) -> dict:
    """Run a model on test data and log basic stats at one threshold."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "threshold": round(threshold, 3),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "f1_churn": round(f1_score(y_test, y_pred), 4),
        "precision_churn": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall_churn": round(recall_score(y_test, y_pred), 4),
    }

    logger.info("\n" + "="*52)
    logger.info(f"  {model_name}  (threshold={threshold:.3f})")
    logger.info("="*52)
    logger.info(f"  ROC-AUC   : {metrics['roc_auc']}")
    logger.info(f"  F1        : {metrics['f1_churn']}")
    logger.info(f"  Precision : {metrics['precision_churn']}")
    logger.info(f"  Recall    : {metrics['recall_churn']}")
    logger.info("\n" + classification_report(y_test, y_pred,
                                       target_names=['Retained', 'Churned']))

    return metrics




def find_optimal_threshold(model, X_val, y_val) -> float:
    """Pick the cutoff on validation data that gives highest F1."""
    y_prob = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = np.where((precision + recall) == 0, 0,
                         2 * precision * recall / (precision + recall))
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    logger.info(f"Optimal threshold: {best_threshold:.4f}")
    return best_threshold



def filter_low_importance_features(model, feature_cols: list,
                                   threshold: float = IMPORTANCE_THRESHOLD) -> list:
    """Keep only features whose normalised importance is above threshold."""
    importances = model.feature_importances_
    normalised = importances / importances.sum()
    dropped = [f for f, imp in zip(feature_cols, normalised) if imp < threshold]
    kept = [f for f, imp in zip(feature_cols, normalised) if imp >= threshold]
    if dropped:
        logger.info(f"Dropping {dropped}")
    else:
        logger.info("No features dropped.")
    return kept



def nested_cross_validation(model, X, y, cv_outer: int = 5) -> dict:
    """Quick outer-loop CV to estimate how a tuned model might behave.

    Returns means and stds for ROC-AUC and F1.
    """
    cv = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
    f1_scores  = cross_val_score(model, X, y, cv=cv,
                                 scoring="f1", n_jobs=-1)
    results = {
        "cv_roc_auc_mean": round(float(auc_scores.mean()), 4),
        "cv_roc_auc_std":  round(float(auc_scores.std()),  4),
        "cv_f1_mean":      round(float(f1_scores.mean()),  4),
        "cv_f1_std":       round(float(f1_scores.std()),   4),
    }
    logger.info(f"Nested CV ROC-AUC: {results['cv_roc_auc_mean']} ± {results['cv_roc_auc_std']}")
    logger.info(f"Nested CV F1     : {results['cv_f1_mean']} ± {results['cv_f1_std']}")
    return results




def run_training():
    ensure_dirs()

    df = load_features()
    X, y, feature_cols = prepare_data(df)

    logger.info(f"Dataset  : {X.shape[0]:,} samples  |  {X.shape[1]} features")
    logger.info(f"Churn rate: {y.mean():.2%}")

    # split data (60/20/20 stratified)
    # 60% train  |  20% validation (threshold tuning)  |  20% test (final eval)
    # All splits are stratified to preserve the ~26.5% churn ratio.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    all_metrics = []

    # ── 1. Logistic Regression — baseline ────────────────────────────────────
    logger.info("\nTraining Logistic Regression...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    metrics_lr = evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")
    all_metrics.append(metrics_lr)
    joblib.dump(lr,     os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # ── 2. Random Forest — ensemble baseline ─────────────────────────────────
    logger.info("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    metrics_rf = evaluate_model(rf, X_test, y_test, "Random Forest")
    all_metrics.append(metrics_rf)
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))

    # ── 3. XGBoost — hyperparameter tuning ───────────────────────────────────
    logger.info("\nTuning XGBoost via RandomizedSearchCV...")

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos

    # Search space — covers the main regularisation and tree structure knobs.
    # RandomizedSearchCV samples n_iter combinations rather than exhaustively
    # trying all combinations, making it practical for a dataset of this size.
    param_dist = {
        "n_estimators":      randint(200, 600),
        "max_depth":         randint(3, 8),
        "learning_rate":     uniform(0.01, 0.2),
        "subsample":         uniform(0.6, 0.4),       # 0.6 – 1.0
        "colsample_bytree":  uniform(0.6, 0.4),       # 0.6 – 1.0
        "min_child_weight":  randint(1, 10),
        "gamma":             uniform(0, 0.5),
        "reg_alpha":         uniform(0, 1.0),          # L1 regularisation
        "reg_lambda":        uniform(0.5, 2.0),        # L2 regularisation
    }

    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=50,               # 50 random combinations
        scoring="roc_auc",       # optimise for discrimination, not accuracy
        cv=cv_inner,
        refit=True,              # refit best params on full training set
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_params = search.best_params_
    logger.info(f"Best params: {best_params}")
    logger.info(f"Best CV ROC-AUC (inner): {search.best_score_:.4f}")

    xgb_tuned = search.best_estimator_

    # ── 3a. Feature importance filtering ─────────────────────────────────────
    logger.info("\nChecking feature importance...")
    kept_features = filter_low_importance_features(xgb_tuned, feature_cols)

    # If any features were dropped, refit the tuned model on the reduced set
    if len(kept_features) < len(feature_cols):
        logger.info(f"Refitting on {len(kept_features)} features...")
        xgb_tuned.fit(X_train[kept_features], y_train)
        X_val_final  = X_val[kept_features]
        X_test_final = X_test[kept_features]
    else:
        X_val_final  = X_val
        X_test_final = X_test

    # ── 3b. Threshold optimisation ────────────────────────────────────────────
    logger.info("\nOptimising decision threshold on validation set...")
    optimal_threshold = find_optimal_threshold(xgb_tuned, X_val_final, y_val)

    # ── 3c. Final evaluation at optimal threshold ─────────────────────────────
    metrics_xgb = evaluate_model(
        xgb_tuned, X_test_final, y_test,
        "XGBoost (tuned)", threshold=optimal_threshold
    )

    # Also record what performance looked like at 0.5 for comparison
    metrics_xgb_default = evaluate_model(
        xgb_tuned, X_test_final, y_test,
        "XGBoost (tuned, threshold=0.5)", threshold=0.5
    )

    all_metrics.append(metrics_xgb)
    all_metrics.append(metrics_xgb_default)

    # ── 3d. Nested cross-validation ───────────────────────────────────────────
    logger.info("\nRunning nested cross-validation...")
    cv_results = nested_cross_validation(xgb_tuned, X[kept_features], y)
    metrics_xgb.update(cv_results)

    # ── Save artefacts ────────────────────────────────────────────────────────
    joblib.dump(xgb_tuned,         os.path.join(MODELS_DIR, "xgboost.pkl"))
    joblib.dump(kept_features,     os.path.join(MODELS_DIR, "feature_cols.pkl"))
    joblib.dump(optimal_threshold, os.path.join(MODELS_DIR, "optimal_threshold.pkl"))

    # Save best hyperparameters for documentation
    with open(os.path.join(MODELS_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # Save full metrics
    metrics_path = os.path.join(MODELS_DIR, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\nAll artefacts saved to: {MODELS_DIR}")

    # Print summary table
    summary = pd.DataFrame(all_metrics)[
        ["model", "threshold", "roc_auc", "f1_churn", "precision_churn", "recall_churn"]
    ]
    print("\n" + "="*75)
    print("FINAL MODEL COMPARISON")
    print("="*75)
    print(summary.to_string(index=False))
    print("="*75)

    if "cv_roc_auc_mean" in metrics_xgb:
        print(f"\nXGBoost nested CV  →  "
              f"ROC-AUC: {metrics_xgb['cv_roc_auc_mean']} ± {metrics_xgb['cv_roc_auc_std']}  |  "
              f"F1: {metrics_xgb['cv_f1_mean']} ± {metrics_xgb['cv_f1_std']}")

    return pd.DataFrame(all_metrics)


if __name__ == "__main__":
    run_training()
