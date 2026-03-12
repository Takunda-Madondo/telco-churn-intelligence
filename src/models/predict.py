"""Helpers to load a saved churn model and score data.

Example:
    predictor = ChurnPredictor('xgboost')
    results = predictor.predict(df)
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import get_logger, MODELS_DIR, PROCESSED_DIR
from src.models.train import encode_categoricals

logger = get_logger("models.predict")


class ChurnPredictor:
    """Simple interface for scoring with a saved churn model."""

    def __init__(self, model_name: str = "xgboost"):
        """model_name must be 'xgboost', 'random_forest' or 'logistic_regression'."""
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
        self.scaler = (
            joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
            if model_name == "logistic_regression" else None
        )
        # load threshold for xgboost, otherwise default 0.5
        threshold_path = os.path.join(MODELS_DIR, "optimal_threshold.pkl")
        if model_name == "xgboost" and os.path.exists(threshold_path):
            self.threshold = joblib.load(threshold_path)
            logger.info(f"Loaded model: {model_name}  |  threshold: {self.threshold:.4f}")
        else:
            self.threshold = 0.5
            logger.info(f"Loaded model: {model_name}  |  threshold: 0.5 (default)")

    def _load_model(self, name: str):
        model_files = {
            "xgboost": "xgboost.pkl",
            "random_forest": "random_forest.pkl",
            "logistic_regression": "logistic_regression.pkl",
        }
        if name not in model_files:
            raise ValueError(f"Unknown model: {name}")
        path = os.path.join(MODELS_DIR, model_files[name])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        return joblib.load(path)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Take a feature table and add scores and labels.

        The XGBoost predictor will use its tuned threshold; others default to 0.5.
        """
        df_encoded = encode_categoricals(df.copy())
        X = df_encoded[self.feature_cols]
        if self.scaler:
            X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)[:, 1]
        pred = (proba >= self.threshold).astype(int)
        result = df.copy()
        result["churn_probability"] = proba.round(4)
        result["churn_prediction"] = pred
        result["threshold_used"] = round(self.threshold, 4)
        result["risk_label"] = pd.cut(proba, bins=[0, 0.3, 0.6, 1], labels=["Low", "Medium", "High"])

        logger.info(f"Scored {len(result):,} customers.")
        logger.info(f"Threshold: {self.threshold:.4f}  |  "
                    f"Predicted churn rate: {pred.mean():.2%}")

        return result


def score_all_customers(model_name: str = "xgboost") -> pd.DataFrame:
    """Load processed features and score all customers. Used for dashboard."""
    path = os.path.join(PROCESSED_DIR, "03_features.csv")
    df = pd.read_csv(path)
    predictor = ChurnPredictor(model_name=model_name)
    return predictor.predict(df)


if __name__ == "__main__":
    results = score_all_customers()
    print(results[["customerID", "churn_probability", "churn_prediction", "risk_label"]].head(10))
    print(f"\nRisk label distribution:\n{results['risk_label'].value_counts()}")
