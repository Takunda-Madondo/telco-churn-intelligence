"""Add predictor features to the cleaned dataset.

Reads 02_cleaned.csv and writes 03_features.csv.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import get_logger, PROCESSED_DIR, ensure_dirs

logger = get_logger("etl.features")

# columns used by engagement score
OPTIONAL_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]



def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple RFM-like proxies based on tenure, services and charges."""
    df = df.copy()
    max_tenure = df["tenure"].max()
    df["rfm_recency"] = (max_tenure - df["tenure"]) / max_tenure
    service_cols = [
        "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["rfm_frequency"] = df[service_cols].sum(axis=1)
    df["rfm_monetary"] = df["TotalCharges"]
    logger.info("Added RFM features.")
    return df



def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """Gauge how many optional services a customer uses."""
    df = df.copy()
    df["engagement_score"] = df[OPTIONAL_SERVICES].sum(axis=1) / len(OPTIONAL_SERVICES)
    df["has_streaming"] = ((df["StreamingTV"] == 1) | (df["StreamingMovies"] == 1)).astype(int)
    df["has_support_services"] = ((df["OnlineSecurity"] == 1) | (df["TechSupport"] == 1)).astype(int)
    logger.info("Added engagement score.")
    return df



CONTRACT_RISK_MAP = {
    "Month-to-month": 3,   # Highest churn risk
    "One year": 2,
    "Two year": 1,         # Lowest churn risk
}

PAYMENT_RISK_MAP = {
    "Electronic check": 3,        # Known high churn method
    "Mailed check": 2,
    "Bank transfer (automatic)": 1,
    "Credit card (automatic)": 1,
}

def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple numeric weights from contract and payment fields."""
    df = df.copy()
    df["contract_risk"] = df["Contract"].map(CONTRACT_RISK_MAP)
    df["payment_risk"] = df["PaymentMethod"].map(PAYMENT_RISK_MAP)
    df["charges_per_month_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    logger.info("Added risk features.")
    return df



def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    """Simple labels based on tenure months."""
    df = df.copy()
    df["tenure_band"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 36, 72],
        labels=["New", "Developing", "Established"],
        include_lowest=True,
    )
    logger.info("Added tenure band.")
    return df



def add_risk_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Quick rule-based bucket for dashboard filters."""
    df = df.copy()
    conditions = [
        (df["contract_risk"] == 3) & (df["payment_risk"] == 3) & (df["engagement_score"] < 0.3),
        (df["contract_risk"] == 1) & (df["payment_risk"] == 1) & (df["engagement_score"] >= 0.5),
    ]
    choices = ["High Risk", "Low Risk"]
    df["risk_segment"] = np.select(conditions, choices, default="Medium Risk")
    logger.info(f"Risk segments:\n{df['risk_segment'].value_counts()}")
    return df



def run_feature_engineering() -> pd.DataFrame:
    ensure_dirs()

    in_path = os.path.join(PROCESSED_DIR, "02_cleaned.csv")
    if not os.path.exists(in_path):
        raise FileNotFoundError("Run clean first.")
    df = pd.read_csv(in_path)
    logger.info(f"Loaded {df.shape}")

    df = add_rfm_features(df)
    df = add_engagement_score(df)
    df = add_risk_features(df)
    df = add_tenure_band(df)
    df = add_risk_segment(df)

    out_path = os.path.join(PROCESSED_DIR, "03_features.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}, shape {df.shape}")
    return df


if __name__ == "__main__":
    run_feature_engineering()
