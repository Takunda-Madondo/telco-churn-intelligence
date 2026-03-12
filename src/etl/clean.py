"""Clean the ingested data.

Fix types, encode Yes/No flags and strip stray spaces.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import get_logger, PROCESSED_DIR, ensure_dirs

logger = get_logger("etl.clean")

# simple sets used in cleaning
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
]

# service flags may have extra "No internet service" values
SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines",
]


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Turn TotalCharges into a float and fill the few blanks."""
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    null_mask = df["TotalCharges"].isnull()
    logger.info(f"TotalCharges nulls: {null_mask.sum()}, filling with MonthlyCharges")
    df.loc[null_mask, "TotalCharges"] = df.loc[null_mask, "MonthlyCharges"]
    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Map simple Yes/No columns to 1/0."""
    df = df.copy()
    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
        logger.info(f"Encoded {col}")
    return df


def encode_service_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the various service flags into 0/1, treating missing as 0."""
    df = df.copy()
    for col in SERVICE_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        logger.info(f"Encoded {col}")
    return df


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Trim strings everywhere."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def run_cleaning() -> pd.DataFrame:
    ensure_dirs()

    in_path = os.path.join(PROCESSED_DIR, "01_ingested.csv")
    if not os.path.exists(in_path):
        raise FileNotFoundError("Run ingest first.")
    df = pd.read_csv(in_path)
    logger.info(f"Loaded {df.shape}")

    df = strip_whitespace(df)
    df = fix_total_charges(df)
    df = encode_binary(df)
    df = encode_service_cols(df)

    remaining_nulls = df.isnull().sum().sum()
    logger.info(f"Nulls after cleaning: {remaining_nulls}")
    assert remaining_nulls == 0, "Nulls remain after cleaning!"

    out_path = os.path.join(PROCESSED_DIR, "02_cleaned.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path} shape {df.shape}")
    return df


if __name__ == "__main__":
    run_cleaning()
