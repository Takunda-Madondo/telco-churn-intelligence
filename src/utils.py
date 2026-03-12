"""Utility helpers for logging, paths and quick checks."""

import os
import logging
import pandas as pd

def get_logger(name: str) -> logging.Logger:
    """Return a logger with a consistent format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


# simple project paths

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

def ensure_dirs():
    """Make sure dirs we write to exist."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


# data checks

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]

def validate_schema(df: pd.DataFrame) -> bool:
    """Return False if any expected column is missing (logs the list)."""
    logger = get_logger("utils.validate_schema")
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False
    logger.info("Schema ok.")
    return True


def summarise_dataframe(df: pd.DataFrame) -> None:
    """Show shape, duplicates, nulls and dtypes."""
    print(f"Shape: {df.shape}")
    print(f"Duplicates: {df.duplicated().sum()}")
    nulls = df.isnull().sum()
    print(f"Null counts:\n{nulls[nulls > 0]}")
    print(f"\nDtypes:\n{df.dtypes}")
