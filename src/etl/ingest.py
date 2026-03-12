"""Basic loader for the raw CSV.

Reads the file, checks the columns and dumps a copy under processed/.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils import get_logger, RAW_DATA_PATH, PROCESSED_DIR, validate_schema, ensure_dirs

logger = get_logger("etl.ingest")


def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the original CSV, error out if it's missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data not found at: {path}\n"
            "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "and put it in data/raw/"
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def run_ingestion() -> pd.DataFrame:
    ensure_dirs()

    df = load_raw()
    if not validate_schema(df):
        raise ValueError("Schema failed, check the raw file.")

    churn_rate = df["Churn"].value_counts(normalize=True).get("Yes", 0)
    logger.info(f"Raw churn rate: {churn_rate:.2%}")
    logger.info(f"Null cells: {df.isnull().sum().sum()}")

    out_path = os.path.join(PROCESSED_DIR, "01_ingested.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote {out_path}")

    return df


if __name__ == "__main__":
    run_ingestion()
