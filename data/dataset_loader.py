import pandas as pd
from pathlib import Path


def load_labels(csv_path: Path):
    """
    Load DAiSEE labels CSV and normalize column names / clip ids.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["ClipID"] = df["ClipID"].str.replace(".avi", "", regex=False)
    return df
