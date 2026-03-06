"""
Carbon Catalogue Data Loader

Downloads and parses The Carbon Catalogue product-level data from:
https://www.kaggle.com/datasets/jeannettesavage/the-carbon-catalogue-public-database

Reference: The Carbon Catalogue (Nature, 2022)
https://www.nature.com/articles/s41597-022-01178-9
"""

import os
from pathlib import Path

import gdown
import pandas as pd

# Project root: carbon-aware-recsys/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CARBON_DIR = PROJECT_ROOT / "data" / "raw" / "carbon_catalogue"

# ── Google Drive file config ───────────────────────────────────────────
CARBON_CATALOGUE_FILE = {
    "id": "1V43TyWwx1nZm-b9lm6ecGIAhkUFyu53k",
    "filename": "carbon_catalogue.csv",
}


def download_carbon_catalogue(force: bool = False) -> Path:
    """Download The Carbon Catalogue CSV from Google Drive.

    File is saved to: data/raw/carbon_catalogue/carbon_catalogue.csv

    Args:
        force: If True, re-download even if file already exists.

    Returns:
        Path to the downloaded CSV file.
    """
    output_path = RAW_CARBON_DIR / CARBON_CATALOGUE_FILE["filename"]
    os.makedirs(output_path.parent, exist_ok=True)

    if output_path.exists() and not force:
        print(f"  Already exists: {output_path.name}")
    else:
        url = f"https://drive.google.com/uc?id={CARBON_CATALOGUE_FILE['id']}"
        print(f"  Downloading: {output_path.name} ...")
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
        print(f"  Saved to: {output_path}")

    return output_path


def load_carbon_catalogue(force_download: bool = False) -> pd.DataFrame:
    """Download (if needed) and load The Carbon Catalogue as a DataFrame.

    Args:
        force_download: If True, re-download even if file already exists.

    Returns:
        DataFrame with product-level carbon footprint data.
    """
    print("Loading Carbon Catalogue ...")
    path = download_carbon_catalogue(force=force_download)
    df = pd.read_csv(path, encoding="latin-1")
    print(f"  Loaded: {len(df):,} products, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    df = load_carbon_catalogue()
    print(f"\n{'='*50}")
    print("Carbon Catalogue")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\n{df.head()}")
