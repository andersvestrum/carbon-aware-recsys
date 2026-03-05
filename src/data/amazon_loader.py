"""
Amazon Review Data Loader

Downloads and parses Amazon review data from:
https://amazon-reviews-2023.github.io/data_processing/5core.html

Used as a proxy for purchase behavior — each review represents a
verified purchase with rating as engagement level.

Subcategories loaded: Electronics, Home_and_Kitchen, Sports_and_Outdoors
"""

import os
from pathlib import Path

import gdown
import pandas as pd

# Project root: carbon-aware-recsys/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_AMAZON_DIR = PROJECT_ROOT / "data" / "raw" / "amazon"

# Google Drive file IDs extracted from share links
AMAZON_ELECTRONICS_FILES = {
    "train": {
        "id": "1A81uvtJFqjL8EKdCHEEjTmMnUgGuoitt",
        "filename": "electronics.csv",
    },
    "val": {
        "id": "1z6YJC1KXNFq95pn3qbqkke9P4GGz_2uO",
        "filename": "electronics.csv",
    },
    "test": {
        "id": "1M2CRih0ikrry4qe2pGMldCAS78TwI6gU",
        "filename": "electronics.csv",
    },
}

AMAZON_HAK_FILES = {
    "train": {
        "id": "1qSmXs9xm7iBT9ZVjwg9nzmxteOGgyPye",
        "filename": "home_and_kitchen.csv",
    },
    "val": {
        "id": "1qrmGkQNTDiXI6krj5NsIC7KOEtq94uS2",
        "filename": "home_and_kitchen.csv",
    },
    "test": {
        "id": "1E85L-Kr80aETu4DdUknOrXwVnzXUMCKB",
        "filename": "home_and_kitchen.csv",
    },
}

AMAZON_SAO_FILES = {
    "train": {
        "id": "1lOKA-lZSsH5SiB1Ud19rs5sCF_7TPQZn",
        "filename": "sports_and_outdoors.csv",
    },
    "val": {
        "id": "1S6XBAJ3wmudCrn-Sja8j22SoJiOw5co-",
        "filename": "sports_and_outdoors.csv",
    },
    "test": {
        "id": "1-uJeiwZeXtVCLZfI6WKQtc3LURjVVWNc",
        "filename": "sports_and_outdoors.csv",
    },
}



def _download_category(name: str, file_config: dict, force: bool = False) -> dict[str, Path]:
    """Download all splits for a single Amazon category.

    Files are saved to: data/raw/amazon/{split}/{filename}
    E.g. data/raw/amazon/train/electronics.csv

    Args:
        name: Category display name (for logging).
        file_config: Dict with 'train'/'val'/'test' keys, each having 'id' and 'filename'.
        force: If True, re-download even if files already exist.

    Returns:
        Dict mapping split name ('train', 'val', 'test') to file paths.
    """
    print(f"Downloading Amazon {name} data ...")
    paths = {}
    for split, info in file_config.items():
        output_path = RAW_AMAZON_DIR / split / info["filename"]
        os.makedirs(output_path.parent, exist_ok=True)

        if output_path.exists() and not force:
            print(f"    Already exists: {split}/{info['filename']}")
        else:
            url = f"https://drive.google.com/uc?id={info['id']}"
            print(f"    Downloading: {split}/{info['filename']} ...")
            gdown.download(url, str(output_path), quiet=False)
            print(f"    Saved to: {output_path}")

        paths[split] = output_path
    print("Done.\n")
    return paths


def download_amazon_sao_data(force: bool = False) -> dict[str, Path]:
    """Download Amazon Sports_and_Outdoors splits to data/raw/amazon/{split}/."""
    return _download_category("Sports_and_Outdoors", AMAZON_SAO_FILES, force)


def download_amazon_electronics_data(force: bool = False) -> dict[str, Path]:
    """Download Amazon Electronics splits to data/raw/amazon/{split}/."""
    return _download_category("Electronics", AMAZON_ELECTRONICS_FILES, force)


def download_amazon_hak_data(force: bool = False) -> dict[str, Path]:
    """Download Amazon Home_and_Kitchen splits to data/raw/amazon/{split}/."""
    return _download_category("Home_and_Kitchen", AMAZON_HAK_FILES, force)


def download_all_amazon_data(force: bool = False) -> dict[str, dict[str, Path]]:
    """Download all Amazon category files to data/raw/amazon/{split}/.

    Args:
        force: If True, re-download even if files already exist.

    Returns:
        Dict mapping category ('electronics', 'hak', 'sao') to split paths.
    """
    return {
        "electronics": download_amazon_electronics_data(force=force),
        "hak": download_amazon_hak_data(force=force),
        "sao": download_amazon_sao_data(force=force),
    }


def _load_splits(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Load a set of split paths into DataFrames.
    
    Automatically detects gzip-compressed files.
    """
    dataframes = {}
    for split, path in paths.items():
        print(f"  Loading {split}: {path.name} ...")
        # Detect gzip by reading magic bytes (1f 8b)
        with open(path, "rb") as f:
            is_gzip = f.read(2) == b"\x1f\x8b"
        compression = "gzip" if is_gzip else "infer"
        df = pd.read_csv(path, compression=compression)
        print(f"    {split}: {len(df):,} rows, {len(df.columns)} columns")
        dataframes[split] = df
    return dataframes


def load_amazon_sao_data(force_download: bool = False) -> dict[str, pd.DataFrame]:
    """Download (if needed) and load Amazon SaO data as DataFrames."""
    paths = download_amazon_sao_data(force=force_download)
    return _load_splits(paths)


def load_amazon_electronics_data(force_download: bool = False) -> dict[str, pd.DataFrame]:
    """Download (if needed) and load Amazon Electronics data as DataFrames."""
    paths = download_amazon_electronics_data(force=force_download)
    return _load_splits(paths)


def load_amazon_hak_data(force_download: bool = False) -> dict[str, pd.DataFrame]:
    """Download (if needed) and load Amazon Home_and_Kitchen data as DataFrames."""
    paths = download_amazon_hak_data(force=force_download)
    return _load_splits(paths)


def load_all_amazon_data(force_download: bool = False) -> dict[str, dict[str, pd.DataFrame]]:
    """Download (if needed) and load all Amazon category data as DataFrames.

    Returns:
        Dict mapping category name to dict of split DataFrames.
        E.g. {'electronics': {'train': df, 'val': df, 'test': df}, ...}
    """
    return {
        "electronics": load_amazon_electronics_data(force_download),
        "hak": load_amazon_hak_data(force_download),
        "sao": load_amazon_sao_data(force_download),
    }


if __name__ == "__main__":
    all_data = load_all_amazon_data()
    for category, splits in all_data.items():
        print(f"\n{'='*50}")
        print(f"Category: {category}")
        print(f"{'='*50}")
        for split, df in splits.items():
            print(f"\n--- {split} ---")
            print(df.head())
