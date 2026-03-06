"""
Amazon Review Data Loader

Downloads and parses Amazon review data from:
https://amazon-reviews-2023.github.io/data_processing/5core.html

Used as a proxy for purchase behavior — each review represents a
verified purchase with rating as engagement level.

Subcategories loaded: Electronics, Home_and_Kitchen, Sports_and_Outdoors
"""

import gzip
import json
import os
from pathlib import Path

import gdown
import pandas as pd

# Project root: carbon-aware-recsys/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_AMAZON_DIR = PROJECT_ROOT / "data" / "raw" / "amazon"
META_DIR = RAW_AMAZON_DIR / "meta"

# ── Cap on metadata rows (set to None to load all) ────────────────────
MAX_META_ROWS = 50_000  # TODO: remove cap for full runs

# ── Metadata file IDs (json.gz from Google Drive) ──────────────────────
# Provide the Google Drive file IDs for each category's metadata file.
AMAZON_META_FILES = {
    "electronics": {
        "id": "1HIW5G-K0SEtroNBiXXAdt-4fxkgUsCeJ",
        "filename": "meta_electronics.json.gz",
    },
    "home_and_kitchen": {
        "id": "1McSEoW1uxeNu12QWSJmNu8L_P2G8Xeue",
        "filename": "meta_home_and_kitchen.json.gz",
    },
    "sports_and_outdoors": {
        "id": "1UYtERfQHl_sKjzMjIT7hXwUjFPKqo6Qm",
        "filename": "meta_sports_and_outdoors.json.gz",
    },
}

# ── Review split file IDs (CSV from Google Drive) ──────────────────────
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


# ── Metadata (json.gz) download & load ────────────────────────────────
def _download_file(file_id: str, output_path: Path, force: bool = False) -> Path:
    """Download a single file from Google Drive if it doesn't already exist."""
    os.makedirs(output_path.parent, exist_ok=True)
    if output_path.exists() and not force:
        print(f"  Already exists: {output_path.name}")
    else:
        print(f"  Downloading: {output_path.name} ...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(output_path), quiet=False)
    return output_path


def load_meta(category: str, force_download: bool = False) -> pd.DataFrame:
    """Download (if needed) and load a category's metadata.

    Workflow:
    1. If a capped CSV already exists, load it directly (fast).
    2. Otherwise, download the json.gz, parse up to MAX_META_ROWS,
       save a CSV, then delete the json.gz to save disk space.
    """
    csv_path = META_DIR / f"meta_{category}.csv"

    if csv_path.exists() and not force_download:
        print(f"  Loading cached CSV: {csv_path.name}")
        df = pd.read_csv(csv_path)
    else:
        # Download json.gz, parse, save as CSV, then remove json.gz
        info = AMAZON_META_FILES[category]
        gz_path = _download_file(info["id"], META_DIR / info["filename"], force=force_download)
        rows = []
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if MAX_META_ROWS is not None and len(rows) >= MAX_META_ROWS:
                    break
        df = pd.DataFrame(rows)

        os.makedirs(csv_path.parent, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path.name} ({csv_path.stat().st_size / 1e6:.1f} MB)")

        # Remove the large json.gz to save disk space
        gz_path.unlink()
        print(f"  Removed: {gz_path.name}")

    print(f"  {category}: {len(df):,} items, {len(df.columns)} columns")
    return df


def load_all_meta(force_download: bool = False) -> dict[str, pd.DataFrame]:
    """Download (if needed) and load all category metadata as DataFrames."""
    print("Loading Amazon metadata ...")
    return {cat: load_meta(cat, force_download) for cat in AMAZON_META_FILES}


if __name__ == "__main__":
    all_data = load_all_amazon_data()
    for category, splits in all_data.items():
        print(f"\n{'='*50}")
        print(f"Category: {category}")
        print(f"{'='*50}")
        for split, df in splits.items():
            print(f"\n--- {split} ---")
            print(df.head())

    # Load metadata
    all_meta = load_all_meta()
    for category, df in all_meta.items():
        print(f"\n{'='*50}")
        print(f"Metadata: {category}")
        print(f"{'='*50}")
        print(df.head())
