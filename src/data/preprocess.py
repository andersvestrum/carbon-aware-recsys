"""
Data Preprocessing

Cleans, merges Amazon interactions with carbon data,
and formats output for RecBole (.inter, .item, .user files).

Step 1 – merge_meta_with_interactions()
    Joins each category's train / val / test interaction CSVs
    with the corresponding metadata CSV on `parent_asin`,
    then saves the merged DataFrames to  data/interim/{split}/{category}.csv
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.data.amazon_loader import (
    RAW_AMAZON_DIR,
    META_DIR,
    load_all_amazon_data,
    load_all_meta,
)

# Project root: carbon-aware-recsys/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

# Metadata columns to keep when merging (drop heavy blob-like fields)
META_COLS_TO_KEEP = [
    "parent_asin",
    "main_category",
    "title",
    "average_rating",
    "rating_number",
    "price",
    "store",
    "categories",
]

# Map short category keys (from amazon_loader) to meta-file keys
CATEGORY_MAP = {
    "electronics": "electronics",
    "hak": "home_and_kitchen",
    "sao": "sports_and_outdoors",
}


def merge_meta_with_interactions(
    force_download: bool = False,
    meta_cols: list[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Merge metadata into every interaction split and save to interim/.

    Args:
        force_download: Passed through to the download helpers.
        meta_cols: Metadata columns to keep (besides parent_asin).
                   Defaults to META_COLS_TO_KEEP.

    Returns:
        Nested dict  {category: {split: merged_df}}
    """
    if meta_cols is None:
        meta_cols = META_COLS_TO_KEEP

    # 1. Load interaction data  {category: {split: df}}
    print("=" * 60)
    print("Loading interaction data …")
    print("=" * 60)
    all_interactions = load_all_amazon_data(force_download=force_download)

    # 2. Load metadata  {category: df}
    print("=" * 60)
    print("Loading metadata …")
    print("=" * 60)
    all_meta = load_all_meta(force_download=force_download)

    merged_all: dict[str, dict[str, pd.DataFrame]] = {}

    for cat_key, meta_key in CATEGORY_MAP.items():
        print(f"\n{'─' * 60}")
        print(f"  Category: {meta_key}")
        print(f"{'─' * 60}")

        meta_df = all_meta[meta_key]

        # Keep only the desired columns that actually exist in this meta file
        available_cols = [c for c in meta_cols if c in meta_df.columns]
        meta_slim = meta_df[available_cols].copy()

        # De-duplicate metadata on parent_asin (keep first occurrence)
        meta_slim = meta_slim.drop_duplicates(subset="parent_asin")

        splits = all_interactions[cat_key]
        merged_splits: dict[str, pd.DataFrame] = {}

        for split_name, inter_df in splits.items():
            merged = inter_df.merge(meta_slim, on="parent_asin", how="inner")

            # Report merge coverage
            n_total = len(merged)
            n_matched = merged["main_category"].notna().sum()
            pct = 100 * n_matched / n_total if n_total else 0
            print(
                f"  {split_name:>5}: {n_total:>10,} rows  |  "
                f"meta matched {n_matched:>10,} ({pct:.1f}%)"
            )

            # Save to data/interim/{split}/{category}.csv  (same layout as raw/)
            out_dir = INTERIM_DIR / split_name
            os.makedirs(out_dir, exist_ok=True)
            out_path = out_dir / f"{meta_key}.csv"
            merged.to_csv(out_path, index=False)
            print(f"         → saved {out_path.relative_to(PROJECT_ROOT)}")

            merged_splits[split_name] = merged

        merged_all[cat_key] = merged_splits

    print(f"\n{'=' * 60}")
    print("All merged files saved to data/interim/")
    print(f"{'=' * 60}")
    return merged_all


if __name__ == "__main__":
    merge_meta_with_interactions()
