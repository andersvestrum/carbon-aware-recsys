#!/usr/bin/env python
"""
Rebuild data/interim/{split}/{category}.csv from raw interaction data,
keeping only rows whose parent_asin has a few-shot LLM PCF prediction,
and attaching pcf + main_category from the Colab predictions.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "amazon"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PCF_PATH = PROJECT_ROOT / "data" / "processed" / "carbon" / "amazon_pcf_predictions.csv"

SPLITS = ["train", "val", "test"]
CATEGORIES = ["electronics", "home_and_kitchen", "sports_and_outdoors"]


def main() -> None:
    print(f"Loading PCF predictions from {PCF_PATH} ...")
    pcf = pd.read_csv(PCF_PATH, usecols=["parent_asin", "pcf", "main_category"], low_memory=False)
    pcf = pcf.drop_duplicates(subset="parent_asin", keep="first")
    print(f"  {len(pcf):,} products with few-shot PCF predictions\n")

    for split in SPLITS:
        for category in CATEGORIES:
            raw_path = RAW_DIR / split / f"{category}.csv"
            out_path = INTERIM_DIR / split / f"{category}.csv"

            df = pd.read_csv(raw_path, compression="gzip", low_memory=False)
            before = len(df)

            df = df[df["parent_asin"].isin(pcf["parent_asin"])].copy()
            df = df.merge(pcf[["parent_asin", "pcf", "main_category"]], on="parent_asin", how="left")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)

            print(
                f"  {split}/{category}: {before:,} -> {len(df):,} rows "
                f"({df['parent_asin'].nunique():,} unique products)"
            )


if __name__ == "__main__":
    main()
