#!/usr/bin/env python
"""
Merge category-sharded Amazon PCF prediction CSVs into one combined file.

Example:
    python scripts/11_merge_amazon_pcf_shards.py \
      --inputs data/processed/carbon/shards/electronics_pcf_predictions.csv \
               data/processed/carbon/shards/home_and_kitchen_pcf_predictions.csv \
               data/processed/carbon/shards/sports_and_outdoors_pcf_predictions.csv \
      --output data/processed/carbon/amazon_pcf_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded Amazon PCF prediction CSVs.")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Shard CSVs produced by scripts/predict_carbon.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/carbon/amazon_pcf_predictions.csv"),
        help="Merged Amazon PCF prediction CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames: list[pd.DataFrame] = []
    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing shard input: {path}")
        df = pd.read_csv(path)
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("No non-empty shard CSVs were provided.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    if "source_category" in merged.columns:
        merged = merged.sort_values(["source_category", "parent_asin"], kind="stable")
    elif "parent_asin" in merged.columns:
        merged = merged.sort_values(["parent_asin"], kind="stable")

    if "parent_asin" in merged.columns and merged["parent_asin"].duplicated().any():
        dupes = merged.loc[merged["parent_asin"].duplicated(), "parent_asin"].head(10).tolist()
        raise ValueError(
            "Duplicate parent_asin values encountered while merging shards. "
            f"Examples: {dupes}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Saved merged Amazon PCF predictions: {args.output}")
    print(f"Rows: {len(merged):,}")


if __name__ == "__main__":
    main()
