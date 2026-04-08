"""
Build the missing data/interim/train/electronics.csv (and home_and_kitchen)
by joining the raw train CSVs against the metadata + PCF columns already
present in the existing val interim files.

Skips the broken preprocessing path that requires loading the empty
amazon_pcf_predictions.csv cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "amazon"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

META_COLS = [
    "parent_asin", "main_category", "title", "average_rating",
    "rating_number", "price", "store", "categories", "pcf",
]


def build(category: str):
    print(f"\n=== {category} ===")
    raw_path = RAW_DIR / "train" / f"{category}.csv"
    val_path = INTERIM_DIR / "val" / f"{category}.csv"
    test_path = INTERIM_DIR / "test" / f"{category}.csv"
    out_path = INTERIM_DIR / "train" / f"{category}.csv"

    print(f"Loading raw train: {raw_path}")
    train_raw = pd.read_csv(raw_path, compression="gzip")
    print(f"  rows={len(train_raw):,} users={train_raw['user_id'].nunique():,}")

    print(f"Loading metadata + PCF from interim val/test")
    meta_frames = []
    for p in [val_path, test_path]:
        if p.exists():
            meta_frames.append(pd.read_csv(p, usecols=META_COLS))
    meta = (
        pd.concat(meta_frames, ignore_index=True)
        .drop_duplicates(subset="parent_asin", keep="first")
    )
    print(f"  unique items with metadata: {len(meta):,}")

    print("Merging (inner join on parent_asin)…")
    merged = train_raw.merge(meta, on="parent_asin", how="inner")
    print(f"  merged rows: {len(merged):,}  "
          f"({100*len(merged)/len(train_raw):.1f}% of raw train)")

    counts = merged.groupby("user_id").size()
    print(f"  users: {merged['user_id'].nunique():,}  "
          f"mean/u: {counts.mean():.1f}  "
          f"≥5: {(counts>=5).sum():,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote → {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="electronics",
                    choices=["electronics", "home_and_kitchen"])
    args = ap.parse_args()
    build(args.category)


if __name__ == "__main__":
    main()
