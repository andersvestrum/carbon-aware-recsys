"""
RecBole Data Formatter

Converts interim interaction DataFrames into RecBole's required file formats:
  - .inter  (user–item interactions)
  - .item   (item features, optional)
  - .user   (user features, optional)

RecBole expects tab-separated files where the header encodes the field name
and type, e.g.:  user_id:token   item_id:token   rating:float   timestamp:float
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Project root: carbon-aware-recsys/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECBOLE_DIR = PROJECT_ROOT / "data" / "processed" / "recbole"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def _concat_interim_splits(
    category: str,
    splits: list[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate interim interaction CSVs for a category.

    Args:
        category: Category filename stem, e.g. ``"electronics"``.
        splits: Which splits to include. Defaults to all three.

    Returns:
        Single concatenated DataFrame with a ``split`` column.
    """
    if splits is None:
        splits = ["train", "val", "test"]

    frames = []
    for split in splits:
        path = INTERIM_DIR / split / f"{category}.csv"
        if not path.exists():
            log.warning("Missing interim file: %s — skipping", path)
            continue
        df = pd.read_csv(path)
        df["split"] = split
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No interim files found for category '{category}'. "
            "Run preprocessing first."
        )

    combined = pd.concat(frames, ignore_index=True)
    log.info(
        "Loaded %s: %s interactions across %d splits",
        category,
        f"{len(combined):,}",
        len(frames),
    )
    return combined


def write_recbole_inter(
    interactions: pd.DataFrame,
    dataset_name: str,
    output_dir: Path | None = None,
) -> Path:
    """Write a RecBole .inter file from an interactions DataFrame.

    Required columns in *interactions*:
        - ``user_id``  (str)
        - ``parent_asin``  (str, used as item_id)

    Optional columns (included when present):
        - ``rating``    → ``rating:float``
        - ``timestamp`` → ``timestamp:float``

    Args:
        interactions: DataFrame with at least user_id and parent_asin.
        dataset_name: RecBole dataset name (directory + file stem).
        output_dir: Where to write. Defaults to ``data/processed/recbole/<dataset_name>/``.

    Returns:
        Path to the written .inter file.
    """
    if output_dir is None:
        output_dir = RECBOLE_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    inter_path = output_dir / f"{dataset_name}.inter"

    out = pd.DataFrame()
    out["user_id:token"] = interactions["user_id"].astype(str)
    out["item_id:token"] = interactions["parent_asin"].astype(str)

    if "rating" in interactions.columns:
        out["rating:float"] = pd.to_numeric(
            interactions["rating"], errors="coerce"
        ).fillna(0.0)

    if "timestamp" in interactions.columns:
        out["timestamp:float"] = pd.to_numeric(
            interactions["timestamp"], errors="coerce"
        ).fillna(0.0)
    else:
        # RecBole needs timestamps for time-ordered splits; use synthetic
        out["timestamp:float"] = np.arange(len(out), dtype=np.float64)

    out.to_csv(inter_path, sep="\t", index=False)
    log.info("Wrote RecBole .inter → %s  (%s rows)", inter_path, f"{len(out):,}")
    return inter_path


def write_recbole_item(
    interactions: pd.DataFrame,
    dataset_name: str,
    output_dir: Path | None = None,
) -> Path | None:
    """Write optional RecBole .item file with carbon footprint as item feature.

    Only written if ``pcf`` column is present in the interactions DataFrame.

    Returns:
        Path to the .item file, or None if not written.
    """
    if "pcf" not in interactions.columns:
        log.info("No 'pcf' column — skipping .item file")
        return None

    if output_dir is None:
        output_dir = RECBOLE_DIR / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    item_path = output_dir / f"{dataset_name}.item"

    # Deduplicate on item_id
    items = (
        interactions[["parent_asin", "pcf"]]
        .drop_duplicates(subset="parent_asin")
        .copy()
    )

    out = pd.DataFrame()
    out["item_id:token"] = items["parent_asin"].astype(str)
    out["pcf:float"] = items["pcf"].astype(float)

    # Include category if available
    if "main_category" in interactions.columns:
        cats = (
            interactions[["parent_asin", "main_category"]]
            .drop_duplicates(subset="parent_asin")
        )
        out["category:token"] = cats["main_category"].astype(str).values

    out.to_csv(item_path, sep="\t", index=False)
    log.info("Wrote RecBole .item → %s  (%s items)", item_path, f"{len(out):,}")
    return item_path


def format_category_for_recbole(
    category: str,
    dataset_name: str | None = None,
) -> tuple[Path, str]:
    """Full pipeline: load interim data → write RecBole files.

    Args:
        category: Category filename stem (e.g. ``"electronics"``).
        dataset_name: RecBole dataset name. Defaults to the category name.

    Returns:
        (output_dir, dataset_name)
    """
    if dataset_name is None:
        dataset_name = category

    interactions = _concat_interim_splits(category)
    output_dir = RECBOLE_DIR / dataset_name

    write_recbole_inter(interactions, dataset_name, output_dir)
    write_recbole_item(interactions, dataset_name, output_dir)

    return output_dir, dataset_name


def format_all_categories_for_recbole(
    categories: list[str] | None = None,
) -> dict[str, tuple[Path, str]]:
    """Format all categories into RecBole datasets.

    Args:
        categories: List of category stems. Defaults to all three.

    Returns:
        Dict mapping category → (output_dir, dataset_name).
    """
    if categories is None:
        categories = ["electronics", "home_and_kitchen", "sports_and_outdoors"]

    results = {}
    for cat in categories:
        log.info("Formatting %s for RecBole …", cat)
        results[cat] = format_category_for_recbole(cat)

    return results
