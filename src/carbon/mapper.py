"""
Carbon Mapper

Maps Amazon product metadata (from data/raw/amazon/meta/) to the Carbon
Catalogue feature space, applies the trained CarbonPredictor, and returns
a ``pcf`` column (predicted kg CO₂e) for every product.

Usage::

    from src.carbon.mapper import CarbonMapper

    mapper = CarbonMapper()
    result = mapper.map(amazon_meta_df, predictor)
    # → DataFrame with parent_asin + pcf columns
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.carbon.predictor import (
    CarbonPredictor,
    COL_COMPANY,
    COL_DOWNSTREAM,
    COL_OPERATIONS,
    COL_PCF,
    COL_PRODUCT_DETAIL,
    COL_PRODUCT_NAME,
    COL_SECTOR,
    COL_UPSTREAM,
    COL_WEIGHT,
)
from src.data.amazon_loader import load_all_meta

log = logging.getLogger(__name__)

# ─── Project paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "output" / "models"

# ─── Amazon main_category → Carbon Catalogue sector mapping ──────────────────
#
# Actual Amazon main_category values (from our 3-category meta files):
#   electronics:        All Electronics, Computers, Camera & Photo,
#                       Cell Phones & Accessories, Home Audio & Theater, ...
#   home_and_kitchen:   Amazon Home, Tools & Home Improvement, ...
#   sports_and_outdoors: Sports & Outdoors, AMAZON FASHION, ...
#
# Actual Carbon Catalogue sectors:
#   Computer, IT & telecom (253)
#   Food & Beverage (139)
#   Home durables, textiles, & equipment (122)
#   Chemicals (116)
#   Automobiles & components (75)
#   Construction & commercial materials (67)
#   Comm. equipm. & capital goods (56)
#   Packaging for consumer goods (38)

AMAZON_TO_CC_SECTOR: dict[str, str] = {
    # ── Electronics meta ─────────────────────────────────────────────────
    "All Electronics":            "Computer, IT & telecom",
    "Computers":                  "Computer, IT & telecom",
    "Camera & Photo":             "Computer, IT & telecom",
    "Cell Phones & Accessories":  "Computer, IT & telecom",
    "Home Audio & Theater":       "Comm. equipm. & capital goods",
    "Car Electronics":            "Computer, IT & telecom",
    "GPS & Navigation":           "Computer, IT & telecom",
    "Headphones":                 "Computer, IT & telecom",
    "Portable Audio & Video":     "Computer, IT & telecom",
    "Wearable Technology":        "Computer, IT & telecom",
    "Television & Video":         "Comm. equipm. & capital goods",
    "Video Projectors":           "Comm. equipm. & capital goods",
    "Security & Surveillance":    "Computer, IT & telecom",
    # ── Home & Kitchen meta ──────────────────────────────────────────────
    "Amazon Home":                "Home durables, textiles, & equipment",
    "Kitchen & Dining":           "Home durables, textiles, & equipment",
    "Tools & Home Improvement":   "Home durables, textiles, & equipment",
    "Home Improvement":           "Construction & commercial materials",
    "Garden & Outdoor":           "Home durables, textiles, & equipment",
    "Industrial & Scientific":    "Chemicals",
    # ── Sports & Outdoors meta ───────────────────────────────────────────
    "Sports & Outdoors":          "Home durables, textiles, & equipment",
    "AMAZON FASHION":             "Home durables, textiles, & equipment",
    "Automotive":                 "Automobiles & components",
    # ── Catch-all for other categories ───────────────────────────────────
    "Toys & Games":               "Packaging for consumer goods",
    "Office Products":            "Computer, IT & telecom",
    "Musical Instruments":        "Comm. equipm. & capital goods",
    "Arts, Crafts & Sewing":      "Chemicals",
    "Pet Supplies":               "Packaging for consumer goods",
    "Baby Products":              "Packaging for consumer goods",
    "Appliances":                 "Home durables, textiles, & equipment",
    "Software":                   "Computer, IT & telecom",
    "Video Games":                "Computer, IT & telecom",
}

# Fallback sector when no mapping is found
FALLBACK_SECTOR = "Home durables, textiles, & equipment"

# ─── Weight heuristic (Amazon has no weight field) ────────────────────────────
#
# Rough median product weight (kg) by CC sector and price tier.

SECTOR_WEIGHT_DEFAULTS: dict[str, dict[str, float]] = {
    "Computer, IT & telecom":              {"cheap": 0.2, "mid": 1.0, "expensive": 3.0},
    "Food & Beverage":                     {"cheap": 0.3, "mid": 0.5, "expensive": 2.0},
    "Home durables, textiles, & equipment":{"cheap": 0.3, "mid": 1.5, "expensive": 5.0},
    "Chemicals":                           {"cheap": 0.2, "mid": 0.5, "expensive": 1.5},
    "Automobiles & components":            {"cheap": 0.5, "mid": 3.0, "expensive": 15.0},
    "Construction & commercial materials": {"cheap": 1.0, "mid": 5.0, "expensive": 20.0},
    "Comm. equipm. & capital goods":       {"cheap": 0.3, "mid": 2.0, "expensive": 8.0},
    "Packaging for consumer goods":        {"cheap": 0.1, "mid": 0.4, "expensive": 1.5},
}

# ── Lifecycle fraction defaults per sector (approximate CC medians) ──────────
# Used because Amazon products don't have lifecycle data.

SECTOR_LIFECYCLE_DEFAULTS: dict[str, dict[str, float]] = {
    "Computer, IT & telecom":              {"upstream": 0.60, "operations": 0.20, "downstream": 0.20},
    "Food & Beverage":                     {"upstream": 0.70, "operations": 0.15, "downstream": 0.15},
    "Home durables, textiles, & equipment":{"upstream": 0.55, "operations": 0.25, "downstream": 0.20},
    "Chemicals":                           {"upstream": 0.65, "operations": 0.20, "downstream": 0.15},
    "Automobiles & components":            {"upstream": 0.50, "operations": 0.25, "downstream": 0.25},
    "Construction & commercial materials": {"upstream": 0.55, "operations": 0.30, "downstream": 0.15},
    "Comm. equipm. & capital goods":       {"upstream": 0.60, "operations": 0.20, "downstream": 0.20},
    "Packaging for consumer goods":        {"upstream": 0.65, "operations": 0.20, "downstream": 0.15},
}

DEFAULT_LIFECYCLE = {"upstream": 0.60, "operations": 0.20, "downstream": 0.20}

PRICE_CHEAP = 15.0
PRICE_EXPENSIVE = 80.0


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_price(val: Any) -> float:
    """Parse an Amazon price field → float. Returns 0.0 on failure."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return max(float(val), 0.0)
    s = str(val).replace("$", "").replace(",", "").strip()
    try:
        return max(float(s), 0.0)
    except ValueError:
        return 0.0


def _price_tier(price: float) -> str:
    if price <= PRICE_CHEAP:
        return "cheap"
    if price >= PRICE_EXPENSIVE:
        return "expensive"
    return "mid"


def _estimate_weight(sector: str, price: float) -> float:
    tiers = SECTOR_WEIGHT_DEFAULTS.get(
        sector, SECTOR_WEIGHT_DEFAULTS["Home durables, textiles, & equipment"],
    )
    return tiers[_price_tier(price)]


# ─── CarbonMapper ────────────────────────────────────────────────────────────

class CarbonMapper:
    """
    Translates Amazon product metadata into the Carbon Catalogue feature
    space and applies a trained :class:`CarbonPredictor` to return a
    ``pcf`` (predicted kg CO₂e) for every Amazon product.

    Usage::

        predictor = CarbonPredictor.load("output/models/carbon_model.joblib")
        mapper = CarbonMapper()
        result = mapper.map(meta_df, predictor)
        # result has columns: parent_asin, pcf, mapped_sector
    """

    def __init__(
        self,
        sector_map: dict[str, str] | None = None,
    ) -> None:
        self._sector_map = sector_map or AMAZON_TO_CC_SECTOR

    def map(
        self,
        meta_df: pd.DataFrame,
        predictor: CarbonPredictor,
    ) -> pd.DataFrame:
        """
        Map Amazon metadata → predicted carbon footprint.

        Parameters
        ----------
        meta_df : Amazon product metadata (must have ``parent_asin`` and
                  ``title``; ``main_category``, ``price``, ``store`` are
                  used when available).
        predictor : A fitted :class:`CarbonPredictor`.

        Returns
        -------
        DataFrame with columns:
            ``parent_asin`` — unique product ID
            ``pcf``         — predicted product carbon footprint (kg CO₂e)
            ``mapped_sector`` — Carbon Catalogue sector used for prediction
        """
        if not predictor.is_fitted:
            raise RuntimeError(
                "Predictor is not fitted. Call predictor.fit() or load a model first."
            )

        log.info("Mapping %s Amazon items → Carbon Catalogue space …", f"{len(meta_df):,}")

        # Build a CC-format DataFrame from Amazon columns
        cc_df = self._translate(meta_df)

        # Run the predictor
        predictions = predictor.predict(cc_df)

        # Result: parent_asin + pcf
        result = pd.DataFrame({
            "parent_asin": meta_df["parent_asin"].values,
            "pcf": predictions["predicted_carbon_kg"].values,
            "mapped_sector": cc_df[COL_SECTOR].values,
        })

        self._log_summary(result)
        return result

    # ── Translation: Amazon schema → Carbon Catalogue schema ──────────────

    def _translate(self, meta_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a DataFrame with Carbon Catalogue column names from Amazon
        metadata so the predictor's ``_build_features()`` works unchanged.
        """
        n = len(meta_df)
        cc = pd.DataFrame(index=range(n))

        # Product name ← title
        cc[COL_PRODUCT_NAME] = (
            meta_df["title"].fillna("").astype(str).str.strip().values
        )

        # Product detail ← concatenation of available text fields
        detail_cols = [c for c in ("categories", "description", "features") if c in meta_df.columns]
        if detail_cols:
            cc[COL_PRODUCT_DETAIL] = (
                meta_df[detail_cols]
                .fillna("")
                .astype(str)
                .apply(lambda row: " ".join(row), axis=1)
                .str.strip()
                .values
            )
        else:
            cc[COL_PRODUCT_DETAIL] = ""

        # Company ← store (or empty)
        if "store" in meta_df.columns:
            cc[COL_COMPANY] = meta_df["store"].fillna("").astype(str).values
        else:
            cc[COL_COMPANY] = ""

        # Sector ← mapped from Amazon main_category
        main_cat = meta_df.get("main_category", pd.Series([""] * n))
        cc[COL_SECTOR] = main_cat.map(self._sector_map).fillna(FALLBACK_SECTOR).values

        # Price → weight heuristic
        if "price" in meta_df.columns:
            prices = meta_df["price"].apply(_clean_price).values
        else:
            prices = np.zeros(n)

        cc[COL_WEIGHT] = [
            _estimate_weight(s, p) for s, p in zip(cc[COL_SECTOR], prices)
        ]

        # Lifecycle fractions ← sector defaults
        upstreams, operations, downstreams = [], [], []
        for sector in cc[COL_SECTOR]:
            lc = SECTOR_LIFECYCLE_DEFAULTS.get(sector, DEFAULT_LIFECYCLE)
            upstreams.append(f"{lc['upstream'] * 100:.2f}%")
            operations.append(f"{lc['operations'] * 100:.2f}%")
            downstreams.append(f"{lc['downstream'] * 100:.2f}%")

        cc[COL_UPSTREAM] = upstreams
        cc[COL_OPERATIONS] = operations
        cc[COL_DOWNSTREAM] = downstreams

        # PCF placeholder (not used during prediction)
        cc[COL_PCF] = np.nan

        log.info(
            "Sector distribution:\n%s",
            cc[COL_SECTOR].value_counts().to_string(),
        )

        return cc

    @staticmethod
    def _log_summary(result: pd.DataFrame) -> None:
        """Log per-sector prediction stats."""
        log.info("=== Carbon Mapping Summary ===")
        log.info(
            "  Overall: median=%.2f kg  mean=%.2f kg  range=[%.2f, %.2f]",
            result["pcf"].median(),
            result["pcf"].mean(),
            result["pcf"].min(),
            result["pcf"].max(),
        )
        stats = (
            result
            .groupby("mapped_sector")["pcf"]
            .agg(["count", "median", "mean", "min", "max"])
        )
        log.info("  Per-sector:\n%s", stats.to_string())


# ─── Convenience entry-point ──────────────────────────────────────────────────

def map_all_amazon_meta(
    model_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    End-to-end: load all Amazon metadata + trained predictor → pcf column.

    Loads meta CSVs from ``data/raw/amazon/meta/``, applies the mapper,
    and saves a single CSV with ``parent_asin, pcf, mapped_sector``.

    Saved to: ``data/processed/carbon/amazon_carbon_estimates.csv``
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if model_path is None:
        model_path = MODEL_DIR / "carbon_model.joblib"
    if output_path is None:
        output_path = PROCESSED_DIR / "carbon" / "amazon_carbon_estimates.csv"

    log.info("=== Carbon Mapper: Amazon Meta → PCF ===")

    # 1. Load all Amazon metadata
    all_meta = load_all_meta()
    meta_frames = []
    for cat, df in all_meta.items():
        df = df.copy()
        df["source_category"] = cat  # track which file it came from
        meta_frames.append(df)
    meta_df = pd.concat(meta_frames, ignore_index=True)

    # De-duplicate on parent_asin (same product may appear in multiple files)
    meta_df = meta_df.drop_duplicates(subset="parent_asin", keep="first")
    log.info("Total unique Amazon products: %s", f"{len(meta_df):,}")

    # 2. Load predictor
    predictor = CarbonPredictor.load(model_path)

    # 3. Map
    mapper = CarbonMapper()
    result = mapper.map(meta_df, predictor)

    # 4. Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    log.info("Saved %s estimates → %s", f"{len(result):,}", output_path)

    return result


if __name__ == "__main__":
    map_all_amazon_meta()
