"""
Carbon Footprint Predictor

Predicts carbon footprint for products not found in The Carbon Catalogue.
Uses product category, attributes, and known carbon values to estimate
footprint for unseen products.

Trained exclusively on the 866-product Carbon Catalogue dataset.
A separate linking step maps Amazon items → this predictor.

Features used from the Carbon Catalogue:
  - TF-IDF on product name + product detail + company (text signal)
  - Sector (ordinal-encoded)
  - Product weight in kg (log-transformed)
  - Lifecycle stage fractions (upstream, operations, downstream)

Target: log1p(PCF in kg CO₂e)  — log-transform handles heavy right tail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from src.data.carbon_loader import load_carbon_catalogue

log = logging.getLogger(__name__)

# ─── Project paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CARBON_DIR = PROJECT_ROOT / "data" / "processed" / "carbon"
MODEL_DIR = PROJECT_ROOT / "output" / "models"

# ─── Carbon Catalogue column names ───────────────────────────────────────────

COL_PCF_ID = "*PCF-ID"
COL_PRODUCT_NAME = "Product name (and functional unit)"
COL_PRODUCT_DETAIL = "Product detail"
COL_COMPANY = "Company"
COL_SECTOR = "*Company's sector"
COL_WEIGHT = "Product weight (kg)"
COL_PCF = "Product's carbon footprint (PCF, kg CO2e)"
COL_INTENSITY = "*Carbon intensity"
COL_UPSTREAM = "*Upstream CO2e (fraction of total PCF)"
COL_OPERATIONS = "*Operations CO2e (fraction of total PCF)"
COL_DOWNSTREAM = "*Downstream CO2e (fraction of total PCF)"

# ─── Default hyper-parameters ─────────────────────────────────────────────────

DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 3,
    "random_state": 42,
}

TFIDF_MAX_FEATURES = 500
CV_FOLDS = 5
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of remaining after test split
RANDOM_STATE = 42


# ─── Report dataclass ────────────────────────────────────────────────────────

@dataclass
class CarbonModelReport:
    """Diagnostics produced during model training and evaluation."""

    # Cross-validation (on train set)
    cv_r2_mean: float = 0.0
    cv_r2_std: float = 0.0
    cv_mae_log_mean: float = 0.0

    # Held-out validation set
    val_r2: float = 0.0
    val_mae_kg: float = 0.0
    val_rmse_kg: float = 0.0

    # Held-out test set
    test_r2: float = 0.0
    test_mae_kg: float = 0.0
    test_rmse_kg: float = 0.0

    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    top_features: dict[str, float] = field(default_factory=dict)
    model_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cv_r2_mean": self.cv_r2_mean,
            "cv_r2_std": self.cv_r2_std,
            "cv_mae_log_mean": self.cv_mae_log_mean,
            "val_r2": self.val_r2,
            "val_mae_kg": self.val_mae_kg,
            "val_rmse_kg": self.val_rmse_kg,
            "test_r2": self.test_r2,
            "test_mae_kg": self.test_mae_kg,
            "test_rmse_kg": self.test_rmse_kg,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "top_20_features": {str(k): float(v) for k, v in self.top_features.items()},
            "model_params": self.model_params,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info("Saved model report → %s", path)


# ─── Feature engineering helpers ──────────────────────────────────────────────

def _parse_fraction(val: Any) -> float:
    """Parse a percentage string like '57.50%' → 0.575. Returns 0 on failure."""
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace("%", "")
    try:
        return float(s) / 100.0
    except ValueError:
        return 0.0


def _build_text(row: pd.Series) -> str:
    """Concatenate textual fields into one TF-IDF input string."""
    parts: list[str] = []
    for col in (COL_PRODUCT_NAME, COL_PRODUCT_DETAIL, COL_COMPANY):
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip().lower())
    return " ".join(parts)


def _build_features(
    df: pd.DataFrame,
    *,
    fit: bool = True,
    encoders: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build a dense feature matrix from Carbon Catalogue columns.

    Parameters
    ----------
    df : Carbon Catalogue DataFrame (raw column names).
    fit : If True, fit new encoders; otherwise reuse existing ones.
    encoders : Previously fitted encoders (required when fit=False).

    Returns
    -------
    X : 2-D float array  (n_products, n_features)
    encoders : dict of fitted encoder objects for reuse at inference.
    """
    if encoders is None:
        encoders = {}

    blocks: list[np.ndarray] = []

    # 1 ── TF-IDF on product name + detail + company ──────────────────────
    texts = df.apply(_build_text, axis=1).tolist()

    if fit or "tfidf" not in encoders:
        tfidf = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        blocks.append(tfidf.fit_transform(texts).toarray())
        encoders["tfidf"] = tfidf
    else:
        blocks.append(encoders["tfidf"].transform(texts).toarray())

    # 2 ── Sector (ordinal-encoded) ───────────────────────────────────────
    sector_arr = (
        df[COL_SECTOR].fillna("unknown").astype(str).to_numpy().reshape(-1, 1)
    )
    if fit or "sector_enc" not in encoders:
        sector_enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
        )
        blocks.append(sector_enc.fit_transform(sector_arr))
        encoders["sector_enc"] = sector_enc
    else:
        blocks.append(encoders["sector_enc"].transform(sector_arr))

    # 3 ── Product weight (log-transformed, 0 if missing) ────────────────
    weight = pd.to_numeric(df[COL_WEIGHT], errors="coerce").fillna(0)
    blocks.append(np.log1p(weight.to_numpy()).reshape(-1, 1))

    # 4 ── Lifecycle stage fractions ──────────────────────────────────────
    for frac_col in (COL_UPSTREAM, COL_OPERATIONS, COL_DOWNSTREAM):
        frac_vals = df[frac_col].apply(_parse_fraction).to_numpy().reshape(-1, 1)
        blocks.append(frac_vals)

    X = np.hstack(blocks)
    log.info("Feature matrix shape: %s", X.shape)
    return X, encoders


def _evaluate(
    model: GradientBoostingRegressor,
    X: np.ndarray,
    y_log: np.ndarray,
    label: str,
) -> dict[str, float]:
    """Evaluate model on a split, return metrics in original kg CO₂e scale."""
    pred_log = model.predict(X)
    pred_kg = np.expm1(pred_log)
    true_kg = np.expm1(y_log)

    r2 = r2_score(y_log, pred_log)
    mae = mean_absolute_error(true_kg, pred_kg)
    rmse = float(np.sqrt(mean_squared_error(true_kg, pred_kg)))

    log.info(
        "%s  R²=%.3f  MAE=%.2f kg  RMSE=%.2f kg  (n=%d)",
        label, r2, mae, rmse, len(y_log),
    )
    return {"r2": r2, "mae_kg": mae, "rmse_kg": rmse}


# ─── Public API ───────────────────────────────────────────────────────────────

class CarbonPredictor:
    """
    Gradient-boosted predictor for product carbon footprints.

    Trained exclusively on The Carbon Catalogue (866 products).
    Uses proper train / val / test splits plus cross-validation on
    the training fold.

    Typical usage::

        predictor = CarbonPredictor()
        report = predictor.fit(carbon_df)          # train + evaluate
        preds = predictor.predict(carbon_df)       # predict on any CC-format df

        predictor.save(Path("output/models/carbon_model.joblib"))
        predictor = CarbonPredictor.load(Path("output/models/carbon_model.joblib"))
    """

    def __init__(
        self,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        self._params = model_params or DEFAULT_MODEL_PARAMS
        self._model: GradientBoostingRegressor | None = None
        self._encoders: dict[str, Any] = {}
        self._report: CarbonModelReport | None = None

    # ── Training ──────────────────────────────────────────────────────────

    def fit(self, carbon_df: pd.DataFrame) -> CarbonModelReport:
        """
        Train on Carbon Catalogue data with train/val/test splits and CV.

        Steps
        -----
        1. Split data → train (70%) / val (15%) / test (15%)
        2. Run k-fold CV on training set to estimate generalisation
        3. Fit final model on training set
        4. Evaluate on validation set (hyper-parameter selection guide)
        5. Evaluate on held-out test set (final reported numbers)

        Parameters
        ----------
        carbon_df : Raw Carbon Catalogue DataFrame (as returned by
                    ``load_carbon_catalogue()``).

        Returns
        -------
        CarbonModelReport with all evaluation metrics.
        """
        log.info(
            "Training carbon predictor on %d labelled products …", len(carbon_df),
        )

        carbon_df = carbon_df.copy()

        # ── Target: log1p(PCF in kg CO₂e) ────────────────────────────────
        carbon_df["_target"] = np.log1p(
            pd.to_numeric(carbon_df[COL_PCF], errors="coerce").fillna(0)
        )

        # ── Train / Val / Test split ─────────────────────────────────────
        train_val_df, test_df = train_test_split(
            carbon_df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        relative_val = VAL_SIZE / (1.0 - TEST_SIZE)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val,
            random_state=RANDOM_STATE,
        )

        log.info(
            "Split sizes — train: %d  val: %d  test: %d",
            len(train_df), len(val_df), len(test_df),
        )

        # ── Build features (fit encoders on training set only) ───────────
        X_train, self._encoders = _build_features(train_df, fit=True)
        y_train = train_df["_target"].to_numpy()

        X_val, _ = _build_features(val_df, fit=False, encoders=self._encoders)
        y_val = val_df["_target"].to_numpy()

        X_test, _ = _build_features(test_df, fit=False, encoders=self._encoders)
        y_test = test_df["_target"].to_numpy()

        # ── Cross-validation on training set ─────────────────────────────
        model = GradientBoostingRegressor(**self._params)

        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        cv_mae = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error",
        )

        log.info(
            "CV on train — R² (log): %.3f ± %.3f   MAE (log): %.3f ± %.3f",
            cv_r2.mean(), cv_r2.std(), -cv_mae.mean(), cv_mae.std(),
        )

        # ── Fit on full training set ─────────────────────────────────────
        model.fit(X_train, y_train)
        self._model = model

        # ── Evaluate on val and test ─────────────────────────────────────
        val_metrics = _evaluate(model, X_val, y_val, "Val ")
        test_metrics = _evaluate(model, X_test, y_test, "Test")

        # ── Feature importances ──────────────────────────────────────────
        n_tfidf = TFIDF_MAX_FEATURES
        feature_names = (
            [f"tfidf_{i}" for i in range(n_tfidf)]
            + ["sector", "log_weight", "frac_upstream", "frac_operations", "frac_downstream"]
        )
        # Guard against shape mismatch if tfidf produced fewer features
        if len(feature_names) != len(model.feature_importances_):
            feature_names = [f"f{i}" for i in range(len(model.feature_importances_))]

        importances = pd.Series(model.feature_importances_, index=feature_names)
        top_features = importances.nlargest(20).to_dict()

        # ── Assemble report ──────────────────────────────────────────────
        self._report = CarbonModelReport(
            cv_r2_mean=float(cv_r2.mean()),
            cv_r2_std=float(cv_r2.std()),
            cv_mae_log_mean=float(-cv_mae.mean()),
            val_r2=val_metrics["r2"],
            val_mae_kg=val_metrics["mae_kg"],
            val_rmse_kg=val_metrics["rmse_kg"],
            test_r2=test_metrics["r2"],
            test_mae_kg=test_metrics["mae_kg"],
            test_rmse_kg=test_metrics["rmse_kg"],
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            top_features=top_features,
            model_params=self._params,
        )
        return self._report

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict carbon footprints for a Carbon-Catalogue-format DataFrame.

        Parameters
        ----------
        df : Must contain the same columns as the Carbon Catalogue
             (product name, sector, weight, etc.).

        Returns
        -------
        DataFrame with columns:
            ``pcf_id``, ``predicted_carbon_kg``, ``predicted_carbon_log``,
            ``sector``.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        log.info("Predicting carbon footprints for %d items …", len(df))

        X, _ = _build_features(df, fit=False, encoders=self._encoders)

        log_preds = self._model.predict(X)
        carbon_kg = np.clip(np.expm1(log_preds), 0.01, 10_000.0)

        results = pd.DataFrame({
            "pcf_id": df[COL_PCF_ID].values if COL_PCF_ID in df.columns else range(len(df)),
            "predicted_carbon_kg": carbon_kg,
            "predicted_carbon_log": log_preds,
            "sector": df[COL_SECTOR].values if COL_SECTOR in df.columns else "unknown",
        })

        log.info(
            "Predictions — median=%.2f kg  mean=%.2f kg  range=[%.2f, %.2f]",
            np.median(carbon_kg), np.mean(carbon_kg),
            carbon_kg.min(), carbon_kg.max(),
        )
        return results

    # ── Persistence (pickle, no joblib) ───────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist trained model + encoders to disk via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "model": self._model,
            "encoders": self._encoders,
            "params": self._params,
        }
        joblib.dump(blob, path)
        log.info("Saved CarbonPredictor → %s", path)

    @classmethod
    def load(cls, path: Path) -> CarbonPredictor:
        """Load a previously saved CarbonPredictor from a joblib file."""
        blob = joblib.load(path)
        predictor = cls(model_params=blob["params"])
        predictor._model = blob["model"]
        predictor._encoders = blob["encoders"]
        log.info("Loaded CarbonPredictor ← %s", path)
        return predictor

    @property
    def report(self) -> CarbonModelReport | None:
        return self._report

    @property
    def is_fitted(self) -> bool:
        return self._model is not None


# ─── Convenience entry-point ─────────────────────────────────────────────────

def train_and_evaluate() -> CarbonModelReport:
    """
    End-to-end: load Carbon Catalogue → train → evaluate → save artefacts.

    Saved to:
        data/processed/carbon/carbon_predictions.csv
        output/models/carbon_model.joblib
        output/models/carbon_model_report.json
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("=== Carbon Predictor: Train & Evaluate ===")

    # 1. Load raw catalogue
    carbon_df = load_carbon_catalogue()

    # 2. Train with train/val/test splits + CV
    predictor = CarbonPredictor()
    report = predictor.fit(carbon_df)

    # 3. Predict on entire catalogue (for inspection / downstream use)
    predictions = predictor.predict(carbon_df)
    out_csv = PROCESSED_CARBON_DIR / "carbon_predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_csv, index=False)
    log.info("Saved predictions → %s", out_csv)

    # 4. Save model + report
    model_path = MODEL_DIR / "carbon_model.joblib"
    predictor.save(model_path)

    report_path = MODEL_DIR / "carbon_model_report.json"
    report.save(report_path)

    # 5. Summary
    log.info("=== Summary ===")
    log.info("  Train/Val/Test: %d / %d / %d", report.n_train, report.n_val, report.n_test)
    log.info("  CV R² (log):  %.3f ± %.3f", report.cv_r2_mean, report.cv_r2_std)
    log.info("  Val  R²=%.3f  MAE=%.2f kg  RMSE=%.2f kg", report.val_r2, report.val_mae_kg, report.val_rmse_kg)
    log.info("  Test R²=%.3f  MAE=%.2f kg  RMSE=%.2f kg", report.test_r2, report.test_mae_kg, report.test_rmse_kg)

    return report


if __name__ == "__main__":
    train_and_evaluate()
