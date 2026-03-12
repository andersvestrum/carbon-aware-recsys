"""
DeepFM Model Wrapper — Pipeline Step 2

DeepFM for CTR (Click-Through Rate) prediction.
Predicts: "Will this user engage with (click/rate/purchase) this item?"
Output: Probability score ∈ [0, 1].

Reference: DeepFM (Guo et al., 2017)
https://arxiv.org/pdf/1703.04247

The wrapper handles:
    • Feature engineering from interim interaction data
    • Sparse (categorical) + dense (numeric) feature extraction
    • Label binarisation (rating ≥ threshold → positive)
    • Training with early stopping
    • Scoring RecBole candidates with the trained model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ─── Feature definitions ────────────────────────────────────────────────────

# Categorical columns → sparse embeddings
SPARSE_FEATURES = ["user_id", "parent_asin", "main_category", "store"]

# Numeric columns → dense inputs
DENSE_FEATURES = ["average_rating", "rating_number", "price", "pcf"]


# ─── Feature engineering ────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    label_encoders: dict[str, LabelEncoder] | None = None,
    scaler: MinMaxScaler | None = None,
    fit: bool = True,
    rating_threshold: float = 4.0,
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], MinMaxScaler, np.ndarray]:
    """Build DeepFM input features from an interaction DataFrame.

    Args:
        df: Interaction data with at least the sparse + dense columns.
        label_encoders: Pre-fitted encoders (for val/test).
        scaler: Pre-fitted MinMaxScaler (for val/test).
        fit: Whether to fit encoders/scaler (True for train).
        rating_threshold: Ratings ≥ this value → label 1.

    Returns:
        (features_df, label_encoders, scaler, labels)
    """
    data = df.copy()

    # ── Fill missing values ───────────────────────────────────────────
    for col in DENSE_FEATURES:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median() if fit else 0.0)
        else:
            data[col] = 0.0

    for col in SPARSE_FEATURES:
        if col in data.columns:
            data[col] = data[col].fillna("__UNKNOWN__").astype(str)
        else:
            data[col] = "__UNKNOWN__"

    # ── Encode sparse features ────────────────────────────────────────
    if label_encoders is None:
        label_encoders = {}

    for col in SPARSE_FEATURES:
        if fit:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            # Handle unseen labels
            data[col] = data[col].map(
                lambda x, _le=le: (
                    _le.transform([x])[0]
                    if x in _le.classes_
                    else 0
                )
            )

    # ── Scale dense features ──────────────────────────────────────────
    if fit:
        scaler = MinMaxScaler()
        data[DENSE_FEATURES] = scaler.fit_transform(data[DENSE_FEATURES])
    else:
        data[DENSE_FEATURES] = scaler.transform(data[DENSE_FEATURES])

    # ── Labels ────────────────────────────────────────────────────────
    labels = (data["rating"].values >= rating_threshold).astype(np.float32)

    return data, label_encoders, scaler, labels


def get_feature_columns(data: pd.DataFrame):
    """Build deepctr-torch feature column lists.

    Returns:
        (sparse_feature_columns, dense_feature_columns)
    """
    from deepctr_torch.inputs import DenseFeat, SparseFeat

    sparse_feature_columns = [
        SparseFeat(col, vocabulary_size=int(data[col].max()) + 1, embedding_dim=8)
        for col in SPARSE_FEATURES
    ]
    dense_feature_columns = [
        DenseFeat(col, dimension=1) for col in DENSE_FEATURES
    ]

    return sparse_feature_columns, dense_feature_columns


# ─── Model builder ──────────────────────────────────────────────────────────

@dataclass
class DeepFMWrapper:
    """Thin wrapper around deepctr-torch DeepFM.

    Attributes:
        dnn_hidden_units: Hidden layer sizes for the DNN component.
        l2_reg_embedding: L2 regularisation for embeddings.
        l2_reg_dnn: L2 regularisation for DNN layers.
        dnn_dropout: Dropout rate for DNN layers.
        epochs: Training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        rating_threshold: Ratings ≥ this → positive label.
        device: ``cpu`` or ``cuda``.
    """

    dnn_hidden_units: tuple[int, ...] = (256, 128, 64)
    l2_reg_embedding: float = 1e-5
    l2_reg_dnn: float = 0.0
    dnn_dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    rating_threshold: float = 4.0
    device: str = "cpu"

    # Fitted state (set after training)
    model_: Any = field(default=None, repr=False)
    label_encoders_: dict | None = field(default=None, repr=False)
    scaler_: Any = field(default=None, repr=False)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Train the DeepFM model.

        Args:
            train_df: Training interactions (must include rating + features).
            val_df: Validation interactions for early stopping.

        Returns:
            Dict with training history metrics.
        """
        from deepctr_torch.models import DeepFM

        log.info("Building features for training (%s rows) …", f"{len(train_df):,}")

        train_data, self.label_encoders_, self.scaler_, train_labels = (
            build_features(
                train_df, fit=True, rating_threshold=self.rating_threshold,
            )
        )

        sparse_cols, dense_cols = get_feature_columns(train_data)
        feature_columns = sparse_cols + dense_cols

        # Build model input dict
        train_input = {
            col: train_data[col].values for col in SPARSE_FEATURES + DENSE_FEATURES
        }

        # Validation data
        val_input, val_labels = None, None
        if val_df is not None and len(val_df) > 0:
            val_data, _, _, val_labels = build_features(
                val_df,
                label_encoders=self.label_encoders_,
                scaler=self.scaler_,
                fit=False,
                rating_threshold=self.rating_threshold,
            )
            val_input = {
                col: val_data[col].values
                for col in SPARSE_FEATURES + DENSE_FEATURES
            }

        # Build DeepFM
        self.model_ = DeepFM(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            dnn_hidden_units=self.dnn_hidden_units,
            l2_reg_embedding=self.l2_reg_embedding,
            l2_reg_dnn=self.l2_reg_dnn,
            dnn_dropout=self.dnn_dropout,
            device=self.device,
            task="binary",
        )

        self.model_.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["binary_crossentropy", "auc"],
            lr=self.learning_rate,
        )

        log.info("Training DeepFM (epochs=%d, batch_size=%d) …", self.epochs, self.batch_size)

        validation_data = None
        if val_input is not None:
            validation_data = (val_input, val_labels)

        history = self.model_.fit(
            train_input,
            train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            verbose=1,
        )

        return {k: [float(v) for v in vals] for k, vals in history.history.items()}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict engagement probability for each row.

        Args:
            df: DataFrame with the same feature columns as training.

        Returns:
            Array of engagement probabilities (shape: ``(n,)``).
        """
        if self.model_ is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        data, _, _, _ = build_features(
            df,
            label_encoders=self.label_encoders_,
            scaler=self.scaler_,
            fit=False,
            rating_threshold=self.rating_threshold,
        )

        model_input = {
            col: data[col].values for col in SPARSE_FEATURES + DENSE_FEATURES
        }

        preds = self.model_.predict(model_input, batch_size=self.batch_size)
        return preds.flatten()

    def score_candidates(
        self,
        candidates_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Score RecBole candidates with the trained DeepFM.

        Joins candidate (user_id, parent_asin) pairs with item features
        from the interactions data, predicts engagement, and returns
        a DataFrame ready for the carbon re-ranker.

        Args:
            candidates_df: RecBole output with
                ``[user_id, parent_asin, relevance_score]``.
            interactions_df: Interim interactions with item features
                (average_rating, price, store, etc.).

        Returns:
            DataFrame with ``[user_id, parent_asin, relevance_score,
            engagement_score]``.
        """
        # Get item features (take the most common / first row per item)
        item_features = (
            interactions_df
            .drop_duplicates(subset="parent_asin")
            [["parent_asin", "main_category", "store",
              "average_rating", "rating_number", "price", "pcf"]]
            .copy()
        )

        # Join candidates with item features
        scored = candidates_df.merge(item_features, on="parent_asin", how="left")

        # We need a rating column for build_features (used for label only,
        # we'll set a dummy since we only need prediction here)
        scored["rating"] = 0.0

        # Predict engagement
        engagement = self.predict(scored)
        scored["engagement_score"] = engagement

        # Return clean output
        keep_cols = ["user_id", "parent_asin", "relevance_score", "engagement_score"]
        return scored[keep_cols].copy()
