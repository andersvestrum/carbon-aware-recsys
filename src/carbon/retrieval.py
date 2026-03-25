"""
Retrieval-based Product Carbon Footprint estimation.

This module estimates Product Carbon Footprint (PCF) values for Amazon
products by retrieving semantically similar products from The Carbon
Catalogue and optionally querying an LLM with zero-shot or retrieval-
augmented few-shot prompts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.amazon_loader import load_all_meta
from src.data.carbon_loader import load_carbon_catalogue

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_CARBON_DIR = PROJECT_ROOT / "data" / "processed" / "carbon"
RESULTS_DIR = PROJECT_ROOT / "output" / "results" / "carbon"

CARBON_ID_COL = "*PCF-ID"
CARBON_TITLE_COL = "Product name (and functional unit)"
CARBON_DETAIL_COL = "Product detail"
CARBON_COMPANY_COL = "Company"
CARBON_SECTOR_COL = "*Company's sector"
CARBON_PCF_COL = "Product's carbon footprint (PCF, kg CO2e)"

AMAZON_ID_COL = "parent_asin"
AMAZON_TITLE_COL = "title"
AMAZON_METADATA_COLS = (
    "main_category",
    "store",
    "categories",
    "description",
    "features",
)
CARBON_CONTEXT_COLS = (
    CARBON_DETAIL_COL,
    CARBON_COMPANY_COL,
    CARBON_SECTOR_COL,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_TOP_K = 5
DEFAULT_SELECTED_PCF_COL = "pcf"
PREDICTION_METHOD_TO_COLUMN = {
    "neighbor_average": "neighbor_average_pcf",
    "zero_shot_llm": "zero_shot_llm_pcf",
    "few_shot_llm": "few_shot_llm_pcf",
}
LLM_METHODS = ("zero_shot_llm", "few_shot_llm")
NEIGHBOR_OUTPUT_SUFFIXES = ("id", "title", "pcf", "similarity")
AMAZON_OUTPUT_BASE_COLS = (
    AMAZON_ID_COL,
    AMAZON_TITLE_COL,
    "main_category",
    "store",
    "price",
    "source_category",
)

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")
_WHITESPACE_RE = re.compile(r"\s+")

# Plausible PCF range (kg CO2e); matches CarbonPredictor clip. Used to bound LLM
# outputs so zero-shot scale errors don't dominate metrics.
PCF_KG_MIN = 0.01
PCF_KG_MAX = 10_000.0


@dataclass
class RetrievalConfig:
    """Configuration for semantic-retrieval PCF estimation."""

    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    top_k: int = DEFAULT_TOP_K
    batch_size: int = 64
    retrieval_chunk_size: int = 2048
    max_text_chars: int = 1000
    use_amazon_metadata: bool = True
    use_carbon_context: bool = True
    device: str | None = None
    random_seed: int = 42
    deterministic: bool = True
    num_threads: int = 1


def set_global_determinism(
    seed: int,
    *,
    deterministic: bool = True,
    num_threads: int = 1,
) -> None:
    """Best-effort deterministic setup for Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if num_threads > 0:
        torch.set_num_threads(num_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class JsonlPredictionCache:
    """Append-only cache for prompt → numeric prediction lookups."""

    def __init__(self, path: Path | None) -> None:
        self._path = Path(path) if path is not None else None
        self._records: dict[str, float] = {}

        if self._path is not None and self._path.exists():
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    key = str(record["key"])
                    value = float(record["value"])
                    self._records[key] = value

    def get(self, key: str) -> float | None:
        return self._records.get(key)

    def set(self, key: str, value: float, **metadata: Any) -> None:
        if key in self._records:
            return

        self._records[key] = value
        if self._path is None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "value": float(value), **metadata}
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


class SentenceTransformerEncoder:
    """Lazy wrapper around `sentence-transformers`."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 64,
        random_seed: int = 42,
        deterministic: bool = True,
        num_threads: int = 1,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.deterministic = deterministic
        self.num_threads = num_threads
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            set_global_determinism(
                self.random_seed,
                deterministic=self.deterministic,
                num_threads=self.num_threads,
            )
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for retrieval-based PCF estimation. "
                    "Install the project requirements before running this pipeline."
                ) from exc

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.eval()
        return self._model

    def encode(
        self,
        texts: Sequence[str],
        *,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32, copy=False)


class OpenAILLMClient:
    """Thin OpenAI client that returns a single numeric prediction."""

    system_prompt = (
        "You estimate product carbon footprints in kilograms of CO2 equivalent. "
        "Return only one non-negative numeric value and no extra text."
    )

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.timeout = timeout
        self._client: Any | None = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def _load_client(self) -> Any:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. LLM baselines are unavailable."
            )

        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai is required for the LLM baselines. "
                    "Install the project requirements before running them."
                ) from exc

            kwargs: dict[str, Any] = {
                "api_key": self.api_key,
                "timeout": self.timeout,
            }
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def predict_numeric(self, prompt: str) -> float:
        client = self._load_client()

        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

        text = response.choices[0].message.content or ""
        return parse_numeric_response(text)


def parse_numeric_response(text: str) -> float:
    """Extract the first non-negative float from an LLM response and clamp to plausible PCF range."""
    match = _FLOAT_RE.search(text)
    if match is None:
        raise ValueError(f"Could not parse a numeric PCF from response: {text!r}")
    value = max(float(match.group(0)), 0.0)
    return float(np.clip(value, PCF_KG_MIN, PCF_KG_MAX))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def normalize_product_text(text: Any) -> str:
    """Lowercase and collapse whitespace for robust matching."""
    if _is_missing(text):
        return ""
    normalized = _WHITESPACE_RE.sub(" ", str(text).strip().lower())
    return normalized


def _flatten_metadata_value(value: Any) -> str:
    if _is_missing(value):
        return ""

    if isinstance(value, str):
        text = value.strip()
        if not text or text == "[]":
            return ""
        if text[0] in "[{" and text[-1] in "]}":
            try:
                value = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return text
        else:
            return text

    if isinstance(value, dict):
        preferred_keys = ("title", "name", "value", "text")
        parts: list[str] = []
        for key in preferred_keys:
            if key in value:
                flat = _flatten_metadata_value(value[key])
                if flat:
                    parts.append(flat)
        if parts:
            return " ".join(parts)
        return " ".join(
            flat for flat in (_flatten_metadata_value(v) for v in value.values()) if flat
        )

    if isinstance(value, (list, tuple, set)):
        return " ".join(
            flat for flat in (_flatten_metadata_value(v) for v in value) if flat
        )

    return str(value).strip()


def _compose_embedding_text(
    title: Any,
    extras: Sequence[Any],
    *,
    max_text_chars: int,
) -> str:
    parts = [_flatten_metadata_value(title)]
    parts.extend(_flatten_metadata_value(value) for value in extras)
    text = normalize_product_text(" ".join(part for part in parts if part))
    if not text:
        return "unknown product"
    return text[:max_text_chars]


def _unique_existing_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> list[str]:
    return list(dict.fromkeys(col for col in columns if col in df.columns))


def _prepare_frame(
    df: pd.DataFrame,
    *,
    id_col: str,
    title_col: str,
    extra_text_cols: Sequence[str],
    pcf_col: str | None = None,
    max_text_chars: int,
) -> pd.DataFrame:
    available_cols = [title_col, *[col for col in extra_text_cols if col in df.columns]]
    prepared = df.copy()

    if id_col not in prepared.columns:
        prepared[id_col] = prepared.index.astype(str)
    if title_col not in prepared.columns:
        prepared[title_col] = ""

    prepared[id_col] = prepared[id_col].astype(str)
    prepared[title_col] = prepared[title_col].fillna("").astype(str)
    prepared["normalized_title"] = prepared[title_col].map(normalize_product_text)

    rows = prepared[available_cols].itertuples(index=False, name=None)
    texts: list[str] = []
    for row in rows:
        title = row[0] if row else ""
        extras = row[1:] if len(row) > 1 else ()
        texts.append(
            _compose_embedding_text(
                title,
                extras,
                max_text_chars=max_text_chars,
            )
        )
    prepared["text_for_embedding"] = texts

    if pcf_col is not None:
        prepared["pcf"] = pd.to_numeric(prepared[pcf_col], errors="coerce")

    return prepared


def prepare_carbon_catalogue(
    carbon_df: pd.DataFrame,
    *,
    use_context: bool = True,
    max_text_chars: int = 1000,
) -> pd.DataFrame:
    extra_cols = CARBON_CONTEXT_COLS if use_context else ()
    keep_cols = _unique_existing_columns(
        carbon_df,
        (CARBON_ID_COL, CARBON_TITLE_COL, CARBON_PCF_COL, *extra_cols),
    )
    prepared = _prepare_frame(
        carbon_df[keep_cols].copy(),
        id_col=CARBON_ID_COL,
        title_col=CARBON_TITLE_COL,
        extra_text_cols=extra_cols,
        pcf_col=CARBON_PCF_COL,
        max_text_chars=max_text_chars,
    )
    prepared = prepared.loc[
        prepared["pcf"].notna() & (prepared["pcf"] >= 0)
    ].copy()
    prepared = prepared.loc[prepared["text_for_embedding"].str.len() > 0].copy()
    prepared["catalogue_index"] = np.arange(len(prepared))
    return prepared.reset_index(drop=True)


def prepare_amazon_metadata(
    amazon_df: pd.DataFrame,
    *,
    use_metadata: bool = True,
    max_text_chars: int = 1000,
) -> pd.DataFrame:
    working = amazon_df.copy()
    if AMAZON_ID_COL not in working.columns:
        working[AMAZON_ID_COL] = working.index.astype(str)
    if AMAZON_TITLE_COL not in working.columns:
        working[AMAZON_TITLE_COL] = ""

    extra_cols = AMAZON_METADATA_COLS if use_metadata else ()
    keep_cols = _unique_existing_columns(
        working,
        (
            AMAZON_ID_COL,
            AMAZON_TITLE_COL,
            "main_category",
            "store",
            "price",
            "source_category",
            *extra_cols,
        ),
    )
    prepared = _prepare_frame(
        working.drop_duplicates(subset=AMAZON_ID_COL)[keep_cols].copy(),
        id_col=AMAZON_ID_COL,
        title_col=AMAZON_TITLE_COL,
        extra_text_cols=extra_cols,
        pcf_col=None,
        max_text_chars=max_text_chars,
    )
    return prepared.reset_index(drop=True)


def build_zero_shot_prompt(product_title: str) -> str:
    """Prompt for the zero-shot LLM baseline."""
    return (
        "Estimate the Product Carbon Footprint (kg CO2e) for the following product. "
        "Typical values are between 1 and 10,000 kg CO2e for most consumer products.\n\n"
        f"Product: {product_title}\n\n"
        "Format: reply with exactly one number (e.g. 150 or 2.5), no scientific notation, "
        "no units or other text. The number is the PCF in kg CO2e."
    )


def build_few_shot_prompt(
    product_title: str,
    neighbors: Sequence[dict[str, Any]],
) -> str:
    """Prompt for retrieval-augmented few-shot estimation."""
    lines = ["Example products with known carbon footprint:"]
    for neighbor in neighbors:
        lines.extend(
            [
                "",
                f"Product: {neighbor['product_title']}",
                f"PCF: {neighbor['pcf']}",
            ]
        )

    lines.extend(
        [
            "",
            "Now estimate the Product Carbon Footprint (kg CO2e) for:",
            f"Product: {product_title}",
            "",
            "Return only a numeric value.",
        ]
    )
    return "\n".join(lines)


def _neighbor_examples_from_row(row: pd.Series, top_k: int) -> list[dict[str, Any]]:
    return [
        {
            "product_title": row[f"neighbor_{rank}_title"],
            "pcf": row[f"neighbor_{rank}_pcf"],
        }
        for rank in range(1, top_k + 1)
        if row.get(f"neighbor_{rank}_title", "")
    ]


def _build_llm_prompt(
    row: pd.Series,
    *,
    method_name: str,
    title_col: str,
    top_k: int,
) -> str:
    title = str(row[title_col])
    if method_name == "zero_shot_llm":
        return build_zero_shot_prompt(title)
    if method_name == "few_shot_llm":
        return build_few_shot_prompt(title, _neighbor_examples_from_row(row, top_k))
    raise ValueError(f"Unsupported LLM method: {method_name}")


def compute_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    """Compute RMSE, MAE, and Spearman correlation."""
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(true_arr) & np.isfinite(pred_arr)

    if not np.any(valid):
        return {"n": 0.0, "rmse": np.nan, "mae": np.nan, "spearman": np.nan}

    true_arr = true_arr[valid]
    pred_arr = pred_arr[valid]

    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    mae = float(np.mean(np.abs(pred_arr - true_arr)))
    spearman = float(
        pd.Series(true_arr).rank(method="average").corr(
            pd.Series(pred_arr).rank(method="average"),
            method="pearson",
        )
    )
    return {"n": float(len(true_arr)), "rmse": rmse, "mae": mae, "spearman": spearman}


def _top_k_cosine(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    *,
    top_k: int,
    chunk_size: int,
    exclude_reference_indices: np.ndarray | None = None,
    query_titles: Sequence[str] | None = None,
    reference_titles: Sequence[str] | None = None,
    exclude_exact_title_matches: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve top-k cosine matches using chunked matrix multiplies."""
    if reference_embeddings.size == 0:
        raise ValueError("Reference embeddings are empty.")

    k = min(top_k, reference_embeddings.shape[0])
    if len(query_embeddings) == 0:
        return (
            np.empty((0, k), dtype=int),
            np.empty((0, k), dtype=np.float32),
        )

    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    reference_titles_arr = (
        np.asarray(reference_titles, dtype=object)
        if reference_titles is not None
        else None
    )

    for start in range(0, len(query_embeddings), chunk_size):
        stop = min(start + chunk_size, len(query_embeddings))
        similarities = query_embeddings[start:stop] @ reference_embeddings.T

        if exclude_reference_indices is not None:
            excluded = exclude_reference_indices[start:stop]
            valid = (excluded >= 0) & (excluded < reference_embeddings.shape[0])
            similarities[np.arange(stop - start)[valid], excluded[valid]] = -np.inf

        if exclude_exact_title_matches and query_titles is not None and reference_titles_arr is not None:
            for local_idx, title in enumerate(query_titles[start:stop]):
                if title:
                    similarities[local_idx, reference_titles_arr == title] = -np.inf

        # Stable sort keeps reference-index order for exact score ties.
        idx = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
        scores = np.take_along_axis(similarities, idx, axis=1)

        invalid = ~np.isfinite(scores)
        idx = np.where(invalid, -1, idx)
        scores = np.where(invalid, np.nan, scores)

        all_indices.append(idx)
        all_scores.append(scores.astype(np.float32))

    return np.vstack(all_indices), np.vstack(all_scores)


def _build_neighbor_columns(
    *,
    query_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    reference_id_col: str,
    reference_title_col: str,
) -> pd.DataFrame:
    result = query_df.copy().reset_index(drop=True)
    n_rows, k = neighbor_indices.shape

    ref_ids = reference_df[reference_id_col].astype(str).to_numpy()
    ref_titles = reference_df[reference_title_col].astype(str).to_numpy()
    ref_pcfs = reference_df["pcf"].astype(float).to_numpy()
    pcf_matrix = np.full((n_rows, k), np.nan, dtype=float)

    for rank in range(k):
        idx = neighbor_indices[:, rank]
        valid = idx >= 0

        id_values = np.full(n_rows, "", dtype=object)
        title_values = np.full(n_rows, "", dtype=object)
        pcf_values = np.full(n_rows, np.nan, dtype=float)
        similarity_values = np.full(n_rows, np.nan, dtype=float)

        if np.any(valid):
            id_values[valid] = ref_ids[idx[valid]]
            title_values[valid] = ref_titles[idx[valid]]
            pcf_values[valid] = ref_pcfs[idx[valid]]
            similarity_values[valid] = neighbor_scores[valid, rank]
            pcf_matrix[valid, rank] = pcf_values[valid]

        result[f"neighbor_{rank + 1}_id"] = id_values
        result[f"neighbor_{rank + 1}_title"] = title_values
        result[f"neighbor_{rank + 1}_pcf"] = pcf_values
        result[f"neighbor_{rank + 1}_similarity"] = similarity_values

    with np.errstate(invalid="ignore"):
        result["neighbor_average_pcf"] = np.nanmean(pcf_matrix, axis=1)

    return result


def _build_empty_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": method_name,
                "n_examples": 0,
                "rmse": np.nan,
                "mae": np.nan,
                "spearman": np.nan,
                "available": False,
            }
            for method_name in PREDICTION_METHOD_TO_COLUMN
        ]
    )


def _build_metrics_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_name, pred_col in PREDICTION_METHOD_TO_COLUMN.items():
        stats = compute_regression_metrics(predictions_df["pcf"], predictions_df[pred_col])
        rows.append(
            {
                "method": method_name,
                "n_examples": int(stats["n"]),
                "rmse": stats["rmse"],
                "mae": stats["mae"],
                "spearman": stats["spearman"],
                "available": bool(stats["n"]),
            }
        )
    return pd.DataFrame(rows)


def _select_final_pcf(df: pd.DataFrame) -> pd.DataFrame:
    few_shot_available = df["few_shot_llm_pcf"].notna()
    df[DEFAULT_SELECTED_PCF_COL] = np.where(
        few_shot_available,
        df["few_shot_llm_pcf"],
        df["neighbor_average_pcf"],
    )
    df["pcf_source"] = np.where(
        few_shot_available,
        "few_shot_llm",
        "neighbor_average",
    )
    return df


def _amazon_output_columns(df: pd.DataFrame, top_k: int) -> list[str]:
    neighbor_cols = [
        f"neighbor_{rank}_{suffix}"
        for rank in range(1, top_k + 1)
        for suffix in NEIGHBOR_OUTPUT_SUFFIXES
    ]
    prediction_cols = [
        "neighbor_average_pcf",
        "zero_shot_llm_pcf",
        "few_shot_llm_pcf",
        DEFAULT_SELECTED_PCF_COL,
        "pcf_source",
    ]
    return _unique_existing_columns(
        df,
        (*AMAZON_OUTPUT_BASE_COLS, *neighbor_cols, *prediction_cols),
    )


def _empty_amazon_predictions(top_k: int) -> pd.DataFrame:
    neighbor_cols = [
        f"neighbor_{rank}_{suffix}"
        for rank in range(1, top_k + 1)
        for suffix in NEIGHBOR_OUTPUT_SUFFIXES
    ]
    prediction_cols = [
        "neighbor_average_pcf",
        "zero_shot_llm_pcf",
        "few_shot_llm_pcf",
        DEFAULT_SELECTED_PCF_COL,
        "pcf_source",
    ]
    columns = [
        *AMAZON_OUTPUT_BASE_COLS,
        *neighbor_cols,
        *prediction_cols,
    ]
    return pd.DataFrame(columns=columns)


class PCFRetrievalEstimator:
    """Semantic-retrieval PCF estimator with optional LLM baselines."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        *,
        encoder: SentenceTransformerEncoder | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        set_global_determinism(
            self.config.random_seed,
            deterministic=self.config.deterministic,
            num_threads=self.config.num_threads,
        )
        self.encoder = encoder or SentenceTransformerEncoder(
            self.config.embedding_model_name,
            device=self.config.device,
            batch_size=self.config.batch_size,
            random_seed=self.config.random_seed,
            deterministic=self.config.deterministic,
            num_threads=self.config.num_threads,
        )
        self._carbon_catalogue: pd.DataFrame | None = None
        self._carbon_embeddings: np.ndarray | None = None

    @property
    def carbon_catalogue(self) -> pd.DataFrame:
        if self._carbon_catalogue is None:
            raise RuntimeError("Carbon Catalogue is not loaded. Call fit_carbon_catalogue() first.")
        return self._carbon_catalogue

    @property
    def carbon_embeddings(self) -> np.ndarray:
        if self._carbon_embeddings is None:
            raise RuntimeError("Carbon Catalogue embeddings are not built yet.")
        return self._carbon_embeddings

    def fit_carbon_catalogue(
        self,
        carbon_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Prepare the Carbon Catalogue and build title embeddings."""
        if carbon_df is None:
            carbon_df = load_carbon_catalogue()

        prepared = prepare_carbon_catalogue(
            carbon_df,
            use_context=self.config.use_carbon_context,
            max_text_chars=self.config.max_text_chars,
        )
        embeddings = self.encoder.encode(
            prepared["text_for_embedding"].tolist(),
            show_progress_bar=True,
        )

        self._carbon_catalogue = prepared
        self._carbon_embeddings = embeddings

        log.info(
            "Built Carbon Catalogue index: %s products, %s-dimensional embeddings",
            f"{len(prepared):,}",
            embeddings.shape[1] if embeddings.ndim == 2 else 0,
        )
        return prepared

    def retrieve_neighbors(
        self,
        query_df: pd.DataFrame,
        query_embeddings: np.ndarray,
        *,
        exclude_reference_indices: np.ndarray | None = None,
        exclude_exact_title_matches: bool = False,
    ) -> pd.DataFrame:
        """Retrieve top-k Carbon Catalogue neighbors for each query row."""
        neighbor_idx, neighbor_scores = _top_k_cosine(
            query_embeddings,
            self.carbon_embeddings,
            top_k=self.config.top_k,
            chunk_size=self.config.retrieval_chunk_size,
            exclude_reference_indices=exclude_reference_indices,
            query_titles=query_df["normalized_title"].tolist(),
            reference_titles=self.carbon_catalogue["normalized_title"].tolist(),
            exclude_exact_title_matches=exclude_exact_title_matches,
        )
        return _build_neighbor_columns(
            query_df=query_df,
            reference_df=self.carbon_catalogue,
            neighbor_indices=neighbor_idx,
            neighbor_scores=neighbor_scores,
            reference_id_col=CARBON_ID_COL,
            reference_title_col=CARBON_TITLE_COL,
        )

    def _run_llm_predictions(
        self,
        df: pd.DataFrame,
        *,
        method_name: str,
        title_col: str,
        llm_client: OpenAILLMClient | None,
        llm_model_name: str | None,
        cache_path: Path | None,
        limit: int | None,
        cache_only: bool = False,
    ) -> np.ndarray:
        predictions = np.full(len(df), np.nan, dtype=float)
        if llm_client is None and not cache_only:
            return predictions
        if llm_client is not None and not llm_client.is_available and not cache_only:
            return predictions

        if limit == 0:
            return predictions

        run_limit = len(df) if limit is None else min(limit, len(df))
        cache = JsonlPredictionCache(cache_path)
        model_name = (
            llm_client.model
            if llm_client is not None
            else (llm_model_name or DEFAULT_LLM_MODEL)
        )
        log.info("Running %s LLM predictions for %d items", method_name, run_limit)

        for row_idx in range(run_limit):
            row = df.iloc[row_idx]
            prompt = _build_llm_prompt(
                row,
                method_name=method_name,
                title_col=title_col,
                top_k=self.config.top_k,
            )

            cache_key = hashlib.sha256(
                f"{model_name}\n{method_name}\n{prompt}".encode("utf-8")
            ).hexdigest()
            cached = cache.get(cache_key)
            if cached is not None:
                predictions[row_idx] = cached
                continue
            if cache_only:
                continue

            try:
                if llm_client is None:
                    raise RuntimeError("Live LLM prediction requested without an LLM client.")
                value = llm_client.predict_numeric(prompt)
            except Exception as exc:
                log.warning(
                    "LLM prediction failed for %s row %d: %s",
                    method_name,
                    row_idx,
                    exc,
                )
                continue

            predictions[row_idx] = value
            cache.set(
                cache_key,
                value,
                method=method_name,
                model=model_name,
            )

        return predictions

    def _apply_llm_methods(
        self,
        df: pd.DataFrame,
        *,
        title_col: str,
        llm_client: OpenAILLMClient | None,
        llm_model_name: str | None,
        llm_cache_path: Path | None,
        llm_limit: int | None,
        llm_cache_only: bool,
    ) -> pd.DataFrame:
        result = df.copy()
        for method_name in LLM_METHODS:
            result[PREDICTION_METHOD_TO_COLUMN[method_name]] = self._run_llm_predictions(
                result,
                method_name=method_name,
                title_col=title_col,
                llm_client=llm_client,
                llm_model_name=llm_model_name,
                cache_path=llm_cache_path,
                limit=llm_limit,
                cache_only=llm_cache_only,
            )
        return result

    def evaluate_on_carbon_catalogue(
        self,
        *,
        limit: int | None = None,
        random_state: int = 42,
        llm_client: OpenAILLMClient | None = None,
        llm_model_name: str | None = None,
        llm_cache_path: Path | None = None,
        llm_cache_only: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate nearest-neighbor, zero-shot LLM, and few-shot LLM methods.

        The Carbon Catalogue is used as a leave-one-out evaluation set.
        """
        catalogue = self.carbon_catalogue.copy()
        if limit is not None and limit < len(catalogue):
            catalogue = catalogue.sample(n=limit, random_state=random_state).sort_index()

        if catalogue.empty:
            return catalogue, _build_empty_metrics_frame()

        query_embeddings = self.carbon_embeddings[catalogue["catalogue_index"].to_numpy()]
        retrieved = self.retrieve_neighbors(
            catalogue,
            query_embeddings,
            exclude_reference_indices=catalogue["catalogue_index"].to_numpy(),
            exclude_exact_title_matches=True,
        )
        retrieved = self._apply_llm_methods(
            retrieved,
            title_col=CARBON_TITLE_COL,
            llm_client=llm_client,
            llm_model_name=llm_model_name,
            llm_cache_path=llm_cache_path,
            llm_limit=len(retrieved),
            llm_cache_only=llm_cache_only,
        )
        return retrieved, _build_metrics_frame(retrieved)

    def predict_amazon_products(
        self,
        amazon_df: pd.DataFrame,
        *,
        llm_client: OpenAILLMClient | None = None,
        llm_model_name: str | None = None,
        llm_cache_path: Path | None = None,
        llm_limit: int | None = None,
        llm_cache_only: bool = False,
    ) -> pd.DataFrame:
        """Predict PCF for Amazon products using retrieval and optional LLMs."""
        prepared = prepare_amazon_metadata(
            amazon_df,
            use_metadata=self.config.use_amazon_metadata,
            max_text_chars=self.config.max_text_chars,
        )
        if prepared.empty:
            return _empty_amazon_predictions(self.config.top_k)
        query_embeddings = self.encoder.encode(
            prepared["text_for_embedding"].tolist(),
            show_progress_bar=True,
        )
        retrieved = self.retrieve_neighbors(prepared, query_embeddings)
        retrieved = self._apply_llm_methods(
            retrieved,
            title_col=AMAZON_TITLE_COL,
            llm_client=llm_client,
            llm_model_name=llm_model_name,
            llm_cache_path=llm_cache_path,
            llm_limit=llm_limit,
            llm_cache_only=llm_cache_only,
        )
        retrieved = _select_final_pcf(retrieved)
        return retrieved[_amazon_output_columns(retrieved, self.config.top_k)].copy()

    @classmethod
    def load_all_amazon_metadata(cls) -> pd.DataFrame:
        """Load and concatenate all Amazon metadata categories."""
        meta_frames = []
        for category, frame in load_all_meta().items():
            copy = frame.copy()
            copy["source_category"] = category
            meta_frames.append(copy)
        return pd.concat(meta_frames, ignore_index=True)
