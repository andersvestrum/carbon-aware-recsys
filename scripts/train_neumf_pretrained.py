"""
Train NeuMF properly via the He et al. (2017) pretraining recipe.

Stage 1: GMF pretraining (matrix-factorization branch only).
Stage 2: MLP pretraining (multi-layer perceptron branch only).
Stage 3: Fused NeuMF fine-tuning, initialized from stage-1 and stage-2
         checkpoints.

Each stage uses k-core filtering (≥5 interactions per user/item) and 4
negative samples per positive — both deviations from the current
configs/recbole/neumf.yaml that are critical for NeuMF on sparse data.

Final scores are saved to:
    output/results/{dataset}_NeuMF_pretrained_scores.parquet

Usage:
    python scripts/train_neumf_pretrained.py --category electronics --max-users 3000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# PyTorch 2.6+ defaults torch.load to weights_only=True, which can't
# unpickle RecBole checkpoints. NeuMF.load_pretrain calls torch.load
# directly without overriding this. Force the legacy behaviour for
# all torch.load calls in this process before importing recbole.
import torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("neumf-pretrain")

CONFIG_DIR = PROJECT_ROOT / "configs" / "recbole"
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
MODEL_DIR = PROJECT_ROOT / "output" / "models" / "recbole_checkpoints"


def _train_stage(
    dataset_name: str,
    config_file: Path,
    overrides: dict[str, Any],
    stage_label: str,
):
    """Run one training stage and return (model, dataset, config, saved_path)."""
    from src.recommender.trainer import build_config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer

    log.info("=" * 70)
    log.info("Stage: %s", stage_label)
    log.info("=" * 70)

    config = build_config(
        dataset_name=dataset_name,
        model_name="NeuMF",
        config_file=config_file,
        overrides=overrides,
    )

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model_class = get_model("NeuMF")
    model = model_class(config, train_data.dataset).to(config["device"])

    trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
    trainer = trainer_class(config, model)

    log.info("Training NeuMF (%s) for up to %d epochs …",
             stage_label, config["epochs"])
    best_valid_score, _ = trainer.fit(
        train_data, valid_data, saved=True, verbose=True,
    )
    log.info("Best validation NDCG@10 (%s): %.4f", stage_label, best_valid_score)

    try:
        test_result = trainer.evaluate(test_data, load_best_model=True)
    except Exception as exc:
        log.warning("Could not reload best checkpoint (%r); using in-memory.", exc)
        test_result = trainer.evaluate(test_data, load_best_model=False)
    log.info("%s test: %s", stage_label, dict(test_result))

    saved_path = Path(trainer.saved_model_file)
    log.info("Checkpoint → %s", saved_path)

    return model, dataset, config, test_result, saved_path


def _extract_and_save_scores(
    model, dataset, config, dataset_name: str, label: str, top_k: int = 100,
):
    from src.recommender.trainer import extract_relevance_scores

    scores_df = extract_relevance_scores(model, dataset, config, top_k)
    out_path = RESULTS_DIR / f"{dataset_name}_{label}_scores.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(out_path, index=False)
    log.info("Scores → %s  (%d rows)", out_path, len(scores_df))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="electronics")
    ap.add_argument("--max-users", type=int, default=3000,
                    help="Subset users for faster runs (None = full dataset)")
    args = ap.parse_args()

    from src.recommender.recbole_formatter import format_category_for_recbole

    log.info("Formatting %s (max_users=%s) …", args.category, args.max_users)
    _, dataset_name = format_category_for_recbole(
        args.category,
        max_users=args.max_users if args.max_users > 0 else None,
    )

    # ── Stage 1: GMF pretraining ───────────────────────────────────────
    _, _, _, gmf_test, gmf_ckpt = _train_stage(
        dataset_name=dataset_name,
        config_file=CONFIG_DIR / "neumf_gmf_pretrain.yaml",
        overrides={},
        stage_label="GMF pretrain",
    )

    # ── Stage 2: MLP pretraining ───────────────────────────────────────
    _, _, _, mlp_test, mlp_ckpt = _train_stage(
        dataset_name=dataset_name,
        config_file=CONFIG_DIR / "neumf_mlp_pretrain.yaml",
        overrides={},
        stage_label="MLP pretrain",
    )

    # ── Stage 3: Fused fine-tuning ─────────────────────────────────────
    fused_overrides = {
        "mf_pretrain_path": str(gmf_ckpt),
        "mlp_pretrain_path": str(mlp_ckpt),
    }
    fused_model, fused_dataset, fused_config, fused_test, fused_ckpt = _train_stage(
        dataset_name=dataset_name,
        config_file=CONFIG_DIR / "neumf_pretrained.yaml",
        overrides=fused_overrides,
        stage_label="NeuMF fused fine-tune",
    )

    # ── Score extraction ───────────────────────────────────────────────
    _extract_and_save_scores(
        fused_model, fused_dataset, fused_config,
        dataset_name=dataset_name,
        label="NeuMF_pretrained",
    )

    # ── Summary ────────────────────────────────────────────────────────
    summary = {
        "category": args.category,
        "dataset": dataset_name,
        "max_users": args.max_users,
        "stage_1_gmf": {k: float(v) for k, v in gmf_test.items()},
        "stage_2_mlp": {k: float(v) for k, v in mlp_test.items()},
        "stage_3_fused": {k: float(v) for k, v in fused_test.items()},
        "checkpoints": {
            "gmf": str(gmf_ckpt),
            "mlp": str(mlp_ckpt),
            "fused": str(fused_ckpt),
        },
    }
    summary_path = RESULTS_DIR / f"{dataset_name}_NeuMF_pretrained_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary → %s", summary_path)

    print("\n" + "=" * 70)
    print("PRETRAINING COMPLETE")
    print("=" * 70)
    for stage, result in [
        ("Stage 1 GMF", gmf_test),
        ("Stage 2 MLP", mlp_test),
        ("Stage 3 Fused", fused_test),
    ]:
        ndcg = result.get("ndcg@10", "?")
        print(f"  {stage:20s}  NDCG@10 = {ndcg}")


if __name__ == "__main__":
    main()
