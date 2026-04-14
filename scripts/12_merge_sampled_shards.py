#!/usr/bin/env python
"""
Merge sampled-shard Electronics PCF prediction outputs.

This script is intended for outputs produced by:
  notebooks/colab_pcf_llm_few_shot_electronics.ipynb

It merges multiple shard CSVs into a single CSV and writes merge metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge sampled shard prediction CSVs into one file.",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Explicit shard CSV paths. If omitted, --input-dir and --glob are used.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory to search for shard CSVs when --inputs is omitted.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="predictions_*_sampled_shard.csv",
        help="Glob pattern under --input-dir for shard CSVs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for merged CSV output.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Optional path for merge metadata JSON "
        "(default: <output_stem>_merge_metadata.json).",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow duplicate parent_asin rows across shard files.",
    )
    return parser.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        paths = [Path(p).expanduser().resolve() for p in args.inputs]
    else:
        if args.input_dir is None:
            raise ValueError("Provide --inputs or --input-dir.")
        input_dir = args.input_dir.expanduser().resolve()
        paths = sorted(input_dir.glob(args.glob))

    if not paths:
        raise FileNotFoundError("No shard CSV files found to merge.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Some input files do not exist:\n" + "\n".join(missing)
        )
    return paths


def main() -> None:
    args = parse_args()
    input_paths = _resolve_inputs(args)

    frames = []
    input_rows: dict[str, int] = {}
    for path in input_paths:
        frame = pd.read_csv(path)
        frames.append(frame)
        input_rows[str(path)] = int(len(frame))

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    merged_rows = int(len(merged))
    duplicate_count = 0

    if "parent_asin" in merged.columns:
        duplicate_mask = merged["parent_asin"].duplicated(keep=False)
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count and not args.allow_duplicates:
            duplicates = (
                merged.loc[duplicate_mask, "parent_asin"]
                .astype(str)
                .drop_duplicates()
                .head(20)
                .tolist()
            )
            preview = "\n".join(duplicates)
            raise ValueError(
                "Duplicate parent_asin values found across shard files. "
                "Rerun with --allow-duplicates if this is intentional.\n"
                f"Duplicate rows: {duplicate_count}\n"
                f"Example duplicate parent_asin values:\n{preview}"
            )
        merged = merged.sort_values("parent_asin").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    metadata_output = (
        args.metadata_output
        if args.metadata_output is not None
        else args.output.with_name(f"{args.output.stem}_merge_metadata.json")
    )
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    merge_metadata = {
        "inputs": [str(path) for path in input_paths],
        "input_rows": input_rows,
        "num_input_files": int(len(input_paths)),
        "merged_rows": merged_rows,
        "duplicate_parent_asin_rows": duplicate_count,
        "allow_duplicates": bool(args.allow_duplicates),
        "output": str(args.output),
    }
    with metadata_output.open("w", encoding="utf-8") as handle:
        json.dump(merge_metadata, handle, indent=2)

    print(f"Merged {len(input_paths)} shard files -> {args.output}")
    print(f"Merged rows: {merged_rows:,}")
    print(f"Merge metadata: {metadata_output}")


if __name__ == "__main__":
    main()
