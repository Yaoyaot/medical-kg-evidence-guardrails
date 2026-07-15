from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from stage3_path_reranker_utils import read_csv, write_csv


RELEVANCE = {"RELEVANT", "PARTIAL", "IRRELEVANT"}
ROLES = {"SUPPORT", "REFUTE", "CONTEXT_ONLY", "UNKNOWN"}
ERROR_TYPES = {
    "NONE",
    "ENTITY_LINKING",
    "RELATION_MISMATCH",
    "INDIRECT_BRIDGE",
    "POLARITY",
    "INSUFFICIENT_SPECIFICITY",
    "OTHER",
}
REQUIRED_COLUMNS = {
    "annotation_id",
    "claim_id",
    "dataset",
    "gold_label",
    "claim",
    "path_text",
    "path_relevance",
    "evidence_role",
    "error_type",
}


def normalize(value: str) -> str:
    return (value or "").strip().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate completed KG path annotations.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--allow-unlabeled", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.data_dir) / "processed/stage3_path_reranker"
    path = output_dir / f"path_annotation_batch_{args.sample_size}.csv"
    rows = read_csv(path)
    errors = []
    if len(rows) != args.sample_size:
        errors.append(f"Expected {args.sample_size} rows, found {len(rows)}.")
    missing_columns = sorted(REQUIRED_COLUMNS - set(rows[0] if rows else []))
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    normalized_rows = []
    unlabeled = 0
    for index, row in enumerate(rows, start=2):
        item = dict(row)
        for field in ("path_relevance", "evidence_role", "error_type"):
            item[field] = normalize(item.get(field, ""))
        values = [item["path_relevance"], item["evidence_role"], item["error_type"]]
        if not any(values):
            unlabeled += 1
        elif not all(values):
            errors.append(f"Row {index}: annotation fields must be filled together.")
        if item["path_relevance"] and item["path_relevance"] not in RELEVANCE:
            errors.append(f"Row {index}: invalid path_relevance={item['path_relevance']!r}.")
        if item["evidence_role"] and item["evidence_role"] not in ROLES:
            errors.append(f"Row {index}: invalid evidence_role={item['evidence_role']!r}.")
        if item["error_type"] and item["error_type"] not in ERROR_TYPES:
            errors.append(f"Row {index}: invalid error_type={item['error_type']!r}.")
        normalized_rows.append(item)

    relevance_counts = Counter(row["path_relevance"] for row in normalized_rows if row["path_relevance"])
    if not args.allow_unlabeled:
        if unlabeled:
            errors.append(f"{unlabeled} rows are still unlabeled.")
        if relevance_counts["RELEVANT"] < 10:
            errors.append("Need at least 10 RELEVANT paths before training.")
        if relevance_counts["IRRELEVANT"] < 10:
            errors.append("Need at least 10 IRRELEVANT paths before training.")

    stats = {
        "input_rows": len(rows),
        "allow_unlabeled": args.allow_unlabeled,
        "unlabeled_rows": unlabeled,
        "complete_rows": len(rows) - unlabeled,
        "relevance_counts": dict(relevance_counts),
        "evidence_role_counts": dict(Counter(row["evidence_role"] for row in normalized_rows if row["evidence_role"])),
        "error_type_counts": dict(Counter(row["error_type"] for row in normalized_rows if row["error_type"])),
        "dataset_counts": dict(Counter(row.get("dataset") for row in normalized_rows)),
        "valid": not errors,
        "errors": errors[:50],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "path_annotation_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    if not errors and not args.allow_unlabeled:
        write_csv(output_dir / "path_annotations_validated.csv", normalized_rows)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
