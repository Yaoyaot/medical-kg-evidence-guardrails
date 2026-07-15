from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 20260618
METHODS = (
    "confidence_only_features",
    "dataset_source_only",
    "kg_evidence_state_only",
    "semantic_rules_only",
    "path_statistics_only",
    "evidence_scorer_only",
    "confidence_plus_kg_rules",
    "confidence_plus_evidence_scorer",
    "full_without_dataset_source",
    "full_with_dataset_source",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def statistics(rows: list[dict], method: str, indices: np.ndarray | None = None) -> dict[str, float]:
    if indices is None:
        indices = np.arange(len(rows))
    support_indices = np.asarray([index for index in indices if rows[index]["candidate_label"] == "SUPPORT"], dtype=np.int32)
    y = np.asarray([int(rows[index]["gold_label"] != "SUPPORT") for index in support_indices])
    scores = np.asarray([float(rows[index][f"risk_{method}"]) for index in support_indices])
    auroc = float(roc_auc_score(y, scores)) if len(set(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, scores)) if len(set(y)) > 1 else float("nan")
    reviewed: set[int] = set()
    for fold in sorted({int(rows[index]["outer_fold"]) for index in indices}):
        fold_all = [index for index in indices if int(rows[index]["outer_fold"]) == fold]
        fold_support = [index for index in fold_all if rows[index]["candidate_label"] == "SUPPORT"]
        budget = min(len(fold_support), math.ceil(0.05 * len(fold_all)))
        ranked = sorted(fold_support, key=lambda index: (-float(rows[index][f"risk_{method}"]), str(rows[index]["id"])))
        reviewed.update(ranked[:budget])
    errors = {index for index in support_indices if rows[index]["gold_label"] != "SUPPORT"}
    detected = len(errors & reviewed)
    accepted = [index for index in indices if index not in reviewed]
    accepted_errors = sum(rows[index]["candidate_label"] != rows[index]["gold_label"] for index in accepted)
    return {
        "auroc": auroc,
        "average_precision": ap,
        "reviewed": len(reviewed),
        "error_detection_precision": detected / len(reviewed) if reviewed else 0.0,
        "error_detection_recall": detected / len(errors) if errors else 0.0,
        "selective_risk": accepted_errors / len(accepted) if accepted else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize risk ablations with group-bootstrap intervals.")
    parser.add_argument("--input", default="data/processed/stage9_eswa_major_revision/formal600_crossfit/formal600_crossfit_risk_scores.jsonl")
    parser.add_argument("--output-dir", default="data/processed/stage9_eswa_major_revision/formal600_crossfit")
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()
    rows = read_jsonl(Path(args.input))
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["pair_group_id"]].append(index)
    group_keys = sorted(groups)
    rng = random.Random(SEED)
    boot = {method: defaultdict(list) for method in METHODS}
    for _ in range(args.iterations):
        sampled = []
        for group in rng.choices(group_keys, k=len(group_keys)):
            sampled.extend(groups[group])
        sampled_rows = [rows[index] for index in sampled]
        for method in METHODS:
            values = statistics(sampled_rows, method)
            for metric in ("auroc", "average_precision", "error_detection_precision", "error_detection_recall", "selective_risk"):
                if not math.isnan(values[metric]):
                    boot[method][metric].append(values[metric])

    output = []
    for method in METHODS:
        point = statistics(rows, method)
        row = {"method": method, **point, "iterations": args.iterations, "resampling_unit": "claim_source_component"}
        for metric in ("auroc", "average_precision", "error_detection_precision", "error_detection_recall", "selective_risk"):
            values = boot[method][metric]
            row[f"{metric}_ci_low"] = percentile(values, 0.025)
            row[f"{metric}_ci_high"] = percentile(values, 0.975)
        output.append(row)
    out = Path(args.output_dir)
    write_csv(out / "formal600_risk_ablation_summary.csv", output)
    print(json.dumps({"methods": len(output), "iterations": args.iterations, "output": str(out / 'formal600_risk_ablation_summary.csv')}, indent=2))


if __name__ == "__main__":
    main()
