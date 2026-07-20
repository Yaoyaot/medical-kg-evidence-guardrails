from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

from repo_paths import find_repo_root


ROOT = find_repo_root()
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_eswa_major_revision import (  # noqa: E402
    risk_ranking_metrics,
    select_fold_budget,
    selective_metrics,
)


DEFAULT_INPUT = (
    ROOT
    / "data/processed/stage11_eswa_nested_crossfit/formal600/"
    "formal600_crossfit_risk_scores.jsonl"
)
DEFAULT_OUTPUT = ROOT / "data/processed/stage12_eswa_fold_component_audit"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fold_rows(rows: list[dict]) -> list[dict]:
    output = []
    for fold in range(5):
        subset = [row for row in rows if int(row["outer_fold"]) == fold]
        labels = Counter(row["gold_label"] for row in subset)
        datasets = Counter(row["dataset"].lower() for row in subset)
        support = [row for row in subset if row["candidate_label"] == "SUPPORT"]
        false_support = [row for row in support if row["gold_label"] != "SUPPORT"]
        output.append(
            {
                "fold": fold,
                "components": len({row["pair_group_id"] for row in subset}),
                "records": len(subset),
                "support_gold": labels["SUPPORT"],
                "refute_gold": labels["REFUTE"],
                "uncertain_gold": labels["UNCERTAIN"],
                "healthver": datasets["healthver"],
                "medaesqa": datasets["medaesqa"],
                "scifact": datasets["scifact"],
                "support_predictions": len(support),
                "false_support_predictions": len(false_support),
                "nominal_5pct_review_count": min(
                    len(support), math.ceil(0.05 * len(subset))
                ),
            }
        )
    return output


def sensitivity_row(scope: str, rows: list[dict], method: str, field: str) -> dict:
    ranking = risk_ranking_metrics(rows, field, method)
    reviewed = select_fold_budget(rows, field, 0.05)
    selective = selective_metrics(rows, reviewed)
    return {
        "scope": scope,
        "method": method,
        "records": len(rows),
        "support_predictions": ranking["support_predictions"],
        "false_support_predictions": ranking["error_events"],
        "auroc": ranking["auroc"],
        "average_precision": ranking["average_precision"],
        "reviewed_count": selective["reviewed_count"],
        "captured_false_support": selective["detected_false_supports"],
        "reviewed_true_support": selective[
            "incorrectly_reviewed_correct_supports"
        ],
        "detection_precision": selective["error_detection_precision"],
        "detection_recall": selective["error_detection_recall"],
        "selective_risk": selective["selective_risk"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report Formal600 fold composition and largest-component sensitivity."
    )
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_INPUT.relative_to(ROOT)),
        help="Frozen strict nested cross-fit risk-score JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT.relative_to(ROOT)),
        help="Directory for fold and sensitivity summaries.",
    )
    args = parser.parse_args()
    input_path = Path(args.input_path)
    output = Path(args.output_dir)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not output.is_absolute():
        output = ROOT / output

    rows = read_jsonl(input_path)
    if len(rows) != 600:
        raise AssertionError(f"Expected 600 records, found {len(rows)}")
    component_counts = Counter(row["pair_group_id"] for row in rows)
    largest_id, largest_n = component_counts.most_common(1)[0]
    largest_rows = [row for row in rows if row["pair_group_id"] == largest_id]
    largest_folds = {int(row["outer_fold"]) for row in largest_rows}
    if len(largest_folds) != 1 or largest_n != 94:
        raise AssertionError(
            f"Unexpected largest component: {largest_id}, n={largest_n}, folds={largest_folds}"
        )

    folds = fold_rows(rows)
    without_largest = [row for row in rows if row["pair_group_id"] != largest_id]
    sensitivity = []
    methods = (
        ("learned", "risk_full_without_dataset_source"),
        ("self_reported_confidence", "risk_confidence"),
    )
    for scope, subset in (("full600", rows), ("without_largest_component", without_largest)):
        for method, field in methods:
            sensitivity.append(sensitivity_row(scope, subset, method, field))

    full = {row["method"]: row for row in sensitivity if row["scope"] == "full600"}
    reduced = {
        row["method"]: row
        for row in sensitivity
        if row["scope"] == "without_largest_component"
    }
    stats = {
        "seed": 20260618,
        "input_records": len(rows),
        "components": len(component_counts),
        "largest_component": {
            "id": largest_id,
            "records": largest_n,
            "share": largest_n / len(rows),
            "fold": next(iter(largest_folds)),
            "datasets": dict(Counter(row["dataset"].lower() for row in largest_rows)),
            "labels": dict(Counter(row["gold_label"] for row in largest_rows)),
            "support_predictions": sum(
                row["candidate_label"] == "SUPPORT" for row in largest_rows
            ),
            "false_support_predictions": sum(
                row["candidate_label"] == "SUPPORT"
                and row["gold_label"] != "SUPPORT"
                for row in largest_rows
            ),
        },
        "sensitivity_protocol": (
            "Remove the largest component from evaluation only; retain frozen OOF "
            "scores and recompute fold-specific 5% budgets and rankings among remaining records."
        ),
        "full_learned_minus_confidence_auroc": (
            full["learned"]["auroc"] - full["self_reported_confidence"]["auroc"]
        ),
        "without_largest_learned_minus_confidence_auroc": (
            reduced["learned"]["auroc"]
            - reduced["self_reported_confidence"]["auroc"]
        ),
        "full_learned_minus_confidence_detection_recall": (
            full["learned"]["detection_recall"]
            - full["self_reported_confidence"]["detection_recall"]
        ),
        "without_largest_learned_minus_confidence_detection_recall": (
            reduced["learned"]["detection_recall"]
            - reduced["self_reported_confidence"]["detection_recall"]
        ),
        "point_advantage_direction_preserved": False,
        "model_retraining": False,
        "llm_api_calls": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "formal600_fold_composition.csv", folds)
    write_csv(output / "largest_component_sensitivity.csv", sensitivity)
    (output / "fold_component_audit_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
