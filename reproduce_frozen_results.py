from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LABELS = ("SUPPORT", "REFUTE", "UNCERTAIN")
RISK_FIELDS = {
    "self_reported_confidence": "risk_confidence",
    "rule_matched_budget": "risk_rule",
    "confidence_only_features": "risk_confidence_only_features",
    "dataset_source_only": "risk_dataset_source_only",
    "kg_evidence_state_only": "risk_kg_evidence_state_only",
    "semantic_rules_only": "risk_semantic_rules_only",
    "path_statistics_only": "risk_path_statistics_only",
    "evidence_scorer_only": "risk_evidence_scorer_only",
    "confidence_plus_kg_rules": "risk_confidence_plus_kg_rules",
    "confidence_plus_evidence_scorer": "risk_confidence_plus_evidence_scorer",
    "full_without_dataset_source": "risk_full_without_dataset_source",
    "full_with_dataset_source": "risk_full_with_dataset_source",
    "oracle": "risk_oracle",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def macro_f1(gold: list[str], pred: list[str]) -> float:
    values = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return sum(values) / len(values)


def classification_metrics(domain: str, method: str, rows: list[dict]) -> dict:
    gold = [row["gold_label"] for row in rows]
    pred = [row["predicted_label"] for row in rows]
    support = [index for index, value in enumerate(pred) if value == "SUPPORT"]
    true_support = sum(gold[index] == "SUPPORT" for index in support)
    false_support = len(support) - true_support
    gold_support = sum(value == "SUPPORT" for value in gold)
    return {
        "domain": domain,
        "method": method,
        "count": len(rows),
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(rows),
        "macro_f1": macro_f1(gold, pred),
        "support_predictions": len(support),
        "support_precision": true_support / len(support) if support else 0.0,
        "support_recall": true_support / gold_support if gold_support else 0.0,
        "support_f1": (
            2
            * (true_support / len(support))
            * (true_support / gold_support)
            / ((true_support / len(support)) + (true_support / gold_support))
            if support and true_support and gold_support
            else 0.0
        ),
        "false_support_count": false_support,
        "false_support_rate": false_support / len(support) if support else 0.0,
    }


def select_fold_budget(rows: list[dict], field: str, budget: float) -> set[str]:
    selected: set[str] = set()
    for fold in range(5):
        fold_rows = [row for row in rows if int(row["outer_fold"]) == fold]
        eligible = [row for row in fold_rows if row["candidate_label"] == "SUPPORT"]
        count = min(len(eligible), math.ceil(budget * len(fold_rows)))
        ranked = sorted(
            eligible, key=lambda row: (-float(row.get(field) or 0.0), row["id"])
        )
        selected.update(row["id"] for row in ranked[:count])
    return selected


def selective_metrics(rows: list[dict], reviewed: set[str]) -> dict:
    accepted = [row for row in rows if row["id"] not in reviewed]
    reviewed_rows = [row for row in rows if row["id"] in reviewed]
    false_supports = [
        row
        for row in rows
        if row["candidate_label"] == "SUPPORT" and row["gold_label"] != "SUPPORT"
    ]
    detected = [
        row
        for row in reviewed_rows
        if row["candidate_label"] == "SUPPORT" and row["gold_label"] != "SUPPORT"
    ]
    false_reviews = [
        row
        for row in reviewed_rows
        if row["candidate_label"] == "SUPPORT" and row["gold_label"] == "SUPPORT"
    ]
    gold = [row["gold_label"] for row in accepted]
    pred = [row["candidate_label"] for row in accepted]
    accepted_support = [row for row in accepted if row["candidate_label"] == "SUPPORT"]
    accepted_false = sum(row["gold_label"] != "SUPPORT" for row in accepted_support)
    base_correct = sum(row["candidate_label"] == row["gold_label"] for row in rows)
    reviewed_errors = sum(
        row["candidate_label"] != row["gold_label"] for row in reviewed_rows
    )
    accuracy = sum(g == p for g, p in zip(gold, pred)) / len(accepted)
    return {
        "reviewed_count": len(reviewed_rows),
        "review_rate": len(reviewed_rows) / len(rows),
        "coverage": len(accepted) / len(rows),
        "selective_accuracy": accuracy,
        "selective_risk": 1.0 - accuracy,
        "selective_macro_f1": macro_f1(gold, pred),
        "detected_false_supports": len(detected),
        "missed_false_supports": len(false_supports) - len(detected),
        "incorrectly_reviewed_correct_supports": len(false_reviews),
        "error_detection_precision": len(detected) / len(reviewed_rows)
        if reviewed_rows
        else 0.0,
        "error_detection_recall": len(detected) / len(false_supports)
        if false_supports
        else 0.0,
        "accepted_support_predictions": len(accepted_support),
        "accepted_support_precision": 1.0
        - accepted_false / len(accepted_support)
        if accepted_support
        else float("nan"),
        "accepted_false_support_rate": accepted_false / len(accepted_support)
        if accepted_support
        else float("nan"),
        "oracle_review_accuracy_upper_bound": (base_correct + reviewed_errors)
        / len(rows),
    }


def assert_close(observed: object, expected: str, label: str) -> None:
    if expected == "":
        return
    try:
        observed_number = float(observed)
        expected_number = float(expected)
    except (TypeError, ValueError):
        if str(observed) != expected:
            raise AssertionError(f"{label}: observed={observed!r}, expected={expected!r}")
        return
    if math.isnan(expected_number) and math.isnan(observed_number):
        return
    if not math.isclose(observed_number, expected_number, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(
            f"{label}: observed={observed_number}, expected={expected_number}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute primary point estimates from released frozen predictions."
    )
    parser.add_argument(
        "--output-dir", default="outputs/reproduced", help="Generated verification tables."
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Verify metrics without rebuilding reviewer figures and compact tables.",
    )
    args = parser.parse_args()
    output = ROOT / args.output_dir

    predictions = [
        *read_jsonl(ROOT / "artifacts/predictions/formal600_predictions.jsonl"),
        *read_jsonl(
            ROOT / "artifacts/predictions/pubmedqa_claim300_predictions.jsonl"
        ),
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in predictions:
        grouped[(row["domain"], row["method"])].append(row)
    classification = [
        classification_metrics(domain, method, rows)
        for (domain, method), rows in sorted(grouped.items())
    ]
    expected_classification = {
        (row["domain"], row["method"]): row
        for row in read_csv(ROOT / "artifacts/results/fair_input_metrics.csv")
    }
    for row in classification:
        expected = expected_classification[(row["domain"], row["method"])]
        for field, value in row.items():
            if field in {"domain", "method"}:
                continue
            assert_close(value, expected[field], f"{row['domain']}/{row['method']}/{field}")
    write_csv(output / "fair_input_metrics_recomputed.csv", classification)

    risk_rows = read_csv(ROOT / "artifacts/predictions/risk_routing_scores.csv")
    risk_results = []
    expected_risk = read_csv(ROOT / "artifacts/results/risk_routing_metrics.csv")
    expected_keys = {(row["method"], float(row["budget"])): row for row in expected_risk}
    for method, field in RISK_FIELDS.items():
        for budget in (0.01, 0.02, 0.05, 0.10, 0.20):
            metrics = selective_metrics(
                risk_rows, select_fold_budget(risk_rows, field, budget)
            )
            row = {"method": method, "budget": budget, **metrics}
            expected = expected_keys.get((method, budget))
            if expected:
                for metric, value in row.items():
                    if metric in {"method", "budget"}:
                        continue
                    assert_close(value, expected[metric], f"{method}/{budget}/{metric}")
            risk_results.append(row)
    for budget in (0.01, 0.02, 0.05, 0.10, 0.20):
        simulations = []
        for simulation in range(1000):
            for row in risk_rows:
                digest = hashlib.sha256(
                    f"20260618:{simulation}:{row['id']}".encode("utf-8")
                ).hexdigest()
                row["risk_random_recomputed"] = int(digest[:16], 16) / 16**16
            simulations.append(
                selective_metrics(
                    risk_rows,
                    select_fold_budget(
                        risk_rows, "risk_random_recomputed", budget
                    ),
                )
            )
        row = {
            "method": "random_review",
            "budget": budget,
            **{
                key: sum(float(item[key]) for item in simulations)
                / len(simulations)
                for key in simulations[0]
            },
        }
        expected = expected_keys[("random_review", budget)]
        for metric, value in row.items():
            if metric in {"method", "budget"}:
                continue
            assert_close(value, expected[metric], f"random_review/{budget}/{metric}")
        risk_results.append(row)
    write_csv(output / "risk_routing_metrics_recomputed.csv", risk_results)

    summary = {
        "status": "passed",
        "classification_rows_verified": len(classification),
        "risk_rows_verified": sum(
            (row["method"], float(row["budget"])) in expected_keys
            for row in risk_results
        ),
        "formal600_records": len(
            {
                row["record_id"]
                for row in predictions
                if row["domain"] == "formal600"
            }
        ),
        "pubmedqa_records": len(
            {
                row["record_id"]
                for row in predictions
                if row["domain"] == "pubmedqa"
            }
        ),
        "network_or_api_calls": False,
        "verification_figures_generated": not args.skip_render,
        "compact_tables_generated": not args.skip_render,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if not args.skip_render:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/figures/build_result_figures.py"),
                "--output-dir",
                str(output / "figures"),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/tables/build_main_tables.py"),
                "--output-dir",
                str(output / "tables"),
            ],
            cwd=ROOT,
            check=True,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
