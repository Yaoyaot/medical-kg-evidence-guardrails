from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create compact reviewer tables from released result CSV files."
    )
    parser.add_argument("--output-dir", default="outputs/reproduced/tables")
    args = parser.parse_args()
    output = ROOT / args.output_dir

    fair = read_csv(ROOT / "artifacts/results/fair_input_metrics.csv")
    write_csv(
        output / "fair_input_metrics.csv",
        fair,
        [
            "domain",
            "method",
            "count",
            "accuracy",
            "macro_f1",
            "support_predictions",
            "support_precision",
            "support_recall",
            "false_support_count",
            "false_support_rate",
        ],
    )

    routing = [
        row
        for row in read_csv(ROOT / "artifacts/results/risk_routing_metrics.csv")
        if float(row["budget"]) == 0.05
        and row["method"]
        in {
            "self_reported_confidence",
            "rule_matched_budget",
            "full_without_dataset_source",
            "random_review",
            "oracle",
        }
    ]
    write_csv(
        output / "routing_at_5_percent.csv",
        routing,
        [
            "method",
            "budget",
            "reviewed_count",
            "coverage",
            "selective_accuracy",
            "selective_risk",
            "detected_false_supports",
            "error_detection_precision",
            "error_detection_recall",
        ],
    )
    print(f"Wrote reviewer tables to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
