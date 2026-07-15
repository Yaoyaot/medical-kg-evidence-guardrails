from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


TASK_LABELS = {
    "path_relevance": ("IRRELEVANT", "PARTIAL", "RELEVANT"),
    "actionability": ("NON_ACTIONABLE", "ACTIONABLE"),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-class and PR-AUC Evidence Scorer metrics.")
    parser.add_argument("--scorer-dir", required=True)
    args = parser.parse_args()
    root = Path(args.scorer_dir)
    rows = [row for row in read_jsonl(root / "oof_predictions.jsonl") if row["model"] == "logistic_regression"]

    per_seed = []
    for task, labels in TASK_LABELS.items():
        task_rows = [row for row in rows if row["task"] == task]
        for seed in sorted({int(row["seed"]) for row in task_rows}):
            seed_rows = [row for row in task_rows if int(row["seed"]) == seed]
            class_aps = []
            for label in labels:
                gold = np.asarray([int(row["gold_label"] == label) for row in seed_rows])
                scores = np.asarray([float(row["probabilities"].get(label, 0.0)) for row in seed_rows])
                ap = float(average_precision_score(gold, scores)) if len(set(gold)) > 1 else float("nan")
                class_aps.append(ap)
                per_seed.append({"task": task, "seed": seed, "class": label, "average_precision": ap, "support": int(gold.sum())})
            per_seed.append({"task": task, "seed": seed, "class": "MACRO", "average_precision": float(np.nanmean(class_aps)), "support": len(seed_rows)})

    grouped = defaultdict(list)
    supports = defaultdict(list)
    for row in per_seed:
        grouped[(row["task"], row["class"])].append(row["average_precision"])
        supports[(row["task"], row["class"])].append(row["support"])
    summary = []
    for key, values in sorted(grouped.items()):
        task, label = key
        summary.append({
            "task": task,
            "class": label,
            "average_precision_mean": float(np.nanmean(values)),
            "average_precision_std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            "support_per_seed": int(round(float(np.mean(supports[key])))),
        })
    write_csv(root / "pr_auc_by_seed.csv", per_seed)
    write_csv(root / "pr_auc_summary.csv", summary)
    print(json.dumps({"scorer_dir": str(root), "rows": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
