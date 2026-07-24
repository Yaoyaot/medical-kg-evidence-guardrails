from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evidence_funnel(output: Path) -> None:
    rows = read_csv(ROOT / "artifacts/results/evidence_conversion_funnel.csv")
    row = next(item for item in rows if item["variant"] == "primekg_fusion_relation_aware")
    labels = ["Entity linked", "Any path", "Direct edge", "Predicate aligned"]
    values = [
        int(row["entity_linking_claims"]),
        int(row["any_path_claims"]),
        int(row["direct_edge_claims"]),
        int(row["predicate_aligned_claims"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, values, color=["#4472C4", "#5B9BD5", "#70AD47", "#ED7D31"])
    ax.set_ylabel("Claims")
    ax.tick_params(axis="x", rotation=15)
    ax.bar_label(bars, padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "evidence_conversion_funnel.png", dpi=300)
    plt.close(fig)


def risk_coverage(output: Path) -> None:
    rows = read_csv(ROOT / "artifacts/results/risk_coverage_curve.csv")
    methods = {
        "self_reported_confidence": ("Self-reported confidence", "#4472C4"),
        "rule_matched_budget": ("Rule", "#A5A5A5"),
        "full_without_dataset_source": ("Learned", "#70AD47"),
        "oracle": ("Oracle", "#ED7D31"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method, (label, color) in methods.items():
        subset = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: float(row["coverage"]),
        )
        ax.plot(
            [float(row["coverage"]) for row in subset],
            [float(row["selective_risk"]) for row in subset],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=label,
            color=color,
        )
    ax.set_xlabel("Accepted coverage")
    ax.set_ylabel("Selective risk")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "risk_coverage_curve.png", dpi=300)
    plt.close(fig)


def entity_audit(output: Path) -> None:
    rows = read_csv(ROOT / "artifacts/results/entity_linking_by_dataset.csv")
    selected = [
        row
        for row in rows
        if row.get("scope") == "dataset"
        and row.get("value") in {"healthver", "medaesqa", "pubmedqa", "scifact"}
    ]
    datasets = [row["value"] for row in selected]
    values = [float(row["mention_f1"]) for row in selected]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(datasets, values, color="#4472C4")
    ax.set_ylabel("Mention F1")
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "entity_linking_by_dataset.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild data-derived verification figures from released CSV files."
    )
    parser.add_argument("--output-dir", default="outputs/reproduced/figures")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    evidence_funnel(output)
    risk_coverage(output)
    entity_audit(output)
    print(f"Wrote three verification figures to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
