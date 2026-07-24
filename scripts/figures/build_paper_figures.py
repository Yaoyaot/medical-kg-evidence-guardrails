from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COLORS = {
    "navy": "#173F5F",
    "blue": "#277DA1",
    "teal": "#2A9D8F",
    "green": "#4C956C",
    "orange": "#E07A3F",
    "red": "#C94C4C",
    "purple": "#7B6DAD",
    "gray": "#657786",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str, dpi: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white")
    fig.savefig(
        output / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def figure3(output: Path, dpi: int) -> None:
    rows = read_csv(ROOT / "artifacts/predictions/risk_routing_scores.csv")
    funnel = [
        len(rows),
        sum(int(row["linked_entity_count"]) > 0 for row in rows),
        sum(int(row["path_count"]) > 0 for row in rows),
        sum(int(row["direct_path_count"]) > 0 for row in rows),
        sum(int(row["predicate_aligned_path_count"]) > 0 for row in rows),
        sum(int(row["strict_structured_candidate_count"]) > 0 for row in rows),
    ]
    labels = [
        "Formal600",
        "Entity\nlinked",
        "Any local\npath",
        "Direct\nedge",
        "Predicate\naligned",
        "Qualifier\ncompatible",
    ]
    colors = [
        COLORS["navy"],
        COLORS["blue"],
        COLORS["purple"],
        COLORS["orange"],
        "#D4942C",
        COLORS["green"],
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    bars = ax.bar(np.arange(len(labels)), funnel, color=colors, width=0.68)
    for bar, count in zip(bars, funnel):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 12,
            f"{count}\n({count / len(rows):.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Claim–evidence pairs")
    ax.set_ylim(0, 690)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9E1E8", linewidth=0.7, alpha=0.7)
    save_figure(fig, output, "figure3_formal600_evidence_funnel", dpi)

    states = Counter(row["evidence_state_nominal"] for row in rows)
    expected_states = {
        "NO_KG_GROUNDING": 161,
        "SINGLE_ENTITY_CONTEXT": 272,
        "TWO_HOP_CONTEXT": 89,
        "DIRECT_RELATION_UNRESOLVED": 43,
        "DIRECT_PREDICATE_OR_DIRECTION_MISMATCH": 24,
        "DIRECT_MATCH_QUALIFIER_INCOMPLETE": 9,
        "DIRECT_MATCH_QUALIFIER_COMPATIBLE": 2,
    }
    if states != Counter(expected_states):
        raise AssertionError(f"Unexpected terminal-state distribution: {states}")


def figure4(output: Path, dpi: int) -> None:
    rows = read_csv(ROOT / "artifacts/results/risk_routing_metrics.csv")
    methods = [
        ("random_review", "Random", COLORS["gray"]),
        ("self_reported_confidence", "Self-reported confidence", COLORS["blue"]),
        ("rule_matched_budget", "Rule-based KG state", COLORS["orange"]),
        ("full_without_dataset_source", "Full model (no dataset source)", COLORS["green"]),
        ("oracle", "Oracle", COLORS["red"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for method, label, color in methods:
        selected = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: float(row["budget"]),
        )
        budgets = [100 * float(row["budget"]) for row in selected]
        axes[0].plot(
            budgets,
            [float(row["error_detection_recall"]) for row in selected],
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
        )
        axes[1].plot(
            budgets,
            [float(row["selective_risk"]) for row in selected],
            marker="o",
            linewidth=1.8,
            label=label,
            color=color,
        )
    axes[0].set_title("False-SUPPORT error detection", weight="bold")
    axes[0].set_ylabel("Error-detection recall")
    axes[1].set_title("Risk among accepted predictions", weight="bold")
    axes[1].set_ylabel("Selective risk")
    for ax in axes:
        ax.set_xlabel("Nominal review budget (%)")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.grid(color="#D9E1E8", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    save_figure(fig, output, "figure4_matched_budget_routing", dpi)


def binned_risk(
    rows: list[dict[str, str]], field: str
) -> tuple[list[float], list[float], list[int]]:
    bins = np.linspace(0, 1, 11)
    scores = np.asarray([float(row[field]) for row in rows])
    events = np.asarray([float(row["gold_label"] != "SUPPORT") for row in rows])
    means: list[float] = []
    observed: list[float] = []
    sizes: list[int] = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (scores >= left) & (scores < right if right < 1 else scores <= right)
        if not np.any(mask):
            continue
        means.append(float(np.mean(scores[mask])))
        observed.append(float(np.mean(events[mask])))
        sizes.append(int(np.sum(mask)))
    return means, observed, sizes


def figure5(output: Path, dpi: int) -> None:
    rows = [
        row
        for row in read_csv(ROOT / "artifacts/predictions/risk_routing_scores.csv")
        if row["candidate_label"] == "SUPPORT"
    ]
    series = [
        ("Self-reported confidence", "risk_confidence", COLORS["blue"]),
        (
            "Full model (no dataset source)",
            "risk_full_without_dataset_source",
            COLORS["green"],
        ),
    ]
    annotation_layouts = {
        "Self-reported confidence": [
            ((0, 4), "center", "bottom"),
            ((-4, 5), "right", "bottom"),
            ((0, 4), "center", "bottom"),
        ],
        "Full model (no dataset source)": [
            ((0, 4), "center", "bottom"),
            ((0, 4), "center", "bottom"),
            ((5, 4), "left", "bottom"),
            ((0, -5), "center", "top"),
            ((0, 4), "center", "bottom"),
            ((0, -5), "center", "top"),
            ((0, 4), "center", "bottom"),
            ((4, -5), "left", "top"),
            ((-4, 5), "right", "bottom"),
        ],
    }
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=COLORS["gray"],
        linewidth=1.2,
        label="Perfect calibration",
    )
    for label, field, color in series:
        means, observed, sizes = binned_risk(rows, field)
        ax.plot(means, observed, marker="o", linewidth=1.8, color=color, label=label)
        layouts = annotation_layouts[label]
        for index, (x, y, size) in enumerate(zip(means, observed, sizes)):
            offset, horizontal_alignment, vertical_alignment = (
                layouts[index]
                if index < len(layouts)
                else ((0, 5), "center", "bottom")
            )
            ax.annotate(
                str(size),
                (x, y),
                xytext=offset,
                textcoords="offset points",
                ha=horizontal_alignment,
                va=vertical_alignment,
                fontsize=9,
                color=color,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                    "pad": 0.2,
                },
            )
    ax.set_xlabel("Mean predicted false-SUPPORT risk")
    ax.set_ylabel("Observed false-SUPPORT frequency")
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0, 1.05)
    ax.grid(color="#D9E1E8", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save_figure(fig, output, "figure5_reliability_diagram", dpi)


def figure6(output: Path, dpi: int) -> None:
    formal = {
        row["method"]: row
        for row in read_csv(ROOT / "artifacts/results/risk_ranking_metrics.csv")
    }
    external = {
        row["method"]: row
        for row in read_csv(
            ROOT / "artifacts/results/pubmedqa_frozen_risk_ranking.csv"
        )
    }
    methods = [
        "self_reported_confidence",
        "rule_matched_budget",
        "full_without_dataset_source",
    ]
    labels = [
        "Self-reported\nconfidence",
        "Rule-based\nKG state",
        "Full model\n(no dataset source)",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, metric, title in zip(
        axes,
        ("auroc", "average_precision"),
        ("AUROC", "Average precision"),
    ):
        x = np.arange(len(methods))
        width = 0.34
        ax.bar(
            x - width / 2,
            [float(formal[method][metric]) for method in methods],
            width,
            label="Formal600 grouped OOF",
            color=COLORS["blue"],
        )
        ax.bar(
            x + width / 2,
            [float(external[method][metric]) for method in methods],
            width,
            label="PubMedQA frozen stress test",
            color=COLORS["orange"],
        )
        if metric == "auroc":
            ax.axhline(
                0.5, color=COLORS["gray"], linestyle="--", linewidth=1
            )
        ax.set_xticks(x, labels)
        ax.set_title(title, weight="bold")
        ax.set_ylabel(title)
        ax.set_ylim(0, 0.8)
        ax.grid(axis="y", color="#D9E1E8", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="upper right")
    save_figure(fig, output, "figure6_frozen_risk_transfer", dpi)


def figure7(output: Path, dpi: int) -> None:
    rows = [
        row
        for row in read_csv(
            ROOT / "artifacts/results/entity_linking_by_dataset.csv"
        )
        if row["scope"] == "dataset"
    ]
    order = ["healthver", "medaesqa", "pubmedqa", "scifact"]
    by_dataset = {row["value"]: row for row in rows}
    metrics = [
        ("mention_precision", "Mention precision", COLORS["blue"]),
        ("mention_recall", "Mention recall", COLORS["purple"]),
        (
            "human_judged_link_accuracy",
            "Human-judged link accuracy",
            COLORS["orange"],
        ),
    ]
    x = np.arange(len(order))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    for index, (metric, label, color) in enumerate(metrics):
        values = np.asarray([float(by_dataset[name][metric]) for name in order])
        lows = np.asarray(
            [float(by_dataset[name][f"{metric}_ci_low"]) for name in order]
        )
        highs = np.asarray(
            [float(by_dataset[name][f"{metric}_ci_high"]) for name in order]
        )
        offsets = x + (index - 1) * width
        bars = ax.bar(
            offsets,
            values,
            width,
            label=label,
            color=color,
            yerr=np.vstack([values - lows, highs - values]),
            capsize=3,
            error_kw={"linewidth": 0.9, "zorder": 3},
            zorder=2,
        )
        for bar, value, high in zip(bars, values, highs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                high + 0.035,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLORS["navy"],
                zorder=5,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.96,
                    "pad": 0.18,
                },
            )
    ax.set_xticks(x, ["HealthVer", "MedAESQA", "PubMedQA", "SciFact"])
    ax.set_ylabel("Audited rate")
    ax.set_ylim(0, 1.16)
    ax.grid(axis="y", color="#D9E1E8", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, output, "figure7_entity_linking_audit", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the five data-derived manuscript figures from released "
            "frozen artifacts. Figures 1 and 2 are conceptual diagrams."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reproduced/paper_figures",
    )
    parser.add_argument("--dpi", type=int, default=1200)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    configure_plotting()
    output = ROOT / args.output_dir
    figure3(output, args.dpi)
    figure4(output, args.dpi)
    figure5(output, args.dpi)
    figure6(output, args.dpi)
    figure7(output, args.dpi)
    print(f"Wrote Figures 3–7 to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
