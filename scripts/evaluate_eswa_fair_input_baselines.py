from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


SEED = 20260618
LABELS = ("SUPPORT", "REFUTE", "UNCERTAIN")
METHOD_LABELS = {
    "direct_llm": "direct_llm",
    "text_rag_llm": "provided_text",
    "provided_text_bm25_llm": "provided_text_plus_extra_bm25",
    "provided_text_kg_llm": "provided_text_plus_kg_paths",
    "medgraphrag_style_llm": "provided_text_plus_kg_plus_extra_bm25",
    "vanilla_graphrag_llm": "kg_only_local_paths_diagnostic",
    "bm25_text_rag_llm": "bm25_only_diagnostic",
}
FAIR_METHODS = (
    "provided_text",
    "provided_text_plus_extra_bm25",
    "provided_text_plus_kg_paths",
    "provided_text_plus_kg_plus_extra_bm25",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dedupe(paths: list[Path]) -> list[dict]:
    rows = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            rows[(row.get("baseline"), row.get("id"))] = row
    return list(rows.values())


def metrics(method: str, rows: list[dict], domain: str) -> dict:
    gold = [row["gold_label"] for row in rows]
    pred = [row["pred_label"] for row in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(gold, pred, labels=list(LABELS), zero_division=0)
    support_indices = [index for index, label in enumerate(pred) if label == "SUPPORT"]
    false_count = sum(gold[index] != "SUPPORT" for index in support_indices)
    return {
        "domain": domain,
        "method": method,
        "count": len(rows),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=list(LABELS), average="macro", zero_division=0),
        "support_predictions": len(support_indices),
        "support_precision": float(precision[0]),
        "support_recall": float(recall[0]),
        "support_f1": float(f1[0]),
        "false_support_count": false_count,
        "false_support_rate": false_count / len(support_indices) if support_indices else 0.0,
    }


def grouped_bootstrap(by_method: dict[str, dict[str, dict]], group_by_id: dict[str, str], iterations: int, domain: str) -> list[dict]:
    reference = "provided_text"
    ordered_ids = sorted(by_method[reference])
    id_to_index = {claim_id: index for index, claim_id in enumerate(ordered_ids)}
    groups = defaultdict(list)
    for claim_id in ordered_ids:
        groups[group_by_id[claim_id]].append(id_to_index[claim_id])
    group_keys = sorted(groups)
    group_indices = [np.asarray(groups[key], dtype=np.int32) for key in group_keys]
    label_to_int = {label: index for index, label in enumerate(LABELS)}
    gold = np.asarray([label_to_int[by_method[reference][claim_id]["gold_label"]] for claim_id in ordered_ids], dtype=np.int8)
    predictions = {
        method: np.asarray([label_to_int[by_method[method][claim_id]["pred_label"]] for claim_id in ordered_ids], dtype=np.int8)
        for method in FAIR_METHODS
    }

    def fast_metrics(sample_indices: np.ndarray, pred: np.ndarray) -> dict[str, float]:
        sample_gold = gold[sample_indices]
        sample_pred = pred[sample_indices]
        accuracy = float(np.mean(sample_gold == sample_pred))
        f1_values = []
        for label in range(len(LABELS)):
            true_positive = int(np.sum((sample_gold == label) & (sample_pred == label)))
            false_positive = int(np.sum((sample_gold != label) & (sample_pred == label)))
            false_negative = int(np.sum((sample_gold == label) & (sample_pred != label)))
            denominator = 2 * true_positive + false_positive + false_negative
            f1_values.append((2 * true_positive / denominator) if denominator else 0.0)
        support_mask = sample_pred == label_to_int["SUPPORT"]
        support_count = int(np.sum(support_mask))
        true_support = int(np.sum(support_mask & (sample_gold == label_to_int["SUPPORT"])))
        support_precision = true_support / support_count if support_count else 0.0
        false_support_rate = 1.0 - support_precision if support_count else 0.0
        return {
            "accuracy": accuracy,
            "macro_f1": float(np.mean(f1_values)),
            "support_precision": support_precision,
            "false_support_rate": false_support_rate,
        }

    rng = random.Random(SEED)
    metric_names = ("accuracy", "macro_f1", "support_precision", "false_support_rate")
    values = {(method, metric): [] for method in FAIR_METHODS if method != reference for metric in metric_names}
    for _ in range(iterations):
        draws = rng.choices(range(len(group_keys)), k=len(group_keys))
        sample_indices = np.concatenate([group_indices[index] for index in draws])
        baseline = fast_metrics(sample_indices, predictions[reference])
        for method in FAIR_METHODS:
            if method == reference:
                continue
            current = fast_metrics(sample_indices, predictions[method])
            for metric in metric_names:
                values[(method, metric)].append(current[metric] - baseline[metric])
    output = []
    for (method, metric), samples in values.items():
        ordered = sorted(samples)
        output.append({
            "domain": domain,
            "method": method,
            "reference": reference,
            "metric": metric,
            "mean_difference": float(np.mean(samples)),
            "ci_low": ordered[int(0.025 * (len(ordered) - 1))],
            "ci_high": ordered[int(0.975 * (len(ordered) - 1))],
            "iterations": iterations,
            "resampling_unit": "claim_source_component" if domain == "formal600" else "pubmedqa_claim",
        })
    return output


def validate(rows: list[dict], expected: int, domain: str) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("baseline") not in METHOD_LABELS:
            continue
        grouped[METHOD_LABELS[row["baseline"]]][row["id"]] = row
    for method in FAIR_METHODS:
        if len(grouped[method]) != expected:
            raise ValueError(f"{domain}/{method}: expected {expected}, found {len(grouped[method])}")
        failures = [row for row in grouped[method].values() if row.get("request_error") or row.get("parse_error")]
        if failures:
            raise RuntimeError(f"{domain}/{method}: {len(failures)} failed results")
    ids = set(grouped[FAIR_METHODS[0]])
    if any(set(grouped[method]) != ids for method in FAIR_METHODS):
        raise ValueError(f"{domain}: fair-input method IDs differ")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Stage 9 provided-text/BM25/KG fair input matrix.")
    parser.add_argument("--fair-dir", default="data/processed/stage9_eswa_major_revision/fair_input_baselines")
    parser.add_argument("--group-manifest", default="data/processed/stage9_eswa_major_revision/grouping/formal600_group_manifest.csv")
    parser.add_argument("--output-dir", default="data/processed/stage9_eswa_major_revision/fair_input_evaluation")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()
    fair = Path(args.fair_dir)
    formal_paths = [
        Path("data/processed/llm_baseline_results_deepseek-v4-flash-formal600.jsonl"),
        Path("data/processed/stage7_hierarchical_scorer/formal600/eswa_enhanced_baselines_formal600.jsonl"),
        *sorted(fair.glob("formal600_*.jsonl")),
    ]
    pubmedqa_paths = [
        Path("data/processed/stage8_pubmedqa_external/external_baseline_results.jsonl"),
        *sorted(fair.glob("pubmedqa_*.jsonl")),
    ]
    formal = validate(dedupe(formal_paths), 600, "formal600")
    pubmedqa = validate(dedupe(pubmedqa_paths), 300, "pubmedqa")
    group_by_id = {row["id"]: row["pair_group_id"] for row in read_csv(Path(args.group_manifest))}
    output = Path(args.output_dir)
    main_rows = []
    for domain, grouped in (("formal600", formal), ("pubmedqa", pubmedqa)):
        for method in FAIR_METHODS:
            main_rows.append(metrics(method, list(grouped[method].values()), domain))
        for diagnostic in ("direct_llm", "kg_only_local_paths_diagnostic", "bm25_only_diagnostic"):
            if diagnostic in grouped and len(grouped[diagnostic]) in {300, 600}:
                main_rows.append(metrics(diagnostic, list(grouped[diagnostic].values()), domain))
    bootstrap = grouped_bootstrap(formal, group_by_id, args.bootstrap_iterations, "formal600")
    bootstrap.extend(grouped_bootstrap(pubmedqa, {claim_id: claim_id for claim_id in pubmedqa["provided_text"]}, args.bootstrap_iterations, "pubmedqa"))
    write_csv(output / "fair_input_main_results.csv", main_rows)
    write_csv(output / "fair_input_paired_bootstrap.csv", bootstrap)
    stats = {
        "formal600_rows_per_fair_method": 600,
        "pubmedqa_rows_per_fair_method": 300,
        "fair_methods": list(FAIR_METHODS),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": SEED,
        "formal600_resampling_unit": "claim_source_component",
        "pubmedqa_resampling_unit": "claim",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "fair_input_evaluation_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
