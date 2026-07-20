from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from repo_paths import find_repo_root


SEED = 20260618
ITERATIONS = 5000
LABELS = ("SUPPORT", "REFUTE", "UNCERTAIN")
FAIR_METHODS = (
    "provided_text",
    "provided_text_plus_extra_bm25",
    "provided_text_plus_kg_paths",
    "provided_text_plus_kg_plus_extra_bm25",
)
METHOD_LABELS = {
    "text_rag_llm": "provided_text",
    "provided_text_bm25_llm": "provided_text_plus_extra_bm25",
    "provided_text_kg_llm": "provided_text_plus_kg_paths",
    "medgraphrag_style_llm": "provided_text_plus_kg_plus_extra_bm25",
}
AUDIT_TOTAL_FIELDS = (
    "mention_tp",
    "mention_fp",
    "mention_fn",
    "predicted_links",
    "incorrect_links",
    "correct_links",
    "resolved_gold",
    "resolved_detected",
    "resolved_exact",
)


ROOT = find_repo_root()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def ratio_metrics(totals: dict[str, float]) -> dict[str, float]:
    tp, fp, fn = totals["mention_tp"], totals["mention_fp"], totals["mention_fn"]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    link_accuracy = (
        totals["correct_links"] / totals["predicted_links"]
        if totals["predicted_links"]
        else float("nan")
    )
    strict_id_accuracy = (
        totals["resolved_exact"] / totals["resolved_detected"]
        if totals["resolved_detected"]
        else float("nan")
    )
    concept_recall = (
        totals["resolved_exact"] / totals["resolved_gold"]
        if totals["resolved_gold"]
        else float("nan")
    )
    return {
        "mention_precision": precision,
        "mention_recall": recall,
        "mention_f1": f1,
        "human_judged_link_accuracy": link_accuracy,
        "strict_id_accuracy_detected_resolved": strict_id_accuracy,
        "end_to_end_resolved_concept_recall": concept_recall,
    }


def entity_audit_analysis(data_root: Path, output: Path, iterations: int) -> dict:
    audit = read_csv(
        data_root
        / "stage11_eswa_nested_crossfit/human_audit_results/entity_linking_claim_metrics.csv"
    )
    adjudication = {
        row["id"]: row
        for row in read_csv(
            data_root
            / "stage9_eswa_major_revision/human_audits/entity_linking_audit120_adjudication.csv"
        )
    }
    formal = read_jsonl(
        data_root
        / "stage11_eswa_nested_crossfit/formal600/formal600_crossfit_risk_scores.jsonl"
    )
    external = read_jsonl(
        data_root
        / "stage11_eswa_nested_crossfit/pubmedqa_external/pubmedqa_frozen_risk_scores.jsonl"
    )
    population_rows = formal + external
    population = defaultdict(int)
    for row in population_rows:
        dataset = str(row["dataset"]).lower().replace("pubmedqa_claim", "pubmedqa")
        status = "linked" if int(row.get("linked_entity_count", 0)) > 0 else "unlinked"
        population[(dataset, status)] += 1

    audit_by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in audit:
        row = dict(row)
        row["dataset"] = row["dataset"].lower()
        row["link_status"] = (
            "linked" if int(float(adjudication[row["id"]]["predicted_link_count"])) > 0 else "unlinked"
        )
        for field in AUDIT_TOTAL_FIELDS:
            row[field] = float(row[field])
        audit_by_stratum[(row["dataset"], row["link_status"])].append(row)

    if set(audit_by_stratum) != set(population):
        raise AssertionError(
            f"Audit/population strata differ: audit={sorted(audit_by_stratum)}, population={sorted(population)}"
        )
    sampling_rows = []
    for stratum in sorted(population):
        n_population = population[stratum]
        n_audit = len(audit_by_stratum[stratum])
        sampling_rows.append(
            {
                "dataset": stratum[0],
                "link_status": stratum[1],
                "population_claims": n_population,
                "audited_claims": n_audit,
                "sampling_fraction": n_audit / n_population,
                "inverse_probability_weight": n_population / n_audit,
            }
        )
    write_csv(output / "entity_audit_sampling_design.csv", sampling_rows)

    targets = {
        "formal600_composition": {"healthver", "medaesqa", "scifact"},
        "pubmedqa300": {"pubmedqa"},
        "combined900_composition": {"healthver", "medaesqa", "scifact", "pubmedqa"},
    }

    def weighted_totals(target_datasets: set[str], sampled: dict[tuple[str, str], list[dict]]) -> dict[str, float]:
        totals = {field: 0.0 for field in AUDIT_TOTAL_FIELDS}
        for stratum, rows in sampled.items():
            if stratum[0] not in target_datasets:
                continue
            weight = population[stratum] / len(rows)
            for row in rows:
                for field in AUDIT_TOTAL_FIELDS:
                    totals[field] += weight * float(row[field])
        return totals

    point_rows = []
    bootstrap_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for target, datasets in targets.items():
        totals = weighted_totals(datasets, audit_by_stratum)
        point_rows.append(
            {
                "estimand": target,
                "audit_design": "dataset_by_link_status_stratified_ratio_estimator",
                "population_claims": sum(
                    count for (dataset, _), count in population.items() if dataset in datasets
                ),
                **totals,
                **ratio_metrics(totals),
            }
        )

    rng = random.Random(SEED)
    strata = sorted(audit_by_stratum)
    for _ in range(iterations):
        sampled: dict[tuple[str, str], list[dict]] = {}
        for stratum in strata:
            rows = audit_by_stratum[stratum]
            sampled[stratum] = [rows[rng.randrange(len(rows))] for _ in rows]
        for target, datasets in targets.items():
            metrics = ratio_metrics(weighted_totals(datasets, sampled))
            for metric, value in metrics.items():
                if not math.isnan(value):
                    bootstrap_values[(target, metric)].append(value)

    ci_rows = []
    for point in point_rows:
        for metric in (
            "mention_precision",
            "mention_recall",
            "mention_f1",
            "human_judged_link_accuracy",
            "strict_id_accuracy_detected_resolved",
            "end_to_end_resolved_concept_recall",
        ):
            values = bootstrap_values[(point["estimand"], metric)]
            ci_rows.append(
                {
                    "estimand": point["estimand"],
                    "metric": metric,
                    "point_estimate": point[metric],
                    "ci_low": percentile(values, 2.5),
                    "ci_high": percentile(values, 97.5),
                    "iterations": iterations,
                    "resampling_design": "within_dataset_by_link_status_stratum",
                    "interpretation": "design_standardized_sensitivity_not_random_population_audit",
                }
            )
    write_csv(output / "entity_audit_design_standardized.csv", point_rows)
    write_csv(output / "entity_audit_stratified_bootstrap.csv", ci_rows)
    return {
        "population_by_stratum": {
            f"{dataset}:{status}": count for (dataset, status), count in sorted(population.items())
        },
        "audit_claims_per_stratum": {
            f"{dataset}:{status}": len(rows) for (dataset, status), rows in sorted(audit_by_stratum.items())
        },
        "estimands": point_rows,
    }


def dedupe_jsonl(paths: list[Path]) -> list[dict]:
    rows = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            baseline = row.get("baseline")
            if baseline in METHOD_LABELS:
                rows[(METHOD_LABELS[baseline], row["id"])] = row
    return list(rows.values())


def classification_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    precision, recall, _, _ = precision_recall_fscore_support(
        gold, pred, labels=list(LABELS), zero_division=0
    )
    support_mask = pred == "SUPPORT"
    false_count = int(np.sum(support_mask & (gold != "SUPPORT")))
    support_count = int(np.sum(support_mask))
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=list(LABELS), average="macro", zero_division=0)),
        "support_precision": float(precision[0]),
        "support_recall": float(recall[0]),
        "support_predictions": float(support_count),
        "false_support_count": float(false_count),
        "false_support_rate": false_count / support_count if support_count else 0.0,
    }


def fair_input_pairwise(data_root: Path, output: Path, iterations: int) -> dict:
    fair_dir = data_root / "stage9_eswa_major_revision/fair_input_baselines"
    paths = [
        data_root / "llm_baseline_results_deepseek-v4-flash-formal600.jsonl",
        data_root / "stage7_hierarchical_scorer/formal600/eswa_enhanced_baselines_formal600.jsonl",
        *sorted(fair_dir.glob("formal600_*.jsonl")),
    ]
    rows = dedupe_jsonl(paths)
    by_method: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_method[METHOD_LABELS[row["baseline"]]][row["id"]] = row
    for method in FAIR_METHODS:
        if len(by_method[method]) != 600:
            raise AssertionError(f"{method}: expected 600 rows, found {len(by_method[method])}")
    ids = sorted(by_method[FAIR_METHODS[0]])
    if any(set(by_method[method]) != set(ids) for method in FAIR_METHODS):
        raise AssertionError("Fair-input method IDs differ")
    group_map = {
        row["id"]: row["pair_group_id"]
        for row in read_csv(data_root / "stage9_eswa_major_revision/grouping/formal600_group_manifest.csv")
    }
    groups: dict[str, list[int]] = defaultdict(list)
    for index, identifier in enumerate(ids):
        groups[group_map[identifier]].append(index)
    group_arrays = [np.asarray(groups[key], dtype=int) for key in sorted(groups)]
    gold = np.asarray([by_method[FAIR_METHODS[0]][identifier]["gold_label"] for identifier in ids])
    pred = {
        method: np.asarray([by_method[method][identifier]["pred_label"] for identifier in ids])
        for method in FAIR_METHODS
    }
    pairs = [
        ("provided_text_plus_extra_bm25", "provided_text"),
        ("provided_text_plus_kg_paths", "provided_text"),
        ("provided_text_plus_kg_plus_extra_bm25", "provided_text"),
        ("provided_text_plus_kg_paths", "provided_text_plus_extra_bm25"),
    ]
    metric_names = tuple(classification_metrics(gold, pred[FAIR_METHODS[0]]))
    point_rows = []
    for method, reference in pairs:
        left = classification_metrics(gold, pred[method])
        right = classification_metrics(gold, pred[reference])
        for metric in metric_names:
            point_rows.append(
                {
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "observed_difference": left[metric] - right[metric],
                }
            )

    rng = random.Random(SEED)
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for _ in range(iterations):
        draws = rng.choices(range(len(group_arrays)), k=len(group_arrays))
        indices = np.concatenate([group_arrays[index] for index in draws])
        sampled_gold = gold[indices]
        sampled = {method: classification_metrics(sampled_gold, values_[indices]) for method, values_ in pred.items()}
        for method, reference in pairs:
            for metric in metric_names:
                values[(method, reference, metric)].append(
                    sampled[method][metric] - sampled[reference][metric]
                )
    rows_out = []
    for point in point_rows:
        samples = values[(point["method"], point["reference"], point["metric"])]
        rows_out.append(
            {
                **point,
                "bootstrap_mean_difference": float(np.mean(samples)),
                "ci_low": percentile(samples, 2.5),
                "ci_high": percentile(samples, 97.5),
                "iterations": iterations,
                "resampling_unit": "claim_source_component",
                "models_retrained_within_bootstrap": False,
            }
        )
    write_csv(output / "fair_input_all_pairwise_bootstrap.csv", rows_out)
    return {"comparisons": len(pairs), "metrics": list(metric_names), "rows": len(rows_out)}


def select_fold_budget(rows: list[dict], score_field: str, budget: float) -> set[str]:
    selected: set[str] = set()
    for fold in range(5):
        fold_rows = [row for row in rows if int(row["outer_fold"]) == fold]
        eligible = [row for row in fold_rows if row["candidate_label"] == "SUPPORT"]
        count = min(len(eligible), math.ceil(budget * len(fold_rows)))
        ranked = sorted(eligible, key=lambda row: (-float(row[score_field]), row["id"]))
        selected.update(row["id"] for row in ranked[:count])
    return selected


def reviewer_sensitivity(data_root: Path, output: Path, iterations: int) -> dict:
    rows = read_jsonl(
        data_root
        / "stage11_eswa_nested_crossfit/formal600/formal600_crossfit_risk_scores.jsonl"
    )
    methods = {
        "confidence": "risk_confidence",
        "rule": "risk_rule",
        "learned": "risk_full_without_dataset_source",
        "oracle": "risk_oracle",
    }
    budgets = (0.01, 0.02, 0.05, 0.10, 0.20)
    lambdas = (0.01, 0.05, 0.10, 0.20, 0.50)
    sensitivities = (0.60, 0.80, 1.00)

    def counts(sample: list[dict], field: str, budget: float) -> tuple[int, int]:
        reviewed = select_fold_budget(sample, field, budget)
        captured = sum(
            row["id"] in reviewed
            and row["candidate_label"] == "SUPPORT"
            and row["gold_label"] != "SUPPORT"
            for row in sample
        )
        return len(reviewed), captured

    point_rows = []
    for method, field in methods.items():
        for budget in budgets:
            reviewed, captured = counts(rows, field, budget)
            for sensitivity in sensitivities:
                for cost in lambdas:
                    point_rows.append(
                        {
                            "method": method,
                            "budget": budget,
                            "reviewed_count": reviewed,
                            "captured_false_supports": captured,
                            "reviewer_sensitivity": sensitivity,
                            "review_cost_ratio_lambda": cost,
                            "normalized_idealized_net_gain": (sensitivity * captured - cost * reviewed) / len(rows),
                        }
                    )
    write_csv(output / "reviewer_sensitivity_cost_point_estimates.csv", point_rows)

    components: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        components[row["pair_group_id"]].append(row)
    keys = sorted(components)
    rng = random.Random(SEED)
    values: dict[tuple[str, float, float, float], list[float]] = defaultdict(list)
    for _ in range(iterations):
        sample = []
        for draw, key in enumerate(rng.choices(keys, k=len(keys))):
            for local, source in enumerate(components[key]):
                sample.append({**source, "id": f"b{draw}-{local}-{source['id']}"})
        for method, field in methods.items():
            for budget in budgets:
                reviewed, captured = counts(sample, field, budget)
                for sensitivity in sensitivities:
                    for cost in lambdas:
                        values[(method, budget, sensitivity, cost)].append(
                            (sensitivity * captured - cost * reviewed) / len(sample)
                        )
    ci_rows = []
    points = {
        (row["method"], row["budget"], row["reviewer_sensitivity"], row["review_cost_ratio_lambda"]): row
        for row in point_rows
    }
    for key, samples in sorted(values.items()):
        point = points[key]
        ci_rows.append(
            {
                "method": key[0],
                "budget": key[1],
                "reviewer_sensitivity": key[2],
                "review_cost_ratio_lambda": key[3],
                "observed_normalized_idealized_net_gain": point["normalized_idealized_net_gain"],
                "bootstrap_mean": float(np.mean(samples)),
                "ci_low": percentile(samples, 2.5),
                "ci_high": percentile(samples, 97.5),
                "iterations": iterations,
                "resampling_unit": "claim_source_component",
                "models_retrained_within_bootstrap": False,
            }
        )
    write_csv(output / "reviewer_sensitivity_cost_bootstrap.csv", ci_rows)
    return {"methods": list(methods), "point_rows": len(point_rows), "bootstrap_rows": len(ci_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-API analyses addressing the consolidated ESWA review recommendations."
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/stage13_eswa_review_revision",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=ITERATIONS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "seed": SEED,
        "bootstrap_iterations": args.bootstrap_iterations,
        "no_api_requests": True,
        "frozen_predictions_only": True,
        "entity_audit": entity_audit_analysis(args.data_root, args.output_dir, args.bootstrap_iterations),
        "fair_input": fair_input_pairwise(args.data_root, args.output_dir, args.bootstrap_iterations),
        "reviewer_sensitivity": reviewer_sensitivity(args.data_root, args.output_dir, args.bootstrap_iterations),
    }
    stats["sha256"] = {}
    for path in sorted(args.output_dir.glob("*.csv")):
        stats["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir / "review_revision_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
