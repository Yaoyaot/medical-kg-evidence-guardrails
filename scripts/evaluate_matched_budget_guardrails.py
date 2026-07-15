from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score

from hierarchical_evidence_features import combined_text, evidence_path_rows, structured_features


BUDGETS = [0.025, 0.05, 0.10, 0.15]
MODEL_SEEDS = [11, 23, 37, 53, 71]
SPLIT_SEED = 20260617


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_fraction(value: str) -> float:
    digest = hashlib.sha256(f"{SPLIT_SEED}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def score_paths(paths: list[dict], bundle: dict) -> list[dict]:
    if not paths:
        return []
    tfidf = bundle["tfidf"]
    vectorizer = bundle["dict_vectorizer"]
    model = bundle["model"]
    x = hstack([
        tfidf.transform([combined_text(row) for row in paths]),
        vectorizer.transform([structured_features(row) for row in paths]),
    ]).tocsr()
    probabilities = model.predict_proba(x)
    output = []
    for row, values in zip(paths, probabilities):
        output.append({**row, "probabilities": {label: float(value) for label, value in zip(model.classes_, values)}})
    return output


def rule_risk(status: str) -> float:
    return {
        "KG_EXPLICIT_CONFLICT": 1.0,
        "KG_TWO_HOP_CONTEXT": 0.85,
        "KG_SINGLE_ENTITY_CONTEXT": 0.65,
        "KG_PARTIAL_DIRECT": 0.50,
        "KG_NO_GROUNDING": 0.30,
        "KG_DUAL_EVIDENCE": 0.10,
    }.get(status, 0.30)


def risk_features(row: dict) -> dict:
    features = {
        "text_confidence": float(row.get("text_confidence", 0.0)),
        "max_actionable_probability": float(row.get("max_actionable_probability", 0.0)),
        "mean_actionable_probability": float(row.get("mean_actionable_probability", 0.0)),
        "max_relevant_probability": float(row.get("max_relevant_probability", 0.0)),
        "mean_relevant_probability": float(row.get("mean_relevant_probability", 0.0)),
        "path_count": float(row.get("path_count", 0)),
        "direct_path_count": float(row.get("direct_path_count", 0)),
        "two_hop_path_count": float(row.get("two_hop_path_count", 0)),
        "predicate_aligned_path_count": float(row.get("predicate_aligned_path_count", 0)),
        "endpoint_aligned_path_count": float(row.get("endpoint_aligned_path_count", 0)),
        "max_path_score": float(row.get("max_path_score", 0.0)),
        "linked_entity_count": float(row.get("linked_entity_count", 0)),
        "has_path": float(row.get("path_count", 0) > 0),
        "has_direct_path": float(row.get("direct_path_count", 0) > 0),
        "has_two_hop_context": float(row.get("two_hop_path_count", 0) > 0),
        "guardrail_status": row.get("guardrail_status", "KG_NO_GROUNDING"),
        "kg_evidence_tier": row.get("kg_evidence_tier", "NO_KG_GROUNDING"),
        "dataset": row.get("dataset", "unknown"),
    }
    for qualifier in row.get("qualifier_flags") or []:
        features[f"qualifier::{qualifier}"] = 1.0
    return features


def ece(y_true: list[int], scores: list[float], bins: int = 10) -> float:
    if not y_true:
        return 0.0
    total = len(y_true)
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [i for i, score in enumerate(scores) if lower <= score < upper or (index == bins - 1 and score == 1.0)]
        if not members:
            continue
        confidence = sum(scores[i] for i in members) / len(members)
        observed = sum(y_true[i] for i in members) / len(members)
        result += len(members) / total * abs(confidence - observed)
    return result


def classify_metrics(rows: list[dict], reviewed_ids: set[str]) -> dict:
    predictions = []
    for row in rows:
        label = row["candidate_label"]
        if row["id"] in reviewed_ids and label == "SUPPORT":
            label = "UNCERTAIN"
        predictions.append(label)
    gold = [row["gold_label"] for row in rows]
    support_indices = [i for i, label in enumerate(predictions) if label == "SUPPORT"]
    support_correct = sum(gold[i] == "SUPPORT" for i in support_indices)
    support_precision = support_correct / len(support_indices) if support_indices else 0.0
    reviewed = [row for row in rows if row["id"] in reviewed_ids]
    reviewed_errors = sum(row["candidate_label"] == "SUPPORT" and row["gold_label"] != "SUPPORT" for row in reviewed)
    all_support_errors = sum(row["candidate_label"] == "SUPPORT" and row["gold_label"] != "SUPPORT" for row in rows)
    return {
        "accuracy": accuracy_score(gold, predictions),
        "macro_f1": f1_score(gold, predictions, labels=["SUPPORT", "REFUTE", "UNCERTAIN"], average="macro", zero_division=0),
        "support_precision": support_precision,
        "support_hallucination_rate": 1.0 - support_precision if support_indices else 0.0,
        "error_detection_recall": reviewed_errors / all_support_errors if all_support_errors else 0.0,
        "review_precision": reviewed_errors / len(reviewed) if reviewed else 0.0,
        "retained_accuracy": accuracy_score(gold, predictions),
        "accepted_coverage": sum(label != "UNCERTAIN" for label in predictions) / len(rows),
        "review_rate": len(reviewed_ids) / len(rows),
        "reviewed_count": len(reviewed_ids),
    }


def choose_reviews(rows: list[dict], score_field: str, budget: float) -> set[str]:
    eligible = [row for row in rows if row["candidate_label"] == "SUPPORT"]
    count = min(len(eligible), max(1, round(budget * len(rows))))
    ranked = sorted(eligible, key=lambda row: (-float(row.get(score_field, 0.0)), str(row["id"])))
    return {row["id"] for row in ranked[:count]}


def bootstrap_differences(rows: list[dict], methods: list[str], budget: float, iterations: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    metrics = ["support_hallucination_rate", "error_detection_recall", "review_precision", "accuracy"]
    values = {(method, metric): [] for method in methods if method != "confidence_only" for metric in metrics}
    for _ in range(iterations):
        sample = [{**rows[rng.randrange(len(rows))], "id": f"bootstrap-{index}"} for index in range(len(rows))]
        baseline_ids = choose_reviews(sample, "risk_confidence_only", budget)
        baseline = classify_metrics(sample, baseline_ids)
        for method in methods:
            if method == "confidence_only":
                continue
            current = classify_metrics(sample, choose_reviews(sample, f"risk_{method}", budget))
            for metric in metrics:
                values[(method, metric)].append(current[metric] - baseline[metric])
    output = []
    for (method, metric), samples in values.items():
        ordered = sorted(samples)
        output.append({
            "method": method,
            "budget": budget,
            "metric": metric,
            "mean_difference_vs_confidence": sum(samples) / len(samples),
            "ci_low": ordered[int(0.025 * (len(ordered) - 1))],
            "ci_high": ordered[int(0.975 * (len(ordered) - 1))],
            "probability_better": sum(value < 0 for value in samples) / len(samples) if "hallucination" in metric else sum(value > 0 for value in samples) / len(samples),
        })
    return output


def calibration_bins(rows: list[dict], method: str, bins: int = 10) -> list[dict]:
    support = [row for row in rows if row["candidate_label"] == "SUPPORT"]
    output = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [row for row in support if lower <= float(row[f"risk_{method}"]) < upper or (index == bins - 1 and float(row[f"risk_{method}"]) == 1.0)]
        output.append({
            "method": method,
            "bin": index + 1,
            "lower": lower,
            "upper": upper,
            "count": len(members),
            "mean_predicted_risk": sum(float(row[f"risk_{method}"]) for row in members) / len(members) if members else "",
            "observed_error_rate": sum(row["gold_label"] != "SUPPORT" for row in members) / len(members) if members else "",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KG guardrails at matched review budgets.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--path-scorer-dir", default="data/processed/stage7_hierarchical_scorer/path_scorer")
    parser.add_argument("--output-dir", default="data/processed/stage7_hierarchical_scorer/matched_budget")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    guardrail_path = data_dir / "processed/stage2_primekg_semantic_clean/stage4_kg_guardrail_formal600/guardrail_results.jsonl"
    evidence_path = data_dir / "processed/stage2_primekg_semantic_clean/strict_verifier/strict_kg_evidence.jsonl"
    old_learned_path = data_dir / "processed/stage6_eswa/learned_guardrail_adjudicated/learned_guardrail_scored_results.jsonl"
    clean_learned_path = data_dir / "processed/stage2_primekg_semantic_clean/learned_guardrail_adjudicated/learned_guardrail_scored_results.jsonl"
    rows = read_jsonl(guardrail_path)
    if len(rows) != 600:
        raise ValueError(f"Expected Formal600 rows, found {len(rows)}")
    ids = {row["id"] for row in rows}
    evidence = {row["id"]: row for row in read_jsonl(evidence_path) if row["id"] in ids}
    old_learned = {row["id"]: row for row in read_jsonl(old_learned_path)}
    clean_learned = {row["id"]: row for row in read_jsonl(clean_learned_path)}
    relevance_bundle = joblib.load(Path(args.path_scorer_dir) / "path_relevance_logistic_regression.joblib")
    actionable_bundle = joblib.load(Path(args.path_scorer_dir) / "actionability_logistic_regression.joblib")

    enriched = []
    for row in rows:
        claim_evidence = evidence.get(row["id"], {})
        paths = evidence_path_rows({**claim_evidence, "qualifier_flags": row.get("qualifier_flags") or []})
        relevance = score_paths(paths, relevance_bundle)
        actionable = score_paths(paths, actionable_bundle)
        relevant_scores = [item["probabilities"].get("RELEVANT", 0.0) for item in relevance]
        actionable_scores = [item["probabilities"].get("ACTIONABLE", 0.0) for item in actionable]
        direct_count = sum(item.get("path_type") == "1-hop" for item in paths)
        two_hop_count = sum(item.get("path_type") == "2-hop" for item in paths)
        item = {
            **row,
            "evaluation_split": "calibration" if stable_fraction(str(row["id"])) < 0.4 else "test",
            "path_count": len(paths),
            "direct_path_count": direct_count,
            "two_hop_path_count": two_hop_count,
            "predicate_aligned_path_count": sum(bool(path.get("predicate_aligned")) for path in paths),
            "endpoint_aligned_path_count": sum(bool(path.get("endpoint_aligned")) for path in paths),
            "max_path_score": max((float(path.get("score", 0.0)) for path in paths), default=0.0),
            "linked_entity_count": len(claim_evidence.get("linked_entities") or []),
            "max_relevant_probability": max(relevant_scores, default=0.0),
            "mean_relevant_probability": sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0,
            "max_actionable_probability": max(actionable_scores, default=0.0),
            "mean_actionable_probability": sum(actionable_scores) / len(actionable_scores) if actionable_scores else 0.0,
            "risk_confidence_only": 1.0 - float(row.get("text_confidence", 0.0)),
            "risk_rule_context": rule_risk(row.get("guardrail_status", "")),
            "risk_original_learned": float(old_learned.get(row["id"], {}).get("learned_support_risk_score", 0.0)),
            "risk_semantic_clean_learned": float(clean_learned.get(row["id"], {}).get("learned_support_risk_score", 0.0)),
            "risk_oracle": float(row["candidate_label"] == "SUPPORT" and row["gold_label"] != "SUPPORT"),
        }
        enriched.append(item)

    calibration_support = [row for row in enriched if row["evaluation_split"] == "calibration" and row["candidate_label"] == "SUPPORT"]
    test_rows = [row for row in enriched if row["evaluation_split"] == "test"]
    if not calibration_support or len({row["gold_label"] == "SUPPORT" for row in calibration_support}) != 2:
        raise RuntimeError("Calibration SUPPORT subset must contain correct and hallucinated predictions")

    vectorizer = DictVectorizer(sparse=True)
    x_calibration = vectorizer.fit_transform([risk_features(row) for row in calibration_support])
    y_calibration = np.asarray([int(row["gold_label"] != "SUPPORT") for row in calibration_support])
    x_all = vectorizer.transform([risk_features(row) for row in enriched])
    ensemble = []
    seed_rows = []
    for seed in MODEL_SEEDS:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(calibration_support), size=len(calibration_support), replace=True)
        model = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=seed)
        model.fit(x_calibration[indices], y_calibration[indices])
        probabilities = model.predict_proba(x_all)[:, list(model.classes_).index(1)]
        ensemble.append(probabilities)
        for row, score in zip(enriched, probabilities):
            row[f"risk_hierarchical_seed_{seed}"] = float(score)
        seed_rows.append({"seed": seed, "calibration_rows": len(indices), "positive_rows": int(y_calibration[indices].sum())})
    mean_risk = np.mean(np.vstack(ensemble), axis=0)
    for row, score in zip(enriched, mean_risk):
        row["risk_hierarchical_learned"] = float(score)

    methods = ["confidence_only", "rule_context", "original_learned", "semantic_clean_learned", "hierarchical_learned", "oracle"]
    result_rows = []
    seed_result_rows = []
    for split_name, split_rows in (("test", test_rows), ("all", enriched)):
        for budget in BUDGETS:
            for method in methods:
                reviewed_ids = choose_reviews(split_rows, f"risk_{method}", budget)
                result_rows.append({"split": split_name, "method": method, "budget": budget, **classify_metrics(split_rows, reviewed_ids)})
            for seed in MODEL_SEEDS:
                reviewed_ids = choose_reviews(split_rows, f"risk_hierarchical_seed_{seed}", budget)
                seed_result_rows.append({"split": split_name, "method": "hierarchical_learned", "seed": seed, "budget": budget, **classify_metrics(split_rows, reviewed_ids)})

    calibration_rows = []
    calibration_bin_rows = []
    support_test = [row for row in test_rows if row["candidate_label"] == "SUPPORT"]
    truth = [int(row["gold_label"] != "SUPPORT") for row in support_test]
    for method in methods[:-1]:
        scores = [float(row[f"risk_{method}"]) for row in support_test]
        calibration_rows.append({
            "method": method,
            "support_test_rows": len(support_test),
            "ece": ece(truth, scores),
            "brier_score": brier_score_loss(truth, scores),
        })
        calibration_bin_rows.extend(calibration_bins(support_test, method))

    bootstrap_rows = []
    for budget in BUDGETS:
        bootstrap_rows.extend(bootstrap_differences(test_rows, methods, budget, args.bootstrap_iterations, SPLIT_SEED + round(budget * 1000)))

    write_csv(output_dir / "matched_budget_results.csv", result_rows)
    write_csv(output_dir / "risk_calibration_metrics.csv", calibration_rows)
    write_csv(output_dir / "risk_calibration_bins.csv", calibration_bin_rows)
    write_csv(output_dir / "paired_bootstrap_differences.csv", bootstrap_rows)
    write_csv(output_dir / "risk_model_seed_manifest.csv", seed_rows)
    write_csv(output_dir / "hierarchical_seed_results.csv", seed_result_rows)
    write_jsonl(output_dir / "claim_risk_scores.jsonl", enriched)
    stats = {
        "formal600_rows": len(enriched),
        "calibration_rows": sum(row["evaluation_split"] == "calibration" for row in enriched),
        "test_rows": len(test_rows),
        "calibration_support_rows": len(calibration_support),
        "test_support_rows": len(support_test),
        "split_seed": SPLIT_SEED,
        "model_seeds": MODEL_SEEDS,
        "review_budgets": BUDGETS,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    (output_dir / "matched_budget_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
