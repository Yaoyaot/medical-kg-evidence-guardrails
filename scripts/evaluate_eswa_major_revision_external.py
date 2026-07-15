from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from build_text_rag_kg_guardrail import build_row
from evaluate_eswa_major_revision import (
    PRIMARY_MODEL,
    ece,
    features,
    path_quality_features,
    risk_ranking_metrics,
    selective_metrics,
)
from evaluate_matched_budget_guardrails import rule_risk


SEED = 20260618
SAMPLE_SIZES = (20, 40, 60, 80, 100)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def top_ids(rows: list[dict], field: str, count: int) -> set[str]:
    eligible = [row for row in rows if row["candidate_label"] == "SUPPORT"]
    ranked = sorted(eligible, key=lambda row: (-float(row[field]), row["id"]))
    return {row["id"] for row in ranked[: min(count, len(ranked))]}


def learning_curve(rows: list[dict], c_value: float, repeats: int) -> list[dict]:
    support = [row for row in rows if row["candidate_label"] == "SUPPORT"]
    output = []
    for sample_size in SAMPLE_SIZES:
        for repeat in range(repeats):
            rng = random.Random(f"{SEED}:{sample_size}:{repeat}")
            indices = list(range(len(support)))
            rng.shuffle(indices)
            train_indices = indices[:sample_size]
            test_indices = indices[sample_size:]
            train = [support[index] for index in train_indices]
            test = [support[index] for index in test_indices]
            y_train = np.asarray([int(row["gold_label"] != "SUPPORT") for row in train])
            y_test = np.asarray([int(row["gold_label"] != "SUPPORT") for row in test])
            if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                continue
            vectorizer = DictVectorizer(sparse=True)
            x_train = vectorizer.fit_transform([features(row, PRIMARY_MODEL) for row in train])
            model = LogisticRegression(C=c_value, penalty="l2", class_weight="balanced", max_iter=5000, random_state=SEED)
            model.fit(x_train, y_train)
            scores = model.predict_proba(vectorizer.transform([features(row, PRIMARY_MODEL) for row in test]))[:, list(model.classes_).index(1)]
            budget = min(len(test), math.ceil(0.05 * len(test)))
            ranked = np.argsort(-scores)[:budget]
            detected = int(y_test[ranked].sum())
            output.append({
                "analysis": "post_hoc_target_domain_adaptation_diagnostic",
                "training_support_predictions": sample_size,
                "test_support_predictions": len(test),
                "repeat": repeat,
                "training_error_events": int(y_train.sum()),
                "test_error_events": int(y_test.sum()),
                "auroc": roc_auc_score(y_test, scores),
                "average_precision": average_precision_score(y_test, scores),
                "top_5pct_reviewed": budget,
                "top_5pct_detected_errors": detected,
                "top_5pct_error_detection_recall": detected / int(y_test.sum()),
            })
    return output


def summarize_learning_curve(rows: list[dict]) -> list[dict]:
    output = []
    metrics = ("auroc", "average_precision", "top_5pct_error_detection_recall")
    for sample_size in SAMPLE_SIZES:
        subset = [row for row in rows if row["training_support_predictions"] == sample_size]
        item = {"training_support_predictions": sample_size, "valid_repeats": len(subset)}
        for metric in metrics:
            values = sorted(float(row[metric]) for row in subset)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_ci_low"] = values[int(0.025 * (len(values) - 1))]
            item[f"{metric}_ci_high"] = values[int(0.975 * (len(values) - 1))]
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Stage 9 frozen Formal600 risk model and post-hoc PubMedQA adaptation diagnostic.")
    parser.add_argument("--stage8-dir", default="data/processed/stage8_pubmedqa_external")
    parser.add_argument("--model-path", default="data/processed/stage9_eswa_major_revision/formal600_crossfit/formal600_full_frozen_risk_model.joblib")
    parser.add_argument("--path-scorer-dir", default="data/processed/stage7_hierarchical_scorer/path_scorer")
    parser.add_argument("--output-dir", default="data/processed/stage9_eswa_major_revision/pubmedqa_external")
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    stage8 = Path(args.stage8_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = [row for row in read_jsonl(stage8 / "external_baseline_results.jsonl") if row.get("baseline") == "text_rag_llm"]
    if len(baseline) != 300:
        raise ValueError(f"Expected 300 provided-text rows, found {len(baseline)}")
    evidence = {row["id"]: row for row in read_jsonl(stage8 / "kg/strict_verifier/strict_kg_evidence.jsonl")}
    relevance_bundle = joblib.load(Path(args.path_scorer_dir) / "path_relevance_logistic_regression.joblib")
    actionable_bundle = joblib.load(Path(args.path_scorer_dir) / "actionability_logistic_regression.joblib")
    frozen = joblib.load(Path(args.model_path))

    feature_rows = []
    gold_by_id = {}
    enriched_rows = []
    for source in baseline:
        gold_by_id[source["id"]] = source["gold_label"]
        no_gold = {key: value for key, value in source.items() if key != "gold_label"}
        rule_row = build_row(no_gold, evidence[source["id"]])
        rule_row.pop("gold_label", None)
        enriched = {
            **rule_row,
            **path_quality_features(evidence[source["id"]], rule_row, relevance_bundle, actionable_bundle),
            "risk_confidence": 1.0 - float(rule_row.get("text_confidence", 0.0)),
            "risk_rule": rule_risk(rule_row.get("guardrail_status", "")),
        }
        if "gold_label" in enriched:
            raise AssertionError("PubMedQA gold entered frozen feature construction")
        feature_rows.append(features(enriched, PRIMARY_MODEL))
        enriched_rows.append(enriched)
    x = frozen["vectorizer"].transform(feature_rows)
    scores = frozen["model"].predict_proba(x)[:, list(frozen["model"].classes_).index(1)]
    rows = []
    for row, score in zip(enriched_rows, scores):
        rows.append({
            **row,
            "risk_full_without_dataset_source": float(score),
            "gold_label": gold_by_id[row["id"]],
            "external_gold_used_for_risk_scoring": False,
        })

    ranking = [
        risk_ranking_metrics(rows, "risk_confidence", "self_reported_confidence"),
        risk_ranking_metrics(rows, "risk_rule", "rule_matched_budget"),
        risk_ranking_metrics(rows, "risk_full_without_dataset_source", PRIMARY_MODEL),
    ]
    review_count = 15
    selective = []
    for method, field in (
        ("self_reported_confidence", "risk_confidence"),
        ("rule_matched_budget", "risk_rule"),
        (PRIMARY_MODEL, "risk_full_without_dataset_source"),
    ):
        selected = top_ids(rows, field, review_count)
        selective.append({"method": method, "review_budget": review_count, **selective_metrics(rows, selected)})

    learning = learning_curve(rows, float(frozen["C"]), args.repeats)
    learning_summary = summarize_learning_curve(learning)
    write_jsonl(output / "pubmedqa_frozen_risk_scores.jsonl", rows)
    write_csv(output / "pubmedqa_frozen_risk_ranking.csv", ranking)
    write_csv(output / "pubmedqa_frozen_selective_metrics.csv", selective)
    write_csv(output / "pubmedqa_target_adaptation_learning_curve.csv", learning)
    write_csv(output / "pubmedqa_target_adaptation_summary.csv", learning_summary)
    stats = {
        "claims": len(rows),
        "support_predictions": sum(row["candidate_label"] == "SUPPORT" for row in rows),
        "review_budget": review_count,
        "formal600_model_training_support_predictions": frozen["training_support_predictions"],
        "formal600_model_training_error_events": frozen["training_error_events"],
        "external_gold_used_for_frozen_risk_scoring": False,
        "frozen_ranking": ranking,
        "frozen_selective": selective,
        "target_adaptation_is_post_hoc_diagnostic": True,
        "target_adaptation_repeats": args.repeats,
    }
    (output / "pubmedqa_major_revision_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
