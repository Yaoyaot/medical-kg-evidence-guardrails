from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from evaluate_matched_budget_guardrails import rule_risk, score_paths
from hierarchical_evidence_features import evidence_path_rows


SEED = 20260618
LABELS = ("SUPPORT", "REFUTE", "UNCERTAIN")
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.20)
CS = (0.01, 0.1, 1.0, 10.0)
PRIMARY_MODEL = "full_without_dataset_source"
ABLATIONS = (
    "confidence_only_features",
    "dataset_source_only",
    "kg_evidence_state_only",
    "semantic_rules_only",
    "path_statistics_only",
    "evidence_scorer_only",
    "confidence_plus_kg_rules",
    "confidence_plus_evidence_scorer",
    "full_without_dataset_source",
    "full_with_dataset_source",
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def path_quality_features(evidence: dict, row: dict, relevance_bundle: dict, actionable_bundle: dict) -> dict:
    paths = evidence_path_rows({**evidence, "qualifier_flags": row.get("qualifier_flags") or []})
    relevance = score_paths(paths, relevance_bundle)
    actionable = score_paths(paths, actionable_bundle)
    relevant_scores = [item["probabilities"].get("RELEVANT", 0.0) for item in relevance]
    actionable_scores = [item["probabilities"].get("ACTIONABLE", 0.0) for item in actionable]
    qualifiers = list(row.get("qualifier_flags") or [])
    direct_paths = list(evidence.get("direct_paths") or [])
    aligned_direct = [item for item in direct_paths if item.get("predicate_aligned")]
    unresolved_direct = [item for item in direct_paths if item.get("claim_predicate_families") == ["UNRESOLVED"]]
    negative_relations = {"contraindicates", "not_presents", "not_expressed"}
    positive_present = any(item.get("relations") and item["relations"][0] not in negative_relations for item in aligned_direct)
    negative_present = any(item.get("relations") and item["relations"][0] in negative_relations for item in aligned_direct)
    if aligned_direct:
        evidence_state = "DIRECT_MATCH_QUALIFIER_INCOMPLETE" if qualifiers else "DIRECT_MATCH_QUALIFIER_COMPATIBLE"
    elif direct_paths and unresolved_direct:
        evidence_state = "DIRECT_RELATION_UNRESOLVED"
    elif direct_paths:
        evidence_state = "DIRECT_PREDICATE_OR_DIRECTION_MISMATCH"
    elif evidence.get("two_hop_context_paths"):
        evidence_state = "TWO_HOP_CONTEXT"
    elif evidence.get("linked_entities"):
        evidence_state = "SINGLE_ENTITY_CONTEXT"
    else:
        evidence_state = "NO_KG_GROUNDING"
    return {
        "path_count": len(paths),
        "direct_path_count": sum(item.get("path_type") == "1-hop" for item in paths),
        "two_hop_path_count": sum(item.get("path_type") == "2-hop" for item in paths),
        "predicate_aligned_path_count": sum(bool(item.get("predicate_aligned")) for item in paths),
        "endpoint_aligned_path_count": sum(bool(item.get("endpoint_aligned")) for item in paths),
        "max_path_score": max((float(item.get("score", 0.0)) for item in paths), default=0.0),
        "linked_entity_count": len(evidence.get("linked_entities") or []),
        "max_relevant_probability": max(relevant_scores, default=0.0),
        "mean_relevant_probability": sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0,
        "max_actionable_probability": max(actionable_scores, default=0.0),
        "mean_actionable_probability": sum(actionable_scores) / len(actionable_scores) if actionable_scores else 0.0,
        "evidence_state_nominal": evidence_state,
        "qualifier_compatible_for_strict_kg": not qualifiers,
        "direct_conflict_present": positive_present and negative_present,
        "strict_structured_candidate_count": len(aligned_direct) if not qualifiers else 0,
    }


def feature_groups(row: dict) -> dict[str, dict]:
    confidence = {"self_reported_confidence": float(row.get("text_confidence", 0.0))}
    dataset = {"dataset_source": str(row.get("dataset", "unknown"))}
    kg_state = {
        "guardrail_status": str(row.get("guardrail_status", "KG_NO_GROUNDING")),
        "evidence_state": str(row.get("evidence_state_nominal", row.get("kg_evidence_tier", "NO_KG_GROUNDING"))),
        "has_any_path": float(row.get("path_count", 0) > 0),
        "has_direct_path": float(row.get("direct_path_count", 0) > 0),
        "has_two_hop_path": float(row.get("two_hop_path_count", 0) > 0),
        "direct_conflict_present": float(bool(row.get("direct_conflict_present"))),
    }
    semantic = {
        "predicate_aligned_path_count": float(row.get("predicate_aligned_path_count", 0)),
        "endpoint_aligned_path_count": float(row.get("endpoint_aligned_path_count", 0)),
        "qualifier_compatible_for_strict_kg": float(bool(row.get("qualifier_compatible_for_strict_kg"))),
        "strict_structured_candidate_count": float(row.get("strict_structured_candidate_count", 0)),
    }
    for qualifier in row.get("qualifier_flags") or []:
        semantic[f"qualifier::{qualifier}"] = 1.0
    path_stats = {
        "path_count": float(row.get("path_count", 0)),
        "direct_path_count": float(row.get("direct_path_count", 0)),
        "two_hop_path_count": float(row.get("two_hop_path_count", 0)),
        "max_path_score": float(row.get("max_path_score", 0.0)),
        "linked_entity_count": float(row.get("linked_entity_count", 0)),
    }
    scorer = {
        "max_actionable_probability": float(row.get("max_actionable_probability", 0.0)),
        "mean_actionable_probability": float(row.get("mean_actionable_probability", 0.0)),
        "max_relevant_probability": float(row.get("max_relevant_probability", 0.0)),
        "mean_relevant_probability": float(row.get("mean_relevant_probability", 0.0)),
    }
    return {"confidence": confidence, "dataset": dataset, "kg_state": kg_state, "semantic": semantic, "path": path_stats, "scorer": scorer}


def features(row: dict, model_name: str) -> dict:
    groups = feature_groups(row)
    selected = {
        "confidence_only_features": ("confidence",),
        "dataset_source_only": ("dataset",),
        "kg_evidence_state_only": ("kg_state",),
        "semantic_rules_only": ("semantic",),
        "path_statistics_only": ("path",),
        "evidence_scorer_only": ("scorer",),
        "confidence_plus_kg_rules": ("confidence", "kg_state", "semantic"),
        "confidence_plus_evidence_scorer": ("confidence", "scorer"),
        "full_without_dataset_source": ("confidence", "kg_state", "semantic", "path", "scorer"),
        "full_with_dataset_source": ("confidence", "dataset", "kg_state", "semantic", "path", "scorer"),
    }[model_name]
    output = {}
    for group in selected:
        output.update(groups[group])
    return output


def safe_auc(y: list[int], scores: list[float]) -> float:
    return roc_auc_score(y, scores) if len(set(y)) == 2 else float("nan")


def safe_ap(y: list[int], scores: list[float]) -> float:
    return average_precision_score(y, scores) if len(set(y)) == 2 else float("nan")


def ece(y: list[int], scores: list[float], bins: int = 10) -> float:
    if not y:
        return float("nan")
    total = len(y)
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, score in enumerate(scores) if low <= score < high or (index == bins - 1 and score == 1.0)]
        if members:
            result += len(members) / total * abs(np.mean([scores[i] for i in members]) - np.mean([y[i] for i in members]))
    return float(result)


def choose_c(rows: list[dict], model_name: str) -> tuple[float, list[dict]]:
    y = np.asarray([int(row["gold_label"] != "SUPPORT") for row in rows])
    groups = np.asarray([row["pair_group_id"] for row in rows])
    unique_groups = len(set(groups))
    splits = min(3, unique_groups)
    diagnostics = []
    if splits < 2 or len(set(y)) < 2:
        return 1.0, diagnostics
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform([features(row, model_name) for row in rows])
    cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=SEED)
    for c_value in CS:
        aps = []
        for train, test in cv.split(x, y, groups):
            if len(set(y[train])) < 2 or len(set(y[test])) < 2:
                continue
            model = LogisticRegression(C=c_value, penalty="l2", class_weight="balanced", max_iter=5000, random_state=SEED)
            model.fit(x[train], y[train])
            scores = model.predict_proba(x[test])[:, list(model.classes_).index(1)]
            aps.append(average_precision_score(y[test], scores))
        diagnostics.append({"model": model_name, "C": c_value, "inner_folds_used": len(aps), "mean_average_precision": float(np.mean(aps)) if aps else float("nan")})
    valid = [row for row in diagnostics if not math.isnan(row["mean_average_precision"])]
    selected = max(valid, key=lambda row: (row["mean_average_precision"], -float(row["C"]))) if valid else {"C": 1.0}
    return float(selected["C"]), diagnostics


def fit_and_score(train: list[dict], test: list[dict], model_name: str) -> tuple[np.ndarray, dict, list[dict]]:
    c_value, diagnostics = choose_c(train, model_name)
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([features(row, model_name) for row in train])
    y_train = np.asarray([int(row["gold_label"] != "SUPPORT") for row in train])
    if len(set(y_train)) < 2:
        scores = np.repeat(float(np.mean(y_train)) if len(y_train) else 0.5, len(test))
        return scores, {"C": c_value, "feature_count": x_train.shape[1], "event_count": int(y_train.sum()), "non_event_count": int(len(y_train) - y_train.sum())}, diagnostics
    model = LogisticRegression(C=c_value, penalty="l2", class_weight="balanced", max_iter=5000, random_state=SEED)
    model.fit(x_train, y_train)
    x_test = vectorizer.transform([features(row, model_name) for row in test])
    scores = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    coefficients = [
        {"feature": name, "coefficient": float(value)}
        for name, value in zip(vectorizer.get_feature_names_out(), model.coef_[0])
    ]
    manifest = {
        "C": c_value,
        "feature_count": x_train.shape[1],
        "event_count": int(y_train.sum()),
        "non_event_count": int(len(y_train) - y_train.sum()),
        "samples_per_feature": len(y_train) / max(x_train.shape[1], 1),
        "coefficients": coefficients,
    }
    return scores, manifest, diagnostics


def select_fold_budget(rows: list[dict], score_field: str, budget: float) -> set[str]:
    selected = set()
    for fold in range(5):
        fold_rows = [row for row in rows if int(row["outer_fold"]) == fold]
        eligible = [row for row in fold_rows if row["candidate_label"] == "SUPPORT"]
        count = min(len(eligible), math.ceil(budget * len(fold_rows)))
        ranked = sorted(eligible, key=lambda row: (-float(row.get(score_field, 0.0)), row["id"]))
        selected.update(row["id"] for row in ranked[:count])
    return selected


def base_metrics(rows: list[dict]) -> dict:
    gold = [row["gold_label"] for row in rows]
    pred = [row["candidate_label"] for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(gold, pred, labels=list(LABELS), zero_division=0)
    support_indices = [i for i, value in enumerate(pred) if value == "SUPPORT"]
    wrong = sum(gold[i] != "SUPPORT" for i in support_indices)
    return {
        "count": len(rows),
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=list(LABELS), average="macro", zero_division=0),
        "support_predictions": len(support_indices),
        "support_precision": float(precision[0]),
        "support_recall": float(recall[0]),
        "support_f1": float(f1[0]),
        "false_support_count": wrong,
        "false_support_rate": wrong / len(support_indices) if support_indices else 0.0,
        **{f"{label.lower()}_support": int(value) for label, value in zip(LABELS, support)},
    }


def selective_metrics(rows: list[dict], reviewed: set[str]) -> dict:
    accepted = [row for row in rows if row["id"] not in reviewed]
    reviewed_rows = [row for row in rows if row["id"] in reviewed]
    false_supports = [row for row in rows if row["candidate_label"] == "SUPPORT" and row["gold_label"] != "SUPPORT"]
    detected = [row for row in reviewed_rows if row["candidate_label"] == "SUPPORT" and row["gold_label"] != "SUPPORT"]
    false_reviews = [row for row in reviewed_rows if row["candidate_label"] == "SUPPORT" and row["gold_label"] == "SUPPORT"]
    accepted_gold = [row["gold_label"] for row in accepted]
    accepted_pred = [row["candidate_label"] for row in accepted]
    accepted_support = [row for row in accepted if row["candidate_label"] == "SUPPORT"]
    accepted_false = sum(row["gold_label"] != "SUPPORT" for row in accepted_support)
    base_correct = sum(row["candidate_label"] == row["gold_label"] for row in rows)
    reviewed_errors = sum(row["candidate_label"] != row["gold_label"] for row in reviewed_rows)
    accuracy = accuracy_score(accepted_gold, accepted_pred) if accepted else float("nan")
    return {
        "reviewed_count": len(reviewed_rows),
        "review_rate": len(reviewed_rows) / len(rows),
        "coverage": len(accepted) / len(rows),
        "selective_accuracy": accuracy,
        "selective_risk": 1.0 - accuracy if accepted else float("nan"),
        "selective_macro_f1": f1_score(accepted_gold, accepted_pred, labels=list(LABELS), average="macro", zero_division=0) if accepted else float("nan"),
        "detected_false_supports": len(detected),
        "missed_false_supports": len(false_supports) - len(detected),
        "incorrectly_reviewed_correct_supports": len(false_reviews),
        "error_detection_precision": len(detected) / len(reviewed_rows) if reviewed_rows else 0.0,
        "error_detection_recall": len(detected) / len(false_supports) if false_supports else 0.0,
        "accepted_support_predictions": len(accepted_support),
        "accepted_support_precision": 1.0 - accepted_false / len(accepted_support) if accepted_support else float("nan"),
        "accepted_false_support_rate": accepted_false / len(accepted_support) if accepted_support else float("nan"),
        "oracle_review_accuracy_upper_bound": (base_correct + reviewed_errors) / len(rows),
    }


def risk_ranking_metrics(rows: list[dict], score_field: str, method: str) -> dict:
    support = [row for row in rows if row["candidate_label"] == "SUPPORT"]
    y = [int(row["gold_label"] != "SUPPORT") for row in support]
    scores = [float(row[score_field]) for row in support]
    return {
        "method": method,
        "support_predictions": len(support),
        "error_events": sum(y),
        "non_error_events": len(y) - sum(y),
        "auroc": safe_auc(y, scores),
        "average_precision": safe_ap(y, scores),
        "brier_score": brier_score_loss(y, scores),
        "ece_10_bins": ece(y, scores),
        "mean_risk_false_support": float(np.mean([score for score, truth in zip(scores, y) if truth])) if any(y) else float("nan"),
        "mean_risk_correct_support": float(np.mean([score for score, truth in zip(scores, y) if not truth])) if not all(y) else float("nan"),
    }


def bootstrap_primary(rows: list[dict], iterations: int) -> list[dict]:
    components = defaultdict(list)
    for row in rows:
        components[row["pair_group_id"]].append(row)
    keys = sorted(components)
    rng = random.Random(SEED)
    metrics = ("error_detection_recall", "error_detection_precision", "selective_risk", "accepted_false_support_rate")
    values = {metric: [] for metric in metrics}
    for _ in range(iterations):
        sample = []
        for draw, key in enumerate(rng.choices(keys, k=len(keys))):
            for local, source in enumerate(components[key]):
                sample.append({**source, "id": f"b{draw}-{local}-{source['id']}"})
        confidence = selective_metrics(sample, select_fold_budget(sample, "risk_confidence", 0.05))
        learned = selective_metrics(sample, select_fold_budget(sample, f"risk_{PRIMARY_MODEL}", 0.05))
        for metric in metrics:
            left, right = learned[metric], confidence[metric]
            if not math.isnan(left) and not math.isnan(right):
                values[metric].append(left - right)
    output = []
    for metric, samples in values.items():
        ordered = sorted(samples)
        output.append({
            "method": PRIMARY_MODEL,
            "reference": "self_reported_confidence",
            "budget": 0.05,
            "metric": metric,
            "mean_difference": float(np.mean(samples)),
            "ci_low": ordered[int(0.025 * (len(ordered) - 1))],
            "ci_high": ordered[int(0.975 * (len(ordered) - 1))],
            "iterations": iterations,
            "resampling_unit": "claim_source_component",
        })
    return output


def prevalence_adjusted(rows: list[dict]) -> list[dict]:
    rates = {}
    for label in LABELS:
        subset = [row for row in rows if row["gold_label"] == label]
        rates[label] = sum(row["candidate_label"] == "SUPPORT" for row in subset) / len(subset)
    scenarios = {
        "balanced_formal600": {"SUPPORT": 1 / 3, "REFUTE": 1 / 3, "UNCERTAIN": 1 / 3},
        "medfact_bench_2000": {"SUPPORT": 1096 / 2000, "REFUTE": 305 / 2000, "UNCERTAIN": 599 / 2000},
        "low_support": {"SUPPORT": 0.20, "REFUTE": 0.30, "UNCERTAIN": 0.50},
        "high_support": {"SUPPORT": 0.70, "REFUTE": 0.15, "UNCERTAIN": 0.15},
    }
    output = []
    for name, prior in scenarios.items():
        denominator = sum(prior[label] * rates[label] for label in LABELS)
        precision = prior["SUPPORT"] * rates["SUPPORT"] / denominator if denominator else float("nan")
        output.append({
            "scenario": name,
            **{f"prior_{label.lower()}": prior[label] for label in LABELS},
            **{f"predicted_support_rate_given_{label.lower()}": rates[label] for label in LABELS},
            "prevalence_adjusted_support_precision": precision,
            "prevalence_adjusted_false_support_rate": 1.0 - precision,
        })
    return output


def fit_final_bundle(rows: list[dict], output: Path) -> dict:
    support = [row for row in rows if row["candidate_label"] == "SUPPORT"]
    c_value, diagnostics = choose_c(support, PRIMARY_MODEL)
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform([features(row, PRIMARY_MODEL) for row in support])
    y = np.asarray([int(row["gold_label"] != "SUPPORT") for row in support])
    model = LogisticRegression(C=c_value, penalty="l2", class_weight="balanced", max_iter=5000, random_state=SEED)
    model.fit(x, y)
    bundle = {
        "model_name": PRIMARY_MODEL,
        "vectorizer": vectorizer,
        "model": model,
        "C": c_value,
        "feature_count": x.shape[1],
        "training_support_predictions": len(support),
        "training_error_events": int(y.sum()),
        "seed": SEED,
    }
    joblib.dump(bundle, output / "formal600_full_frozen_risk_model.joblib")
    coefficients = [
        {"model": PRIMARY_MODEL, "feature": name, "coefficient": float(value), "C": c_value}
        for name, value in zip(vectorizer.get_feature_names_out(), model.coef_[0])
    ]
    write_csv(output / "formal600_final_risk_coefficients.csv", coefficients)
    write_csv(output / "formal600_final_inner_cv.csv", diagnostics)
    return {key: value for key, value in bundle.items() if key not in {"vectorizer", "model"}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ESWA major-revision grouped cross-fitting and selective-risk analysis.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--group-manifest", default="data/processed/stage9_eswa_major_revision/grouping/formal600_group_manifest.csv")
    parser.add_argument("--output-dir", default="data/processed/stage9_eswa_major_revision/formal600_crossfit")
    parser.add_argument("--path-scorer-dir", default="data/processed/stage7_hierarchical_scorer/path_scorer")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    data = Path(args.data_dir) / "processed"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {row["id"]: row for row in read_csv(Path(args.group_manifest))}
    guardrail = read_jsonl(data / "stage2_primekg_semantic_clean/stage4_kg_guardrail_formal600/guardrail_results.jsonl")
    evidence = {row["id"]: row for row in read_jsonl(data / "stage2_primekg_semantic_clean/strict_verifier/strict_kg_evidence.jsonl")}
    relevance_bundle = joblib.load(Path(args.path_scorer_dir) / "path_relevance_logistic_regression.joblib")
    actionable_bundle = joblib.load(Path(args.path_scorer_dir) / "actionability_logistic_regression.joblib")
    if len(guardrail) != 600 or set(manifest) != {row["id"] for row in guardrail}:
        raise ValueError("Formal600 group manifest does not match guardrail rows")

    rows = []
    for row in guardrail:
        group = manifest[row["id"]]
        enriched = {
            **row,
            **path_quality_features(evidence[row["id"]], row, relevance_bundle, actionable_bundle),
            "pair_group_id": group["pair_group_id"],
            "outer_fold": int(group["outer_fold"]),
            "risk_confidence": 1.0 - float(row.get("text_confidence", 0.0)),
            "risk_rule": rule_risk(row.get("guardrail_status", "")),
            "risk_oracle": float(row.get("candidate_label") == "SUPPORT" and row.get("gold_label") != "SUPPORT"),
        }
        rows.append(enriched)

    model_manifests = []
    inner_rows = []
    coefficient_rows = []
    for outer_fold in range(5):
        train = [row for row in rows if row["outer_fold"] != outer_fold and row["candidate_label"] == "SUPPORT"]
        test = [row for row in rows if row["outer_fold"] == outer_fold and row["candidate_label"] == "SUPPORT"]
        for model_name in ABLATIONS:
            scores, model_manifest, diagnostics = fit_and_score(train, test, model_name)
            for row, score in zip(test, scores):
                row[f"risk_{model_name}"] = float(score)
            model_manifests.append({
                "outer_fold": outer_fold,
                "model": model_name,
                "training_support_predictions": len(train),
                "test_support_predictions": len(test),
                **{key: value for key, value in model_manifest.items() if key != "coefficients"},
            })
            inner_rows.extend({"outer_fold": outer_fold, **item} for item in diagnostics)
            coefficient_rows.extend({"outer_fold": outer_fold, "model": model_name, **item} for item in model_manifest.get("coefficients", []))

    for row in rows:
        if row["candidate_label"] != "SUPPORT":
            for model_name in ABLATIONS:
                row[f"risk_{model_name}"] = 0.0

    risk_methods = {
        "self_reported_confidence": "risk_confidence",
        "rule_matched_budget": "risk_rule",
        **{model_name: f"risk_{model_name}" for model_name in ABLATIONS},
        "oracle": "risk_oracle",
    }
    ranking = [risk_ranking_metrics(rows, field, method) for method, field in risk_methods.items()]
    selective = []
    risk_curve = []
    for method, field in risk_methods.items():
        for budget in BUDGETS:
            reviewed = select_fold_budget(rows, field, budget)
            selective.append({"method": method, "budget": budget, **selective_metrics(rows, reviewed)})
        for percent in range(0, 21):
            budget = percent / 100
            risk_curve.append({"method": method, "budget": budget, **selective_metrics(rows, select_fold_budget(rows, field, budget))})
    # Random-review expectation at matched budgets; random scores are independent of gold.
    for budget in BUDGETS:
        simulations = []
        for simulation in range(1000):
            for row in rows:
                digest = hashlib.sha256(f"{SEED}:{simulation}:{row['id']}".encode("utf-8")).hexdigest()
                row["risk_random"] = int(digest[:16], 16) / 16**16
            simulations.append(selective_metrics(rows, select_fold_budget(rows, "risk_random", budget)))
        selective.append({"method": "random_review", "budget": budget, **{key: float(np.mean([item[key] for item in simulations])) for key in simulations[0]}})

    aurc_rows = []
    for method in risk_methods:
        points = sorted([row for row in risk_curve if row["method"] == method], key=lambda row: row["coverage"])
        aurc = float(np.trapz([row["selective_risk"] for row in points], [row["coverage"] for row in points]))
        aurc_rows.append({"method": method, "coverage_min": points[0]["coverage"], "coverage_max": points[-1]["coverage"], "aurc_0_to_20pct_review": aurc})

    base = base_metrics(rows)
    matrix = confusion_matrix([row["gold_label"] for row in rows], [row["candidate_label"] for row in rows], labels=list(LABELS))
    confusion_rows = [
        {"gold_label": gold, "predicted_label": pred, "count": int(matrix[i, j])}
        for i, gold in enumerate(LABELS) for j, pred in enumerate(LABELS)
    ]
    bootstrap = bootstrap_primary(rows, args.bootstrap_iterations)
    final_manifest = fit_final_bundle(rows, output)

    write_jsonl(output / "formal600_crossfit_risk_scores.jsonl", rows)
    write_csv(output / "formal600_base_metrics.csv", [{"method": "provided_text_verifier", **base}])
    write_csv(output / "formal600_confusion_matrix.csv", confusion_rows)
    write_csv(output / "formal600_risk_ranking.csv", ranking)
    write_csv(output / "formal600_selective_metrics.csv", selective)
    write_csv(output / "formal600_risk_coverage_curve.csv", risk_curve)
    write_csv(output / "formal600_aurc.csv", aurc_rows)
    write_csv(output / "formal600_group_bootstrap.csv", bootstrap)
    write_csv(output / "formal600_prevalence_adjusted_precision.csv", prevalence_adjusted(rows))
    write_csv(output / "formal600_outer_model_manifest.csv", model_manifests)
    write_csv(output / "formal600_inner_cv.csv", inner_rows)
    write_csv(output / "formal600_outer_coefficients.csv", coefficient_rows)

    primary_ranking = next(row for row in ranking if row["method"] == PRIMARY_MODEL)
    confidence_ranking = next(row for row in ranking if row["method"] == "self_reported_confidence")
    primary_5 = next(row for row in selective if row["method"] == PRIMARY_MODEL and row["budget"] == 0.05)
    confidence_5 = next(row for row in selective if row["method"] == "self_reported_confidence" and row["budget"] == 0.05)
    recall_ci = next(row for row in bootstrap if row["metric"] == "error_detection_recall")
    gate = (
        primary_5["error_detection_recall"] > confidence_5["error_detection_recall"]
        and recall_ci["ci_low"] > 0
    )
    stats = {
        "evaluation_design": "five-fold claim/source/near-duplicate grouped cross-fitting",
        "rows": len(rows),
        "groups": len({row["pair_group_id"] for row in rows}),
        "folds": 5,
        "base": base,
        "primary_model": PRIMARY_MODEL,
        "primary_ranking": primary_ranking,
        "confidence_ranking": confidence_ranking,
        "primary_5pct": primary_5,
        "confidence_5pct": confidence_5,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": SEED,
        "final_frozen_model": final_manifest,
        "result_gate_passed": gate,
        "result_gate_rule": "learned 5% error-detection recall > confidence and group-bootstrap CI lower bound > 0",
    }
    (output / "formal600_major_revision_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
