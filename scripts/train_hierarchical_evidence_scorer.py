from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from hierarchical_evidence_features import (
    build_adverse_event_pairs,
    combined_text,
    semantic_clean_path,
    structured_features,
)


RELEVANCE_LABELS = ["IRRELEVANT", "PARTIAL", "RELEVANT"]
ACTION_LABELS = ["NON_ACTIONABLE", "ACTIONABLE"]
SEEDS = [11, 23, 37, 53, 71]


def read_csv(path: Path) -> list[dict]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Unable to decode {path}")


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


def prepare_rows(rows: list[dict], adverse_pairs: set[tuple[str, str]]) -> tuple[list[dict], int]:
    output = []
    remapped_total = 0
    for source in rows:
        if str(source.get("artifact_excluded", "")).lower() == "true":
            continue
        row = dict(source)
        cleaned, relations, remapped = semantic_clean_path(row.get("path_text", ""), row.get("relations", ""), adverse_pairs)
        row["path_text"] = cleaned
        row["relations"] = ";".join(relations)
        row["semantic_relation_remapped"] = str(bool(remapped)).lower()
        row["actionability"] = "ACTIONABLE" if row.get("evidence_role") in {"SUPPORT", "REFUTE"} else "NON_ACTIONABLE"
        remapped_total += remapped
        output.append(row)
    return output, remapped_total


def make_matrices(train_rows: list[dict], test_rows: list[dict]):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000, sublinear_tf=True)
    vectorizer = DictVectorizer(sparse=True)
    x_text_train = tfidf.fit_transform([combined_text(row) for row in train_rows])
    x_text_test = tfidf.transform([combined_text(row) for row in test_rows])
    x_struct_train = vectorizer.fit_transform([structured_features(row) for row in train_rows])
    x_struct_test = vectorizer.transform([structured_features(row) for row in test_rows])
    return hstack([x_text_train, x_struct_train]).tocsr(), hstack([x_text_test, x_struct_test]).tocsr(), tfidf, vectorizer


def model_for(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0, random_state=seed)
    return RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    result = {
        "count": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }
    for index, label in enumerate(labels):
        key = label.lower()
        result[f"{key}_precision"] = precision[index]
        result[f"{key}_recall"] = recall[index]
        result[f"{key}_f1"] = f1[index]
        result[f"{key}_support"] = int(support[index])
    return result


def rule_prediction(row: dict, task: str) -> str:
    tier = str(row.get("evidence_tier") or "")
    if task == "actionability":
        return "ACTIONABLE" if tier == "DIRECT_PREDICATE_MATCH" else "NON_ACTIONABLE"
    if tier == "DIRECT_PREDICATE_MATCH":
        return "RELEVANT"
    if tier in {"DIRECT_RELATION_UNRESOLVED", "DIRECT_PREDICATE_MISMATCH", "TWO_HOP_CONTEXT", "SINGLE_ENTITY_CONTEXT"}:
        return "PARTIAL"
    return "IRRELEVANT"


def fit_final(rows: list[dict], task: str, model_name: str, seed: int):
    x, _, tfidf, vectorizer = make_matrices(rows, rows[:1])
    labels = [row[task] for row in rows]
    model = model_for(model_name, seed)
    model.fit(x, labels)
    return {"tfidf": tfidf, "dict_vectorizer": vectorizer, "model": model, "task": task}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical path evidence scorers with grouped CV.")
    parser.add_argument("--annotations-path", default="data/processed/stage6_eswa/annotations_gold/path_annotations_modeling_pool.csv")
    parser.add_argument("--primekg-graph-dir", default="data/processed/stage2_primekg_semantic_clean/primekg_graph")
    parser.add_argument("--output-dir", default="data/processed/stage7_hierarchical_scorer/path_scorer")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--expected-rows", type=int, default=474)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_csv(Path(args.annotations_path))
    artifact_count = sum(str(row.get("artifact_excluded", "")).lower() == "true" for row in raw_rows)
    rows, remapped = prepare_rows(raw_rows, build_adverse_event_pairs(Path(args.primekg_graph_dir)))
    if args.expected_rows and len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} modeling rows, found {len(rows)}")

    groups = np.asarray([row.get("claim_cluster_id") or row["claim_id"] for row in rows])
    stratify = np.asarray([row["actionability"] for row in rows])
    tasks = {"path_relevance": RELEVANCE_LABELS, "actionability": ACTION_LABELS}
    fold_metrics, oof_rows, leakage_rows = [], [], []

    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train_indices, test_indices) in enumerate(splitter.split(rows, stratify, groups), start=1):
            train_groups = set(groups[train_indices])
            test_groups = set(groups[test_indices])
            overlap = train_groups & test_groups
            leakage_rows.append({"seed": seed, "fold": fold, "leaking_group_count": len(overlap)})
            if overlap:
                raise RuntimeError(f"Group leakage for seed={seed}, fold={fold}")
            train_rows = [rows[index] for index in train_indices]
            test_rows = [rows[index] for index in test_indices]
            if len(set(row["actionability"] for row in test_rows)) != 2:
                raise RuntimeError(f"Actionability class missing for seed={seed}, fold={fold}")
            x_train, x_test, _, _ = make_matrices(train_rows, test_rows)
            for task, labels in tasks.items():
                y_train = [row[task] for row in train_rows]
                y_test = [row[task] for row in test_rows]
                majority = Counter(y_train).most_common(1)[0][0]
                for baseline_name, predictions in (
                    ("majority_baseline", [majority] * len(test_rows)),
                    ("rule_baseline", [rule_prediction(row, task) for row in test_rows]),
                ):
                    result = metrics(y_test, predictions, labels)
                    fold_metrics.append({"seed": seed, "fold": fold, "task": task, "model": baseline_name, **result})
                    for local_index, source_index in enumerate(test_indices):
                        oof_rows.append({
                            "annotation_id": rows[source_index]["annotation_id"],
                            "claim_id": rows[source_index]["claim_id"],
                            "claim_cluster_id": groups[source_index],
                            "seed": seed,
                            "fold": fold,
                            "task": task,
                            "model": baseline_name,
                            "gold_label": y_test[local_index],
                            "pred_label": predictions[local_index],
                            "probabilities": {},
                        })
                for model_name in ("logistic_regression", "random_forest"):
                    model = model_for(model_name, seed)
                    model.fit(x_train, y_train)
                    predictions = model.predict(x_test).tolist()
                    probabilities = model.predict_proba(x_test)
                    result = metrics(y_test, predictions, labels)
                    fold_metrics.append({"seed": seed, "fold": fold, "task": task, "model": model_name, **result})
                    for local_index, source_index in enumerate(test_indices):
                        oof_rows.append({
                            "annotation_id": rows[source_index]["annotation_id"],
                            "claim_id": rows[source_index]["claim_id"],
                            "claim_cluster_id": groups[source_index],
                            "seed": seed,
                            "fold": fold,
                            "task": task,
                            "model": model_name,
                            "gold_label": y_test[local_index],
                            "pred_label": predictions[local_index],
                            "probabilities": {label: round(float(value), 8) for label, value in zip(model.classes_, probabilities[local_index])},
                        })

    summaries = []
    for (task, model_name), items in sorted(defaultdict(list, {
        key: [row for row in fold_metrics if (row["task"], row["model"]) == key]
        for key in {(row["task"], row["model"]) for row in fold_metrics}
    }).items()):
        for metric_name in ("accuracy", "macro_f1"):
            values = np.asarray([float(row[metric_name]) for row in items])
            summaries.append({"task": task, "model": model_name, "metric": metric_name, "mean": values.mean(), "std": values.std(ddof=1)})

    confusion_rows = []
    for seed in SEEDS:
        for task, labels in tasks.items():
            for model_name in ("majority_baseline", "rule_baseline", "logistic_regression", "random_forest"):
                selected = [row for row in oof_rows if row["seed"] == seed and row["task"] == task and row["model"] == model_name]
                if not selected:
                    continue
                matrix = confusion_matrix([row["gold_label"] for row in selected], [row["pred_label"] for row in selected], labels=labels)
                for gold_index, gold_label in enumerate(labels):
                    for pred_index, pred_label in enumerate(labels):
                        confusion_rows.append({"seed": seed, "task": task, "model": model_name, "gold_label": gold_label, "pred_label": pred_label, "count": int(matrix[gold_index, pred_index])})

    final_seed = 20260618
    for task in tasks:
        for model_name in ("logistic_regression", "random_forest"):
            bundle = fit_final(rows, task, model_name, final_seed)
            joblib.dump(bundle, output_dir / f"{task}_{model_name}.joblib")

    write_csv(output_dir / "semantic_clean_annotation_pool.csv", rows)
    write_csv(output_dir / "cv_fold_metrics.csv", fold_metrics)
    write_csv(output_dir / "cv_summary.csv", summaries)
    write_csv(output_dir / "confusion_matrices.csv", confusion_rows)
    write_csv(output_dir / "group_leakage_checks.csv", leakage_rows)
    write_jsonl(output_dir / "oof_predictions.jsonl", oof_rows)
    manifest_path = Path(args.annotations_path).with_name("annotation_gold_manifest.json")
    annotation_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    stats = {
        "original_annotation_rows": int(annotation_manifest.get("a500_rows", len(raw_rows))),
        "input_rows": len(raw_rows),
        "modeling_rows": len(rows),
        "artifact_excluded_rows": int(annotation_manifest.get("artifact_excluded_rows", artifact_count)),
        "semantic_relation_remaps": remapped,
        "claim_cluster_count": len(set(groups)),
        "seeds": SEEDS,
        "folds_per_seed": args.folds,
        "label_counts": {
            "path_relevance": Counter(row["path_relevance"] for row in rows),
            "actionability": Counter(row["actionability"] for row in rows),
        },
        "group_leakage_count": sum(row["leaking_group_count"] for row in leakage_rows),
    }
    (output_dir / "path_scorer_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
