from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_eswa_major_revision import PRIMARY_MODEL, features


SEED = 20260618
C_VALUE = 0.1


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


def fit(train: list[dict]) -> tuple[DictVectorizer, LogisticRegression]:
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform([features(row, PRIMARY_MODEL) for row in train])
    y = np.asarray([int(row["gold_label"] != "SUPPORT") for row in train])
    model = LogisticRegression(C=C_VALUE, penalty="l2", class_weight="balanced", max_iter=5000, random_state=SEED)
    model.fit(x, y)
    return vectorizer, model


def predict(vectorizer: DictVectorizer, model: LogisticRegression, rows: list[dict]) -> np.ndarray:
    x = vectorizer.transform([features(row, PRIMARY_MODEL) for row in rows])
    return model.predict_proba(x)[:, list(model.classes_).index(1)]


def ranking_metrics(test_all: list[dict], test_support: list[dict], scores: np.ndarray) -> dict:
    y = np.asarray([int(row["gold_label"] != "SUPPORT") for row in test_support])
    budget = min(len(test_support), math.ceil(0.05 * len(test_all)))
    selected = np.argsort(-scores)[:budget]
    detected = int(y[selected].sum())
    return {
        "test_rows": len(test_all),
        "test_support_predictions": len(test_support),
        "test_error_events": int(y.sum()),
        "auroc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
        "reviewed_count": budget,
        "detected_false_supports": detected,
        "error_detection_precision": detected / budget if budget else 0.0,
        "error_detection_recall": detected / int(y.sum()) if y.sum() else 0.0,
    }


def repeated_group_splits(rows: list[dict], repeats: int) -> list[dict]:
    groups = sorted({row["pair_group_id"] for row in rows})
    output = []
    for repeat in range(repeats):
        rng = random.Random(f"{SEED}:split:{repeat}")
        shuffled = list(groups)
        rng.shuffle(shuffled)
        train_groups = set(shuffled[: round(0.70 * len(shuffled))])
        train = [row for row in rows if row["pair_group_id"] in train_groups and row["candidate_label"] == "SUPPORT"]
        test_all = [row for row in rows if row["pair_group_id"] not in train_groups]
        test = [row for row in test_all if row["candidate_label"] == "SUPPORT"]
        y_train = {row["gold_label"] != "SUPPORT" for row in train}
        y_test = {row["gold_label"] != "SUPPORT" for row in test}
        if len(y_train) < 2 or len(y_test) < 2:
            continue
        vectorizer, model = fit(train)
        output.append({"repeat": repeat, "training_support_predictions": len(train), **ranking_metrics(test_all, test, predict(vectorizer, model, test))})
    return output


def leave_one_dataset_out(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        train = [row for row in rows if row["dataset"] != dataset and row["candidate_label"] == "SUPPORT"]
        test_all = [row for row in rows if row["dataset"] == dataset]
        test = [row for row in test_all if row["candidate_label"] == "SUPPORT"]
        if len({row["gold_label"] != "SUPPORT" for row in train}) < 2 or len({row["gold_label"] != "SUPPORT" for row in test}) < 2:
            output.append({"held_out_dataset": dataset, "status": "insufficient_class_variation", "training_support_predictions": len(train), "test_support_predictions": len(test)})
            continue
        vectorizer, model = fit(train)
        output.append({"held_out_dataset": dataset, "status": "ok", "training_support_predictions": len(train), **ranking_metrics(test_all, test, predict(vectorizer, model, test))})
    return output


def learning_curve(rows: list[dict], repeats: int) -> list[dict]:
    groups = sorted({row["pair_group_id"] for row in rows})
    output = []
    for target in (20, 40, 60, 80, 100):
        for repeat in range(repeats):
            rng = random.Random(f"{SEED}:learning:{target}:{repeat}")
            shuffled = list(groups)
            rng.shuffle(shuffled)
            train_groups = set()
            train_support_count = 0
            for group in shuffled:
                group_support = sum(row["candidate_label"] == "SUPPORT" and row["pair_group_id"] == group for row in rows)
                if train_support_count >= target:
                    break
                train_groups.add(group)
                train_support_count += group_support
            train = [row for row in rows if row["pair_group_id"] in train_groups and row["candidate_label"] == "SUPPORT"]
            test_all = [row for row in rows if row["pair_group_id"] not in train_groups]
            test = [row for row in test_all if row["candidate_label"] == "SUPPORT"]
            if len({row["gold_label"] != "SUPPORT" for row in train}) < 2 or len({row["gold_label"] != "SUPPORT" for row in test}) < 2:
                continue
            vectorizer, model = fit(train)
            output.append({"target_training_support_predictions": target, "repeat": repeat, "actual_training_support_predictions": len(train), **ranking_metrics(test_all, test, predict(vectorizer, model, test))})
    return output


def bootstrap_ranking_stability(rows: list[dict], bootstraps: int) -> tuple[list[dict], dict]:
    support = [row for row in rows if row["candidate_label"] == "SUPPORT"]
    y = np.asarray([int(row["gold_label"] != "SUPPORT") for row in support])
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform([features(row, PRIMARY_MODEL) for row in support])
    coefficient_rows = []
    top_sets = []
    for bootstrap in range(bootstraps):
        rng = np.random.default_rng(SEED + bootstrap)
        indices = rng.choice(len(support), size=len(support), replace=True)
        if len(set(y[indices])) < 2:
            continue
        model = LogisticRegression(C=C_VALUE, penalty="l2", class_weight="balanced", max_iter=5000, random_state=SEED + bootstrap)
        model.fit(x[indices], y[indices])
        scores = model.predict_proba(x)[:, list(model.classes_).index(1)]
        top_sets.append({support[index]["id"] for index in np.argsort(-scores)[:30]})
        for feature, value in zip(vectorizer.get_feature_names_out(), model.coef_[0]):
            coefficient_rows.append({"bootstrap": bootstrap, "feature": feature, "coefficient": float(value)})
    jaccards = [len(left & right) / len(left | right) for left, right in itertools.combinations(top_sets, 2)]
    summary = {
        "bootstrap_models": len(top_sets),
        "top_b": 30,
        "pairwise_jaccard_mean": float(np.mean(jaccards)),
        "pairwise_jaccard_std": float(np.std(jaccards, ddof=1)),
        "pairwise_jaccard_min": float(np.min(jaccards)),
        "pairwise_jaccard_max": float(np.max(jaccards)),
    }
    return coefficient_rows, summary


def summarize(rows: list[dict], group_field: str) -> list[dict]:
    output = []
    metrics = ("auroc", "average_precision", "error_detection_precision", "error_detection_recall")
    for group in sorted({row[group_field] for row in rows}):
        subset = [row for row in rows if row[group_field] == group]
        item = {group_field: group, "valid_repeats": len(subset)}
        for metric in metrics:
            values = [float(row[metric]) for row in subset]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze grouped-split, leave-one-source-out, learning-curve, coefficient, and ranking stability.")
    parser.add_argument("--scores-path", default="data/processed/stage9_eswa_major_revision/formal600_crossfit/formal600_crossfit_risk_scores.jsonl")
    parser.add_argument("--output-dir", default="data/processed/stage9_eswa_major_revision/risk_stability")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--bootstrap-models", type=int, default=50)
    args = parser.parse_args()
    rows = read_jsonl(Path(args.scores_path))
    repeated = repeated_group_splits(rows, args.repeats)
    loso = leave_one_dataset_out(rows)
    learning = learning_curve(rows, args.repeats)
    coefficients, ranking_stability = bootstrap_ranking_stability(rows, args.bootstrap_models)
    output = Path(args.output_dir)
    write_csv(output / "repeated_group_splits.csv", repeated)
    write_csv(output / "leave_one_dataset_out.csv", loso)
    write_csv(output / "formal600_learning_curve.csv", learning)
    write_csv(output / "formal600_learning_curve_summary.csv", summarize(learning, "target_training_support_predictions"))
    write_csv(output / "bootstrap_coefficient_distribution.csv", coefficients)
    stats = {
        "repeated_group_splits": {"requested": args.repeats, "valid": len(repeated)},
        "leave_one_dataset_out": loso,
        "learning_curve_repeats": args.repeats,
        "ranking_stability": ranking_stability,
        "model": PRIMARY_MODEL,
        "C": C_VALUE,
        "seed": SEED,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "risk_stability_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
