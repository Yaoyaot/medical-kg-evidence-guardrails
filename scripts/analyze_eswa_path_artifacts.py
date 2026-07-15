from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from hierarchical_evidence_features import combined_text, structured_features


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score(rows: list[dict], bundle: dict) -> tuple[list[str], list[dict]]:
    tfidf = bundle["tfidf"]
    vectorizer = bundle["dict_vectorizer"]
    model = bundle["model"]
    x = hstack([
        tfidf.transform([combined_text(row) for row in rows]),
        vectorizer.transform([structured_features(row) for row in rows]),
    ]).tocsr()
    predictions = model.predict(x).tolist()
    probabilities = model.predict_proba(x)
    return predictions, [
        {label: float(value) for label, value in zip(model.classes_, values)}
        for values in probabilities
    ]


def task_metrics(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    return {
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=labels, average="macro", zero_division=0),
        "per_label": {
            label: {"precision": float(p), "recall": float(r), "f1": float(v), "support": int(n)}
            for label, p, r, v, n in zip(labels, precision, recall, f1, support)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the 26 excluded graph-path artifacts with frozen Evidence Scorers.")
    parser.add_argument("--artifact-path", default="data/processed/stage6_eswa/annotations_gold/path_annotations_artifact_error_analysis.csv")
    parser.add_argument("--path-scorer-dir", default="data/processed/stage7_hierarchical_scorer/path_scorer")
    parser.add_argument("--semantic-clean-stats", default="data/processed/stage2_primekg_semantic_clean/primekg_graph/hetionet_graph_stats.json")
    parser.add_argument("--output-dir", default="data/processed/stage9_eswa_major_revision/path_artifacts")
    args = parser.parse_args()

    rows = read_csv(Path(args.artifact_path))
    if len(rows) != 26:
        raise ValueError(f"Expected 26 artifacts, found {len(rows)}")
    for row in rows:
        row["actionability"] = "ACTIONABLE" if row.get("evidence_role") in {"SUPPORT", "REFUTE"} else "NON_ACTIONABLE"
    path_dir = Path(args.path_scorer_dir)
    relevance_pred, relevance_prob = score(rows, joblib.load(path_dir / "path_relevance_logistic_regression.joblib"))
    actionable_pred, actionable_prob = score(rows, joblib.load(path_dir / "actionability_logistic_regression.joblib"))
    scored = []
    for row, rp, rprob, ap, aprob in zip(rows, relevance_pred, relevance_prob, actionable_pred, actionable_prob):
        scored.append({
            **row,
            "pred_path_relevance": rp,
            "path_relevance_probabilities": json.dumps(rprob, ensure_ascii=False),
            "pred_actionability": ap,
            "actionability_probabilities": json.dumps(aprob, ensure_ascii=False),
        })
    semantic = json.loads(Path(args.semantic_clean_stats).read_text(encoding="utf-8"))
    reasons = Counter()
    for row in rows:
        for reason in str(row.get("artifact_reasons", "")).split("|"):
            if reason:
                reasons[reason] += 1
    stats = {
        "artifact_rows": len(rows),
        "artifact_reason_counts": dict(reasons),
        "artifact_flag_recall_on_known_artifacts": sum(str(row.get("artifact_excluded", "")).lower() == "true" for row in rows) / len(rows),
        "artifact_flag_precision": None,
        "artifact_flag_precision_note": "Not estimable from an artifact-only audit set.",
        "path_relevance_on_artifacts": task_metrics([row["path_relevance"] for row in rows], relevance_pred, ["IRRELEVANT", "PARTIAL", "RELEVANT"]),
        "actionability_on_artifacts": task_metrics([row["actionability"] for row in rows], actionable_pred, ["ACTIONABLE", "NON_ACTIONABLE"]),
        "semantic_cleaning": {
            "reversed_drug_effect_edges_removed": semantic["action_counts"]["reversed_drug_effect_removed"],
            "drug_effect_edges_downgraded_to_context": semantic["action_counts"]["drug_effect_downgraded_to_context"],
            "edges_before": semantic["edges_before"],
            "edges_after": semantic["edges_after"],
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "artifact_path_predictions.csv", scored)
    (output / "artifact_sensitivity_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
