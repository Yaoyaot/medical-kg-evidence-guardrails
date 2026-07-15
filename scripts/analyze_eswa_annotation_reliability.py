from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


SEED = 20260618
FIELDS = {
    "path_relevance": ["IRRELEVANT", "PARTIAL", "RELEVANT"],
    "evidence_role": ["CONTEXT_ONLY", "REFUTE", "SUPPORT", "UNKNOWN"],
    "error_type": ["ENTITY_LINKING", "INDIRECT_BRIDGE", "INSUFFICIENT_SPECIFICITY", "NONE", "POLARITY", "RELATION_MISMATCH"],
}


def read_csv(path: Path) -> list[dict]:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Cannot decode {path}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(left: list[str], right: list[str], iterations: int) -> tuple[float, float]:
    rng = random.Random(SEED)
    values = []
    for _ in range(iterations):
        indices = [rng.randrange(len(left)) for _ in left]
        sample_left = [left[index] for index in indices]
        sample_right = [right[index] for index in indices]
        values.append(cohen_kappa_score(sample_left, sample_right))
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reliability statistics for the 200 overlapping path annotations.")
    parser.add_argument("--annotator-a", default="data/processed/stage6_eswa/annotations/path_annotations_annotator_a_overlap200_key_bcleaned_continue.csv")
    parser.add_argument("--annotator-b", default="data/processed/stage6_eswa/annotations/path_annotations_annotator_b_overlap200_cleaned_continue.csv")
    parser.add_argument("--output-dir", default="data/processed/stage9_eswa_major_revision/annotation_reliability")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    a = read_csv(Path(args.annotator_a))
    b = read_csv(Path(args.annotator_b))
    if len(a) != 200 or len(b) != 200:
        raise ValueError(f"Expected 200 rows per annotator, found A={len(a)}, B={len(b)}")
    key = lambda row: (row.get("claim_id", ""), row.get("path_text", ""))
    b_by_key = {key(row): row for row in b}
    pairs = [(row, b_by_key[key(row)]) for row in a if key(row) in b_by_key]
    if len(pairs) != 200:
        raise ValueError(f"Expected 200 aligned overlap rows, found {len(pairs)}")

    summary = []
    confusion_rows = []
    for field, labels in FIELDS.items():
        left = [row[field].strip().upper() for row, _ in pairs]
        right = [row[field].strip().upper() for _, row in pairs]
        agreement = float(np.mean([x == y for x, y in zip(left, right)]))
        kappa = cohen_kappa_score(left, right, labels=labels)
        ci_low, ci_high = bootstrap_ci(left, right, args.bootstrap_iterations)
        summary.append({
            "field": field,
            "rows": len(left),
            "raw_agreement": agreement,
            "cohens_kappa": kappa,
            "kappa_ci_low": ci_low,
            "kappa_ci_high": ci_high,
            "pabak_descriptive": 2 * agreement - 1,
            "annotator_a_distribution": json.dumps(Counter(left), ensure_ascii=False),
            "annotator_b_distribution": json.dumps(Counter(right), ensure_ascii=False),
        })
        matrix = confusion_matrix(left, right, labels=labels)
        for i, a_label in enumerate(labels):
            for j, b_label in enumerate(labels):
                confusion_rows.append({"field": field, "annotator_a_label": a_label, "annotator_b_label": b_label, "count": int(matrix[i, j])})

    output = Path(args.output_dir)
    write_csv(output / "annotation_reliability_summary.csv", summary)
    write_csv(output / "annotation_confusion_matrices.csv", confusion_rows)
    stats = {
        "overlap_rows": len(pairs),
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": SEED,
        "summary": summary,
        "pabak_note": "PABAK is reported only as a descriptive prevalence-adjusted sensitivity statistic for reviewer context.",
    }
    (output / "annotation_reliability_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
