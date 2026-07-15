from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ENUMS = {
    "path_relevance": {"RELEVANT", "PARTIAL", "IRRELEVANT"},
    "evidence_role": {"SUPPORT", "REFUTE", "CONTEXT_ONLY", "UNKNOWN"},
    "error_type": {
        "NONE",
        "ENTITY_LINKING",
        "RELATION_MISMATCH",
        "INDIRECT_BRIDGE",
        "POLARITY",
        "INSUFFICIENT_SPECIFICITY",
        "OTHER",
    },
}
EDITABLE_FIELDS = {*ENUMS, "notes"}
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")


def read_csv(path: Path) -> list[dict]:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path} with supported encodings: {CSV_ENCODINGS}") from last_error


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def normalized(value: str) -> str:
    return str(value or "").strip().upper()


def index_rows(rows: list[dict], label: str) -> dict[str, dict]:
    ids = [row.get("annotation_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{label} contains duplicate annotation_id values.")
    return {item: row for item, row in zip(ids, rows)}


def validate(rows: list[dict], label: str, expected_rows: int) -> list[dict]:
    if len(rows) != expected_rows:
        raise RuntimeError(f"{label}: expected {expected_rows} rows, found {len(rows)}.")
    output = []
    for index, row in enumerate(rows, start=2):
        item = dict(row)
        for field, allowed in ENUMS.items():
            item[field] = normalized(item.get(field, ""))
            if not item[field]:
                raise RuntimeError(f"{label} row {index}: {field} is empty.")
            if item[field] not in allowed:
                raise RuntimeError(f"{label} row {index}: invalid {field}={item[field]!r}.")
        output.append(item)
    return output


def kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    if len(a) != len(b) or not a:
        raise RuntimeError("Cannot calculate agreement for empty or misaligned labels.")
    observed = sum(left == right for left, right in zip(a, b)) / len(a)
    labels = sorted(set(a) | set(b))
    left = Counter(a)
    right = Counter(b)
    expected = sum((left[label] / len(a)) * (right[label] / len(b)) for label in labels)
    score = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return round(observed, 6), round(score, 6)


def resolve_input_path(path_value: str | None, output_dir: Path, default_name: str) -> Path:
    if not path_value:
        return output_dir / default_name
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def prefixed(output_dir: Path, stem: str, prefix: str, suffix: str) -> Path:
    return output_dir / f"{stem}_{prefix}{suffix}" if prefix else output_dir / f"{stem}{suffix}"


def write_summary(path: Path, result: dict) -> None:
    lines = [
        "# Annotation Agreement Summary",
        "",
        f"- Annotated overlap rows: {result['annotated_rows']}",
        f"- Row-level agreement: {result['row_level_agreement']:.4f}",
        f"- Disagreement rows: {result['disagreement_rows']}",
        "",
        "| Field | Raw agreement | Cohen's kappa | Annotator A distribution | Annotator B distribution |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for field, stats in result["field_agreement"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    field,
                    f"{stats['agreement']:.4f}",
                    f"{stats['cohens_kappa']:.4f}",
                    json.dumps(stats["annotator_a_distribution"], ensure_ascii=False),
                    json.dumps(stats["annotator_b_distribution"], ensure_ascii=False),
                ]
            )
            + " |"
        )
    if result["adjudication_required"]:
        lines.extend(
            [
                "",
                "Disagreements remain and should be reviewed before using adjudicated labels.",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate independent strict-path annotation agreement.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--annotator-a-path")
    parser.add_argument("--annotator-b-path")
    parser.add_argument("--expected-rows", type=int, default=200)
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()

    output_dir = Path(args.data_dir) / "processed/submission_strengthening/annotations"
    a_path = resolve_input_path(args.annotator_a_path, output_dir, "path_annotations_annotator_a.csv")
    b_path = resolve_input_path(args.annotator_b_path, output_dir, "path_annotations_annotator_b.csv")
    a_rows = validate(read_csv(a_path), "annotator_a", args.expected_rows)
    b_rows = validate(read_csv(b_path), "annotator_b", args.expected_rows)
    a = index_rows(a_rows, "annotator_a")
    b = index_rows(b_rows, "annotator_b")
    if set(a) != set(b):
        raise RuntimeError("Annotator sheets have different annotation_id sets.")

    ids = sorted(a)
    for item in ids:
        immutable_fields = (set(a[item]) | set(b[item])) - EDITABLE_FIELDS
        for field in immutable_fields:
            if a[item].get(field, "") != b[item].get(field, ""):
                raise RuntimeError(f"Annotator sheets differ in read-only field {field!r} for {item}.")
    agreement = {}
    for field in ENUMS:
        observed, score = kappa([a[item][field] for item in ids], [b[item][field] for item in ids])
        agreement[field] = {
            "agreement": observed,
            "cohens_kappa": score,
            "annotator_a_distribution": dict(Counter(a[item][field] for item in ids)),
            "annotator_b_distribution": dict(Counter(b[item][field] for item in ids)),
        }

    disagreements = []
    adjudication = []
    for item in ids:
        left = a[item]
        right = b[item]
        differs = any(left[field] != right[field] for field in ENUMS)
        base = {key: left.get(key, "") for key in left if key not in {*ENUMS, "notes"}}
        row = {
            **base,
            **{f"annotator_a_{field}": left[field] for field in ENUMS},
            **{f"annotator_b_{field}": right[field] for field in ENUMS},
            "annotator_a_notes": left.get("notes", ""),
            "annotator_b_notes": right.get("notes", ""),
            "final_path_relevance": left["path_relevance"] if not differs else "",
            "final_evidence_role": left["evidence_role"] if not differs else "",
            "final_error_type": left["error_type"] if not differs else "",
            "adjudication_notes": "",
        }
        adjudication.append(row)
        if differs:
            disagreements.append(row)

    output_prefix = args.output_prefix.strip()
    write_csv(prefixed(output_dir, "annotation_disagreements", output_prefix, ".csv"), disagreements, list(adjudication[0]))
    write_csv(prefixed(output_dir, "path_annotations_adjudication", output_prefix, ".csv"), adjudication)
    result = {
        "annotated_rows": len(ids),
        "annotator_a_path": str(a_path),
        "annotator_b_path": str(b_path),
        "disagreement_rows": len(disagreements),
        "row_level_agreement": round(1 - len(disagreements) / len(ids), 6),
        "field_agreement": agreement,
        "adjudication_required": bool(disagreements),
    }
    agreement_path = prefixed(output_dir, "annotation_agreement", output_prefix, ".json")
    agreement_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_name = f"annotation_agreement_{output_prefix}_summary.md" if output_prefix else "annotation_agreement_summary.md"
    write_summary(Path("outputs/paper_assets") / summary_name, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
