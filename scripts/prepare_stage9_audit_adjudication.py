from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any


PUBMED_ENUMS = {
    "claim_faithfulness": {"VALID", "INVALID"},
    "atomicity": {"ATOMIC", "NON_ATOMIC"},
    "label_compatibility": {"COMPATIBLE", "INCOMPATIBLE", "AMBIGUOUS"},
    "pico_preservation": {"COMPLETE", "PARTIAL", "NOT_APPLICABLE"},
    "modality_strength": {"PRESERVED", "WEAKENED", "STRENGTHENED", "CHANGED"},
}

ENTITY_ENUMS = {
    "abbreviation_ambiguity": {"YES", "NO"},
    "overall_linking_judgment": {"CORRECT", "PARTIAL", "INCORRECT"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            text = raw.decode(encoding)
            return list(csv.DictReader(io.StringIO(text, newline="")))
        except UnicodeDecodeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_json_array(value: str, row_label: str, field: str, errors: list[str]) -> list[Any] | None:
    raw = clean(value)
    if not raw:
        errors.append(f"{row_label}: {field} is blank")
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(f"{row_label}: {field} is invalid JSON ({exc.msg})")
        return None
    if not isinstance(parsed, list):
        errors.append(f"{row_label}: {field} must be a JSON array")
        return None
    return parsed


def canonical_json(value: str) -> str | None:
    try:
        parsed = json.loads(clean(value))
    except (json.JSONDecodeError, TypeError):
        return None

    def normalized(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: normalized(item[key]) for key in sorted(item)}
        if isinstance(item, list):
            normalized_items = [normalized(part) for part in item]
            return sorted(
                normalized_items,
                key=lambda part: json.dumps(part, ensure_ascii=False, sort_keys=True),
            )
        if isinstance(item, str):
            return item.strip()
        return item

    return json.dumps(normalized(parsed), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def values_agree(field: str, left: str, right: str) -> bool:
    if field.endswith("_json"):
        left_json = canonical_json(left)
        right_json = canonical_json(right)
        return left_json is not None and left_json == right_json
    return clean(left).upper() == clean(right).upper()


def preserve_completed_adjudication(
    path: Path,
    rows: list[dict[str, Any]],
    final_fields: list[str],
) -> None:
    """Carry forward human-entered final labels when regenerating a merged table."""
    if not path.exists():
        return
    existing = {clean(row.get("id")): row for row in read_csv(path)}
    for row in rows:
        previous = existing.get(clean(row.get("id")))
        if previous is None:
            continue
        for field in final_fields:
            value = clean(previous.get(field))
            if value:
                row[field] = value
        notes = clean(previous.get("adjudication_notes"))
        if notes:
            row["adjudication_notes"] = previous.get("adjudication_notes", "")


def index_and_validate_alignment(
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    expected_count: int,
    static_fields: list[str],
    audit_name: str,
    errors: list[str],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    if len(rows_a) != expected_count:
        errors.append(f"{audit_name}: annotator A has {len(rows_a)} rows; expected {expected_count}")
    if len(rows_b) != expected_count:
        errors.append(f"{audit_name}: annotator B has {len(rows_b)} rows; expected {expected_count}")

    ids_a = [clean(row.get("id")) for row in rows_a]
    ids_b = [clean(row.get("id")) for row in rows_b]
    for annotator, ids in (("A", ids_a), ("B", ids_b)):
        duplicates = [item for item, count in Counter(ids).items() if item and count > 1]
        if duplicates:
            errors.append(f"{audit_name}: annotator {annotator} duplicate ids: {duplicates[:10]}")
        if any(not item for item in ids):
            errors.append(f"{audit_name}: annotator {annotator} contains blank id")

    by_id_b = {clean(row.get("id")): row for row in rows_b}
    missing_in_b = sorted(set(ids_a) - set(ids_b))
    extra_in_b = sorted(set(ids_b) - set(ids_a))
    if missing_in_b:
        errors.append(f"{audit_name}: ids missing from annotator B: {missing_in_b[:10]}")
    if extra_in_b:
        errors.append(f"{audit_name}: unexpected ids in annotator B: {extra_in_b[:10]}")

    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for row_a in rows_a:
        row_id = clean(row_a.get("id"))
        row_b = by_id_b.get(row_id)
        if row_b is None:
            continue
        for field in static_fields:
            if clean(row_a.get(field)) != clean(row_b.get(field)):
                errors.append(f"{audit_name} {row_id}: immutable field differs: {field}")
        pairs.append((row_a, row_b))
    return pairs


def validate_enum(
    row: dict[str, str],
    fields: dict[str, set[str]],
    row_label: str,
    errors: list[str],
) -> None:
    for field, allowed in fields.items():
        value = clean(row.get(field)).upper()
        if value not in allowed:
            errors.append(f"{row_label}: {field}={value!r}; expected one of {sorted(allowed)}")


def prepare_pubmedqa(input_dir: Path, errors: list[str]) -> dict[str, Any]:
    error_count_before = len(errors)
    path_a = input_dir / "pubmedqa_label_mapping_audit60_annotator_a.csv"
    path_b = input_dir / "pubmedqa_label_mapping_audit60_annotator_b.csv"
    rows_a, rows_b = read_csv(path_a), read_csv(path_b)
    static = [
        "audit_order", "id", "pubid", "raw_label", "mapped_label", "question",
        "converted_claim", "source_context",
    ]
    pairs = index_and_validate_alignment(rows_a, rows_b, 60, static, "pubmedqa", errors)
    annotation_fields = list(PUBMED_ENUMS)
    merged: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    field_disagreements: Counter[str] = Counter()

    for row_a, row_b in pairs:
        row_id = clean(row_a.get("id"))
        validate_enum(row_a, PUBMED_ENUMS, f"pubmedqa A {row_id}", errors)
        validate_enum(row_b, PUBMED_ENUMS, f"pubmedqa B {row_id}", errors)
        out: dict[str, Any] = {field: row_a.get(field, "") for field in static}
        disagreements: list[str] = []
        for field in annotation_fields:
            a_value = clean(row_a.get(field)).upper()
            b_value = clean(row_b.get(field)).upper()
            out[f"annotator_a_{field}"] = a_value
            out[f"annotator_b_{field}"] = b_value
            if values_agree(field, a_value, b_value):
                out[f"final_{field}"] = a_value
            else:
                out[f"final_{field}"] = ""
                disagreements.append(field)
                field_disagreements[field] += 1
        out["annotator_a_notes"] = row_a.get("notes", "")
        out["annotator_b_notes"] = row_b.get("notes", "")
        out["disagreement_fields"] = ";".join(disagreements)
        out["adjudication_notes"] = ""
        merged.append(out)
        if disagreements:
            disagreement_rows.append(out)

    fieldnames = static.copy()
    for field in annotation_fields:
        fieldnames.extend([f"annotator_a_{field}", f"annotator_b_{field}", f"final_{field}"])
    fieldnames.extend(["annotator_a_notes", "annotator_b_notes", "disagreement_fields", "adjudication_notes"])
    prepared = len(errors) == error_count_before
    if prepared:
        adjudication_path = input_dir / "pubmedqa_label_mapping_audit60_adjudication.csv"
        preserve_completed_adjudication(
            adjudication_path,
            merged,
            [f"final_{field}" for field in annotation_fields],
        )
        write_csv(adjudication_path, merged, fieldnames)
        write_csv(input_dir / "pubmedqa_label_mapping_audit60_disagreements.csv", disagreement_rows, fieldnames)
    return {
        "adjudication_prepared": prepared,
        "rows": len(merged),
        "rows_with_any_disagreement": len(disagreement_rows),
        "agreements_on_all_fields": len(merged) - len(disagreement_rows),
        "field_disagreements": dict(field_disagreements),
    }


def validate_entity_json(
    row: dict[str, str], annotator: str, row_id: str, errors: list[str]
) -> None:
    prefix = f"entity {annotator} {row_id}"
    json_fields = [
        "gold_biomedical_mentions_json", "gold_concept_links_json",
        "missed_mentions_json", "incorrect_predicted_links_json",
    ]
    parsed: dict[str, list[Any] | None] = {}
    for field in json_fields:
        parsed[field] = parse_json_array(row.get(field, ""), prefix, field, errors)

    concepts = parsed.get("gold_concept_links_json")
    if concepts is not None:
        for index, item in enumerate(concepts):
            if not isinstance(item, dict):
                errors.append(f"{prefix}: gold_concept_links_json[{index}] must be an object")
                continue
            missing = [key for key in ("mention", "concept_id", "entity_type") if not clean(item.get(key))]
            if missing:
                errors.append(f"{prefix}: gold_concept_links_json[{index}] missing {missing}")


def prepare_entity(input_dir: Path, errors: list[str]) -> dict[str, Any]:
    error_count_before = len(errors)
    path_a = input_dir / "entity_linking_audit120_annotator_a.csv"
    path_b = input_dir / "entity_linking_audit120_annotator_b.csv"
    rows_a, rows_b = read_csv(path_a), read_csv(path_b)
    static = [
        "audit_order", "id", "dataset", "claim", "gold_label",
        "predicted_link_count", "predicted_links_json",
    ]
    pairs = index_and_validate_alignment(rows_a, rows_b, 120, static, "entity", errors)
    annotation_fields = [
        "gold_biomedical_mentions_json", "gold_concept_links_json", "missed_mentions_json",
        "incorrect_predicted_links_json", "abbreviation_ambiguity", "overall_linking_judgment",
    ]
    merged: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    field_disagreements: Counter[str] = Counter()

    for row_a, row_b in pairs:
        row_id = clean(row_a.get("id"))
        validate_enum(row_a, ENTITY_ENUMS, f"entity A {row_id}", errors)
        validate_enum(row_b, ENTITY_ENUMS, f"entity B {row_id}", errors)
        validate_entity_json(row_a, "A", row_id, errors)
        validate_entity_json(row_b, "B", row_id, errors)
        out: dict[str, Any] = {field: row_a.get(field, "") for field in static}
        disagreements: list[str] = []
        for field in annotation_fields:
            a_value = clean(row_a.get(field))
            b_value = clean(row_b.get(field))
            if field in ENTITY_ENUMS:
                a_value, b_value = a_value.upper(), b_value.upper()
            out[f"annotator_a_{field}"] = a_value
            out[f"annotator_b_{field}"] = b_value
            if values_agree(field, a_value, b_value):
                out[f"final_{field}"] = a_value
            else:
                out[f"final_{field}"] = ""
                disagreements.append(field)
                field_disagreements[field] += 1
        out["annotator_a_notes"] = row_a.get("notes", "")
        out["annotator_b_notes"] = row_b.get("notes", "")
        out["disagreement_fields"] = ";".join(disagreements)
        out["adjudication_notes"] = ""
        merged.append(out)
        if disagreements:
            disagreement_rows.append(out)

    fieldnames = static.copy()
    for field in annotation_fields:
        fieldnames.extend([f"annotator_a_{field}", f"annotator_b_{field}", f"final_{field}"])
    fieldnames.extend(["annotator_a_notes", "annotator_b_notes", "disagreement_fields", "adjudication_notes"])
    prepared = len(errors) == error_count_before
    if prepared:
        adjudication_path = input_dir / "entity_linking_audit120_adjudication.csv"
        preserve_completed_adjudication(
            adjudication_path,
            merged,
            [f"final_{field}" for field in annotation_fields],
        )
        write_csv(adjudication_path, merged, fieldnames)
        write_csv(input_dir / "entity_linking_audit120_disagreements.csv", disagreement_rows, fieldnames)
    return {
        "adjudication_prepared": prepared,
        "rows": len(merged),
        "rows_with_any_disagreement": len(disagreement_rows),
        "agreements_on_all_fields": len(merged) - len(disagreement_rows),
        "field_disagreements": dict(field_disagreements),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage 9 dual annotations and prepare adjudication tables.")
    parser.add_argument(
        "--audit-dir",
        default="data/processed/stage9_eswa_major_revision/human_audits",
    )
    args = parser.parse_args()
    audit_dir = Path(args.audit_dir).resolve()
    errors: list[str] = []
    report = {
        "audit_dir": str(audit_dir),
        "pubmedqa": prepare_pubmedqa(audit_dir, errors),
        "entity_linking": prepare_entity(audit_dir, errors),
        "validation_errors": errors,
        "validation_passed": not errors,
    }
    report_path = audit_dir / "stage9_annotation_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
