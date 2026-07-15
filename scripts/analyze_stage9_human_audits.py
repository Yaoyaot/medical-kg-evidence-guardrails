from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "data/processed/stage9_eswa_major_revision/human_audits"
OUT_DIR = ROOT / "data/processed/stage9_eswa_major_revision/human_audit_results"
STAGE8 = ROOT / "data/processed/stage8_pubmedqa_external"
SEED = 20260618
BOOTSTRAPS = 5000
LABELS = ("SUPPORT", "REFUTE", "UNCERTAIN")

PUBMED_FIELDS = {
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
AUDIT_FILES = [
    "pubmedqa_label_mapping_audit60_annotator_a.csv",
    "pubmedqa_label_mapping_audit60_annotator_b.csv",
    "pubmedqa_label_mapping_audit60_adjudication.csv",
    "entity_linking_audit120_annotator_a.csv",
    "entity_linking_audit120_annotator_b.csv",
    "entity_linking_audit120_adjudication.csv",
]


def decode_text(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode {path}")


def read_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    text, encoding = decode_text(path)
    return list(csv.DictReader(io.StringIO(text, newline=""))), encoding


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    text, _ = decode_text(path)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_array(value: str, label: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(parsed, list):
        raise RuntimeError(f"{label}: expected a JSON array")
    return parsed


def canonical_json(value: str) -> str:
    parsed = json.loads(value)

    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: normalize(item[key]) for key in sorted(item)}
        if isinstance(item, list):
            normalized = [normalize(part) for part in item]
            return sorted(normalized, key=lambda part: json.dumps(part, ensure_ascii=False, sort_keys=True))
        if isinstance(item, str):
            return item.strip()
        return item

    return json.dumps(normalize(parsed), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def norm_mention(value: Any) -> str:
    text = str(value or "").casefold().replace("‐", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def percentile(values: list[float], probability: float) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    position = (len(finite) - 1) * probability
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return finite[low]
    fraction = position - low
    return finite[low] * (1 - fraction) + finite[high] * fraction


def bootstrap_ci(
    rows: list[Any],
    metric: Callable[[list[Any]], float | None],
    seed_offset: int = 0,
    iterations: int = BOOTSTRAPS,
) -> tuple[float | None, float | None, int]:
    rng = random.Random(SEED + seed_offset)
    values: list[float] = []
    for _ in range(iterations):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        value = metric(sample)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return percentile(values, 0.025), percentile(values, 0.975), len(values)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return centre - margin, centre + margin


def cohen_kappa(left: list[str], right: list[str]) -> tuple[float, float | None, str]:
    if len(left) != len(right) or not left:
        raise RuntimeError("Kappa inputs must be non-empty and aligned")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    labels = sorted(set(left) | set(right))
    expected = sum((left.count(label) / len(left)) * (right.count(label) / len(right)) for label in labels)
    if math.isclose(expected, 1.0):
        return observed, None, "not_estimable_zero_marginal_variance"
    return observed, (observed - expected) / (1 - expected), "estimable"


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_f1(gold: list[str], pred: list[str]) -> float:
    values = []
    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        values.append(f1_from_counts(tp, fp, fn)[2])
    return sum(values) / len(values)


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [row["gold_label"] for row in rows]
    pred = [row["pred_label"] for row in rows]
    correct = sum(g == p for g, p in zip(gold, pred))
    support_tp = sum(g == "SUPPORT" and p == "SUPPORT" for g, p in zip(gold, pred))
    support_fp = sum(g != "SUPPORT" and p == "SUPPORT" for g, p in zip(gold, pred))
    support_fn = sum(g == "SUPPORT" and p != "SUPPORT" for g, p in zip(gold, pred))
    support_precision, support_recall, support_f1 = f1_from_counts(support_tp, support_fp, support_fn)
    return {
        "count": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1": macro_f1(gold, pred) if rows else 0.0,
        "support_predictions": support_tp + support_fp,
        "support_precision": support_precision,
        "support_recall": support_recall,
        "support_f1": support_f1,
        "false_support_count": support_fp,
        "false_support_rate": support_fp / (support_tp + support_fp) if support_tp + support_fp else 0.0,
        "reviewed_count": sum(bool(row.get("reviewed")) for row in rows),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_alignment(a: list[dict[str, str]], b: list[dict[str, str]], final: list[dict[str, str]], expected: int, name: str) -> None:
    if not (len(a) == len(b) == len(final) == expected):
        raise RuntimeError(f"{name}: expected {expected} aligned rows, found A={len(a)}, B={len(b)}, final={len(final)}")
    ids = [{row.get("id", "").strip() for row in rows} for rows in (a, b, final)]
    if len(ids[0]) != expected or ids[0] != ids[1] or ids[0] != ids[2]:
        raise RuntimeError(f"{name}: duplicate, blank, or misaligned IDs")


def validate_pubmed(a: list[dict[str, str]], b: list[dict[str, str]], final: list[dict[str, str]]) -> None:
    validate_alignment(a, b, final, 60, "PubMedQA audit")
    by_a = {row["id"]: row for row in a}
    by_b = {row["id"]: row for row in b}
    for row in final:
        row_id = row["id"]
        disagreements = []
        for field, allowed in PUBMED_FIELDS.items():
            av = row[f"annotator_a_{field}"].strip().upper()
            bv = row[f"annotator_b_{field}"].strip().upper()
            fv = row[f"final_{field}"].strip().upper()
            if av not in allowed or bv not in allowed or fv not in allowed:
                raise RuntimeError(f"{row_id}: invalid {field} value")
            if av != by_a[row_id][field].strip().upper() or bv != by_b[row_id][field].strip().upper():
                raise RuntimeError(f"{row_id}: adjudication does not reproduce A/B {field}")
            if av == bv and fv != av:
                raise RuntimeError(f"{row_id}: agreed {field} was changed during adjudication")
            if av != bv:
                disagreements.append(field)
        declared = [part for part in row.get("disagreement_fields", "").split(";") if part]
        if set(disagreements) != set(declared):
            raise RuntimeError(f"{row_id}: disagreement_fields mismatch")
        if disagreements and not row.get("adjudication_notes", "").strip():
            raise RuntimeError(f"{row_id}: disagreement lacks adjudication notes")


def validate_entity(a: list[dict[str, str]], b: list[dict[str, str]], final: list[dict[str, str]]) -> None:
    validate_alignment(a, b, final, 120, "Entity-linking audit")
    by_a = {row["id"]: row for row in a}
    by_b = {row["id"]: row for row in b}
    json_fields = [
        "gold_biomedical_mentions_json", "gold_concept_links_json", "missed_mentions_json",
        "incorrect_predicted_links_json",
    ]
    for row in final:
        row_id = row["id"]
        parse_array(row["predicted_links_json"], f"{row_id}.predicted_links_json")
        disagreements = []
        for field in json_fields:
            for prefix, source in (("annotator_a_", by_a[row_id]), ("annotator_b_", by_b[row_id])):
                value = row[prefix + field]
                parse_array(value, f"{row_id}.{prefix}{field}")
                if canonical_json(value) != canonical_json(source[field]):
                    raise RuntimeError(f"{row_id}: adjudication does not reproduce {prefix}{field}")
            parse_array(row["final_" + field], f"{row_id}.final_{field}")
            a_value = canonical_json(row["annotator_a_" + field])
            b_value = canonical_json(row["annotator_b_" + field])
            final_value = canonical_json(row["final_" + field])
            if a_value != b_value:
                disagreements.append(field)
            elif final_value != a_value:
                raise RuntimeError(f"{row_id}: agreed {field} was changed during adjudication")
        for field, allowed in ENTITY_ENUMS.items():
            av = row[f"annotator_a_{field}"].strip().upper()
            bv = row[f"annotator_b_{field}"].strip().upper()
            fv = row[f"final_{field}"].strip().upper()
            if av not in allowed or bv not in allowed or fv not in allowed:
                raise RuntimeError(f"{row_id}: invalid {field} value")
            if av != by_a[row_id][field].strip().upper() or bv != by_b[row_id][field].strip().upper():
                raise RuntimeError(f"{row_id}: adjudication does not reproduce A/B {field}")
            if av != bv:
                disagreements.append(field)
            elif fv != av:
                raise RuntimeError(f"{row_id}: agreed {field} was changed during adjudication")
        declared = [part for part in row.get("disagreement_fields", "").split(";") if part]
        if set(disagreements) != set(declared):
            raise RuntimeError(f"{row_id}: disagreement_fields mismatch")
        if disagreements and not row.get("adjudication_notes", "").strip():
            raise RuntimeError(f"{row_id}: disagreement lacks adjudication notes")


def pubmed_analysis(final: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agreement_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for field_index, (field, allowed) in enumerate(PUBMED_FIELDS.items()):
        left = [row[f"annotator_a_{field}"].strip().upper() for row in final]
        right = [row[f"annotator_b_{field}"].strip().upper() for row in final]
        observed, kappa, status = cohen_kappa(left, right)
        agreement_ci = bootstrap_ci(final, lambda sample, f=field: sum(row[f"annotator_a_{f}"] == row[f"annotator_b_{f}"] for row in sample) / len(sample), 100 + field_index)

        def kappa_metric(sample: list[dict[str, str]], f: str = field) -> float | None:
            return cohen_kappa([row[f"annotator_a_{f}"] for row in sample], [row[f"annotator_b_{f}"] for row in sample])[1]

        kappa_ci = bootstrap_ci(final, kappa_metric, 200 + field_index)
        agreement_rows.append({
            "field": field,
            "n": len(final),
            "agreements": sum(a == b for a, b in zip(left, right)),
            "raw_agreement": observed,
            "agreement_ci_low": agreement_ci[0],
            "agreement_ci_high": agreement_ci[1],
            "cohens_kappa": kappa,
            "kappa_status": status,
            "kappa_ci_low": kappa_ci[0],
            "kappa_ci_high": kappa_ci[1],
            "estimable_bootstrap_kappa_samples": kappa_ci[2],
            "annotator_a_distribution": json.dumps(Counter(left), sort_keys=True),
            "annotator_b_distribution": json.dumps(Counter(right), sort_keys=True),
        })
        final_values = [row[f"final_{field}"].strip().upper() for row in final]
        counts = Counter(final_values)
        for value in sorted(allowed):
            low, high = wilson_interval(counts[value], len(final))
            distribution_rows.append({
                "field": field,
                "value": value,
                "count": counts[value],
                "n": len(final),
                "rate": counts[value] / len(final),
                "ci_low": low,
                "ci_high": high,
            })
    return agreement_rows, distribution_rows


def normalized_mention_set(value: str) -> set[str]:
    return {norm_mention(item) for item in parse_array(value, "mention set") if norm_mention(item)}


def normalized_concept_set(value: str) -> set[tuple[str, str, str]]:
    output = set()
    for item in parse_array(value, "concept set"):
        if not isinstance(item, dict):
            raise RuntimeError("Concept entry must be an object")
        output.add((norm_mention(item.get("mention")), str(item.get("concept_id", "")).strip(), str(item.get("entity_type", "")).strip()))
    return output


def set_agreement_stats(rows: list[dict[str, str]], field: str, parser: Callable[[str], set[Any]]) -> tuple[float, float]:
    intersections = total_a = total_b = exact = 0
    for row in rows:
        left = parser(row["annotator_a_" + field])
        right = parser(row["annotator_b_" + field])
        intersections += len(left & right)
        total_a += len(left)
        total_b += len(right)
        exact += int(left == right)
    overlap_f1 = 2 * intersections / (total_a + total_b) if total_a + total_b else 1.0
    return overlap_f1, exact / len(rows)


def entity_agreement(final: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, (field, parser) in enumerate((
        ("gold_biomedical_mentions_json", normalized_mention_set),
        ("gold_concept_links_json", normalized_concept_set),
    )):
        overlap, exact = set_agreement_stats(final, field, parser)
        overlap_ci = bootstrap_ci(final, lambda sample, f=field, p=parser: set_agreement_stats(sample, f, p)[0], 300 + index)
        exact_ci = bootstrap_ci(final, lambda sample, f=field, p=parser: set_agreement_stats(sample, f, p)[1], 320 + index)
        output.append({
            "field": field,
            "metric_type": "set_overlap",
            "n": len(final),
            "overlap_f1": overlap,
            "overlap_f1_ci_low": overlap_ci[0],
            "overlap_f1_ci_high": overlap_ci[1],
            "exact_row_agreement": exact,
            "exact_row_ci_low": exact_ci[0],
            "exact_row_ci_high": exact_ci[1],
            "raw_agreement": None,
            "cohens_kappa": None,
            "kappa_status": "not_applicable_free_text_sets",
        })
    for index, field in enumerate(ENTITY_ENUMS):
        left = [row["annotator_a_" + field].strip().upper() for row in final]
        right = [row["annotator_b_" + field].strip().upper() for row in final]
        observed, kappa, status = cohen_kappa(left, right)
        agreement_ci = bootstrap_ci(final, lambda sample, f=field: sum(row["annotator_a_" + f] == row["annotator_b_" + f] for row in sample) / len(sample), 340 + index)
        kappa_ci = bootstrap_ci(final, lambda sample, f=field: cohen_kappa([row["annotator_a_" + f] for row in sample], [row["annotator_b_" + f] for row in sample])[1], 360 + index)
        output.append({
            "field": field,
            "metric_type": "categorical",
            "n": len(final),
            "overlap_f1": None,
            "overlap_f1_ci_low": None,
            "overlap_f1_ci_high": None,
            "exact_row_agreement": None,
            "exact_row_ci_low": None,
            "exact_row_ci_high": None,
            "raw_agreement": observed,
            "agreement_ci_low": agreement_ci[0],
            "agreement_ci_high": agreement_ci[1],
            "cohens_kappa": kappa,
            "kappa_ci_low": kappa_ci[0],
            "kappa_ci_high": kappa_ci[1],
            "kappa_status": status,
        })
    return output


def entity_claim_counts(row: dict[str, str], mapping_audit: list[dict[str, Any]]) -> dict[str, Any]:
    row_id = row["id"]
    predictions = parse_array(row["predicted_links_json"], f"{row_id}.predicted_links_json")
    gold_mentions = normalized_mention_set(row["final_gold_biomedical_mentions_json"])
    missed = normalized_mention_set(row["final_missed_mentions_json"])
    if not missed <= gold_mentions:
        raise RuntimeError(f"{row_id}: final missed mentions are not a subset of gold mentions")
    predicted_mentions = {norm_mention(item.get("matched_alias") or item.get("name")) for item in predictions if norm_mention(item.get("matched_alias") or item.get("name"))}
    mention_tp = len(gold_mentions - missed)
    if mention_tp > len(predicted_mentions):
        raise RuntimeError(f"{row_id}: adjudicated mention TP exceeds emitted unique mentions")
    mention_fp = len(predicted_mentions) - mention_tp
    mention_fn = len(missed)

    prediction_keys: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        key = (str(prediction.get("node_id", "")).strip(), norm_mention(prediction.get("matched_alias") or prediction.get("name")))
        prediction_keys[key].append(index)
    incorrect_indices: set[int] = set()
    for error in parse_array(row["final_incorrect_predicted_links_json"], f"{row_id}.final_incorrect_predicted_links_json"):
        if not isinstance(error, dict):
            raise RuntimeError(f"{row_id}: incorrect link entry must be an object")
        key = (str(error.get("node_id", "")).strip(), norm_mention(error.get("mention")))
        matches = prediction_keys.get(key, [])
        strategy = "node_id_and_normalized_mention"
        if len(matches) == 0:
            node_matches = [index for index, prediction in enumerate(predictions) if str(prediction.get("node_id", "")).strip() == key[0]]
            if len(node_matches) == 1:
                matches = node_matches
                strategy = "unique_node_id_fallback_gold_span_recorded"
            else:
                raise RuntimeError(
                    f"{row_id}: incorrect link {key} has no exact node+mention match and maps to "
                    f"{len(node_matches)} node-only predictions; expected exactly one"
                )
        elif len(matches) != 1:
            raise RuntimeError(f"{row_id}: incorrect link {key} maps to {len(matches)} predictions; expected exactly one")
        if matches[0] in incorrect_indices:
            raise RuntimeError(f"{row_id}: duplicate incorrect-link judgment for prediction {matches[0]}")
        incorrect_indices.add(matches[0])
        prediction = predictions[matches[0]]
        mapping_audit.append({
            "id": row_id,
            "error_node_id": key[0],
            "error_mention": error.get("mention", ""),
            "error_reason": error.get("reason", ""),
            "predicted_mention": prediction.get("matched_alias") or prediction.get("name") or "",
            "prediction_index": matches[0],
            "mapping_strategy": strategy,
        })

    gold_concepts = parse_array(row["final_gold_concept_links_json"], f"{row_id}.final_gold_concept_links_json")
    resolved_total = resolved_detected = resolved_exact = 0
    type_gold: Counter[str] = Counter()
    type_detected: Counter[str] = Counter()
    for concept in gold_concepts:
        mention = norm_mention(concept.get("mention"))
        entity_type = str(concept.get("entity_type") or "UNKNOWN").strip() or "UNKNOWN"
        type_gold[entity_type] += 1
        if mention not in missed:
            type_detected[entity_type] += 1
        concept_id = str(concept.get("concept_id", "")).strip()
        if not concept_id or concept_id.upper() == "UNRESOLVED":
            continue
        resolved_total += 1
        candidates = [prediction for prediction in predictions if norm_mention(prediction.get("matched_alias") or prediction.get("name")) == mention]
        if candidates:
            resolved_detected += 1
        if any(str(prediction.get("node_id", "")).strip() == concept_id for prediction in candidates):
            resolved_exact += 1

    type_link_total: Counter[str] = Counter()
    type_link_correct: Counter[str] = Counter()
    for index, prediction in enumerate(predictions):
        kind = str(prediction.get("kind") or "UNKNOWN").strip() or "UNKNOWN"
        type_link_total[kind] += 1
        if index not in incorrect_indices:
            type_link_correct[kind] += 1

    return {
        "id": row_id,
        "dataset": row["dataset"],
        "abbreviation_ambiguity": row["final_abbreviation_ambiguity"].strip().upper(),
        "overall_linking_judgment": row["final_overall_linking_judgment"].strip().upper(),
        "mention_tp": mention_tp,
        "mention_fp": mention_fp,
        "mention_fn": mention_fn,
        "predicted_links": len(predictions),
        "incorrect_links": len(incorrect_indices),
        "correct_links": len(predictions) - len(incorrect_indices),
        "resolved_gold": resolved_total,
        "resolved_detected": resolved_detected,
        "resolved_exact": resolved_exact,
        "type_gold": dict(type_gold),
        "type_detected": dict(type_detected),
        "type_link_total": dict(type_link_total),
        "type_link_correct": dict(type_link_correct),
    }


def sum_counts(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for row in rows:
        for field in ("mention_tp", "mention_fp", "mention_fn", "predicted_links", "incorrect_links", "correct_links", "resolved_gold", "resolved_detected", "resolved_exact"):
            totals[field] += int(row[field])
    return totals


def entity_metric_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = sum_counts(rows)
    precision, recall, f1 = f1_from_counts(counts["mention_tp"], counts["mention_fp"], counts["mention_fn"])
    return {
        "mention_precision": precision,
        "mention_recall": recall,
        "mention_f1": f1,
        "human_judged_link_accuracy": counts["correct_links"] / counts["predicted_links"] if counts["predicted_links"] else 0.0,
        "strict_id_accuracy_detected_resolved": counts["resolved_exact"] / counts["resolved_detected"] if counts["resolved_detected"] else 0.0,
        "end_to_end_resolved_concept_recall": counts["resolved_exact"] / counts["resolved_gold"] if counts["resolved_gold"] else 0.0,
    }


def entity_summary_row(scope: str, value: str, rows: list[dict[str, Any]], seed_base: int) -> dict[str, Any]:
    counts = sum_counts(rows)
    metrics = entity_metric_values(rows)
    output: dict[str, Any] = {"scope": scope, "value": value, "claims": len(rows), **counts, **metrics}
    for index, metric in enumerate(metrics):
        ci = bootstrap_ci(rows, lambda sample, m=metric: entity_metric_values(sample)[m], seed_base + index)
        output[metric + "_ci_low"] = ci[0]
        output[metric + "_ci_high"] = ci[1]
    return output


def type_rows(claim_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    types = sorted({key for row in claim_rows for field in ("type_gold", "type_link_total") for key in row[field]})
    output: list[dict[str, Any]] = []
    for type_index, entity_type in enumerate(types):
        gold = sum(row["type_gold"].get(entity_type, 0) for row in claim_rows)
        detected = sum(row["type_detected"].get(entity_type, 0) for row in claim_rows)
        emitted = sum(row["type_link_total"].get(entity_type, 0) for row in claim_rows)
        correct = sum(row["type_link_correct"].get(entity_type, 0) for row in claim_rows)

        def mention_recall(sample: list[dict[str, Any]]) -> float | None:
            denominator = sum(row["type_gold"].get(entity_type, 0) for row in sample)
            numerator = sum(row["type_detected"].get(entity_type, 0) for row in sample)
            return numerator / denominator if denominator else None

        def link_accuracy(sample: list[dict[str, Any]]) -> float | None:
            denominator = sum(row["type_link_total"].get(entity_type, 0) for row in sample)
            numerator = sum(row["type_link_correct"].get(entity_type, 0) for row in sample)
            return numerator / denominator if denominator else None

        recall_ci = bootstrap_ci(claim_rows, mention_recall, 600 + type_index * 2)
        accuracy_ci = bootstrap_ci(claim_rows, link_accuracy, 601 + type_index * 2)
        output.append({
            "entity_type": entity_type,
            "gold_mentions": gold,
            "detected_gold_mentions": detected,
            "mention_recall": detected / gold if gold else None,
            "mention_recall_ci_low": recall_ci[0],
            "mention_recall_ci_high": recall_ci[1],
            "emitted_links": emitted,
            "human_judged_correct_links": correct,
            "human_judged_link_accuracy": correct / emitted if emitted else None,
            "link_accuracy_ci_low": accuracy_ci[0],
            "link_accuracy_ci_high": accuracy_ci[1],
        })
    return output


def external_sensitivity(pubmed_final: list[dict[str, str]]) -> list[dict[str, Any]]:
    compatible_ids = {row["id"] for row in pubmed_final if row["final_label_compatibility"].strip().upper() == "COMPATIBLE"}
    if len(compatible_ids) != 44:
        raise RuntimeError(f"Expected 44 compatible audit IDs, found {len(compatible_ids)}")
    baseline_rows = read_jsonl(STAGE8 / "external_baseline_results.jsonl")
    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baseline_names = {
        "direct_llm": "direct_llm",
        "text_rag_llm": "provided_text_verifier",
        "vanilla_graphrag_llm": "kg_only_local_path_prompting",
        "medgraphrag_style_llm": "text_kg_bm25_joint_prompting",
    }
    for row in baseline_rows:
        if row["id"] in compatible_ids:
            methods[baseline_names[row["baseline"]]].append({
                "id": row["id"], "gold_label": row["gold_label"], "pred_label": row["pred_label"], "reviewed": False,
            })

    risk_rows = read_jsonl(STAGE8 / "external_guardrail_scores.jsonl")
    eligible = [row for row in risk_rows if row["candidate_label"] == "SUPPORT"]
    budget = round(0.05 * len(risk_rows))

    def top_ids(field: str) -> set[str]:
        return {row["id"] for row in sorted(eligible, key=lambda item: (-float(item[field]), str(item["id"])))[:budget]}

    confidence_ids = top_ids("risk_confidence_only")
    learned_ids = top_ids("risk_hierarchical_learned")
    for row in risk_rows:
        if row["id"] not in compatible_ids:
            continue
        candidate = row["candidate_label"]
        methods["confidence_review_full300_top15"].append({
            "id": row["id"], "gold_label": row["gold_label"],
            "pred_label": "UNCERTAIN" if row["id"] in confidence_ids and candidate == "SUPPORT" else candidate,
            "reviewed": row["id"] in confidence_ids,
        })
        rule_review = candidate == "SUPPORT" and row.get("guardrail_status") == "KG_TWO_HOP_CONTEXT"
        methods["rule_guardrail_original_trigger"].append({
            "id": row["id"], "gold_label": row["gold_label"],
            "pred_label": "UNCERTAIN" if rule_review else candidate, "reviewed": rule_review,
        })
        methods["learned_review_full300_top15"].append({
            "id": row["id"], "gold_label": row["gold_label"],
            "pred_label": "UNCERTAIN" if row["id"] in learned_ids and candidate == "SUPPORT" else candidate,
            "reviewed": row["id"] in learned_ids,
        })

    output = []
    for method_index, (method, rows) in enumerate(sorted(methods.items())):
        if len(rows) != 44 or len({row["id"] for row in rows}) != 44:
            raise RuntimeError(f"{method}: expected 44 unique compatible rows, found {len(rows)}")
        metrics = classification_metrics(rows)
        record: dict[str, Any] = {"method": method, "analysis_scope": "post_audit_compatible44", **metrics}
        for metric_index, metric in enumerate(("accuracy", "macro_f1", "support_precision", "false_support_rate")):
            ci = bootstrap_ci(rows, lambda sample, m=metric: float(classification_metrics(sample)[m]), 800 + method_index * 10 + metric_index)
            record[metric + "_ci_low"] = ci[0]
            record[metric + "_ci_high"] = ci[1]
        output.append(record)
    return output


def fmt_rate(value: float | None) -> str:
    return "NA" if value is None else f"{value:.3f}"


def build_summary(
    pubmed_agreement: list[dict[str, Any]],
    pubmed_distribution: list[dict[str, Any]],
    entity_agreement_rows: list[dict[str, Any]],
    entity_overall: dict[str, Any],
    sensitivity: list[dict[str, Any]],
    encodings: dict[str, str],
) -> str:
    final_lookup = {(row["field"], row["value"]): row for row in pubmed_distribution}
    lines = [
        "# Stage 9 finalized human-audit analysis",
        "",
        f"- Seed: `{SEED}`",
        f"- Claim-level bootstrap replicates: `{BOOTSTRAPS}`",
        "- PubMedQA audit: 60 dual-annotated and adjudicated records.",
        "- Entity-linking audit: 120 dual-annotated and adjudicated records (30 per dataset).",
        "",
        "## PubMedQA conversion and label mapping",
        "",
        "| Dimension | Raw agreement | Cohen's kappa | Final positive/desired rate |",
        "|---|---:|---:|---:|",
    ]
    desired = {
        "claim_faithfulness": "VALID",
        "atomicity": "ATOMIC",
        "label_compatibility": "COMPATIBLE",
        "pico_preservation": "COMPLETE",
        "modality_strength": "PRESERVED",
    }
    for row in pubmed_agreement:
        final = final_lookup[(row["field"], desired[row["field"]])]
        lines.append(f"| {row['field']} | {row['raw_agreement']:.3f} | {fmt_rate(row['cohens_kappa'])} | {final['count']}/60 ({final['rate']:.1%}) |")
    lines.extend([
        "",
        "The claim conversion passed the original faithfulness and atomicity gates, but mapped-label compatibility was only 44/60 (73.3%). The PubMedQA-Claim-300 experiment must therefore be described as a QA-derived label-noise stress test rather than a reliable external gold evaluation.",
        "",
        "## Entity-linking audit",
        "",
        f"- Mention precision/recall/F1: {entity_overall['mention_precision']:.3f} / {entity_overall['mention_recall']:.3f} / {entity_overall['mention_f1']:.3f}.",
        f"- Human-judged link accuracy: {entity_overall['human_judged_link_accuracy']:.3f} ({entity_overall['correct_links']}/{entity_overall['predicted_links']}).",
        f"- Strict ID accuracy among detected resolved concepts: {entity_overall['strict_id_accuracy_detected_resolved']:.3f}.",
        f"- End-to-end resolved-concept recall: {entity_overall['end_to_end_resolved_concept_recall']:.3f}.",
        "",
        "## Annotation agreement",
        "",
    ])
    for row in entity_agreement_rows:
        if row["metric_type"] == "set_overlap":
            lines.append(f"- {row['field']}: overlap F1={row['overlap_f1']:.3f}; exact-row agreement={row['exact_row_agreement']:.3f}.")
        else:
            lines.append(f"- {row['field']}: raw agreement={row['raw_agreement']:.3f}; kappa={fmt_rate(row['cohens_kappa'])}.")
    lines.extend([
        "",
        "## Post-audit compatible-44 sensitivity analysis",
        "",
        "This analysis is descriptive, small-sample, and post-audit. Review sets remain the frozen full-300 selections; no budget is reallocated within the 44 records.",
        "",
        "| Method | Accuracy | Macro-F1 | SUPPORT precision | False-support rate |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in sensitivity:
        lines.append(f"| {row['method']} | {row['accuracy']:.3f} | {row['macro_f1']:.3f} | {row['support_precision']:.3f} | {row['false_support_rate']:.3f} |")
    lines.extend(["", "## Source encodings", ""])
    for name, encoding in encodings.items():
        lines.append(f"- `{name}`: `{encoding}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Stage 9 PubMedQA and entity-linking human audits.")
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    audit_dir = args.audit_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, list[dict[str, str]]] = {}
    encodings: dict[str, str] = {}
    for name in AUDIT_FILES:
        loaded[name], encodings[name] = read_csv(audit_dir / name)

    pub_a = loaded[AUDIT_FILES[0]]
    pub_b = loaded[AUDIT_FILES[1]]
    pub_final = loaded[AUDIT_FILES[2]]
    entity_a = loaded[AUDIT_FILES[3]]
    entity_b = loaded[AUDIT_FILES[4]]
    entity_final = loaded[AUDIT_FILES[5]]
    validate_pubmed(pub_a, pub_b, pub_final)
    validate_entity(entity_a, entity_b, entity_final)

    hashes = {
        "generated_at_note": "Hashes identify the frozen adjudicated inputs; timestamps are intentionally excluded.",
        "files": [
            {"path": name, "bytes": (audit_dir / name).stat().st_size, "sha256": sha256(audit_dir / name), "detected_encoding": encodings[name]}
            for name in AUDIT_FILES
        ],
    }
    write_json(output_dir / "stage9_human_audit_sha256.json", hashes)

    pub_agreement, pub_distribution = pubmed_analysis(pub_final)
    write_csv(output_dir / "pubmedqa_audit_agreement.csv", pub_agreement)
    write_csv(output_dir / "pubmedqa_audit_final_distribution.csv", pub_distribution)

    entity_agreement_rows = entity_agreement(entity_final)
    write_csv(output_dir / "entity_annotation_agreement.csv", entity_agreement_rows)
    mapping_audit: list[dict[str, Any]] = []
    entity_claim_rows = [entity_claim_counts(row, mapping_audit) for row in entity_final]
    write_csv(output_dir / "entity_link_error_mapping_audit.csv", mapping_audit)
    if len(mapping_audit) != sum(row["incorrect_links"] for row in entity_claim_rows):
        raise RuntimeError("Incorrect-link mapping audit does not reconcile with adjudicated error counts")
    claim_flat = []
    for row in entity_claim_rows:
        claim_flat.append({key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in row.items()})
    write_csv(output_dir / "entity_linking_claim_metrics.csv", claim_flat)

    overall = entity_summary_row("overall", "all", entity_claim_rows, 400)
    by_dataset = [entity_summary_row("dataset", dataset, [row for row in entity_claim_rows if row["dataset"] == dataset], 450 + index * 10) for index, dataset in enumerate(sorted({row["dataset"] for row in entity_claim_rows}))]
    by_abbreviation = [entity_summary_row("abbreviation_ambiguity", value, [row for row in entity_claim_rows if row["abbreviation_ambiguity"] == value], 500 + index * 10) for index, value in enumerate(("NO", "YES"))]
    write_csv(output_dir / "entity_linking_overall_metrics.csv", [overall])
    write_csv(output_dir / "entity_linking_by_dataset.csv", by_dataset)
    write_csv(output_dir / "entity_linking_by_abbreviation.csv", by_abbreviation)
    write_csv(output_dir / "entity_linking_by_type.csv", type_rows(entity_claim_rows))

    judgments = []
    for dataset in ["ALL", *sorted({row["dataset"] for row in entity_claim_rows})]:
        selected = entity_claim_rows if dataset == "ALL" else [row for row in entity_claim_rows if row["dataset"] == dataset]
        counts = Counter(row["overall_linking_judgment"] for row in selected)
        for value in sorted(ENTITY_ENUMS["overall_linking_judgment"]):
            low, high = wilson_interval(counts[value], len(selected))
            judgments.append({"dataset": dataset, "judgment": value, "count": counts[value], "n": len(selected), "rate": counts[value] / len(selected), "ci_low": low, "ci_high": high})
    write_csv(output_dir / "entity_linking_judgment_distribution.csv", judgments)

    sensitivity = external_sensitivity(pub_final)
    write_csv(output_dir / "pubmedqa_audit60_compatible44_sensitivity.csv", sensitivity)

    stats = {
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "validation": {"passed": True, "pubmedqa_rows": len(pub_final), "entity_rows": len(entity_final)},
        "pubmedqa": {"agreement": pub_agreement, "final_distribution": pub_distribution},
        "entity_linking": {"agreement": entity_agreement_rows, "overall": overall, "by_dataset": by_dataset, "by_abbreviation": by_abbreviation},
        "compatible44_sensitivity": sensitivity,
        "input_hashes": hashes,
    }
    write_json(output_dir / "stage9_human_audit_stats.json", stats)
    summary = build_summary(pub_agreement, pub_distribution, entity_agreement_rows, overall, sensitivity, encodings)
    (output_dir / "stage9_human_audit_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "validation_passed": True,
        "pubmedqa_rows": len(pub_final),
        "entity_rows": len(entity_final),
        "pubmedqa_label_compatible": sum(row["final_label_compatibility"] == "COMPATIBLE" for row in pub_final),
        "entity_overall": overall,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
