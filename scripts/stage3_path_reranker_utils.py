from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


NEGATIVE_RELATIONS = {
    "contraindicates",
    "negatively_regulates",
    "inhibits",
    "not_presents",
    "not_expressed",
}
NUMERIC_FEATURES = [
    "path_score",
    "lexical_overlap",
    "relation_priority",
    "hop_count",
    "linked_entity_count",
    "avg_entity_score",
    "max_entity_score",
    "candidate_path_count",
    "is_direct_path",
    "has_negative_relation",
]


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def stable_fraction(value: str) -> float:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_name(value: str, calibration_ratio: float = 0.30) -> str:
    return "calibration" if stable_fraction(value) < calibration_ratio else "test"


def candidate_paths(row: dict, limit: int | None = None) -> list[dict]:
    paths = row.get("top_evidence_paths") or row.get("evidence_paths") or []
    return paths[:limit] if limit is not None else paths


def normalize_excerpt(text: str, limit: int = 800) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    return value[:limit]


def alias_source_family(source: str) -> str:
    value = (source or "unknown").lower()
    if value.startswith("umls:"):
        return "umls"
    if "primekg" in value:
        return "primekg"
    if "hetionet" in value:
        return "hetionet"
    if "hgnc" in value:
        return "hgnc"
    if "mondo" in value:
        return "mondo"
    return "other"


def entity_summary(entities: list[dict]) -> str:
    values = []
    for entity in entities:
        name = entity.get("matched_alias") or entity.get("name") or entity.get("node_id")
        values.append(f"{name} [{entity.get('kind', 'Unknown')}]")
    return "; ".join(values)


def path_feature_dict(row: dict, path: dict) -> dict[str, float]:
    entities = row.get("linked_entities") or []
    entity_scores = [float(item.get("score", 0.0)) for item in entities]
    relations = path.get("relations") or []
    path_type = path.get("path_type") or "unknown"
    hop_count = 1 if path_type == "1-hop" else 2 if path_type == "2-hop" else len(relations)
    features: dict[str, float] = {
        "path_score": float(path.get("score", 0.0)),
        "lexical_overlap": float(path.get("lexical_overlap", 0.0)),
        "relation_priority": float(path.get("relation_priority", 0.0)),
        "hop_count": float(hop_count),
        "linked_entity_count": float(len(entities)),
        "avg_entity_score": sum(entity_scores) / len(entity_scores) if entity_scores else 0.0,
        "max_entity_score": max(entity_scores) if entity_scores else 0.0,
        "candidate_path_count": float(len(candidate_paths(row))),
        "is_direct_path": 1.0 if path_type == "1-hop" else 0.0,
        "has_negative_relation": 1.0 if any(rel in NEGATIVE_RELATIONS for rel in relations) else 0.0,
        f"dataset={row.get('dataset') or 'unknown'}": 1.0,
        f"path_type={path_type}": 1.0,
    }
    for relation in sorted(set(relations)):
        features[f"relation={relation}"] = 1.0
    kinds = sorted({str(item.get("kind") or "Unknown") for item in entities})
    if kinds:
        features[f"entity_kinds={'|'.join(kinds)}"] = 1.0
    for family in sorted({alias_source_family(str(item.get("matched_alias_source") or "")) for item in entities}):
        features[f"alias_source={family}"] = 1.0
    return features


def build_feature_names(feature_rows: list[dict[str, float]]) -> list[str]:
    categorical_counts = Counter(
        key
        for row in feature_rows
        for key in row
        if key not in NUMERIC_FEATURES
    )
    categorical = sorted(key for key, count in categorical_counts.items() if count >= 2)
    return ["bias", *NUMERIC_FEATURES, *categorical]


def vectorize(features: dict[str, float], feature_names: list[str]) -> list[float]:
    return [1.0 if name == "bias" else float(features.get(name, 0.0)) for name in feature_names]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def predict_probability(features: dict[str, float], model: dict) -> float:
    vector = vectorize(features, model["feature_names"])
    value = sum(float(weight) * item for weight, item in zip(model["weights"], vector))
    return sigmoid(value)


def existing_quality_score(row: dict, path: dict | None) -> float:
    if not path:
        return 0.0
    entity_count = len(row.get("linked_entities") or [])
    entity_coverage = min(entity_count / 2, 1.0)
    return max(
        0.0,
        min(
            1.0,
            0.35 * float(path.get("score", 0.0))
            + 0.25 * float(path.get("lexical_overlap", 0.0))
            + 0.15 * entity_coverage
            + 0.15 * float(path.get("relation_priority", 0.0))
            + 0.10 * min(len(candidate_paths(row)) / 10, 1.0),
        ),
    )

