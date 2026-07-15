from __future__ import annotations

import json
import re
from pathlib import Path


QUALIFIERS = ("NUMERIC", "DIRECTIONAL", "POPULATION", "TEMPORAL", "DOSAGE", "CONDITIONAL")


def bool_value(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def split_values(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [item.strip() for item in re.split(r"[;,|]", text) if item.strip()]


def build_adverse_event_pairs(primekg_graph_dir: Path) -> set[tuple[str, str]]:
    nodes_path = primekg_graph_dir / "hetionet_nodes.tsv"
    edges_path = primekg_graph_dir / "hetionet_edges.tsv"
    if not nodes_path.exists() or not edges_path.exists():
        return set()
    names: dict[str, str] = {}
    with nodes_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        id_index = header.index("id")
        name_index = header.index("name")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > max(id_index, name_index):
                names[parts[id_index]] = normalize_name(parts[name_index])
    pairs: set[tuple[str, str]] = set()
    with edges_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        source_index = header.index("source_id")
        target_index = header.index("target_id")
        relation_index = header.index("relation")
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(source_index, target_index, relation_index):
                continue
            if parts[relation_index] != "adverse_event_context":
                continue
            source_name = names.get(parts[source_index], "")
            target_name = names.get(parts[target_index], "")
            if source_name and target_name:
                pairs.add((source_name, target_name))
    return pairs


def normalize_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def semantic_clean_path(path_text: str, relations, adverse_pairs: set[tuple[str, str]]) -> tuple[str, list[str], int]:
    relation_list = split_values(relations)
    segments = [item.strip() for item in str(path_text or "").split(";") if item.strip()]
    output_segments: list[str] = []
    output_relations: list[str] = []
    remapped = 0
    pattern = re.compile(r"^(.+?)\s+--([^>]+?)-->\s+(.+?)$")
    for index, segment in enumerate(segments):
        match = pattern.match(segment)
        relation = relation_list[index] if index < len(relation_list) else (match.group(2).strip() if match else "")
        if match:
            source, target = match.group(1).strip(), match.group(3).strip()
            if relation == "causes" and (normalize_name(source), normalize_name(target)) in adverse_pairs:
                relation = "adverse_event_context"
                segment = f"{source} --{relation}--> {target}"
                remapped += 1
        output_segments.append(segment)
        output_relations.append(relation)
    if not segments:
        output_segments = [str(path_text or "")]
        output_relations = relation_list
    return " ; ".join(output_segments), output_relations, remapped


def combined_text(row: dict) -> str:
    return f"claim: {row.get('claim', '')} path: {row.get('path_text', '')}"


def structured_features(row: dict) -> dict[str, float | str]:
    relations = split_values(row.get("relations"))
    qualifiers = split_values(row.get("qualifier_flags"))
    path_type = str(row.get("path_type") or "none")
    features: dict[str, float | str] = {
        "path_score": float(row.get("path_score") or row.get("score") or 0.0),
        "hop_count": 2.0 if path_type.startswith("2") else (1.0 if path_type.startswith("1") else 0.0),
        "endpoint_aligned": float(bool_value(row.get("endpoint_aligned"))),
        "predicate_aligned": float(bool_value(row.get("predicate_aligned"))),
        "relation_count": float(len(relations)),
        "qualifier_count": float(len(qualifiers)),
        "path_type": path_type,
        "evidence_tier": str(row.get("evidence_tier") or "NO_KG_GROUNDING"),
        "kg_relation_family": str(row.get("kg_relation_family") or "UNKNOWN"),
        "dataset": str(row.get("dataset") or "unknown"),
    }
    for relation in relations:
        features[f"relation::{relation}"] = 1.0
    for qualifier in qualifiers:
        features[f"qualifier::{qualifier}"] = 1.0
    for predicate in split_values(row.get("claim_predicate_families")):
        features[f"predicate::{predicate}"] = 1.0
    return features


def evidence_path_rows(claim_row: dict) -> list[dict]:
    output = []
    for path in [*(claim_row.get("direct_paths") or []), *(claim_row.get("two_hop_context_paths") or [])]:
        item = dict(path)
        item["claim_id"] = claim_row.get("id")
        item["claim"] = claim_row.get("claim")
        item["dataset"] = claim_row.get("dataset")
        item["qualifier_flags"] = claim_row.get("qualifier_flags") or []
        output.append(item)
    return output
