from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_NODE_TYPES = {
    "Disease",
    "Compound",
    "Gene",
    "Side Effect",
    "SideEffect",
    "Symptom",
    "Anatomy",
}

DEFAULT_RELATIONS = {
    "treats",
    "palliates",
    "associates",
    "causes",
    "interacts",
    "targets",
    "resembles",
    "presents",
    "expresses",
    "regulates",
    "upregulates",
    "downregulates",
}

EXPANDED_NODE_TYPES = DEFAULT_NODE_TYPES | {
    "Biological Process",
    "Cellular Component",
    "Molecular Function",
    "Pathway",
    "Pharmacologic Class",
}

EXPANDED_RELATIONS = DEFAULT_RELATIONS | {
    "includes",
    "localizes",
    "participates",
}

RELATION_CANONICAL = {
    "binds": "targets",
    "binding": "targets",
    "target": "targets",
    "targets": "targets",
    "treat": "treats",
    "treats": "treats",
    "palliate": "palliates",
    "palliates": "palliates",
    "associate": "associates",
    "associates": "associates",
    "associated": "associates",
    "cause": "causes",
    "causes": "causes",
    "interact": "interacts",
    "interacts": "interacts",
    "resembles": "resembles",
    "present": "presents",
    "presents": "presents",
    "expresses": "expresses",
    "regulates": "regulates",
    "upregulates": "upregulates",
    "downregulates": "downregulates",
}

HETIONET_ABBREVIATION_MAP = {
    "AeG": "expresses",
    "AdG": "downregulates",
    "AuG": "upregulates",
    "CbG": "targets",
    "CcSE": "causes",
    "CdG": "downregulates",
    "CpD": "palliates",
    "CrC": "resembles",
    "CtD": "treats",
    "CuG": "upregulates",
    "DaG": "associates",
    "DdG": "downregulates",
    "DlA": "localizes",
    "DrD": "resembles",
    "DuG": "upregulates",
    "GcG": "covaries",
    "GiG": "interacts",
    "GpBP": "participates",
    "GpCC": "participates",
    "GpMF": "participates",
    "GpPW": "participates",
    "Gr>G": "regulates",
    "PCiC": "includes",
    "PWpPW": "participates",
}


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def canonical_relation(value: str) -> str:
    key = normalize_text(value)
    return RELATION_CANONICAL.get(key, key)


def split_arg(value: str | None, default: set[str]) -> set[str]:
    if not value:
        return set(default)
    return {item.strip() for item in value.split(",") if item.strip()}


def find_column(fieldnames: Iterable[str], candidates: list[str]) -> str | None:
    by_normalized = {normalize_text(name): name for name in fieldnames}
    for candidate in candidates:
        key = normalize_text(candidate)
        if key in by_normalized:
            return by_normalized[key]
    return None


def read_nodes(nodes_path: Path, allowed_types: set[str]) -> dict[str, dict[str, str]]:
    with nodes_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Node file has no header: {nodes_path}")

        id_col = find_column(reader.fieldnames, ["id", "identifier", "node_id"])
        name_col = find_column(reader.fieldnames, ["name", "label"])
        kind_col = find_column(reader.fieldnames, ["kind", "type", "category", "metanode"])

        missing = [
            label
            for label, column in [("id", id_col), ("name", name_col), ("kind", kind_col)]
            if column is None
        ]
        if missing:
            raise ValueError(f"Missing required node columns {missing} in {nodes_path}")

        allowed_normalized = {normalize_text(item) for item in allowed_types}
        nodes: dict[str, dict[str, str]] = {}
        for row in reader:
            node_id = row[id_col].strip()
            name = row[name_col].strip()
            kind = row[kind_col].strip()
            if normalize_text(kind) not in allowed_normalized:
                continue
            if not node_id or not name:
                continue
            nodes[node_id] = {"id": node_id, "name": name, "kind": kind}
        return nodes


def parse_metaedge_phrase(value: str) -> str | None:
    parts = [part.strip() for part in re.split(r"\s+-\s+", value) if part.strip()]
    if len(parts) >= 3:
        return parts[1]
    return None


def read_metaedges(metaedges_path: Path) -> dict[str, str]:
    if not metaedges_path.exists():
        return dict(HETIONET_ABBREVIATION_MAP)

    mapping = dict(HETIONET_ABBREVIATION_MAP)
    with metaedges_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return mapping

        abbreviation_col = find_column(reader.fieldnames, ["abbreviation", "abbr", "code"])
        relation_col = find_column(
            reader.fieldnames,
            ["kind", "relationship", "relation", "predicate", "edge kind"],
        )
        metaedge_col = find_column(reader.fieldnames, ["metaedge", "name"])

        if abbreviation_col is None:
            return mapping

        for row in reader:
            abbreviation = row.get(abbreviation_col, "").strip()
            if not abbreviation:
                continue

            relation = row.get(relation_col, "").strip() if relation_col else ""
            if not relation and metaedge_col:
                relation = parse_metaedge_phrase(row.get(metaedge_col, "").strip()) or ""
            if relation:
                mapping[abbreviation] = canonical_relation(relation)
    return mapping


def iter_edges(edges_path: Path) -> Iterable[tuple[str, str, str]]:
    opener = gzip.open if edges_path.suffix == ".gz" else open
    with opener(edges_path, "rt", encoding="utf-8", newline="") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed edge line {line_number}: {line[:120]}")
            source, metaedge, target = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if source.lower() == "source" and target.lower() in {"target", "destination"}:
                continue
            yield source, metaedge, target


def build_aliases(nodes: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    aliases: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for node in nodes.values():
        candidates = {node["name"], node["id"]}
        if "::" in node["id"]:
            candidates.add(node["id"].split("::", 1)[1])
        for alias in candidates:
            alias = alias.strip()
            normalized = normalize_text(alias)
            if not normalized:
                continue
            key = (node["id"], normalized)
            if key in seen:
                continue
            seen.add(key)
            aliases.append(
                {
                    "alias": alias,
                    "normalized_alias": normalized,
                    "node_id": node["id"],
                    "name": node["name"],
                    "kind": node["kind"],
                }
            )
    aliases.sort(key=lambda item: (item["normalized_alias"], item["node_id"]))
    return aliases


def write_tsv(path: Path, rows: Iterable[dict[str, str]], fields: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_graph(
    raw_dir: Path,
    out_dir: Path,
    allowed_types: set[str],
    allowed_relations: set[str],
) -> dict:
    nodes_path = raw_dir / "hetionet" / "hetionet-v1.0-nodes.tsv"
    edges_path = raw_dir / "hetionet" / "hetionet-v1.0-edges.sif.gz"
    metaedges_path = raw_dir / "hetionet" / "metaedges.tsv"

    for path in [nodes_path, edges_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required Hetionet file not found: {path}")

    nodes = read_nodes(nodes_path, allowed_types)
    relation_map = read_metaedges(metaedges_path)
    allowed_relation_normalized = {canonical_relation(item) for item in allowed_relations}

    edges: list[dict[str, str]] = []
    skipped_relation = Counter()
    skipped_node_type = 0
    for source, metaedge, target in iter_edges(edges_path):
        if source not in nodes or target not in nodes:
            skipped_node_type += 1
            continue

        relation = canonical_relation(relation_map.get(metaedge, metaedge))
        if relation not in allowed_relation_normalized:
            skipped_relation[relation] += 1
            continue

        source_node = nodes[source]
        target_node = nodes[target]
        edges.append(
            {
                "source_id": source,
                "source_name": source_node["name"],
                "source_kind": source_node["kind"],
                "relation": relation,
                "metaedge": metaedge,
                "target_id": target,
                "target_name": target_node["name"],
                "target_kind": target_node["kind"],
            }
        )

    connected_node_ids = {edge["source_id"] for edge in edges} | {edge["target_id"] for edge in edges}
    filtered_nodes = [nodes[node_id] for node_id in sorted(connected_node_ids)]
    aliases = build_aliases({node["id"]: node for node in filtered_nodes})

    out_dir.mkdir(parents=True, exist_ok=True)
    node_count = write_tsv(out_dir / "hetionet_nodes.tsv", filtered_nodes, ["id", "name", "kind"])
    edge_count = write_tsv(
        out_dir / "hetionet_edges.tsv",
        edges,
        [
            "source_id",
            "source_name",
            "source_kind",
            "relation",
            "metaedge",
            "target_id",
            "target_name",
            "target_kind",
        ],
    )
    alias_count = write_jsonl(out_dir / "hetionet_aliases.jsonl", aliases)

    stats = {
        "nodes": node_count,
        "edges": edge_count,
        "aliases": alias_count,
        "node_type_counts": dict(Counter(node["kind"] for node in filtered_nodes)),
        "relation_counts": dict(Counter(edge["relation"] for edge in edges)),
        "metaedge_counts": dict(Counter(edge["metaedge"] for edge in edges)),
        "skipped_edges_due_to_filtered_node_type": skipped_node_type,
        "skipped_edges_due_to_relation": dict(skipped_relation),
        "allowed_node_types": sorted(allowed_types),
        "allowed_relations": sorted(allowed_relation_normalized),
        "outputs": {
            "nodes": str(out_dir / "hetionet_nodes.tsv"),
            "edges": str(out_dir / "hetionet_edges.tsv"),
            "aliases": str(out_dir / "hetionet_aliases.jsonl"),
            "stats": str(out_dir / "hetionet_graph_stats.json"),
        },
    }

    stats_path = out_dir / "hetionet_graph_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MVP Hetionet graph tables.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--node-types", default=None, help="Comma-separated node type allowlist.")
    parser.add_argument("--relations", default=None, help="Comma-separated relation allowlist.")
    parser.add_argument("--expanded", action="store_true", help="Keep additional mechanism and pathway node types.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    default_node_types = EXPANDED_NODE_TYPES if args.expanded else DEFAULT_NODE_TYPES
    default_relations = EXPANDED_RELATIONS if args.expanded else DEFAULT_RELATIONS
    stats = build_graph(
        raw_dir=data_dir / "raw",
        out_dir=Path(args.output_dir) if args.output_dir else data_dir / "processed",
        allowed_types=split_arg(args.node_types, default_node_types),
        allowed_relations=split_arg(args.relations, default_relations),
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
