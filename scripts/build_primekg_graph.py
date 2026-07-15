from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


TYPE_MAP = {
    "anatomy": "Anatomy",
    "biological_process": "Biological Process",
    "cellular_component": "Cellular Component",
    "disease": "Disease",
    "drug": "Compound",
    "effect/phenotype": "Symptom",
    "gene/protein": "Gene",
    "molecular_function": "Molecular Function",
    "pathway": "Pathway",
}

DISPLAY_RELATION_MAP = {
    "associated with": "associates",
    "carrier": "interacts",
    "contraindication": "contraindicates",
    "enzyme": "interacts",
    "expression absent": "not_expressed",
    "expression present": "expresses",
    "indication": "treats",
    "off label use": "palliates",
    "phenotype absent": "not_presents",
    "phenotype present": "presents",
    "target": "targets",
    "transporter": "interacts",
}

RELATION_FALLBACK_MAP = {
    "disease_disease": "associates",
    "disease_phenotype_negative": "not_presents",
    "disease_phenotype_positive": "presents",
    "disease_protein": "associates",
    "drug_disease": "treats",
    "drug_effect": "causes",
    "drug_protein": "targets",
    "pathway_protein": "participates",
    "phenotype_protein": "associates",
}

DEFAULT_RELATIONS = {
    "associates",
    "causes",
    "contraindicates",
    "expresses",
    "interacts",
    "not_expressed",
    "not_presents",
    "palliates",
    "participates",
    "presents",
    "targets",
    "treats",
}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("_", " ").replace("-", " "))


def canonical_relation(row: dict[str, str]) -> str | None:
    display = normalize(row["display_relation"])
    relation = row["relation"].strip().lower()
    return DISPLAY_RELATION_MAP.get(display) or RELATION_FALLBACK_MAP.get(relation)


def node_id(row: dict[str, str], side: str, kind: str) -> str:
    external_id = row[f"{side}_id"].strip()
    source = row[f"{side}_source"].strip() or "unknown"
    if kind == "Gene":
        return f"Gene::{external_id}"
    if kind == "Compound" and external_id.startswith("DB"):
        return f"Compound::{external_id}"
    if kind == "Anatomy" and source.upper() == "UBERON":
        suffix = external_id if external_id.startswith("UBERON:") else f"UBERON:{external_id}"
        return f"Anatomy::{suffix}"
    safe_source = re.sub(r"[^A-Za-z0-9_.:-]+", "_", source)
    return f"PrimeKG::{kind}::{safe_source}:{external_id}"


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a filtered PrimeKG graph using the local Hetionet TSV schema.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-ppi", action="store_true")
    parser.add_argument("--include-anatomy-expression", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    input_path = Path(args.input_path) if args.input_path else data_dir / "raw" / "primekg" / "kg.csv"
    out_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed" / "stage2_primekg" / "primekg_graph"
    allowed_relations = set(DEFAULT_RELATIONS)
    if args.include_ppi:
        allowed_relations.add("ppi")

    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}
    skipped = Counter()
    raw_relation_counts = Counter()
    relation_counts = Counter()

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_relation_counts[(row["relation"], row["display_relation"])] += 1
            x_kind = TYPE_MAP.get(row["x_type"].strip())
            y_kind = TYPE_MAP.get(row["y_type"].strip())
            if not x_kind or not y_kind:
                skipped["unsupported_node_type"] += 1
                continue
            relation = canonical_relation(row)
            if row["display_relation"].strip().lower() == "ppi" and args.include_ppi:
                relation = "ppi"
            if normalize(row["display_relation"]) == "parent child":
                skipped["hierarchy_relation"] += 1
                continue
            if relation in {"expresses", "not_expressed"} and not args.include_anatomy_expression:
                skipped["anatomy_expression_relation"] += 1
                continue
            if not relation or relation not in allowed_relations:
                skipped["unsupported_relation"] += 1
                continue

            source_id = node_id(row, "x", x_kind)
            target_id = node_id(row, "y", y_kind)
            source = {
                "id": source_id,
                "name": row["x_name"].strip(),
                "kind": x_kind,
                "external_id": row["x_id"].strip(),
                "external_source": row["x_source"].strip(),
            }
            target = {
                "id": target_id,
                "name": row["y_name"].strip(),
                "kind": y_kind,
                "external_id": row["y_id"].strip(),
                "external_source": row["y_source"].strip(),
            }
            nodes.setdefault(source_id, source)
            nodes.setdefault(target_id, target)
            key = (source_id, relation, target_id)
            edges.setdefault(
                key,
                {
                    "source_id": source_id,
                    "source_name": source["name"],
                    "source_kind": x_kind,
                    "relation": relation,
                    "metaedge": f"primekg:{row['relation']}:{row['display_relation']}",
                    "target_id": target_id,
                    "target_name": target["name"],
                    "target_kind": y_kind,
                },
            )
            relation_counts[relation] += 1

    node_rows = sorted(nodes.values(), key=lambda row: row["id"])
    edge_rows = sorted(edges.values(), key=lambda row: (row["source_id"], row["relation"], row["target_id"]))
    alias_rows = [
        {
            "alias": node["name"],
            "normalized_alias": normalize(node["name"]),
            "node_id": node["id"],
            "name": node["name"],
            "kind": node["kind"],
            "alias_source": "primekg",
            "external_id": node["external_id"],
        }
        for node in node_rows
        if node["name"].strip()
    ]

    write_tsv(out_dir / "hetionet_nodes.tsv", node_rows, ["id", "name", "kind"])
    write_tsv(
        out_dir / "hetionet_edges.tsv",
        edge_rows,
        ["source_id", "source_name", "source_kind", "relation", "metaedge", "target_id", "target_name", "target_kind"],
    )
    write_jsonl(out_dir / "hetionet_aliases.jsonl", alias_rows)
    stats = {
        "nodes": len(node_rows),
        "edges": len(edge_rows),
        "aliases": len(alias_rows),
        "include_ppi": args.include_ppi,
        "include_anatomy_expression": args.include_anatomy_expression,
        "node_type_counts": dict(Counter(row["kind"] for row in node_rows)),
        "relation_counts": dict(Counter(row["relation"] for row in edge_rows)),
        "skipped": dict(skipped),
        "raw_relation_counts": [
            {"relation": relation, "display_relation": display, "count": count}
            for (relation, display), count in raw_relation_counts.most_common()
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hetionet_graph_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
