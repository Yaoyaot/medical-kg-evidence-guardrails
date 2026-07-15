from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


NODE_FIELDS = ["id", "name", "kind"]
EDGE_FIELDS = ["source_id", "source_name", "source_kind", "relation", "metaedge", "target_id", "target_name", "target_kind"]


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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


def is_primekg_drug_effect_side_effect(edge: dict) -> bool:
    return str(edge.get("metaedge", "")).lower() == "primekg:drug_effect:side effect"


def clean_edge(edge: dict) -> tuple[dict | None, str]:
    if not is_primekg_drug_effect_side_effect(edge):
        return dict(edge), "unchanged"
    if edge.get("source_kind") == "Compound" and edge.get("target_kind") in {"Symptom", "Side Effect"}:
        item = dict(edge)
        item["relation"] = "adverse_event_context"
        return item, "drug_effect_downgraded_to_context"
    if edge.get("source_kind") in {"Symptom", "Side Effect"} and edge.get("target_kind") == "Compound":
        return None, "reversed_drug_effect_removed"
    return None, "unsupported_drug_effect_removed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a semantic-clean PrimeKG graph from an existing PrimeKG graph.")
    parser.add_argument("--input-dir", default="data/processed/stage2_primekg/primekg_graph")
    parser.add_argument("--output-dir", default="data/processed/stage2_primekg_semantic_clean/primekg_graph")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    node_rows = list(read_tsv(input_dir / "hetionet_nodes.tsv"))
    alias_rows = list(read_jsonl(input_dir / "hetionet_aliases.jsonl"))

    edges: dict[tuple[str, str, str], dict] = {}
    action_counts = Counter()
    relation_counts_before = Counter()
    relation_counts_after = Counter()
    metaedge_counts_before = Counter()
    metaedge_counts_after = Counter()
    removed_examples = []
    changed_examples = []
    for edge in read_tsv(input_dir / "hetionet_edges.tsv"):
        relation_counts_before[edge["relation"]] += 1
        metaedge_counts_before[edge.get("metaedge", "")] += 1
        cleaned, action = clean_edge(edge)
        action_counts[action] += 1
        if cleaned is None:
            if len(removed_examples) < 20:
                removed_examples.append(edge)
            continue
        key = (cleaned["source_id"], cleaned["relation"], cleaned["target_id"])
        edges.setdefault(key, cleaned)
        relation_counts_after[cleaned["relation"]] += 1
        metaedge_counts_after[cleaned.get("metaedge", "")] += 1
        if action != "unchanged" and len(changed_examples) < 20:
            changed_examples.append({"before": edge, "after": cleaned})

    edge_rows = sorted(edges.values(), key=lambda row: (row["source_id"], row["relation"], row["target_id"]))
    node_ids_in_edges = {row["source_id"] for row in edge_rows} | {row["target_id"] for row in edge_rows}
    retained_node_rows = [row for row in node_rows if row["id"] in node_ids_in_edges]
    retained_alias_rows = [row for row in alias_rows if row.get("node_id") in node_ids_in_edges]

    write_tsv(output_dir / "hetionet_nodes.tsv", retained_node_rows, NODE_FIELDS)
    write_tsv(output_dir / "hetionet_edges.tsv", edge_rows, EDGE_FIELDS)
    write_jsonl(output_dir / "hetionet_aliases.jsonl", retained_alias_rows)
    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "nodes_before": len(node_rows),
        "nodes_after": len(retained_node_rows),
        "edges_before": sum(relation_counts_before.values()),
        "edges_after": len(edge_rows),
        "aliases_before": len(alias_rows),
        "aliases_after": len(retained_alias_rows),
        "action_counts": dict(action_counts),
        "relation_counts_before": dict(relation_counts_before),
        "relation_counts_after": dict(relation_counts_after),
        "drug_effect_side_effect_edges_before": metaedge_counts_before.get("primekg:drug_effect:side effect", 0),
        "drug_effect_side_effect_edges_after": metaedge_counts_after.get("primekg:drug_effect:side effect", 0),
        "removed_examples": removed_examples,
        "changed_examples": changed_examples,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hetionet_graph_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
