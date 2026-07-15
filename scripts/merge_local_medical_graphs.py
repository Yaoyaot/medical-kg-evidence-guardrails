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


def write_tsv(path: Path, rows, fields: list[str]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge graph directories using the local Hetionet-compatible TSV schema.")
    parser.add_argument("--graph-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    graph_dirs = [Path(path) for path in args.graph_dir]
    out_dir = Path(args.output_dir)
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}
    aliases: dict[tuple[str, str], dict] = {}

    for graph_dir in graph_dirs:
        for row in read_tsv(graph_dir / "hetionet_nodes.tsv"):
            nodes.setdefault(row["id"], row)
        for row in read_tsv(graph_dir / "hetionet_edges.tsv"):
            edges.setdefault((row["source_id"], row["relation"], row["target_id"]), row)
        for row in read_jsonl(graph_dir / "hetionet_aliases.jsonl"):
            aliases.setdefault((row["node_id"], row["normalized_alias"]), row)

    node_rows = sorted(nodes.values(), key=lambda row: row["id"])
    edge_rows = sorted(edges.values(), key=lambda row: (row["source_id"], row["relation"], row["target_id"]))
    alias_rows = sorted(aliases.values(), key=lambda row: (row["normalized_alias"], row["node_id"]))
    write_tsv(out_dir / "hetionet_nodes.tsv", node_rows, NODE_FIELDS)
    write_tsv(out_dir / "hetionet_edges.tsv", edge_rows, EDGE_FIELDS)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "hetionet_aliases.jsonl").open("w", encoding="utf-8") as f:
        for row in alias_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "graph_dirs": [str(path) for path in graph_dirs],
        "nodes": len(node_rows),
        "edges": len(edge_rows),
        "aliases": len(alias_rows),
        "node_type_counts": dict(Counter(row["kind"] for row in node_rows)),
        "relation_counts": dict(Counter(row["relation"] for row in edge_rows)),
    }
    (out_dir / "hetionet_graph_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
