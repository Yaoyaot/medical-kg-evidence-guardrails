from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def read_tsv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def alias_row(alias: str, node_id: str, kind: str, source: str, external_id: str) -> dict | None:
    normalized = normalize_text(alias)
    if not normalized:
        return None
    return {
        "alias": alias.strip(),
        "normalized_alias": normalized,
        "node_id": node_id,
        "kind": kind,
        "alias_source": source,
        "external_id": external_id,
    }


def load_graph_nodes(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in read_tsv(path)}


def hgnc_aliases(path: Path, graph_nodes: dict[str, dict]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = payload.get("response", {}).get("docs", [])
    rows = []
    fields = ["symbol", "name", "alias_symbol", "alias_name", "prev_symbol", "prev_name"]
    for doc in docs:
        entrez_ids = values(doc.get("entrez_id"))
        if not entrez_ids:
            continue
        node_id = f"Gene::{entrez_ids[0]}"
        if node_id not in graph_nodes:
            continue
        external_id = str(doc.get("hgnc_id", ""))
        for field in fields:
            for alias in values(doc.get(field)):
                row = alias_row(alias, node_id, "Gene", f"hgnc:{field}", external_id)
                if row:
                    rows.append(row)
    return rows


def mondo_aliases(path: Path, graph_nodes: dict[str, dict]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    graphs = payload.get("graphs", [])
    rows = []
    for graph in graphs:
        for node in graph.get("nodes", []):
            meta = node.get("meta", {})
            xrefs = meta.get("xrefs", [])
            doids = []
            for xref in xrefs:
                value = str(xref.get("val", "") if isinstance(xref, dict) else xref)
                match = re.search(r"DOID[:_](\d+)", value, flags=re.IGNORECASE)
                if match:
                    doids.append(match.group(1))
            if not doids:
                continue
            aliases = [str(node.get("lbl", "")).strip()]
            for synonym in meta.get("synonyms", []):
                aliases.append(str(synonym.get("val", "") if isinstance(synonym, dict) else synonym).strip())
            for doid in sorted(set(doids)):
                node_id = f"Disease::DOID:{doid}"
                if node_id not in graph_nodes:
                    continue
                for alias in aliases:
                    if not alias:
                        continue
                    row = alias_row(alias, node_id, "Disease", "mondo", str(node.get("id", "")))
                    if row:
                        rows.append(row)
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    output = {}
    for row in rows:
        key = (row["node_id"], row["normalized_alias"])
        output.setdefault(key, row)
    return sorted(output.values(), key=lambda row: (row["node_id"], row["normalized_alias"], row["alias_source"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HGNC and Mondo aliases aligned to a local Hetionet graph.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    graph_dir = Path(args.graph_dir) if args.graph_dir else data_dir / "processed" / "stage1_umls" / "expanded_graph"
    out_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed" / "stage1_umls"
    terminology_dir = data_dir / "raw" / "terminology"
    graph_nodes = load_graph_nodes(graph_dir / "hetionet_nodes.tsv")

    rows = dedupe(
        hgnc_aliases(terminology_dir / "hgnc_complete_set.json", graph_nodes)
        + mondo_aliases(terminology_dir / "mondo.json", graph_nodes)
    )
    out_path = out_dir / "open_aliases.jsonl"
    write_jsonl(out_path, rows)
    stats = {
        "aliases": len(rows),
        "matched_nodes": len({row["node_id"] for row in rows}),
        "alias_source_counts": dict(Counter(row["alias_source"].split(":", 1)[0] for row in rows)),
        "kind_counts": dict(Counter(row["kind"] for row in rows)),
        "output": str(out_path),
    }
    stats_path = out_dir / "open_alias_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
