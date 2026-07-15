from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

RELATION_PRIORITY = {
    "treats": 1.0,
    "palliates": 0.95,
    "causes": 0.95,
    "targets": 0.90,
    "interacts": 0.80,
    "associates": 0.75,
    "regulates": 0.70,
    "upregulates": 0.70,
    "downregulates": 0.70,
    "expresses": 0.65,
    "resembles": 0.60,
    "presents": 0.60,
    "localizes": 0.60,
    "participates": 0.55,
    "includes": 0.55,
    "contraindicates": 0.95,
    "not_presents": 0.85,
    "not_expressed": 0.75,
    "ppi": 0.55,
    "adverse_event_context": 0.45,
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


def tokens(value: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(value) if tok.lower() not in STOPWORDS}


def lexical_overlap(text: str, claim: str) -> float:
    text_tokens = tokens(text)
    claim_tokens = tokens(claim)
    if not text_tokens or not claim_tokens:
        return 0.0
    return len(text_tokens & claim_tokens) / max(len(text_tokens), 1)


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


def read_tsv(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        yield from reader


def load_graph(
    graph_dir: Path,
) -> tuple[dict[str, dict], dict[str, list[dict]], dict[tuple[str, str], list[dict]], dict[str, dict]]:
    nodes_path = graph_dir / "hetionet_nodes.tsv"
    edges_path = graph_dir / "hetionet_edges.tsv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Hetionet nodes file not found: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Hetionet edges file not found: {edges_path}")

    nodes = {
        row["id"]: {"node_id": row["id"], "name": row["name"], "kind": row["kind"]}
        for row in read_tsv(nodes_path)
    }

    adjacency: dict[str, list[dict]] = defaultdict(list)
    between: dict[tuple[str, str], list[dict]] = defaultdict(list)
    edge_lookup: dict[str, dict] = {}
    for edge_index, row in enumerate(read_tsv(edges_path)):
        edge = {
            "edge_id": f"e{edge_index}",
            "source_id": row["source_id"],
            "source_name": row["source_name"],
            "source_kind": row["source_kind"],
            "relation": row["relation"],
            "metaedge": row["metaedge"],
            "target_id": row["target_id"],
            "target_name": row["target_name"],
            "target_kind": row["target_kind"],
        }
        source_id = edge["source_id"]
        target_id = edge["target_id"]
        adjacency[source_id].append(edge)
        adjacency[target_id].append(edge)
        between[tuple(sorted([source_id, target_id]))].append(edge)
        edge_lookup[edge["edge_id"]] = edge
    return nodes, adjacency, between, edge_lookup


def edge_text(edge: dict) -> str:
    return f'{edge["source_name"]} --{edge["relation"]}--> {edge["target_name"]}'


def path_text(edges: list[dict]) -> str:
    return " ; ".join(edge_text(edge) for edge in edges)


def other_node(edge: dict, node_id: str) -> str:
    return edge["target_id"] if edge["source_id"] == node_id else edge["source_id"]


def relation_priority(edges: list[dict]) -> float:
    if not edges:
        return 0.0
    return sum(RELATION_PRIORITY.get(edge["relation"], 0.50) for edge in edges) / len(edges)


def relation_kind_compatible(edge: dict) -> bool:
    pair = frozenset([edge["source_kind"], edge["target_kind"]])
    allowed_pairs = {
        "associates": {
            frozenset(["Disease", "Gene"]),
            frozenset(["Disease"]),
            frozenset(["Gene"]),
            frozenset(["Gene", "Symptom"]),
        },
        "causes": {
            frozenset(["Compound", "Side Effect"]),
            frozenset(["Compound", "Symptom"]),
        },
        "adverse_event_context": {
            frozenset(["Compound", "Side Effect"]),
            frozenset(["Compound", "Symptom"]),
        },
        "contraindicates": {frozenset(["Compound", "Disease"])},
        "downregulates": {frozenset(["Gene"]), frozenset(["Anatomy", "Gene"]), frozenset(["Compound", "Gene"])},
        "expresses": {frozenset(["Anatomy", "Gene"])},
        "includes": {frozenset(["Compound", "Pharmacologic Class"])},
        "interacts": {frozenset(["Gene"]), frozenset(["Compound", "Gene"])},
        "localizes": {frozenset(["Anatomy", "Disease"])},
        "not_expressed": {frozenset(["Anatomy", "Gene"])},
        "not_presents": {frozenset(["Disease", "Symptom"])},
        "palliates": {frozenset(["Compound", "Disease"])},
        "participates": {
            frozenset(["Biological Process", "Gene"]),
            frozenset(["Cellular Component", "Gene"]),
            frozenset(["Gene", "Molecular Function"]),
            frozenset(["Gene", "Pathway"]),
        },
        "ppi": {frozenset(["Gene"])},
        "presents": {frozenset(["Disease", "Symptom"])},
        "regulates": {frozenset(["Gene"])},
        "resembles": {frozenset(["Compound"]), frozenset(["Disease"])},
        "targets": {frozenset(["Compound", "Gene"])},
        "treats": {frozenset(["Compound", "Disease"])},
        "upregulates": {frozenset(["Gene"]), frozenset(["Anatomy", "Gene"]), frozenset(["Compound", "Gene"])},
    }
    return pair in allowed_pairs.get(edge["relation"], set())


def path_score(
    claim: str,
    edges: list[dict],
    entity_score: float,
    path_shortness: float,
    zero_overlap_score_cap: float,
    lexical_overlap_weight: float,
) -> float:
    evidence = path_text(edges)
    overlap = lexical_overlap(evidence, claim)
    score = (
        0.45 * entity_score
        + 0.25 * relation_priority(edges)
        + lexical_overlap_weight * overlap
        + 0.10 * path_shortness
    )
    if overlap == 0:
        score = min(score, zero_overlap_score_cap)
    return round(score, 4)


def seed_scores(linked_entities: list[dict]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for entity in linked_entities:
        node_id = entity.get("node_id")
        if not node_id:
            continue
        scores[node_id] = max(scores.get(node_id, 0.0), float(entity.get("score", 0.0)))
    return scores


def edge_sort_key(edge: dict, claim: str) -> tuple[float, float, str]:
    return (
        RELATION_PRIORITY.get(edge["relation"], 0.50),
        lexical_overlap(edge_text(edge), claim),
        edge["edge_id"],
    )


def make_path_record(
    claim: str,
    edges: list[dict],
    seed_ids: list[str],
    entity_score: float,
    path_shortness: float,
    path_type: str,
    zero_overlap_score_cap: float,
    lexical_overlap_weight: float,
    alignment_seed_ids: list[str] | None = None,
) -> dict:
    node_ids: list[str] = []
    for edge in edges:
        node_ids.extend([edge["source_id"], edge["target_id"]])
    node_ids = sorted(set(node_ids))
    aligned_seed_count = len(set(alignment_seed_ids or seed_ids) & set(node_ids))
    semantic_compatible = all(relation_kind_compatible(edge) for edge in edges)
    return {
        "path_type": path_type,
        "seed_ids": seed_ids,
        "node_ids": node_ids,
        "edge_ids": [edge["edge_id"] for edge in edges],
        "relations": [edge["relation"] for edge in edges],
        "path_text": path_text(edges),
        "score": path_score(
            claim,
            edges,
            entity_score,
            path_shortness,
            zero_overlap_score_cap,
            lexical_overlap_weight,
        ),
        "lexical_overlap": round(lexical_overlap(path_text(edges), claim), 4),
        "relation_priority": round(relation_priority(edges), 4),
        "aligned_seed_count": aligned_seed_count,
        "semantic_compatible": semantic_compatible,
        "verification_eligible": semantic_compatible and aligned_seed_count >= 2,
    }


def retrieve_for_claim(
    row: dict,
    nodes: dict[str, dict],
    adjacency: dict[str, list[dict]],
    between: dict[tuple[str, str], list[dict]],
    edge_lookup: dict[str, dict],
    max_paths: int,
    max_nodes: int,
    max_edges: int,
    max_neighbors_per_seed: int,
    max_common_neighbors: int,
    max_hop: int,
    prioritize_lexical_overlap: bool,
    lexical_overlap_weight: float,
    min_entity_score: float,
    zero_overlap_score_cap: float,
    relation_aware_filter: bool,
) -> dict:
    claim = str(row.get("claim", ""))
    linked_entities = [
        entity
        for entity in row.get("linked_entities", [])
        if entity.get("node_id") in nodes
        and float(entity.get("score", 0.0)) >= min_entity_score
    ]
    scores = seed_scores(linked_entities)
    seed_ids = list(scores.keys())

    candidate_paths: list[dict] = []
    seed_top_edges: dict[str, list[dict]] = {}

    for seed_id in seed_ids:
        seed_edges = sorted(
            adjacency.get(seed_id, []),
            key=lambda edge: edge_sort_key(edge, claim),
            reverse=True,
        )[:max_neighbors_per_seed]
        seed_top_edges[seed_id] = seed_edges
        for edge in seed_edges:
            candidate_paths.append(
                make_path_record(
                    claim=claim,
                    edges=[edge],
                    seed_ids=[seed_id],
                    entity_score=scores.get(seed_id, 0.0),
                    path_shortness=1.0,
                    path_type="1-hop",
                    zero_overlap_score_cap=zero_overlap_score_cap,
                    lexical_overlap_weight=lexical_overlap_weight,
                    alignment_seed_ids=seed_ids,
                )
            )

    if max_hop >= 2 and len(seed_ids) >= 2:
        neighbor_sets = {
            seed_id: {other_node(edge, seed_id) for edge in seed_top_edges.get(seed_id, [])}
            for seed_id in seed_ids
        }
        for left_index, left_seed in enumerate(seed_ids):
            for right_seed in seed_ids[left_index + 1 :]:
                common_neighbors = sorted(neighbor_sets[left_seed] & neighbor_sets[right_seed])[:max_common_neighbors]
                for middle in common_neighbors:
                    left_edges = sorted(
                        between.get(tuple(sorted([left_seed, middle])), []),
                        key=lambda edge: edge_sort_key(edge, claim),
                        reverse=True,
                    )[:3]
                    right_edges = sorted(
                        between.get(tuple(sorted([middle, right_seed])), []),
                        key=lambda edge: edge_sort_key(edge, claim),
                        reverse=True,
                    )[:3]
                    for left_edge in left_edges:
                        for right_edge in right_edges:
                            candidate_paths.append(
                                make_path_record(
                                    claim=claim,
                                    edges=[left_edge, right_edge],
                                    seed_ids=[left_seed, right_seed],
                                    entity_score=(scores[left_seed] + scores[right_seed]) / 2,
                                    path_shortness=0.7,
                                    path_type="2-hop",
                                    zero_overlap_score_cap=zero_overlap_score_cap,
                                    lexical_overlap_weight=lexical_overlap_weight,
                                )
                            )

    if prioritize_lexical_overlap:
        candidate_paths.sort(
            key=lambda item: (item["lexical_overlap"] > 0, item["score"], item["relation_priority"]),
            reverse=True,
        )
    else:
        candidate_paths.sort(
            key=lambda item: (item["score"], item["relation_priority"]),
            reverse=True,
        )

    unfiltered_candidate_path_count = len(candidate_paths)
    if relation_aware_filter:
        candidate_paths = [path for path in candidate_paths if path["verification_eligible"]]

    selected_paths: list[dict] = []
    selected_edge_ids: set[str] = set()
    selected_node_ids: set[str] = set()
    seen_path_text: set[str] = set()

    for path in candidate_paths:
        if path["path_text"] in seen_path_text:
            continue
        next_edge_ids = selected_edge_ids | set(path["edge_ids"])
        next_node_ids = selected_node_ids | set(path["node_ids"])
        if len(next_edge_ids) > max_edges or len(next_node_ids) > max_nodes:
            continue
        selected_paths.append(path)
        selected_edge_ids = next_edge_ids
        selected_node_ids = next_node_ids
        seen_path_text.add(path["path_text"])
        if len(selected_paths) >= max_paths:
            break

    subgraph_nodes = [
        nodes[node_id]
        for node_id in sorted(selected_node_ids)
        if node_id in nodes
    ]
    subgraph_edges = [
        edge_lookup[edge_id]
        for edge_id in sorted(selected_edge_ids)
        if edge_id in edge_lookup
    ]

    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "claim": claim,
        "source": row.get("source", ""),
        "label": row.get("label"),
        "raw_label": row.get("raw_label"),
        "linked_entities": linked_entities,
        "top_evidence_paths": (
            [
                path
                for path in selected_paths
                if path["lexical_overlap"] > 0
            ][:10]
            or selected_paths[:10]
        )
        if prioritize_lexical_overlap
        else selected_paths[:10],
        "evidence_paths": selected_paths,
        "subgraph_nodes": subgraph_nodes,
        "subgraph_edges": subgraph_edges,
        "retrieval_stats": {
            "seed_count": len(seed_ids),
            "candidate_path_count": len(candidate_paths),
            "unfiltered_candidate_path_count": unfiltered_candidate_path_count,
            "selected_path_count": len(selected_paths),
            "subgraph_node_count": len(subgraph_nodes),
            "subgraph_edge_count": len(subgraph_edges),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve 1-hop and 2-hop local Hetionet subgraphs.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--max-paths", type=int, default=50)
    parser.add_argument("--max-nodes", type=int, default=300)
    parser.add_argument("--max-edges", type=int, default=1000)
    parser.add_argument("--max-neighbors-per-seed", type=int, default=50)
    parser.add_argument("--max-common-neighbors", type=int, default=50)
    parser.add_argument("--max-hop", type=int, choices=[1, 2], default=2)
    parser.add_argument("--disable-lexical-overlap-priority", action="store_true")
    parser.add_argument("--disable-lexical-overlap-scoring", action="store_true")
    parser.add_argument("--min-entity-score", type=float, default=0.70)
    parser.add_argument("--zero-overlap-score-cap", type=float, default=0.55)
    parser.add_argument("--relation-aware-filter", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    graph_dir = Path(args.graph_dir) if args.graph_dir else data_dir / "processed"
    linked_path = Path(args.input_path) if args.input_path else data_dir / "processed" / "linked_claims.jsonl"
    if not linked_path.exists():
        raise FileNotFoundError(f"Linked claims file not found: {linked_path}")

    nodes, adjacency, between, edge_lookup = load_graph(graph_dir)
    rows = [
        retrieve_for_claim(
            row=row,
            nodes=nodes,
            adjacency=adjacency,
            between=between,
            edge_lookup=edge_lookup,
            max_paths=args.max_paths,
            max_nodes=args.max_nodes,
            max_edges=args.max_edges,
            max_neighbors_per_seed=args.max_neighbors_per_seed,
            max_common_neighbors=args.max_common_neighbors,
            max_hop=args.max_hop,
            prioritize_lexical_overlap=not args.disable_lexical_overlap_priority,
            lexical_overlap_weight=0.0 if args.disable_lexical_overlap_scoring else 0.20,
            min_entity_score=args.min_entity_score,
            zero_overlap_score_cap=args.zero_overlap_score_cap,
            relation_aware_filter=args.relation_aware_filter,
        )
        for row in read_jsonl(linked_path)
    ]

    out_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "local_subgraphs.jsonl"
    write_jsonl(out_path, rows)

    pathful = sum(1 for row in rows if row["evidence_paths"])
    coverage = pathful / max(len(rows), 1)
    selected_paths = [path for row in rows for path in row["evidence_paths"]]
    zero_overlap_paths = [path for path in selected_paths if float(path.get("lexical_overlap", 0.0)) == 0.0]
    stats = {
        "input_claims": len(rows),
        "claims_with_evidence_paths": pathful,
        "evidence_path_coverage": round(coverage, 4),
        "avg_selected_paths": round(
            sum(row["retrieval_stats"]["selected_path_count"] for row in rows) / max(len(rows), 1),
            4,
        ),
        "avg_selected_path_score": round(
            sum(float(path.get("score", 0.0)) for path in selected_paths) / max(len(selected_paths), 1),
            4,
        ),
        "zero_overlap_path_count": len(zero_overlap_paths),
        "avg_zero_overlap_path_score": round(
            sum(float(path.get("score", 0.0)) for path in zero_overlap_paths) / max(len(zero_overlap_paths), 1),
            4,
        ),
        "path_type_counts": dict(Counter(path["path_type"] for row in rows for path in row["evidence_paths"])),
        "relation_counts": dict(Counter(rel for row in rows for path in row["evidence_paths"] for rel in path["relations"])),
        "claims_with_top_evidence_paths": sum(1 for row in rows if row["top_evidence_paths"]),
        "max_hop": args.max_hop,
        "prioritize_lexical_overlap": not args.disable_lexical_overlap_priority,
        "lexical_overlap_weight": 0.0 if args.disable_lexical_overlap_scoring else 0.20,
        "min_entity_score": args.min_entity_score,
        "zero_overlap_score_cap": args.zero_overlap_score_cap,
        "relation_aware_filter": args.relation_aware_filter,
        "verification_eligible_path_count": sum(
            1 for path in selected_paths if path.get("verification_eligible")
        ),
        "output": str(out_path),
    }
    stats_path = out_dir / "local_subgraph_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
