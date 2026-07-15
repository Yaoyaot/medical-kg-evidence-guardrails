from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from strict_relation_alignment import (
    CONTEXT_ONLY_RELATIONS,
    automatic_support_allowed,
    extract_claim_predicate_families,
    predicate_aligned,
    relation_family,
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def retained_entities(row: dict, min_score: float) -> list[dict]:
    by_node = {}
    for entity in row.get("linked_entities") or []:
        node_id = str(entity.get("node_id") or "")
        score = float(entity.get("score", 0.0))
        if node_id and score >= min_score and score > float(by_node.get(node_id, {}).get("score", -1)):
            by_node[node_id] = entity
    return list(by_node.values())


def direct_path_record(edge: dict, claim_families: list[str], entities_by_id: dict[str, dict]) -> dict:
    relation = edge["relation"]
    left = entities_by_id[edge["source_id"]]
    right = entities_by_id[edge["target_id"]]
    aligned = predicate_aligned(claim_families, relation)
    if aligned:
        tier = "DIRECT_PREDICATE_MATCH"
    elif claim_families == ["UNRESOLVED"]:
        tier = "DIRECT_RELATION_UNRESOLVED"
    else:
        tier = "DIRECT_PREDICATE_MISMATCH"
    entity_score = (float(left.get("score", 0.0)) + float(right.get("score", 0.0))) / 2
    score = 0.55 * entity_score + 0.25 * (1.0 if aligned else 0.0) + 0.20 * (0.0 if relation in CONTEXT_ONLY_RELATIONS else 1.0)
    return {
        "path_type": "1-hop",
        "evidence_tier": tier,
        "endpoint_aligned": True,
        "claim_predicate_families": claim_families,
        "kg_relation_family": relation_family(relation),
        "predicate_aligned": aligned,
        "automatic_support_allowed": automatic_support_allowed(claim_families, relation),
        "path_text": f"{edge['source_name']} --{relation}--> {edge['target_name']}",
        "relations": [relation],
        "node_ids": [edge["source_id"], edge["target_id"]],
        "linked_entities": [left, right],
        "score": round(score, 4),
    }


def context_path_record(path: dict, claim_families: list[str]) -> dict:
    return {
        "path_type": path.get("path_type", "2-hop"),
        "evidence_tier": "TWO_HOP_CONTEXT",
        "endpoint_aligned": int(path.get("aligned_seed_count", 0)) >= 2,
        "claim_predicate_families": claim_families,
        "kg_relation_family": "CONTEXT_ONLY",
        "predicate_aligned": False,
        "automatic_support_allowed": False,
        "path_text": path.get("path_text"),
        "relations": path.get("relations") or [],
        "node_ids": path.get("node_ids") or [],
        "linked_entities": [],
        "score": float(path.get("score", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build strict direct-edge KG evidence tiers.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--min-entity-score", type=float, default=0.70)
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--linked-path", default=None)
    parser.add_argument("--subgraphs-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variant", default="primekg_fusion_relation_aware")
    args = parser.parse_args()

    processed = Path(args.data_dir) / "processed"
    linked_path = Path(args.linked_path) if args.linked_path else processed / "stage2_primekg/fused_linking/linked_claims.jsonl"
    graph_path = Path(args.graph_path) if args.graph_path else processed / "stage2_primekg/fused_graph/hetionet_edges.tsv"
    old_subgraphs_path = Path(args.subgraphs_path) if args.subgraphs_path else processed / "stage2_primekg/variants/primekg_fusion_relation_aware/local_subgraphs.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else processed / "stage3_strict_verifier"

    linked_rows = list(read_jsonl(linked_path))
    old_subgraphs = {row["id"]: row for row in read_jsonl(old_subgraphs_path)}
    entities_by_claim = {}
    global_seed_ids = set()
    for row in linked_rows:
        entities = retained_entities(row, args.min_entity_score)
        entities_by_claim[row["id"]] = entities
        global_seed_ids.update(entity["node_id"] for entity in entities)

    direct_edges_by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    scanned_edges = 0
    retained_direct_edges = 0
    with graph_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for edge in reader:
            scanned_edges += 1
            source_id = edge["source_id"]
            target_id = edge["target_id"]
            if source_id == target_id or source_id not in global_seed_ids or target_id not in global_seed_ids:
                continue
            direct_edges_by_pair[tuple(sorted([source_id, target_id]))].append(edge)
            retained_direct_edges += 1

    output_rows = []
    tier_counts = Counter()
    for row in linked_rows:
        claim_id = row["id"]
        entities = entities_by_claim[claim_id]
        entity_by_id = {item["node_id"]: item for item in entities}
        seed_ids = sorted(entity_by_id)
        families = extract_claim_predicate_families(str(row.get("claim") or ""))
        direct_paths = []
        for left_index, left in enumerate(seed_ids):
            for right in seed_ids[left_index + 1 :]:
                for edge in direct_edges_by_pair.get(tuple(sorted([left, right])), []):
                    direct_paths.append(direct_path_record(edge, families, entity_by_id))
        direct_paths = sorted(
            direct_paths,
            key=lambda item: (
                item["automatic_support_allowed"],
                item["predicate_aligned"],
                item["score"],
                item["path_text"],
            ),
            reverse=True,
        )
        old_paths = (old_subgraphs.get(claim_id, {}).get("top_evidence_paths") or [])
        contexts = [
            context_path_record(path, families)
            for path in old_paths
            if path.get("path_type") == "2-hop"
        ][:20]
        if direct_paths:
            primary_tier = direct_paths[0]["evidence_tier"]
        elif contexts:
            primary_tier = "TWO_HOP_CONTEXT"
        elif len(seed_ids) == 1:
            primary_tier = "SINGLE_ENTITY_CONTEXT"
        else:
            primary_tier = "NO_KG_GROUNDING"
        tier_counts[primary_tier] += 1
        output_rows.append(
            {
                "id": claim_id,
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "claim": row.get("claim"),
                "source": row.get("source"),
                "label": row.get("label"),
                "raw_label": row.get("raw_label"),
                "claim_predicate_families": families,
                "linked_entities": entities,
                "direct_paths": direct_paths,
                "two_hop_context_paths": contexts,
                "primary_evidence_tier": primary_tier,
                "evidence_stats": {
                    "seed_count": len(seed_ids),
                    "direct_path_count": len(direct_paths),
                    "predicate_aligned_direct_path_count": sum(item["automatic_support_allowed"] for item in direct_paths),
                    "two_hop_context_count": len(contexts),
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "strict_kg_evidence.jsonl"
    write_jsonl(output_path, output_rows)
    total = len(output_rows)
    linked_claims = sum(bool(row["linked_entities"]) for row in output_rows)
    any_path_claims = sum(bool(row["direct_paths"] or row["two_hop_context_paths"]) for row in output_rows)
    direct_edge_claims = sum(bool(row["direct_paths"]) for row in output_rows)
    predicate_aligned_claims = sum(any(path["automatic_support_allowed"] for path in row["direct_paths"]) for row in output_rows)
    stats = {
        "variant": args.variant,
        "input_claims": total,
        "min_entity_score": args.min_entity_score,
        "graph_path": str(graph_path),
        "linked_path": str(linked_path),
        "subgraphs_path": str(old_subgraphs_path),
        "scanned_graph_edges": scanned_edges,
        "retained_seed_to_seed_graph_edges": retained_direct_edges,
        "entity_linking_claims": linked_claims,
        "any_path_claims": any_path_claims,
        "direct_edge_claims": direct_edge_claims,
        "predicate_aligned_claims": predicate_aligned_claims,
        "entity_linking_coverage": round(linked_claims / total, 4),
        "any_path_coverage": round(any_path_claims / total, 4),
        "direct_edge_coverage": round(direct_edge_claims / total, 4),
        "predicate_aligned_coverage": round(predicate_aligned_claims / total, 4),
        "primary_evidence_tier_counts": dict(tier_counts),
        "output": str(output_path),
    }
    (output_dir / "evidence_tier_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
