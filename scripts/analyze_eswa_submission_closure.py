from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from detect_claim_qualifiers import detect_qualifiers  # noqa: E402
from link_claim_entities import link_claim, load_aliases  # noqa: E402
from retrieve_local_subgraphs import load_graph, retrieve_for_claim  # noqa: E402
from strict_relation_alignment import (  # noqa: E402
    automatic_support_allowed,
    extract_claim_predicate_families,
)


SEED = 20260618
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.20)
LAMBDAS = (0.01, 0.05, 0.10, 0.20, 0.50)
METHODS = {
    "random": "risk_random",
    "confidence": "risk_confidence",
    "rule": "risk_rule",
    "learned": "risk_full_without_dataset_source",
    "oracle": "risk_oracle",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile_interval(values: list[float]) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def clone_component_sample(rows: list[dict], rng: random.Random) -> list[dict]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[str(row["pair_group_id"])].append(row)
    keys = sorted(by_group)
    sample: list[dict] = []
    for draw, key in enumerate(rng.choices(keys, k=len(keys))):
        for local, source in enumerate(by_group[key]):
            sample.append({**source, "id": f"b{draw}-{local}-{source['id']}"})
    return sample


def random_score(identifier: str, iteration: int) -> float:
    digest = hashlib.sha256(f"{SEED}:{iteration}:{identifier}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / 16**16


def select_fold_budget(rows: list[dict], score_field: str, budget: float, random_iteration: int | None = None) -> set[str]:
    selected: set[str] = set()
    for fold in range(5):
        fold_rows = [row for row in rows if int(row["outer_fold"]) == fold]
        eligible = [row for row in fold_rows if row["candidate_label"] == "SUPPORT"]
        count = min(len(eligible), math.ceil(budget * len(fold_rows)))
        if random_iteration is None:
            ranked = sorted(eligible, key=lambda row: (-float(row.get(score_field, 0.0)), row["id"]))
        else:
            ranked = sorted(eligible, key=lambda row: (-random_score(row["id"], random_iteration), row["id"]))
        selected.update(row["id"] for row in ranked[:count])
    return selected


def captured_counts(rows: list[dict], reviewed: set[str]) -> tuple[int, int, int]:
    reviewed_rows = [row for row in rows if row["id"] in reviewed]
    captured = sum(row["candidate_label"] == "SUPPORT" and row["gold_label"] != "SUPPORT" for row in reviewed_rows)
    correct_support = sum(row["candidate_label"] == "SUPPORT" and row["gold_label"] == "SUPPORT" for row in reviewed_rows)
    return int(captured), len(reviewed_rows), int(correct_support)


def cost_utility(rows: list[dict], iterations: int, output: Path) -> dict:
    point_rows: list[dict] = []
    for method, field in METHODS.items():
        for budget in BUDGETS:
            if method == "random":
                simulations = []
                for simulation in range(1000):
                    reviewed = select_fold_budget(rows, field, budget, random_iteration=simulation)
                    simulations.append(captured_counts(rows, reviewed))
                captured = float(np.mean([item[0] for item in simulations]))
                reviewed_count = float(np.mean([item[1] for item in simulations]))
                correct_support = float(np.mean([item[2] for item in simulations]))
            else:
                reviewed = select_fold_budget(rows, field, budget)
                captured, reviewed_count, correct_support = captured_counts(rows, reviewed)
            for cost_ratio in LAMBDAS:
                point_rows.append({
                    "method": method,
                    "budget": budget,
                    "reviewed_count": reviewed_count,
                    "captured_false_supports": captured,
                    "reviewed_correct_supports": correct_support,
                    "review_cost_ratio_lambda": cost_ratio,
                    "normalized_net_gain": (captured - cost_ratio * reviewed_count) / len(rows),
                })

    rng = random.Random(SEED)
    distributions: dict[tuple[str, float, float], list[float]] = defaultdict(list)
    for iteration in range(iterations):
        sample = clone_component_sample(rows, rng)
        for method, field in METHODS.items():
            for budget in BUDGETS:
                reviewed = select_fold_budget(
                    sample,
                    field,
                    budget,
                    random_iteration=iteration if method == "random" else None,
                )
                captured, reviewed_count, _ = captured_counts(sample, reviewed)
                for cost_ratio in LAMBDAS:
                    distributions[(method, budget, cost_ratio)].append(
                        (captured - cost_ratio * reviewed_count) / len(sample)
                    )

    bootstrap_rows: list[dict] = []
    for key, values in sorted(distributions.items()):
        method, budget, cost_ratio = key
        low, high = percentile_interval(values)
        bootstrap_rows.append({
            "method": method,
            "budget": budget,
            "review_cost_ratio_lambda": cost_ratio,
            "bootstrap_iterations": iterations,
            "mean_normalized_net_gain": statistics.fmean(values),
            "ci_low": low,
            "ci_high": high,
        })

    write_csv(output / "cost_utility_point_estimates.csv", point_rows)
    write_csv(output / "cost_utility_group_bootstrap.csv", bootstrap_rows)
    paired_rows = []
    for budget in BUDGETS:
        for cost_ratio in LAMBDAS:
            learned = distributions[("learned", budget, cost_ratio)]
            confidence = distributions[("confidence", budget, cost_ratio)]
            differences = [left - right for left, right in zip(learned, confidence)]
            low, high = percentile_interval(differences)
            paired_rows.append({
                "comparison": "learned_minus_confidence",
                "budget": budget,
                "review_cost_ratio_lambda": cost_ratio,
                "bootstrap_iterations": iterations,
                "mean_difference": statistics.fmean(differences),
                "ci_low": low,
                "ci_high": high,
            })
    write_csv(output / "cost_utility_paired_differences.csv", paired_rows)
    return {
        "formula": "(captured_false_supports - lambda * reviewed_count) / N",
        "budgets": list(BUDGETS),
        "review_cost_ratios": list(LAMBDAS),
        "bootstrap_iterations": iterations,
        "bootstrap_unit": "claim/source/near-duplicate connected component",
        "ranking_uses_gold": False,
        "gold_use": "evaluation of captured false SUPPORT cases only",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kg_metadata(output: Path) -> dict:
    hetionet_stats = read_json(ROOT / "data/processed/stage1_umls/expanded_graph/hetionet_graph_stats.json")
    primekg_raw = read_json(ROOT / "data/processed/stage2_primekg/primekg_raw_stats.json")
    primekg_filtered = read_json(ROOT / "data/processed/stage2_primekg/primekg_graph/hetionet_graph_stats.json")
    primekg_clean = read_json(ROOT / "data/processed/stage2_primekg_semantic_clean/primekg_graph/hetionet_graph_stats.json")
    fused_clean = read_json(ROOT / "data/processed/stage2_primekg_semantic_clean/fused_graph/hetionet_graph_stats.json")
    umls = read_json(ROOT / "data/processed/stage1_umls/umls_manifest.json")
    mondo_path = ROOT / "data/raw/terminology/mondo.json"
    hgnc_path = ROOT / "data/raw/terminology/hgnc_complete_set.json"
    primekg_path = ROOT / "data/raw/primekg/kg.csv"
    metadata = {
        "resources": [
            {"resource": "Hetionet", "version_or_date": "v1.0", "role": "base biomedical graph"},
            {"resource": "PrimeKG", "version_or_date": "Harvard Dataverse datafile 6180620; downloaded 2026-05-30", "role": "filtered relation enrichment"},
            {"resource": "Mondo", "version_or_date": "2026-05-05", "role": "disease terminology"},
            {"resource": "HGNC", "version_or_date": "downloaded 2026-05-30", "role": "gene terminology"},
            {"resource": "UMLS", "version_or_date": ", ".join(umls["umls_content_version"]), "role": "licensed terminology normalization"},
        ],
        "counts": {
            "hetionet_expanded": {"nodes": hetionet_stats["nodes"], "edges": hetionet_stats["edges"], "aliases": hetionet_stats["aliases"]},
            "primekg_raw": {"rows": primekg_raw["rows"], "unique_nodes": primekg_raw["unique_node_indices"]},
            "primekg_filtered": {"nodes": primekg_filtered["nodes"], "edges": primekg_filtered["edges"], "aliases": primekg_filtered["aliases"]},
            "primekg_cleaned": {"nodes": primekg_clean["nodes_after"], "edges": primekg_clean["edges_after"], "aliases": primekg_clean["aliases_after"]},
            "fused_cleaned": {"nodes": fused_clean["nodes"], "edges": fused_clean["edges"], "aliases": fused_clean["aliases"]},
        },
        "semantic_cleaning": primekg_clean["action_counts"],
        "retrieval_parameters": {
            "max_hop": 2,
            "max_paths": 50,
            "max_prompt_paths": 10,
            "max_nodes": 300,
            "max_edges": 1000,
            "max_neighbors_per_seed": 50,
            "max_common_neighbors": 50,
            "max_edge_combinations_per_common_neighbor": 9,
            "min_entity_score": 0.70,
        },
        "provenance_policy": {
            "surviving_edge_field": "metaedge",
            "cross_graph_duplicate_key": ["source_id", "relation", "target_id"],
            "duplicate_resolution": "first-seen edge retained",
            "multi_source_provenance_union": False,
        },
        "raw_file_provenance": {
            "primekg_sha256": sha256(primekg_path),
            "primekg_bytes": primekg_path.stat().st_size,
            "mondo_sha256": sha256(mondo_path),
            "mondo_bytes": mondo_path.stat().st_size,
            "hgnc_sha256": sha256(hgnc_path),
            "hgnc_bytes": hgnc_path.stat().st_size,
        },
    }
    (output / "kg_resource_and_build_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def quantiles(values: list[float]) -> dict:
    return {
        "mean_ms": statistics.fmean(values) * 1000,
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": float(np.percentile(values, 95)) * 1000,
        "min_ms": min(values) * 1000,
        "max_ms": max(values) * 1000,
    }


def load_combined_aliases(paths: list[Path]) -> tuple[list[dict], dict[str, set[int]]]:
    aliases: list[dict] = []
    inverted: dict[str, set[int]] = defaultdict(set)
    for path in paths:
        loaded, _, _ = load_aliases(path, min_alias_length=3, disable_numeric_aliases=True)
        offset = len(aliases)
        aliases.extend(loaded)
        for index, alias in enumerate(loaded, start=offset):
            for token in alias["alias_tokens"]:
                inverted[token].add(index)
    return aliases, inverted


def audit_claim(row: dict, between: dict[tuple[str, str], list[dict]]) -> dict:
    entities = [item for item in row.get("linked_entities", []) if float(item.get("score", 0.0)) >= 0.70]
    seed_ids = sorted({item["node_id"] for item in entities})
    families = extract_claim_predicate_families(str(row.get("claim", "")))
    qualifiers = detect_qualifiers(str(row.get("claim", "")))
    direct = []
    for left_index, left in enumerate(seed_ids):
        for right in seed_ids[left_index + 1:]:
            for edge in between.get(tuple(sorted((left, right))), []):
                direct.append(automatic_support_allowed(families, edge["relation"]))
    strict = any(direct) and not qualifiers
    return {"direct_edges": len(direct), "strict_candidate": strict, "qualifier_count": len(qualifiers)}


def runtime_benchmark(output: Path, repeats: int) -> dict:
    graph_dir = ROOT / "data/processed/stage2_primekg_semantic_clean/fused_graph"
    alias_paths = [
        graph_dir / "hetionet_aliases.jsonl",
        ROOT / "data/processed/stage1_umls/open_aliases.jsonl",
        ROOT / "data/private/stage1_umls/umls_aliases.jsonl",
    ]
    missing = [str(path) for path in alias_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing benchmark alias resources: {missing}")

    formal_ids = {row["id"] for row in read_csv(ROOT / "data/processed/stage7_hierarchical_scorer/formal600/eswa_formal600_manifest.csv")}
    source_rows = []
    for row in read_jsonl(ROOT / "data/processed/stage2_primekg_semantic_clean/variants/primekg_semantic_clean_relation_aware/local_subgraphs.jsonl"):
        if row["id"] in formal_ids:
            source_rows.append({key: row.get(key) for key in ("id", "dataset", "split", "claim", "source", "label", "raw_label")})
    source_rows.sort(key=lambda row: row["id"])
    if len(source_rows) != 600:
        raise RuntimeError(f"Expected 600 Formal600 source rows, found {len(source_rows)}")

    alias_load_times = []
    graph_load_times = []
    aliases = inverted = None
    nodes = adjacency = between = edge_lookup = None
    for _ in range(repeats):
        start = time.perf_counter()
        aliases, inverted = load_combined_aliases(alias_paths)
        alias_load_times.append(time.perf_counter() - start)
        del aliases, inverted
        gc.collect()

        start = time.perf_counter()
        nodes, adjacency, between, edge_lookup = load_graph(graph_dir)
        graph_load_times.append(time.perf_counter() - start)
        del nodes, adjacency, between, edge_lookup
        gc.collect()

    aliases, inverted = load_combined_aliases(alias_paths)
    nodes, adjacency, between, edge_lookup = load_graph(graph_dir)
    detail_rows = []
    for repeat in range(repeats):
        for source in source_rows:
            total_start = time.perf_counter()
            start = time.perf_counter()
            linked = link_claim(source, aliases, inverted, top_k=5, token_threshold=0.75)
            link_seconds = time.perf_counter() - start

            start = time.perf_counter()
            retrieved = retrieve_for_claim(
                row=linked,
                nodes=nodes,
                adjacency=adjacency,
                between=between,
                edge_lookup=edge_lookup,
                max_paths=50,
                max_nodes=300,
                max_edges=1000,
                max_neighbors_per_seed=50,
                max_common_neighbors=50,
                max_hop=2,
                prioritize_lexical_overlap=True,
                lexical_overlap_weight=0.20,
                min_entity_score=0.70,
                zero_overlap_score_cap=0.55,
                relation_aware_filter=True,
            )
            retrieval_seconds = time.perf_counter() - start

            start = time.perf_counter()
            audited = audit_claim(retrieved, between)
            audit_seconds = time.perf_counter() - start
            total_seconds = time.perf_counter() - total_start
            detail_rows.append({
                "repeat": repeat + 1,
                "id": source["id"],
                "entity_link_seconds": link_seconds,
                "local_subgraph_seconds": retrieval_seconds,
                "semantic_audit_seconds": audit_seconds,
                "end_to_end_kg_audit_seconds": total_seconds,
                "linked_entity_count": len(linked["linked_entities"]),
                "selected_path_count": retrieved["retrieval_stats"]["selected_path_count"],
                "candidate_path_count": retrieved["retrieval_stats"]["candidate_path_count"],
                **audited,
            })

    write_csv(output / "kg_runtime_per_claim.csv", detail_rows)
    summary_rows = []
    for stage, field in (
        ("entity_link", "entity_link_seconds"),
        ("local_subgraph_1_2_hop", "local_subgraph_seconds"),
        ("semantic_audit", "semantic_audit_seconds"),
        ("end_to_end_kg_audit", "end_to_end_kg_audit_seconds"),
    ):
        summary_rows.append({"stage": stage, "observations": len(detail_rows), **quantiles([float(row[field]) for row in detail_rows])})
    summary_rows.append({"stage": "alias_index_load", "observations": repeats, **quantiles(alias_load_times)})
    summary_rows.append({"stage": "graph_load", "observations": repeats, **quantiles(graph_load_times)})
    write_csv(output / "kg_runtime_summary.csv", summary_rows)

    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu = platform.processor() or platform.uname().processor
    except Exception:
        memory_gb = None
        if os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                memory_gb = status.total_physical / (1024**3)
        cpu = platform.processor() or "not reported"
    environment = {
        "benchmark_claims": len(source_rows),
        "repeats": repeats,
        "claim_observations": len(detail_rows),
        "fixed_order": "lexicographic claim id",
        "os_cache_cleared_between_loads": False,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "memory_gb": memory_gb,
        "threading": "single-process Python; no explicit worker pool",
        "network_or_llm_calls": False,
        "alias_count_loaded": len(aliases),
        "graph_node_count_loaded": len(nodes),
        "graph_edge_count_loaded": len(edge_lookup),
    }
    (output / "kg_runtime_environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"summary": summary_rows, "environment": environment}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen Stage 10 ESWA submission-closure analyses.")
    parser.add_argument("--output-dir", default="data/processed/stage10_eswa_submission_closure")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--runtime-repeats", type=int, default=3)
    parser.add_argument("--skip-runtime", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    risk_rows = list(read_jsonl(ROOT / "data/processed/stage9_eswa_major_revision/formal600_crossfit/formal600_crossfit_risk_scores.jsonl"))
    if len(risk_rows) != 600:
        raise RuntimeError(f"Expected 600 frozen OOF rows, found {len(risk_rows)}")
    if any("pair_group_id" not in row or "outer_fold" not in row for row in risk_rows):
        raise RuntimeError("Frozen OOF rows lack grouped-crossfit identifiers")

    started = time.time()
    utility = cost_utility(risk_rows, args.bootstrap_iterations, output)
    metadata = kg_metadata(output)
    if args.skip_runtime:
        runtime = {
            "summary": read_csv(output / "kg_runtime_summary.csv"),
            "environment": read_json(output / "kg_runtime_environment.json"),
        }
        if runtime["environment"].get("memory_gb") is None and os.name == "nt":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong), ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                runtime["environment"]["memory_gb"] = status.total_physical / (1024**3)
                (output / "kg_runtime_environment.json").write_text(
                    json.dumps(runtime["environment"], ensure_ascii=False, indent=2), encoding="utf-8"
                )
    else:
        runtime = runtime_benchmark(output, args.runtime_repeats)
    manifest = {
        "stage": "stage10_eswa_submission_closure",
        "seed": SEED,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "frozen_inputs": {
            "risk_scores": "data/processed/stage9_eswa_major_revision/formal600_crossfit/formal600_crossfit_risk_scores.jsonl",
            "formal600_rows": 600,
            "pubmedqa_used_for_risk_adjustment": False,
            "model_retraining": False,
            "llm_api_calls": False,
        },
        "cost_utility": utility,
        "kg_fused_counts": metadata["counts"]["fused_cleaned"],
        "runtime_environment": runtime["environment"],
        "outputs": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "stage10_submission_closure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
