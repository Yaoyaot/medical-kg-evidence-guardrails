from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from detect_claim_qualifiers import detect_qualifiers
from repo_paths import find_repo_root


SEED = 20260618
LABELS = ("SUPPORT", "REFUTE", "UNCERTAIN")
N_PER_LABEL = 200
N_REPLICATES = 10
NEAR_DUPLICATE_THRESHOLD = 0.90

ROOT = find_repo_root()
OUTPUT = ROOT / "data/processed/stage14_alternative_balanced_samples"
COMPACT_INPUT = OUTPUT / "full_pool_structural_features.csv"


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_frozen_strict_file(explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else ROOT / explicit
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = [
        ROOT / "data/processed/stage2_primekg_semantic_clean/strict_verifier/strict_kg_evidence.jsonl",
        ROOT.parent.parent / "data/processed/stage2_primekg_semantic_clean/strict_verifier/strict_kg_evidence.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "The compact structural input is absent and the frozen 6,454-row strict-evidence file was not found."
    )


def build_compact_input(strict_evidence: Path | None = None) -> list[dict]:
    source_path = find_frozen_strict_file(strict_evidence)
    rows: list[dict] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            claim = str(item.get("claim", ""))
            source = str(item.get("source", ""))
            direct = list(item.get("direct_paths") or [])
            two_hop = list(item.get("two_hop_context_paths") or [])
            aligned = any(bool(path.get("predicate_aligned")) for path in direct)
            qualifiers = detect_qualifiers(claim)
            rows.append(
                {
                    "id": item["id"],
                    "dataset": item.get("dataset", ""),
                    "gold_label": item.get("label", ""),
                    "normalized_claim": normalize(claim),
                    "normalized_source_sha256": sha256_text(normalize(source)),
                    "entity_linked": int(bool(item.get("linked_entities"))),
                    "has_local_path": int(bool(direct or two_hop)),
                    "has_direct_edge": int(bool(direct)),
                    "has_predicate_aligned_direct": int(aligned),
                    "has_qualifier_compatible_direct": int(aligned and not qualifiers),
                }
            )
    if len(rows) != 6454 or len({row["id"] for row in rows}) != 6454:
        raise ValueError(f"Expected 6,454 unique frozen rows, found {len(rows)}")
    write_csv(COMPACT_INPUT, rows)
    provenance = {
        "source_logical_id": "frozen_stage2_relation_aware_strict_evidence_v1",
        "source_repository_path": "data/processed/stage2_primekg_semantic_clean/strict_verifier/strict_kg_evidence.jsonl",
        "source_location_class": "upstream full workspace input; not included in the compact anonymous package",
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "compact_file": COMPACT_INPUT.relative_to(ROOT).as_posix(),
        "compact_sha256": hashlib.sha256(COMPACT_INPUT.read_bytes()).hexdigest(),
        "records": len(rows),
        "created_without_api_calls": True,
    }
    (OUTPUT / "full_pool_structural_features_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def union_exact(values: list[str], union: UnionFind) -> None:
    first: dict[str, int] = {}
    for index, value in enumerate(values):
        if not value:
            continue
        if value in first:
            union.union(index, first[value])
        else:
            first[value] = index


def component_stats(rows: list[dict]) -> dict:
    claims = [row["normalized_claim"] for row in rows]
    sources = [row["normalized_source_sha256"] for row in rows]
    union = UnionFind(len(rows))
    union_exact(claims, union)
    union_exact(sources, union)
    matrix = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), lowercase=False, norm="l2"
    ).fit_transform(claims)
    similarities = (matrix @ matrix.T).tocoo()
    for left, right, score in zip(similarities.row, similarities.col, similarities.data):
        if left < right and score >= NEAR_DUPLICATE_THRESHOLD and claims[left] != claims[right]:
            union.union(int(left), int(right))
    sizes = Counter(union.find(index) for index in range(len(rows)))
    return {
        "components": len(sizes),
        "largest_component_records": max(sizes.values()),
        "largest_component_share": max(sizes.values()) / len(rows),
        "unique_normalized_claims": len(set(claims)),
        "unique_source_hashes": len(set(sources)),
    }


def select_by_hash(rows: list[dict], seed: int) -> list[dict]:
    selected: list[dict] = []
    for label in LABELS:
        candidates = [row for row in rows if row["gold_label"] == label]
        ranked = sorted(
            candidates,
            key=lambda row: (sha256_text(f"{seed}:{row['id']}"), row["id"]),
        )
        if len(ranked) < N_PER_LABEL:
            raise ValueError(f"Insufficient {label} rows: {len(ranked)}")
        selected.extend(ranked[:N_PER_LABEL])
    return selected


def select_original(rows: list[dict]) -> list[dict]:
    selected: list[dict] = []
    for label in LABELS:
        selected.extend(sorted((row for row in rows if row["gold_label"] == label), key=lambda row: row["id"])[:N_PER_LABEL])
    return selected


def summarize(sample_id: str, frame: str, seed: str, rows: list[dict]) -> dict:
    if len(rows) != 600 or Counter(row["gold_label"] for row in rows) != Counter({label: 200 for label in LABELS}):
        raise ValueError(f"Invalid balanced sample {sample_id}")
    datasets = Counter(row["dataset"] for row in rows)
    components = component_stats(rows)
    summary = {
        "sample_id": sample_id,
        "sampling_frame": frame,
        "selection_seed": seed,
        "records": len(rows),
        "support": 200,
        "refute": 200,
        "uncertain": 200,
        "pubhealth": datasets.get("pubhealth", 0),
        "healthver": datasets.get("healthver", 0),
        "medaesqa": datasets.get("medaesqa", 0),
        "scifact": datasets.get("scifact", 0),
        **components,
    }
    for field in (
        "entity_linked",
        "has_local_path",
        "has_direct_edge",
        "has_predicate_aligned_direct",
        "has_qualifier_compatible_direct",
    ):
        count = sum(int(row[field]) for row in rows)
        summary[f"{field}_count"] = count
        summary[f"{field}_rate"] = count / len(rows)
    return summary


def interval_text(values: list[float], decimals: int = 1) -> str:
    median = float(np.median(values))
    low, high = min(values), max(values)
    return f"{median:.{decimals}f} [{low:.{decimals}f}-{high:.{decimals}f}]"


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for frame in ("full_6454_pool", "matched_source_frame"):
        selected = [row for row in rows if row["sampling_frame"] == frame]
        output.append(
            {
                "sampling_frame": frame,
                "replicates": len(selected),
                "dataset_composition_median_range": "; ".join(
                    f"{name} {interval_text([float(row[name]) for row in selected], 0)}"
                    for name in ("pubhealth", "healthver", "medaesqa", "scifact")
                ),
                "components_median_range": interval_text([float(row["components"]) for row in selected], 0),
                "largest_component_records_median_range": interval_text([float(row["largest_component_records"]) for row in selected], 0),
                "largest_component_share_pct_median_range": interval_text([100 * float(row["largest_component_share"]) for row in selected], 1),
                "entity_linked_pct_median_range": interval_text([100 * float(row["entity_linked_rate"]) for row in selected], 1),
                "local_path_pct_median_range": interval_text([100 * float(row["has_local_path_rate"]) for row in selected], 1),
                "direct_edge_pct_median_range": interval_text([100 * float(row["has_direct_edge_rate"]) for row in selected], 1),
                "predicate_aligned_pct_median_range": interval_text([100 * float(row["has_predicate_aligned_direct_rate"]) for row in selected], 2),
                "qualifier_compatible_count_median_range": interval_text([float(row["has_qualifier_compatible_direct_count"]) for row in selected], 0),
            }
        )
    return output


def main() -> None:
    global OUTPUT, COMPACT_INPUT
    parser = argparse.ArgumentParser(
        description="Post-hoc structural sensitivity over deterministic label-balanced samples."
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/stage14_alternative_balanced_samples",
    )
    parser.add_argument(
        "--compact-input",
        help="Optional precomputed 6,454-row structural feature CSV.",
    )
    parser.add_argument(
        "--strict-evidence",
        help="Optional full strict-evidence JSONL used to build the compact input.",
    )
    args = parser.parse_args()
    OUTPUT = Path(args.output_dir)
    if not OUTPUT.is_absolute():
        OUTPUT = ROOT / OUTPUT
    COMPACT_INPUT = (
        Path(args.compact_input)
        if args.compact_input
        else OUTPUT / "full_pool_structural_features.csv"
    )
    if not COMPACT_INPUT.is_absolute():
        COMPACT_INPUT = ROOT / COMPACT_INPUT
    strict_evidence = Path(args.strict_evidence) if args.strict_evidence else None

    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = (
        read_csv(COMPACT_INPUT)
        if COMPACT_INPUT.exists()
        else build_compact_input(strict_evidence)
    )
    numeric_fields = {
        "entity_linked", "has_local_path", "has_direct_edge",
        "has_predicate_aligned_direct", "has_qualifier_compatible_direct",
    }
    for row in rows:
        for field in numeric_fields:
            row[field] = int(row[field])

    detailed = [summarize("formal600_id_sorted", "id_sorted_original", "not_applicable", select_original(rows))]
    membership: list[dict] = []
    frames = {
        "full_6454_pool": rows,
        "matched_source_frame": [row for row in rows if row["dataset"] in {"healthver", "medaesqa", "scifact"}],
    }
    for frame_name, frame_rows in frames.items():
        for offset in range(N_REPLICATES):
            seed = SEED + offset
            sample_id = f"{frame_name}_seed_{seed}"
            sample = select_by_hash(frame_rows, seed)
            detailed.append(summarize(sample_id, frame_name, str(seed), sample))
            membership.extend(
                {"sample_id": sample_id, "sampling_frame": frame_name, "selection_seed": seed, "id": row["id"]}
                for row in sample
            )

    write_csv(OUTPUT / "alternative_sample_metrics.csv", detailed)
    write_csv(OUTPUT / "alternative_sample_membership.csv", membership)
    aggregate_rows = aggregate(detailed)
    write_csv(OUTPUT / "alternative_sample_summary.csv", aggregate_rows)
    manifest = {
        "master_seed": SEED,
        "replicate_seeds": [SEED + offset for offset in range(N_REPLICATES)],
        "replicates_per_frame": N_REPLICATES,
        "labels": list(LABELS),
        "records_per_label": N_PER_LABEL,
        "near_duplicate_rule": "normalized claim/source exact union plus claim char_wb TF-IDF 3-5 similarity >= 0.90",
        "selection_rule": "ascending SHA256(seed:id) independently within each label",
        "full_frame_records": len(rows),
        "matched_source_frame_records": len(frames["matched_source_frame"]),
        "no_llm_or_api_calls": True,
        "outputs": [
            "full_pool_structural_features.csv",
            "alternative_sample_metrics.csv",
            "alternative_sample_membership.csv",
            "alternative_sample_summary.csv",
        ],
    }
    (OUTPUT / "alternative_sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "summary": aggregate_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
