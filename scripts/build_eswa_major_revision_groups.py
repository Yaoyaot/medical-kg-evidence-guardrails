from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


SEED = 20260618
LABELS = ("SUPPORT", "REFUTE", "UNCERTAIN")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


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


def union_exact(values: list[str], union: UnionFind, reason: str, audit: list[dict], ids: list[str]) -> int:
    first: dict[str, int] = {}
    links = 0
    for index, value in enumerate(values):
        if not value:
            continue
        if value in first:
            union.union(index, first[value])
            links += 1
            audit.append({
                "left_id": ids[first[value]],
                "right_id": ids[index],
                "match_type": reason,
                "similarity": 1.0,
            })
        else:
            first[value] = index
    return links


def near_duplicate_links(claims: list[str], threshold: float, union: UnionFind, audit: list[dict], ids: list[str]) -> int:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False, norm="l2")
    matrix = vectorizer.fit_transform(claims)
    similarities = (matrix @ matrix.T).tocoo()
    links = 0
    for left, right, score in zip(similarities.row, similarities.col, similarities.data):
        if left >= right or score < threshold or claims[left] == claims[right]:
            continue
        union.union(int(left), int(right))
        links += 1
        audit.append({
            "left_id": ids[int(left)],
            "right_id": ids[int(right)],
            "match_type": "claim_char_tfidf",
            "similarity": round(float(score), 8),
        })
    return links


def component_rows(rows: list[dict], roots: list[int]) -> list[dict]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, root in enumerate(roots):
        grouped[root].append(index)
    output = []
    for root, indices in grouped.items():
        output.append({
            "root": root,
            "indices": indices,
            "size": len(indices),
            "labels": Counter(rows[index]["gold_label"] for index in indices),
            "datasets": Counter(rows[index]["dataset"] for index in indices),
        })
    return output


def objective(folds: list[dict], totals: dict) -> float:
    score = 0.0
    for fold in folds:
        score += ((fold["size"] - totals["size"] / 5) / max(totals["size"] / 5, 1)) ** 2
        for label in LABELS:
            target = totals["labels"][label] / 5
            score += 1.5 * ((fold["labels"][label] - target) / max(target, 1)) ** 2
        for dataset, count in totals["datasets"].items():
            target = count / 5
            score += 0.75 * ((fold["datasets"][dataset] - target) / max(target, 1)) ** 2
    return score


def assign_folds(components: list[dict], rows: list[dict], restarts: int, seed: int) -> tuple[dict[int, int], float]:
    totals = {
        "size": len(rows),
        "labels": Counter(row["gold_label"] for row in rows),
        "datasets": Counter(row["dataset"] for row in rows),
    }
    best_assignment: dict[int, int] | None = None
    best_score = float("inf")
    rng = random.Random(seed)
    base = sorted(components, key=lambda item: (-item["size"], item["root"]))
    for restart in range(restarts):
        if restart == 0:
            ordered = list(base)
        else:
            ordered = sorted(base, key=lambda item: (-item["size"] + rng.random() * 4.0, rng.random()))
        folds = [{"size": 0, "labels": Counter(), "datasets": Counter()} for _ in range(5)]
        assignment = {}
        for component in ordered:
            candidates = []
            for fold_index in range(5):
                fold = folds[fold_index]
                fold["size"] += component["size"]
                fold["labels"].update(component["labels"])
                fold["datasets"].update(component["datasets"])
                candidates.append((objective(folds, totals), fold_index))
                fold["size"] -= component["size"]
                fold["labels"].subtract(component["labels"])
                fold["datasets"].subtract(component["datasets"])
            _, selected = min(candidates, key=lambda item: (item[0], item[1]))
            fold = folds[selected]
            fold["size"] += component["size"]
            fold["labels"].update(component["labels"])
            fold["datasets"].update(component["datasets"])
            assignment[component["root"]] = selected
        score = objective(folds, totals)
        if score < best_score:
            best_score = score
            best_assignment = assignment
    assert best_assignment is not None
    return best_assignment, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-controlled Formal600 claim/source groups and five folds.")
    parser.add_argument("--risk-path", default="data/processed/stage7_hierarchical_scorer/matched_budget/claim_risk_scores.jsonl")
    parser.add_argument("--subgraphs-path", default="data/processed/stage2_primekg_semantic_clean/variants/primekg_semantic_clean_relation_aware/local_subgraphs.jsonl")
    parser.add_argument("--output-dir", default="data/processed/stage9_eswa_major_revision/grouping")
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--restarts", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.risk_path))
    if len(rows) != 600 or len({row["id"] for row in rows}) != 600:
        raise ValueError("Expected 600 unique Formal600 rows")
    by_id = {row["id"]: row for row in rows}
    subgraphs = {row["id"]: row for row in read_jsonl(Path(args.subgraphs_path)) if row["id"] in by_id}
    if len(subgraphs) != 600:
        raise ValueError(f"Expected 600 matching subgraphs, found {len(subgraphs)}")

    ids = [row["id"] for row in rows]
    claims = [normalize(row.get("claim", "")) for row in rows]
    sources = [normalize(subgraphs[row["id"]].get("source", "")) for row in rows]
    union = UnionFind(len(rows))
    audit: list[dict] = []
    exact_claim_links = union_exact(claims, union, "exact_claim", audit, ids)
    exact_source_links = union_exact(sources, union, "exact_source", audit, ids)
    near_links = near_duplicate_links(claims, args.near_duplicate_threshold, union, audit, ids)
    roots = [union.find(index) for index in range(len(rows))]
    components = component_rows(rows, roots)
    assignment, split_objective = assign_folds(components, rows, args.restarts, args.seed)

    root_to_component = {
        root: f"pair_component_{position:04d}"
        for position, root in enumerate(sorted({union.find(index) for index in range(len(rows))}), start=1)
    }
    manifest = []
    for index, row in enumerate(rows):
        root = union.find(index)
        source = sources[index]
        manifest.append({
            "id": row["id"],
            "dataset": row.get("dataset", ""),
            "gold_label": row.get("gold_label", ""),
            "claim": row.get("claim", ""),
            "normalized_claim_sha256": hashlib.sha256(claims[index].encode("utf-8")).hexdigest(),
            "normalized_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "pair_group_id": root_to_component[root],
            "outer_fold": assignment[root],
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "selection_seed": args.seed,
        })
    manifest.sort(key=lambda item: (int(item["outer_fold"]), item["pair_group_id"], item["id"]))

    id_to_fold = {row["id"]: int(row["outer_fold"]) for row in manifest}
    for link in audit:
        link["same_fold"] = id_to_fold[link["left_id"]] == id_to_fold[link["right_id"]]
    if any(not row["same_fold"] for row in audit):
        raise RuntimeError("A duplicate claim/source link crosses outer folds")

    fold_stats = []
    for fold in range(5):
        selected = [row for row in rows if id_to_fold[row["id"]] == fold]
        fold_stats.append({
            "fold": fold,
            "rows": len(selected),
            "groups": len({next(item["pair_group_id"] for item in manifest if item["id"] == row["id"]) for row in selected}),
            "labels": dict(Counter(row["gold_label"] for row in selected)),
            "datasets": dict(Counter(row["dataset"] for row in selected)),
            "provided_text_support_predictions": sum(row.get("candidate_label") == "SUPPORT" for row in selected),
        })

    output = Path(args.output_dir)
    write_csv(output / "formal600_group_manifest.csv", manifest)
    write_csv(output / "duplicate_and_near_duplicate_links.csv", sorted(audit, key=lambda item: (item["match_type"], item["left_id"], item["right_id"])))
    stats = {
        "rows": len(rows),
        "unique_normalized_claims": len(set(claims)),
        "unique_normalized_sources": len(set(sources)),
        "pair_components": len(root_to_component),
        "largest_component": max(Counter(root_to_component[union.find(index)] for index in range(len(rows))).values()),
        "exact_claim_links": exact_claim_links,
        "exact_source_links": exact_source_links,
        "near_duplicate_links": near_links,
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "fold_assignment_objective": split_objective,
        "seed": args.seed,
        "folds": fold_stats,
        "leakage_checks": {
            "duplicate_links_crossing_folds": sum(not row["same_fold"] for row in audit),
            "pair_group_crossing_folds": 0,
        },
    }
    (output / "formal600_grouping_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
