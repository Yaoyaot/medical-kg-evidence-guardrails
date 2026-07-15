from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

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


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokens(value: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(value) if tok.lower() not in STOPWORDS}


def is_numeric_alias(alias: str) -> bool:
    return bool(re.fullmatch(r"\d+", alias.strip()))


def is_short_alias_allowed(alias: str, min_alias_length: int) -> bool:
    compact = re.sub(r"\s+", "", alias)
    return len(compact) >= min_alias_length


def phrase_in_text(phrase: str, text: str) -> bool:
    if not phrase or not text:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


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


def load_aliases(
    path: Path,
    min_alias_length: int,
    disable_numeric_aliases: bool,
) -> tuple[list[dict], dict[str, set[int]], dict[str, int]]:
    aliases: list[dict] = []
    inverted: dict[str, set[int]] = defaultdict(set)
    stats = {
        "raw_aliases": 0,
        "filtered_numeric_aliases": 0,
        "filtered_short_aliases": 0,
        "filtered_empty_aliases": 0,
        "filtered_tokenless_aliases": 0,
    }
    for row in read_jsonl(path):
        stats["raw_aliases"] += 1
        alias = str(row.get("alias", "")).strip()
        kind = str(row.get("kind", "")).strip()
        normalized = str(row.get("normalized_alias") or normalize_text(alias)).strip()
        if not normalized:
            stats["filtered_empty_aliases"] += 1
            continue
        if disable_numeric_aliases and is_numeric_alias(normalized):
            stats["filtered_numeric_aliases"] += 1
            continue
        if not is_short_alias_allowed(normalized, min_alias_length):
            stats["filtered_short_aliases"] += 1
            continue

        alias_tokens = tokens(normalized)
        if not alias_tokens:
            stats["filtered_tokenless_aliases"] += 1
            continue

        item = {
            "alias": alias,
            "normalized_alias": normalized,
            "alias_tokens": alias_tokens,
            "node_id": row.get("node_id", ""),
            "name": row.get("name", ""),
            "kind": kind,
            "alias_source": row.get("alias_source", "hetionet"),
            "external_id": row.get("external_id", ""),
        }
        alias_id = len(aliases)
        aliases.append(item)
        for token in alias_tokens:
            inverted[token].add(alias_id)
    return aliases, inverted, stats


def score_alias(
    claim: str,
    claim_norm: str,
    claim_tokens: set[str],
    alias: dict,
    token_threshold: float,
) -> tuple[float, str] | None:
    alias_norm = alias["normalized_alias"]
    alias_tokens = alias["alias_tokens"]
    if (
        str(alias.get("alias_source", "")).startswith("hgnc")
        and len(alias_tokens) == 1
        and len(alias_norm) <= 4
        and not phrase_in_text(str(alias.get("alias", "")), claim)
    ):
        return None

    if alias_norm == claim_norm:
        return 1.0, "exact_claim"
    if alias_norm in claim_norm and phrase_in_text(alias_norm, claim_norm):
        return 0.85, "alias_in_claim"
    if claim_norm and len(claim_norm) >= 8 and claim_norm in alias_norm and phrase_in_text(claim_norm, alias_norm):
        return 0.70, "claim_in_alias"

    overlap_tokens = alias_tokens & claim_tokens
    if len(alias_tokens) < 2 or not any(not token.isdigit() for token in overlap_tokens):
        return None

    overlap = len(overlap_tokens) / max(len(alias_tokens), 1)
    if overlap >= token_threshold:
        return min(0.65, 0.50 + 0.15 * overlap), "token_overlap"
    return None


def link_claim(row: dict, aliases: list[dict], inverted: dict[str, set[int]], top_k: int, token_threshold: float) -> dict:
    claim = str(row.get("claim", "")).strip()
    claim_norm = normalize_text(claim)
    claim_tokens = tokens(claim_norm)

    candidate_alias_ids: set[int] = set()
    for token in claim_tokens:
        candidate_alias_ids.update(inverted.get(token, set()))

    best_by_node: dict[str, dict] = {}
    for alias_id in candidate_alias_ids:
        alias = aliases[alias_id]
        scored = score_alias(claim, claim_norm, claim_tokens, alias, token_threshold)
        if scored is None:
            continue
        score, match_type = scored
        node_id = alias["node_id"]
        existing = best_by_node.get(node_id)
        if existing and existing["score"] >= score:
            continue
        best_by_node[node_id] = {
            "node_id": node_id,
            "name": alias["name"],
            "kind": alias["kind"],
            "matched_alias": alias["alias"],
            "normalized_alias": alias["normalized_alias"],
            "match_type": match_type,
            "score": round(score, 4),
            "matched_alias_source": alias.get("alias_source", "hetionet"),
            "external_id": alias.get("external_id", ""),
        }

    linked_entities = sorted(
        best_by_node.values(),
        key=lambda item: (-item["score"], item["kind"], len(item["normalized_alias"]), item["name"]),
    )[:top_k]

    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "claim": claim,
        "source": row.get("source", ""),
        "label": row.get("label"),
        "raw_label": row.get("raw_label"),
        "linked_entities": linked_entities,
        "entity_linking_stats": {
            "candidate_aliases": len(candidate_alias_ids),
            "linked_entity_count": len(linked_entities),
        },
    }


def input_rows(data_dir: Path, input_paths: list[Path] | None = None) -> Iterable[dict]:
    if input_paths:
        for path in input_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required normalized claim file not found: {path}")
            yield from read_jsonl(path)
        return
    paths = [
        data_dir / "processed" / "medfact_bench.sample.jsonl",
        data_dir / "processed" / "pubhealth.sample.jsonl",
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required normalized claim file not found: {path}")
        yield from read_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Link claim entities to Hetionet aliases.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--alias-path", action="append", default=None)
    parser.add_argument("--input-path", action="append", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token-threshold", type=float, default=0.75)
    parser.add_argument("--min-alias-length", type=int, default=3)
    parser.add_argument("--disable-numeric-aliases", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    graph_dir = Path(args.graph_dir) if args.graph_dir else data_dir / "processed"
    alias_paths = [Path(path) for path in args.alias_path] if args.alias_path else [graph_dir / "hetionet_aliases.jsonl"]
    aliases = []
    inverted: dict[str, set[int]] = defaultdict(set)
    alias_filter_stats = Counter()
    for aliases_path in alias_paths:
        if not aliases_path.exists():
            raise FileNotFoundError(f"Alias file not found: {aliases_path}")
        loaded, _, loaded_stats = load_aliases(
            aliases_path,
            min_alias_length=args.min_alias_length,
            disable_numeric_aliases=args.disable_numeric_aliases,
        )
        offset = len(aliases)
        aliases.extend(loaded)
        for index, alias in enumerate(loaded, start=offset):
            for token in alias["alias_tokens"]:
                inverted[token].add(index)
        alias_filter_stats.update(loaded_stats)
    rows = [
        link_claim(row, aliases, inverted, args.top_k, args.token_threshold)
        for row in input_rows(data_dir, [Path(path) for path in args.input_path] if args.input_path else None)
    ]

    out_dir = Path(args.output_dir) if args.output_dir else data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "linked_claims.jsonl"
    write_jsonl(out_path, rows)

    coverage = sum(1 for row in rows if row["linked_entities"]) / max(len(rows), 1)
    match_types = Counter(
        entity["match_type"]
        for row in rows
        for entity in row["linked_entities"]
    )
    top_aliases = Counter(
        entity["matched_alias"]
        for row in rows
        for entity in row["linked_entities"]
    ).most_common(20)
    linked_kind_counts = Counter(
        entity["kind"]
        for row in rows
        for entity in row["linked_entities"]
    )

    summary = {
        "input_claims": len(rows),
        "aliases_loaded": len(aliases),
        **dict(alias_filter_stats),
        "alias_paths": [str(path) for path in alias_paths],
        "linked_claims": sum(1 for row in rows if row["linked_entities"]),
        "entity_linking_coverage": round(coverage, 4),
        "match_type_counts": dict(match_types),
        "linked_kind_counts": dict(linked_kind_counts),
        "top_matched_aliases": top_aliases,
        "output": str(out_path),
    }
    summary_path = out_dir / "entity_linking_stats.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
