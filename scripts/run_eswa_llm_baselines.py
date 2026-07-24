from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

from run_llm_baselines import (
    LABELS,
    append_jsonl,
    chat_completion,
    clip_text,
    completed_keys,
    extract_json,
    kg_context,
    load_inputs,
    normalize_label,
    read_jsonl,
    stratified_sample,
)


ESWA_BASELINES = [
    "bm25_text_rag_llm",
    "tfidf_text_rag_llm",
    "provided_text_bm25_llm",
    "provided_text_kg_llm",
    "vanilla_graphrag_llm",
    "medgraphrag_style_llm",
    "llm_self_consistency_3",
    "text_rag_llm_judge",
]

_BM25_CACHE: dict[int, tuple[list[dict], list[Counter], list[int], float, Counter]] = {}
_TFIDF_CACHE: dict[int, tuple[list[dict], list[Counter], Counter]] = {}


def select_manifest_rows(corpus: list[dict], manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    ids = [str(row.get("id", "")).strip() for row in manifest]
    if not ids or any(not item for item in ids):
        raise ValueError(f"Sample manifest contains empty IDs: {manifest_path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Sample manifest contains duplicate IDs: {manifest_path}")
    by_id = {str(row["id"]): row for row in corpus}
    missing = [item for item in ids if item not in by_id]
    if missing:
        raise ValueError(f"Sample IDs missing from corpus ({len(missing)}): {missing[:5]}")
    selected = []
    for manifest_row in manifest:
        item = dict(by_id[manifest_row["id"]])
        item["has_kg_evidence"] = str(manifest_row.get("has_kg_evidence", "")).lower() == "true"
        item["sample_manifest_source"] = str(manifest_path.resolve())
        item["sample_order"] = int(manifest_row.get("sample_order") or len(selected) + 1)
        selected.append(item)
    return selected


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def top_bm25(query: str, corpus: list[dict], k: int, exclude_id: str | None = None) -> list[dict]:
    query_terms = Counter(tokens(query))
    if not query_terms:
        return []
    cache_key = id(corpus)
    if cache_key not in _BM25_CACHE:
        doc_terms = [Counter(tokens(row.get("source", ""))) for row in corpus]
        doc_lengths = [sum(terms.values()) for terms in doc_terms]
        avg_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1.0
        df = Counter()
        for terms in doc_terms:
            for term in terms:
                df[term] += 1
        _BM25_CACHE[cache_key] = (corpus, doc_terms, doc_lengths, avg_len, df)
    indexed_corpus, doc_terms, doc_lengths, avg_len, df = _BM25_CACHE[cache_key]
    n_docs = len(corpus)
    scores = []
    for row, terms, length in zip(indexed_corpus, doc_terms, doc_lengths):
        if str(row.get("id")) == str(exclude_id):
            continue
        score = 0.0
        for term, qtf in query_terms.items():
            tf = terms.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf + 1.5 * (1 - 0.75 + 0.75 * length / max(avg_len, 1))
            score += idf * (tf * 2.5 / denom) * qtf
        if score > 0:
            scores.append((score, row))
    return [
        {
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "score": round(score, 4),
            "source": row.get("source", ""),
        }
        for score, row in sorted(scores, key=lambda item: item[0], reverse=True)[:k]
    ]


def top_tfidf(query: str, corpus: list[dict], k: int, exclude_id: str | None = None) -> list[dict]:
    query_terms = Counter(tokens(query))
    if not query_terms:
        return []
    cache_key = id(corpus)
    if cache_key not in _TFIDF_CACHE:
        doc_terms = [Counter(tokens(row.get("source", ""))) for row in corpus]
        df = Counter()
        for terms in doc_terms:
            for term in terms:
                df[term] += 1
        _TFIDF_CACHE[cache_key] = (corpus, doc_terms, df)
    indexed_corpus, doc_terms, df = _TFIDF_CACHE[cache_key]
    n_docs = len(corpus)
    query_vec = {}
    for term, tf in query_terms.items():
        query_vec[term] = (1 + math.log(tf)) * math.log((n_docs + 1) / (df.get(term, 0) + 1))
    query_norm = math.sqrt(sum(value * value for value in query_vec.values())) or 1.0
    scores = []
    for row, terms in zip(indexed_corpus, doc_terms):
        if str(row.get("id")) == str(exclude_id):
            continue
        dot = 0.0
        doc_norm_sq = 0.0
        for term, tf in terms.items():
            value = (1 + math.log(tf)) * math.log((n_docs + 1) / (df.get(term, 0) + 1))
            doc_norm_sq += value * value
            dot += value * query_vec.get(term, 0.0)
        denom = query_norm * (math.sqrt(doc_norm_sq) or 1.0)
        score = dot / denom
        if score > 0:
            scores.append((score, row))
    return [
        {
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "score": round(score, 4),
            "source": row.get("source", ""),
        }
        for score, row in sorted(scores, key=lambda item: item[0], reverse=True)[:k]
    ]


def retrieved_text(snippets: list[dict], max_chars: int) -> str:
    if not snippets:
        return "No retrieved text snippets."
    parts = []
    for index, item in enumerate(snippets, start=1):
        parts.append(
            f"[Snippet {index} | id={item.get('id')} | dataset={item.get('dataset')} | score={item.get('score')}]\n"
            + clip_text(item.get("source", ""), max_chars // max(len(snippets), 1))
        )
    return "\n\n".join(parts)


def schema_text() -> str:
    return (
        'Return ONLY valid JSON with keys: "label", "confidence", "reasoning". '
        'The label must be exactly one of "SUPPORT", "REFUTE", "UNCERTAIN". '
        "Confidence must be a number between 0 and 1."
    )


def classifier_prompt(claim: str, evidence: str) -> str:
    return f"""You are a strict medical fact-checking system.

Task: Determine whether the claim is supported, refuted, or uncertain.

[Claim]
{claim}

[Evidence]
{evidence}

Rules:
- SUPPORT means the evidence clearly entails the claim.
- REFUTE means the evidence clearly contradicts the claim.
- UNCERTAIN means evidence is missing, ambiguous, unrelated, or insufficient.
- Do not invent evidence.

{schema_text()}
"""


def judge_prompt(row: dict, candidate: dict) -> str:
    return f"""You are a medical evidence judge.

The first model produced a candidate fact-checking label. Verify whether that label is justified by the source text.

[Claim]
{row.get('claim')}

[Source]
{clip_text(row.get('source', ''), 5000)}

[Candidate label]
{candidate.get('pred_label')}

[Candidate reasoning]
{candidate.get('reasoning')}

Return ONLY valid JSON with keys: "label", "confidence", "reasoning".
If the source does not justify the candidate label, choose UNCERTAIN.
"""


def build_prompt_payload(
    row: dict,
    baseline: str,
    corpus: list[dict],
    top_k: int,
    max_source_chars: int,
    max_kg_chars: int,
) -> tuple[str, dict]:
    claim = row.get("claim", "")
    retrieved = []
    if baseline == "bm25_text_rag_llm":
        retrieved = top_bm25(claim, corpus, top_k, row.get("id"))
        evidence = retrieved_text(retrieved, max_source_chars)
    elif baseline == "tfidf_text_rag_llm":
        retrieved = top_tfidf(claim, corpus, top_k, row.get("id"))
        evidence = retrieved_text(retrieved, max_source_chars)
    elif baseline == "provided_text_bm25_llm":
        retrieved = top_bm25(claim, corpus, top_k, row.get("id"))
        evidence = (
            "[Provided claim-associated text evidence]\n"
            + clip_text(row.get("source", ""), max_source_chars // 2)
            + "\n\n[Additional BM25-retrieved passages]\n"
            + retrieved_text(retrieved, max_source_chars // 2)
        )
    elif baseline == "provided_text_kg_llm":
        evidence = (
            "[Provided claim-associated text evidence]\n"
            + clip_text(row.get("source", ""), max_source_chars // 2)
            + "\n\n[Local KG paths]\n"
            + kg_context(row, max_kg_chars)
        )
    elif baseline == "vanilla_graphrag_llm":
        evidence = "[Local KG paths]\n" + kg_context(row, max_kg_chars)
    elif baseline == "medgraphrag_style_llm":
        retrieved = top_bm25(claim, corpus, max(1, top_k - 1), row.get("id"))
        evidence = (
            "[Text evidence]\n"
            + clip_text(row.get("source", ""), max_source_chars // 2)
            + "\n\n[Local KG paths]\n"
            + kg_context(row, max_kg_chars // 2)
            + "\n\n[Retrieved related snippets]\n"
            + retrieved_text(retrieved, max_source_chars // 2)
        )
    else:
        evidence = clip_text(row.get("source", ""), max_source_chars)
    prompt = classifier_prompt(claim, evidence)
    metadata = {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_chars": len(prompt),
        "retrieved_evidence_ids": [item.get("id") for item in retrieved],
        "kg_path_count": len(row.get("kg_paths") or []),
        "subgraph_source": row.get("subgraph_source", ""),
        "has_kg_evidence": bool(row.get("has_kg_evidence", row.get("kg_paths"))),
        "sample_manifest_source": row.get("sample_manifest_source", ""),
    }
    return prompt, metadata


def build_prompt(row: dict, baseline: str, corpus: list[dict], top_k: int, max_source_chars: int, max_kg_chars: int) -> str:
    return build_prompt_payload(row, baseline, corpus, top_k, max_source_chars, max_kg_chars)[0]


def call_json(prompt: str, api_key: str, base_url: str, model: str, timeout: int, max_retries: int) -> tuple[str, dict, bool, str | None, float]:
    started = time.time()
    try:
        raw = chat_completion(prompt, api_key, base_url, model, timeout, max_retries)
        parsed, parse_error = extract_json(raw)
        return raw, parsed, parse_error, None, time.time() - started
    except Exception as exc:
        return "", {}, True, str(exc), time.time() - started


def run_once(
    row: dict,
    baseline: str,
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_retries: int,
    metadata: dict | None = None,
) -> dict:
    raw, parsed, parse_error, request_error, latency = call_json(prompt, api_key, base_url, model, timeout, max_retries)
    pred_label = normalize_label(parsed.get("label"))
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "baseline": baseline,
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "claim": row.get("claim"),
        "gold_label": row.get("gold_label"),
        "pred_label": pred_label,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "parse_error": parse_error,
        "request_error": request_error,
        "raw_response": raw,
        "latency_seconds": round(latency, 4),
        "model": model,
        "request_count": 1,
        **(metadata or {}),
    }


def majority_vote(results: list[dict]) -> dict:
    counts = Counter(row["pred_label"] for row in results)
    label = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    confidences = [float(row.get("confidence", 0.0)) for row in results if row["pred_label"] == label]
    base = dict(results[-1])
    base["pred_label"] = label
    base["confidence"] = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    base["reasoning"] = "Self-consistency majority vote over three Text-RAG calls."
    base["self_consistency_votes"] = counts
    base["raw_runs"] = results
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ESWA-strengthened LLM baselines.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--verification-path",
        default="data/processed/stage2_primekg_semantic_clean/variants/primekg_semantic_clean_relation_aware/claim_verification_results.jsonl",
    )
    parser.add_argument(
        "--subgraph-path",
        default="data/processed/stage2_primekg_semantic_clean/variants/primekg_semantic_clean_relation_aware/local_subgraphs.jsonl",
    )
    parser.add_argument("--output-dir", default="data/processed/stage7_hierarchical_scorer/baselines")
    parser.add_argument("--sample-ids-path")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--baseline", choices=["all", *ESWA_BASELINES], default="all")
    parser.add_argument("--per-label-limit", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-name", default="eswa_baseline_smoke_semantic_clean.jsonl")
    parser.add_argument("--prompt-preview-name", default="eswa_baseline_smoke_prompt_preview.jsonl")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-v4-flash")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("LLM_MAX_RETRIES", "3")))
    parser.add_argument("--max-source-chars", type=int, default=6000)
    parser.add_argument("--max-kg-chars", type=int, default=6000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    verification_path = Path(args.verification_path)
    subgraph_path = Path(args.subgraph_path)
    corpus = load_inputs(data_dir, verification_path, subgraph_path)
    for row in corpus:
        row["subgraph_source"] = str(subgraph_path.resolve())
    if args.sample_ids_path:
        rows = select_manifest_rows(corpus, Path(args.sample_ids_path))
        if args.limit is not None:
            rows = rows[: args.limit]
    else:
        rows = stratified_sample(corpus, args.per_label_limit, args.limit)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-count must be positive and shard-index must be in [0, shard-count)")
    shard_start = len(rows) * args.shard_index // args.shard_count
    shard_end = len(rows) * (args.shard_index + 1) // args.shard_count
    rows = rows[shard_start:shard_end]
    baselines = ESWA_BASELINES if args.baseline == "all" else [args.baseline]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / args.output_name
    done = completed_keys(out_path, args.retry_failed) if args.resume else set()

    if args.dry_run:
        preview_rows = []
        for baseline in baselines:
            for row in rows[: min(3, len(rows))]:
                prompt, metadata = build_prompt_payload(row, baseline, corpus, args.top_k, args.max_source_chars, args.max_kg_chars)
                preview_rows.append(
                    {
                        "baseline": baseline,
                        "id": row.get("id"),
                        "claim": row.get("claim"),
                        "prompt": prompt,
                        **metadata,
                    }
                )
        append_jsonl(output_dir / args.prompt_preview_name, preview_rows)
        print(json.dumps({"dry_run": True, "preview_rows": len(preview_rows), "selected_claims": len(rows), "baselines": baselines}, indent=2))
        return

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or LLM_API_KEY environment variable.")

    written = 0
    attempted = 0
    for baseline in baselines:
        for row in rows:
            key = (baseline, row["id"])
            if key in done:
                continue
            attempted += 1
            prompt, metadata = build_prompt_payload(row, baseline, corpus, args.top_k, args.max_source_chars, args.max_kg_chars)
            if baseline == "llm_self_consistency_3":
                runs = [
                    run_once(row, baseline + "_run", prompt, api_key, args.base_url, args.model, args.timeout, args.max_retries, metadata)
                    for _ in range(3)
                ]
                result = majority_vote(runs)
                result["baseline"] = baseline
                result["request_count"] = 3
            elif baseline == "text_rag_llm_judge":
                first = run_once(row, baseline + "_candidate", prompt, api_key, args.base_url, args.model, args.timeout, args.max_retries, metadata)
                judged = run_once(row, baseline, judge_prompt(row, first), api_key, args.base_url, args.model, args.timeout, args.max_retries)
                judged["candidate_label"] = first["pred_label"]
                judged["candidate_confidence"] = first["confidence"]
                judged.update(metadata)
                judged["request_count"] = 2
                result = judged
            else:
                result = run_once(row, baseline, prompt, api_key, args.base_url, args.model, args.timeout, args.max_retries, metadata)
            append_jsonl(out_path, [result])
            written += 1
            print(json.dumps({"baseline": baseline, "id": row["id"], "pred": result["pred_label"], "request_error": bool(result.get("request_error"))}, ensure_ascii=False))
    print(json.dumps({"selected_input_claims": len(rows), "baselines": baselines, "attempted_requests": attempted, "written_results": written, "output": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
