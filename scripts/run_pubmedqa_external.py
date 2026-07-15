from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from run_llm_baselines import (
    append_jsonl,
    build_prompt as build_base_prompt,
    completed_keys,
    load_inputs,
    run_baseline,
)
from run_eswa_llm_baselines import build_prompt_payload, run_once


METHODS = (
    "direct_llm",
    "text_rag_llm",
    "provided_text_bm25_llm",
    "provided_text_kg_llm",
    "vanilla_graphrag_llm",
    "medgraphrag_style_llm",
)
LABELS = ("SUPPORT", "REFUTE", "UNCERTAIN")
DEFAULT_STAGE8 = Path("data/processed/stage8_pubmedqa_external")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 60:
        raise ValueError(f"Expected 60 audit rows, found {len(rows)}")
    if any(str(row.get("claim_faithfulness", "")).strip().upper() != "VALID" for row in rows):
        raise ValueError("Audit faithfulness gate has not passed")
    if any(str(row.get("atomicity", "")).strip().upper() != "ATOMIC" for row in rows):
        raise ValueError("Audit atomicity gate has not passed")
    ids = [str(row["id"]).strip() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Audit contains duplicate IDs")
    return ids


def select_scope(corpus: list[dict], audit_path: Path, scope: str) -> list[dict]:
    by_id = {str(row["id"]): row for row in corpus}
    if len(by_id) != 300:
        raise ValueError(f"Expected Claim-300 corpus, found {len(by_id)} unique IDs")
    pilot = audit_ids(audit_path)
    missing = [claim_id for claim_id in pilot if claim_id not in by_id]
    if missing:
        raise ValueError(f"Audit IDs missing from corpus: {missing[:5]}")
    if scope == "audit60":
        ids = pilot
    elif scope == "remaining240":
        pilot_set = set(pilot)
        ids = [claim_id for claim_id in sorted(by_id) if claim_id not in pilot_set]
    elif scope == "all300":
        ids = [claim_id for claim_id in sorted(by_id)]
    else:
        raise ValueError(f"Unknown scope: {scope}")
    rows = [by_id[claim_id] for claim_id in ids]
    expected = {"audit60": 60, "remaining240": 240, "all300": 300}[scope]
    if len(rows) != expected:
        raise ValueError(f"Scope {scope} expected {expected} rows, found {len(rows)}")
    return rows


def prompt_for(row: dict, method: str, corpus: list[dict], top_k: int, max_source_chars: int, max_kg_chars: int) -> tuple[str, dict]:
    if method in {"direct_llm", "text_rag_llm"}:
        prompt = build_base_prompt(row, method, max_source_chars, max_kg_chars)
        return prompt, {
            "prompt_chars": len(prompt),
            "kg_path_count": len(row.get("kg_paths") or []),
            "has_kg_evidence": bool(row.get("kg_paths")),
            "retrieved_evidence_ids": [],
        }
    return build_prompt_payload(row, method, corpus, top_k, max_source_chars, max_kg_chars)


def assert_prompt_safe(prompt: str, row: dict) -> None:
    forbidden_markers = ("[Gold Label]", "final_decision", "long_answer", "raw_label")
    hits = [marker for marker in forbidden_markers if marker.lower() in prompt.lower()]
    if hits:
        raise AssertionError(f"Prompt contains forbidden markers: {hits}")
    if str(row.get("gold_label", "")) and "gold_label" in prompt.lower():
        raise AssertionError("Prompt exposes gold_label field")


def preview_path(stage8: Path, scope: str) -> Path:
    return stage8 / f"{scope}_prompt_preview.jsonl"


def result_path(stage8: Path, scope: str) -> Path:
    return stage8 / {
        "audit60": "pilot60_baseline_results.jsonl",
        "remaining240": "remaining240_baseline_results.jsonl",
        "all300": "external_baseline_results.jsonl",
    }[scope]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen PubMedQA external-validation baselines.")
    parser.add_argument("--stage8-dir", default=str(DEFAULT_STAGE8))
    parser.add_argument("--scope", choices=("audit60", "remaining240", "all300"), default="audit60")
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--verification-path")
    parser.add_argument("--subgraph-path")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-v4-flash")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("LLM_MAX_RETRIES", "3")))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-source-chars", type=int, default=6000)
    parser.add_argument("--max-kg-chars", type=int, default=6000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--output-path", help="Optional method-shard output path.")
    args = parser.parse_args()

    stage8 = Path(args.stage8_dir)
    verification = Path(args.verification_path) if args.verification_path else stage8 / "kg/claim_verification_results.jsonl"
    subgraphs = Path(args.subgraph_path) if args.subgraph_path else stage8 / "kg/local_subgraphs.jsonl"
    corpus = load_inputs(Path("data"), verification, subgraphs)
    rows = select_scope(corpus, stage8 / "pubmedqa_claim_audit60.csv", args.scope)
    methods = METHODS if args.method == "all" else (args.method,)

    if Counter(row.get("gold_label") for row in rows) != (
        Counter({label: 20 for label in LABELS}) if args.scope == "audit60" else Counter(row.get("gold_label") for row in rows)
    ):
        raise ValueError("audit60 is not balanced 20/20/20")

    if args.dry_run:
        previews = []
        for method in methods:
            for row in rows[:3]:
                prompt, metadata = prompt_for(row, method, corpus, args.top_k, args.max_source_chars, args.max_kg_chars)
                assert_prompt_safe(prompt, row)
                previews.append({"method": method, "id": row["id"], "claim": row["claim"], "prompt": prompt, **metadata})
        path = preview_path(stage8, args.scope)
        write_mode = "w"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(write_mode, encoding="utf-8") as handle:
            for row in previews:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"dry_run": True, "scope": args.scope, "scope_rows": len(rows), "preview_rows": len(previews), "output": str(path)}, indent=2))
        return

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or LLM_API_KEY")
    output = Path(args.output_path) if args.output_path else result_path(stage8, args.scope)
    done = completed_keys(output, args.retry_failed) if args.resume else set()
    written = 0
    for method in methods:
        for row in rows:
            if (method, row["id"]) in done:
                continue
            prompt, metadata = prompt_for(row, method, corpus, args.top_k, args.max_source_chars, args.max_kg_chars)
            assert_prompt_safe(prompt, row)
            if method in {"direct_llm", "text_rag_llm"}:
                result = run_baseline(
                    row, method, api_key, args.base_url, args.model, args.timeout, args.max_retries,
                    args.max_source_chars, args.max_kg_chars,
                )
                result.update(metadata)
                result["request_count"] = 1
            else:
                result = run_once(
                    row, method, prompt, api_key, args.base_url, args.model, args.timeout,
                    args.max_retries, metadata,
                )
            append_jsonl(output, [result])
            written += 1
            print(json.dumps({"scope": args.scope, "method": method, "id": row["id"], "pred": result["pred_label"], "failed": bool(result.get("request_error") or result.get("parse_error"))}, ensure_ascii=False))
    print(json.dumps({"scope": args.scope, "scope_rows": len(rows), "methods": methods, "written": written, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
