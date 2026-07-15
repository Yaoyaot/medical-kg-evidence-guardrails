from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import requests


LABELS = ["SUPPORT", "REFUTE", "UNCERTAIN"]
BASELINES = ["direct_llm", "text_rag_llm", "kg_rag_llm"]


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_inputs(
    data_dir: Path,
    verification_path: Path | None = None,
    subgraph_path: Path | None = None,
) -> list[dict]:
    verification_path = verification_path or data_dir / "processed" / "claim_verification_results.jsonl"
    subgraph_path = subgraph_path or data_dir / "processed" / "local_subgraphs.jsonl"
    if not verification_path.exists():
        raise FileNotFoundError(f"Verification results not found: {verification_path}")
    if not subgraph_path.exists():
        raise FileNotFoundError(f"Local subgraphs not found: {subgraph_path}")

    by_id = {row["id"]: row for row in read_jsonl(verification_path)}
    for row in read_jsonl(subgraph_path):
        if row["id"] not in by_id:
            continue
        by_id[row["id"]]["source"] = row.get("source", "")
        by_id[row["id"]]["kg_paths"] = [
            path.get("path_text", "")
            for path in (row.get("top_evidence_paths") or row.get("evidence_paths") or [])
            if path.get("path_text")
        ][:20]
    return list(by_id.values())


def stratified_sample(rows: list[dict], per_label_limit: int | None, limit: int | None) -> list[dict]:
    if per_label_limit is None and limit is None:
        return rows

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: str(item.get("id", ""))):
        buckets[row.get("gold_label", "UNCERTAIN")].append(row)

    selected: list[dict] = []
    if per_label_limit is not None:
        for label in LABELS:
            selected.extend(buckets.get(label, [])[:per_label_limit])
    else:
        selected = [row for label in LABELS for row in buckets.get(label, [])]

    if limit is not None:
        selected = selected[:limit]
    return selected


def normalize_label(value: object) -> str:
    text = str(value or "").strip().upper()
    text = text.replace("-", " ").replace("_", " ")
    if text in {"SUPPORT", "SUPPORTED", "TRUE", "ENTAILMENT"}:
        return "SUPPORT"
    if text in {"REFUTE", "REFUTED", "FALSE", "CONTRADICT", "CONTRADICTION"}:
        return "REFUTE"
    if text in {"UNCERTAIN", "UNKNOWN", "NEI", "NOT ENOUGH INFO", "NOT ENOUGH INFORMATION"}:
        return "UNCERTAIN"
    if "NOT ENOUGH" in text:
        return "UNCERTAIN"
    if "SUPPORT" in text:
        return "SUPPORT"
    if "REFUT" in text or "CONTRADICT" in text:
        return "REFUTE"
    return "UNCERTAIN"


def extract_json(text: str) -> tuple[dict, bool]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), True
            except json.JSONDecodeError:
                pass
    return {}, True


def clip_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def kg_context(row: dict, max_chars: int) -> str:
    paths = row.get("kg_paths", [])
    if not paths:
        return "No local KG evidence paths were retrieved."
    text = "\n".join(f"{idx + 1}. {path}" for idx, path in enumerate(paths))
    return clip_text(text, max_chars)


def build_prompt(row: dict, baseline: str, max_source_chars: int, max_kg_chars: int) -> str:
    claim = row.get("claim", "")
    schema = (
        'Return ONLY valid JSON with keys: "label", "confidence", "reasoning". '
        'The label must be exactly one of "SUPPORT", "REFUTE", "UNCERTAIN". '
        'Confidence must be a number between 0 and 1.'
    )

    if baseline == "direct_llm":
        evidence = "No external evidence is provided. Use your medical knowledge, and choose UNCERTAIN if unsure."
    elif baseline == "text_rag_llm":
        evidence = (
            "Use ONLY the following source text as evidence. If the source does not contain enough "
            f"information, choose UNCERTAIN.\n\n[Source]\n{clip_text(row.get('source', ''), max_source_chars)}"
        )
    elif baseline == "kg_rag_llm":
        evidence = (
            "Use ONLY the following local medical KG evidence paths. If the paths do not contain enough "
            f"information, choose UNCERTAIN.\n\n[KG Evidence Paths]\n{kg_context(row, max_kg_chars)}"
        )
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    return f"""You are a strict medical fact-checking system.

Task: Determine whether the claim is supported, refuted, or uncertain.

[Claim]
{claim}

{evidence}

Rules:
- SUPPORT means the evidence or your direct medical judgment clearly entails the claim.
- REFUTE means the evidence or your direct medical judgment clearly contradicts the claim.
- UNCERTAIN means evidence is missing, ambiguous, unrelated, or insufficient.
- Do not invent evidence.

{schema}
"""


def chat_completion(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_retries: int,
) -> str:
    base_url = base_url.rstrip("/")
    endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"LLM request failed after {max_retries} attempts: {last_error}")


def completed_keys(path: Path, retry_failed: bool) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        if retry_failed and (row.get("request_error") or row.get("parse_error")):
            continue
        keys.add((row.get("baseline", ""), row.get("id", "")))
    return keys


def run_baseline(
    row: dict,
    baseline: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_retries: int,
    max_source_chars: int,
    max_kg_chars: int,
) -> dict:
    prompt = build_prompt(row, baseline, max_source_chars, max_kg_chars)
    started = time.time()
    try:
        raw = chat_completion(prompt, api_key, base_url, model, timeout, max_retries)
        parsed, parse_error = extract_json(raw)
        pred_label = normalize_label(parsed.get("label"))
        confidence = parsed.get("confidence", 0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(confidence, 1.0))
        reasoning = str(parsed.get("reasoning", "")).strip()
        request_error = None
    except Exception as exc:
        raw = ""
        pred_label = "UNCERTAIN"
        confidence = 0.0
        reasoning = ""
        parse_error = True
        request_error = str(exc)

    return {
        "baseline": baseline,
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "claim": row.get("claim"),
        "gold_label": row.get("gold_label"),
        "pred_label": pred_label,
        "confidence": round(confidence, 4),
        "reasoning": reasoning,
        "parse_error": parse_error,
        "request_error": request_error,
        "raw_response": raw,
        "latency_seconds": round(time.time() - started, 4),
        "model": model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI-compatible LLM baselines.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--baseline", choices=["all", *BASELINES], default="all")
    parser.add_argument("--per-label-limit", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--output-name", default="llm_baseline_results.jsonl")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "qwen3.5-flash")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("LLM_MAX_RETRIES", "3")))
    parser.add_argument("--max-source-chars", type=int, default=5000)
    parser.add_argument("--max-kg-chars", type=int, default=5000)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or LLM_API_KEY environment variable.")
    if not args.model:
        raise RuntimeError("Missing model. Set OPENAI_MODEL/LLM_MODEL or pass --model.")

    data_dir = Path(args.data_dir)
    baselines = BASELINES if args.baseline == "all" else [args.baseline]
    rows = stratified_sample(load_inputs(data_dir), args.per_label_limit, args.limit)
    out_path = data_dir / "processed" / args.output_name
    done = completed_keys(out_path, args.retry_failed) if args.resume else set()

    written = 0
    attempted = 0
    for baseline in baselines:
        for row in rows:
            key = (baseline, row["id"])
            if key in done:
                continue
            attempted += 1
            result = run_baseline(
                row=row,
                baseline=baseline,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_source_chars=args.max_source_chars,
                max_kg_chars=args.max_kg_chars,
            )
            append_jsonl(out_path, [result])
            written += 1
            print(
                json.dumps(
                    {
                        "baseline": baseline,
                        "id": row["id"],
                        "gold": row.get("gold_label"),
                        "pred": result["pred_label"],
                        "parse_error": result["parse_error"],
                        "request_error": bool(result["request_error"]),
                    },
                    ensure_ascii=False,
                )
            )

    print(
        json.dumps(
            {
                "selected_input_claims": len(rows),
                "baselines": baselines,
                "attempted_requests": attempted,
                "written_results": written,
                "output": str(out_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
