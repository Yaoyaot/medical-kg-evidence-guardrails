from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlencode

import requests


DATASET = "qiaojin/PubMedQA"
CONFIG = "pqa_labeled"
SPLIT = "train"
DATASET_SERVER = "https://datasets-server.huggingface.co"
DEFAULT_OUTPUT_DIR = Path("data/processed/stage8_pubmedqa_external")
LABEL_MAP = {"yes": "SUPPORT", "no": "REFUTE", "maybe": "UNCERTAIN"}
LABEL_ORDER = ("SUPPORT", "REFUTE", "UNCERTAIN")
FORBIDDEN_GENERATION_FIELDS = ("context", "long_answer", "final_decision", "gold_label", "raw_label")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "in", "into", "is",
    "it", "may", "might", "of", "on", "or", "should", "than", "that", "the", "their", "there",
    "these", "this", "to", "was", "were", "what", "when", "where", "whether", "which", "with",
    "would",
}

CANDIDATE_FIELDS = [
    "id", "pubid", "dataset", "split", "question", "claim", "source", "gold_label", "raw_label",
    "sample_order", "label_sample_order", "conversion_model", "conversion_batch", "conversion_attempt",
    "word_count", "sentence_count", "content_token_overlap", "missing_numbers", "missing_acronyms",
    "is_empty", "has_question_mark", "length_ok", "single_sentence", "numbers_preserved",
    "acronyms_preserved", "content_overlap_ok", "duplicate_claim", "automatic_check_pass",
    "automatic_check_flags",
]

AUDIT_FIELDS = [
    "audit_order", "id", "pubid", "gold_label", "question", "claim", "automatic_check_pass",
    "automatic_check_flags", "claim_faithfulness", "atomicity", "notes",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict], mode: str = "w") -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def request_json(url: str, timeout: int = 60) -> dict:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def download_pubmedqa(cache_path: Path, refresh: bool = False) -> tuple[list[dict], dict]:
    if cache_path.exists() and not refresh:
        rows = read_jsonl(cache_path)
        if len(rows) != 1000:
            raise ValueError(f"Cached PubMedQA row count is {len(rows)}, expected 1000: {cache_path}")
        return rows, {"cache_used": True, "cache_path": str(cache_path), "downloaded_rows": len(rows)}

    params = urlencode({"dataset": DATASET, "config": CONFIG, "split": SPLIT})
    first = request_json(f"{DATASET_SERVER}/rows?{params}&offset=0&length=100")
    total = int(first.get("num_rows_total", 0))
    pages = [first]
    for offset in range(100, total, 100):
        pages.append(request_json(f"{DATASET_SERVER}/rows?{params}&offset={offset}&length=100"))

    rows = [entry["row"] for page in pages for entry in page.get("rows", [])]
    if total != 1000 or len(rows) != total:
        raise ValueError(f"Unexpected PubMedQA size: api_total={total}, downloaded={len(rows)}")
    if len({str(row.get('pubid')) for row in rows}) != total:
        raise ValueError("PubMedQA contains duplicate pubids")
    write_jsonl(cache_path, rows)
    return rows, {
        "cache_used": False,
        "cache_path": str(cache_path),
        "downloaded_rows": len(rows),
        "pages": len(pages),
    }


def dataset_revision() -> str:
    try:
        info = request_json(f"https://huggingface.co/api/datasets/{DATASET}")
        return str(info.get("sha", ""))
    except Exception:
        return ""


def stratified_sample(rows: list[dict], per_label: int, seed: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        raw_label = str(row.get("final_decision", "")).strip().lower()
        if raw_label not in LABEL_MAP:
            raise ValueError(f"Unexpected PubMedQA label: {raw_label!r}")
        buckets[raw_label].append(row)

    selected: list[dict] = []
    label_sample_order = Counter()
    for raw_label in ("yes", "no", "maybe"):
        bucket = sorted(buckets[raw_label], key=lambda item: int(item["pubid"]))
        if len(bucket) < per_label:
            raise ValueError(f"Not enough {raw_label} examples: {len(bucket)} < {per_label}")
        rng = random.Random(f"{seed}:sample:{raw_label}")
        for row in rng.sample(bucket, per_label):
            item = dict(row)
            item["raw_label"] = raw_label
            item["gold_label"] = LABEL_MAP[raw_label]
            label_sample_order[item["gold_label"]] += 1
            item["label_sample_order"] = label_sample_order[item["gold_label"]]
            selected.append(item)

    ordering_rng = random.Random(f"{seed}:sample-order")
    ordering_rng.shuffle(selected)
    for index, row in enumerate(selected, start=1):
        row["sample_order"] = index
    return selected


def generation_prompt(rows: list[dict]) -> str:
    items = [{"pubid": int(row["pubid"]), "question": str(row["question"]).strip()} for row in rows]
    if any(set(item) != {"pubid", "question"} for item in items):
        raise AssertionError("Generation items may contain only pubid and question fields")
    prompt = (
        "Rewrite each biomedical yes/no/maybe question as one atomic declarative proposition whose truth is "
        "exactly what the question asks. Do not answer the question and do not add facts. Preserve negation, "
        "modality, comparisons, populations, interventions, outcomes, time qualifiers, numbers, and biomedical "
        "names. Use 5-60 words, one sentence, and no question mark.\n\n"
        "Return only valid JSON in this exact shape:\n"
        '{"items":[{"pubid":123,"claim":"One declarative proposition."}]}\n\n'
        "Items:\n" + json.dumps(items, ensure_ascii=False)
    )
    return prompt


def parse_json_object(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM response is not a JSON object")
    return value


def parse_conversion_response(text: str, expected_pubids: set[int]) -> dict[int, str]:
    data = parse_json_object(text)
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("LLM response does not contain an items list")
    output: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict) or "pubid" not in item or "claim" not in item:
            raise ValueError("Malformed conversion item")
        pubid = int(item["pubid"])
        if pubid in output:
            raise ValueError(f"Duplicate pubid in response: {pubid}")
        if pubid not in expected_pubids:
            raise ValueError(f"Unexpected pubid in response: {pubid}")
        output[pubid] = re.sub(r"\s+", " ", str(item["claim"])).strip()
    if set(output) != expected_pubids:
        missing = sorted(expected_pubids - set(output))
        raise ValueError(f"Response is missing pubids: {missing}")
    return output


def call_conversion_api(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_retries: int,
) -> str:
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Conversion request failed after {max_retries} attempts: {last_error}")


def normalized_claim(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def lexical_tokens(text: str) -> set[str]:
    return {
        token.lower() for token in re.findall(r"[A-Za-z0-9]+", text)
        if token.lower() not in STOPWORDS and len(token) > 1
    }


def numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", text))


def acronyms(text: str) -> set[str]:
    return {
        value.strip("-")
        for value in re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9-]{1,}(?![A-Za-z0-9])", text)
        if value.strip("-")
    }


def quality_checks(question: str, claim: str, duplicate: bool = False) -> dict:
    word_count = len(re.findall(r"\b[\w'-]+\b", claim))
    sentence_count = len(re.findall(r"[.!?](?:\s|$)", claim.strip()))
    if claim.strip() and sentence_count == 0:
        sentence_count = 1
    q_tokens = lexical_tokens(question)
    c_tokens = lexical_tokens(claim)
    overlap = len(q_tokens & c_tokens) / len(q_tokens) if q_tokens else 1.0
    missing_numbers = sorted(numbers(question) - numbers(claim))
    missing_acronyms = sorted(acronyms(question) - acronyms(claim))
    checks = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "content_token_overlap": round(overlap, 4),
        "missing_numbers": "|".join(missing_numbers),
        "missing_acronyms": "|".join(missing_acronyms),
        "is_empty": not bool(claim.strip()),
        "has_question_mark": "?" in claim,
        "length_ok": 5 <= word_count <= 60,
        "single_sentence": sentence_count == 1,
        "numbers_preserved": not missing_numbers,
        "acronyms_preserved": not missing_acronyms,
        "content_overlap_ok": overlap >= 0.45,
        "duplicate_claim": duplicate,
    }
    failures = [
        name for name in (
            "is_empty", "has_question_mark", "length_ok", "single_sentence", "numbers_preserved",
            "acronyms_preserved", "content_overlap_ok", "duplicate_claim",
        )
        if (checks[name] if name in {"is_empty", "has_question_mark", "duplicate_claim"} else not checks[name])
    ]
    checks["automatic_check_pass"] = not failures
    checks["automatic_check_flags"] = "|".join(failures)
    return checks


def needs_retry(question: str, claim: str) -> bool:
    checks = quality_checks(question, claim)
    return any(
        flag in checks["automatic_check_flags"].split("|")
        for flag in (
            "is_empty", "has_question_mark", "length_ok", "single_sentence", "numbers_preserved",
            "acronyms_preserved",
        )
    )


def append_log(path: Path, row: dict) -> None:
    write_jsonl(path, [row], mode="a")


def convert_rows(
    rows: list[dict],
    output_dir: Path,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    timeout: int,
    max_retries: int,
    retry_rounds: int,
    caller: Callable[[str, str, str, str, int, int], str] = call_conversion_api,
) -> dict[int, dict]:
    result_path = output_dir / "pubmedqa_claim_conversion_results.jsonl"
    log_path = output_dir / "pubmedqa_claim_conversion_requests.jsonl"
    existing_rows = read_jsonl(result_path)
    converted = {int(row["pubid"]): row for row in existing_rows if str(row.get("claim", "")).strip()}
    by_pubid = {int(row["pubid"]): row for row in rows}

    def run_batch(batch: list[dict], round_number: int, batch_number: int) -> None:
        prompt = generation_prompt(batch)
        started = time.time()
        error = ""
        raw = ""
        try:
            raw = caller(prompt, api_key, base_url, model, timeout, max_retries)
            parsed = parse_conversion_response(raw, {int(row["pubid"]) for row in batch})
            result_rows = []
            for row in batch:
                pubid = int(row["pubid"])
                entry = {
                    "pubid": pubid,
                    "claim": parsed[pubid],
                    "conversion_model": model,
                    "conversion_batch": batch_number,
                    "conversion_attempt": round_number,
                    "prompt_sha256": sha256_text(prompt),
                }
                converted[pubid] = entry
                result_rows.append(entry)
            write_jsonl(result_path, result_rows, mode="a")
        except Exception as exc:
            error = str(exc)
        append_log(log_path, {
            "round": round_number,
            "batch": batch_number,
            "pubids": [int(row["pubid"]) for row in batch],
            "questions": [str(row["question"]) for row in batch],
            "prompt_sha256": sha256_text(prompt),
            "prompt_chars": len(prompt),
            "forbidden_field_check": True,
            "response_sha256": sha256_text(raw) if raw else "",
            "latency_seconds": round(time.time() - started, 4),
            "success": not bool(error),
            "error": error,
        })
        if error:
            print(json.dumps({"batch": batch_number, "round": round_number, "error": error}, ensure_ascii=False))

    pending = [row for row in rows if int(row["pubid"]) not in converted]
    batch_number = 0
    for start in range(0, len(pending), batch_size):
        batch_number += 1
        run_batch(pending[start:start + batch_size], 1, batch_number)
        print(json.dumps({"round": 1, "batch": batch_number, "converted": len(converted)}, ensure_ascii=False))

    for round_number in range(2, retry_rounds + 2):
        retry_items = [
            by_pubid[pubid] for pubid in sorted(by_pubid)
            if pubid not in converted or needs_retry(str(by_pubid[pubid]["question"]), str(converted[pubid]["claim"]))
        ]
        if not retry_items:
            break
        for row in retry_items:
            batch_number += 1
            run_batch([row], round_number, batch_number)

    missing = sorted(set(by_pubid) - set(converted))
    if missing:
        raise RuntimeError(f"No conversion result for {len(missing)} pubids: {missing[:10]}")
    return converted


def joined_source(row: dict) -> str:
    context = row.get("context") or {}
    values = context.get("contexts", []) if isinstance(context, dict) else []
    return "\n\n".join(re.sub(r"\s+", " ", str(value)).strip() for value in values if str(value).strip())


def build_candidates(rows: list[dict], converted: dict[int, dict], model: str) -> list[dict]:
    claims = Counter(normalized_claim(str(converted[int(row["pubid"])]["claim"])) for row in rows)
    output = []
    for row in sorted(rows, key=lambda item: int(item["sample_order"])):
        pubid = int(row["pubid"])
        conversion = converted[pubid]
        claim = str(conversion["claim"]).strip()
        norm = normalized_claim(claim)
        item = {
            "id": f"pubmedqa_{pubid}",
            "pubid": pubid,
            "dataset": "pubmedqa_claim",
            "split": "pqa_labeled_train",
            "question": str(row["question"]).strip(),
            "claim": claim,
            "source": joined_source(row),
            "gold_label": row["gold_label"],
            "raw_label": row["raw_label"],
            "sample_order": row["sample_order"],
            "label_sample_order": row["label_sample_order"],
            "conversion_model": conversion.get("conversion_model", model),
            "conversion_batch": conversion.get("conversion_batch", ""),
            "conversion_attempt": conversion.get("conversion_attempt", ""),
        }
        item.update(quality_checks(item["question"], claim, duplicate=bool(norm and claims[norm] > 1)))
        output.append(item)
    return output


def build_audit_sample(candidates: list[dict], per_label: int, seed: int) -> list[dict]:
    output = []
    for label in LABEL_ORDER:
        pool = sorted((row for row in candidates if row["gold_label"] == label), key=lambda row: row["pubid"])
        rng = random.Random(f"{seed}:audit:{label}")
        output.extend(rng.sample(pool, per_label))
    random.Random(f"{seed}:audit-order").shuffle(output)
    return [
        {
            "audit_order": index,
            "id": row["id"],
            "pubid": row["pubid"],
            "gold_label": row["gold_label"],
            "question": row["question"],
            "claim": row["claim"],
            "automatic_check_pass": row["automatic_check_pass"],
            "automatic_check_flags": row["automatic_check_flags"],
            "claim_faithfulness": "",
            "atomicity": "",
            "notes": "",
        }
        for index, row in enumerate(output, start=1)
    ]


def validate_outputs(candidates: list[dict], audit: list[dict]) -> dict:
    if len(candidates) != 300:
        raise ValueError(f"Expected 300 candidates, found {len(candidates)}")
    label_counts = Counter(row["gold_label"] for row in candidates)
    if label_counts != Counter({label: 100 for label in LABEL_ORDER}):
        raise ValueError(f"Candidate label imbalance: {dict(label_counts)}")
    if len({row["pubid"] for row in candidates}) != 300:
        raise ValueError("Candidate pubids are not unique")
    if len({normalized_claim(row["claim"]) for row in candidates}) != 300:
        raise ValueError("Candidate claims are not unique")
    if len(audit) != 60:
        raise ValueError(f"Expected 60 audit rows, found {len(audit)}")
    audit_counts = Counter(row["gold_label"] for row in audit)
    if audit_counts != Counter({label: 20 for label in LABEL_ORDER}):
        raise ValueError(f"Audit label imbalance: {dict(audit_counts)}")
    candidate_ids = {row["id"] for row in candidates}
    if not {row["id"] for row in audit}.issubset(candidate_ids):
        raise ValueError("Audit sample is not a subset of Claim-300")
    return {
        "candidate_count": len(candidates),
        "candidate_label_counts": dict(label_counts),
        "unique_pubids": len({row["pubid"] for row in candidates}),
        "unique_claims": len({normalized_claim(row["claim"]) for row in candidates}),
        "automatic_check_pass_count": sum(bool(row["automatic_check_pass"]) for row in candidates),
        "automatic_check_pass_rate": round(sum(bool(row["automatic_check_pass"]) for row in candidates) / 300, 4),
        "automatic_check_flag_counts": dict(Counter(
            flag for row in candidates for flag in str(row["automatic_check_flags"]).split("|") if flag
        )),
        "audit_count": len(audit),
        "audit_label_counts": dict(audit_counts),
        "audit_is_subset": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-controlled PubMedQA-Claim-300 and audit60.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--per-label", type=int, default=100)
    parser.add_argument("--audit-per-label", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-v4-flash")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("LLM_TIMEOUT_SECONDS", "90")))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("LLM_MAX_RETRIES", "3")))
    parser.add_argument("--retry-rounds", type=int, default=2)
    parser.add_argument("--refresh-download", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY or LLM_API_KEY")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, download_stats = download_pubmedqa(output_dir / "pubmedqa_pqa_labeled_raw.jsonl", args.refresh_download)
    sampled = stratified_sample(raw_rows, args.per_label, args.seed)
    write_jsonl(output_dir / "pubmedqa_sampled300_private.jsonl", sampled)

    converted = convert_rows(
        sampled, output_dir, api_key, args.base_url, args.model, args.batch_size, args.timeout,
        args.max_retries, args.retry_rounds,
    )
    candidates = build_candidates(sampled, converted, args.model)
    audit = build_audit_sample(candidates, args.audit_per_label, args.seed)
    stats = validate_outputs(candidates, audit)

    write_csv(output_dir / "pubmedqa_claim_candidates.csv", candidates, CANDIDATE_FIELDS)
    claim_rows = [
        {
            "id": row["id"],
            "dataset": row["dataset"],
            "split": row["split"],
            "claim": row["claim"],
            "source": row["source"],
            "label": row["gold_label"],
            "gold_label": row["gold_label"],
            "raw_label": row["raw_label"],
            "pubid": row["pubid"],
            "question": row["question"],
            "provenance": {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "conversion_model": row["conversion_model"],
                "conversion_prompt_excludes_answer_fields": True,
            },
            "automatic_quality": {
                "pass": row["automatic_check_pass"],
                "flags": row["automatic_check_flags"],
                "content_token_overlap": row["content_token_overlap"],
            },
        }
        for row in candidates
    ]
    write_jsonl(output_dir / "pubmedqa_claim300.jsonl", claim_rows)
    write_csv(output_dir / "pubmedqa_claim_audit60.csv", audit, AUDIT_FIELDS)

    sample_fingerprint = sha256_text("\n".join(str(row["pubid"]) for row in sampled))
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "dataset_revision": dataset_revision(),
        "dataset_server": DATASET_SERVER,
        "seed": args.seed,
        "sample_per_label": args.per_label,
        "audit_per_label": args.audit_per_label,
        "batch_size": args.batch_size,
        "model": args.model,
        "prompt_sha256_example": sha256_text(generation_prompt(sampled[:args.batch_size])),
        "generation_input_fields": ["pubid", "question"],
        "forbidden_generation_fields": list(FORBIDDEN_GENERATION_FIELDS),
        "sample_pubid_fingerprint": sample_fingerprint,
        "download": download_stats,
        "outputs": {
            "candidates": str(output_dir / "pubmedqa_claim_candidates.csv"),
            "claim300": str(output_dir / "pubmedqa_claim300.jsonl"),
            "audit60": str(output_dir / "pubmedqa_claim_audit60.csv"),
        },
    }
    (output_dir / "pubmedqa_claim300_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "pubmedqa_claim_quality_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({**stats, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
