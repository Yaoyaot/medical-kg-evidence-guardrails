from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from detect_claim_qualifiers import detect_qualifiers
from strict_relation_alignment import NEGATIVE_RELATIONS, relation_family


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


def best_path(paths: list[dict]) -> dict | None:
    return max(paths, key=lambda path: float(path.get("score", 0.0))) if paths else None


def explicit_conflict_path(evidence: dict) -> tuple[dict | None, str]:
    families = set(evidence.get("claim_predicate_families") or [])
    for path in evidence.get("direct_paths") or []:
        relations = path.get("relations") or []
        if len(relations) != 1:
            continue
        relation = relations[0]
        family = relation_family(relation)
        if "TREATS" in families and (relation == "contraindicates" or family == "CONTRAINDICATES"):
            return path, "claim_treats_but_kg_contraindicates"
        if "PRESENTS" in families and relation == "not_presents":
            return path, "claim_presents_but_kg_not_presents"
        if "EXPRESSION" in families and relation == "not_expressed":
            return path, "claim_expression_but_kg_not_expressed"
    return None, ""


def guardrail_status(text_row: dict, evidence: dict) -> tuple[str, dict | None, str, list[str]]:
    qualifiers = detect_qualifiers(str(text_row.get("claim") or ""))
    conflict, reason = explicit_conflict_path(evidence)
    if conflict:
        return "KG_EXPLICIT_CONFLICT", conflict, reason, qualifiers
    direct_paths = evidence.get("direct_paths") or []
    aligned = [path for path in direct_paths if path.get("automatic_support_allowed")]
    if aligned:
        chosen = best_path(aligned)
        if qualifiers:
            return "KG_PARTIAL_DIRECT", chosen, "claim_has_unencoded_qualifiers", qualifiers
        return "KG_DUAL_EVIDENCE", chosen, "", qualifiers
    if direct_paths:
        return "KG_PARTIAL_DIRECT", best_path(direct_paths), "direct_edge_without_full_predicate_alignment", qualifiers
    contexts = evidence.get("two_hop_context_paths") or []
    if contexts:
        return "KG_TWO_HOP_CONTEXT", best_path(contexts), "", qualifiers
    if evidence.get("linked_entities"):
        return "KG_SINGLE_ENTITY_CONTEXT", None, "", qualifiers
    return "KG_NO_GROUNDING", None, "", qualifiers


def build_row(text_row: dict, evidence: dict) -> dict:
    status, path, reason, qualifiers = guardrail_status(text_row, evidence)
    final_label = text_row["pred_label"]
    action = "ACCEPT_TEXT_RAG"
    confidence_tier = "TEXT_ONLY"
    if status == "KG_DUAL_EVIDENCE":
        confidence_tier = "HIGH_TEXT_KG_DUAL"
    elif status == "KG_EXPLICIT_CONFLICT":
        final_label = "UNCERTAIN"
        action = "REVIEW_REQUIRED"
        confidence_tier = "CONFLICT_REVIEW"
    elif status == "KG_PARTIAL_DIRECT":
        confidence_tier = "TEXT_WITH_PARTIAL_KG"
    elif status in {"KG_TWO_HOP_CONTEXT", "KG_SINGLE_ENTITY_CONTEXT"}:
        confidence_tier = "TEXT_WITH_KG_CONTEXT"
    return {
        "id": text_row["id"],
        "dataset": text_row.get("dataset"),
        "claim": text_row.get("claim"),
        "gold_label": text_row.get("gold_label"),
        "candidate_label": text_row.get("pred_label"),
        "final_label": final_label,
        "text_confidence": float(text_row.get("confidence", 0.0)),
        "guardrail_status": status,
        "action": action,
        "confidence_tier": confidence_tier,
        "kg_evidence_tier": evidence.get("primary_evidence_tier"),
        "kg_path": path,
        "kg_conflict_reason": reason,
        "qualifier_flags": qualifiers,
        "text_reasoning": text_row.get("reasoning"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Text-RAG predictions through a strict KG risk guardrail.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--llm-input-name", default="llm_baseline_results_deepseek-v4-flash.jsonl")
    parser.add_argument("--evidence-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default="pilot150")
    args = parser.parse_args()

    processed = Path(args.data_dir) / "processed"
    llm_rows = [
        row
        for row in read_jsonl(processed / args.llm_input_name)
        if row.get("baseline") == "text_rag_llm"
    ]
    evidence_path = Path(args.evidence_path) if args.evidence_path else processed / "stage3_strict_verifier/strict_kg_evidence.jsonl"
    evidence = {
        row["id"]: row
        for row in read_jsonl(evidence_path)
    }
    missing = [row["id"] for row in llm_rows if row["id"] not in evidence]
    if missing:
        raise RuntimeError(f"Missing strict KG evidence for {len(missing)} rows: {missing[:5]}")
    rows = [build_row(row, evidence[row["id"]]) for row in llm_rows]
    output_dir = Path(args.output_dir) if args.output_dir else processed / "stage4_kg_guardrail"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "guardrail_results.jsonl"
    write_jsonl(output_path, rows)
    stats = {
        "input_rows": len(rows),
        "run_name": args.run_name,
        "evidence_path": str(evidence_path),
        "guardrail_status_counts": dict(Counter(row["guardrail_status"] for row in rows)),
        "action_counts": dict(Counter(row["action"] for row in rows)),
        "confidence_tier_counts": dict(Counter(row["confidence_tier"] for row in rows)),
        "qualifier_flag_counts": dict(Counter(flag for row in rows for flag in row["qualifier_flags"])),
        "output": str(output_path),
    }
    (output_dir / "guardrail_status_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
