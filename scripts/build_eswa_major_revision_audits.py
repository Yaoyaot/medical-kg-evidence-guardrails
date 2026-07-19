from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260618


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_key(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def pubmedqa_audit(stage8: Path, output: Path) -> dict:
    prior = read_csv(stage8 / "pubmedqa_claim_audit60.csv")
    candidates = {row["id"]: row for row in read_jsonl(stage8 / "pubmedqa_claim300.jsonl")}
    if len(prior) != 60:
        raise ValueError("Expected existing PubMedQA audit60")
    rows = []
    for index, item in enumerate(prior, start=1):
        source = candidates[item["id"]]
        rows.append({
            "audit_order": index,
            "id": item["id"],
            "pubid": item["pubid"],
            "raw_label": source.get("raw_label", ""),
            "mapped_label": item["gold_label"],
            "question": item["question"],
            "converted_claim": item["claim"],
            "source_context": source.get("source", ""),
            "claim_faithfulness": "",
            "atomicity": "",
            "label_compatibility": "",
            "pico_preservation": "",
            "modality_strength": "",
            "notes": "",
        })
    for annotator in ("A", "B"):
        write_csv(output / f"pubmedqa_label_mapping_audit60_annotator_{annotator.lower()}.csv", rows)
    write_csv(output / "pubmedqa_label_mapping_audit60_adjudication.csv", [
        {
            **{key: value for key, value in row.items() if key not in {"claim_faithfulness", "atomicity", "label_compatibility", "pico_preservation", "modality_strength", "notes"}},
            "annotator_a_claim_faithfulness": "",
            "annotator_b_claim_faithfulness": "",
            "final_claim_faithfulness": "",
            "annotator_a_atomicity": "",
            "annotator_b_atomicity": "",
            "final_atomicity": "",
            "annotator_a_label_compatibility": "",
            "annotator_b_label_compatibility": "",
            "final_label_compatibility": "",
            "annotator_a_pico_preservation": "",
            "annotator_b_pico_preservation": "",
            "final_pico_preservation": "",
            "annotator_a_modality_strength": "",
            "annotator_b_modality_strength": "",
            "final_modality_strength": "",
            "adjudication_notes": "",
        }
        for row in rows
    ])
    return {"rows": len(rows), "labels": dict(Counter(row["mapped_label"] for row in rows))}


def entity_audit(stage8: Path, group_manifest: Path, formal_evidence_path: Path, output: Path) -> dict:
    formal_manifest = read_csv(group_manifest)
    formal_evidence = {row["id"]: row for row in read_jsonl(formal_evidence_path)}
    pubmedqa_claims = {row["id"]: row for row in read_jsonl(stage8 / "pubmedqa_claim300.jsonl")}
    pubmedqa_evidence = {row["id"]: row for row in read_jsonl(stage8 / "kg/strict_verifier/strict_kg_evidence.jsonl")}
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in formal_manifest:
        evidence = formal_evidence[row["id"]]
        by_dataset[row["dataset"]].append({
            "id": row["id"],
            "dataset": row["dataset"],
            "claim": row["claim"],
            "gold_label": row["gold_label"],
            "predicted_links": evidence.get("linked_entities") or [],
        })
    for claim_id, source in pubmedqa_claims.items():
        evidence = pubmedqa_evidence[claim_id]
        by_dataset["pubmedqa"].append({
            "id": claim_id,
            "dataset": "pubmedqa",
            "claim": source["claim"],
            "gold_label": source["label"],
            "predicted_links": evidence.get("linked_entities") or [],
        })
    expected = {"scifact", "healthver", "medaesqa", "pubmedqa"}
    if set(by_dataset) != expected:
        raise ValueError(f"Unexpected entity-audit datasets: {set(by_dataset)}")

    selected = []
    for dataset in sorted(expected):
        candidates = sorted(by_dataset[dataset], key=lambda row: stable_key(row["id"]))
        linked = [row for row in candidates if row["predicted_links"]]
        unlinked = [row for row in candidates if not row["predicted_links"]]
        chosen = linked[:15] + unlinked[:15]
        if len(chosen) < 30:
            used = {row["id"] for row in chosen}
            chosen.extend(row for row in candidates if row["id"] not in used and len(chosen) < 30)
        selected.extend(chosen)
    selected.sort(key=lambda row: (row["dataset"], stable_key(row["id"])))
    audit_rows = []
    for order, row in enumerate(selected, start=1):
        links = row.pop("predicted_links")
        audit_rows.append({
            "audit_order": order,
            **row,
            "predicted_link_count": len(links),
            "predicted_links_json": json.dumps(links, ensure_ascii=False),
            "gold_biomedical_mentions_json": "",
            "gold_concept_links_json": "",
            "missed_mentions_json": "",
            "incorrect_predicted_links_json": "",
            "abbreviation_ambiguity": "",
            "overall_linking_judgment": "",
            "notes": "",
        })
    if len(audit_rows) != 120:
        raise ValueError(f"Expected 120 entity audit rows, found {len(audit_rows)}")
    for annotator in ("A", "B"):
        write_csv(output / f"entity_linking_audit120_annotator_{annotator.lower()}.csv", audit_rows)
    write_csv(output / "entity_linking_audit120_adjudication.csv", [
        {
            **{key: value for key, value in row.items() if key not in {"gold_biomedical_mentions_json", "gold_concept_links_json", "missed_mentions_json", "incorrect_predicted_links_json", "abbreviation_ambiguity", "overall_linking_judgment", "notes"}},
            "annotator_a_gold_mentions_json": "",
            "annotator_b_gold_mentions_json": "",
            "final_gold_mentions_json": "",
            "annotator_a_gold_concepts_json": "",
            "annotator_b_gold_concepts_json": "",
            "final_gold_concepts_json": "",
            "final_incorrect_predicted_links_json": "",
            "final_abbreviation_ambiguity": "",
            "adjudication_notes": "",
        }
        for row in audit_rows
    ])
    return {
        "rows": len(audit_rows),
        "datasets": dict(Counter(row["dataset"] for row in audit_rows)),
        "linked_status": dict(Counter("linked" if int(row["predicted_link_count"]) else "unlinked" for row in audit_rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dual-annotator Stage 9 PubMedQA mapping and entity-linking audit packages.")
    parser.add_argument("--stage8-dir", default="data/processed/stage8_pubmedqa_external")
    parser.add_argument("--group-manifest", default="data/processed/stage9_eswa_major_revision/grouping/formal600_group_manifest.csv")
    parser.add_argument("--formal-evidence", default="data/processed/stage2_primekg_semantic_clean/strict_verifier/strict_kg_evidence.jsonl")
    parser.add_argument("--output-dir", default="data/processed/stage9_eswa_major_revision/human_audits")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    modeling_pool = read_csv(Path("data/processed/stage6_eswa/annotations_gold/path_annotations_modeling_pool.csv"))
    dual_pool = [row for row in modeling_pool if row.get("gold_source") in {"dual_agreed", "dual_adjudicated"}]
    if len(dual_pool) != 224:
        raise ValueError(f"Expected 224 dual-agreed/adjudicated paths, found {len(dual_pool)}")
    write_csv(output / "dual_agreed_or_adjudicated_path_pool224.csv", dual_pool)
    stats = {
        "seed": SEED,
        "pubmedqa_label_mapping": pubmedqa_audit(Path(args.stage8_dir), output),
        "entity_linking": entity_audit(Path(args.stage8_dir), Path(args.group_manifest), Path(args.formal_evidence), output),
        "path_sensitivity_pool": {"rows": len(dual_pool), "sources": dict(Counter(row.get("gold_source", "") for row in dual_pool))},
        "independent_annotation_required": True,
        "gold_fields_must_be_completed_before_metrics": True,
    }
    guideline = """# Stage 9 dual-annotator audit instructions

## Independence

Annotators A and B complete their files independently and must not inspect the other annotator's answers. Adjudication begins only after both files are frozen.

## PubMedQA label-mapping audit

- `claim_faithfulness`: VALID or INVALID. The declarative claim must preserve the question proposition.
- `atomicity`: ATOMIC or NON_ATOMIC.
- `label_compatibility`: COMPATIBLE, INCOMPATIBLE, or AMBIGUOUS. Judge whether yes/no/maybe, after mapping, functions as SUPPORT/REFUTE/UNCERTAIN for the converted claim given the supplied context.
- `pico_preservation`: COMPLETE, PARTIAL, or NOT_APPLICABLE. Check population, intervention/exposure, comparator, and outcome when present.
- `modality_strength`: PRESERVED, WEAKENED, STRENGTHENED, or CHANGED. Pay special attention to association, prediction, causality, necessity, and uncertainty.

## Entity-linking audit

Annotate every biomedical mention in the claim, including mentions missed by the system. JSON fields must contain valid JSON arrays. A concept record should use `{"mention":"...","concept_id":"...","entity_type":"..."}`. Record incorrect model links separately, and mark abbreviation ambiguity YES or NO. `overall_linking_judgment` is CORRECT, PARTIAL, or INCORRECT.

Do not infer clinical truth. Judge conversion/label semantics or entity identity only.
"""
    (output / "stage9_dual_annotation_guideline.md").write_text(guideline, encoding="utf-8")
    (output / "stage9_human_audit_manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
