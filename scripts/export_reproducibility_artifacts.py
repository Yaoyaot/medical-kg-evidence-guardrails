from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from repo_paths import find_repo_root


ROOT = find_repo_root()
METHOD_LABELS = {
    "direct_llm": "direct_llm",
    "text_rag_llm": "provided_text",
    "provided_text_bm25_llm": "provided_text_plus_extra_bm25",
    "provided_text_kg_llm": "provided_text_plus_kg_paths",
    "medgraphrag_style_llm": "provided_text_plus_kg_plus_extra_bm25",
    "vanilla_graphrag_llm": "kg_only_local_paths_diagnostic",
    "bm25_text_rag_llm": "bm25_only_diagnostic",
}
LABELS = {"SUPPORT", "REFUTE", "UNCERTAIN"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(data)


def text_hash(value: object) -> str:
    """Hash upstream text without redistributing the text itself."""
    return sha256_bytes(str(value or "").encode("utf-8"))


def detect_encoding(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            path.read_text(encoding=encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Unable to decode {path}")


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding=detect_encoding(path), newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding=detect_encoding(path)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )


def copy_text_normalized(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding=detect_encoding(source))
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.replace("\r\n", "\n").replace("\r", "\n"))


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def clean_error(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "none", "null", "false"} else text


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def export_splits(organized: Path, output: Path) -> dict[str, dict]:
    source = require(
        organized
        / "data/processed/stage9_eswa_major_revision/grouping/formal600_group_manifest.csv"
    )
    rows = read_csv(source)
    selection_order = {}
    for label in LABELS:
        label_ids = sorted(row["id"] for row in rows if row["gold_label"] == label)
        selection_order.update(
            {record_id: index + 1 for index, record_id in enumerate(label_ids)}
        )
    membership = []
    for row in rows:
        membership.append(
            {
                "record_id": row["id"],
                "source_record_id": row["id"],
                "dataset": row["dataset"],
                "frozen_gold_label": row["gold_label"],
                "sampling_frame": "standardized_claim_evidence_pool_6454",
                "selection_rule": "lexicographic_record_id_within_label_first_200",
                "selection_order_within_label": selection_order[row["id"]],
                "normalized_claim_sha256": row["normalized_claim_sha256"],
                "normalized_source_sha256": row["normalized_source_sha256"],
                "component_id": row["pair_group_id"],
                "outer_fold": int(row["outer_fold"]),
                "near_duplicate_threshold": float(row["near_duplicate_threshold"]),
                "selection_seed": int(row["selection_seed"]),
            }
        )
    membership.sort(key=lambda row: row["record_id"])
    target = output / "data_splits/formal600_membership.csv"
    write_csv(target, membership)

    outer = [
        {
            "record_id": row["record_id"],
            "component_id": row["component_id"],
            "outer_fold": row["outer_fold"],
            "selection_seed": row["selection_seed"],
        }
        for row in membership
    ]
    write_csv(output / "data_splits/formal600_outer_folds.csv", outer)

    component_rows = []
    by_component: dict[str, list[dict]] = defaultdict(list)
    for row in membership:
        by_component[row["component_id"]].append(row)
    for component_id, component_members in sorted(by_component.items()):
        claim_counts = Counter(
            row["normalized_claim_sha256"] for row in component_members
        )
        source_counts = Counter(
            row["normalized_source_sha256"] for row in component_members
        )
        component_rows.append(
            {
                "component_id": component_id,
                "component_size": len(component_members),
                "outer_fold": component_members[0]["outer_fold"],
                "record_ids": ";".join(
                    sorted(row["record_id"] for row in component_members)
                ),
                "datasets": ";".join(
                    sorted({row["dataset"] for row in component_members})
                ),
                "exact_claim_hash_collision": any(
                    count > 1 for count in claim_counts.values()
                ),
                "exact_source_hash_collision": any(
                    count > 1 for count in source_counts.values()
                ),
                "construction_rule": (
                    "connected_components(exact_claim OR exact_source OR "
                    "claim_char_ngram_tfidf_cosine>=0.90)"
                ),
            }
        )
    write_csv(output / "data_splits/claim_component_map.csv", component_rows)

    provenance_source = require(
        organized
        / "data/processed/stage11_eswa_nested_crossfit/formal600/"
        "evidence_scorer_feature_provenance.csv"
    )
    provenance = read_csv(provenance_source)
    inner = [
        {
            "outer_fold": int(row["outer_fold"]),
            "record_id": row["record_id"],
            "record_fold": int(row["record_fold"]),
            "feature_role": row["feature_role"],
            "scorer_excluded_folds": row["scorer_excluded_folds"],
            "own_fold_excluded": truthy(row["own_fold_excluded"]),
            "outer_test_fold_excluded": truthy(row["outer_test_fold_excluded"]),
        }
        for row in provenance
    ]
    write_csv(output / "data_splits/formal600_inner_feature_assignments.csv", inner)
    write_csv(output / "data_splits/formal600_inner_folds.csv", inner)

    path_map_source = require(
        organized
        / "data/processed/stage11_eswa_nested_crossfit/formal600/"
        "path_annotation_component_map.csv"
    )
    path_map = read_csv(path_map_source)
    write_csv(output / "data_splits/path_component_map.csv", path_map)

    scorer_source = require(
        organized
        / "data/processed/stage11_eswa_nested_crossfit/formal600/"
        "path_scorer_nested_manifest.csv"
    )
    scorer_rows = read_csv(scorer_source)
    write_csv(output / "data_splits/scorer_exclusion_manifest.csv", scorer_rows)
    annotations_by_fold: dict[str, list[str]] = defaultdict(list)
    for row in path_map:
        for fold in str(row.get("linked_outer_folds") or "").split(";"):
            fold = fold.strip()
            if fold:
                annotations_by_fold[fold].append(row["annotation_id"])
    scorer_json = {
        "manifest_type": "outer-fold Evidence Scorer exclusion",
        "reason": (
            "path annotation linked to a claim/source component assigned to "
            "the outer test fold"
        ),
        "folds": [],
    }
    for scorer_row in scorer_rows:
        excluded_key = str(scorer_row["excluded_outer_folds"])
        excluded_folds = [
            fold.strip() for fold in excluded_key.split(";") if fold.strip()
        ]
        excluded_ids = sorted(
            {
                annotation_id
                for fold in excluded_folds
                for annotation_id in annotations_by_fold.get(fold, [])
            }
        )
        scorer_json["folds"].append(
            {
                "scorer_key": scorer_row["scorer_key"],
                "excluded_component_folds": [int(fold) for fold in excluded_folds],
                "excluded_annotation_ids": excluded_ids,
                "excluded_count": int(scorer_row["excluded_annotation_rows"]),
                "retained_count": int(scorer_row["training_annotation_rows"]),
                "test_linked_annotations_in_training": int(
                    scorer_row["test_linked_annotations_in_training"]
                ),
                "relevance_model_sha256": scorer_row["relevance_model_sha256"],
                "actionability_model_sha256": scorer_row[
                    "actionability_model_sha256"
                ],
            }
        )
    write_json(
        output / "data_splits/scorer_exclusion_manifest.json", scorer_json
    )

    stats_source = require(
        organized
        / "data/processed/stage11_eswa_nested_crossfit/formal600/"
        "strict_nested_crossfit_stats.json"
    )
    stats = json.loads(stats_source.read_text(encoding="utf-8"))
    manifest = {
        "records": len(membership),
        "components": len({row["component_id"] for row in membership}),
        "outer_fold_counts": dict(
            sorted(Counter(str(row["outer_fold"]) for row in membership).items())
        ),
        "feature_assignment_rows": len(inner),
        "path_annotation_rows": stats["path_annotations"],
        "all_feature_rows_exclude_own_component_fold": stats[
            "all_feature_rows_exclude_own_component_fold"
        ],
        "all_risk_training_features_exclude_outer_test_fold": stats[
            "all_risk_training_features_exclude_outer_test_fold"
        ],
        "seed": 20260618,
        "sampling_frame": "standardized_claim_evidence_pool_6454",
        "formal600_selection_rule": (
            "sort record IDs lexicographically within each frozen label and "
            "take the first 200"
        ),
        "component_rule": (
            "connected components over exact normalized claim, exact normalized "
            "source, or claim character n-gram TF-IDF cosine similarity >=0.90"
        ),
        "inner_assignment_note": (
            "formal600_inner_folds.csv records nested Evidence-Scorer feature "
            "assignments; it is not a newly sampled partition"
        ),
    }
    write_json(output / "data_splits/sampling_and_crossfit_manifest.json", manifest)
    write_json(output / "data_splits/sampling_manifest.json", manifest)
    files = sorted((output / "data_splits").glob("*"))
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    }


def dedupe_prediction_sources(paths: list[Path]) -> list[dict]:
    rows: dict[tuple[str, str], dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            baseline = str(row.get("baseline") or "")
            identifier = str(row.get("id") or "")
            if baseline and identifier:
                rows[(baseline, identifier)] = row
    return list(rows.values())


def public_predictions(
    rows: list[dict], domain: str, fold_by_id: dict[str, int]
) -> list[dict]:
    output = []
    for row in rows:
        baseline = row.get("baseline")
        if baseline not in METHOD_LABELS:
            continue
        identifier = str(row["id"])
        gold = str(row["gold_label"])
        pred = str(row["pred_label"])
        if gold not in LABELS or pred not in LABELS:
            raise ValueError(f"Unexpected label in {domain}/{baseline}/{identifier}")
        parsed_record = {
            "id": identifier,
            "method": METHOD_LABELS[baseline],
            "predicted_label": pred,
            "confidence": float(row.get("confidence") or 0.0),
        }
        output.append(
            {
                "domain": domain,
                "record_id": identifier,
                "dataset": row.get("dataset") or domain,
                "method": METHOD_LABELS[baseline],
                "source_baseline_name": baseline,
                "model_identifier": row.get("model") or "deepseek-v4-flash",
                "outer_fold": fold_by_id.get(identifier),
                "gold_label": gold,
                "predicted_label": pred,
                "confidence": float(row.get("confidence") or 0.0),
                "correct": pred == gold,
                "support_prediction": pred == "SUPPORT",
                "false_support": pred == "SUPPORT" and gold != "SUPPORT",
                "parse_error": truthy(row.get("parse_error")),
                "request_error": clean_error(row.get("request_error")),
                "prompt_sha256": row.get("prompt_sha256") or "",
                "prompt_chars": int(row.get("prompt_chars") or 0),
                "request_count": int(row.get("request_count") or 1),
                "parsed_prediction_sha256": canonical_hash(parsed_record),
            }
        )
    output.sort(key=lambda row: (row["method"], row["record_id"]))
    return output


def export_predictions(
    workspace: Path, organized: Path, output: Path
) -> dict[str, dict]:
    membership = read_csv(output / "data_splits/formal600_membership.csv")
    fold_by_id = {row["record_id"]: int(row["outer_fold"]) for row in membership}

    formal_sources = [
        organized / "data/processed/llm_baseline_results_deepseek-v4-flash-formal600.jsonl",
        organized
        / "data/processed/stage7_hierarchical_scorer/formal600/"
        "eswa_enhanced_baselines_formal600.jsonl",
        *sorted(
            (
                organized
                / "data/processed/stage9_eswa_major_revision/fair_input_baselines"
            ).glob("formal600_*.jsonl")
        ),
    ]
    formal = public_predictions(
        dedupe_prediction_sources(formal_sources), "formal600", fold_by_id
    )
    write_jsonl(output / "predictions/formal600_predictions.jsonl", formal)

    pub_sources = [
        organized
        / "data/processed/stage8_pubmedqa_external/external_baseline_results.jsonl",
        organized
        / "data/processed/stage9_eswa_major_revision/fair_input_baselines/"
        "pubmedqa_provided_text_bm25_llm.jsonl",
        organized
        / "data/processed/stage9_eswa_major_revision/fair_input_baselines/"
        "pubmedqa_provided_text_kg_llm.jsonl",
    ]
    pubmedqa = public_predictions(
        dedupe_prediction_sources(pub_sources), "pubmedqa", {}
    )
    write_jsonl(
        output / "predictions/pubmedqa_claim300_predictions.jsonl", pubmedqa
    )

    oof_source = require(
        workspace
        / "data/processed/stage7_hierarchical_scorer/path_scorer/oof_predictions.jsonl"
    )
    oof = read_jsonl(oof_source)
    oof_public = [
        {
            "annotation_id": row["annotation_id"],
            "claim_id": row["claim_id"],
            "claim_cluster_id": row["claim_cluster_id"],
            "seed": int(row["seed"]),
            "fold": int(row["fold"]),
            "task": row["task"],
            "model": row["model"],
            "gold_label": row["gold_label"],
            "predicted_label": row["pred_label"],
            "probabilities": row["probabilities"],
        }
        for row in oof
    ]
    write_jsonl(
        output / "predictions/evidence_scorer_oof_predictions.jsonl", oof_public
    )

    risk_source = require(
        organized
        / "data/processed/stage11_eswa_nested_crossfit/formal600/"
        "formal600_crossfit_risk_scores.jsonl"
    )
    risk_fields = [
        "id",
        "dataset",
        "gold_label",
        "candidate_label",
        "pair_group_id",
        "outer_fold",
        "text_confidence",
        "guardrail_status",
        "action",
        "path_count",
        "direct_path_count",
        "two_hop_path_count",
        "predicate_aligned_path_count",
        "endpoint_aligned_path_count",
        "max_path_score",
        "linked_entity_count",
        "max_relevant_probability",
        "mean_relevant_probability",
        "max_actionable_probability",
        "mean_actionable_probability",
        "evidence_state_nominal",
        "qualifier_compatible_for_strict_kg",
        "direct_conflict_present",
        "strict_structured_candidate_count",
        "evidence_scorer_excluded_folds",
        "risk_confidence",
        "risk_rule",
        "risk_oracle",
        "risk_confidence_only_features",
        "risk_dataset_source_only",
        "risk_kg_evidence_state_only",
        "risk_semantic_rules_only",
        "risk_path_statistics_only",
        "risk_evidence_scorer_only",
        "risk_confidence_plus_kg_rules",
        "risk_confidence_plus_evidence_scorer",
        "risk_full_without_dataset_source",
        "risk_full_with_dataset_source",
        "risk_random",
    ]
    risk_rows = read_jsonl(risk_source)
    risk_public = [{field: row.get(field) for field in risk_fields} for row in risk_rows]
    write_csv(output / "predictions/risk_routing_scores.csv", risk_public, risk_fields)

    request_hashes = [
        {
            "domain": row["domain"],
            "record_id": row["record_id"],
            "method": row["method"],
            "prompt_sha256": row["prompt_sha256"],
            "prompt_hash_status": "recorded"
            if row["prompt_sha256"]
            else "not_recorded_for_legacy_run",
            "request_payload_sha256": "",
            "request_payload_hash_status": (
                "not_persisted_in_historical_run"
            ),
            "parsed_prediction_sha256": row["parsed_prediction_sha256"],
            "request_status": "failed"
            if row["request_error"] or row["parse_error"]
            else "success",
            "request_count": row["request_count"],
            "retry_count": max(0, int(row["request_count"]) - 1),
        }
        for row in [*formal, *pubmedqa]
    ]
    write_csv(output / "api_manifest/request_hashes.csv", request_hashes)

    files = [
        output / "predictions/formal600_predictions.jsonl",
        output / "predictions/pubmedqa_claim300_predictions.jsonl",
        output / "predictions/evidence_scorer_oof_predictions.jsonl",
        output / "predictions/risk_routing_scores.csv",
    ]
    manifest = {
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "formal600_method_counts": dict(
            sorted(Counter(row["method"] for row in formal).items())
        ),
        "pubmedqa_method_counts": dict(
            sorted(Counter(row["method"] for row in pubmedqa).items())
        ),
        "formal600_expected_rows_per_method": 600,
        "pubmedqa_expected_rows_per_method": 300,
        "formal600_duplicate_method_record_pairs": len(formal)
        - len({(row["method"], row["record_id"]) for row in formal}),
        "pubmedqa_duplicate_method_record_pairs": len(pubmedqa)
        - len({(row["method"], row["record_id"]) for row in pubmedqa}),
        "raw_api_responses_included": False,
        "generated_by": "scripts/export_reproducibility_artifacts.py",
        "release_date": "2026-07-24",
    }
    write_json(output / "predictions/prediction_manifest.json", manifest)
    return {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }


def export_audits(workspace: Path, organized: Path, output: Path) -> dict[str, dict]:
    clean_source = require(
        organized
        / "data/processed/stage7_hierarchical_scorer/path_scorer/"
        "semantic_clean_annotation_pool.csv"
    )
    clean_rows = read_csv(clean_source)
    clean_fields = [
        "annotation_id",
        "claim_id",
        "dataset",
        "gold_label",
        "claim_sha256",
        "path_text",
        "relations",
        "path_type",
        "evidence_tier",
        "claim_predicate_families",
        "kg_relation_family",
        "predicate_aligned",
        "endpoint_aligned",
        "path_score",
        "entity_pair_key",
        "claim_cluster_id",
        "duplicate_group_size",
        "path_relevance",
        "evidence_role",
        "error_type",
        "semantic_relation_remapped",
        "actionability",
    ]
    clean_public = []
    for row in clean_rows:
        public_row = {field: row.get(field, "") for field in clean_fields}
        public_row["claim_sha256"] = text_hash(row.get("claim", ""))
        clean_public.append(public_row)
    write_csv(
        output / "audits/path_annotations_modeling_pool_anonymized.csv",
        clean_public,
        clean_fields,
    )

    overlap_source = require(
        workspace
        / "data/processed/stage6_eswa/annotations_gold/"
        "path_annotations_adjudication_overlap200.csv"
    )
    overlap_rows = read_csv(overlap_source)
    overlap_fields = [
        "annotation_id",
        "claim_id",
        "dataset",
        "annotator_a_path_relevance",
        "annotator_a_evidence_role",
        "annotator_a_error_type",
        "annotator_b_path_relevance",
        "annotator_b_evidence_role",
        "annotator_b_error_type",
        "final_path_relevance",
        "final_evidence_role",
        "final_error_type",
    ]
    write_csv(
        output / "audits/path_overlap200_adjudication_anonymized.csv",
        [{field: row.get(field, "") for field in overlap_fields} for row in overlap_rows],
        overlap_fields,
    )

    artifact_source = require(
        workspace
        / "data/processed/stage6_eswa/annotations_gold/"
        "path_annotations_artifact_error_analysis.csv"
    )
    artifacts = read_csv(artifact_source)
    artifact_fields = [
        "annotation_id",
        "claim_id",
        "dataset",
        "path_relevance",
        "evidence_role",
        "error_type",
        "artifact_excluded",
        "artifact_reasons",
    ]
    write_csv(
        output / "audits/path_exclusion_manifest.csv",
        [{field: row.get(field, "") for field in artifact_fields} for row in artifacts],
        artifact_fields,
    )

    pub_source = require(
        organized
        / "data/processed/stage9_eswa_major_revision/human_audits/"
        "pubmedqa_label_mapping_audit60_adjudication.csv"
    )
    pub_rows = read_csv(pub_source)
    pub_fields = [
        "audit_order",
        "id",
        "pubid",
        "raw_label",
        "mapped_label",
        "question_sha256",
        "converted_claim_sha256",
        "annotator_a_claim_faithfulness",
        "annotator_b_claim_faithfulness",
        "final_claim_faithfulness",
        "annotator_a_atomicity",
        "annotator_b_atomicity",
        "final_atomicity",
        "annotator_a_label_compatibility",
        "annotator_b_label_compatibility",
        "final_label_compatibility",
        "annotator_a_pico_preservation",
        "annotator_b_pico_preservation",
        "final_pico_preservation",
        "annotator_a_modality_strength",
        "annotator_b_modality_strength",
        "final_modality_strength",
        "disagreement_fields",
    ]
    pub_public = []
    for row in pub_rows:
        public_row = {field: row.get(field, "") for field in pub_fields}
        public_row["question_sha256"] = text_hash(row.get("question", ""))
        public_row["converted_claim_sha256"] = text_hash(
            row.get("converted_claim", "")
        )
        pub_public.append(public_row)
    write_csv(
        output / "audits/pubmedqa_mapping_audit_anonymized.csv",
        pub_public,
        pub_fields,
    )

    entity_source = require(
        organized
        / "data/processed/stage9_eswa_major_revision/human_audits/"
        "entity_linking_audit120_adjudication.csv"
    )
    entity_rows = read_csv(entity_source)
    entity_fields = [
        "audit_order",
        "id",
        "dataset",
        "claim_sha256",
        "gold_label",
        "predicted_link_count",
        "predicted_links_json",
        "annotator_a_gold_biomedical_mentions_json",
        "annotator_b_gold_biomedical_mentions_json",
        "final_gold_biomedical_mentions_json",
        "annotator_a_gold_concept_links_json",
        "annotator_b_gold_concept_links_json",
        "final_gold_concept_links_json",
        "annotator_a_incorrect_predicted_links_json",
        "annotator_b_incorrect_predicted_links_json",
        "final_incorrect_predicted_links_json",
        "annotator_a_abbreviation_ambiguity",
        "annotator_b_abbreviation_ambiguity",
        "final_abbreviation_ambiguity",
        "annotator_a_overall_linking_judgment",
        "annotator_b_overall_linking_judgment",
        "final_overall_linking_judgment",
        "disagreement_fields",
    ]
    entity_public = []
    for row in entity_rows:
        public_row = {field: row.get(field, "") for field in entity_fields}
        public_row["claim_sha256"] = text_hash(row.get("claim", ""))
        entity_public.append(public_row)
    write_csv(
        output / "audits/entity_linking_audit_anonymized.csv",
        entity_public,
        entity_fields,
    )

    guideline_source = require(
        organized
        / "data/processed/stage9_eswa_major_revision/human_audits/"
        "stage9_dual_annotation_guideline.md"
    )
    copy_text_normalized(
        guideline_source, output / "audits/human_audit_guideline.md"
    )

    audit_files = sorted(
        path
        for path in (output / "audits").glob("*")
        if path.name != "audit_manifest.json"
    )
    overlap_disagreements = sum(
        any(
            row.get(f"annotator_a_{field}") != row.get(f"annotator_b_{field}")
            for field in ("path_relevance", "evidence_role", "error_type")
        )
        for row in overlap_rows
    )
    manifest = {
        "path_modeling_rows": len(clean_rows),
        "path_modeling_expected_rows": 474,
        "path_overlap_rows": len(overlap_rows),
        "path_overlap_expected_rows": 200,
        "path_overlap_disagreement_rows": overlap_disagreements,
        "artifact_rows": len(artifacts),
        "artifact_expected_rows": 26,
        "pubmedqa_audit_rows": len(pub_rows),
        "pubmedqa_audit_expected_rows": 60,
        "pubmedqa_disagreement_rows": sum(
            bool(str(row.get("disagreement_fields") or "").strip())
            for row in pub_rows
        ),
        "entity_linking_audit_rows": len(entity_rows),
        "entity_linking_audit_expected_rows": 120,
        "entity_linking_disagreement_rows": sum(
            bool(str(row.get("disagreement_fields") or "").strip())
            for row in entity_rows
        ),
        "identity_fields_removed": True,
        "free_text_notes_removed": True,
        "source_context_removed": True,
        "full_upstream_claim_and_question_text_removed": True,
        "short_annotated_biomedical_mention_spans_retained": True,
        "generated_by": "scripts/export_reproducibility_artifacts.py",
        "analyzed_by": "scripts/analyze_stage9_human_audits.py",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in audit_files
        ],
    }
    write_json(output / "audits/audit_manifest.json", manifest)
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in audit_files
    }


def export_alternative_samples(organized: Path, output: Path) -> dict[str, dict]:
    source_dir = (
        organized
        / "data/processed/stage14_alternative_balanced_samples"
    )
    (output / "alternative_samples").mkdir(parents=True, exist_ok=True)
    for name in (
        "alternative_sample_membership.csv",
        "alternative_sample_metrics.csv",
        "alternative_sample_summary.csv",
        "alternative_sample_manifest.json",
    ):
        copy_text_normalized(
            require(source_dir / name), output / "alternative_samples" / name
        )

    features = read_csv(require(source_dir / "full_pool_structural_features.csv"))
    public_features = [
        {
            "record_id": row["id"],
            "dataset": row["dataset"],
            "gold_label": row["gold_label"],
            "normalized_claim_sha256": sha256_bytes(
                row["normalized_claim"].encode("utf-8")
            ),
            "normalized_source_sha256": row["normalized_source_sha256"],
            "entity_linked": row["entity_linked"],
            "has_local_path": row["has_local_path"],
            "has_direct_edge": row["has_direct_edge"],
            "has_predicate_aligned_direct": row["has_predicate_aligned_direct"],
            "has_qualifier_compatible_direct": row[
                "has_qualifier_compatible_direct"
            ],
        }
        for row in features
    ]
    write_csv(
        output / "alternative_samples/full_pool_structural_features.csv",
        public_features,
    )

    upstream = json.loads(
        require(source_dir / "full_pool_structural_features_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    upstream["source_file"] = "logical:standardized_full_pool_6454"
    upstream["public_export"] = (
        "normalized claim text removed and replaced with SHA-256"
    )
    write_json(
        output / "alternative_samples/full_pool_structural_features_provenance.json",
        upstream,
    )
    files = sorted((output / "alternative_samples").glob("*"))
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    }


def copy_result(
    source: Path, target: Path, logical_source: str, provenance: list[dict]
) -> None:
    require(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    copy_text_normalized(source, target)
    provenance.append(
        {
            "logical_source": logical_source,
            "input_sha256": sha256_file(source),
            "output": target.relative_to(ROOT).as_posix(),
            "rows": max(0, sum(1 for _ in target.open("r", encoding=detect_encoding(target))) - 1)
            if target.suffix == ".csv"
            else None,
            "sha256": sha256_file(target),
        }
    )


def export_results(
    workspace: Path, organized: Path, output: Path
) -> dict[str, dict]:
    provenance: list[dict] = []
    mappings = {
        "fair_input_metrics.csv": organized
        / "data/processed/stage9_eswa_major_revision/fair_input_evaluation/"
        "fair_input_main_results.csv",
        "fair_input_pairwise_bootstrap.csv": organized
        / "data/processed/stage13_eswa_review_revision/"
        "fair_input_all_pairwise_bootstrap.csv",
        "evidence_conversion_funnel.csv": workspace
        / "data/processed/submission_strengthening/"
        "strict_evidence_conversion_funnel.csv",
        "evidence_scorer_cv_summary.csv": organized
        / "data/processed/stage7_hierarchical_scorer/path_scorer/cv_summary.csv",
        "evidence_scorer_fold_metrics.csv": organized
        / "data/processed/stage7_hierarchical_scorer/path_scorer/cv_fold_metrics.csv",
        "risk_routing_metrics.csv": organized
        / "data/processed/stage11_eswa_nested_crossfit/formal600/"
        "formal600_selective_metrics.csv",
        "risk_ranking_metrics.csv": organized
        / "data/processed/stage11_eswa_nested_crossfit/formal600/"
        "formal600_risk_ranking.csv",
        "risk_coverage_curve.csv": organized
        / "data/processed/stage11_eswa_nested_crossfit/formal600/"
        "formal600_risk_coverage_curve.csv",
        "bootstrap_contrasts.csv": organized
        / "data/processed/stage11_eswa_nested_crossfit/formal600/"
        "formal600_group_bootstrap.csv",
        "risk_ablation_summary.csv": organized
        / "data/processed/stage11_eswa_nested_crossfit/formal600/"
        "formal600_risk_ablation_summary.csv",
        "fold_composition.csv": organized
        / "data/processed/stage12_eswa_fold_component_audit/"
        "formal600_fold_composition.csv",
        "largest_component_sensitivity.csv": organized
        / "data/processed/stage12_eswa_fold_component_audit/"
        "largest_component_sensitivity.csv",
        "entity_linking_by_dataset.csv": organized
        / "data/processed/stage11_eswa_nested_crossfit/human_audit_results/"
        "entity_linking_by_dataset.csv",
        "pubmedqa_audit_distribution.csv": organized
        / "data/processed/stage11_eswa_nested_crossfit/human_audit_results/"
        "pubmedqa_audit_final_distribution.csv",
        "reviewer_sensitivity_point_estimates.csv": organized
        / "data/processed/stage13_eswa_review_revision/"
        "reviewer_sensitivity_cost_point_estimates.csv",
        "reviewer_sensitivity_bootstrap.csv": organized
        / "data/processed/stage13_eswa_review_revision/"
        "reviewer_sensitivity_cost_bootstrap.csv",
        "cost_utility_point_estimates.csv": organized
        / "data/processed/stage11_eswa_nested_crossfit/submission_closure/"
        "cost_utility_point_estimates.csv",
        "kg_runtime_summary.csv": organized
        / "data/processed/stage11_eswa_nested_crossfit/submission_closure/"
        "kg_runtime_summary.csv",
    }
    for target_name, source in mappings.items():
        copy_result(
            source,
            output / "results" / target_name,
            f"logical:{source.name}",
            provenance,
        )
    write_json(
        output / "provenance/result_provenance.json",
        {
            "seed": 20260618,
            "bootstrap_iterations": 5000,
            "release_date": "2026-07-24",
            "reference_python": "3.10.11",
            "dependency_lock_sha256": sha256_file(ROOT / "requirements-lock.txt"),
            "configuration_sha256": sha256_file(
                ROOT / "config/experiment_config.json"
            ),
            "export_code_sha256": sha256_file(
                ROOT / "scripts/export_reproducibility_artifacts.py"
            ),
            "execution_status": "passed",
            "absolute_source_paths_removed": True,
            "outputs": provenance,
        },
    )
    feature_source = require(
        organized
        / "data/processed/stage11_eswa_nested_crossfit/formal600/"
        "evidence_scorer_feature_provenance.csv"
    )
    write_json(
        output / "provenance/feature_provenance.json",
        {
            "scope": "nested Evidence Scorer feature assignments",
            "logical_source": "logical:evidence_scorer_feature_provenance.csv",
            "input_sha256": sha256_file(feature_source),
            "released_assignment_file": (
                "artifacts/data_splits/formal600_inner_feature_assignments.csv"
            ),
            "released_assignment_sha256": sha256_file(
                output
                / "data_splits/formal600_inner_feature_assignments.csv"
            ),
            "path_component_map": "artifacts/data_splits/path_component_map.csv",
            "scorer_exclusion_manifest": (
                "artifacts/data_splits/scorer_exclusion_manifest.json"
            ),
            "seed": 20260618,
            "generated_by": "scripts/export_reproducibility_artifacts.py",
            "execution_status": "passed",
        },
    )
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted((output / "results").glob("*"))
    }


def export_api_manifest(output: Path) -> dict[str, dict]:
    config = json.loads((ROOT / "config/experiment_config.json").read_text())
    request_rows = read_csv(output / "api_manifest/request_hashes.csv")
    failed_requests = sum(row["request_status"] != "success" for row in request_rows)
    retry_count = sum(int(row.get("retry_count") or 0) for row in request_rows)
    manifest = {
        **config["llm"],
        "expected_request_records": 6000,
        "actual_request_records": len(request_rows),
        "failed_request_records": failed_requests,
        "recorded_retry_count": retry_count,
        "request_hash_rows": len(request_rows),
        "recorded_prompt_hash_rows": sum(
            row["prompt_hash_status"] == "recorded" for row in request_rows
        ),
        "legacy_prompt_hash_rows": sum(
            row["prompt_hash_status"] == "not_recorded_for_legacy_run"
            for row in request_rows
        ),
        "recorded_request_payload_hash_rows": sum(
            row["request_payload_hash_status"] == "recorded"
            for row in request_rows
        ),
        "request_payload_hash_limitation": (
            "Historical payload hashes were not persisted; the released "
            "prompt hashes and parsed-record hashes are reported without "
            "back-filling unverifiable payload hashes."
        ),
        "raw_response_text_included": False,
        "limitations": [
            "Some legacy baseline runs predated per-request prompt hashing.",
            "Hosted model weights may change without notice.",
            "Parsed frozen predictions are the authoritative released outputs.",
        ],
    }
    write_json(output / "api_manifest/frozen_request_manifest.json", manifest)
    parsed_files = [
        output / "predictions/formal600_predictions.jsonl",
        output / "predictions/pubmedqa_claim300_predictions.jsonl",
    ]
    write_json(
        output / "api_manifest/response_file_hashes.json",
        {
            "raw_response_files_released": False,
            "raw_response_file_hashes_available": False,
            "reason": (
                "Raw hosted-model response archives are intentionally withheld "
                "and historical archive hashes were not consistently persisted."
            ),
            "released_parsed_prediction_files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "rows": sum(
                        1
                        for line in path.open("r", encoding="utf-8")
                        if line.strip()
                    ),
                    "sha256": sha256_file(path),
                    "parser": "scripts/export_reproducibility_artifacts.py",
                }
                for path in parsed_files
            ],
        },
    )
    prompt_path = output / "prompts/prompt_templates.md"
    write_json(
        output / "prompts/prompt_template_manifest.json",
        {
            "path": prompt_path.relative_to(ROOT).as_posix(),
            "bytes": prompt_path.stat().st_size,
            "sha256": sha256_file(prompt_path),
            "templates": [
                "evidence_conditioned_classifier",
                "pubmedqa_question_to_claim_conversion",
            ],
        },
    )
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted((output / "api_manifest").glob("*"))
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a small, anonymized reproducibility view from the private workspace."
    )
    parser.add_argument(
        "--workspace-root",
        required=True,
        help="Root containing data/ and organized/medical_kg_evidence_guardrails/.",
    )
    args = parser.parse_args()
    workspace = Path(args.workspace_root).resolve()
    organized = workspace / "organized/medical_kg_evidence_guardrails"
    output = ROOT / "artifacts"
    if not organized.exists():
        raise FileNotFoundError(organized)

    sections = {
        "splits": export_splits(organized, output),
        "predictions": export_predictions(workspace, organized, output),
        "audits": export_audits(workspace, organized, output),
        "alternative_samples": export_alternative_samples(organized, output),
        "results": export_results(workspace, organized, output),
    }
    sections["api_manifest"] = export_api_manifest(output)

    report = {
        "status": "passed",
        "workspace_paths_persisted": False,
        "sections": sections,
    }
    write_json(output / "provenance/local_export_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
