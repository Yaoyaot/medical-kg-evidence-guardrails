from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

from evaluate_eswa_major_revision import (
    ABLATIONS,
    BUDGETS,
    LABELS,
    PRIMARY_MODEL,
    SEED,
    base_metrics,
    bootstrap_primary,
    fit_and_score,
    fit_final_bundle,
    path_quality_features,
    prevalence_adjusted,
    read_csv,
    read_jsonl,
    risk_ranking_metrics,
    select_fold_budget,
    selective_metrics,
    write_csv,
    write_jsonl,
)
from evaluate_matched_budget_guardrails import rule_risk
from train_hierarchical_evidence_scorer import (
    fit_final as fit_path_scorer,
    prepare_rows as prepare_annotation_rows,
    read_csv as read_annotation_csv,
)
from hierarchical_evidence_features import build_adverse_event_pairs


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_annotations_to_components(
    annotations: list[dict],
    manifest_rows: list[dict],
    formal_sources: dict[str, str],
    threshold: float,
) -> tuple[list[dict], dict[str, set[int]], dict[str, set[str]]]:
    """Conservatively link path annotations to Formal600 components.

    Links are created from an identical originating record ID, claim character
    TF-IDF similarity, or a substantial normalized source-prefix match. An
    annotation linked to more than one component is excluded from every fold
    represented by those components.
    """

    formal = []
    for row in manifest_rows:
        formal.append(
            {
                **row,
                "normalized_claim": normalize(row.get("claim", "")),
                "normalized_source": normalize(formal_sources.get(row["id"], "")),
            }
        )
    formal_by_id = {row["id"]: row for row in formal}
    component_fold = {row["pair_group_id"]: int(row["outer_fold"]) for row in formal}

    annotation_claims = [normalize(row.get("claim", "")) for row in annotations]
    formal_claims = [row["normalized_claim"] for row in formal]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), lowercase=False, norm="l2"
    )
    matrix = vectorizer.fit_transform(formal_claims + annotation_claims)
    similarities = (matrix[len(formal_claims) :] @ matrix[: len(formal_claims)].T).tocoo()

    matched_formal_indices: dict[int, set[int]] = defaultdict(set)
    maximum_similarity: dict[int, float] = defaultdict(float)
    for annotation_index, formal_index, score in zip(
        similarities.row, similarities.col, similarities.data
    ):
        maximum_similarity[int(annotation_index)] = max(
            maximum_similarity[int(annotation_index)], float(score)
        )
        if score >= threshold:
            matched_formal_indices[int(annotation_index)].add(int(formal_index))

    mapping_rows: list[dict] = []
    linked_folds: dict[str, set[int]] = {}
    linked_components: dict[str, set[str]] = {}
    for index, annotation in enumerate(annotations):
        reasons: set[str] = set()
        formal_ids: set[str] = set()
        components: set[str] = set()
        claim_id = str(annotation.get("claim_id", ""))
        if claim_id in formal_by_id:
            reasons.add("same_record_id")
            formal_ids.add(claim_id)
            components.add(formal_by_id[claim_id]["pair_group_id"])

        for formal_index in matched_formal_indices.get(index, set()):
            formal_row = formal[formal_index]
            formal_ids.add(formal_row["id"])
            components.add(formal_row["pair_group_id"])
            if annotation_claims[index] == formal_row["normalized_claim"]:
                reasons.add("exact_normalized_claim")
            else:
                reasons.add("near_duplicate_claim")

        source_excerpt = normalize(annotation.get("source_excerpt", ""))
        if len(source_excerpt) >= 100:
            for formal_row in formal:
                source = formal_row["normalized_source"]
                if source and (source.startswith(source_excerpt) or source_excerpt.startswith(source)):
                    reasons.add("normalized_source_prefix")
                    formal_ids.add(formal_row["id"])
                    components.add(formal_row["pair_group_id"])

        folds = {component_fold[component] for component in components}
        annotation_id = annotation["annotation_id"]
        linked_folds[annotation_id] = folds
        linked_components[annotation_id] = components
        mapping_rows.append(
            {
                "annotation_id": annotation_id,
                "claim_id": claim_id,
                "claim_cluster_id": annotation.get("claim_cluster_id", ""),
                "link_reasons": ";".join(sorted(reasons)),
                "linked_formal_ids": ";".join(sorted(formal_ids)),
                "linked_component_ids": ";".join(sorted(components)),
                "linked_outer_folds": ";".join(map(str, sorted(folds))),
                "linked_component_count": len(components),
                "maximum_claim_similarity": maximum_similarity.get(index, 0.0),
                "retained_in_every_scorer_fold": not folds,
            }
        )
    return mapping_rows, linked_folds, linked_components


def annotation_class_counts(rows: list[dict]) -> dict:
    relevance = Counter(row["path_relevance"] for row in rows)
    actionability = Counter(row["actionability"] for row in rows)
    return {
        "irrelevant": relevance.get("IRRELEVANT", 0),
        "partial": relevance.get("PARTIAL", 0),
        "relevant": relevance.get("RELEVANT", 0),
        "non_actionable": actionability.get("NON_ACTIONABLE", 0),
        "actionable": actionability.get("ACTIONABLE", 0),
    }


def scorer_key(excluded_folds: tuple[int, ...]) -> str:
    return "exclude_" + "_".join(map(str, excluded_folds))


def train_scorers(
    annotations: list[dict],
    linked_folds: dict[str, set[int]],
    exclusion_sets: list[tuple[int, ...]],
    model_dir: Path,
) -> tuple[dict[tuple[int, ...], tuple[dict, dict]], list[dict]]:
    bundles: dict[tuple[int, ...], tuple[dict, dict]] = {}
    manifests: list[dict] = []
    model_dir.mkdir(parents=True, exist_ok=True)
    for excluded in exclusion_sets:
        excluded_set = set(excluded)
        train_rows = [
            row
            for row in annotations
            if not (linked_folds[row["annotation_id"]] & excluded_set)
        ]
        excluded_rows = [row for row in annotations if row not in train_rows]
        if any(linked_folds[row["annotation_id"]] & excluded_set for row in train_rows):
            raise AssertionError(f"Path-annotation leakage in scorer {excluded}")
        counts = annotation_class_counts(train_rows)
        if min(counts["irrelevant"], counts["partial"], counts["relevant"]) == 0:
            raise RuntimeError(f"Missing relevance class for scorer {excluded}: {counts}")
        if min(counts["non_actionable"], counts["actionable"]) == 0:
            raise RuntimeError(f"Missing actionability class for scorer {excluded}: {counts}")

        relevance = fit_path_scorer(
            train_rows, "path_relevance", "logistic_regression", SEED
        )
        actionable = fit_path_scorer(
            train_rows, "actionability", "logistic_regression", SEED
        )
        key = scorer_key(excluded)
        relevance_path = model_dir / f"{key}_path_relevance.joblib"
        actionable_path = model_dir / f"{key}_actionability.joblib"
        joblib.dump(relevance, relevance_path)
        joblib.dump(actionable, actionable_path)
        bundles[excluded] = (relevance, actionable)
        manifests.append(
            {
                "scorer_key": key,
                "excluded_outer_folds": ";".join(map(str, excluded)),
                "training_annotation_rows": len(train_rows),
                "excluded_annotation_rows": len(excluded_rows),
                "training_claim_clusters": len(
                    {row.get("claim_cluster_id") or row["claim_id"] for row in train_rows}
                ),
                **counts,
                "test_linked_annotations_in_training": 0,
                "relevance_model": relevance_path.name,
                "relevance_model_sha256": sha256(relevance_path),
                "actionability_model": actionable_path.name,
                "actionability_model_sha256": sha256(actionable_path),
            }
        )
    return bundles, manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict nested Evidence-Scorer and claim-risk cross-fitting for Formal600."
    )
    parser.add_argument(
        "--annotations-path",
        default="data/processed/stage6_eswa/annotations_gold/path_annotations_modeling_pool.csv",
    )
    parser.add_argument(
        "--group-manifest",
        default="data/processed/stage9_eswa_major_revision/grouping/formal600_group_manifest.csv",
    )
    parser.add_argument(
        "--formal-subgraphs",
        default="data/processed/stage2_primekg_semantic_clean/variants/primekg_semantic_clean_relation_aware/local_subgraphs.jsonl",
    )
    parser.add_argument(
        "--guardrail-results",
        default="data/processed/stage2_primekg_semantic_clean/stage4_kg_guardrail_formal600/guardrail_results.jsonl",
    )
    parser.add_argument(
        "--strict-evidence",
        default="data/processed/stage2_primekg_semantic_clean/strict_verifier/strict_kg_evidence.jsonl",
    )
    parser.add_argument(
        "--primekg-graph-dir",
        default="data/processed/stage2_primekg_semantic_clean/primekg_graph",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/stage11_eswa_nested_crossfit/formal600",
    )
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_csv(Path(args.group_manifest))
    manifest = {row["id"]: row for row in manifest_rows}
    guardrail = read_jsonl(Path(args.guardrail_results))
    evidence = {row["id"]: row for row in read_jsonl(Path(args.strict_evidence))}
    subgraphs = {row["id"]: row for row in read_jsonl(Path(args.formal_subgraphs))}
    if len(manifest) != 600 or len(guardrail) != 600:
        raise ValueError("Expected 600 Formal600 manifest and guardrail rows")
    if set(manifest) != {row["id"] for row in guardrail}:
        raise ValueError("Formal600 manifest and guardrail IDs differ")
    if not set(manifest) <= set(evidence) or not set(manifest) <= set(subgraphs):
        raise ValueError("Strict evidence or source subgraphs are incomplete")

    raw_annotations = read_annotation_csv(Path(args.annotations_path))
    annotations, remapped = prepare_annotation_rows(
        raw_annotations, build_adverse_event_pairs(Path(args.primekg_graph_dir))
    )
    if len(annotations) != 474:
        raise ValueError(f"Expected 474 clean annotations, found {len(annotations)}")
    mapping_rows, linked_folds, linked_components = link_annotations_to_components(
        annotations,
        manifest_rows,
        {identifier: subgraphs[identifier].get("source", "") for identifier in manifest},
        args.near_duplicate_threshold,
    )
    write_csv(output / "path_annotation_component_map.csv", mapping_rows)

    exclusion_sets = [(fold,) for fold in range(5)] + [
        (left, right) for left in range(5) for right in range(left + 1, 5)
    ]
    bundles, scorer_manifests = train_scorers(
        annotations,
        linked_folds,
        exclusion_sets,
        output / "nested_path_scorer_models",
    )
    write_csv(output / "path_scorer_nested_manifest.csv", scorer_manifests)

    base_rows: dict[str, dict] = {}
    for row in guardrail:
        group = manifest[row["id"]]
        base_rows[row["id"]] = {
            **row,
            "pair_group_id": group["pair_group_id"],
            "outer_fold": int(group["outer_fold"]),
            "risk_confidence": 1.0 - float(row.get("text_confidence", 0.0)),
            "risk_rule": rule_risk(row.get("guardrail_status", "")),
            "risk_oracle": float(
                row.get("candidate_label") == "SUPPORT"
                and row.get("gold_label") != "SUPPORT"
            ),
        }

    feature_cache: dict[tuple[tuple[int, ...], str], dict] = {}

    def enriched(identifier: str, excluded: tuple[int, ...]) -> dict:
        excluded = tuple(sorted(excluded))
        cache_key = (excluded, identifier)
        if cache_key not in feature_cache:
            relevance, actionable = bundles[excluded]
            base = base_rows[identifier]
            feature_cache[cache_key] = {
                **base,
                **path_quality_features(
                    evidence[identifier], base, relevance, actionable
                ),
                "evidence_scorer_excluded_folds": list(excluded),
            }
        return dict(feature_cache[cache_key])

    final_by_id: dict[str, dict] = {}
    model_manifests: list[dict] = []
    inner_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    provenance_rows: list[dict] = []
    for outer_fold in range(5):
        test_ids = [
            identifier
            for identifier, row in base_rows.items()
            if row["outer_fold"] == outer_fold
        ]
        test_all = [enriched(identifier, (outer_fold,)) for identifier in test_ids]
        for row in test_all:
            final_by_id[row["id"]] = row
            provenance_rows.append(
                {
                    "outer_fold": outer_fold,
                    "record_id": row["id"],
                    "record_fold": outer_fold,
                    "feature_role": "outer_test",
                    "scorer_excluded_folds": str(outer_fold),
                    "own_fold_excluded": True,
                    "outer_test_fold_excluded": True,
                }
            )

        train_all: list[dict] = []
        for inner_fold in range(5):
            if inner_fold == outer_fold:
                continue
            excluded = tuple(sorted((outer_fold, inner_fold)))
            identifiers = [
                identifier
                for identifier, row in base_rows.items()
                if row["outer_fold"] == inner_fold
            ]
            block = [enriched(identifier, excluded) for identifier in identifiers]
            train_all.extend(block)
            for row in block:
                provenance_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "record_id": row["id"],
                        "record_fold": inner_fold,
                        "feature_role": "risk_training_inner_oof",
                        "scorer_excluded_folds": ";".join(map(str, excluded)),
                        "own_fold_excluded": inner_fold in excluded,
                        "outer_test_fold_excluded": outer_fold in excluded,
                    }
                )
        if len(train_all) + len(test_all) != 600:
            raise AssertionError(f"Outer fold {outer_fold} does not partition 600 rows")
        if not all(
            row["outer_fold"] in row["evidence_scorer_excluded_folds"]
            and outer_fold in row["evidence_scorer_excluded_folds"]
            for row in train_all
        ):
            raise AssertionError(f"Inner OOF Evidence Scorer leakage in outer fold {outer_fold}")
        if not all(
            outer_fold in row["evidence_scorer_excluded_folds"] for row in test_all
        ):
            raise AssertionError(f"Outer-test Evidence Scorer leakage in fold {outer_fold}")

        train = [row for row in train_all if row["candidate_label"] == "SUPPORT"]
        test = [row for row in test_all if row["candidate_label"] == "SUPPORT"]
        for model_name in ABLATIONS:
            scores, model_manifest, diagnostics = fit_and_score(
                train, test, model_name
            )
            for row, score in zip(test, scores):
                final_by_id[row["id"]][f"risk_{model_name}"] = float(score)
            model_manifests.append(
                {
                    "outer_fold": outer_fold,
                    "model": model_name,
                    "training_support_predictions": len(train),
                    "test_support_predictions": len(test),
                    "evidence_scorer_outer_test_exclusion": str(outer_fold),
                    "risk_training_scorer_design": "inner_oof_excludes_outer_and_record_fold",
                    **{
                        key: value
                        for key, value in model_manifest.items()
                        if key != "coefficients"
                    },
                }
            )
            inner_rows.extend(
                {"outer_fold": outer_fold, **item} for item in diagnostics
            )
            coefficient_rows.extend(
                {
                    "outer_fold": outer_fold,
                    "model": model_name,
                    **item,
                }
                for item in model_manifest.get("coefficients", [])
            )

    if len(final_by_id) != 600:
        raise AssertionError(f"Expected 600 final OOF rows, found {len(final_by_id)}")
    rows = [final_by_id[row["id"]] for row in guardrail]
    for row in rows:
        if row["candidate_label"] != "SUPPORT":
            for model_name in ABLATIONS:
                row[f"risk_{model_name}"] = 0.0

    if not all(row["own_fold_excluded"] and row["outer_test_fold_excluded"] for row in provenance_rows):
        raise AssertionError("Feature provenance contains a non-excluded component fold")
    write_csv(output / "evidence_scorer_feature_provenance.csv", provenance_rows)

    risk_methods = {
        "self_reported_confidence": "risk_confidence",
        "rule_matched_budget": "risk_rule",
        **{model_name: f"risk_{model_name}" for model_name in ABLATIONS},
        "oracle": "risk_oracle",
    }
    ranking = [
        risk_ranking_metrics(rows, field, method)
        for method, field in risk_methods.items()
    ]
    selective: list[dict] = []
    risk_curve: list[dict] = []
    for method, field in risk_methods.items():
        for budget in BUDGETS:
            reviewed = select_fold_budget(rows, field, budget)
            selective.append(
                {"method": method, "budget": budget, **selective_metrics(rows, reviewed)}
            )
        for percent in range(21):
            budget = percent / 100
            risk_curve.append(
                {
                    "method": method,
                    "budget": budget,
                    **selective_metrics(
                        rows, select_fold_budget(rows, field, budget)
                    ),
                }
            )
    for budget in BUDGETS:
        simulations = []
        for simulation in range(1000):
            for row in rows:
                digest = hashlib.sha256(
                    f"{SEED}:{simulation}:{row['id']}".encode("utf-8")
                ).hexdigest()
                row["risk_random"] = int(digest[:16], 16) / 16**16
            simulations.append(
                selective_metrics(
                    rows, select_fold_budget(rows, "risk_random", budget)
                )
            )
        selective.append(
            {
                "method": "random_review",
                "budget": budget,
                **{
                    key: float(np.mean([item[key] for item in simulations]))
                    for key in simulations[0]
                },
            }
        )

    aurc_rows = []
    for method in risk_methods:
        points = sorted(
            [row for row in risk_curve if row["method"] == method],
            key=lambda row: row["coverage"],
        )
        aurc_rows.append(
            {
                "method": method,
                "coverage_min": points[0]["coverage"],
                "coverage_max": points[-1]["coverage"],
                "partial_aurc_coverage_0_805_to_1_000": float(
                    np.trapezoid(
                        [row["selective_risk"] for row in points],
                        [row["coverage"] for row in points],
                    )
                ),
            }
        )

    base = base_metrics(rows)
    matrix = confusion_matrix(
        [row["gold_label"] for row in rows],
        [row["candidate_label"] for row in rows],
        labels=list(LABELS),
    )
    confusion_rows = [
        {"gold_label": gold, "predicted_label": pred, "count": int(matrix[i, j])}
        for i, gold in enumerate(LABELS)
        for j, pred in enumerate(LABELS)
    ]
    bootstrap = bootstrap_primary(rows, args.bootstrap_iterations)
    final_manifest = fit_final_bundle(rows, output)

    deployment_dir = output / "deployment_path_scorer"
    deployment_dir.mkdir(exist_ok=True)
    deployment_relevance = fit_path_scorer(
        annotations, "path_relevance", "logistic_regression", SEED
    )
    deployment_actionable = fit_path_scorer(
        annotations, "actionability", "logistic_regression", SEED
    )
    joblib.dump(
        deployment_relevance,
        deployment_dir / "path_relevance_logistic_regression.joblib",
    )
    joblib.dump(
        deployment_actionable,
        deployment_dir / "actionability_logistic_regression.joblib",
    )

    write_jsonl(output / "formal600_crossfit_risk_scores.jsonl", rows)
    write_csv(output / "formal600_base_metrics.csv", [{"method": "provided_text_verifier", **base}])
    write_csv(output / "formal600_confusion_matrix.csv", confusion_rows)
    write_csv(output / "formal600_risk_ranking.csv", ranking)
    write_csv(output / "formal600_selective_metrics.csv", selective)
    write_csv(output / "formal600_risk_coverage_curve.csv", risk_curve)
    write_csv(output / "formal600_aurc.csv", aurc_rows)
    write_csv(output / "formal600_group_bootstrap.csv", bootstrap)
    write_csv(output / "formal600_prevalence_adjusted_precision.csv", prevalence_adjusted(rows))
    write_csv(output / "formal600_outer_model_manifest.csv", model_manifests)
    write_csv(output / "formal600_inner_cv.csv", inner_rows)
    write_csv(output / "formal600_outer_coefficients.csv", coefficient_rows)

    primary_ranking = next(row for row in ranking if row["method"] == PRIMARY_MODEL)
    confidence_ranking = next(
        row for row in ranking if row["method"] == "self_reported_confidence"
    )
    primary_5 = next(
        row
        for row in selective
        if row["method"] == PRIMARY_MODEL and row["budget"] == 0.05
    )
    confidence_5 = next(
        row
        for row in selective
        if row["method"] == "self_reported_confidence" and row["budget"] == 0.05
    )
    recall_ci = next(
        row for row in bootstrap if row["metric"] == "error_detection_recall"
    )
    linked_rows = [row for row in mapping_rows if row["linked_outer_folds"]]
    stats = {
        "evaluation_design": "strict nested Evidence-Scorer and claim-risk component cross-fitting",
        "rows": len(rows),
        "groups": len({row["pair_group_id"] for row in rows}),
        "folds": 5,
        "path_annotations": len(annotations),
        "path_annotation_claim_clusters": len(
            {row.get("claim_cluster_id") or row["claim_id"] for row in annotations}
        ),
        "path_annotations_linked_to_formal_components": len(linked_rows),
        "formal_components_linked_to_path_annotations": len(
            {
                component
                for components in linked_components.values()
                for component in components
            }
        ),
        "outer_path_scorer_training_rows": {
            str(fold): next(
                row["training_annotation_rows"]
                for row in scorer_manifests
                if row["scorer_key"] == scorer_key((fold,))
            )
            for fold in range(5)
        },
        "inner_feature_rows": sum(
            row["feature_role"] == "risk_training_inner_oof"
            for row in provenance_rows
        ),
        "outer_test_feature_rows": sum(
            row["feature_role"] == "outer_test" for row in provenance_rows
        ),
        "all_feature_rows_exclude_own_component_fold": True,
        "all_risk_training_features_exclude_outer_test_fold": True,
        "semantic_relation_remapped": remapped,
        "base": base,
        "primary_model": PRIMARY_MODEL,
        "primary_ranking": primary_ranking,
        "confidence_ranking": confidence_ranking,
        "primary_5pct": primary_5,
        "confidence_5pct": confidence_5,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_seed": SEED,
        "final_frozen_model": final_manifest,
        "result_gate_rule": (
            "At the fold-allocated nominal 5% review budget, learned routing must "
            "have higher observed false-SUPPORT detection recall than self-reported-"
            "confidence routing and the lower bound of the 5000-replicate claim/source-"
            "component bootstrap interval for learned-minus-confidence recall must exceed zero."
        ),
        "result_gate_timing": "primary analysis decision rule; not registered as a prospective protocol",
        "result_gate_passed": (
            primary_5["error_detection_recall"]
            > confidence_5["error_detection_recall"]
            and recall_ci["ci_low"] > 0
        ),
        "llm_api_calls": False,
    }
    (output / "strict_nested_crossfit_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
