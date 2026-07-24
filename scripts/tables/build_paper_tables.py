from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABELS = ("SUPPORT", "REFUTE", "UNCERTAIN")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def macro_f1(rows: list[dict], prediction: str = "pred_label") -> float:
    values = []
    for label in LABELS:
        tp = sum(
            row["gold_label"] == label and row[prediction] == label for row in rows
        )
        fp = sum(
            row["gold_label"] != label and row[prediction] == label for row in rows
        )
        fn = sum(
            row["gold_label"] == label and row[prediction] != label for row in rows
        )
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return sum(values) / len(values)


def classification_metrics(rows: list[dict]) -> dict:
    support = [row for row in rows if row["pred_label"] == "SUPPORT"]
    true_support = sum(row["gold_label"] == "SUPPORT" for row in support)
    false_support = len(support) - true_support
    gold_support = sum(row["gold_label"] == "SUPPORT" for row in rows)
    return {
        "count": len(rows),
        "accuracy": sum(
            row["gold_label"] == row["pred_label"] for row in rows
        )
        / len(rows),
        "macro_f1": macro_f1(rows),
        "support_predictions": len(support),
        "support_precision": true_support / len(support)
        if support
        else "not estimable",
        "support_recall": true_support / gold_support if gold_support else 0.0,
        "false_support_count": false_support,
        "false_support_rate": false_support / len(support)
        if support
        else "not estimable",
    }


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if len(a) != len(b) or not a:
        raise ValueError("Kappa vectors must have equal nonzero length")
    categories = sorted(set(a) | set(b))
    observed = sum(left == right for left, right in zip(a, b)) / len(a)
    left_counts = Counter(a)
    right_counts = Counter(b)
    expected = sum(
        left_counts[value] * right_counts[value] for value in categories
    ) / len(a) ** 2
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def table1() -> list[dict]:
    resources = json.loads(
        (ROOT / "config/kg_resources.json").read_text(encoding="utf-8")
    )
    return [
        {
            "resource_or_build_stage": "Hetionet terminology-expanded graph",
            "version_or_retrieval_date": "v1.0",
            "nodes": 45102,
            "edges_or_rows": 2188507,
            "aliases": 135306,
            "role": "Base biomedical graph",
        },
        {
            "resource_or_build_stage": "PrimeKG raw file",
            "version_or_retrieval_date": "Dataverse file 6180620; 2026-05-30",
            "nodes": "129375 unique",
            "edges_or_rows": 8100498,
            "aliases": "",
            "role": "Source snapshot",
        },
        {
            "resource_or_build_stage": "PrimeKG filtered graph",
            "version_or_retrieval_date": "Same snapshot",
            "nodes": 41263,
            "edges_or_rows": resources["primekg"]["filtered_edges"],
            "aliases": 41263,
            "role": "Relation/type filtering",
        },
        {
            "resource_or_build_stage": "PrimeKG semantically cleaned graph",
            "version_or_retrieval_date": "Same snapshot",
            "nodes": 41263,
            "edges_or_rows": resources["primekg"]["semantically_cleaned_edges"],
            "aliases": 41263,
            "role": "Direction and relation cleaning",
        },
        {
            "resource_or_build_stage": "Mondo",
            "version_or_retrieval_date": "2026-05-05",
            "nodes": "",
            "edges_or_rows": "",
            "aliases": "",
            "role": "Disease terminology",
        },
        {
            "resource_or_build_stage": "HGNC",
            "version_or_retrieval_date": "Retrieved 2026-05-30",
            "nodes": "",
            "edges_or_rows": "",
            "aliases": "",
            "role": "Gene terminology",
        },
        {
            "resource_or_build_stage": "UMLS",
            "version_or_retrieval_date": "2026AA",
            "nodes": "",
            "edges_or_rows": "",
            "aliases": "",
            "role": "Licensed normalization only",
        },
        {
            "resource_or_build_stage": "Final fused and cleaned graph",
            "version_or_retrieval_date": "Frozen deterministic build",
            "nodes": resources["merged_clean_graph"]["nodes"],
            "edges_or_rows": resources["merged_clean_graph"]["edges"],
            "aliases": resources["merged_clean_graph"]["aliases"],
            "role": "Experimental KG",
        },
    ]


def table2() -> list[dict]:
    return [
        {
            "evidence_state": "No KG grounding",
            "operational_condition": "No reliable concept link",
            "permitted_use": "Absence/context signal only",
        },
        {
            "evidence_state": "Single-entity context",
            "operational_condition": "Concepts linked but no claim-spanning path",
            "permitted_use": "Context only",
        },
        {
            "evidence_state": "Two-hop context",
            "operational_condition": "Linked endpoints connected through an intermediate",
            "permitted_use": "Context/risk signal only",
        },
        {
            "evidence_state": "Direct relation unresolved",
            "operational_condition": "Direct edge; claim predicate unresolved",
            "permitted_use": "Context/risk signal only",
        },
        {
            "evidence_state": "Direct predicate/direction mismatch",
            "operational_condition": "Direct edge conflicts with predicate or order",
            "permitted_use": "Conflict/risk signal",
        },
        {
            "evidence_state": "Direct match; qualifier incomplete",
            "operational_condition": "Direct semantic match; critical qualifier unrepresented",
            "permitted_use": "Risk/context signal only",
        },
        {
            "evidence_state": "Direct match; qualifier compatible",
            "operational_condition": "Endpoints, predicate, direction, and qualifier gate pass",
            "permitted_use": "Strict structured candidate",
        },
    ]


def table3() -> list[dict]:
    return [
        {
            "set": "Formal600",
            "size": 600,
            "labels": "200/200/200",
            "text_supplied_to_primary_verifier": (
                "Source text paired by HealthVer, MedAESQA, or SciFact"
            ),
            "role": "Grouped internal evaluation",
        },
        {
            "set": "Path annotations",
            "size": "474 clean paths",
            "labels": "Relevance and actionability",
            "text_supplied_to_primary_verifier": (
                "Claim, source excerpt, linked path"
            ),
            "role": "Nested Evidence Scorer development",
        },
        {
            "set": "PubMedQA-Claim-300",
            "size": 300,
            "labels": "100/100/100",
            "text_supplied_to_primary_verifier": (
                "PubMedQA abstract context; no long answer"
            ),
            "role": "QA-derived label-noise stress test",
        },
    ]


def table4() -> list[dict]:
    display = {
        "provided_text": "Provided text",
        "provided_text_plus_extra_bm25": "Provided text + extra BM25",
        "provided_text_plus_kg_paths": "Provided text + KG paths",
        "provided_text_plus_kg_plus_extra_bm25": (
            "Provided text + KG + BM25"
        ),
    }
    output = []
    for row in read_csv(ROOT / "artifacts/results/fair_input_metrics.csv"):
        if row["method"] not in display:
            continue
        output.append(
            {
                "domain": (
                    "Formal600"
                    if row["domain"] == "formal600"
                    else "PubMedQA stress test"
                ),
                "input_condition": display[row["method"]],
                "accuracy": f"{float(row['accuracy']):.4f}",
                "macro_f1": f"{float(row['macro_f1']):.4f}",
                "support_n_false_n": (
                    f"{int(row['support_predictions'])} / "
                    f"{int(row['false_support_count'])}"
                ),
                "support_precision_false_support_rate": (
                    f"{float(row['support_precision']):.4f} / "
                    f"{float(row['false_support_rate']):.4f}"
                ),
                "support_recall": f"{float(row['support_recall']):.4f}",
            }
        )
    return output


def table5() -> list[dict]:
    display = {
        "confidence_only_features": "Confidence only",
        "dataset_source_only": "Dataset source only",
        "kg_evidence_state_only": "KG evidence state only",
        "semantic_rules_only": "Qualifier, endpoint, and predicate rules",
        "path_statistics_only": "Path statistics only",
        "evidence_scorer_only": "Evidence Scorer only",
        "confidence_plus_kg_rules": "Confidence + KG rules",
        "confidence_plus_evidence_scorer": "Confidence + Evidence Scorer",
        "full_without_dataset_source": "Full without dataset source",
        "full_with_dataset_source": "Full with dataset source",
    }
    output = []
    for row in read_csv(ROOT / "artifacts/results/risk_ablation_summary.csv"):
        output.append(
            {
                "feature_set": display[row["method"]],
                "auroc_95ci": (
                    f"{float(row['auroc']):.4f} "
                    f"({float(row['auroc_ci_low']):.4f}–"
                    f"{float(row['auroc_ci_high']):.4f})"
                ),
                "ap_95ci": (
                    f"{float(row['average_precision']):.4f} "
                    f"({float(row['average_precision_ci_low']):.4f}–"
                    f"{float(row['average_precision_ci_high']):.4f})"
                ),
                "five_pct_detection_recall": (
                    f"{float(row['error_detection_recall']):.4f}"
                ),
                "five_pct_selective_risk": (
                    f"{float(row['selective_risk']):.4f}"
                ),
            }
        )
    return output


def table6() -> list[dict]:
    display = {
        "self_reported_confidence": "Self-reported confidence",
        "rule_matched_budget": "Rule-based KG state",
        "full_without_dataset_source": "Full learned model, no dataset source",
        "oracle": "Oracle error ranking",
    }
    output = []
    for row in read_csv(ROOT / "artifacts/results/risk_routing_metrics.csv"):
        if float(row["budget"]) != 0.05 or row["method"] not in display:
            continue
        output.append(
            {
                "ranking": display[row["method"]],
                "captured_false_support": int(
                    float(row["detected_false_supports"])
                ),
                "reviewed_true_support": int(
                    float(row["incorrectly_reviewed_correct_supports"])
                ),
                "detection_precision": (
                    f"{float(row['error_detection_precision']):.4f}"
                ),
                "detection_recall": (
                    f"{float(row['error_detection_recall']):.4f}"
                ),
                "coverage": f"{float(row['coverage']):.4f}",
                "selective_risk": f"{float(row['selective_risk']):.4f}",
            }
        )
    return output


def table7() -> list[dict]:
    rows = read_csv(
        ROOT / "artifacts/audits/pubmedqa_mapping_audit_anonymized.csv"
    )
    dimensions = [
        (
            "Claim faithfulness",
            "claim_faithfulness",
            "VALID",
        ),
        ("Atomicity", "atomicity", "ATOMIC"),
        ("Label compatibility", "label_compatibility", "COMPATIBLE"),
        ("PICO preservation", "pico_preservation", "COMPLETE"),
        (
            "Modality/strength preservation",
            "modality_strength",
            "PRESERVED",
        ),
    ]
    output = []
    for label, field, positive in dimensions:
        a = [row[f"annotator_a_{field}"] for row in rows]
        b = [row[f"annotator_b_{field}"] for row in rows]
        final = [row[f"final_{field}"] for row in rows]
        agreement = sum(left == right for left, right in zip(a, b)) / len(rows)
        kappa = cohen_kappa(a, b)
        count = sum(value == positive for value in final)
        output.append(
            {
                "dimension": label,
                "raw_agreement": f"{agreement:.3f}",
                "cohens_kappa": (
                    "Not estimable" if kappa is None else f"{kappa:.3f}"
                ),
                "final_adjudicated_outcome": (
                    f"{count}/{len(rows)} {positive} "
                    f"({count / len(rows):.1%})"
                ),
            }
        )
    return output


def supplementary_table1() -> list[dict]:
    audit = read_csv(
        ROOT / "artifacts/audits/pubmedqa_mapping_audit_anonymized.csv"
    )
    compatible = {
        row["id"]
        for row in audit
        if row["final_label_compatibility"] == "COMPATIBLE"
    }
    if len(compatible) != 44:
        raise AssertionError(f"Expected 44 compatible IDs, found {len(compatible)}")

    prediction_names = {
        "direct_llm": "Direct LLM",
        "provided_text": "Provided-text verifier",
        "kg_only_local_paths_diagnostic": "KG-only local-path prompting",
        "provided_text_plus_kg_plus_extra_bm25": (
            "Text + KG + BM25 joint prompting"
        ),
    }
    methods: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(
        ROOT / "artifacts/predictions/pubmedqa_claim300_predictions.jsonl"
    ):
        if row["record_id"] not in compatible or row["method"] not in prediction_names:
            continue
        methods[prediction_names[row["method"]]].append(
            {
                "id": row["record_id"],
                "gold_label": row["gold_label"],
                "pred_label": row["predicted_label"],
                "reviewed": False,
            }
        )

    risk_rows = read_csv(
        ROOT / "artifacts/predictions/pubmedqa_frozen_risk_scores.csv"
    )
    eligible = [row for row in risk_rows if row["candidate_label"] == "SUPPORT"]
    confidence_top = {
        row["id"]
        for row in sorted(
            eligible,
            key=lambda row: (-float(row["risk_confidence"]), row["id"]),
        )[:15]
    }
    learned_top = {
        row["id"]
        for row in sorted(
            eligible,
            key=lambda row: (
                -float(row["risk_full_without_dataset_source"]),
                row["id"],
            ),
        )[:15]
    }
    for row in risk_rows:
        if row["id"] not in compatible:
            continue
        candidate = row["candidate_label"]
        rules = [
            (
                "Confidence review, frozen full-300 top 15",
                row["id"] in confidence_top,
            ),
            (
                "Rule guardrail, original trigger",
                candidate == "SUPPORT"
                and row["guardrail_status"] == "KG_TWO_HOP_CONTEXT",
            ),
            (
                "Learned review, frozen full-300 top 15",
                row["id"] in learned_top,
            ),
        ]
        for name, reviewed in rules:
            methods[name].append(
                {
                    "id": row["id"],
                    "gold_label": row["gold_label"],
                    "pred_label": (
                        "UNCERTAIN"
                        if reviewed and candidate == "SUPPORT"
                        else candidate
                    ),
                    "reviewed": reviewed,
                }
            )

    order = [
        "Direct LLM",
        "Provided-text verifier",
        "KG-only local-path prompting",
        "Text + KG + BM25 joint prompting",
        "Confidence review, frozen full-300 top 15",
        "Rule guardrail, original trigger",
        "Learned review, frozen full-300 top 15",
    ]
    output = []
    for name in order:
        rows = methods[name]
        if len(rows) != 44 or len({row["id"] for row in rows}) != 44:
            raise AssertionError(f"{name}: expected 44 unique rows")
        metrics = classification_metrics(rows)
        precision = metrics["support_precision"]
        false_rate = metrics["false_support_rate"]
        output.append(
            {
                "method": name,
                "accuracy": f"{metrics['accuracy']:.3f}",
                "macro_f1": f"{metrics['macro_f1']:.3f}",
                "support_n": metrics["support_predictions"],
                "support_precision": (
                    "Not estimable"
                    if isinstance(precision, str)
                    else f"{precision:.3f}"
                ),
                "false_support_n_rate": (
                    "0 / not estimable"
                    if isinstance(false_rate, str)
                    else (
                        f"{metrics['false_support_count']} / "
                        f"{false_rate:.3f}"
                    )
                ),
                "reviewed_in_subset": sum(row["reviewed"] for row in rows),
            }
        )
    return output


def supplementary_table2() -> list[dict]:
    output = []
    for row in read_csv(ROOT / "artifacts/results/fold_composition.csv"):
        output.append(
            {
                "fold": row["fold"],
                "components": row["components"],
                "records": row["records"],
                "gold_s_r_u": (
                    f"{row['support_gold']}/{row['refute_gold']}/"
                    f"{row['uncertain_gold']}"
                ),
                "data_h_m_s": (
                    f"{row['healthver']}/{row['medaesqa']}/{row['scifact']}"
                ),
                "support_predictions": row["support_predictions"],
                "false_support": row["false_support_predictions"],
                "five_pct_review": row["nominal_5pct_review_count"],
            }
        )
    return output


def supplementary_table3() -> list[dict]:
    display_scope = {
        "full600": "Full 600",
        "without_largest_component": "Without largest",
    }
    display_method = {
        "learned": "Learned",
        "self_reported_confidence": "Self-conf.",
    }
    output = []
    for row in read_csv(
        ROOT / "artifacts/results/largest_component_sensitivity.csv"
    ):
        output.append(
            {
                "scope": display_scope[row["scope"]],
                "ranking": display_method[row["method"]],
                "support_predictions": row["support_predictions"],
                "false_support": row["false_support_predictions"],
                "auroc_ap": (
                    f"{float(row['auroc']):.4f} / "
                    f"{float(row['average_precision']):.4f}"
                ),
                "reviews": row["reviewed_count"],
                "captured_recall": (
                    f"{row['captured_false_support']} / "
                    f"{float(row['detection_recall']):.4f}"
                ),
                "selective_risk": f"{float(row['selective_risk']):.4f}",
            }
        )
    return output


def supplementary_tables4_and5() -> tuple[list[dict], list[dict]]:
    metrics = read_csv(
        ROOT / "artifacts/alternative_samples/alternative_sample_metrics.csv"
    )
    formal = next(row for row in metrics if row["sample_id"] == "formal600_id_sorted")
    summary = {
        row["sampling_frame"]: row
        for row in read_csv(
            ROOT
            / "artifacts/alternative_samples/alternative_sample_summary.csv"
        )
    }
    table4_rows = [
        {
            "sampling_frame": "Formal600 ID-sorted",
            "dataset_composition_n": (
                f"H {formal['healthver']}; M {formal['medaesqa']}; "
                f"S {formal['scifact']}; P {formal['pubhealth']}"
            ),
            "components": formal["components"],
            "largest_component_n_pct": (
                f"{formal['largest_component_records']} "
                f"({100 * float(formal['largest_component_share']):.1f}%)"
            ),
        }
    ]
    table5_rows = [
        {
            "sampling_frame": "Formal600 ID-sorted",
            "entity_linked_pct": f"{100 * float(formal['entity_linked_rate']):.1f}",
            "local_path_pct": f"{100 * float(formal['has_local_path_rate']):.1f}",
            "direct_edge_pct": f"{100 * float(formal['has_direct_edge_rate']):.1f}",
            "predicate_aligned_pct": (
                f"{100 * float(formal['has_predicate_aligned_direct_rate']):.2f}"
            ),
            "qualifier_compatible_n": formal[
                "has_qualifier_compatible_direct_count"
            ],
        }
    ]
    labels = {
        "full_6454_pool": "Full pool, 10 samples",
        "matched_source_frame": "Matched-source, 10 samples",
    }
    for key in ("full_6454_pool", "matched_source_frame"):
        row = summary[key]
        composition = (
            row["dataset_composition_median_range"]
            .replace("pubhealth", "P")
            .replace("healthver", "H")
            .replace("medaesqa", "M")
            .replace("scifact", "S")
        )
        table4_rows.append(
            {
                "sampling_frame": labels[key],
                "dataset_composition_n": composition,
                "components": row["components_median_range"],
                "largest_component_n_pct": (
                    f"{row['largest_component_records_median_range']}; "
                    f"{row['largest_component_share_pct_median_range']}"
                ),
            }
        )
        table5_rows.append(
            {
                "sampling_frame": labels[key],
                "entity_linked_pct": row["entity_linked_pct_median_range"],
                "local_path_pct": row["local_path_pct_median_range"],
                "direct_edge_pct": row["direct_edge_pct_median_range"],
                "predicate_aligned_pct": (
                    row["predicate_aligned_pct_median_range"]
                ),
                "qualifier_compatible_n": (
                    row["qualifier_compatible_count_median_range"]
                ),
            }
        )
    return table4_rows, table5_rows


def supplementary_table6() -> list[dict]:
    point = [
        row
        for row in read_csv(
            ROOT / "artifacts/results/reviewer_sensitivity_point_estimates.csv"
        )
        if float(row["budget"]) == 0.05
        and float(row["reviewer_sensitivity"]) == 0.8
        and float(row["review_cost_ratio_lambda"]) == 0.1
    ]
    random = next(
        row
        for row in read_csv(ROOT / "artifacts/results/risk_routing_metrics.csv")
        if row["method"] == "random_review" and float(row["budget"]) == 0.05
    )
    output = [
        {
            "ranking": "Random",
            "reviewed": int(float(random["reviewed_count"])),
            "false_support_captured": f"{float(random['detected_false_supports']):.3f}",
            "correct_support_reviewed": (
                f"{float(random['incorrectly_reviewed_correct_supports']):.3f}"
            ),
            "normalized_net_gain": f"{(0.8 * float(random['detected_false_supports']) - 0.1 * float(random['reviewed_count'])) / 600:.4f}",
        }
    ]
    names = {
        "confidence": "Self-reported confidence",
        "rule": "Rule-based KG state",
        "learned": "Full learned model, no dataset source",
        "oracle": "Oracle error ranking",
    }
    base_routing = {
        row["method"]: row
        for row in read_csv(ROOT / "artifacts/results/risk_routing_metrics.csv")
        if float(row["budget"]) == 0.05
    }
    routing_key = {
        "confidence": "self_reported_confidence",
        "rule": "rule_matched_budget",
        "learned": "full_without_dataset_source",
        "oracle": "oracle",
    }
    for row in point:
        routing = base_routing[routing_key[row["method"]]]
        output.append(
            {
                "ranking": names[row["method"]],
                "reviewed": row["reviewed_count"],
                "false_support_captured": row["captured_false_supports"],
                "correct_support_reviewed": routing[
                    "incorrectly_reviewed_correct_supports"
                ],
                "normalized_net_gain": (
                    f"{float(row['normalized_idealized_net_gain']):.4f}"
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild manuscript Tables 1–7 and Supplementary Tables S1–S6 "
            "from released configuration, audit, prediction, and result files."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reproduced/paper_tables",
    )
    args = parser.parse_args()
    output = ROOT / args.output_dir
    main_tables = [
        table1(),
        table2(),
        table3(),
        table4(),
        table5(),
        table6(),
        table7(),
    ]
    for index, rows in enumerate(main_tables, 1):
        write_csv(output / f"table{index}.csv", rows)
    s4, s5 = supplementary_tables4_and5()
    supplementary = [
        supplementary_table1(),
        supplementary_table2(),
        supplementary_table3(),
        s4,
        s5,
        supplementary_table6(),
    ]
    for index, rows in enumerate(supplementary, 1):
        write_csv(output / f"table_s{index}.csv", rows)
    print(f"Wrote Tables 1–7 and S1–S6 to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
