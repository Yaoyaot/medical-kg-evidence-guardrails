from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual: str | float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"{label}: expected {expected}, observed {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check released artifacts against manuscript-facing invariants."
    )
    parser.add_argument(
        "--tables-dir",
        default="outputs/reproduced/paper_tables",
    )
    parser.add_argument(
        "--figures-dir",
        default="outputs/reproduced/paper_figures",
    )
    parser.add_argument(
        "--skip-generated-files",
        action="store_true",
        help="Check source artifacts without requiring generated tables or figures.",
    )
    args = parser.parse_args()

    risk = read_csv(ROOT / "artifacts/predictions/risk_routing_scores.csv")
    if len(risk) != 600 or len({row["id"] for row in risk}) != 600:
        raise AssertionError("Formal600 risk-score artifact must contain 600 unique IDs")
    funnel = {
        "records": len(risk),
        "entity_linked": sum(int(row["linked_entity_count"]) > 0 for row in risk),
        "any_path": sum(int(row["path_count"]) > 0 for row in risk),
        "direct_edge": sum(int(row["direct_path_count"]) > 0 for row in risk),
        "predicate_aligned": sum(
            int(row["predicate_aligned_path_count"]) > 0 for row in risk
        ),
        "qualifier_compatible": sum(
            int(row["strict_structured_candidate_count"]) > 0 for row in risk
        ),
    }
    expected_funnel = {
        "records": 600,
        "entity_linked": 439,
        "any_path": 167,
        "direct_edge": 78,
        "predicate_aligned": 11,
        "qualifier_compatible": 2,
    }
    if funnel != expected_funnel:
        raise AssertionError(f"Figure 3 funnel mismatch: {funnel}")
    states = Counter(row["evidence_state_nominal"] for row in risk)
    expected_states = Counter(
        {
            "NO_KG_GROUNDING": 161,
            "SINGLE_ENTITY_CONTEXT": 272,
            "TWO_HOP_CONTEXT": 89,
            "DIRECT_RELATION_UNRESOLVED": 43,
            "DIRECT_PREDICATE_OR_DIRECTION_MISMATCH": 24,
            "DIRECT_MATCH_QUALIFIER_INCOMPLETE": 9,
            "DIRECT_MATCH_QUALIFIER_COMPATIBLE": 2,
        }
    )
    if states != expected_states:
        raise AssertionError(f"Terminal-state mismatch: {states}")

    external = read_csv(
        ROOT / "artifacts/predictions/pubmedqa_frozen_risk_scores.csv"
    )
    if len(external) != 300 or len({row["id"] for row in external}) != 300:
        raise AssertionError("External risk artifact must contain 300 unique IDs")
    forbidden = {"claim", "source", "question", "reasoning", "long_answer"}
    if forbidden.intersection(external[0]):
        raise AssertionError("External risk export contains a forbidden text field")
    if any(
        row["external_gold_used_for_risk_scoring"].strip().lower() != "false"
        for row in external
    ):
        raise AssertionError("External gold-use guard is not false for every row")

    ranking = {
        row["method"]: row
        for row in read_csv(
            ROOT / "artifacts/results/pubmedqa_frozen_risk_ranking.csv"
        )
    }
    assert_close(
        ranking["full_without_dataset_source"]["auroc"],
        0.5118256710071751,
        "Figure 6 learned external AUROC",
    )
    assert_close(
        ranking["full_without_dataset_source"]["average_precision"],
        0.4839238198922775,
        "Figure 6 learned external AP",
    )
    assert_close(
        ranking["self_reported_confidence"]["auroc"],
        0.6101514748870581,
        "Figure 6 confidence external AUROC",
    )
    assert_close(
        ranking["self_reported_confidence"]["average_precision"],
        0.4920841890423587,
        "Figure 6 confidence external AP",
    )

    fair = {
        (row["domain"], row["method"]): row
        for row in read_csv(ROOT / "artifacts/results/fair_input_metrics.csv")
    }
    assert_close(
        fair[("formal600", "provided_text")]["macro_f1"],
        0.5857622328210564,
        "Table 4 Formal600 provided-text Macro-F1",
    )
    assert_close(
        fair[("formal600", "provided_text_plus_kg_paths")]["macro_f1"],
        0.6140038362687483,
        "Table 4 Formal600 text+KG Macro-F1",
    )

    routing = {
        (row["method"], float(row["budget"])): row
        for row in read_csv(ROOT / "artifacts/results/risk_routing_metrics.csv")
    }
    learned = routing[("full_without_dataset_source", 0.05)]
    confidence = routing[("self_reported_confidence", 0.05)]
    if int(float(learned["reviewed_count"])) != 32:
        raise AssertionError("Table 6 learned review count must be 32")
    if int(float(learned["detected_false_supports"])) != 15:
        raise AssertionError("Table 6 learned captured count must be 15")
    if int(float(confidence["detected_false_supports"])) != 13:
        raise AssertionError("Table 6 confidence captured count must be 13")

    audit = read_csv(
        ROOT / "artifacts/audits/pubmedqa_mapping_audit_anonymized.csv"
    )
    audit_counts = {
        "faithful": sum(row["final_claim_faithfulness"] == "VALID" for row in audit),
        "atomic": sum(row["final_atomicity"] == "ATOMIC" for row in audit),
        "compatible": sum(
            row["final_label_compatibility"] == "COMPATIBLE" for row in audit
        ),
        "pico": sum(row["final_pico_preservation"] == "COMPLETE" for row in audit),
        "modality": sum(
            row["final_modality_strength"] == "PRESERVED" for row in audit
        ),
    }
    if audit_counts != {
        "faithful": 60,
        "atomic": 55,
        "compatible": 44,
        "pico": 60,
        "modality": 60,
    }:
        raise AssertionError(f"Table 7 audit counts mismatch: {audit_counts}")

    generated = {"tables": 0, "figure_files": 0}
    if not args.skip_generated_files:
        tables = ROOT / args.tables_dir
        figures = ROOT / args.figures_dir
        expected_tables = [
            *(f"table{index}.csv" for index in range(1, 8)),
            *(f"table_s{index}.csv" for index in range(1, 7)),
        ]
        missing_tables = [name for name in expected_tables if not (tables / name).is_file()]
        if missing_tables:
            raise AssertionError(f"Missing generated tables: {missing_tables}")
        expected_figures = [
            f"figure{index}_{stem}.{suffix}"
            for index, stem in (
                (3, "formal600_evidence_funnel"),
                (4, "matched_budget_routing"),
                (5, "reliability_diagram"),
                (6, "frozen_risk_transfer"),
                (7, "entity_linking_audit"),
            )
            for suffix in ("png", "svg", "pdf")
        ]
        missing_figures = [
            name for name in expected_figures if not (figures / name).is_file()
        ]
        if missing_figures:
            raise AssertionError(f"Missing generated figures: {missing_figures}")
        generated = {
            "tables": len(expected_tables),
            "figure_files": len(expected_figures),
        }

    report = {
        "status": "passed",
        "manuscript_tables_checked": ["Table 4", "Table 6", "Table 7"],
        "manuscript_figures_checked": ["Figure 3", "Figure 6"],
        "formal600_funnel": funnel,
        "terminal_evidence_states": dict(sorted(states.items())),
        "external_risk_rows": len(external),
        "external_gold_used_for_risk_scoring": False,
        "generated_outputs": generated,
        "network_or_api_calls": False,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
