from __future__ import annotations

import argparse
import json
from pathlib import Path

from detect_claim_qualifiers import detect_qualifiers
from hierarchical_evidence_features import structured_features
from repo_paths import find_repo_root
from strict_relation_alignment import (
    automatic_support_allowed,
    extract_claim_predicate_families,
    relation_family,
)


ROOT = find_repo_root()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a no-network synthetic smoke test of semantic evidence rules."
    )
    parser.add_argument(
        "--cases", default="examples/minimal/cases.json", help="Synthetic cases JSON."
    )
    parser.add_argument(
        "--expected",
        default="examples/minimal/expected.json",
        help="Expected summary JSON.",
    )
    args = parser.parse_args()

    cases = json.loads((ROOT / args.cases).read_text(encoding="utf-8"))
    qualifier_ok = []
    predicate_ok = []
    relation_ok = []
    action_ok = []
    features = []
    for case in cases:
        qualifiers = detect_qualifiers(case["claim"])
        predicates = extract_claim_predicate_families(case["claim"])
        family = relation_family(case["relation"])
        allowed = automatic_support_allowed(predicates, case["relation"])
        qualifier_ok.append(qualifiers == case["expected_qualifiers"])
        predicate_ok.append(predicates == case["expected_predicates"])
        relation_ok.append(family == case["expected_relation_family"])
        action_ok.append(allowed is case["expected_support_allowed"])
        features.append(
            structured_features(
                {
                    "dataset": "synthetic",
                    "path_type": case["path_type"],
                    "path_score": case["path_score"],
                    "relations": [case["relation"]],
                    "qualifier_flags": qualifiers,
                    "claim_predicate_families": predicates,
                    "kg_relation_family": family,
                    "endpoint_aligned": case["endpoint_aligned"],
                    "predicate_aligned": case["predicate_aligned"],
                }
            )
        )

    summary = {
        "cases": len(cases),
        "all_qualifier_checks_pass": all(qualifier_ok),
        "all_predicate_checks_pass": all(predicate_ok),
        "all_relation_checks_pass": all(relation_ok),
        "all_action_checks_pass": all(action_ok),
        "structured_feature_rows": len(features),
    }
    expected = json.loads((ROOT / args.expected).read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))
    if summary != expected:
        raise SystemExit("Synthetic example did not match expected output.")


if __name__ == "__main__":
    main()
