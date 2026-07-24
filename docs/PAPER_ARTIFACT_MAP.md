# Paper artifact map

The table maps the main empirical claims to released inputs and executable
verification routes. Conceptual system diagrams are not data-derived and are
therefore not rebuilt from CSV files.

| Paper item | Frozen input | Verification or generation route |
|---|---|---|
| Fair-input verifier comparison | `artifacts/predictions/formal600_predictions.jsonl` | `python reproduce_frozen_results.py` |
| PubMedQA label-noise stress test | `artifacts/predictions/pubmedqa_claim300_predictions.jsonl` | `python reproduce_frozen_results.py` |
| Evidence-conversion funnel | `artifacts/results/evidence_conversion_funnel.csv` | `scripts/figures/build_result_figures.py` |
| Evidence Scorer performance | `artifacts/predictions/evidence_scorer_oof_predictions.jsonl`; `artifacts/results/evidence_scorer_*` | `scripts/train_hierarchical_evidence_scorer.py` or direct inspection |
| Nested scorer leakage controls | `artifacts/data_splits/*`; `artifacts/predictions/risk_routing_scores.csv` | `scripts/evaluate_eswa_nested_path_crossfit.py`; manifest checks |
| Matched-budget routing | `artifacts/predictions/risk_routing_scores.csv` | `python reproduce_frozen_results.py` |
| Risk discrimination and ablation | `artifacts/results/risk_ranking_metrics.csv`; `risk_ablation_summary.csv` | direct machine-readable results |
| Risk-coverage analysis | `artifacts/results/risk_coverage_curve.csv` | `scripts/figures/build_result_figures.py` |
| Largest-component sensitivity | `artifacts/results/fold_composition.csv`; `largest_component_sensitivity.csv` | direct machine-readable results |
| Entity-link audit | `artifacts/audits/entity_linking_audit_anonymized.csv`; `artifacts/results/entity_linking_by_dataset.csv` | `scripts/analyze_stage9_human_audits.py`; `scripts/figures/build_result_figures.py` |
| PubMedQA conversion/label audit | `artifacts/audits/pubmedqa_mapping_audit_anonymized.csv` | `scripts/analyze_stage9_human_audits.py` |
| Alternative balanced samples | `artifacts/alternative_samples/*` | `scripts/analyze_alternative_balanced_samples.py` |
| KG runtime | `artifacts/results/kg_runtime_summary.csv` | descriptive benchmark source |
| Cost-sensitive analysis | `artifacts/results/cost_utility_point_estimates.csv` | `scripts/analyze_eswa_submission_closure.py` |

Generated verification outputs are written under `outputs/reproduced/`, which
is intentionally ignored by Git.
