# Paper artifact map

`python reproduce_paper_artifacts.py` is the paper-facing entry point. It
recomputes frozen point estimates, rebuilds every manuscript table and every
data-derived figure, and checks key values. Conceptual Figures 1 and 2 are not
derived from result tables.

| Paper item | Released input | Generator or verification route |
|---|---|---|
| Table 1: KG resources | `config/kg_resources.json` | `scripts/tables/build_paper_tables.py` |
| Table 2: evidence states | frozen rule taxonomy in code and documentation | `scripts/tables/build_paper_tables.py` |
| Table 3: evaluation sets | split manifests and prediction counts | `scripts/tables/build_paper_tables.py` |
| Table 4: fair input matrix | `artifacts/results/fair_input_metrics.csv` | `scripts/tables/build_paper_tables.py` |
| Table 5: risk ablations | `artifacts/results/risk_ablation_summary.csv` | `scripts/tables/build_paper_tables.py` |
| Table 6: 5% routing | `artifacts/results/risk_routing_metrics.csv` | `scripts/tables/build_paper_tables.py` |
| Table 7: PubMedQA audit | `artifacts/audits/pubmedqa_mapping_audit_anonymized.csv` | `scripts/tables/build_paper_tables.py` |
| Figure 3: Formal600 KG funnel | `artifacts/predictions/risk_routing_scores.csv` | `scripts/figures/build_paper_figures.py` |
| Figure 4: matched-budget routing | `artifacts/results/risk_routing_metrics.csv` | `scripts/figures/build_paper_figures.py` |
| Figure 5: reliability diagram | `artifacts/predictions/risk_routing_scores.csv` | `scripts/figures/build_paper_figures.py` |
| Figure 6: frozen risk transfer | `artifacts/results/risk_ranking_metrics.csv`; `pubmedqa_frozen_risk_ranking.csv` | `scripts/figures/build_paper_figures.py` |
| Figure 7: entity-link audit | `artifacts/results/entity_linking_by_dataset.csv` | `scripts/figures/build_paper_figures.py` |
| Table S1: compatible-44 sensitivity | PubMedQA predictions, audit IDs, and `pubmedqa_frozen_risk_scores.csv` | `scripts/tables/build_paper_tables.py` |
| Tables S2–S3: fold/component sensitivity | `fold_composition.csv`; `largest_component_sensitivity.csv` | `scripts/tables/build_paper_tables.py` |
| Tables S4–S5: alternative samples | `artifacts/alternative_samples/*` | `scripts/tables/build_paper_tables.py` |
| Table S6: reviewer sensitivity | `reviewer_sensitivity_point_estimates.csv`; `risk_routing_metrics.csv` | `scripts/tables/build_paper_tables.py` |

`scripts/verify_paper_artifacts.py` checks the Formal600 funnel and terminal
states, frozen external AUROC/AP, 5% routing counts, PubMedQA audit totals, and
the presence of all generated paper artifacts.

Generated outputs are written under `outputs/reproduced/`, which is
intentionally ignored by Git.
