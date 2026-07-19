# Experimental pipeline

Run every command from the repository root. Paths shown below are repository-relative defaults; use `python scripts/<name>.py --help` to inspect overrideable arguments. Generated files are not committed.

## 0. Code-only validation

```bash
python -m pip install -r requirements.txt
python scripts/validate_repository.py
```

This check requires no data and issues no API requests.

## 1. Claims and biomedical resources

1. `prepare_week1_datasets.py` standardizes source claim–evidence datasets.
2. `build_hetionet_graph.py` constructs the Hetionet base graph.
3. `build_open_terminology_aliases.py` constructs Mondo/HGNC aliases.
4. `build_umls_aliases.py` constructs licensed UMLS aliases.
5. `build_primekg_graph.py` imports the filtered PrimeKG graph.
6. `build_primekg_semantic_clean_graph.py` removes erroneous reverse drug-effect edges and downgrades retained drug-to-adverse-event edges to contextual relations.
7. `merge_local_medical_graphs.py` merges the cleaned resources.

## 2. Local KG evidence

1. `link_claim_entities.py` links claim mentions to graph concepts.
2. `retrieve_local_subgraphs.py` retrieves bounded 1-hop and 2-hop paths.
3. `detect_claim_qualifiers.py` detects negation, uncertainty, modality, comparison, and related qualifiers.
4. `strict_relation_alignment.py` maps claim predicate families to permitted KG relation families.
5. `build_strict_kg_evidence.py` assigns nominal evidence states from grounding, topology, endpoints, predicates, direction, qualifiers, and conflicts.

## 3. Path annotations and Evidence Scorer

1. `validate_path_annotations.py` validates enumerations and annotation completeness.
2. `evaluate_annotation_agreement.py` reports overlap and agreement.
3. `train_hierarchical_evidence_scorer.py` trains and evaluates the path-level Evidence Scorer with grouped validation.
4. `build_eswa_major_revision_audits.py` creates the dual-annotator PubMedQA and entity-linking audit templates.
5. `prepare_stage9_audit_adjudication.py` validates completed independent annotations and creates adjudication tables.
6. `analyze_stage9_human_audits.py` computes adjudicated conversion, label-compatibility, and entity-linking audit statistics.

Completed annotation and adjudication files are intentionally excluded from this public code repository.

## 4. Formal600 verifier inputs

1. `build_eswa_major_revision_groups.py` constructs connected claim/source components.
2. `run_eswa_llm_baselines.py` runs the matched input matrix; `run_llm_baselines.py` contains shared prompting and API utilities.
3. `evaluate_eswa_fair_input_baselines.py` evaluates frozen verifier outputs.

Use `--dry-run` where provided to inspect prompts before an API call. Gold labels must not be inserted into prompts or risk-ranking features.

## 5. Strict nested cross-fitting and review routing

1. `evaluate_eswa_nested_path_crossfit.py` is the primary leakage-controlled analysis. For each outer fold it excludes path annotations linked to that fold, generates inner out-of-fold Evidence Scorer aggregates for risk-model training, and produces unseen-fold risk scores.
2. `evaluate_matched_budget_guardrails.py` and `summarize_eswa_risk_ablations.py` compute matched-budget routing and ablations.
3. `analyze_eswa_fold_component_sensitivity.py` reports five-fold composition and recomputes frozen-score sensitivity after removing the 94-record largest component.
4. `analyze_eswa_risk_stability.py`, `analyze_eswa_annotation_reliability.py`, `analyze_eswa_evidence_scorer_detail.py`, and `analyze_eswa_path_artifacts.py` provide stability, class-level, and artifact diagnostics.
5. `analyze_eswa_submission_closure.py` computes runtime and cost-sensitive analyses from an explicitly supplied frozen risk-score file.

`evaluate_eswa_major_revision.py` remains a shared implementation and diagnostic entry point. Claims of end-to-end leakage control should rely on the strict nested output from `evaluate_eswa_nested_path_crossfit.py`.

## 6. QA-derived label-noise stress test

1. `build_pubmedqa_claim300.py` samples and converts PubMedQA questions using only `pubid + question` in the conversion prompt.
2. `run_pubmedqa_external.py` runs the frozen methods with dry-run/resume support.
3. `evaluate_eswa_major_revision_external.py` evaluates frozen transfer descriptively.
4. `analyze_eswa_review_revision.py` recomputes reviewer-requested audit and frozen-result diagnostics without API calls.

PubMedQA-Claim-300 is analyzed as a label-noise stress test, not clinical validation or a clean confirmatory external gold standard.

## 7. Reviewer-motivated structural sensitivity

`analyze_alternative_balanced_samples.py` creates ten deterministic label-balanced samples in each of two sampling frames and summarizes component structure plus candidate KG coverage. This analysis is post-hoc, issues no API requests, and does not compare verifier accuracy. Samples can overlap; reported minimum–maximum ranges are descriptive summaries rather than confidence intervals or independent sampling replicates.
