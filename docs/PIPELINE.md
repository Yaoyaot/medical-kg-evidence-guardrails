# Experimental pipeline

All paths below are relative to the repository root. Use each script's `--help` output for concrete arguments and overrideable paths.

## 1. Claims and biomedical resources

1. `prepare_week1_datasets.py` standardizes the source claim datasets.
2. `build_hetionet_graph.py` constructs the base heterogeneous graph.
3. `build_open_terminology_aliases.py` and `build_umls_aliases.py` construct terminology aliases.
4. `build_primekg_graph.py`, `build_primekg_semantic_clean_graph.py`, and `merge_local_medical_graphs.py` integrate and clean PrimeKG.

## 2. Local KG evidence

1. `link_claim_entities.py` links claim mentions to KG concepts.
2. `retrieve_local_subgraphs.py` retrieves bounded 1-hop and 2-hop paths.
3. `detect_claim_qualifiers.py` detects negation, uncertainty, modality, comparison, and related qualifiers.
4. `strict_relation_alignment.py` maps claim predicate families to permitted KG relations.
5. `build_strict_kg_evidence.py` assigns evidence states from grounding, topology, endpoints, predicates, direction, qualifiers, and conflicts.

## 3. Evidence learning and guardrails

1. `validate_path_annotations.py` and `evaluate_annotation_agreement.py` validate path labels and agreement.
2. `train_hierarchical_evidence_scorer.py` trains the frozen path Evidence Scorer with grouped validation.
3. `build_text_rag_kg_guardrail.py` derives rule and learned KG-risk features without using evaluation gold labels.

## 4. Formal600 and frozen LLM inputs

1. `build_eswa_major_revision_groups.py` creates connected claim/source grouping components for leakage-aware evaluation.
2. `run_eswa_llm_baselines.py` runs the matched input matrix. `run_llm_baselines.py` contains shared prompting and API utilities.
3. `evaluate_eswa_fair_input_baselines.py` reports frozen verifier results.
4. `evaluate_eswa_major_revision.py` performs grouped cross-fitting, matched-budget review routing, and group bootstrap analysis.
5. The remaining `analyze_eswa_*` scripts compute ablations, stability, artifact, calibration, audit, runtime, and cost-sensitive diagnostics from frozen predictions.

## 5. QA-derived stress test

1. `build_pubmedqa_claim300.py` samples and converts PubMedQA questions using only `pubid + question` in the conversion prompt.
2. `run_pubmedqa_external.py` runs frozen methods and supports dry-run/resume modes.
3. `evaluate_eswa_major_revision_external.py` evaluates frozen transfer descriptively.
4. `analyze_stage9_human_audits.py` quantifies conversion, label-compatibility, and entity-linking audits.

The PubMedQA-derived collection is analyzed as a **label-noise stress test**, not as clinical validation or a clean confirmatory external gold standard.
