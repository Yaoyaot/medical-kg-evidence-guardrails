# Medical KG Evidence Guardrails

Reproducibility materials for **risk-aware, evidence-conditioned biomedical claim verification with local medical knowledge graphs**.

The study evaluates a practical expert-system architecture in which provided textual evidence and local KG evidence are processed in parallel. KG connectivity is treated as a candidate audit signal rather than automatic claim-level support. A fixed-budget routing layer can send high-risk SUPPORT candidates for expert review while preserving the candidate label as unresolved.

## Repository contents

- `scripts/`: grouping, evaluation, Evidence Scorer, risk-routing, audit, figure, and manuscript-generation code;
- `data/processed/`: frozen predictions, connected-component IDs, annotations, adjudication records, grouped-bootstrap results, Stage 10 cost analysis, and runtime benchmarks;
- `outputs/figures_major_revision/`: publication figures in PNG, SVG, and PDF;
- `outputs/manuscript/`: near-submission manuscript source, DOCX, structural audit, and visual-QA record;
- `REPRODUCIBILITY.md`: recommended reconstruction and verification order;
- `DATA_AVAILABILITY.md`: redistribution and licensing boundaries;
- `release_manifest.json`: SHA-256 hashes for every tracked release file except the manifest itself.

## Key evaluation constraints

- Formal600 uses claim/source connected-component grouped cross-fitting.
- Risk routing uses out-of-fold predictions and fold-level matched review budgets.
- PubMedQA-Claim-300 is reported as a QA-derived label-noise stress test, not clinical validation.
- Stage 10 uses frozen predictions and seed `20260618`; it makes no new LLM calls and performs no risk-model retraining.
- UMLS 2026AA content is not redistributed.

## Quick verification

```bash
python scripts/verify_release_manifest.py
python -m compileall -q scripts
```

The complete experimental environment depends on licensed terminology resources and the original public datasets. See `REPRODUCIBILITY.md` and `DATA_AVAILABILITY.md` before attempting a full reconstruction.

## Manuscript status

The included manuscript is a near-submission version. Its scientific claims are intentionally conservative: the learned router does not reliably outperform self-reported confidence under leakage-controlled grouped evaluation, and frozen PubMedQA transfer cannot separate domain shift from reference-label mismatch.

## License

No software or data license has yet been assigned to this repository. Third-party datasets and graph resources remain governed by their original licenses.
