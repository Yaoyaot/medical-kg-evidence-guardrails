# Medical KG Evidence Guardrails

Core experimental code for **risk-aware, evidence-conditioned biomedical claim verification with local medical knowledge graphs**.

This repository contains the executable research pipeline only. Manuscripts, DOCX files, figures, generated tables, raw model predictions, and completed annotation forms are intentionally excluded.

## What is included

- biomedical claim normalization and PubMedQA-to-claim construction;
- Hetionet/PrimeKG graph building and semantic cleaning;
- Mondo, HGNC, and licensed UMLS terminology integration;
- rule-based entity linking and bounded 1/2-hop local-subgraph retrieval;
- predicate, direction, endpoint, and qualifier compatibility checks;
- path-level Evidence Scorer training and evaluation;
- provided-text and text/KG evidence-conditioned LLM baselines;
- grouped cross-fitting, risk ablations, matched-budget review routing, bootstrap evaluation, and transfer diagnostics;
- annotation reliability, path-artifact, and entity-linking audit analysis.

## Repository layout

```text
scripts/               Core experiment and analysis programs
config/kg_resources.json  Frozen resource/version metadata
docs/PIPELINE.md       Execution order and stage descriptions
docs/DATA_REQUIREMENTS.md  Expected local inputs and licensing notes
requirements.txt       Minimal Python dependencies
.env.example           API configuration template without credentials
```

## Installation

Python 3.11 was used for the reported experiments.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Copy `.env.example` to your preferred local environment configuration and set credentials outside version control. The scripts use repository-relative paths and write generated artifacts under `data/processed/` or `outputs/`; these paths are ignored by Git.

## Reproduction scope

The full sequence is documented in [docs/PIPELINE.md](docs/PIPELINE.md). Expensive LLM calls support dry-run/resume behavior where applicable. The final analyses use seed `20260618`; grouped bootstrap evaluation uses 5,000 resamples in the major-revision analyses.

The code does not redistribute UMLS content. Reconstructing UMLS-derived aliases requires a valid UMLS license and a local `UMLS_API_KEY`. Public datasets and knowledge resources remain subject to their original terms.

## Important interpretation

Local KG connectivity is treated as a candidate evidence signal, not automatically as claim-level support. The system separates text-conditioned verification, KG evidence-state assessment, risk ranking, and a fixed-budget human-review action.

## Citation and license

Citation metadata will be added after publication. No software license is granted unless a `LICENSE` file is added explicitly.
