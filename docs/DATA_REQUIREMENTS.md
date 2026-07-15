# Data and licensing requirements

Generated data are not committed to this code-only repository. The pipeline expects local copies of the following resources or reconstructs them with the included scripts.

| Resource | Role | Notes |
|---|---|---|
| MedFact-Bench, PubHealth and project source datasets | Claim/evidence pairs | Follow each dataset's original license and access terms. |
| Hetionet v1.0 | Base biomedical KG | Public upstream resource. |
| PrimeKG | KG enrichment | Public Dataverse resource; semantic cleaning is implemented in code. |
| Mondo and HGNC | Open terminology aliases | Retain upstream provenance and terms. |
| UMLS 2026AA | Licensed terminology aliases | Requires a valid UMLS license; derived licensed content is not redistributed. |
| PubMedQA `pqa_labeled` | QA-derived stress-test construction | Downloaded through the Hugging Face Dataset Viewer workflow. |

Expected generated directories include `data/raw/`, `data/interim/`, `data/processed/`, and `outputs/`. They are excluded from Git because they contain large resources, API outputs, audit forms, or paper-specific generated artifacts.

For exact reproduction, preserve split/component IDs, prompt logs, model identifiers, graph metadata, and random seed `20260618`. Never commit API keys or licensed UMLS files.
