# Data and licensing requirements

Generated data are not committed to this code-only repository. Full reconstruction requires local copies of the following resources and preserves their original licenses.

| Resource | Role | Notes |
|---|---|---|
| MedFact-Bench, PubHealth, SciFact, HealthVer, and MedAESQA/project sources | Claim–evidence pairs | Follow each upstream license and access condition. |
| Hetionet v1.0 | Base biomedical KG | Public upstream resource. |
| PrimeKG | KG enrichment | Public Dataverse resource; semantic cleaning is implemented in code. |
| Mondo and HGNC | Open terminology aliases | Retain upstream provenance and terms. |
| UMLS 2026AA | Licensed terminology aliases | Requires a valid UMLS license; licensed derived content is not redistributed. |
| PubMedQA `pqa_labeled` | QA-derived stress-test construction | Downloaded through the Hugging Face Dataset Viewer workflow. |

## Expected local directory classes

```text
data/raw/         Downloaded public resources and local source datasets
data/private/     Licensed UMLS-derived files; never commit
data/interim/     Standardized intermediate records
data/processed/   Graphs, paths, predictions, annotations, and analyses
outputs/          Optional logs, prompt previews, and generated reports
```

All of these generated locations are ignored by Git. Scripts create output directories when possible, but they do not fabricate missing upstream data.

## Inputs for the primary strict nested analysis

`evaluate_eswa_nested_path_crossfit.py` expects, by default:

- the 474-row clean path-annotation modeling pool;
- the 600-row connected-component group manifest;
- Formal600 local subgraphs and strict evidence records;
- frozen guardrail/verifier results;
- the cleaned PrimeKG graph directory used to reconstruct adverse-event relation context.

Each path is overrideable through the script CLI. The group manifest and verifier results must contain exactly the same 600 IDs. Path annotations connected to an outer test component are excluded from that fold's Evidence Scorer training.

## Reproducibility records to preserve locally

For exact reproduction, retain:

- claim/source component IDs and fold assignments;
- annotation guidelines, independent labels, and adjudication decisions;
- prompt templates, request payload hashes, response hashes, and model identifier;
- KG resource versions, cleaning parameters, and relation mappings;
- random seed `20260618` and bootstrap iteration counts;
- the environment configuration used for each run.

Never commit API keys, licensed UMLS files, patient-identifying information, generated model responses, completed audit forms, or paper manuscripts to this code-only repository.
