# Third-party data and reconstruction

This repository does not redistribute large upstream graphs, licensed UMLS
content, raw API response archives, or third-party source text. Exact resource
versions are recorded in `config/dataset_versions.json` and
`config/kg_resources.json`.

| Resource | Expected local role | Access and restriction |
|---|---|---|
| MedFact-Bench revision `249028c` and component sources | Standardized claim-evidence pool | Obtain from the upstream project and follow each component license |
| PubMedQA `qiaojin/PubMedQA`, `pqa_labeled/train` | QA-derived claim stress test | Obtain from Hugging Face or the upstream PubMedQA project |
| Hetionet v1.0 | Base graph | Obtain from the Hetionet project |
| PrimeKG datafile 6180620 | Graph enrichment | Obtain from Harvard Dataverse |
| Mondo 2026-05-05 | Disease aliases | Obtain from the Mondo release |
| HGNC snapshot retrieved 2026-05-30 | Gene aliases | Obtain from HGNC |
| UMLS 2026AA | Licensed terminology aliases | Requires a valid UMLS license; never commit downloaded or derived licensed tables |

Place downloaded public resources under `data/raw/` and licensed UMLS material
under `data/private/`. Both locations are ignored by Git. Scripts accept path
overrides; run `python <script> --help` for the exact expected inputs.

The released `artifacts/data_splits/` files contain record IDs, labels, hashes,
component IDs, and fold assignments without redistributing paired source text.
The path-modeling export replaces complete claim text with a SHA-256 hash and
retains the KG path fields and labels required to audit the Evidence Scorer.
The PubMedQA and entity-linking audit exports likewise replace complete
questions and claims with hashes. Source excerpts, linked alias lists,
annotator notes, and local paths are removed. Short annotated biomedical
mention spans are retained because they are required to recompute mention- and
concept-linking metrics.

After obtaining the upstream resources:

1. run `scripts/prepare_week1_datasets.py`;
2. construct terminology and graph resources in the order in `docs/PIPELINE.md`;
3. verify the expected graph counts in `config/kg_resources.json`;
4. run the strict evidence and nested-cross-fitting stages;
5. compare generated outputs with `artifacts/results/`.

Failure to obtain a licensed or third-party resource is expected to stop the
corresponding full-reconstruction stage with a missing-input message. Quick and
frozen-result verification remain available.
