# Third-party materials

The MIT License in this repository applies only to original code and
repository documentation created for this project. It does not relicense
third-party datasets, knowledge graphs, terminologies, model services, or
their derived restricted content.

| Resource | Use in this study | Redistribution in this repository |
|---|---|---|
| MedFact-Bench and its component sources | Claim-evidence benchmark construction | Record IDs, labels, hashes, and derived predictions only; upstream text is not redistributed here |
| PubHealth, SciFact, HealthVer, and MedAESQA/project sources | Upstream claim-evidence records | Follow the license and access terms of each upstream source |
| PubMedQA `pqa_labeled` | QA-derived label-noise stress test | IDs, labels, text hashes, and audit decisions only; complete questions and converted claims are not redistributed |
| Hetionet v1.0 | Base biomedical knowledge graph | Full graph files are not redistributed; obtain them from the upstream project |
| PrimeKG, Harvard Dataverse datafile 6180620 | Graph enrichment | Full graph files are not redistributed; obtain them from Harvard Dataverse |
| Mondo | Disease terminology normalization | Full source files are not redistributed; obtain them from the upstream release |
| HGNC | Gene terminology normalization | Full source files are not redistributed; obtain them from HGNC |
| UMLS 2026AA | Licensed terminology normalization | No UMLS files, definitions, or licensed alias tables are redistributed |
| DeepSeek API | Hosted verifier and conversion requests | No credentials or raw response archives are redistributed |

Upstream URLs, versions, retrieval dates, access restrictions, and expected
local locations are documented in `docs/THIRD_PARTY_DATA.md` and
`config/dataset_versions.json`. Users are responsible for complying with all
upstream terms.
