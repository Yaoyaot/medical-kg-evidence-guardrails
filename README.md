# Medical KG Evidence Guardrails

Anonymous review repository for **risk-aware, evidence-conditioned biomedical
claim verification with local medical knowledge graphs**.

## Scope

The repository contains:

- core experiment, graph, audit, and statistical analysis code;
- frozen Formal600 and PubMedQA parsed predictions without raw API responses;
- component, fold, and Evidence Scorer exclusion records;
- anonymized path, PubMedQA mapping, and entity-linking audit materials;
- prompt templates and request-hash manifests;
- machine-readable result tables and provenance;
- a no-network synthetic smoke test;
- one-command quick, frozen-result, and paper-artifact verification.

It excludes API credentials, UMLS licensed content, large knowledge graphs,
third-party paired source text, raw API response archives, manuscripts,
identity-bearing annotation notes, caches, logs, and debugging outputs.

## Repository layout

```text
artifacts/                 Curated splits, predictions, audits, and results
config/                    Frozen resource and experiment metadata
docs/                      Pipeline, data, artifact, and license documentation
environment/               Reference environment information
examples/minimal/          No-network synthetic smoke test
scripts/                   Core experiment and validation programs
reproduce_quick.py         Dependency-light repository verification
reproduce_frozen_results.py Recompute primary point estimates
reproduce_paper_artifacts.py Rebuild manuscript tables and data-derived figures
reproduce_full_pipeline.py  Preflight externally resourced reconstruction
```

## Installation

The frozen statistical/model environment used Python 3.10.11. A compatible
environment can be created with either pip or conda.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Alternatively:

```bash
conda env create -f environment/environment.yml
conda activate medical-kg-evidence-guardrails
```

The looser `requirements.txt` is retained for development compatibility;
`requirements-lock.txt` is the authoritative frozen package set.

## Quick validation

```bash
python reproduce_quick.py
```

Expected final line:

```text
Quick validation passed: repository, manifests, and synthetic rules.
```

This command does not use the network, an API credential, UMLS, or experiment
data outside the repository.

## Frozen-result verification

```bash
python reproduce_frozen_results.py
```

The command recomputes:

- Formal600 and PubMedQA fair-input classification metrics;
- fold-allocated matched-budget review-routing point estimates;
- primary 5% routing quantities from the released unseen-component scores.
- compact verification tables and three data-derived reviewer figures.

It checks the recomputed values against `artifacts/results/` and writes only
ignored outputs under `outputs/reproduced/`. It does not call an API.

## Manuscript table and figure reconstruction

```bash
python reproduce_paper_artifacts.py
```

This no-network command:

- reruns frozen point-estimate verification;
- rebuilds manuscript Tables 1–7 and Supplementary Tables S1–S6;
- rebuilds data-derived Figures 3–7 as PNG (1200 dpi), SVG, and PDF;
- checks manuscript-facing invariants, including the Formal600 evidence funnel,
  PubMedQA frozen-risk transfer values, 5% routing counts, and audit totals.

Figures 1 and 2 are conceptual system diagrams and are not derived from result
tables. Generated files are written under the ignored
`outputs/reproduced/` directory and are not versioned.

## Full reconstruction

Full reconstruction requires upstream datasets, Hetionet, PrimeKG, Mondo,
HGNC, and a valid UMLS license for the UMLS branch. Hosted-model baselines also
require a DeepSeek credential and may not reproduce identical text because
provider-side weights can change.

Follow:

1. `docs/THIRD_PARTY_DATA.md`;
2. `docs/DATA_REQUIREMENTS.md`;
3. `docs/PIPELINE.md`.

Run the non-destructive preflight before starting:

```bash
python reproduce_full_pipeline.py --strict
```

Add `--require-umls` and/or `--require-api` only for branches that use those
resources. The entry point checks availability and prints the ordered
reconstruction route; it does not download licensed data or silently initiate
paid requests.

All command-line entry points expose their accepted paths through `--help`.
The primary component-controlled analysis is:

```bash
python scripts/evaluate_eswa_nested_path_crossfit.py --help
```

For each outer fold it excludes linked path annotations from the Evidence
Scorer and uses inner out-of-fold scorer features for risk-model training.

## Released artifacts

### Data splits and leakage controls

`artifacts/data_splits/` contains Formal600 membership, claim/source component
IDs, outer folds, inner feature assignments, path-to-component mappings, and
scorer exclusion manifests. Claim and paired-source text are represented by
hashes in the split files.

### Predictions and risk scores

`artifacts/predictions/` contains parsed labels, confidence, error status,
prompt-hash availability, and risk scores. The compact PubMedQA frozen-risk
export contains only IDs, labels, audit state, and risk fields required for
Figure 6 and Supplementary Table S1. Claim text, source text, raw response text,
and model reasoning are not included.

### Human audits

`artifacts/audits/` contains anonymized path labels, overlap adjudication,
artifact exclusions, PubMedQA conversion/label audit, and entity-linking
audit. Full upstream claims, questions, source excerpts, annotator identity,
free-text notes, and local paths have been removed. Stable IDs and SHA-256
text hashes support reconstruction and integrity checks after users obtain the
upstream datasets. Short biomedical mention spans required to recompute the
entity-linking audit are retained.

### Results

`artifacts/results/` contains the machine-readable inputs for the main
empirical tables and figures. `docs/PAPER_ARTIFACT_MAP.md` maps claims and
outputs to their verification route.

### Reviewer-motivated structural sensitivity

The ten deterministic alternative samples within each frame may overlap.
Their minimum–maximum ranges are descriptive summaries of selected seeds, not
confidence intervals, independent sampling replicates, or external performance
validation.

## API/model disclosure

The reported requests used:

- DeepSeek official API;
- request model identifier `deepseek-v4-flash`;
- temperature `0`;
- no generation seed;
- provider-default maximum tokens;
- up to three retries with exponential backoff.

`.env.example` contains no credential. The scripts read variables from the
current process and do not automatically load a local `.env`.

```powershell
$env:OPENAI_BASE_URL = "https://api.deepseek.com"
$env:OPENAI_MODEL = "deepseek-v4-flash"
$env:OPENAI_API_KEY = "<your-key>"
```

API access is not needed for released frozen-result verification.

## Integrity

`CODE_MANIFEST.json` is the canonical manifest. `SHA256SUMS.txt` is generated
from the same file set. Neither manifest includes itself.

```bash
python scripts/rebuild_repository_manifest.py --check
python scripts/validate_repository.py
```

The validator supports both a Git checkout and a downloaded archive without
`.git`.

## Interpretation boundaries

- Formal600 is a frozen label-balanced diagnostic subset, not a probability
  sample of the standardized pool.
- PubMedQA-Claim-300 is a QA-derived label-noise stress test, not clinical
  validation or a clean confirmatory external gold standard.
- Automatic KG coverage is a candidate-coverage funnel, not reliable semantic
  coverage.
- Local KG connectivity is not automatic claim-level support.
- The hosted request identifier does not guarantee immutable provider weights.

## Licensing and data availability

Original repository code is released under the MIT License. Third-party
resources retain their upstream licenses and are described in
`THIRD_PARTY_LICENSES.md`.

This anonymous repository is intended for peer review. Citation metadata and a
permanent archival identifier will be added after publication.
