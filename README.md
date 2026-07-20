# Medical KG Evidence Guardrails

Core experimental code for **risk-aware, evidence-conditioned biomedical claim verification with local medical knowledge graphs**.

This repository contains executable research code only. Manuscripts, figures, generated tables, raw or frozen model predictions, completed annotation forms are intentionally excluded.

## Included pipeline

- biomedical claim normalization and PubMedQA-to-claim construction;
- Hetionet/PrimeKG graph building, semantic cleaning, and terminology integration;
- rule-based entity linking and bounded 1/2-hop local-subgraph retrieval;
- predicate, direction, endpoint, and qualifier compatibility checks;
- path-level Evidence Scorer training and strict nested cross-fitting;
- provided-text and text/KG evidence-conditioned LLM baselines;
- grouped risk evaluation, matched-budget review routing, bootstrap analysis, and transfer diagnostics;
- dual-annotator audit preparation, adjudication, and reliability analysis;
- largest-component and alternative label-balanced structural sensitivity analyses.

## Repository layout

```text
scripts/                  Core experiment, audit, and analysis programs
config/kg_resources.json Frozen KG resource/version metadata
docs/PIPELINE.md          Execution order and stage descriptions
docs/DATA_REQUIREMENTS.md Required local inputs and licensing notes
requirements.txt          Runtime Python dependencies
.env.example              Environment-variable template without credentials
```

## Installation

The reported experiments used Python 3.11. The repository validation also runs under Python 3.12.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the declared dependencies and validate the code-only checkout:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/validate_repository.py
```

The validation command parses every Python file, checks declared third-party dependencies, runs `--help` for every command-line entry point, verifies the DeepSeek configuration template, and rejects tracked manuscripts or generated artifacts. It does not call an API or require experiment data.

## Configuration

`.env.example` records the endpoint and request identifier used by the reported runs. The scripts read environment variables from the current process; they do **not** automatically load a local `.env` file. Set credentials in the shell or pass the corresponding command-line options.

```powershell
# Windows PowerShell example
$env:OPENAI_BASE_URL = "https://api.deepseek.com"
$env:OPENAI_MODEL = "deepseek-v4-flash"
$env:OPENAI_API_KEY = "<your-key>"
```

```bash
# Linux/macOS example
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-v4-flash"
export OPENAI_API_KEY="<your-key>"
```

UMLS reconstruction additionally requires `UMLS_API_KEY` and a valid UMLS license. Never commit credentials or licensed UMLS-derived files.

## Running the research pipeline

Run commands from the repository root. Generated inputs and outputs are expected under `data/` and `outputs/`, which are ignored by Git. The repository does not include those files, so the full experiments require the resources and intermediate schemas documented in [docs/DATA_REQUIREMENTS.md](docs/DATA_REQUIREMENTS.md).

Use [docs/PIPELINE.md](docs/PIPELINE.md) for the ordered workflow. Every CLI exposes its accepted paths and options:

```bash
python scripts/evaluate_eswa_nested_path_crossfit.py --help
python scripts/analyze_eswa_fold_component_sensitivity.py --help
python scripts/analyze_alternative_balanced_samples.py --help
```

The primary leakage-controlled Formal600 analysis is `evaluate_eswa_nested_path_crossfit.py`. It excludes outer-fold-linked path annotations when fitting the Evidence Scorer and uses inner out-of-fold scorer features for risk-model training. The alternative balanced-sample analysis is reviewer-motivated and post-hoc; its deterministic samples may overlap and its minimum–maximum ranges are descriptive, not confidence intervals.

## Interpretation boundaries

- Formal600 is a frozen label-balanced diagnostic subset, not a probability sample of the full standardized pool.
- PubMedQA-Claim-300 is a QA-derived label-noise stress test, not clinical validation or a clean confirmatory external gold standard.
- Local KG connectivity is a candidate evidence signal, not automatic claim-level support.
- The frozen request identifier is `deepseek-v4-flash`; hosted model weights may change without notice.
- No software license is granted unless a `LICENSE` file is added explicitly.

## Citation

Citation metadata will be added after publication.
