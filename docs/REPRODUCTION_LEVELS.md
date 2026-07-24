# Reproduction levels

The repository distinguishes three scopes so that a reviewer can verify the
paper without obtaining licensed resources or making paid API calls.

## Quick validation

```bash
python reproduce_quick.py
```

This checks required files, hashes, anonymity patterns, Python syntax, and a
small synthetic evidence-rule example. It requires no network, UMLS, KG, or API
credential.

## Frozen-result verification

```bash
python reproduce_frozen_results.py
```

This reads released parsed predictions and risk scores, recomputes the main
classification and matched-budget routing point estimates, and checks them
against the released result tables. It makes no network or API calls.

## Full reconstruction

Full reconstruction starts from upstream data, terminology, and graph
resources and follows `docs/PIPELINE.md`. It requires resources described in
`docs/THIRD_PARTY_DATA.md`, including a licensed UMLS installation for the UMLS
normalization branch. Hosted-model outputs may differ because the service can
change provider-side weights without changing the request identifier.

Use `python reproduce_full_pipeline.py --strict` as a non-destructive resource
preflight. Add `--require-umls` or `--require-api` for branches that depend on
those resources. The preflight performs no downloads and no hosted-model
requests; the actual ordered stage commands remain explicit in
`docs/PIPELINE.md` so reviewers can inspect every boundary.

The frozen-result verification is therefore the authoritative exact
verification route for the reported LLM outputs.
