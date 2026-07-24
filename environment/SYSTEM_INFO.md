# Reference execution environments

## Frozen statistical/model environment

- Python: 3.10.11
- scikit-learn: 1.5.2
- numerical and analysis packages: see `requirements-lock.txt`
- GPU: not required for the released frozen-result verification
- network: not required for quick or frozen-result verification

## KG runtime benchmark environment

- Python: 3.12.13
- operating system: Windows 11, build 26120
- logical CPU count: 16
- memory: approximately 15.9 GB
- execution mode: single-process Python, no explicit worker pool
- graph loaded for the benchmark: 71,410 nodes and 2,941,128 edges

Hardware timings are descriptive and will vary across machines. Full graph
reconstruction requires the upstream graph files and substantially more disk
space than the frozen-result verification. UMLS reconstruction additionally
requires a valid UMLS license. Hosted-model baselines require a DeepSeek API
credential and may not reproduce identical outputs because provider-side
weights can change.
