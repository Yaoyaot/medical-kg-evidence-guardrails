# Frozen prompt templates

## Evidence-conditioned classifier

```text
You are a strict medical fact-checking system.

Task: Determine whether the claim is supported, refuted, or uncertain.

[Claim]
{claim}

[Evidence]
{condition-specific evidence}

Rules:
- SUPPORT means the evidence clearly entails the claim.
- REFUTE means the evidence clearly contradicts the claim.
- UNCERTAIN means evidence is missing, ambiguous, unrelated, or insufficient.
- Do not invent evidence.

Return ONLY valid JSON with keys: "label", "confidence", "reasoning".
The label must be exactly one of "SUPPORT", "REFUTE", "UNCERTAIN".
Confidence must be a number between 0 and 1.
```

The evidence block was instantiated as one of the following:

- provided claim-associated text;
- provided text followed by additional BM25 passages;
- provided text followed by local KG paths;
- provided text, local KG paths, and additional BM25 passages;
- BM25-only or KG-only evidence for diagnostic conditions.

## PubMedQA question-to-claim conversion

```text
Rewrite each biomedical yes/no/maybe question as one atomic declarative
proposition whose truth is exactly what the question asks. Do not answer the
question and do not add facts. Preserve negation, modality, comparisons,
populations, interventions, outcomes, time qualifiers, numbers, and biomedical
names. Use 5-60 words, one sentence, and no question mark.

Return only valid JSON in this exact shape:
{"items":[{"pubid":123,"claim":"One declarative proposition."}]}

Items:
[{pubid and question only}]
```

The conversion request contained only `pubid` and `question`. It excluded
`final_decision`, mapped label, context, and long answer.

## Request disclosure

- Provider: DeepSeek official API
- Request model identifier: `deepseek-v4-flash`
- Temperature: `0`
- Generation seed: not set
- Top-p: not set
- Maximum tokens: provider default
- Retry policy: at most three retries with exponential backoff

Some legacy baseline records predated per-request prompt hashing. Their
released parsed predictions are marked accordingly in
`artifacts/api_manifest/request_hashes.csv`.
