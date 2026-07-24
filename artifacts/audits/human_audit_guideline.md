# Stage 9 dual-annotator audit instructions

For the detailed Chinese decision rules, JSON schemas, and worked examples, see `stage9_dual_annotation_guideline_detailed_zh.md` in the same directory.

## Independence

Annotators A and B complete their files independently and must not inspect the other annotator's answers. Adjudication begins only after both files are frozen.

## PubMedQA label-mapping audit

- `claim_faithfulness`: VALID or INVALID. The declarative claim must preserve the question proposition.
- `atomicity`: ATOMIC or NON_ATOMIC.
- `label_compatibility`: COMPATIBLE, INCOMPATIBLE, or AMBIGUOUS. Judge whether yes/no/maybe, after mapping, functions as SUPPORT/REFUTE/UNCERTAIN for the converted claim given the supplied context.
- `pico_preservation`: COMPLETE, PARTIAL, or NOT_APPLICABLE. Check population, intervention/exposure, comparator, and outcome when present.
- `modality_strength`: PRESERVED, WEAKENED, STRENGTHENED, or CHANGED. Pay special attention to association, prediction, causality, necessity, and uncertainty.

## Entity-linking audit

Annotate every biomedical mention in the claim, including mentions missed by the system. JSON fields must contain valid JSON arrays. A concept record should use `{"mention":"...","concept_id":"...","entity_type":"..."}`. Record incorrect model links separately, and mark abbreviation ambiguity YES or NO. `overall_linking_judgment` is CORRECT, PARTIAL, or INCORRECT.

Do not infer clinical truth. Judge conversion/label semantics or entity identity only.
