from __future__ import annotations

import re


CONTEXT_ONLY_RELATIONS = {"participates", "includes", "localizes", "resembles", "adverse_event_context"}
NEGATIVE_RELATIONS = {"contraindicates", "not_presents", "not_expressed"}
RELATION_FAMILIES = {
    "treats": "TREATS",
    "palliates": "TREATS",
    "causes": "CAUSES",
    "associates": "ASSOCIATES",
    "interacts": "INTERACTS",
    "targets": "TARGETS",
    "expresses": "EXPRESSION",
    "upregulates": "EXPRESSION",
    "downregulates": "EXPRESSION",
    "regulates": "EXPRESSION",
    "presents": "PRESENTS",
    "contraindicates": "CONTRAINDICATES",
    "not_presents": "PRESENTS",
    "not_expressed": "EXPRESSION",
}

# Ordered from more specific expressions to broader ones.
PREDICATE_PATTERNS = [
    ("CONTRAINDICATES", [r"\bcontraindicat(?:e|es|ed|ion|ions)\b", r"\bshould not be used\b"]),
    ("TREATS", [r"\btreat(?:s|ed|ing|ment|ments)?\b", r"\btherap(?:y|ies|eutic)\b", r"\bpalliat(?:e|es|ed|ive)\b", r"\bprevent(?:s|ed|ing|ion)?\b"]),
    ("TARGETS", [r"\btarget(?:s|ed|ing)?\b"]),
    ("INTERACTS", [r"\binteract(?:s|ed|ing|ion|ions)?\b", r"\bbind(?:s|ing)?\b"]),
    ("EXPRESSION", [r"\bexpress(?:es|ed|ion)?\b", r"\bupregulat(?:e|es|ed|ion)\b", r"\bdownregulat(?:e|es|ed|ion)\b", r"\bregulat(?:e|es|ed|ion)\b"]),
    ("PRESENTS", [r"\bpresent(?:s|ed|ing)?\s+(?:as|with)\b", r"\bsymptom(?:s|atic)?\b"]),
    ("CAUSES", [r"\bcause(?:s|d|ing)?\b", r"\blead(?:s|ing)?\s+to\b", r"\bresult(?:s|ed|ing)?\s+in\b", r"\bincreas(?:e|es|ed|ing)\s+(?:the\s+)?risk\b", r"\bside effect(?:s)?\b"]),
    ("ASSOCIATES", [r"\bassociat(?:e|es|ed|ion)\b", r"\bcorrelat(?:e|es|ed|ion)\b", r"\blink(?:s|ed|ing)?\s+(?:to|with)\b", r"\brelated\s+to\b"]),
]


def extract_claim_predicate_families(claim: str) -> list[str]:
    text = (claim or "").lower()
    output = []
    for family, patterns in PREDICATE_PATTERNS:
        if any(re.search(pattern, text) for pattern in patterns):
            output.append(family)
    return output or ["UNRESOLVED"]


def relation_family(relation: str) -> str:
    if relation in CONTEXT_ONLY_RELATIONS:
        return "CONTEXT_ONLY"
    return RELATION_FAMILIES.get(relation, "UNRESOLVED")


def predicate_aligned(claim_families: list[str], relation: str) -> bool:
    family = relation_family(relation)
    return family not in {"CONTEXT_ONLY", "UNRESOLVED"} and family in claim_families


def automatic_support_allowed(claim_families: list[str], relation: str) -> bool:
    return (
        predicate_aligned(claim_families, relation)
        and relation not in NEGATIVE_RELATIONS
        and relation_family(relation) != "CONTRAINDICATES"
    )
