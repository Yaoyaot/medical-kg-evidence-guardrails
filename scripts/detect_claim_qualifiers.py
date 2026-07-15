from __future__ import annotations

import re


QUALIFIER_PATTERNS = {
    "NUMERIC": [
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b\d+(?:,\d{3})+\b",
        r"\b\d+(?:\.\d+)?\s*(?:mg|g|ml|years?|months?|days?|hours?)\b",
        r"\bprevalence\b",
        r"\bincidence\b",
        r"\bmortality\b",
    ],
    "DIRECTIONAL": [
        r"\bbetter\b",
        r"\bworse\b",
        r"\bpoorer\b",
        r"\bhigher\b",
        r"\blower\b",
        r"\bincreas(?:e|es|ed|ing)\b",
        r"\bdecreas(?:e|es|ed|ing)\b",
        r"\breduc(?:e|es|ed|ing|tion)\b",
        r"\brais(?:e|es|ed|ing)\b",
    ],
    "POPULATION": [
        r"\bpatients?\b",
        r"\bsubjects?\b",
        r"\bparticipants?\b",
        r"\badults?\b",
        r"\bchildren\b",
        r"\binfants?\b",
        r"\bwomen\b",
        r"\bmen\b",
        r"\bmales?\b",
        r"\bfemales?\b",
        r"\baged?\b",
    ],
    "TEMPORAL": [
        r"\bbefore\b",
        r"\bafter\b",
        r"\bduring\b",
        r"\blong[- ]term\b",
        r"\bshort[- ]term\b",
        r"\binitial\b",
        r"\bearly\b",
        r"\blate\b",
    ],
    "DOSAGE": [
        r"\bdos(?:e|es|age)\b",
        r"\bdaily\b",
        r"\btwice\b",
        r"\bmg\b",
        r"\bml\b",
    ],
    "CONDITIONAL": [
        r"\bif\b",
        r"\bwhen\b",
        r"\bunder\b",
        r"\bin vitro\b",
        r"\bin vivo\b",
        r"\bhomozygous\b",
        r"\bheterozygous\b",
        r"\btrait\b",
    ],
}


def detect_qualifiers(claim: str) -> list[str]:
    text = (claim or "").lower()
    return [
        name
        for name, patterns in QUALIFIER_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    ]

