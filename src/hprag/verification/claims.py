"""Split a generated answer into atomic, checkable claims.

Why claim-level, not answer-level?
    An answer can be 90% correct and contain one invented number. Scoring the
    whole answer at once ("is this answer faithful?") averages that error away.
    Splitting into one claim per sentence lets us find, report and remove the
    single bad statement while keeping the good ones.

Hedges ("The sources do not say when...") are statements *about the evidence*,
not about the world. They are kept in the answer but excluded from faithfulness
scoring, because penalising honesty would push the system toward guessing.
"""

from __future__ import annotations

import re

from hprag.logging_utils import get_logger
from hprag.text import extract_citations, split_sentences
from hprag.types import Claim

log = get_logger("verification.claims")

_HEDGE = re.compile(
    r"\b(sources?|documents?|context|provided (text|information))\b.*\b(do(es)? not|don't|doesn't|"
    r"no information|not (mention|specify|say|state|contain|include|cover))"
    r"|\b(not (mentioned|specified|stated|covered) in)\b|\bno (information|details?) (is|are) (given|provided)",
    re.IGNORECASE,
)


def is_hedge(sentence: str) -> bool:
    """True if the sentence is about the evidence ('the sources do not say ...'), not the world."""
    return bool(_HEDGE.search(sentence))


def decompose(answer: str) -> tuple[list[Claim], list[str]]:
    """Return (claims to verify, hedge sentences)."""
    claims: list[Claim] = []
    hedges: list[str] = []
    pending: list[int] = []
    sentences = split_sentences(answer)
    for raw in sentences:
        text, cited = extract_citations(raw)
        if not text or len(text) < 3:
            # A bare "[2]." on its own line belongs to the previous sentence.
            if cited and claims:
                claims[-1].cited = sorted(set(claims[-1].cited + cited))
            continue
        if is_hedge(text):
            hedges.append(text)
            continue
        claims.append(Claim(text=text, cited=sorted(set(cited + pending))))
        pending = []
    # Logged so a reader can see exactly which sentences were checked and which were
    # set aside as honest hedges (hedges are never penalised; see the module docstring).
    log.debug("claims.split", extra={"claims": len(claims), "hedges": len(hedges),
                                     "uncited": sum(not c.cited for c in claims)})
    return claims, hedges
