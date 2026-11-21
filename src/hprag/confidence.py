"""Gate 1: retrieval confidence — decide *before* generation whether there is evidence at all.

The cheapest hallucination to prevent is the one you never ask the LLM to write.
If retrieval found nothing relevant, any fluent answer can only come from the
model's memory (or imagination). So we score the evidence first and abstain
early when it is weak. This also saves an LLM call.

The score combines two signals, each in [0, 1]:

``top_score``
    Reranker score of the best chunk: "is at least one passage clearly about this?"

``coverage``
    IDF-weighted share of the question's content words that appear anywhere in
    the retrieved chunks. Rare words dominate. If the user asks about the
    "Zephyr module" and the corpus never mentions "zephyr", coverage collapses
    even if other words ("module") match — exactly the case where models invent.

``confidence = 0.5 * top_score + 0.5 * coverage``, with a hard penalty when a
rare, high-IDF query term is completely unknown to the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hprag.logging_utils import get_logger
from hprag.text import content_terms, query_terms
from hprag.types import ScoredChunk

log = get_logger("guard.confidence")


@dataclass
class ConfidenceReport:
    """Gate 1's score and the evidence behind it, so an abstention can explain itself."""
    confidence: float
    top_score: float
    coverage: float
    missing_terms: list[str] = field(default_factory=list)
    unknown_terms: list[str] = field(default_factory=list)  # never seen anywhere in the corpus


def retrieval_confidence(query: str, hits: list[ScoredChunk], idf, in_vocab) -> ConfidenceReport:
    """Score how well the retrieved chunks cover the question (see module docstring).

    ``idf`` and ``in_vocab`` come from the BM25 index, so rare words weigh more
    and words the corpus has never seen can be detected.
    """
    q_terms = list(dict.fromkeys(query_terms(query)))
    if not hits or not q_terms:
        return ConfidenceReport(0.0, 0.0, 0.0, missing_terms=q_terms)

    top = max(0.0, min(1.0, hits[0].scores.get("rerank", hits[0].score)))
    evidence_terms = set()
    for h in hits:
        evidence_terms.update(content_terms(h.chunk.text))

    unknown = [t for t in q_terms if not in_vocab(t)]
    # Unknown words get the maximum IDF weight: they are the most informative signal.
    max_idf = max([idf(t) for t in q_terms if in_vocab(t)] or [1.0])
    weights = {t: (idf(t) if in_vocab(t) else max_idf) for t in q_terms}
    total = sum(weights.values()) or 1.0
    missing = [t for t in q_terms if t not in evidence_terms]
    coverage = 1.0 - sum(weights[t] for t in missing) / total

    conf = 0.5 * top + 0.5 * coverage
    if unknown:
        # A specific entity/term the corpus has never heard of is the classic set-up
        # for a fabricated answer ("What did the 2019 report say about X?").
        conf *= 0.5
    report = ConfidenceReport(round(conf, 4), round(top, 4), round(coverage, 4), missing, unknown)
    log.info("scored", extra={"confidence": report.confidence, "top": report.top_score,
                              "coverage": report.coverage, "missing": missing, "unknown": unknown})
    return report
