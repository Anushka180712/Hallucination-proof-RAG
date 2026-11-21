"""Second-stage rerankers.

Why rerank?
    First-stage retrievers optimise for *recall* over the whole corpus and score
    the query and passage independently. A reranker looks at each (query,
    passage) pair *together* and optimises for *precision* in the top few
    results. The top few are all the generator sees, so their precision directly
    controls how much irrelevant text the LLM can be distracted by.

``LexicalReranker`` (default, offline)
    Scores query-term coverage weighted by IDF, plus a proximity bonus when
    matched terms appear close together. Cheap, transparent, deterministic.

``CrossEncoderReranker`` (``pip install hprag[ml]``)
    A neural cross-encoder (e.g. ms-marco MiniLM). Much stronger; use it in prod.
"""

from __future__ import annotations

import math
from typing import Protocol

from hprag.logging_utils import get_logger
from hprag.text import content_terms, query_terms
from hprag.types import ScoredChunk

log = get_logger("retrieval.reranker")


class Reranker(Protocol):
    """Interface: reorder candidate chunks for one query, setting ``scores['rerank']``."""
    name: str

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """Return the hits re-sorted by relevance to the query."""
        ...


class NoReranker:
    """Keeps the fused order unchanged."""
    name = "none"

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """Return hits unchanged."""
        return hits


class LexicalReranker:
    """IDF-weighted query coverage plus a proximity bonus. Offline and explainable."""
    name = "lexical"

    def __init__(self, idf: dict[str, float] | None = None) -> None:
        self.idf = idf or {}

    def _score(self, q_terms: list[str], text: str) -> float:
        terms = content_terms(text)
        if not q_terms or not terms:
            return 0.0
        uniq_q = list(dict.fromkeys(q_terms))
        weights = {t: self.idf.get(t, 1.0) for t in uniq_q}
        total = sum(weights.values()) or 1.0
        present = [t for t in uniq_q if t in set(terms)]
        coverage = sum(weights[t] for t in present) / total
        # Proximity: smallest window containing all matched terms, relative to length.
        positions = [i for i, t in enumerate(terms) if t in present]
        proximity = 0.0
        if len(present) > 1 and positions:
            span = positions[-1] - positions[0] + 1
            proximity = len(present) / span
        return 0.8 * coverage + 0.2 * min(1.0, proximity)

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """Score each hit in [0, 1] and sort, best first."""
        q_terms = query_terms(query)
        for h in hits:
            h.scores["rerank"] = round(self._score(q_terms, h.chunk.text), 4)
            h.score = h.scores["rerank"]
        hits.sort(key=lambda h: -h.score)
        # Debug, not info: one line per question is enough at info level (retrieve.done).
        # Turn on --log-level DEBUG to see why the reranker preferred one chunk over another.
        log.debug("rerank.done", extra={"terms": q_terms, "top": [h.chunk.chunk_id for h in hits[:3]],
                                        "scores": [h.score for h in hits[:3]]})
        return hits


class CrossEncoderReranker:  # pragma: no cover - needs model download
    """Neural cross-encoder that reads query and passage together (needs hprag[ml])."""
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = f"ce:{model}"
        self._model = CrossEncoder(model)

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """Score each (query, passage) pair, squash logits to [0, 1], sort."""
        if not hits:
            return hits
        raw = self._model.predict([(query, h.chunk.text) for h in hits])
        for h, s in zip(hits, raw):
            h.scores["rerank"] = round(1 / (1 + math.exp(-float(s))), 4)  # logits -> [0, 1]
            h.score = h.scores["rerank"]
        hits.sort(key=lambda h: -h.score)
        return hits
