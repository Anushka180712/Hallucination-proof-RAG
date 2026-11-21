"""Okapi BM25 lexical retrieval, implemented from scratch.

Why lexical retrieval in a modern RAG system?
    Dense embeddings are great at paraphrase but bad at exact tokens: product
    codes, names, numbers, rare jargon. Those exact tokens are precisely what
    a factual answer hinges on. BM25 nails them. Combining both (see
    :mod:`hprag.retrieval.hybrid`) gives better recall than either alone, and
    higher recall means fewer "the evidence wasn't there so the model guessed".
"""

from __future__ import annotations

import math
from collections import Counter

from hprag.logging_utils import get_logger
from hprag.text import content_terms, query_terms
from hprag.types import Chunk

log = get_logger("retrieval.bm25")


class BM25Index:
    """Okapi BM25 over stemmed content words."""
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        # k1 controls term-frequency saturation; b controls length normalisation.
        # 1.5 / 0.75 are the textbook defaults and work well on prose.
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._len: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 0.0

    def build(self, chunks: list[Chunk]) -> BM25Index:
        """Compute term frequencies, document frequencies and average length for the chunks."""
        self.chunks = list(chunks)
        self._tf = [Counter(content_terms(c.text)) for c in self.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._df = Counter(t for tf in self._tf for t in tf)
        self._avgdl = (sum(self._len) / len(self._len)) if self._len else 0.0
        log.info("built", extra={"chunks": len(self.chunks), "vocab": len(self._df)})
        return self

    def idf(self, term: str) -> float:
        """Inverse document frequency: high for rare words, which carry the most meaning."""
        n, df = len(self.chunks), self._df.get(term, 0)
        # The "+1" variant keeps IDF positive even for very common terms.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """Return up to ``k`` (chunk index, score) pairs with a positive score, best first."""
        terms = query_terms(query)
        if not terms or not self.chunks:
            return []
        scores = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._len[i] / (self._avgdl or 1))
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self.idf(t) * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        log.debug("search", extra={"terms": terms, "hits": len(scores)})
        return scores[:k]
