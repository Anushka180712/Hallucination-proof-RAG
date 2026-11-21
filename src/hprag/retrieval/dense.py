"""In-memory dense vector index (exact cosine search with numpy).

Why not FAISS / a vector database?
    For up to a few hundred thousand chunks, a single matrix multiply is fast
    (milliseconds), exact (no approximate-search recall loss), and has zero
    operational overhead. Swapping in a vector DB later only requires
    re-implementing ``build`` and ``search``.
"""

from __future__ import annotations

import numpy as np

from hprag.logging_utils import get_logger
from hprag.retrieval.embeddings import Embedder

log = get_logger("retrieval.dense")


class DenseIndex:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.matrix = np.zeros((0, embedder.dim), dtype=np.float32)

    def build(self, texts: list[str]) -> DenseIndex:
        self.matrix = self.embedder.embed(texts) if texts else self.matrix
        log.info("built", extra={"vectors": len(self.matrix), "dim": self.embedder.dim,
                                 "embedder": self.embedder.name})
        return self

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        if len(self.matrix) == 0:
            return []
        q = self.embedder.embed([query])[0]
        sims = self.matrix @ q  # rows are L2-normalised, so dot product == cosine
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx if sims[i] > 0]
