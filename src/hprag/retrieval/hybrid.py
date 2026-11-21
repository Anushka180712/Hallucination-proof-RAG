"""Hybrid retriever: BM25 + dense search, fused with Reciprocal Rank Fusion, then reranked.

Why Reciprocal Rank Fusion (RRF)?
    BM25 scores are unbounded; cosine similarities live in [-1, 1]. Adding them
    directly needs fragile per-corpus normalisation. RRF ignores raw scores and
    uses only *ranks*: ``score = sum(1 / (k + rank))``. It is parameter-light,
    robust, and consistently strong in the IR literature (Cormack et al., 2009).

The retriever also persists itself to a directory (``chunks.jsonl``,
``vectors.npy``, ``meta.json``) so ingestion happens once, not on every query.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hprag.config import RetrievalConfig
from hprag.logging_utils import get_logger, stage
from hprag.retrieval.bm25 import BM25Index
from hprag.retrieval.dense import DenseIndex
from hprag.retrieval.embeddings import make_embedder
from hprag.retrieval.reranker import (
    CrossEncoderReranker,
    LexicalReranker,
    NoReranker,
    Reranker,
)
from hprag.types import Chunk, ScoredChunk

log = get_logger("retrieval.hybrid")


def rrf(rankings: dict[str, list[tuple[int, float]]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for hits in rankings.values():
        for rank, (idx, _score) in enumerate(hits, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self, cfg: RetrievalConfig | None = None) -> None:
        self.cfg = cfg or RetrievalConfig()
        self.chunks: list[Chunk] = []
        self.bm25 = BM25Index()
        self.dense = DenseIndex(make_embedder(self.cfg.embedder, self.cfg.embedding_model))
        self.reranker: Reranker = NoReranker()

    # ------------------------------------------------------------------ building
    def build(self, chunks: list[Chunk]) -> HybridRetriever:
        with stage("index.build", log, chunks=len(chunks)):
            self.chunks = list(chunks)
            self.bm25.build(self.chunks)
            self.dense.build([c.text for c in self.chunks])
            self._init_reranker()
        return self

    def _init_reranker(self) -> None:
        kind = self.cfg.reranker
        if kind == "lexical":
            # Share BM25's IDF table so rare query words weigh more in reranking too.
            self.reranker = LexicalReranker({t: self.bm25.idf(t) for t in self.bm25._df})
        elif kind == "cross-encoder":
            self.reranker = CrossEncoderReranker(self.cfg.cross_encoder_model)
        else:
            self.reranker = NoReranker()

    # ----------------------------------------------------------------- searching
    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        top_k = top_k or self.cfg.top_k
        with stage("retrieve", log, query=query) as out:
            lexical = self.bm25.search(query, self.cfg.candidate_k)
            dense = self.dense.search(query, self.cfg.candidate_k)
            fused = rrf({"bm25": lexical, "dense": dense}, self.cfg.rrf_k)
            lex_map, den_map = dict(lexical), dict(dense)
            candidates = [
                ScoredChunk(
                    chunk=self.chunks[i],
                    score=s,
                    scores={"bm25": round(lex_map.get(i, 0.0), 4),
                            "dense": round(den_map.get(i, 0.0), 4),
                            "rrf": round(s, 5)},
                )
                for i, s in sorted(fused.items(), key=lambda x: -x[1])
            ]
            ranked = self.reranker.rerank(query, candidates)[:top_k]
            out.update(bm25_hits=len(lexical), dense_hits=len(dense), fused=len(candidates),
                       returned=[h.chunk.chunk_id for h in ranked])
        for rank, h in enumerate(ranked, 1):
            log.debug("hit", extra={"rank": rank, "chunk": h.chunk.chunk_id, **h.scores})
        return ranked

    def idf(self, term: str) -> float:
        return self.bm25.idf(term)

    def in_vocab(self, term: str) -> bool:
        return term in self.bm25._df

    # --------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        np.save(p / "vectors.npy", self.dense.matrix)
        (p / "meta.json").write_text(json.dumps({"retrieval": asdict(self.cfg),
                                                 "n_chunks": len(self.chunks)}, indent=2))
        log.info("index.saved", extra={"path": str(p), "chunks": len(self.chunks)})

    @classmethod
    def load(cls, path: str | Path, cfg: RetrievalConfig | None = None) -> HybridRetriever:
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        saved = RetrievalConfig(**meta["retrieval"])
        cfg = cfg or saved
        if cfg.embedder != saved.embedder:
            # Vectors from a different embedder live in a different space: refuse
            # rather than return silently wrong neighbours.
            raise ValueError(f"Index built with embedder {saved.embedder!r}, config asks for "
                             f"{cfg.embedder!r}. Re-run `hprag ingest`.")
        r = cls(cfg)
        r.chunks = [Chunk(**json.loads(ln)) for ln in
                    (p / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if ln]
        r.bm25.build(r.chunks)
        r.dense.matrix = np.load(p / "vectors.npy")
        r._init_reranker()
        log.info("index.loaded", extra={"path": str(p), "chunks": len(r.chunks)})
        return r
