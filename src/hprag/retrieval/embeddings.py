"""Pluggable text embedders.

``HashingEmbedder`` (default)
    Feature hashing of word unigrams + bigrams + character trigrams into a fixed
    vector. No model download, no GPU, fully deterministic. It captures
    morphology ("verify"/"verification") and local phrasing, which is enough
    for a working demo and for CI. It is *not* semantic like a neural model.

``SentenceTransformerEmbedder`` (``pip install hprag[ml]``)
    A real neural embedding model for production-quality paraphrase recall.

Both satisfy the same tiny protocol (``embed(list[str]) -> np.ndarray``), so the
rest of the system never needs to know which one is in use.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from hprag.logging_utils import get_logger
from hprag.text import STOPWORDS, stem, tokenize

log = get_logger("retrieval.embeddings")


class Embedder(Protocol):
    """Interface: turn texts into L2-normalised vectors of size ``dim``."""
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""
        ...


def _h(s: str) -> int:
    # md5 rather than Python's hash(): hash() is salted per process, which would
    # make saved indexes useless after a restart.
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "little")


class HashingEmbedder:
    """Feature-hashed word, bigram and character-trigram vectors: offline and deterministic."""
    name = "hashing"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        toks = [stem(t) for t in tokenize(text) if t not in STOPWORDS]
        feats = [(f"w:{t}", 1.0) for t in toks]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(toks, toks[1:])]
        for t in toks:
            padded = f"#{t}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.3) for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        """Hash each feature into a signed bucket, then L2-normalise each row."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = _h(feat)
                # The sign bit halves collision bias (standard hashing-trick practice).
                out[row, h % self.dim] += weight if (h >> 63) & 1 else -weight
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:  # pragma: no cover - needs model download
    """A neural embedding model from sentence-transformers (needs hprag[ml])."""
    def __init__(self, model: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model}"
        self._model = SentenceTransformer(model)
        self.dim = self._model.get_sentence_embedding_dimension()
        log.info("loaded", extra={"model": model, "dim": self.dim})

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode with the neural model, normalised for cosine search."""
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def make_embedder(kind: str, model: str = "") -> Embedder:
    """Build the embedder named in the config."""
    if kind == "hashing":
        return HashingEmbedder()
    if kind in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder(model)
    raise ValueError(f"Unknown embedder: {kind}")
