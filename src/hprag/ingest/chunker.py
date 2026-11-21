"""Sentence-aware chunking with overlap and exact character offsets.

Why not fixed-size token windows?
    A window that cuts a sentence in half produces a chunk that contains half a
    fact. The generator then sees "revenue grew 12%" without the "in 2023" that
    followed it, and a confident-but-wrong answer is born. Packing *whole
    sentences* into chunks avoids that failure mode.

Why keep offsets?
    Each chunk remembers ``start``/``end`` in the source document so a citation
    can point at the exact characters it relies on. That is what makes the
    answer auditable by a human.
"""

from __future__ import annotations

import re

from hprag.logging_utils import get_logger
from hprag.text import ends_with_abbreviation
from hprag.types import Chunk, Document

log = get_logger("ingest.chunker")
_SENT_SPAN = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")


class SentenceChunker:
    """Packs whole sentences into chunks of at most ``max_chars``, overlapping by N sentences."""
    def __init__(self, max_chars: int = 600, overlap_sentences: int = 1) -> None:
        if max_chars < 50:
            raise ValueError("max_chars must be >= 50")
        self.max_chars = max_chars
        self.overlap = max(0, overlap_sentences)

    def _sentences(self, text: str) -> list[tuple[int, int]]:
        spans = []
        for m in _SENT_SPAN.finditer(text):
            s, e = m.start(), m.end()
            while s < e and text[s].isspace():
                s += 1
            while e > s and text[e - 1].isspace():
                e -= 1
            if e > s:
                # "Dr. Okafor" must not become two sentences: glue spans that end in
                # a known abbreviation onto the next one.
                if spans and ends_with_abbreviation(text[spans[-1][0]:spans[-1][1]]):
                    spans[-1] = (spans[-1][0], e)
                else:
                    spans.append((s, e))
        return spans

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split one document into chunks with exact character offsets."""
        sents = self._sentences(doc.text)
        chunks: list[Chunk] = []
        i = 0
        while i < len(sents):
            j = i
            # Greedily add whole sentences until the next one would overflow.
            # A single sentence longer than max_chars still becomes its own chunk
            # rather than being cut (never split a fact).
            while j + 1 < len(sents) and sents[j + 1][1] - sents[i][0] <= self.max_chars:
                j += 1
            start, end = sents[i][0], sents[j][1]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks)}",
                    doc_id=doc.doc_id,
                    text=doc.text[start:end],
                    start=start,
                    end=end,
                    metadata=dict(doc.metadata),
                )
            )
            if j + 1 >= len(sents):
                break
            # Step forward, re-using the last `overlap` sentences as context, but
            # always make progress to avoid an infinite loop.
            i = max(i + 1, j + 1 - self.overlap)
        log.debug("chunked", extra={"doc_id": doc.doc_id, "sentences": len(sents), "chunks": len(chunks)})
        return chunks

    def chunk_all(self, docs: list[Document]) -> list[Chunk]:
        """Chunk every document."""
        out = [c for d in docs for c in self.chunk(d)]
        log.info("chunked.all", extra={"docs": len(docs), "chunks": len(out)})
        return out
