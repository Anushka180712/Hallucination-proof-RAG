"""The end-to-end hallucination-proof RAG pipeline.

    question
       │
       ▼
  1. retrieve      hybrid BM25 + dense, RRF fusion, rerank            (hprag.retrieval)
       │
       ▼
  2. GATE 1        retrieval confidence high enough?  ── no ──► abstain (no LLM call)
       │ yes
       ▼
  3. generate      grounded prompt, cite every sentence               (hprag.generation)
       │
       ▼
  4. verify        split into claims, check each against evidence     (hprag.verification)
       │
       ▼
  5. repair        drop / regenerate unsupported claims
       │
       ▼
  6. GATE 2        faithfulness high enough and no contradictions? ── no ──► abstain
       │ yes
       ▼
     verified answer + citations + per-claim verdicts + trace id

Defence in depth: each gate catches a different failure. Gate 1 stops
questions the corpus cannot answer. The prompt discourages guessing. The
verifier catches what the model invents anyway. Gate 2 guarantees the final
output meets an explicit, configurable faithfulness bar.
"""

from __future__ import annotations

import copy
import re
import time
from dataclasses import asdict

from hprag.confidence import retrieval_confidence
from hprag.config import Settings
from hprag.generation import ABSTAIN_TOKEN, GenerationRefused, Generator, make_generator
from hprag.guardrails import DocumentGuard, GuardEvent, InputGuard, OutputGuard, QueryCache
from hprag.ingest import SentenceChunker, load_directory
from hprag.logging_utils import get_logger, new_trace_id, stage
from hprag.retrieval import HybridRetriever
from hprag.types import Answer, Citation, Claim, Document, ScoredChunk, Verdict
from hprag.verification import Verifier, decompose, make_verifier

log = get_logger("pipeline")

ABSTAIN_MESSAGE = "I don't have enough reliable evidence in the knowledge base to answer that."
BLOCKED_MESSAGE = ("I can't help with that request. I answer questions about the documents in my "
                   "knowledge base, using only what they say.")


class RAGPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: HybridRetriever | None = None,
        generator: Generator | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self.s = settings or Settings()
        self.retriever = retriever or HybridRetriever(self.s.retrieval)
        self.generator = generator or make_generator(self.s.llm)
        self._verifier = verifier
        gr = self.s.guardrails
        self.input_guard = InputGuard(gr)
        self.output_guard = OutputGuard(gr)
        self.document_guard = DocumentGuard(gr)
        self.cache = QueryCache(gr.cache_size if gr.enabled else 0)
        self.quarantined: list[GuardEvent] = []   # passages kept out of the index, for audit/UI
        self.last_quarantined: list[GuardEvent] = []
        log.info("pipeline.ready", extra={"generator": self.generator.name,
                                          "verifier": self.s.verifier.kind,
                                          "reranker": self.s.retrieval.reranker})

    # ------------------------------------------------------------------ ingestion
    @property
    def verifier(self) -> Verifier:
        # Built lazily because the lexical verifier needs IDF stats from the index.
        if self._verifier is None:
            self._verifier = make_verifier(self.s.verifier, self.retriever.idf)
        return self._verifier

    def ingest_documents(self, docs: list[Document]) -> int:
        chunker = SentenceChunker(self.s.chunking.max_chars, self.s.chunking.overlap_sentences)
        docs, self.quarantined = self._screen(docs)
        chunks = chunker.chunk_all(docs)
        self.retriever.build(chunks)
        self._verifier = None
        self.cache.invalidate()
        return len(chunks)

    def _screen(self, docs: list[Document]) -> tuple[list[Document], list[GuardEvent]]:
        """Run the document guardrail (indirect prompt-injection scan) before chunking."""
        if not self.s.guardrails.enabled:
            return docs, []
        return self.document_guard.scan(docs)

    def add_documents(self, docs: list[Document]) -> int:
        """Add documents to the existing index instead of replacing it.

        BM25 statistics (IDF) depend on the whole corpus, so the index is rebuilt
        from old + new chunks rather than patched; at demo/team scale this takes
        milliseconds and keeps scores exactly consistent.
        """
        chunker = SentenceChunker(self.s.chunking.max_chars, self.s.chunking.overlap_sentences)
        new_ids = {d.doc_id for d in docs}
        # Re-adding a document with the same id replaces its old version.
        kept = [c for c in self.retriever.chunks if c.doc_id not in new_ids]
        docs, flagged = self._screen(docs)
        new_chunks = chunker.chunk_all(docs)
        self.quarantined = [e for e in self.quarantined
                            if e.detail.split(":")[0] not in new_ids] + flagged
        self.last_quarantined = flagged
        self.retriever.build(kept + new_chunks)
        self._verifier = None
        self.cache.invalidate()
        log.info("documents.added", extra={"docs": len(docs), "new_chunks": len(new_chunks),
                                           "total_chunks": len(self.retriever.chunks)})
        return len(new_chunks)

    def ingest_path(self, path: str) -> int:
        return self.ingest_documents(load_directory(path))

    # -------------------------------------------------------------------- answering
    def ask(self, question: str) -> Answer:
        """Answer a question, with guardrails around the hallucination-proof core."""
        trace = new_trace_id()
        gr = self.s.guardrails
        if not gr.enabled:
            return self._answer(question, trace)

        # Input guardrails: block attacks and harmful requests, redact personal data.
        decision = self.input_guard.check(question)
        if decision.blocked:
            return self._blocked(decision, trace)
        q = decision.question
        events = list(decision.events)

        # Repeated question? Serve the already-verified answer.
        hit = self.cache.get(q)
        if hit is not None:
            ans = copy.deepcopy(hit)
            ans.trace_id, ans.cached = trace, True
            ans.guardrails = [asdict(e) for e in events] + [asdict(GuardEvent(
                "cache", "cached", "same question (after normalisation) was answered before"))]
            log.info("answer.cached", extra={"question": q})
            return ans

        ans = self._answer(q, trace)

        # Output guardrails: mask personal data / profanity that came from the documents.
        if not ans.abstained:
            ans.answer, out_events = self.output_guard.apply(ans.answer)
            events += out_events
        for c in ans.citations:
            # Sentences removed by the document guardrail leave runs of spaces; tidy them.
            c.quote, _ = self.output_guard.apply(re.sub(r" {2,}", " ", c.quote))
        ans.guardrails = [asdict(e) for e in events]
        self.cache.put(q, ans)
        return ans

    def _blocked(self, d, trace: str) -> Answer:
        ev = d.blocked
        reason = f"blocked by the {ev.guard.replace('_', ' ')} guardrail: {ev.detail}"
        log.warning("answer.blocked", extra={"guard": ev.guard})
        return Answer(
            question=d.question if ev.guard != "prompt_injection" else d.question[:200],
            answer=d.message or BLOCKED_MESSAGE,
            abstained=True,
            abstain_reason=reason,
            confidence=0.0,
            faithfulness=0.0,
            claims=[],
            citations=[],
            trace_id=trace,
            guardrails=[asdict(e) for e in d.events],
            blocked_by=ev.guard,
        )

    def _answer(self, question: str, trace: str) -> Answer:
        """The hallucination-proof core: retrieve, gate, generate, verify, repair, gate."""
        timings: dict[str, float] = {}
        g = self.s.guard
        log.info("question.received", extra={"question": question})

        # 1. Retrieve -------------------------------------------------------------
        t = time.perf_counter()
        hits = self.retriever.retrieve(question)
        timings["retrieve"] = _ms(t)

        # 2. Gate 1: is there evidence at all? -----------------------------------
        report = retrieval_confidence(question, hits, self.retriever.idf, self.retriever.in_vocab)
        if report.confidence < g.min_retrieval_confidence:
            reason = (f"retrieval confidence {report.confidence:.2f} < "
                      f"{g.min_retrieval_confidence:.2f}")
            if report.unknown_terms:
                reason += f"; the knowledge base never mentions: {', '.join(report.unknown_terms)}"
            return self._abstain(question, reason, report.confidence, hits, timings, trace)

        # 3. Generate --------------------------------------------------------------
        sources = [h.chunk.text for h in hits]
        t = time.perf_counter()
        try:
            draft = self.generator.generate(question, hits)
        except GenerationRefused as e:
            return self._abstain(question, f"generator refused: {e}", report.confidence, hits,
                                 timings, trace)
        timings["generate"] = _ms(t)
        if ABSTAIN_TOKEN in draft:
            return self._abstain(question, "generator found the sources insufficient",
                                 report.confidence, hits, timings, trace, draft=draft)

        # 4. Verify ---------------------------------------------------------------
        t = time.perf_counter()
        claims, hedges = self._verify(draft, sources)
        timings["verify"] = _ms(t)

        # 5. Repair ---------------------------------------------------------------
        answer_text = draft
        attempts = 0
        while (g.repair == "regenerate" and attempts < g.max_regenerations
               and any(c.verdict != Verdict.SUPPORTED for c in claims)):
            attempts += 1
            feedback = "\n".join(f"- {c.text} ({c.verdict.value}: {c.reason})"
                                 for c in claims if c.verdict != Verdict.SUPPORTED)
            with stage("repair.regenerate", log, attempt=attempts):
                retry = self.generator.generate(question, hits, feedback=feedback)
            if ABSTAIN_TOKEN in retry:
                break
            new_claims, new_hedges = self._verify(retry, sources)
            if _faith(new_claims) >= _faith(claims):
                answer_text, claims, hedges = retry, new_claims, new_hedges

        contradicted = [c for c in claims if c.verdict == Verdict.CONTRADICTED]
        faith_before = _faith(claims)
        kept = claims
        if g.repair == "drop":
            kept = [c for c in claims if c.verdict == Verdict.SUPPORTED]
            if len(kept) < len(claims):
                log.warning("repair.dropped", extra={
                    "dropped": [c.text for c in claims if c.verdict != Verdict.SUPPORTED]})
            answer_text = _render(kept, hedges)

        # 6. Gate 2: final faithfulness bar ----------------------------------------
        if contradicted and g.contradiction_aborts:
            return self._abstain(question, f"{len(contradicted)} claim(s) contradicted by the sources",
                                 report.confidence, hits, timings, trace, draft=draft, claims=claims)
        if not kept:
            return self._abstain(question, "no claim in the draft could be verified",
                                 report.confidence, hits, timings, trace, draft=draft, claims=claims)
        # With "drop", the *draft* must still clear the bar: if most of what the model
        # wrote was unsupported, the few surviving sentences are not trustworthy either.
        if faith_before < g.min_faithfulness:
            return self._abstain(question, f"faithfulness {faith_before:.2f} < {g.min_faithfulness:.2f}",
                                 report.confidence, hits, timings, trace, draft=draft, claims=claims)

        result = Answer(
            question=question,
            answer=answer_text,
            abstained=False,
            abstain_reason=None,
            confidence=report.confidence,
            faithfulness=round(_faith(kept), 4),
            claims=claims,
            citations=_citations(kept, hits),
            trace_id=trace,
            draft=draft,
            timings_ms=timings,
        )
        log.info("answer.returned", extra={"faithfulness": result.faithfulness,
                                           "claims": len(claims), "kept": len(kept),
                                           "confidence": report.confidence, **timings})
        return result

    # ------------------------------------------------------------------ helpers
    def _verify(self, text: str, sources: list[str]) -> tuple[list[Claim], list[str]]:
        with stage("verify", log) as out:
            claims, hedges = decompose(text)
            self.verifier.verify(claims, sources, self.s.guard.min_claim_support)
            out.update(claims=len(claims), hedges=len(hedges),
                       supported=sum(c.verdict == Verdict.SUPPORTED for c in claims),
                       contradicted=sum(c.verdict == Verdict.CONTRADICTED for c in claims))
        return claims, hedges

    def _abstain(self, question: str, reason: str, conf: float, hits: list[ScoredChunk],
                 timings: dict[str, float], trace: str, draft: str | None = None,
                 claims: list[Claim] | None = None) -> Answer:
        log.warning("answer.abstained", extra={"reason": reason, "confidence": conf})
        return Answer(
            question=question,
            answer=ABSTAIN_MESSAGE,
            abstained=True,
            abstain_reason=reason,
            confidence=conf,
            faithfulness=_faith(claims or []),
            claims=claims or [],
            # Still show the closest passages: useful to a human even when we won't answer.
            citations=[Citation(i, h.chunk.chunk_id, h.chunk.doc_id, h.chunk.text[:240],
                                h.chunk.start, h.chunk.end) for i, h in enumerate(hits[:3], 1)],
            trace_id=trace,
            draft=draft,
            timings_ms=timings,
        )


def _faith(claims: list[Claim]) -> float:
    if not claims:
        return 0.0
    return sum(c.verdict == Verdict.SUPPORTED for c in claims) / len(claims)


def _render(claims: list[Claim], hedges: list[str]) -> str:
    """Rebuild the answer from verified claims, citing the sources that actually support them."""
    parts = []
    for c in claims:
        refs = c.evidence or c.cited
        text = c.text.rstrip(".")
        parts.append(f"{text} [{', '.join(map(str, refs))}]." if refs else f"{text}.")
    parts += [h if h.endswith(".") else h + "." for h in hedges]
    return " ".join(parts)


def _citations(claims: list[Claim], hits: list[ScoredChunk]) -> list[Citation]:
    used = sorted({n for c in claims for n in (c.evidence or c.cited) if 1 <= n <= len(hits)})
    return [Citation(n, hits[n - 1].chunk.chunk_id, hits[n - 1].chunk.doc_id,
                     hits[n - 1].chunk.text, hits[n - 1].chunk.start, hits[n - 1].chunk.end)
            for n in used]


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
