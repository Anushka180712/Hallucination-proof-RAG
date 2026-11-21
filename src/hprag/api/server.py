"""REST API (FastAPI).

Endpoints
---------
``GET  /``           the web UI (a single static HTML file, no build step)
``GET  /examples``   sample questions for the bundled corpus, grouped
``GET  /health``     liveness + index size
``GET  /guardrails`` which guardrails are active, what each protects against, quarantined passages
``POST /ask``        ``{"question": "..."}`` -> verified answer with claims and citations
``POST /documents``  ``{"documents": [...]}`` -> add to the index (keeps existing documents)
``POST /ingest``     ``{"documents": [{"id": "...", "text": "..."}]}`` -> replace the index

Every response carries an ``X-Trace-Id`` header that matches the ``trace_id``
in the logs, so a user's bug report can be tied to the exact log lines.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from hprag.guardrails import EXPLANATIONS, RateLimiter
from hprag.logging_utils import get_logger
from hprag.pipeline import RAGPipeline
from hprag.types import Document

log = get_logger("api")
STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    """Body of POST /ask. The length cap is a cheap first defence against oversized payloads."""
    question: str = Field(..., min_length=1, max_length=2000)


class DocIn(BaseModel):
    """One document to index: a unique id (also used as its title) and its text."""
    id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Body of POST /ingest and POST /documents: one or more documents."""
    documents: list[DocIn] = Field(..., min_length=1)


def create_app(rag: RAGPipeline) -> FastAPI:
    """Build the FastAPI app around an existing pipeline.

    The pipeline is passed in (not created here) so the CLI, tests and Docker
    image can each configure it their own way and share one app factory.
    """
    app = FastAPI(
        title="Hallucination-Proof RAG",
        version="1.2.0",
        description="Grounded answers with citations, claim-level verification and abstention.",
    )
    # Re-indexing swaps internal state; a lock keeps /ask from reading a half-built index.
    lock = threading.RLock()
    gr = rag.s.guardrails
    limiter = RateLimiter(gr.rate_limit_per_minute if gr.enabled else 0)

    def check_rate(request: Request) -> None:
        """Apply the per-client rate limit; raise HTTP 429 with Retry-After when exceeded."""
        client = request.client.host if request.client else "unknown"
        ok, wait = limiter.allow(client)
        if not ok:
            log.warning("guard.rate_limited", extra={"client": client, "retry_after_s": wait})
            msg = (f"Too many requests: the limit is {gr.rate_limit_per_minute} per minute. "
                   f"Try again in {wait:.0f} s.")
            raise HTTPException(429, msg, headers={"Retry-After": str(int(wait) + 1)})

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        """Serve the single-file web UI."""
        return FileResponse(STATIC / "index.html")

    @app.get("/examples")
    def examples() -> dict:
        """Return the grouped sample questions shown in the UI (and checked by a test)."""
        return json.loads((STATIC / "examples.json").read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict:
        """Liveness probe plus a summary of what the server is running with."""
        return {"status": "ok", "chunks": len(rag.retriever.chunks),
                "generator": rag.generator.name, "verifier": rag.verifier.name,
                "guardrails": gr.enabled, "quarantined": len(rag.quarantined)}

    @app.get("/guardrails")
    def guardrails() -> dict:
        """List every guardrail, whether it is on, and what it protects against."""
        active = {
            "prompt_injection": gr.block_injection, "unsafe_request": gr.block_unsafe,
            "self_harm": gr.block_unsafe, "invalid_input": True, "pii_input": gr.redact_pii,
            "pii_output": gr.redact_pii, "profanity_output": gr.mask_profanity,
            "document_injection": gr.scan_documents, "cache": gr.cache_size > 0,
        }
        items = [{"guard": k, "active": gr.enabled and v, "protects_against": EXPLANATIONS[k]}
                 for k, v in active.items()]
        items.append({"guard": "rate_limit", "active": gr.enabled and gr.rate_limit_per_minute > 0,
                      "protects_against": f"Stops one client from flooding the server: at most "
                                          f"{gr.rate_limit_per_minute} requests per minute."})
        return {"enabled": gr.enabled, "guardrails": items,
                "quarantined": [e.detail for e in rag.quarantined]}

    @app.post("/ask")
    def ask(req: AskRequest, request: Request):
        """Answer one question through guardrails and the hallucination-proof core."""
        check_rate(request)
        if not rag.retriever.chunks:
            raise HTTPException(409, "Index is empty: POST /ingest first")
        with lock:
            ans = rag.ask(req.question)
        return JSONResponse(ans.to_dict(), headers={"X-Trace-Id": ans.trace_id})

    @app.post("/documents")
    def add_documents(req: IngestRequest, request: Request) -> dict:
        """Add documents to the existing index (same id replaces the old version)."""
        check_rate(request)
        docs = [Document(doc_id=d.id, text=d.text, metadata={"title": d.id, **d.metadata})
                for d in req.documents]
        with lock:
            n = rag.add_documents(docs)
        return {"documents": len(docs), "chunks": n, "total_chunks": len(rag.retriever.chunks),
                "quarantined": [e.detail for e in rag.last_quarantined]}

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        """Replace the whole index with the given documents."""
        docs = [Document(doc_id=d.id, text=d.text, metadata={"title": d.id, **d.metadata})
                for d in req.documents]
        with lock:
            n = rag.ingest_documents(docs)
        log.info("api.ingested", extra={"docs": len(docs), "chunks": n})
        return {"documents": len(docs), "chunks": n}

    return app
