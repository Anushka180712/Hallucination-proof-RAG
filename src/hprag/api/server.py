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
    question: str = Field(..., min_length=1, max_length=2000)


class DocIn(BaseModel):
    id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[DocIn] = Field(..., min_length=1)


def create_app(rag: RAGPipeline) -> FastAPI:
    app = FastAPI(
        title="Hallucination-Proof RAG",
        version="1.1.0",
        description="Grounded answers with citations, claim-level verification and abstention.",
    )
    # Re-indexing swaps internal state; a lock keeps /ask from reading a half-built index.
    lock = threading.RLock()
    gr = rag.s.guardrails
    limiter = RateLimiter(gr.rate_limit_per_minute if gr.enabled else 0)

    def check_rate(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        ok, wait = limiter.allow(client)
        if not ok:
            log.warning("guard.rate_limited", extra={"client": client, "retry_after_s": wait})
            msg = (f"Too many requests: the limit is {gr.rate_limit_per_minute} per minute. "
                   f"Try again in {wait:.0f} s.")
            raise HTTPException(429, msg, headers={"Retry-After": str(int(wait) + 1)})

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/examples")
    def examples() -> dict:
        return json.loads((STATIC / "examples.json").read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(rag.retriever.chunks),
                "generator": rag.generator.name, "verifier": rag.verifier.name,
                "guardrails": gr.enabled, "quarantined": len(rag.quarantined)}

    @app.get("/guardrails")
    def guardrails() -> dict:
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
        check_rate(request)
        if not rag.retriever.chunks:
            raise HTTPException(409, "Index is empty: POST /ingest first")
        with lock:
            ans = rag.ask(req.question)
        return JSONResponse(ans.to_dict(), headers={"X-Trace-Id": ans.trace_id})

    @app.post("/documents")
    def add_documents(req: IngestRequest, request: Request) -> dict:
        check_rate(request)
        docs = [Document(doc_id=d.id, text=d.text, metadata={"title": d.id, **d.metadata})
                for d in req.documents]
        with lock:
            n = rag.add_documents(docs)
        return {"documents": len(docs), "chunks": n, "total_chunks": len(rag.retriever.chunks),
                "quarantined": [e.detail for e in rag.last_quarantined]}

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        docs = [Document(doc_id=d.id, text=d.text, metadata={"title": d.id, **d.metadata})
                for d in req.documents]
        with lock:
            n = rag.ingest_documents(docs)
        log.info("api.ingested", extra={"docs": len(docs), "chunks": n})
        return {"documents": len(docs), "chunks": n}

    return app
