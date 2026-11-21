"""REST API (FastAPI).

Endpoints
---------
``GET  /health``   liveness + index size
``POST /ask``      ``{"question": "..."}`` -> verified answer with claims and citations
``POST /ingest``   ``{"documents": [{"id": "...", "text": "..."}]}`` -> rebuild the index

Every response carries an ``X-Trace-Id`` header that matches the ``trace_id``
in the logs, so a user's bug report can be tied to the exact log lines.
"""

from __future__ import annotations

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hprag.logging_utils import get_logger
from hprag.pipeline import RAGPipeline
from hprag.types import Document

log = get_logger("api")


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
        version="1.0.0",
        description="Grounded answers with citations, claim-level verification and abstention.",
    )
    # Re-indexing swaps internal state; a lock keeps /ask from reading a half-built index.
    lock = threading.RLock()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(rag.retriever.chunks),
                "generator": rag.generator.name}

    @app.post("/ask")
    def ask(req: AskRequest):
        from fastapi.responses import JSONResponse

        if not rag.retriever.chunks:
            raise HTTPException(409, "Index is empty: POST /ingest first")
        with lock:
            ans = rag.ask(req.question)
        return JSONResponse(ans.to_dict(), headers={"X-Trace-Id": ans.trace_id})

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        docs = [Document(doc_id=d.id, text=d.text, metadata={"title": d.id, **d.metadata})
                for d in req.documents]
        with lock:
            n = rag.ingest_documents(docs)
        log.info("api.ingested", extra={"docs": len(docs), "chunks": n})
        return {"documents": len(docs), "chunks": n}

    return app
