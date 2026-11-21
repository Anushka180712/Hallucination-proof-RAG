"""The JSON the web UI needs, as plain functions with no web framework.

Why a separate module?
    The same UI runs in two places: served by FastAPI (``hprag ui``), and fully
    inside the browser on GitHub Pages, where Python runs on WebAssembly (Pyodide)
    and there is no server at all (see :mod:`hprag.browser`). Both call these
    functions, so the two versions can never drift apart in what they show.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hprag.guardrails import EXPLANATIONS
from hprag.logging_utils import get_logger
from hprag.pipeline import RAGPipeline
from hprag.types import Document

log = get_logger("api")
STATIC = Path(__file__).parent / "static"


def examples() -> dict[str, Any]:
    """The grouped sample questions shown in the UI (a test checks each behaves as labelled)."""
    return json.loads((STATIC / "examples.json").read_text(encoding="utf-8"))


def health(rag: RAGPipeline) -> dict[str, Any]:
    """Liveness plus a summary of what the pipeline is running with."""
    return {"status": "ok", "chunks": len(rag.retriever.chunks),
            "generator": rag.generator.name, "verifier": rag.verifier.name,
            "guardrails": rag.s.guardrails.enabled, "quarantined": len(rag.quarantined)}


def guardrails(rag: RAGPipeline) -> dict[str, Any]:
    """Every guardrail, whether it is on, and what it protects against."""
    gr = rag.s.guardrails
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


def to_documents(raw: list[dict[str, Any]]) -> list[Document]:
    """Turn ``[{"id", "text", "metadata"?}]`` into Documents; the id doubles as the title."""
    return [Document(doc_id=d["id"], text=d["text"], metadata={"title": d["id"], **d.get("metadata", {})})
            for d in raw]


def add_documents(rag: RAGPipeline, docs: list[Document]) -> dict[str, Any]:
    """Add documents to the index (same id replaces the old version) and report what was quarantined."""
    n = rag.add_documents(docs)
    return {"documents": len(docs), "chunks": n, "total_chunks": len(rag.retriever.chunks),
            "quarantined": [e.detail for e in rag.last_quarantined]}
