"""Run the web UI's backend inside the browser, with no server.

Why this exists
    GitHub Pages can only serve static files, so it can't run ``hprag ui``. Instead,
    the hosted demo loads Python itself into the page with Pyodide (CPython compiled
    to WebAssembly), installs this package, and answers questions locally. Nothing
    leaves the visitor's machine and there is no server to pay for or secure.

How it is wired
    The UI calls ``api(path, body)``. On GitHub Pages, ``web/bridge.js`` routes that
    call here, to :meth:`BrowserApp.handle`, instead of to an HTTP server. It takes
    the same paths and returns the same JSON as the FastAPI server, because both use
    :mod:`hprag.api.views`.

The pipeline is the default offline one (BM25 + hashing embedder, extractive
generator, lexical verifier), so it needs only numpy, which Pyodide ships.
"""

from __future__ import annotations

import json
from typing import Any

from hprag.api import views
from hprag.config import Settings
from hprag.logging_utils import configure_logging, get_logger
from hprag.pipeline import RAGPipeline

log = get_logger("browser")


class BrowserApp:
    """The server's endpoints as one method, for a page that has no server."""

    def __init__(self, corpus_dir: str, settings: Settings | None = None) -> None:
        s = settings or Settings()
        # One visitor, one tab: a rate limit protects nobody here, so the UI shows it as off.
        s.guardrails.rate_limit_per_minute = 0
        # Logs go to the browser's developer console (Pyodide maps stderr to console),
        # so a curious visitor can watch every step, trace id included.
        configure_logging(s.log_level, s.log_format)
        self.rag = RAGPipeline(s)
        n = self.rag.ingest_path(corpus_dir)
        log.info("browser.ready", extra={"corpus": corpus_dir, "chunks": n})

    def handle(self, path: str, body: str | None = None) -> str:
        """Answer one UI request. ``body`` and the result are JSON strings (easy to pass to JS)."""
        data: dict[str, Any] = json.loads(body) if body else {}
        if path == "/health":
            out = views.health(self.rag)
        elif path == "/examples":
            out = views.examples()
        elif path == "/guardrails":
            out = views.guardrails(self.rag)
        elif path == "/ask":
            question = str(data.get("question", ""))[:2000]  # same cap as the server's request model
            out = self.rag.ask(question).to_dict()
        elif path == "/documents":
            out = views.add_documents(self.rag, views.to_documents(data["documents"]))
        else:
            raise ValueError(f"unknown path {path!r}")
        return json.dumps(out)
