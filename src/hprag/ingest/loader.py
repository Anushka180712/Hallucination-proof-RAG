"""Load source documents from disk.

Only plain-text formats are supported out of the box (``.txt``, ``.md``,
``.rst``, ``.jsonl``). This is intentional: PDF/HTML extraction is a large,
lossy problem of its own, and silently garbled extraction is a common hidden
cause of "hallucinations" that are really retrieval of broken text. Convert
those formats upstream with a tool you trust, then ingest the text here.
"""

from __future__ import annotations

import json
from pathlib import Path

from hprag.logging_utils import get_logger
from hprag.types import Document

log = get_logger("ingest.loader")
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def load_file(path: str | Path) -> list[Document]:
    """Load one file. ``.jsonl`` gives one document per line; text formats give one document."""
    p = Path(path)
    if p.suffix == ".jsonl":
        # One {"id": ..., "text": ..., ...metadata} object per line.
        docs = []
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.pop("text")
            doc_id = str(row.pop("id", f"{p.stem}-{i}"))
            docs.append(Document(doc_id=doc_id, text=text, metadata={"source": str(p), **row}))
        log.info("loaded.jsonl", extra={"path": str(p), "docs": len(docs)})
        return docs
    if p.suffix.lower() in TEXT_SUFFIXES:
        text = p.read_text(encoding="utf-8")
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.strip()), p.stem)
        log.debug("loaded.file", extra={"path": str(p), "chars": len(text)})
        return [Document(doc_id=p.stem, text=text, metadata={"source": str(p), "title": title})]
    log.warning("skipped.unsupported", extra={"path": str(p), "suffix": p.suffix})
    return []


def load_directory(path: str | Path) -> list[Document]:
    """Load every supported file under a directory (or a single file), in sorted order."""
    root = Path(path)
    if root.is_file():
        return load_file(root)
    docs: list[Document] = []
    for p in sorted(root.rglob("*")):  # sorted => deterministic doc order => reproducible indexes
        if p.is_file():
            docs.extend(load_file(p))
    log.info("loaded.directory", extra={"path": str(root), "docs": len(docs)})
    return docs
