"""hprag — Hallucination-Proof Retrieval-Augmented Generation.

Quick start::

    from hprag import RAGPipeline
    rag = RAGPipeline()
    rag.ingest_path("data/corpus")
    print(rag.ask("When was the Halcyon station commissioned?").answer)
"""

import logging as _logging

from hprag.config import Settings
from hprag.logging_utils import configure_logging
from hprag.pipeline import RAGPipeline
from hprag.types import Answer, Chunk, Claim, Document, Verdict

__version__ = "1.2.0"
__all__ = ["Answer", "Chunk", "Claim", "Document", "RAGPipeline", "Settings", "Verdict",
           "configure_logging", "__version__"]

# Library etiquette: stay silent unless the application configures logging.
_logging.getLogger("hprag").addHandler(_logging.NullHandler())
