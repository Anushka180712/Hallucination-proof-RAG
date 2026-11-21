"""Core data types shared by every stage.

Why plain dataclasses?
    They are fast, have no dependency, are trivially serialisable with
    :func:`dataclasses.asdict`, and make the data flow obvious when reading the
    code: ``Document -> Chunk -> ScoredChunk -> Claim -> Answer``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


@dataclass
class Document:
    """A source file before chunking."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable span of a document.

    ``start``/``end`` are character offsets into the original document, so a
    citation can always be traced back to the exact text it came from.
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """A chunk plus every score it collected on its way through retrieval."""

    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)  # e.g. {"bm25": .., "dense": .., "rerank": ..}


class Verdict(str, Enum):
    """Outcome of checking one claim against the evidence."""

    SUPPORTED = "supported"          # evidence entails the claim
    CONTRADICTED = "contradicted"    # evidence says the opposite (worst case)
    UNSUPPORTED = "unsupported"      # evidence is silent; claim may be invented


@dataclass
class Claim:
    """One atomic statement from a generated answer."""

    text: str
    cited: list[int] = field(default_factory=list)       # citation numbers written by the generator, e.g. [1, 3]
    verdict: Verdict = Verdict.UNSUPPORTED
    score: float = 0.0                                    # support score in [0, 1]
    evidence: list[int] = field(default_factory=list)    # sources that actually support it (1-based)
    citation_valid: bool = False                          # does at least one cited source support it?
    reason: str = ""


@dataclass
class Citation:
    number: int
    chunk_id: str
    doc_id: str
    quote: str
    start: int
    end: int


@dataclass
class Answer:
    """The final, verified response returned to the caller."""

    question: str
    answer: str
    abstained: bool
    abstain_reason: str | None
    confidence: float                       # retrieval confidence in [0, 1]
    faithfulness: float                     # share of claims supported by evidence, in [0, 1]
    claims: list[Claim]
    citations: list[Citation]
    trace_id: str = "-"
    draft: str | None = None                # the unverified draft, kept for transparency
    timings_ms: dict[str, float] = field(default_factory=dict)
    guardrails: list[dict[str, str]] = field(default_factory=list)  # guardrail decisions, for the UI
    blocked_by: str | None = None           # guardrail that refused the request, if any
    cached: bool = False                    # served from the verified-answer cache

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for c in d["claims"]:
            c["verdict"] = c["verdict"].value if isinstance(c["verdict"], Verdict) else c["verdict"]
        return d
