"""Guardrails: checks that run around the RAG pipeline.

Where they sit::

    question ─► InputGuard ─► (cache) ─► retrieve ─► gates/generate/verify ─► OutputGuard ─► answer
                  │ block: injection, jailbreak, harmful request, invalid input
                  │ redact: personal information
    documents ─► DocumentGuard (at ingest) ─► remove sentences that try to instruct the AI
    API      ─► RateLimiter ─► 429 when one client floods the server

Hallucination checks ask "is this answer true to the sources?". Guardrails ask
different questions: "should this request be processed at all?", "is anything
in it private?", "is a document trying to hijack the model?", "is anything in
the output unfit to show?". Both kinds are needed for a system people can trust.

Each decision is a :class:`GuardEvent` with a plain-language explanation. The
events travel with the answer to the UI and are logged with the trace id.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from hprag.config import GuardrailsConfig
from hprag.guardrails import rules
from hprag.guardrails.pii import redact, strip_redacted
from hprag.logging_utils import get_logger
from hprag.text import query_terms
from hprag.types import Document

log = get_logger("guardrails")

EXPLANATIONS = {
    "prompt_injection": "Protects against prompt injection: text that tries to override the system's rules, "
                        "change its role, or make it reveal its instructions.",
    "unsafe_request": "Protects against requests for seriously harmful content, such as weapons or cyberattacks.",
    "self_harm": "Responds with support instead of a document answer when a message suggests "
                 "someone may be at risk.",
    "invalid_input": "Protects the system from empty, oversized or meaningless input.",
    "pii_input": "Protects privacy: personal details in your question are removed before it is searched, "
                 "logged or sent to any AI model.",
    "pii_output": "Protects privacy: personal details that appear in source documents are masked in the answer.",
    "profanity_output": "Keeps offensive language from source documents out of the displayed answer.",
    "document_injection": "Protects against indirect prompt injection: passages inside documents that try to "
                          "give the AI instructions are kept out of the knowledge base.",
    "cache": "Repeated questions are answered from a cache of already-verified answers, which is faster and "
             "cheaper. The cache is cleared whenever documents change.",
}


@dataclass
class GuardEvent:
    guard: str          # e.g. "prompt_injection"
    action: str         # "blocked" | "redacted" | "flagged" | "quarantined" | "cached"
    detail: str         # what exactly fired, e.g. the rule name and matched text
    explanation: str = ""

    def __post_init__(self) -> None:
        self.explanation = self.explanation or EXPLANATIONS.get(self.guard, "")


@dataclass
class InputDecision:
    question: str                      # possibly redacted
    events: list[GuardEvent] = field(default_factory=list)
    blocked: GuardEvent | None = None
    message: str | None = None         # custom reply when blocked (e.g. self-harm support)


class InputGuard:
    def __init__(self, cfg: GuardrailsConfig) -> None:
        self.cfg = cfg

    def check(self, question: str) -> InputDecision:
        q = question.strip()
        d = InputDecision(question=q)

        # 1. Basic validation: cheapest check first.
        if not q or not re.search(r"[A-Za-z]{2,}", q):
            return self._block(d, "invalid_input", "the question has no words in it")
        if len(q) > self.cfg.max_question_chars:
            return self._block(d, "invalid_input",
                               f"the question is {len(q)} characters; the limit is {self.cfg.max_question_chars}")

        # 2. Self-harm: answered with care, never with a refusal lecture.
        if self.cfg.block_unsafe and rules.SELF_HARM.search(q):
            return self._block(d, "self_harm", "the message suggests the writer may be at risk",
                               message=rules.SELF_HARM_MESSAGE)

        # 3. Prompt injection / jailbreak attempts.
        if self.cfg.block_injection:
            hit = rules.first_match(q, rules.INJECTION_PATTERNS)
            if hit:
                return self._block(d, "prompt_injection", f"rule '{hit[0]}' matched: \"{hit[1]}\"")

        # 4. Harmful requests.
        if self.cfg.block_unsafe:
            hit = rules.first_match(q, rules.UNSAFE_PATTERNS)
            if hit:
                return self._block(d, "unsafe_request", f"category '{hit[0]}' matched: \"{hit[1]}\"")

        # 5. Personal information: redact, don't block.
        if self.cfg.redact_pii:
            clean, kinds = redact(q)
            if kinds:
                d.question = strip_redacted(clean) or clean
                d.events.append(GuardEvent("pii_input", "redacted",
                                           f"removed {', '.join(sorted(set(kinds)))} from the question"))
                log.warning("guard.pii_input", extra={"kinds": kinds})
        return d

    def _block(self, d: InputDecision, guard: str, detail: str, message: str | None = None) -> InputDecision:
        ev = GuardEvent(guard, "blocked", detail)
        d.events.append(ev)
        d.blocked = ev
        d.message = message
        # The raw question is deliberately *not* logged here: it may be an attack payload
        # or contain sensitive text. The rule name and match are enough to audit the decision.
        log.warning("guard.blocked", extra={"guard": guard, "detail": detail})
        return d


class DocumentGuard:
    """Scan documents at ingest time for planted instructions (indirect prompt injection).

    Only the offending *sentences* are removed, not the whole document: a supplier
    note with one malicious line still has useful facts in it. Removed sentences
    are overwritten with spaces of the same length, so every character offset
    (and therefore every citation) still points at the right place in the file.
    """

    _SENT = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")

    def __init__(self, cfg: GuardrailsConfig) -> None:
        self.cfg = cfg

    def scan(self, docs: list[Document]) -> tuple[list[Document], list[GuardEvent]]:
        if not self.cfg.scan_documents:
            return docs, []
        out, events = [], []
        for doc in docs:
            text = doc.text
            for m in self._SENT.finditer(doc.text):
                hit = rules.first_match(m.group(0), rules.DOCUMENT_INJECTION_PATTERNS)
                if not hit:
                    continue
                text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]
                sentence = " ".join(m.group(0).split())
                events.append(GuardEvent("document_injection", "quarantined",
                                         f"{doc.doc_id}: rule '{hit[0]}' removed \"{sentence[:120]}\""))
                log.warning("guard.document_quarantined",
                            extra={"doc_id": doc.doc_id, "rule": hit[0], "sentence": sentence[:120]})
            out.append(Document(doc.doc_id, text, dict(doc.metadata)) if text != doc.text else doc)
        return out, events


class OutputGuard:
    def __init__(self, cfg: GuardrailsConfig) -> None:
        self.cfg = cfg

    def apply(self, text: str) -> tuple[str, list[GuardEvent]]:
        events = []
        if self.cfg.redact_pii:
            text, kinds = redact(text)
            if kinds:
                events.append(GuardEvent("pii_output", "redacted",
                                         f"masked {', '.join(sorted(set(kinds)))} in the answer"))
        if self.cfg.mask_profanity:
            masked, n = rules.PROFANITY.subn(lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 1), text)
            if n:
                text = masked
                events.append(GuardEvent("profanity_output", "redacted", f"masked {n} offensive word(s)"))
        for e in events:
            log.warning("guard.output", extra={"guard": e.guard, "detail": e.detail})
        return text, events


class QueryCache:
    """LRU cache of verified answers, keyed by the question's normalised content words.

    "How many crew are there?" and "how many crew are there" share a key, as do
    word-order variants, because the key is the sorted set of stemmed content
    terms. Keys include an index version, so adding documents invalidates
    everything at once and a stale answer can never be served.
    """

    def __init__(self, size: int = 256) -> None:
        self.size = size
        self._data: OrderedDict[tuple, object] = OrderedDict()
        self._lock = threading.Lock()
        self.version = 0

    def key(self, question: str) -> tuple | None:
        terms = tuple(sorted(set(query_terms(question))))
        return (self.version, terms) if terms else None

    def get(self, question: str):
        k = self.key(question)
        if k is None or self.size <= 0:
            return None
        with self._lock:
            if k in self._data:
                self._data.move_to_end(k)
                return self._data[k]
        return None

    def put(self, question: str, value) -> None:
        k = self.key(question)
        if k is None or self.size <= 0:
            return
        with self._lock:
            self._data[k] = value
            self._data.move_to_end(k)
            while len(self._data) > self.size:
                self._data.popitem(last=False)

    def invalidate(self) -> None:
        with self._lock:
            self._data.clear()
            self.version += 1


class RateLimiter:
    """Token bucket per client: ``rate`` requests per minute, with bursts up to ``rate``.

    Protects the server (and any paid LLM API behind it) from one client sending
    a flood of requests, accidentally or on purpose.
    """

    def __init__(self, per_minute: int) -> None:
        self.capacity = float(per_minute)
        self.refill = per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str) -> tuple[bool, float]:
        """Return (allowed, seconds until the next request would be allowed)."""
        if self.capacity <= 0:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(client, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens >= 1:
                self._buckets[client] = (tokens - 1, now)
                return True, 0.0
            self._buckets[client] = (tokens, now)
            return False, round((1 - tokens) / self.refill, 1)
