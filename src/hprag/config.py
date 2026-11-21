"""All tunable settings in one typed place.

Every threshold that decides "answer or abstain" lives here with a comment
explaining what it does, so the safety behaviour of the system is reviewable
without reading the pipeline code.

Settings are resolved in this order (later wins):
    defaults below  ->  TOML file (``--config``)  ->  ``HPRAG_*`` environment variables
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 uses the identical backport
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class ChunkingConfig:
    max_chars: int = 600       # small enough that one chunk ~ one topic, so citations stay precise
    overlap_sentences: int = 1  # carry one sentence across boundaries so facts aren't cut in half


@dataclass
class RetrievalConfig:
    top_k: int = 5              # chunks handed to the generator (more = more noise to hallucinate from)
    candidate_k: int = 20       # chunks fetched from each retriever before fusion/reranking
    rrf_k: int = 60             # Reciprocal Rank Fusion constant (60 is the value from the original paper)
    embedder: str = "hashing"   # "hashing" (offline, zero deps) | "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker: str = "lexical"   # "lexical" (offline) | "cross-encoder" | "none"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class GuardConfig:
    # --- Gate 1: before generation -------------------------------------------------
    min_retrieval_confidence: float = 0.35   # below this, abstain without calling the LLM at all
    # --- Gate 2: after verification ------------------------------------------------
    min_claim_support: float = 0.55          # a claim needs at least this support score to count as supported
    min_faithfulness: float = 0.6            # share of claims that must be supported to return an answer
    contradiction_aborts: bool = True        # any contradicted claim => answer is never returned as-is
    repair: str = "drop"                     # what to do with unsupported claims: "drop" | "regenerate" | "none"
    max_regenerations: int = 1


@dataclass
class LLMConfig:
    provider: str = "extractive"             # "extractive" (offline, cannot hallucinate) | "anthropic"
    model: str = "claude-opus-5-5"           # latest Claude Opus; override with HPRAG_LLM_MODEL
    max_tokens: int = 2048
    effort: str = "medium"                   # Claude effort level: low | medium | high | xhigh | max


@dataclass
class VerifierConfig:
    kind: str = "lexical"                    # "lexical" (offline) | "nli" | "llm" | "ensemble"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    judge_model: str = "claude-opus-5-5"


@dataclass
class GuardrailsConfig:
    enabled: bool = True
    block_injection: bool = True        # refuse prompt-injection / jailbreak attempts in questions
    block_unsafe: bool = True           # refuse clearly harmful requests (weapons, cyberattacks, violence)
    redact_pii: bool = True             # strip emails, phones, cards... from questions and answers
    mask_profanity: bool = True         # mask offensive words that appear in answers
    scan_documents: bool = True         # quarantine document passages that try to instruct the AI
    max_question_chars: int = 500       # long inputs are a common vehicle for injection payloads
    cache_size: int = 256               # verified answers kept for repeated questions (0 = off)
    rate_limit_per_minute: int = 60     # API requests per client per minute (0 = off)


@dataclass
class Settings:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    guardrails: GuardrailsConfig = field(default_factory=GuardrailsConfig)
    log_level: str = "INFO"
    log_format: str = "pretty"               # "pretty" | "json"

    @classmethod
    def load(cls, path: str | Path | None = None, env: dict[str, str] | None = None) -> Settings:
        s = cls()
        if path:
            with open(path, "rb") as f:
                _apply(s, tomllib.load(f))
        _apply_env(s, os.environ if env is None else env)
        return s


def _coerce(value: Any, current: Any) -> Any:
    if isinstance(current, bool):
        return value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def _apply(obj: Any, data: dict[str, Any]) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise ValueError(f"Unknown config key: {key!r}")
        current = getattr(obj, key)
        if isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(obj, key, _coerce(value, current))


def _apply_env(s: Settings, env: dict[str, str] | os._Environ) -> None:
    """``HPRAG_GUARD_MIN_FAITHFULNESS=0.9`` -> ``settings.guard.min_faithfulness = 0.9``."""
    for key, value in env.items():
        if not key.startswith("HPRAG_"):
            continue
        parts = key[6:].lower().split("_", 1)
        if len(parts) == 2 and hasattr(s, parts[0]) and hasattr(getattr(s, parts[0]), parts[1]):
            section = getattr(s, parts[0])
            setattr(section, parts[1], _coerce(value, getattr(section, parts[1])))
        elif hasattr(s, key[6:].lower()):
            attr = key[6:].lower()
            setattr(s, attr, _coerce(value, getattr(s, attr)))
