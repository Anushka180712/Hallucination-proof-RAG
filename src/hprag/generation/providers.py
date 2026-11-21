"""Pluggable answer generators.

Every provider implements one method::

    generate(question, hits, feedback=None) -> str   # text with [n] citations

``ExtractiveGenerator`` (default, offline)
    Selects the source sentences that best answer the question and returns them
    verbatim with citations. It *cannot* invent facts, which makes it a perfect
    baseline, a CI-friendly default, and a fallback when no API key is set.

``AnthropicGenerator`` (``pip install hprag[anthropic]``, needs ``ANTHROPIC_API_KEY``)
    Fluent abstractive answers from Claude (default ``claude-opus-5-5``). Its
    output is *not* trusted: it always passes through claim verification.

``ScriptedGenerator``
    Returns pre-written answers. Used in tests and demos to show the verifier
    catching a deliberately hallucinated answer.

Adding a provider (OpenAI, a local model via Ollama, ...) means writing one class
with a ``generate`` method and registering it in :func:`make_generator`.
"""

from __future__ import annotations

import os
from typing import Protocol

from hprag.config import LLMConfig
from hprag.generation.prompts import ABSTAIN_TOKEN, GROUNDED_SYSTEM, build_user_prompt
from hprag.logging_utils import get_logger
from hprag.text import STOPWORDS, content_terms, query_terms, split_sentences, tokenize
from hprag.types import ScoredChunk

log = get_logger("generation")


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: list[ScoredChunk], feedback: str | None = None) -> str: ...


class GenerationRefused(RuntimeError):
    """The provider declined to answer (e.g. a safety refusal)."""


# --------------------------------------------------------------------------- extractive
class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_overlap: float = 0.34, relative: float = 0.75) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap
        self.relative = relative

    def generate(self, question: str, hits: list[ScoredChunk], feedback: str | None = None) -> str:
        q = set(query_terms(question))
        if not q:
            return ABSTAIN_TOKEN
        # Question words like "when"/"how many" hint at the answer type we want.
        wants_number = bool({"many", "much", "when", "year", "number", "percent"} &
                            set(tokenize(question)))
        scored: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        for src, h in enumerate(hits, 1):
            for sent in split_sentences(h.chunk.text):
                key = sent.lower()
                if key in seen or sent.startswith("#"):
                    continue
                seen.add(key)
                terms = set(content_terms(sent))
                if not terms:
                    continue
                overlap = len(q & terms) / len(q)
                bonus = 0.1 if wants_number and any(ch.isdigit() for ch in sent) else 0.0
                rank_prior = 0.05 / src  # mild preference for higher-ranked sources
                scored.append((overlap + bonus + rank_prior, src, sent))
        scored.sort(key=lambda x: -x[0])
        # Keep only sentences close to the best one: a sentence that matches half as
        # well is usually about something else and would just add noise.
        best = scored[0][0] if scored else 0.0
        picked = [s for s in scored if s[0] >= max(self.min_overlap, self.relative * best)]
        picked = picked[: self.max_sentences]
        if not picked:
            log.info("extractive.abstain", extra={"best": round(scored[0][0], 3) if scored else 0})
            return ABSTAIN_TOKEN
        out = []
        for _score, src, sent in picked:
            sent = sent.rstrip()
            if sent[-1] in ".!?":
                sent = sent[:-1]
            out.append(f"{sent} [{src}].")
        log.debug("extractive.picked", extra={"n": len(out), "scores": [round(p[0], 3) for p in picked]})
        return " ".join(out)


# --------------------------------------------------------------------------- scripted
class ScriptedGenerator:
    name = "scripted"

    def __init__(self, answers: list[str] | str) -> None:
        self.answers = [answers] if isinstance(answers, str) else list(answers)
        self.calls = 0

    def generate(self, question: str, hits: list[ScoredChunk], feedback: str | None = None) -> str:
        ans = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return ans


# --------------------------------------------------------------------------- anthropic
class AnthropicGenerator:
    """Claude via the official Anthropic SDK."""

    def __init__(self, cfg: LLMConfig, client=None) -> None:
        self.cfg = cfg
        self.name = f"anthropic:{cfg.model}"
        if client is None:
            import anthropic  # imported lazily so the core install stays dependency-free

            client = anthropic.Anthropic()
        self.client = client

    def generate(self, question: str, hits: list[ScoredChunk], feedback: str | None = None) -> str:
        prompt = build_user_prompt(question, hits, feedback)
        log.info("llm.request", extra={"model": self.cfg.model, "sources": len(hits),
                                       "prompt_chars": len(prompt), "repair": bool(feedback)})
        # Server-side fallback: if the primary model declines on a safety
        # classifier, the API re-runs the request on a suitable fallback model
        # inside the same call instead of returning nothing.
        resp = self.client.beta.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=GROUNDED_SYSTEM,
            output_config={"effort": self.cfg.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if resp.stop_reason == "refusal":
            raise GenerationRefused("Model declined to answer")
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        usage = getattr(resp, "usage", None)
        log.info("llm.response", extra={"model": getattr(resp, "model", self.cfg.model),
                                        "stop_reason": resp.stop_reason, "chars": len(text),
                                        "in_tokens": getattr(usage, "input_tokens", None),
                                        "out_tokens": getattr(usage, "output_tokens", None)})
        return text


def make_generator(cfg: LLMConfig) -> Generator:
    if cfg.provider == "anthropic":
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            log.warning("anthropic.no_credentials",
                        extra={"hint": "set ANTHROPIC_API_KEY; the SDK will also try `ant auth login`"})
        return AnthropicGenerator(cfg)
    if cfg.provider == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"Unknown LLM provider: {cfg.provider}")


__all__ = ["ABSTAIN_TOKEN", "AnthropicGenerator", "ExtractiveGenerator", "GenerationRefused",
           "Generator", "ScriptedGenerator", "STOPWORDS", "make_generator"]
