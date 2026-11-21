"""Gate 2: verify every claim against the retrieved evidence.

All verifiers fill in, for each :class:`~hprag.types.Claim`:
``verdict`` (supported / contradicted / unsupported), ``score`` (support in
[0, 1]), ``evidence`` (which sources support it) and ``reason`` (human-readable).

Available verifiers — pick with ``verifier.kind`` in the config:

``lexical`` (default, offline, deterministic)
    IDF-weighted term coverage + strict checks on numbers, negation and
    antonyms. Fast and transparent; catches invented entities, numeric drift,
    and flipped polarity. Cannot recognise deep paraphrase.

``nli`` (``pip install hprag[ml]``)
    A Natural Language Inference cross-encoder (DeBERTa) that predicts
    entailment / contradiction / neutral for (evidence, claim). Handles paraphrase.

``llm`` (``pip install hprag[anthropic]``)
    Claude as a judge, using structured outputs so the verdicts are always valid
    JSON matching a schema. Strongest, costs one extra call per answer.

``ensemble``
    Lexical + one model-based verifier, combined conservatively: a claim is only
    supported if *both* agree, and any contradiction wins. Use when false
    "supported" verdicts are much more costly than false alarms.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Protocol

from hprag.config import VerifierConfig
from hprag.logging_utils import get_logger
from hprag.text import content_terms, has_negation, numbers, split_sentences
from hprag.types import Claim, Verdict

log = get_logger("verification")

ANTONYMS = {
    "increas": "decreas", "rise": "fall", "higher": "lower", "more": "less", "larger": "smaller",
    "before": "after", "above": "below", "gain": "loss", "maximum": "minimum", "earli": "lat",
    "accept": "reject", "allow": "forbid", "succeed": "fail", "open": "clos", "start": "end",
}
ANTONYMS.update({v: k for k, v in list(ANTONYMS.items())})


class Verifier(Protocol):
    name: str

    def verify(self, claims: list[Claim], sources: list[str], threshold: float) -> list[Claim]: ...


def _premises(source: str) -> list[str]:
    """Single sentences and adjacent sentence pairs.

    Pairs let a claim that merges two neighbouring sentences ("X launched in
    2019 and is based in Oslo") still find support, without letting it
    combine facts from far-apart parts of a document.
    """
    sents = [s for s in split_sentences(source) if not s.startswith("#")] or [source]
    return sents + [f"{a} {b}" for a, b in zip(sents, sents[1:])]


# --------------------------------------------------------------------------- lexical
class LexicalVerifier:
    name = "lexical"

    def __init__(self, idf: Callable[[str], float] | None = None) -> None:
        self.idf = idf or (lambda _t: 1.0)

    def score_pair(self, claim: str, premise: str) -> tuple[float, bool, str]:
        """Return (support score, contradiction flag, reason)."""
        c_terms = list(dict.fromkeys(t for t in content_terms(claim) if not t[0].isdigit()))
        p_terms = set(content_terms(premise))
        if not c_terms:
            return 0.0, False, "claim has no content words"
        w = {t: max(self.idf(t), 0.1) for t in c_terms}
        total = sum(w.values())
        missing = [t for t in c_terms if t not in p_terms]
        coverage = 1 - sum(w[t] for t in missing) / total

        c_nums, p_nums = numbers(claim), numbers(premise)
        bad_nums = c_nums - p_nums
        topical = coverage >= 0.6  # premise is clearly about the same thing

        if bad_nums and topical and p_nums:
            return 0.0, True, f"number mismatch: claim says {sorted(bad_nums)}, source says {sorted(p_nums)}"
        if bad_nums:
            return min(coverage, 0.3), False, f"numbers {sorted(bad_nums)} not found in source"
        if topical and has_negation(claim) != has_negation(premise):
            return 0.0, True, "negation mismatch between claim and source"
        if topical:
            for t in c_terms:
                anti = next((a for k, a in ANTONYMS.items() if t.startswith(k)), None)
                if anti and t not in p_terms and any(p.startswith(anti) for p in p_terms):
                    return 0.0, True, f"opposite meaning: claim uses '{t}', source uses '{anti}…'"
        reason = "all key terms found" if not missing else f"missing terms: {missing[:6]}"
        return round(coverage, 4), False, reason

    def verify(self, claims: list[Claim], sources: list[str], threshold: float) -> list[Claim]:
        for claim in claims:
            best, best_reason, support, contra = 0.0, "no evidence", [], None
            for n, src in enumerate(sources, 1):
                src_best, src_reason, src_contra = 0.0, "", None
                for prem in _premises(src):
                    s, is_contra, why = self.score_pair(claim.text, prem)
                    if is_contra and src_contra is None:
                        src_contra = why
                    if s > src_best:
                        src_best, src_reason = s, why
                if src_best >= threshold:
                    support.append(n)
                elif src_contra and contra is None:
                    contra = f"[{n}] {src_contra}"
                if src_best > best:
                    best, best_reason = src_best, f"[{n}] {src_reason}"
            _finalise(claim, best, support, contra, best_reason, threshold)
        return claims


# --------------------------------------------------------------------------- NLI
class NLIVerifier:  # pragma: no cover - needs model download
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self.name = f"nli:{model}"
        self._model = CrossEncoder(model)
        # Label order for the common NLI cross-encoders: contradiction, entailment, neutral.
        labels = getattr(self._model.model.config, "id2label", {0: "contradiction", 1: "entailment",
                                                                 2: "neutral"})
        self._idx = {v.lower(): int(k) for k, v in labels.items()}

    def verify(self, claims: list[Claim], sources: list[str], threshold: float) -> list[Claim]:
        for claim in claims:
            pairs = [(n, p) for n, s in enumerate(sources, 1) for p in _premises(s)]
            logits = self._model.predict([(p, claim.text) for _, p in pairs])
            best, support, contra, reason = 0.0, [], None, "no evidence"
            per_src: dict[int, float] = {}
            for (n, prem), row in zip(pairs, logits):
                ex = [math.exp(float(x)) for x in row]
                probs = [x / sum(ex) for x in ex]
                ent, con = probs[self._idx["entailment"]], probs[self._idx["contradiction"]]
                per_src[n] = max(per_src.get(n, 0.0), ent)
                if ent > best:
                    best, reason = ent, f"[{n}] entailment p={ent:.2f}"
                if con > 0.8 and contra is None:
                    contra = f"[{n}] contradiction p={con:.2f}: {prem[:80]}"
            support = [n for n, s in per_src.items() if s >= threshold]
            _finalise(claim, best, support, contra, reason, threshold)
        return claims


# --------------------------------------------------------------------------- LLM judge
JUDGE_SYSTEM = """You are a strict fact-checker. For each numbered claim, decide whether the numbered
sources SUPPORT it (a source states it, possibly paraphrased), CONTRADICT it (a source states something
incompatible), or leave it UNSUPPORTED (the sources do not say). Use only the sources, never outside
knowledge. Partial support counts as unsupported. List the numbers of every source that supports a claim."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["supported", "contradicted", "unsupported"]},
                    "sources": {"type": "array", "items": {"type": "integer"}},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "verdict", "sources", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


class LLMJudgeVerifier:
    def __init__(self, model: str = "claude-opus-5-5", client=None) -> None:
        self.name = f"llm:{model}"
        self.model = model
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client

    def verify(self, claims: list[Claim], sources: list[str], threshold: float) -> list[Claim]:
        if not claims:
            return claims
        src = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(sources, 1))
        cl = "\n".join(f"{i}. {c.text}" for i, c in enumerate(claims, 1))
        log.info("judge.request", extra={"model": self.model, "claims": len(claims)})
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=JUDGE_SYSTEM,
            output_config={"effort": "medium",
                           "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
            messages=[{"role": "user", "content": f"Sources:\n{src}\n\nClaims:\n{cl}"}],
        )
        if resp.stop_reason == "refusal":
            # Fail closed: if the judge will not judge, nothing counts as verified.
            log.warning("judge.refused")
            for c in claims:
                _finalise(c, 0.0, [], None, "judge declined", threshold)
            return claims
        text = next(b.text for b in resp.content if b.type == "text")
        by_idx = {v["claim"]: v for v in json.loads(text)["verdicts"]}
        for i, c in enumerate(claims, 1):
            v = by_idx.get(i, {"verdict": "unsupported", "sources": [], "reason": "not judged"})
            valid = [n for n in v["sources"] if 1 <= n <= len(sources)]
            score = {"supported": 1.0, "contradicted": 0.0, "unsupported": 0.2}[v["verdict"]]
            contra = v["reason"] if v["verdict"] == "contradicted" else None
            _finalise(c, score, valid if v["verdict"] == "supported" else [], contra, v["reason"],
                      threshold)
        return claims


# --------------------------------------------------------------------------- ensemble
class EnsembleVerifier:
    """Conservative AND-combination of several verifiers."""

    def __init__(self, members: list[Verifier]) -> None:
        self.members = members
        self.name = "ensemble(" + ",".join(m.name for m in members) + ")"

    def verify(self, claims: list[Claim], sources: list[str], threshold: float) -> list[Claim]:
        runs = []
        for m in self.members:
            copies = [Claim(text=c.text, cited=list(c.cited)) for c in claims]
            runs.append(m.verify(copies, sources, threshold))
        for i, claim in enumerate(claims):
            results = [r[i] for r in runs]
            contra = next((r.reason for r in results if r.verdict == Verdict.CONTRADICTED), None)
            support = sorted(set.intersection(*(set(r.evidence) for r in results)))
            score = min(r.score for r in results)
            reason = " | ".join(f"{m.name}: {r.reason}" for m, r in zip(self.members, results))
            _finalise(claim, score, support, contra, reason, threshold)
        return claims


# --------------------------------------------------------------------------- shared
def _finalise(claim: Claim, best: float, support: list[int], contra: str | None, reason: str,
              threshold: float) -> None:
    claim.score = round(best, 4)
    claim.evidence = sorted(support)
    if support and best >= threshold:
        claim.verdict, claim.reason = Verdict.SUPPORTED, reason
    elif contra:
        claim.verdict, claim.reason = Verdict.CONTRADICTED, contra
    else:
        claim.verdict, claim.reason = Verdict.UNSUPPORTED, reason
    # Citation check: did the generator cite a source that actually supports this?
    claim.citation_valid = bool(set(claim.cited) & set(claim.evidence))
    log.debug("claim.checked", extra={"claim": claim.text, "verdict": claim.verdict.value,
                                      "score": claim.score, "cited": claim.cited,
                                      "evidence": claim.evidence, "reason": claim.reason})


def make_verifier(cfg: VerifierConfig, idf: Callable[[str], float] | None = None) -> Verifier:
    if cfg.kind == "lexical":
        return LexicalVerifier(idf)
    if cfg.kind == "nli":
        return NLIVerifier(cfg.nli_model)
    if cfg.kind == "llm":
        return LLMJudgeVerifier(cfg.judge_model)
    if cfg.kind == "ensemble":
        second = NLIVerifier(cfg.nli_model) if cfg.nli_model else LLMJudgeVerifier(cfg.judge_model)
        return EnsembleVerifier([LexicalVerifier(idf), second])
    raise ValueError(f"Unknown verifier: {cfg.kind}")
