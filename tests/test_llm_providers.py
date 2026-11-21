"""Claude-backed components, tested with a fake client (no network, no API key)."""

import json
from types import SimpleNamespace as NS

import pytest

from hprag.config import LLMConfig
from hprag.generation import AnthropicGenerator, GenerationRefused
from hprag.types import Chunk, Claim, ScoredChunk, Verdict
from hprag.verification import LLMJudgeVerifier


class FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kw):
        self.calls.append(kw)
        return NS(content=[NS(type="text", text=self.text)], stop_reason=self.stop_reason,
                  model=kw["model"], usage=NS(input_tokens=10, output_tokens=5))


def fake_client(text, stop_reason="end_turn"):
    m = FakeMessages(text, stop_reason)
    return NS(messages=m, beta=NS(messages=m)), m


HITS = [ScoredChunk(Chunk("d#0", "d", "Halcyon has a crew of 9.", 0, 24), 1.0)]


def test_anthropic_generator_sends_grounded_request():
    client, m = fake_client("Halcyon has a crew of 9 [1].")
    gen = AnthropicGenerator(LLMConfig(provider="anthropic"), client=client)
    assert gen.generate("How many crew?", HITS) == "Halcyon has a crew of 9 [1]."
    kw = m.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert "[1] (from: d)" in kw["messages"][0]["content"]
    assert kw["fallbacks"] == "default"


def test_anthropic_refusal_raises():
    client, _ = fake_client("", stop_reason="refusal")
    with pytest.raises(GenerationRefused):
        AnthropicGenerator(LLMConfig(), client=client).generate("q", HITS)


def test_llm_judge_parses_structured_verdicts():
    payload = {"verdicts": [
        {"claim": 1, "verdict": "supported", "sources": [1], "reason": "stated"},
        {"claim": 2, "verdict": "contradicted", "sources": [], "reason": "source says 9"},
    ]}
    client, m = fake_client(json.dumps(payload))
    claims = [Claim("Crew of 9.", cited=[1]), Claim("Crew of 12.", cited=[1])]
    LLMJudgeVerifier(client=client).verify(claims, ["Halcyon has a crew of 9."], 0.55)
    assert claims[0].verdict == Verdict.SUPPORTED and claims[0].citation_valid
    assert claims[1].verdict == Verdict.CONTRADICTED
    assert m.calls[0]["output_config"]["format"]["type"] == "json_schema"


def test_llm_judge_fails_closed_on_refusal():
    client, _ = fake_client("", stop_reason="refusal")
    claims = [Claim("Crew of 9.")]
    LLMJudgeVerifier(client=client).verify(claims, ["Halcyon has a crew of 9."], 0.55)
    assert claims[0].verdict == Verdict.UNSUPPORTED
