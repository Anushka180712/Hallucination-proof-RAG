from hprag import RAGPipeline, Settings
from hprag.generation import ScriptedGenerator


def test_answers_with_citation(rag):
    a = rag.ask("How much power do the solar arrays generate?")
    assert not a.abstained
    assert "240" in a.answer and "[1]" in a.answer
    assert a.faithfulness == 1.0
    assert a.citations[0].doc_id == "power_and_life_support"


def test_abstains_on_unknown_topic_without_calling_llm(rag):
    gen = ScriptedGenerator("should never be used [1].")
    rag.generator = gen
    a = rag.ask("Who is the station commander?")
    assert a.abstained and "commander" in a.abstain_reason
    assert gen.calls == 0  # gate 1 saved an LLM call


def test_drops_invented_claim(rag):
    rag.generator = ScriptedGenerator(
        "Halcyon was commissioned on 14 March 2041 [1]. It orbits at 412 kilometres [1]. "
        "The station carries a nuclear reactor for backup power [1]."
    )
    a = rag.ask("When was Halcyon commissioned?")
    assert not a.abstained
    assert "reactor" not in a.answer and "2041" in a.answer
    assert "reactor" in a.draft  # the original draft is kept for transparency


def test_blocks_contradiction(rag):
    rag.generator = ScriptedGenerator("Open flames are permitted anywhere on the station [1].")
    a = rag.ask("Are open flames allowed on the station?")
    assert a.abstained and "contradicted" in a.abstain_reason


def test_mostly_invented_draft_is_rejected(rag):
    rag.generator = ScriptedGenerator(
        "Halcyon was commissioned in 2041 [1]. It has a swimming pool [1]. It has a cinema [1]."
    )
    a = rag.ask("When was Halcyon commissioned?")
    assert a.abstained and "faithfulness" in a.abstain_reason


def test_regenerate_repair_uses_feedback(corpus_path):
    s = Settings()
    s.guard.repair = "regenerate"
    gen = ScriptedGenerator([
        "Halcyon was commissioned in 2041 [1]. It has a swimming pool [1].",
        "Halcyon was commissioned on 14 March 2041 [1].",
    ])
    rag = RAGPipeline(s, generator=gen)
    rag.ingest_path(str(corpus_path))
    a = rag.ask("When was Halcyon commissioned?")
    assert gen.calls == 2 and not a.abstained and "pool" not in a.answer


def test_generator_abstain_token(rag):
    rag.generator = ScriptedGenerator("INSUFFICIENT_EVIDENCE")
    a = rag.ask("When was Halcyon commissioned?")
    assert a.abstained


def test_answer_serialises(rag):
    d = rag.ask("Who is the chief scientist?").to_dict()
    assert d["claims"][0]["verdict"] == "supported"
