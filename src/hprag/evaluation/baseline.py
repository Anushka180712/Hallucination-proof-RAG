"""A deliberately naive RAG baseline, for honest comparisons.

The baseline uses the *same* retriever and corpus but switches every safety
mechanism off: no confidence gate, no verification gate, no repair, and a
generator that always answers with the closest-looking text. This mirrors what a
typical "retrieve top-k and ask the LLM" setup does, so the difference between
the two eval reports is the value added by the guards.
"""

from __future__ import annotations

import copy

from hprag.config import Settings
from hprag.generation import ExtractiveGenerator
from hprag.pipeline import RAGPipeline


def naive_settings(base: Settings) -> Settings:
    s = copy.deepcopy(base)
    s.guard.min_retrieval_confidence = 0.0
    s.guard.min_faithfulness = 0.0
    s.guard.min_claim_support = 0.0
    s.guard.contradiction_aborts = False
    s.guard.repair = "none"
    return s


def naive_pipeline(base: Settings, retriever=None) -> RAGPipeline:
    # min_overlap=0 => it never says "I don't know", like an LLM that always answers.
    gen = ExtractiveGenerator(max_sentences=1, min_overlap=0.0, relative=0.0)
    return RAGPipeline(naive_settings(base), retriever=retriever, generator=gen)
