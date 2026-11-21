"""Show the verifier catching a hallucinated draft.

A ScriptedGenerator stands in for an LLM that invents one fact and gets one
number wrong. Run from the repository root:  python examples/catch_hallucination.py
"""

from hprag import RAGPipeline
from hprag.generation import ScriptedGenerator

draft = (
    "Halcyon was commissioned on 14 March 2041 [1]. "
    "It has a permanent crew of 9 people [1]. "
    "Each Kestrel capsule carries 5,000 kilograms of supplies [2]. "
    "The station has a swimming pool in the Atrium module [1]."
)

rag = RAGPipeline(generator=ScriptedGenerator(draft))
rag.ingest_path("data/corpus")
answer = rag.ask("Tell me about Halcyon's commissioning and crew.")

print("DRAFT :", answer.draft)
print("FINAL :", answer.answer)
print("STATUS:", "abstained - " + answer.abstain_reason if answer.abstained else "answered")
for c in answer.claims:
    print(f"  {c.verdict.value:<12} {c.text}\n               reason: {c.reason}")
