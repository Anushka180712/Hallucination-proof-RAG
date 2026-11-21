"""Minimal end-to-end example: ingest, ask, inspect every claim.

Run from the repository root:  python examples/quickstart.py
"""

from hprag import RAGPipeline, configure_logging

configure_logging("WARNING")  # use "DEBUG" to watch every step

rag = RAGPipeline()
rag.ingest_path("data/corpus")

for question in [
    "How many people are in the permanent crew?",
    "What is the annual budget of the Meridian Space Consortium?",
]:
    a = rag.ask(question)
    print(f"\nQ: {question}\nA: {a.answer}")
    if a.abstained:
        print(f"   (abstained: {a.abstain_reason})")
    for c in a.claims:
        print(f"   - {c.verdict.value:<12} {c.text}  (evidence: {c.evidence})")
