# Changelog

## 1.1.0 (2026-11-21)

- Web UI (`hprag ui`): ask your own questions and see verified answers, clickable citations,
  per-statement fact-checks, highlighted sources and refusal reasons
- Clickable sample questions (answerable, tricky, not in the documents), checked by a test
- Add pasted text to the knowledge base from the UI (`POST /documents`, `RAGPipeline.add_documents`)
- Fix: irregular verbs ("began"/"begin") and possessives ("Halcyon's") no longer cause false refusals

## 1.0.0 (2026-11-21)

First release.

- Hybrid retrieval: BM25 plus dense vectors, Reciprocal Rank Fusion and reranking
- Gate 1: retrieval-confidence abstention with unknown-term detection
- Grounded prompting with mandatory citations and an abstain token
- Claim-level verification: lexical, NLI, Claude-as-judge (structured outputs) and ensemble
- Contradiction detection for numbers, negation and antonyms, plus citation validation
- Repair strategies (`drop`, `regenerate`) and a final faithfulness gate
- Providers: offline extractive and Claude (`claude-opus-5-5`)
- Evaluation harness with hallucination metrics and a naive baseline
- Structured, trace-aware logging (pretty and JSON)
- CLI, FastAPI service, Docker image, GitHub Actions CI, 41 offline tests
