# Changelog

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
