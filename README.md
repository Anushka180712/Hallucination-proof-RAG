# 🛡️ Hallucination-Proof RAG

**A question-answering system that would rather say "I don't know" than make something up.**

It answers questions from your own documents. Every sentence in an answer carries a citation, every claim is fact-checked against the sources before you see it, and when the evidence is weak it says so and does not guess.

[![CI](https://github.com/Anushka180712/hallucination-proof-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Anushka180712/hallucination-proof-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-41%20passing-brightgreen)

---

## 🤔 What problem does this solve?

**RAG** (Retrieval-Augmented Generation) means: find the relevant passages in your documents, then ask an AI model to answer using them. It helps a lot, but models still **hallucinate**. They invent numbers, mix up facts, cite the wrong source, or answer confidently when the documents say nothing.

This project adds **safety checks at every step**, so a made-up answer never reaches the user.

| Without guards (naive RAG) | With this project |
|---|---|
| Answers every question, even ones the documents can't answer | **Abstains** and explains why |
| Citations may point at the wrong source | Every citation is **checked** |
| One invented sentence slips through inside a good answer | Each claim is **verified on its own**, and bad ones are removed |
| "2039" instead of "2041" goes unnoticed | **Number, negation and opposite-meaning checks** catch it |
| You can't tell why it said something | **Structured logs** with a trace id show every decision |

## 📊 Results

Measured on the bundled evaluation set (20 answerable and 10 unanswerable questions), fully offline. Re-run it with `make eval`.

| Metric | Naive RAG | **Hallucination-Proof** |
|---|---|---|
| Hallucination rate ↓ | 33.3% | **0.0%** |
| Correctly refused unanswerable questions ↑ | 0% | **100%** |
| Citation precision ↑ | 17.7% | **95.5%** |
| Answer accuracy ↑ | 100% | 95.0% |
| Wrongly refused answerable questions ↓ | 0% | 5.0% |

The trade-off is visible on purpose. The system refused 1 of 20 answerable questions (it could not match "maximum length" to "may not last longer than" with the offline word-based checker). In exchange, it never gave a wrong answer. Turning on the neural reranker or NLI verifier (see [Configuration](#%EF%B8%8F-configuration)) closes most of that gap.

## 🚀 Quick start (2 minutes, no API key needed)

```bash
git clone https://github.com/Anushka180712/hallucination-proof-rag.git
cd hallucination-proof-rag
pip install -e ".[dev]"

hprag demo     # a guided tour of every safety feature
```

The demo shows four cases on the bundled knowledge base:

1. A normal question, answered with a citation.
2. A question the documents can't answer, refused **before** any AI call is made.
3. An AI draft with one invented fact. The fact is removed and the rest is kept.
4. An AI draft that contradicts the sources. The answer is blocked.

### Use your own documents

```bash
hprag ingest path/to/your/docs          # .md, .txt, .rst or .jsonl
hprag ask "What is our refund policy?"
hprag ask "What is our refund policy?" --json    # full details: claims, scores, citations
```

### Use Claude for fluent answers

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
hprag ask "When was Halcyon commissioned?" --provider anthropic
hprag ask "..." --provider anthropic --verifier llm   # Claude also fact-checks each claim
```

The default model is `claude-opus-5-5`. Change it with `HPRAG_LLM_MODEL`. Claude's answers are **never trusted blindly**: they go through the same verification as everything else.

### Run as an API

```bash
pip install -e ".[api]"
hprag serve --corpus data/corpus --port 8000
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "How often do resupply missions arrive?"}'
```

Interactive docs are served at `http://localhost:8000/docs`.

### Run with Docker

```bash
docker build -t hprag .
docker run -p 8000:8000 hprag                                   # offline mode
docker run -p 8000:8000 -e ANTHROPIC_API_KEY -e HPRAG_LLM_PROVIDER=anthropic hprag
```

### Use it from Python

```python
from hprag import RAGPipeline, configure_logging

configure_logging("INFO")              # see every step as it happens
rag = RAGPipeline()
rag.ingest_path("data/corpus")

answer = rag.ask("How much power do the solar arrays generate?")
print(answer.answer)        # "... a combined 240 kilowatts at peak sunlight [1]."
print(answer.faithfulness)  # 1.0: every claim is supported
print(answer.abstained)     # False
for claim in answer.claims:
    print(claim.verdict, claim.text, claim.reason)
```

## 🧠 How it works

```mermaid
flowchart LR
    Q([Question]) --> R[1 · Hybrid retrieval<br/>BM25 + vectors + rerank]
    R --> G1{2 · Gate 1<br/>Enough evidence?}
    G1 -- no --> A1([Abstain<br/>no AI call made])
    G1 -- yes --> GEN[3 · Grounded generation<br/>cite every sentence]
    GEN --> V[4 · Claim verification<br/>check each sentence]
    V --> REP[5 · Repair<br/>drop or rewrite bad claims]
    REP --> G2{6 · Gate 2<br/>Faithful enough?<br/>No contradictions?}
    G2 -- no --> A2([Abstain<br/>with reason])
    G2 -- yes --> OUT([Verified answer<br/>+ citations + verdicts])
```

| Step | What it does | Why |
|---|---|---|
| **1. Hybrid retrieval** | Keyword search (BM25) and vector search run together, are merged with Reciprocal Rank Fusion, then reranked | Keywords catch exact names and numbers. Vectors catch rephrasing. Better retrieval means less temptation to guess. |
| **2. Gate 1: confidence** | Scores how well the passages cover the question. Words the documents have never mentioned count heavily. | If the evidence isn't there, any answer would be invented. Stopping here also saves an AI call. |
| **3. Grounded generation** | A strict prompt: use only the sources, cite every sentence, copy numbers exactly, say `INSUFFICIENT_EVIDENCE` if unsure | Each rule targets a known way models hallucinate |
| **4. Claim verification** | Splits the answer into single claims and checks each against the sources: supported, contradicted, or unsupported | One wrong sentence can hide inside a mostly right answer |
| **5. Repair** | Removes unsupported claims, or asks the model to rewrite with feedback | Keep what's true, drop what isn't |
| **6. Gate 2: faithfulness** | Blocks the answer if too much was unsupported or anything was contradicted | A clear, configurable quality bar for every answer |

📐 Full details with more diagrams: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
💡 The reasoning behind every choice: **[docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md)**
🔎 Reading and using the logs: **[docs/LOGGING.md](docs/LOGGING.md)**

## ✨ Features

**Retrieval**
- Hybrid search: BM25 (written from scratch) plus dense vectors, merged with Reciprocal Rank Fusion
- Reranking: a built-in offline reranker, or a neural cross-encoder
- Sentence-aware chunking with overlap and exact character offsets, so each citation points at the right text
- Saved indexes that load instantly and refuse to load with the wrong embedder

**Hallucination defences**
- Retrieval-confidence gate that abstains before generation
- Grounded prompt where each rule targets a specific failure
- Claim-by-claim verification with four verifiers: `lexical`, `nli`, `llm` (Claude as judge with structured JSON output) and a conservative `ensemble`
- Checks for wrong numbers, flipped negation ("never" vs "always") and opposite words ("increases" vs "decreases")
- Citation validation: does the cited source actually support the claim?
- Repair strategies: `drop`, `regenerate` (with feedback) or `none`
- Final faithfulness gate. Contradictions always block the answer.
- Honest hedges ("the sources don't mention X") are kept and never penalised

**Engineering**
- Pluggable providers: offline extractive (cannot hallucinate), Claude (`claude-opus-5-5`), or your own
- Evaluation harness with hallucination metrics and a naive baseline for comparison
- Structured logs (readable or JSON) with a trace id per question and timing for every stage
- CLI (`ingest`, `ask`, `eval`, `serve`, `demo`), REST API, Docker image, GitHub Actions CI
- 41 tests that run offline in under a second
- The core install needs only `numpy`

## ⚙️ Configuration

Every setting has a sensible default. Override settings with a TOML file (`--config hprag.toml`, see [`examples/hprag.toml`](examples/hprag.toml)) or environment variables named `HPRAG_<SECTION>_<KEY>`:

| Setting | Default | What it controls |
|---|---|---|
| `HPRAG_LLM_PROVIDER` | `extractive` | `extractive` (offline) or `anthropic` |
| `HPRAG_LLM_MODEL` | `claude-opus-5-5` | Claude model to use |
| `HPRAG_VERIFIER_KIND` | `lexical` | `lexical`, `nli`, `llm` or `ensemble` |
| `HPRAG_RETRIEVAL_RERANKER` | `lexical` | `lexical`, `cross-encoder` or `none` |
| `HPRAG_RETRIEVAL_EMBEDDER` | `hashing` | `hashing` (offline) or `sentence-transformers` |
| `HPRAG_GUARD_MIN_RETRIEVAL_CONFIDENCE` | `0.35` | Gate 1: below this, abstain without generating |
| `HPRAG_GUARD_MIN_CLAIM_SUPPORT` | `0.55` | Score a claim needs to count as supported |
| `HPRAG_GUARD_MIN_FAITHFULNESS` | `0.6` | Gate 2: share of claims that must be supported |
| `HPRAG_GUARD_REPAIR` | `drop` | `drop`, `regenerate` or `none` |
| `HPRAG_LOG_FORMAT` | `pretty` | `pretty` or `json` |

**Recommended production setup:** `pip install -e ".[all]"`, then set `HPRAG_LLM_PROVIDER=anthropic`, `HPRAG_RETRIEVAL_EMBEDDER=sentence-transformers`, `HPRAG_RETRIEVAL_RERANKER=cross-encoder` and `HPRAG_VERIFIER_KIND=ensemble`.

## 📁 Project structure

```
src/hprag/
├── pipeline.py          # The orchestrator: all 6 steps, start here
├── confidence.py        # Gate 1: retrieval confidence
├── ingest/              # Loading files and sentence-aware chunking
├── retrieval/           # BM25, dense vectors, RRF fusion, rerankers, index save/load
├── generation/          # Grounded prompts and providers (extractive, Claude, scripted)
├── verification/        # Claim splitting and the four verifiers
├── evaluation/          # Metrics, harness and naive baseline
├── api/server.py        # FastAPI service
├── cli.py               # The `hprag` command
├── config.py            # Every threshold, with comments
├── logging_utils.py     # Structured, trace-aware logging
├── text.py              # Shared tokeniser, stemmer and sentence splitter
└── types.py             # Data types: Document → Chunk → Claim → Answer
data/corpus/             # Demo knowledge base (a fictional space station)
data/eval/               # Labelled questions for evaluation
docs/                    # Architecture, design decisions, logging guide
tests/                   # 41 offline tests
```

**New to the code?** Read `pipeline.py` from top to bottom. Its docstring has the whole flow, and each step calls into one folder.

## ❓ FAQ

**Why is the demo knowledge base about a space station that doesn't exist?**
Because the facts are invented, no AI model can know them from training. Every correct answer *must* come from retrieval, which makes the test honest.

**Can it ever hallucinate?**
No system can promise zero errors. This one makes each kind of error much harder, shows its reasoning in every response, and lets you set how strict it is. Measure it on your own data with `hprag eval`.

**Does it need a GPU or an API key?**
No. The default setup runs fully offline on a laptop. Claude and the neural models are optional upgrades.

**How do I add another model provider?**
Write a class with a `generate(question, hits, feedback=None) -> str` method and register it in `generation/providers.py`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## 🧪 Development

```bash
make install   # pip install -e ".[dev,api,anthropic]"
make test      # 41 tests, offline
make lint      # ruff
make eval      # benchmark against the naive baseline
make demo
```

## 📜 License

MIT © Udisha Singh
