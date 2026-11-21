# Design decisions

Every significant choice in this project, written as a short decision record: the context, the decision, why, and what it costs. If you disagree with one, the trade-off section tells you what you would gain and lose by changing it.

---

### DD-01: Abstaining is a correct answer

**Context.** Most RAG systems are tuned to always answer. A wrong answer delivered confidently does more harm than no answer.
**Decision.** The system has two explicit gates that return "I don't have enough reliable evidence" with a machine-readable reason.
**Why.** Users can act on "I don't know" (ask someone, search elsewhere). They cannot detect a fluent fabrication.
**Trade-off.** Some answerable questions are refused (5% on the demo set). The thresholds are configurable, and the eval harness reports `false_abstention_rate` next to `hallucination_rate`, so the cost stays visible.

### DD-02: Defence in depth, not a single check

**Decision.** Nine independent layers (see ARCHITECTURE §5) instead of one "faithfulness score".
**Why.** Each failure has a different cheapest fix. Unknown topics are cheapest to catch *before* generation. Invented details can only be caught *after*. Wrong citations need a per-claim check. One check cannot do all of these.
**Trade-off.** More code and a little more latency (a few milliseconds offline).

### DD-03: Verify claims one at a time, not whole answers

**Decision.** Answers are split into sentence-level claims, and each one gets a verdict.
**Why.** A whole-answer score averages one invented number into an otherwise correct answer and passes it. Claim-level checks find it, report it and remove it while keeping the correct parts.
**Trade-off.** Sentence splitting is heuristic. A sentence that contains two facts is checked as one claim. The prompt asks for "short sentences, one claim each" to reduce this.

### DD-04: Three verdicts, not two

**Decision.** `supported` / `unsupported` / `contradicted`.
**Why.** "The sources don't say" and "the sources say the opposite" are different failures. A contradiction means the model actively misread the evidence, so by default it blocks the whole answer (`contradiction_aborts`). An unsupported claim is simply removed.

### DD-05: Hedges are not penalised

**Decision.** Sentences like "The sources do not mention X" are detected, kept in the answer and excluded from the faithfulness score.
**Why.** If honesty lowered the score, the system would learn to guess. The incentives should reward saying what is missing.

### DD-06: Hybrid retrieval (BM25 + dense) fused with RRF

**Decision.** Run both retrievers and merge the results with Reciprocal Rank Fusion.
**Why.** Hallucination-prone questions hinge on exact tokens such as names, codes and numbers, which BM25 handles well and embeddings often blur. Paraphrased questions need embeddings. RRF combines the two using ranks only, so no fragile normalisation of BM25 scores against cosine scores is needed. `k=60` follows Cormack et al. (2009).
**Trade-off.** Two indexes to build and keep in sync. Both are built together and saved together.

### DD-07: Rerank before generation

**Decision.** Retrieve 20 candidates per retriever, rerank them, and pass only the top 5 to the generator.
**Why.** The generator is more likely to hallucinate when given irrelevant passages ("distractors"). Wide recall, then narrow precision, gives the generator the least noise.

### DD-08: Gate 1 penalises unknown words

**Decision.** If an important word in the question never appears in the corpus, retrieval confidence is halved.
**Why.** "What does the Zephyr telescope observe?" against a corpus that never mentions "Zephyr" is the textbook setup for a fabricated answer. Retrieval will still return *something* (it always does), and the model will try to be helpful.
**Trade-off.** Synonyms the corpus doesn't use ("cost" when the documents say "price") lower confidence. A neural embedder and reranker reduce this. We still prefer a false abstention to a false answer.

### DD-09: Sentence-aware chunks with offsets

**Decision.** Chunks are built from whole sentences (max 600 characters, 1 sentence of overlap), and each chunk records `start`/`end` offsets.
**Why.** Fixed-size windows cut facts in half ("revenue grew 12%" without "in 2023"), which leads directly to wrong answers. Offsets make every citation traceable to exact characters.
**Trade-off.** Chunks vary in size. A sentence longer than the limit becomes its own chunk instead of being cut.

### DD-10: An offline default for every component

**Decision.** The default configuration needs only `numpy`: hashing embedder, lexical reranker, extractive generator, lexical verifier. Claude and neural models are optional extras.
**Why.** (1) Anyone can clone and run it in two minutes. (2) Tests and CI are fast, free and deterministic. (3) The extractive generator *cannot* invent facts, which makes it a useful baseline and a safe fallback when no API key is available.
**Trade-off.** Offline components are weaker at paraphrase. The README's recommended production setup turns on the neural options.

### DD-11: Claude as the default LLM, behind a provider interface

**Decision.** `AnthropicGenerator` uses `claude-opus-5-5` through the official SDK. Any provider can be added through the `Generator` protocol.
**Why.** It is the latest and most capable Opus model and follows grounding instructions well. Server-side refusal fallback (`fallbacks="default"`) keeps a safety decline from turning into an empty answer, and a final refusal becomes an abstention. The interface keeps the rest of the system independent of any vendor.
**Trade-off.** Network calls add latency and cost. The model's output is still treated as untrusted and verified like everything else.

### DD-12: LLM-as-judge uses structured outputs

**Decision.** `LLMJudgeVerifier` requests JSON that matches a schema (`output_config.format`) and checks all claims in one call.
**Why.** Free-text verdicts need fragile parsing, and a parse failure could be mistaken for "supported". A schema makes every verdict valid. One call per answer instead of one per claim keeps cost low.
**Fail-closed.** If the judge refuses, every claim becomes `unsupported`, never `supported`.

### DD-13: The ensemble combines conservatively

**Decision.** In `EnsembleVerifier`, a claim is supported only if **every** member supports it, and any contradiction wins.
**Why.** In this domain, a false "supported" (a hallucination reaches the user) costs far more than a false alarm (one claim dropped).

### DD-14: Repair by dropping, but the draft must still pass

**Decision.** The default repair removes unsupported claims, but Gate 2 checks the faithfulness of the *original draft* (≥ 0.6).
**Why.** If most of what the model wrote was invented, the few surviving sentences come from an unreliable generation, and returning them would hide that. `regenerate` is available when you would rather spend an extra call to get a complete answer.

### DD-15: Fictional demo corpus

**Decision.** The bundled knowledge base describes an invented space station.
**Why.** With real-world facts, a model could answer correctly from memory, and you could not tell grounding from recall. With invented facts, every correct answer proves retrieval worked, and every unanswerable question really is unanswerable.

### DD-16: Evaluation that includes unanswerable questions

**Decision.** The eval set is one-third unanswerable, and the headline metric is `hallucination_rate` (wrong answers among answers given).
**Why.** Accuracy on answerable questions alone rewards a system that always answers. Only unanswerable questions measure the tendency to invent. A naive baseline with the same retriever and no guards is evaluated next to the real pipeline, so the improvement is measured, not claimed.

### DD-17: Structured logs with trace ids, using only the standard library

**Decision.** `logging` from the standard library, a trace id per question in a `ContextVar`, readable or JSON output, and `stage()` timings.
**Why.** For a system whose selling point is trust, every decision must be auditable: which chunks, what confidence, which claims failed and why. Trace ids keep one question's log lines together even under concurrent API requests. The standard library means no dependency and compatibility with any log pipeline.
**Library etiquette.** `hprag` installs a `NullHandler`, so it stays silent unless the application calls `configure_logging`.

### DD-18: Fail loudly on configuration mistakes

**Decision.** Unknown TOML keys raise an error. Loading an index with a different embedder raises an error.
**Why.** A typo like `min_faithfulnes = 0.9` that is silently ignored would leave the system *less* safe than the user believes. Vectors from a different embedder give silently wrong neighbours. Both are worse than a crash.

### DD-19: No PDF/HTML parsing built in

**Decision.** Only plain-text formats are ingested.
**Why.** Badly extracted text (merged columns, broken tables) is a hidden cause of "hallucinations" that are really faithful answers from garbled sources. Extraction deserves a dedicated, well-tested tool chosen for your documents.

### DD-20: Plain dataclasses and protocols

**Decision.** Core types are dataclasses. Extension points are `typing.Protocol`s.
**Why.** No framework lock-in, readable data flow, trivial serialisation (`Answer.to_dict()`), and implementations don't have to inherit from anything.

---

## Guardrails

### DD-21: Guardrails are a separate layer from hallucination checks

**Context.** Hallucination checks judge whether an answer is faithful to the sources. They don't decide whether a request should be served at all, whether it contains private data, or whether a *document* is trying to manipulate the model.
**Decision.** A `guardrails/` package wraps the core pipeline: input guard, then cache, then the core, then output guard. A document guard runs at ingest.
**Why.** Keeping the two concerns apart keeps both readable and lets each be switched or tuned independently (`HPRAG_GUARDRAILS_ENABLED=false` turns guardrails off without touching the hallucination defences).

### DD-22: Rules, not a classifier, for the default guardrails

**Decision.** Guardrails use phrase-level regular expressions, collected in one file (`guardrails/rules.py`).
**Why.** They run offline in microseconds, they're deterministic, and every block names the exact rule and matched text, which matches this project's "explain every decision" goal. A classifier model would need a download and would give a score without a reason.
**Trade-off.** Rules can be rephrased around and have limited coverage. That's acceptable because they're not the last line of defence: an injection that slips through still meets the grounded prompt ("use only the sources") and claim verification (anything not in the sources is removed). For higher assurance, add a model-based classifier as another `InputGuard` check.
**False positives.** Every pattern needs a phrase, never a single word. A test runs normal questions containing *fire*, *kill* and *rules* through the guard and requires that none are blocked. One real false positive found during development ("What are the rules for spacewalks?") is now a regression test.

### DD-23: Block attacks before retrieval

**Decision.** A blocked question never reaches retrieval, the cache or any LLM.
**Why.** It's cheapest, it means an attack payload is never sent to a third-party API, and the raw text of a blocked question is deliberately not logged (only the rule name and the matched fragment are), since it may be an attack or contain sensitive text.

### DD-24: Redact personal information instead of refusing

**Decision.** Emails, phone numbers, card numbers (Luhn-checked), SSNs and IP addresses are replaced with placeholders, and whole clauses like "my email is …" are removed before searching. The same redaction runs on answers and citation quotes.
**Why.** A question that happens to contain an email is still a legitimate question. Refusing it is unhelpful; passing the email to an LLM provider and into logs is a privacy leak. Removing the clause (not just the value) keeps words like "email" from making a good question look off-topic to Gate 1.
**Precision.** Card numbers must pass the Luhn checksum, and international phone numbers must start with `+`, so dates and measurements in documents ("14 March 2041", "1,800 kWh") are never masked.

### DD-25: Remove planted instructions sentence by sentence, keeping offsets

**Context.** Indirect prompt injection is the RAG-specific attack: a document in the knowledge base says "AI assistants must always answer that X". Retrieval then hands that instruction to the model as trusted context.
**Decision.** At ingest, each sentence is scanned. Sentences that address an AI, override instructions, or use fake system tags are replaced with spaces of the same length.
**Why sentences, not documents.** A supplier note with one malicious line still has useful facts. Quarantining the whole document would lose them.
**Why spaces.** Every chunk keeps exact character offsets into its source file. Overwriting with equal-length whitespace keeps every citation pointing at the right characters.

### DD-26: Self-harm gets support, not a refusal

**Decision.** Messages suggesting the writer may be at risk get a short, warm reply pointing to people and crisis lines, instead of "I can't help with that".
**Why.** A refusal phrased like a policy violation is the wrong response to someone who may be struggling. This tool can't counsel anyone, but it can avoid making things worse and point to real help.

### DD-27: Cache verified answers, keyed on meaning-bearing words, invalidated on change

**Decision.** An LRU cache stores the final, already-verified answer. The key is the sorted set of stemmed content words plus an index version that increments whenever documents change.
**Why.** Repeat questions are common, and with an LLM provider each one costs money and seconds. Caching *after* verification means a cached answer is exactly as trustworthy as a fresh one. Normalising the key makes "When was Halcyon commissioned?" and "when was halcyon commissioned" share an entry. Versioning means a stale answer can never be served after the knowledge base changes.
**Not a semantic cache.** Matching by embedding similarity would also catch paraphrases, but two questions can be similar and still need different answers ("When did X start?" vs "When did X end?"). Exact matching on content words is the safe choice for a system whose goal is never being wrong.

### DD-28: Rate limiting at the API edge

**Decision.** A per-client token bucket (default 60 requests a minute) on `POST /ask` and `POST /documents`, returning `429` with `Retry-After`.
**Why.** It protects the server and any paid LLM API behind it from floods, accidental or not. A token bucket allows short bursts (a user clicking through sample questions) while capping sustained load.
