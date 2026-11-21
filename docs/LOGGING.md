# Logging guide

The logs are designed so that someone reading them can follow exactly how an answer was produced, or why it was refused.

## Turning logs on

```bash
hprag ask "..." --log-level DEBUG                 # everything, readable
hprag ask "..." --log-format json                 # one JSON object per line
HPRAG_LOG_LEVEL=DEBUG HPRAG_LOG_FORMAT=json hprag serve
```

```python
from hprag import configure_logging
configure_logging("DEBUG", "pretty")   # or "json"
```

As a library, `hprag` is silent until you call `configure_logging`.

## Reading a trace

Every question gets a 12-character `trace_id`, shown in square brackets. The API returns the same id in the `X-Trace-Id` header. Here is a real DEBUG trace for one question, lightly trimmed:

```
INFO [5e4b3ba44c7d] pipeline          question.received  question=How much power do the solar arrays generate?
DEBU [5e4b3ba44c7d] retrieval.bm25    search  terms=["power", "solar", "array", "generate"] hits=2
INFO [5e4b3ba44c7d] retrieval.hybrid  retrieve.end  bm25_hits=2 dense_hits=8 fused=8 returned=[...] ms=0.8
DEBU [5e4b3ba44c7d] retrieval.hybrid  hit  rank=1 chunk=power_and_life_support#0 bm25=6.41 dense=0.29 rrf=0.0328 rerank=0.89
INFO [5e4b3ba44c7d] guard.confidence  scored  confidence=0.9445 top=0.8889 coverage=1.0 missing=[] unknown=[]
DEBU [5e4b3ba44c7d] generation        extractive.picked  n=1 scores=[1.15]
DEBU [5e4b3ba44c7d] verification      claim.checked  claim=Halcyon is powered by ... verdict=supported score=1.0 cited=[1] evidence=[1]
INFO [5e4b3ba44c7d] pipeline          verify.end  claims=1 hedges=0 supported=1 contradicted=0 ms=2.05
INFO [5e4b3ba44c7d] pipeline          answer.returned  faithfulness=1.0 claims=1 kept=1 confidence=0.9445 retrieve=1.07 generate=0.84 verify=2.13
```

Reading it top to bottom: the stemmed search terms, how many hits each retriever found, every score each chunk collected, the confidence breakdown, the verdict for each claim with its evidence, and the time spent in each stage.

An abstention explains itself:

```
WARN [3bd7573ef82b] pipeline  answer.abstained  reason=retrieval confidence 0.17 < 0.35; the knowledge base never mentions: commander
```

A dropped hallucination is logged as a warning:

```
WARN [c87c6d990dd1] pipeline  repair.dropped  dropped=["The station also carries a small nuclear reactor for backup power."]
```

## Event reference

| Logger | Event | Level | Key fields |
|---|---|---|---|
| `ingest.loader` | `loaded.directory`, `loaded.file`, `skipped.unsupported` | INFO / DEBUG / WARN | `path`, `docs`, `chars` |
| `ingest.chunker` | `chunked`, `chunked.all` | DEBUG / INFO | `sentences`, `chunks` |
| `retrieval.bm25` | `built`, `search` | INFO / DEBUG | `vocab`, `terms`, `hits` |
| `retrieval.dense` | `built` | INFO | `vectors`, `dim`, `embedder` |
| `retrieval.hybrid` | `index.build.end`, `retrieve.end`, `hit`, `index.saved`, `index.loaded` | INFO / DEBUG | `returned`, `rank`, `bm25`, `dense`, `rrf`, `rerank`, `ms` |
| `guard.confidence` | `scored` | INFO | `confidence`, `top`, `coverage`, `missing`, `unknown` |
| `generation` | `llm.request`, `llm.response`, `extractive.picked`, `extractive.abstain` | INFO / DEBUG | `model`, `stop_reason`, `in_tokens`, `out_tokens` |
| `verification` | `claim.checked`, `judge.request`, `judge.refused` | DEBUG / INFO / WARN | `verdict`, `score`, `cited`, `evidence`, `reason` |
| `pipeline` | `question.received`, `verify.end`, `repair.dropped`, `repair.regenerate.end`, `answer.returned`, `answer.abstained` | INFO / WARN | `faithfulness`, `reason`, timings |
| `evaluation` | `eval.item`, `eval.done` | INFO | metrics |
| `api` | `api.ingested` | INFO | `docs`, `chunks` |

## Useful queries on JSON logs

```bash
# Every line for one question
jq 'select(.trace_id == "5e4b3ba44c7d")' app.log

# Why did answers get refused today?
jq -r 'select(.event == "answer.abstained") | .reason' app.log | sort | uniq -c

# Which invented claims were removed?
jq -r 'select(.event == "repair.dropped") | .dropped[]' app.log

# Slowest stages
jq 'select(.ms) | {event, ms}' app.log | jq -s 'sort_by(-.ms) | .[:10]'
```
