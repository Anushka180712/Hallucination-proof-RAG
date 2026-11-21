"""Evaluation harness with hallucination-focused metrics.

"It seems to work on my three questions" is not evidence. This module runs a
labelled question set through the pipeline and reports numbers you can track
over time and compare across configurations.

Dataset format (JSONL, one object per line)::

    {"id": "a01", "question": "...", "answerable": true,
     "must_include": ["2041"], "gold_docs": ["halcyon_station_overview"]}
    {"id": "u01", "question": "...", "answerable": false}

Unanswerable questions are as important as answerable ones: they are the
only way to measure whether the system *invents* answers.

Metrics
-------
answer_accuracy
    Answerable questions answered with every ``must_include`` fact present.
hallucination_rate
    Of all answers actually given, the share that are wrong: an answer to an
    unanswerable question, or an answer missing the required facts.
    **Lower is better. This is the headline number.**
abstention_precision / abstention_recall
    Precision: when it abstained, was abstaining right? Recall: of the
    unanswerable questions, how many did it correctly refuse?
false_abstention_rate
    Answerable questions it refused. The price paid for safety; keep it visible.
citation_precision
    Share of citations pointing at a gold document.
retrieval_recall
    Answerable questions where a gold document was in the top-k retrieved chunks.
mean_faithfulness
    Average verified-claim ratio of the answers given (from the verifier).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hprag.logging_utils import get_logger
from hprag.pipeline import RAGPipeline

log = get_logger("evaluation")


@dataclass
class EvalItem:
    id: str
    question: str
    answerable: bool
    must_include: list[str] = field(default_factory=list)
    gold_docs: list[str] = field(default_factory=list)


@dataclass
class ItemResult:
    id: str
    question: str
    answerable: bool
    abstained: bool
    correct: bool
    hallucinated: bool
    answer: str
    abstain_reason: str | None
    faithfulness: float
    confidence: float
    citation_hits: int
    citations: int
    retrieval_hit: bool | None
    ms: float


@dataclass
class EvalReport:
    name: str
    n: int
    metrics: dict[str, float]
    items: list[ItemResult]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "n": self.n, "metrics": self.metrics,
                "items": [asdict(i) for i in self.items]}


def load_dataset(path: str | Path) -> list[EvalItem]:
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(EvalItem(**json.loads(line)))
    return items


def _contains_all(answer: str, facts: list[str]) -> bool:
    a = answer.lower()
    return all(f.lower() in a for f in facts)


def evaluate(pipeline: RAGPipeline, items: list[EvalItem], name: str = "pipeline") -> EvalReport:
    results: list[ItemResult] = []
    for it in items:
        t0 = time.perf_counter()
        ans = pipeline.ask(it.question)
        ms = round((time.perf_counter() - t0) * 1000, 2)

        retrieval_hit = None
        if it.answerable and it.gold_docs:
            hits = pipeline.retriever.retrieve(it.question)
            retrieval_hit = any(h.chunk.doc_id in it.gold_docs for h in hits)

        if ans.abstained:
            correct, hallucinated = (not it.answerable), False
        elif it.answerable:
            correct = _contains_all(ans.answer, it.must_include)
            hallucinated = not correct
        else:
            correct, hallucinated = False, True  # answered a question with no answer in the corpus

        cites = [] if ans.abstained else ans.citations
        results.append(ItemResult(
            id=it.id, question=it.question, answerable=it.answerable, abstained=ans.abstained,
            correct=correct, hallucinated=hallucinated, answer=ans.answer,
            abstain_reason=ans.abstain_reason, faithfulness=ans.faithfulness,
            confidence=ans.confidence,
            citation_hits=sum(c.doc_id in it.gold_docs for c in cites), citations=len(cites),
            retrieval_hit=retrieval_hit, ms=ms,
        ))
        log.info("eval.item", extra={"id": it.id, "correct": correct, "hallucinated": hallucinated,
                                     "abstained": ans.abstained})
    report = EvalReport(name=name, n=len(results), metrics=compute_metrics(results), items=results)
    log.info("eval.done", extra={"run": name, **report.metrics})
    return report


def _ratio(a: float, b: float) -> float:
    return round(a / b, 4) if b else 0.0


def compute_metrics(rs: list[ItemResult]) -> dict[str, float]:
    ans = [r for r in rs if r.answerable]
    unans = [r for r in rs if not r.answerable]
    answered = [r for r in rs if not r.abstained]
    abstained = [r for r in rs if r.abstained]
    with_gold = [r for r in ans if r.retrieval_hit is not None]
    return {
        "answer_accuracy": _ratio(sum(r.correct for r in ans), len(ans)),
        "hallucination_rate": _ratio(sum(r.hallucinated for r in answered), len(answered)),
        "abstention_precision": _ratio(sum(not r.answerable for r in abstained), len(abstained)),
        "abstention_recall": _ratio(sum(r.abstained for r in unans), len(unans)),
        "false_abstention_rate": _ratio(sum(r.abstained for r in ans), len(ans)),
        "citation_precision": _ratio(sum(r.citation_hits for r in answered),
                                     sum(r.citations for r in answered)),
        "retrieval_recall": _ratio(sum(bool(r.retrieval_hit) for r in with_gold), len(with_gold)),
        "mean_faithfulness": _ratio(sum(r.faithfulness for r in answered), len(answered)),
        "mean_latency_ms": _ratio(sum(r.ms for r in rs), len(rs)),
    }


def format_report(reports: list[EvalReport]) -> str:
    """Side-by-side markdown table of several reports."""
    keys = list(reports[0].metrics)
    lower_better = {"hallucination_rate", "false_abstention_rate", "mean_latency_ms"}
    head = "| metric | " + " | ".join(r.name for r in reports) + " |"
    sep = "|---|" + "---|" * len(reports)
    rows = []
    for k in keys:
        arrow = " ↓" if k in lower_better else " ↑"
        rows.append(f"| {k}{arrow} | " + " | ".join(f"{r.metrics[k]:.3f}" for r in reports) + " |")
    return "\n".join([head, sep, *rows])
