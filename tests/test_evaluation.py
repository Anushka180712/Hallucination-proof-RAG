from conftest import EVAL_SET

from hprag.evaluation import evaluate, load_dataset
from hprag.evaluation.baseline import naive_pipeline


def test_guarded_pipeline_beats_naive_baseline(rag):
    items = load_dataset(EVAL_SET)
    guarded = evaluate(rag, items, "guarded").metrics
    naive = evaluate(naive_pipeline(rag.s, retriever=rag.retriever), items, "naive").metrics
    assert guarded["hallucination_rate"] < naive["hallucination_rate"]
    assert guarded["hallucination_rate"] <= 0.05
    assert guarded["abstention_recall"] >= 0.9
    assert guarded["answer_accuracy"] >= 0.85
