from hprag.config import RetrievalConfig
from hprag.ingest import SentenceChunker, load_directory
from hprag.retrieval import HashingEmbedder, HybridRetriever, rrf


def _retriever(corpus_path, **kw):
    chunks = SentenceChunker().chunk_all(load_directory(corpus_path))
    return HybridRetriever(RetrievalConfig(**kw)).build(chunks)


def test_rrf_rewards_agreement():
    fused = rrf({"a": [(1, 9.0), (2, 5.0)], "b": [(1, 0.9), (3, 0.5)]}, k=60)
    assert max(fused, key=fused.get) == 1


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(dim=256)
    a, b = e.embed(["solar arrays generate power"]), e.embed(["solar arrays generate power"])
    assert (a == b).all()
    assert abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5


def test_hybrid_finds_exact_entity(corpus_path):
    r = _retriever(corpus_path)
    hits = r.retrieve("What does the Aquifer loop recover?")
    assert hits[0].chunk.doc_id == "power_and_life_support"
    assert {"bm25", "dense", "rrf", "rerank"} <= set(hits[0].scores)


def test_index_roundtrip(tmp_path, corpus_path):
    r = _retriever(corpus_path)
    r.save(tmp_path)
    r2 = HybridRetriever.load(tmp_path)
    q = "Who is the chief scientist?"
    assert [h.chunk.chunk_id for h in r.retrieve(q)] == [h.chunk.chunk_id for h in r2.retrieve(q)]


def test_load_refuses_mismatched_embedder(tmp_path, corpus_path):
    import pytest

    _retriever(corpus_path).save(tmp_path)
    with pytest.raises(ValueError):
        HybridRetriever.load(tmp_path, RetrievalConfig(embedder="sentence-transformers"))
