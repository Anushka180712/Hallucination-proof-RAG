from hprag.ingest import SentenceChunker, load_directory
from hprag.types import Document


def test_chunks_respect_sentences_and_offsets():
    text = "One fact here. " * 30 + "Final sentence."
    doc = Document("d", text)
    chunks = SentenceChunker(max_chars=120, overlap_sentences=1).chunk(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert doc.text[c.start:c.end] == c.text          # offsets are exact
        assert c.text.endswith(".")                        # never cut mid-sentence
        assert len(c.text) <= 120


def test_overlap_repeats_last_sentence():
    sents = [f"Sentence number {w} is here." for w in ("one", "two", "three", "four", "five")]
    chunks = SentenceChunker(max_chars=60, overlap_sentences=1).chunk(Document("d", " ".join(sents)))
    assert len(chunks) >= 2
    assert chunks[1].text.startswith(chunks[0].text.split(". ")[-1].rstrip("."))


def test_long_sentence_is_kept_whole():
    long = "word " * 40 + "end."
    chunks = SentenceChunker(max_chars=60).chunk(Document("d", long))
    assert len(chunks) == 1


def test_load_directory(corpus_path):
    docs = load_directory(corpus_path)
    assert len(docs) == 6
    assert all(d.metadata["title"] for d in docs)
