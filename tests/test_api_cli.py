import json

import pytest

from hprag.cli import main


def test_cli_ingest_then_ask(tmp_path, corpus_path, capsys):
    idx = str(tmp_path / "idx")
    main(["ingest", str(corpus_path), "--index", idx, "--log-level", "ERROR"])
    main(["ask", "Who is the chief scientist?", "--index", idx, "--json", "--log-level", "ERROR"])
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    assert "Okafor" in payload["answer"] and payload["abstained"] is False


def test_api(rag):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from hprag.api.server import create_app

    client = TestClient(create_app(rag))
    assert client.get("/health").json()["chunks"] > 0
    r = client.post("/ask", json={"question": "How often do resupply missions arrive?"})
    assert r.status_code == 200 and "60 days" in r.json()["answer"]
    assert r.headers["X-Trace-Id"] == r.json()["trace_id"]
    r = client.post("/ingest", json={"documents": [{"id": "x", "text": "The vault code is 4417."}]})
    assert r.json()["chunks"] == 1
    assert "4417" in client.post("/ask", json={"question": "What is the vault code?"}).json()["answer"]


def test_ui_and_examples_are_served(rag):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from hprag.api.server import create_app

    client = TestClient(create_app(rag))
    page = client.get("/")
    assert page.status_code == 200 and "Hallucination-Proof RAG" in page.text
    groups = client.get("/examples").json()["groups"]
    assert {g["title"] for g in groups} >= {"Answerable", "Not in the documents"}
    assert client.get("/health").json()["verifier"] == "lexical"


def test_sample_questions_behave_as_labelled(rag):
    """Every sample shown in the UI must do what its group promises."""
    from hprag.api.server import STATIC

    for g in json.loads((STATIC / "examples.json").read_text())["groups"]:
        for q in g["questions"]:
            a = rag.ask(q)
            outcome = "blocked" if a.blocked_by else "refuse" if a.abstained else "answer"
            assert outcome == g["expect"], (q, outcome, a.abstain_reason)


def test_add_documents_keeps_existing_corpus(rag):
    from hprag.types import Document

    before = len(rag.retriever.chunks)
    rag.add_documents([Document("cafe", "The cafeteria opens at 7:30 every morning.")])
    assert len(rag.retriever.chunks) == before + 1
    assert "7:30" in rag.ask("When does the cafeteria open?").answer
    assert not rag.ask("How often do resupply missions arrive?").abstained  # old docs still there
    rag.add_documents([Document("cafe", "The cafeteria opens at 8:00 every morning.")])
    assert len(rag.retriever.chunks) == before + 1  # same id replaces, doesn't duplicate
