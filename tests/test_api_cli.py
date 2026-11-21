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
