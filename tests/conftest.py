"""Shared fixtures. Every test runs offline: no API keys, no model downloads."""

from pathlib import Path

import pytest

from hprag import RAGPipeline, Settings

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
EVAL_SET = Path(__file__).resolve().parents[1] / "data" / "eval" / "halcyon_qa.jsonl"


@pytest.fixture(scope="session")
def corpus_path() -> Path:
    return CORPUS


@pytest.fixture()
def rag() -> RAGPipeline:
    p = RAGPipeline(Settings())
    p.ingest_path(str(CORPUS))
    return p
