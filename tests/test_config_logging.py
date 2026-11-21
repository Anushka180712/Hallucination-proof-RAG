import io
import json
import logging

import pytest

from hprag import Settings
from hprag.logging_utils import configure_logging, get_logger, new_trace_id, stage


def test_env_overrides():
    s = Settings.load(env={"HPRAG_GUARD_MIN_FAITHFULNESS": "0.9", "HPRAG_LLM_PROVIDER": "anthropic",
                           "HPRAG_GUARD_CONTRADICTION_ABORTS": "false", "HPRAG_LOG_LEVEL": "DEBUG"})
    assert s.guard.min_faithfulness == 0.9
    assert s.llm.provider == "anthropic"
    assert s.guard.contradiction_aborts is False
    assert s.log_level == "DEBUG"


def test_toml_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[guard]\nmin_faithfulnes = 0.9\n")
    with pytest.raises(ValueError):
        Settings.load(p, env={})


def test_json_logs_carry_trace_and_fields():
    buf = io.StringIO()
    configure_logging("INFO", "json", stream=buf)
    tid = new_trace_id()
    with stage("retrieve", get_logger("t")) as out:
        out["hits"] = 3
    line = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert line["trace_id"] == tid and line["event"] == "retrieve.end" and line["hits"] == 3
    assert "ms" in line
    logging.getLogger("hprag").handlers.clear()
