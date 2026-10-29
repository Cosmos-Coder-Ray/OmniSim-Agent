import json
import logging

from omnisim.audit_log import JsonFormatter, bind_log_context, reset_log_context
from omnisim.llm.call_log import provider_host, provider_hosts_for_run, write_call_log


def test_call_log_keeps_fields_and_omits_secrets_and_prompts(tmp_path) -> None:
    path = tmp_path / "nested" / "calls.jsonl"
    write_call_log(
        {
            "role": "fast",
            "model": "fake-model",
            "attempt": 1,
            "ok": False,
            "error": "invalid_json",
            "api_key": "secret",
            "prompt": "private prompt",
        },
        path=path,
    )

    line = path.read_text(encoding="utf-8")
    row = json.loads(line)
    assert row["role"] == "fast"
    assert row["ok"] is False
    assert row["ts"]
    assert "secret" not in line
    assert "private prompt" not in line


def test_provider_host_keeps_only_a_hostname() -> None:
    assert provider_host("https://user:secret@api.test.invalid/v1") == "api.test.invalid"
    assert provider_host("http://192.0.2.1:8080") is None


def test_provider_hosts_for_run_reads_matching_safe_hosts(tmp_path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_text(
        '{"run_id":"run-a","provider_host":"api.test.invalid"}\n'
        '{"run_id":"run-a","provider_host":"https://user:secret@other.test.invalid/v1"}\n'
        '{"run_id":"run-b","provider_host":"ignored.test.invalid"}\n'
        "invalid\n",
        encoding="utf-8",
    )

    assert provider_hosts_for_run("run-a", path=path) == [
        "api.test.invalid",
        "other.test.invalid",
    ]


def test_json_formatter_whitelists_and_redacts_context_and_fields() -> None:
    tokens = bind_log_context(request_id="a" * 32, run_id="b" * 32)
    try:
        record = logging.LogRecord(
            "omnisim.audit", logging.ERROR, "test.py", 1, "sentinel prompt", (), None
        )
        record.event = "provider_call"
        record.provider_host = "https://user:pass@provider.example/v1?key=sentinel-secret"
        record.error_type = "RuntimeError"
        record.unlisted = "sentinel-body"
        row = json.loads(JsonFormatter().format(record))
    finally:
        reset_log_context(tokens)
    serialized = json.dumps(row)
    assert row["run_id"] == "b" * 32
    assert row["provider_host"] == "provider.example"
    assert row["error_type"] == "RuntimeError"
    assert "sentinel" not in serialized and "user" not in serialized and "pass" not in serialized
