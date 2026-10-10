import json

from omnisim.llm.call_log import write_call_log


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
