import json
from types import SimpleNamespace
from typing import Any, cast

import httpx
import openai
import pytest
from pydantic import BaseModel

from omnisim.llm import models, nebius
from omnisim.llm.budget import BudgetExceeded, LLMError, Usage, request_slot
from omnisim.llm.call_log import write_call_log


class Answer(BaseModel):
    answer: int


def response(
    content: str | None,
    *,
    usage: tuple[int, int] | None = (2, 3),
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    response_usage = (
        SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=response_usage,
    )


class FakeClient:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict] = []
        self.model_requests = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.models = SimpleNamespace(list=self.list_models)

    def list_models(self) -> list[SimpleNamespace]:
        self.model_requests += 1
        return [SimpleNamespace(id="nvidia/nemotron-test")]

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


@pytest.fixture
def setup_chat(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("NEBIUS_FAST_MODEL", "nvidia/nemotron-test")
    monkeypatch.setattr(nebius, "check", lambda role, **kwargs: None)
    monkeypatch.setattr(nebius, "record", lambda role, prompt, completion, **kwargs: None)
    monkeypatch.setattr(
        nebius,
        "write_call_log",
        lambda entry: write_call_log(entry, path=tmp_path / "calls.jsonl"),
    )
    monkeypatch.setattr(
        nebius,
        "request_slot",
        lambda role, *, cap: request_slot(role, cap=cap, db_path=tmp_path / "usage.sqlite3"),
    )
    monkeypatch.setattr(
        models,
        "write_call_log",
        lambda entry: write_call_log(entry, path=tmp_path / "calls.jsonl"),
    )
    return tmp_path / "calls.jsonl"


def run_chat(client: FakeClient, *, retries: int = 1, max_tokens: int = 8):
    return nebius.chat_json(
        "fast",
        "system prompt",
        "user prompt",
        Answer,
        temperature=0,
        max_tokens=max_tokens,
        retries=retries,
        run_id="run-1",
    )


def test_empty_content_uses_reasoning_content(monkeypatch: pytest.MonkeyPatch, setup_chat) -> None:
    client = FakeClient([response("", reasoning_content='Thinking... {"answer": 4}')])
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    parsed, _ = run_chat(client)

    assert parsed.answer == 4


def test_reasoning_uses_last_schema_valid_object(
    monkeypatch: pytest.MonkeyPatch, setup_chat
) -> None:
    client = FakeClient(
        [response("", reasoning_content='draft {"answer": 1}\nfinal {"answer": 9}')]
    )
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    parsed, _ = run_chat(client)

    assert parsed.answer == 9


def test_content_checks_candidates_from_first_to_last(
    monkeypatch: pytest.MonkeyPatch, setup_chat
) -> None:
    client = FakeClient([response('{"wrong": 1} then {"answer": 6}')])
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    parsed, _ = run_chat(client)

    assert parsed.answer == 6


def test_reasoning_tries_earlier_candidate_if_last_fails_schema(
    monkeypatch: pytest.MonkeyPatch, setup_chat
) -> None:
    client = FakeClient([response("", reasoning_content='draft {"answer": 6} final {"wrong": 1}')])
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    parsed, _ = run_chat(client)

    assert parsed.answer == 6


def test_invalid_json_is_repaired(monkeypatch: pytest.MonkeyPatch, setup_chat) -> None:
    client = FakeClient([response("not json"), response('{"answer": 3}')])
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    parsed, _ = run_chat(client)

    assert parsed.answer == 3
    assert "Fix this error" in client.requests[1]["messages"][1]["content"]


def test_schema_error_is_repaired(monkeypatch: pytest.MonkeyPatch, setup_chat) -> None:
    client = FakeClient([response('{"wrong": 1}'), response('{"answer": 3}')])
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    parsed, _ = run_chat(client)

    assert parsed.answer == 3


def test_invalid_output_raises_after_retries(monkeypatch: pytest.MonkeyPatch, setup_chat) -> None:
    client = FakeClient([response("bad"), response("still bad")])
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    with pytest.raises(nebius.LLMOutputError) as exc:
        run_chat(client, retries=1)

    assert len(client.requests) == 2
    assert isinstance(exc.value.last_error, ValueError)


def test_length_finish_doubles_max_tokens(monkeypatch: pytest.MonkeyPatch, setup_chat) -> None:
    client = FakeClient([response("{", finish_reason="length"), response('{"answer": 3}')])
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    run_chat(client)

    assert [request["max_tokens"] for request in client.requests] == [8, 16]


def test_transport_timeout_retries_once(monkeypatch: pytest.MonkeyPatch, setup_chat) -> None:
    timeout = openai.APITimeoutError(
        request=cast(Any, httpx.Request("POST", "https://test.invalid"))
    )
    client = FakeClient([timeout, response('{"answer": 7}')])
    monkeypatch.setattr(nebius, "get_client", lambda: client)
    monkeypatch.setattr(nebius.time, "sleep", lambda delay: None)

    parsed, _ = run_chat(client)

    assert parsed.answer == 7
    assert len(client.requests) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_do_not_retry(
    monkeypatch: pytest.MonkeyPatch, setup_chat, status: int
) -> None:
    request = httpx.Request("POST", "https://test.invalid")
    error = openai.APIStatusError(
        "request failed", response=cast(Any, httpx.Response(status, request=request)), body={}
    )
    client = FakeClient([error, response('{"answer": 7}')])
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    with pytest.raises(nebius.LLMUnavailable):
        run_chat(client)

    assert len(client.requests) == 1


@pytest.mark.parametrize("status", [429, 503])
def test_rate_limit_and_server_errors_retry(
    monkeypatch: pytest.MonkeyPatch, setup_chat, status: int
) -> None:
    request = httpx.Request("POST", "https://test.invalid")
    error = openai.APIStatusError(
        "request failed", response=cast(Any, httpx.Response(status, request=request)), body={}
    )
    client = FakeClient([error, response('{"answer": 7}')])
    monkeypatch.setattr(nebius, "get_client", lambda: client)
    monkeypatch.setattr(nebius.time, "sleep", lambda delay: None)

    parsed, _ = run_chat(client)

    assert parsed.answer == 7
    assert len(client.requests) == 2


def test_budget_exceeded_sends_no_request(monkeypatch: pytest.MonkeyPatch, setup_chat) -> None:
    client = FakeClient([response('{"answer": 7}')])
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    def reject_budget(role: str, **kwargs: object) -> None:
        raise BudgetExceeded("cap")

    monkeypatch.setattr(nebius, "check", reject_budget)

    with pytest.raises(BudgetExceeded):
        run_chat(client)

    assert client.requests == []
    assert client.model_requests == 0


def test_missing_nemotron_raises_llm_error_before_completion(
    monkeypatch: pytest.MonkeyPatch, setup_chat
) -> None:
    client = FakeClient([response('{"answer": 7}')])
    client.models.list = lambda: [SimpleNamespace(id="other/model")]
    monkeypatch.delenv("NEBIUS_FAST_MODEL", raising=False)
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    with pytest.raises(LLMError) as exc:
        run_chat(client)

    assert isinstance(exc.value, nebius.LLMUnavailable)
    assert client.requests == []


def test_usage_is_summed_across_attempts(monkeypatch: pytest.MonkeyPatch, setup_chat) -> None:
    client = FakeClient([response("bad", usage=(2, 3)), response('{"answer": 7}', usage=(4, 5))])
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    _, usage = run_chat(client)

    assert usage == Usage(prompt_tokens=6, completion_tokens=8)


def test_missing_usage_records_zero_and_logs_warning(
    monkeypatch: pytest.MonkeyPatch, setup_chat, caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeClient([response('{"answer": 7}', usage=None)])
    records: list[tuple[int, int]] = []
    monkeypatch.setattr(nebius, "get_client", lambda: client)
    monkeypatch.setattr(
        nebius,
        "record",
        lambda role, prompt, completion, **kwargs: records.append((prompt, completion)),
    )

    with caplog.at_level("WARNING", logger="omnisim.llm.nebius"):
        parsed, usage = run_chat(client)

    assert parsed.answer == 7
    assert usage == Usage()
    assert records == [(0, 0)]
    assert "llm response has no usage data" in caplog.text


def test_attempt_logs_have_status_and_no_prompt_or_key(
    monkeypatch: pytest.MonkeyPatch, setup_chat
) -> None:
    path = setup_chat
    client = FakeClient([response("bad"), response('{"answer": 7}')])
    monkeypatch.setattr(nebius, "get_client", lambda: client)

    run_chat(client)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["ok"] for row in rows] == [True, False, True]
    text = path.read_text(encoding="utf-8")
    assert "system prompt" not in text
    assert "user prompt" not in text
    assert "secret" not in text


def test_client_disables_sdk_retries_and_uses_fixed_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def make_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(nebius, "_client", None)
    monkeypatch.setattr(nebius, "OpenAI", make_client)
    monkeypatch.setenv("NEBIUS_API_KEY", "fake-key")
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://tokenfactory.invalid/v1/")

    nebius.get_client()

    assert captured["max_retries"] == 0
    assert captured["timeout"] == nebius.TIMEOUT_S
