from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from omnisim.agents.adversary import AdversaryResponse
from omnisim.llm.benchmark_cache import (
    BenchmarkLLMCache,
    ReplayCacheError,
    cache_size,
    prompt_file_hash,
)
from omnisim.llm.budget import Usage


def _live_calls(response: AdversaryResponse):
    calls: list[tuple[object, ...]] = []

    def call(*args: object, **kwargs: object) -> tuple[AdversaryResponse, Usage]:
        calls.append((*args, kwargs))
        return response, Usage(11, 13)

    return calls, call


def _call(cache: BenchmarkLLMCache) -> tuple[AdversaryResponse, Usage]:
    return cache(
        "fast",
        "system prompt secret",
        "user prompt secret",
        AdversaryResponse,
        temperature=0.7,
        max_tokens=4096,
    )


def test_cache_hit_replays_parsed_result_and_usage_without_chat_call(tmp_path: Path) -> None:
    expected = AdversaryResponse(hypothesis="cached", scenarios=[{"push_force_n": 5}])
    calls, chat_json = _live_calls(expected)
    live = BenchmarkLLMCache(
        tmp_path,
        salt="run-a",
        prompt_hash="p" * 64,
        chat_json=chat_json,
        model_resolver=lambda role: "nvidia/nemotron-a",
    )
    first, usage = _call(live)
    replay = BenchmarkLLMCache(
        tmp_path,
        salt="run-a",
        prompt_hash="p" * 64,
        replay_only=True,
        model_resolver=lambda role: pytest.fail("replay resolved a live model"),
    )

    second, cached_usage = _call(replay)

    assert first == second == expected
    assert usage == cached_usage == Usage(11, 13)
    assert len(calls) == 1
    assert replay.last_cache_hit
    assert replay.last_model_id == "nvidia/nemotron-a"
    entry_text = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert "system prompt secret" not in entry_text
    assert "user prompt secret" not in entry_text
    assert set(json.loads(entry_text)) == {
        "parsed",
        "prompt_hash",
        "resolved_model_id",
        "salt",
        "usage",
    }
    assert cache_size(tmp_path) > 0


def test_cache_key_changes_with_request_settings_and_model(tmp_path: Path) -> None:
    cache = BenchmarkLLMCache(
        tmp_path,
        salt="run-a",
        prompt_hash="p" * 64,
        model_resolver=lambda role: "nvidia/nemotron-a",
    )
    base = cache.cache_path("fast", "sys", "user", AdversaryResponse, 0.7, 4096, "model-a")

    assert base != cache.cache_path(
        "fast", "sys", "changed", AdversaryResponse, 0.7, 4096, "model-a"
    )
    assert base != cache.cache_path("fast", "sys", "user", AdversaryResponse, 0.8, 4096, "model-a")
    assert base != cache.cache_path("fast", "sys", "user", AdversaryResponse, 0.7, 2048, "model-a")
    assert base != cache.cache_path("fast", "sys", "user", AdversaryResponse, 0.7, 4096, "model-b")
    other_salt = BenchmarkLLMCache(tmp_path, salt="run-b", prompt_hash="p" * 64)
    assert base != other_salt.cache_path(
        "fast", "sys", "user", AdversaryResponse, 0.7, 4096, "model-a"
    )
    other_hash = BenchmarkLLMCache(tmp_path, salt="run-a", prompt_hash="q" * 64)
    assert base != other_hash.cache_path(
        "fast", "sys", "user", AdversaryResponse, 0.7, 4096, "model-a"
    )


def test_corrupt_entry_is_a_miss_and_is_replaced(tmp_path: Path) -> None:
    expected = AdversaryResponse(hypothesis="fresh", scenarios=[])
    calls, chat_json = _live_calls(expected)
    cache = BenchmarkLLMCache(
        tmp_path,
        salt="run-a",
        prompt_hash="p" * 64,
        chat_json=chat_json,
        model_resolver=lambda role: "nvidia/nemotron-a",
    )
    path = cache.cache_path(
        "fast",
        "system prompt secret",
        "user prompt secret",
        AdversaryResponse,
        0.7,
        4096,
        "nvidia/nemotron-a",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")

    result, _ = _call(cache)

    assert result == expected
    assert len(calls) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["parsed"]["hypothesis"] == "fresh"
    assert not list(tmp_path.glob("*.tmp"))


def test_replay_only_fails_on_miss(tmp_path: Path) -> None:
    replay = BenchmarkLLMCache(
        tmp_path,
        salt="missing",
        prompt_hash="p" * 64,
        replay_only=True,
        model_resolver=lambda role: pytest.fail("replay resolved a live model"),
    )

    with pytest.raises(ReplayCacheError, match="cache miss"):
        _call(replay)


def test_replay_only_rejects_conflicting_model_ids(tmp_path: Path) -> None:
    expected = AdversaryResponse(hypothesis="x", scenarios=[])
    for model_id in ("nvidia/nemotron-a", "nvidia/nemotron-b"):
        _, chat_json = _live_calls(expected)
        live = BenchmarkLLMCache(
            tmp_path,
            salt="run-a",
            prompt_hash="p" * 64,
            chat_json=chat_json,
            model_resolver=lambda role, value=model_id: value,
        )
        _call(live)
    replay = BenchmarkLLMCache(
        tmp_path,
        salt="run-a",
        prompt_hash="p" * 64,
        replay_only=True,
    )

    with pytest.raises(ReplayCacheError, match="multiple model IDs"):
        _call(replay)


def test_prompt_hash_is_sha256_of_file(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("prompt", encoding="utf-8")

    assert prompt_file_hash(prompt) == hashlib.sha256(b"prompt").hexdigest()
