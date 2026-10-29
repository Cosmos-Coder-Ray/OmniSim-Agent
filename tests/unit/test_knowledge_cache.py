import importlib
import json
from datetime import UTC, datetime

from omnisim.knowledge import cache, tavily_agent
from omnisim.knowledge.usage import get_knowledge_usage


class FakeClient:
    def __init__(self):
        self.search_calls = []
        self.extract_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {"results": [{"url": "https://docs.test/page", "title": "Manual", "score": 1}]}

    def extract(self, **kwargs):
        self.extract_calls.append(kwargs)
        return {
            "results": [
                {"url": url, "raw_content": f"Synthetic page for {url}"} for url in kwargs["urls"]
            ]
        }


def test_cache_key_includes_all_parameters(tmp_path) -> None:
    first = cache.cache_file("search", {"query": "robot", "depth": "advanced"}, cache_dir=tmp_path)
    second = cache.cache_file("search", {"query": "robot", "depth": "basic"}, cache_dir=tmp_path)
    assert first != second


def test_search_cache_hit_does_not_call_client(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    call_log = tmp_path / "calls.jsonl"
    params = {
        "query": "Panda",
        "depth": "advanced",
        "max_results": 20,
        "include_domains": ["docs.test"],
        "extract_depth": "advanced",
    }
    path = cache.cache_file("search", params, cache_dir=tmp_path / "cache")
    cache.write_cache(
        path, "search", params, {"results": [{"url": "https://docs.test/page", "score": 1}]}
    )

    monkeypatch.setattr(tavily_agent, "_client", lambda: (_ for _ in ()).throw(AssertionError()))
    results = tavily_agent._search(
        "Panda",
        ("docs.test",),
        cache_dir=tmp_path / "cache",
        call_log_path=call_log,
    )
    assert results[0]["url"] == "https://docs.test/page"
    assert get_knowledge_usage(path=call_log)["cache_hits"] == 1


def test_extract_fetches_only_missing_urls(tmp_path) -> None:
    client = FakeClient()
    cache_dir = tmp_path / "cache"
    first_url = "https://docs.test/one"
    params = {
        "query": None,
        "depth": None,
        "max_results": None,
        "include_domains": None,
        "extract_depth": "advanced",
        "url": first_url,
    }
    cache.write_cache(
        cache.cache_file("extract", params, cache_dir=cache_dir),
        "extract",
        params,
        {"content": "Synthetic cached page one"},
    )
    pages = tavily_agent._extract(
        [first_url, "https://docs.test/two"],
        cache_dir=cache_dir,
        call_log_path=tmp_path / "calls.jsonl",
        client=client,
    )
    assert pages[first_url] == "Synthetic cached page one"
    assert client.extract_calls == [
        {"urls": ["https://docs.test/two"], "extract_depth": "advanced", "timeout": 30.0}
    ]
    assert "Synthetic page for https://docs.test/two" in pages["https://docs.test/two"]


def test_search_filters_unapproved_domains_and_canonicalizes_urls(tmp_path) -> None:
    class SearchClient:
        def search(self, **kwargs):
            return {
                "results": [
                    {"url": "https://docs.test/manual/?tracking=1#section", "score": 0.8},
                    {"url": "https://evil.test/manual", "score": 1.0},
                ]
            }

    results = tavily_agent._search(
        "Panda",
        ("docs.test",),
        cache_dir=tmp_path / "cache",
        call_log_path=tmp_path / "calls.jsonl",
        client=SearchClient(),
    )
    assert results == [{"url": "https://docs.test/manual", "score": 0.8}]


def test_corrupt_cache_is_a_miss_and_logs_warning(tmp_path, caplog) -> None:
    path = cache.cache_file("search", {"query": "x"}, cache_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{", encoding="utf-8")
    assert cache.read_cache(path) is None
    assert "invalid tavily cache file" in caplog.text


def test_cache_write_replaces_file_atomically(tmp_path) -> None:
    path = tmp_path / "nested" / "result.json"
    cache.write_cache(path, "search", {"query": "synthetic"}, {"results": []})
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["op"] == "search"
    assert row["params"] == {"query": "synthetic"}
    assert row["schema_version"] == cache.SCHEMA_VERSION
    assert list(path.parent.iterdir()) == [path]


def test_import_with_empty_env_does_not_create_client(monkeypatch) -> None:
    for key in ("TAVILY_API_KEY", "NEBIUS_API_KEY", "NEBIUS_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tavily_agent, "_client", lambda: (_ for _ in ()).throw(AssertionError()))
    import omnisim.knowledge as knowledge

    assert hasattr(importlib.reload(knowledge), "build_spec")


def test_daily_usage_uses_utc_and_only_counts_live_calls(tmp_path) -> None:
    path = tmp_path / "calls.jsonl"
    from omnisim.knowledge.usage import log_call

    log_call(
        {"ts": "2026-10-16T01:00:00+00:00", "op": "search", "cache": "miss", "ok": True}, path=path
    )
    log_call(
        {"ts": "2026-10-16T02:00:00+00:00", "op": "extract", "cache": "hit", "ok": True}, path=path
    )
    log_call(
        {"ts": "2026-10-15T23:00:00+00:00", "op": "search", "cache": "miss", "ok": True}, path=path
    )
    usage = get_knowledge_usage(path=path, now=datetime(2026, 10, 16, 4, tzinfo=UTC))
    assert usage == {
        "date": "2026-10-16",
        "searches": 1,
        "extracts": 0,
        "cache_hits": 1,
        "cache_misses": 1,
    }
