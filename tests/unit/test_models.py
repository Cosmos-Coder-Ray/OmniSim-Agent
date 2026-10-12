from types import SimpleNamespace

import pytest

from omnisim.llm import models
from omnisim.llm.budget import LLMUnavailable
from omnisim.llm.models import _list_models, list_models, resolve_model


class FakeClient:
    def __init__(self, ids: list[str]) -> None:
        self.calls = 0
        self.models = SimpleNamespace(list=self._list)
        self.ids = ids

    def _list(self) -> list[SimpleNamespace]:
        self.calls += 1
        return [SimpleNamespace(id=item) for item in self.ids]


@pytest.fixture(autouse=True)
def clear_model_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _list_models.cache_clear()
    monkeypatch.setattr(models, "write_call_log", lambda entry: None)


def test_resolve_model_uses_configured_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEBIUS_FAST_MODEL", "nvidia/nemotron-custom")
    client = FakeClient(["nvidia/nemotron-custom", "nvidia/nemotron-lightning"])

    assert resolve_model(client, "fast") == "nvidia/nemotron-custom"


@pytest.mark.parametrize("configured", ["other/model", "other/nemotron-custom"])
def test_resolve_model_ignores_configured_non_nemotron(
    monkeypatch: pytest.MonkeyPatch, configured: str
) -> None:
    monkeypatch.setenv("NEBIUS_FAST_MODEL", configured)
    client = FakeClient([configured, "nvidia/nemotron-lightning"])

    assert resolve_model(client, "fast") == "nvidia/nemotron-lightning"


def test_resolve_model_uses_role_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEBIUS_DEEP_MODEL", raising=False)
    client = FakeClient(["nvidia/nemotron-nano", "nvidia/nemotron-super"])

    assert resolve_model(client, "deep") == "nvidia/nemotron-super"


def test_resolve_model_falls_back_when_configured_id_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEBIUS_FAST_MODEL", "nvidia/nemotron-unavailable")
    client = FakeClient(["nvidia/nemotron-lightning"])

    assert resolve_model(client, "fast") == "nvidia/nemotron-lightning"


def test_resolve_model_fails_without_nemotron(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEBIUS_FAST_MODEL", raising=False)

    with pytest.raises(LLMUnavailable, match="no Nemotron model"):
        resolve_model(FakeClient(["other/model"]), "fast")


def test_resolve_model_never_selects_non_nvidia_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEBIUS_FAST_MODEL", raising=False)
    client = FakeClient(["aaa/nemotron-lightning", "other/lightning", "nvidia/nemotron-other"])

    assert resolve_model(client, "fast") == "nvidia/nemotron-other"


def test_list_models_is_cached_per_client() -> None:
    client = FakeClient(["nvidia/nemotron-super"])

    assert list_models(client) == ["nvidia/nemotron-super"]
    assert list_models(client) == ["nvidia/nemotron-super"]
    assert client.calls == 1
