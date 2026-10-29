from dataclasses import fields
from pathlib import Path

import pytest

from omnisim.config import Settings

SETTING_NAMES = {
    "NEBIUS_API_KEY",
    "NEBIUS_BASE_URL",
    "NEBIUS_FAST_MODEL",
    "NEBIUS_DEEP_MODEL",
    "NEBIUS_ULTRA_MODEL",
    "TAVILY_API_KEY",
    "OMNISIM_DAILY_TOKEN_CAP",
    "OMNISIM_DEMO_MODE",
    "OMNISIM_API_CORS_ORIGINS",
    "OMNISIM_API_POST_RATE_LIMIT",
    "OMNISIM_API_OTHER_RATE_LIMIT",
    "OMNISIM_API_MAX_CONCURRENT_CAMPAIGNS",
    "OMNISIM_API_MAX_QUEUE",
    "OMNISIM_API_MAX_BUDGET",
    "OMNISIM_API_MAX_BODY_BYTES",
    "OMNISIM_TRUST_PROXY_HEADERS",
    "OMNISIM_API_QUICK_MIN_TOKENS",
    "OMNISIM_API_FULL_MIN_TOKENS",
    "OMNISIM_API_FULL_VERIFY_MIN_TOKENS",
    "OMNISIM_API_QUICK_TIMEOUT_S",
    "OMNISIM_API_FULL_TIMEOUT_S",
    "OMNISIM_API_FULL_VERIFY_TIMEOUT_S",
    "OMNISIM_API_MODEL_RESOLUTION_TIMEOUT_S",
    "OMNISIM_API_DEMO_RUN",
}


def test_settings_use_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.NEBIUS_API_KEY == ""
    assert settings.NEBIUS_BASE_URL == ""
    assert settings.NEBIUS_FAST_MODEL == ""
    assert settings.NEBIUS_DEEP_MODEL == ""
    assert settings.NEBIUS_ULTRA_MODEL == ""
    assert settings.TAVILY_API_KEY == ""
    assert settings.OMNISIM_DAILY_TOKEN_CAP == 400000
    assert settings.OMNISIM_DEMO_MODE is False
    assert settings.OMNISIM_API_CORS_ORIGINS == "http://localhost:5173"
    assert settings.OMNISIM_API_POST_RATE_LIMIT == "3/hour"
    assert settings.OMNISIM_API_OTHER_RATE_LIMIT == "120/minute"
    assert settings.OMNISIM_API_MAX_CONCURRENT_CAMPAIGNS == 1
    assert settings.OMNISIM_API_MAX_QUEUE == 3
    assert settings.OMNISIM_API_MAX_BUDGET == 100
    assert settings.OMNISIM_API_MAX_BODY_BYTES == 65536
    assert settings.OMNISIM_TRUST_PROXY_HEADERS is False
    assert settings.OMNISIM_API_QUICK_MIN_TOKENS == 60000
    assert settings.OMNISIM_API_FULL_MIN_TOKENS == 200000
    assert settings.OMNISIM_API_FULL_VERIFY_MIN_TOKENS == 250000
    assert settings.OMNISIM_API_QUICK_TIMEOUT_S == 180
    assert settings.OMNISIM_API_FULL_TIMEOUT_S == 1200
    assert settings.OMNISIM_API_FULL_VERIFY_TIMEOUT_S == 3600
    assert settings.OMNISIM_API_MODEL_RESOLUTION_TIMEOUT_S == 10
    assert settings.OMNISIM_API_DEMO_RUN == "hero"


def test_settings_read_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "NEBIUS_API_KEY": "nebius-test-key",
        "NEBIUS_BASE_URL": "https://api.example.invalid/v1/",
        "NEBIUS_FAST_MODEL": "fast-test-model",
        "NEBIUS_DEEP_MODEL": "deep-test-model",
        "NEBIUS_ULTRA_MODEL": "ultra-test-model",
        "TAVILY_API_KEY": "tavily-test-key",
        "OMNISIM_DAILY_TOKEN_CAP": "12345",
        "OMNISIM_DEMO_MODE": "yes",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = Settings.from_env()

    assert settings.NEBIUS_API_KEY == values["NEBIUS_API_KEY"]
    assert settings.NEBIUS_BASE_URL == values["NEBIUS_BASE_URL"]
    assert settings.NEBIUS_FAST_MODEL == values["NEBIUS_FAST_MODEL"]
    assert settings.NEBIUS_DEEP_MODEL == values["NEBIUS_DEEP_MODEL"]
    assert settings.NEBIUS_ULTRA_MODEL == values["NEBIUS_ULTRA_MODEL"]
    assert settings.TAVILY_API_KEY == values["TAVILY_API_KEY"]
    assert settings.OMNISIM_DAILY_TOKEN_CAP == 12345
    assert settings.OMNISIM_DEMO_MODE is True


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        (
            "OMNISIM_DAILY_TOKEN_CAP",
            "not-an-int",
            "OMNISIM_DAILY_TOKEN_CAP must be an integer",
        ),
        ("OMNISIM_DEMO_MODE", "sometimes", "OMNISIM_DEMO_MODE must be a boolean"),
        (
            "OMNISIM_API_MAX_CONCURRENT_CAMPAIGNS",
            "0",
            "OMNISIM_API_MAX_CONCURRENT_CAMPAIGNS must be positive",
        ),
        ("OMNISIM_API_MAX_BODY_BYTES", "0", "OMNISIM_API_MAX_BODY_BYTES must be positive"),
    ],
)
def test_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        Settings.from_env()


def test_settings_repr_hides_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "nebius-secret-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-secret-value")

    rendered = repr(Settings.from_env())

    assert "nebius-secret-value" not in rendered
    assert "tavily-secret-value" not in rendered


def test_env_example_names_match_settings_fields() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env.example"
    names = {
        line.partition("=")[0]
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    setting_fields = {item.name for item in fields(Settings)}

    assert names == SETTING_NAMES
    assert setting_fields == SETTING_NAMES
