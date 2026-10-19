import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    NEBIUS_API_KEY: str = field(default="", repr=False)
    NEBIUS_BASE_URL: str = ""
    NEBIUS_FAST_MODEL: str = ""
    NEBIUS_DEEP_MODEL: str = ""
    NEBIUS_ULTRA_MODEL: str = ""
    TAVILY_API_KEY: str = field(default="", repr=False)
    OMNISIM_DAILY_TOKEN_CAP: int = 400000
    OMNISIM_DEMO_MODE: bool = False
    OMNISIM_API_CORS_ORIGINS: str = "http://localhost:5173"
    OMNISIM_API_POST_RATE_LIMIT: str = "3/hour"
    OMNISIM_API_OTHER_RATE_LIMIT: str = "120/minute"
    OMNISIM_API_MAX_CONCURRENT_CAMPAIGNS: int = 1
    OMNISIM_API_MAX_QUEUE: int = 3
    OMNISIM_API_MAX_BUDGET: int = 100
    # UNMEASURED admission estimates. Replace after live runs with about 1.3x measured usage.
    OMNISIM_API_QUICK_MIN_TOKENS: int = 60000
    OMNISIM_API_FULL_MIN_TOKENS: int = 200000
    OMNISIM_API_FULL_VERIFY_MIN_TOKENS: int = 250000
    OMNISIM_API_QUICK_TIMEOUT_S: int = 180
    OMNISIM_API_FULL_TIMEOUT_S: int = 1200
    OMNISIM_API_FULL_VERIFY_TIMEOUT_S: int = 3600
    OMNISIM_API_MODEL_RESOLUTION_TIMEOUT_S: int = 10
    OMNISIM_API_DEMO_RUN: str = "hero"

    @classmethod
    def from_env(cls) -> "Settings":
        integer_defaults = {
            "OMNISIM_DAILY_TOKEN_CAP": 400000,
            "OMNISIM_API_MAX_CONCURRENT_CAMPAIGNS": 1,
            "OMNISIM_API_MAX_QUEUE": 3,
            "OMNISIM_API_MAX_BUDGET": 100,
            "OMNISIM_API_QUICK_MIN_TOKENS": 60000,
            "OMNISIM_API_FULL_MIN_TOKENS": 200000,
            "OMNISIM_API_FULL_VERIFY_MIN_TOKENS": 250000,
            "OMNISIM_API_QUICK_TIMEOUT_S": 180,
            "OMNISIM_API_FULL_TIMEOUT_S": 1200,
            "OMNISIM_API_FULL_VERIFY_TIMEOUT_S": 3600,
            "OMNISIM_API_MODEL_RESOLUTION_TIMEOUT_S": 10,
        }
        values: dict[str, int] = {}
        for name, default in integer_defaults.items():
            raw = os.environ.get(name, str(default))
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer") from exc
            if name == "OMNISIM_DAILY_TOKEN_CAP":
                if value < 0:
                    raise ValueError(f"{name} must not be negative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
            values[name] = value

        demo_mode = os.environ.get("OMNISIM_DEMO_MODE", "false").strip().lower()
        if demo_mode in {"true", "1", "yes", "on"}:
            demo = True
        elif demo_mode in {"false", "0", "no", "off"}:
            demo = False
        else:
            raise ValueError("OMNISIM_DEMO_MODE must be a boolean")

        return cls(
            NEBIUS_API_KEY=os.environ.get("NEBIUS_API_KEY", ""),
            NEBIUS_BASE_URL=os.environ.get("NEBIUS_BASE_URL", ""),
            NEBIUS_FAST_MODEL=os.environ.get("NEBIUS_FAST_MODEL", ""),
            NEBIUS_DEEP_MODEL=os.environ.get("NEBIUS_DEEP_MODEL", ""),
            NEBIUS_ULTRA_MODEL=os.environ.get("NEBIUS_ULTRA_MODEL", ""),
            TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY", ""),
            OMNISIM_DEMO_MODE=demo,
            OMNISIM_API_CORS_ORIGINS=os.environ.get(
                "OMNISIM_API_CORS_ORIGINS", "http://localhost:5173"
            ),
            OMNISIM_API_POST_RATE_LIMIT=os.environ.get("OMNISIM_API_POST_RATE_LIMIT", "3/hour"),
            OMNISIM_API_OTHER_RATE_LIMIT=os.environ.get(
                "OMNISIM_API_OTHER_RATE_LIMIT", "120/minute"
            ),
            OMNISIM_API_DEMO_RUN=os.environ.get("OMNISIM_API_DEMO_RUN", "hero"),
            **values,
        )
