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

    @classmethod
    def from_env(cls) -> "Settings":
        cap = os.environ.get("OMNISIM_DAILY_TOKEN_CAP", "400000")
        try:
            daily_cap = int(cap)
        except ValueError as exc:
            raise ValueError("OMNISIM_DAILY_TOKEN_CAP must be an integer") from exc

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
            OMNISIM_DAILY_TOKEN_CAP=daily_cap,
            OMNISIM_DEMO_MODE=demo,
        )
