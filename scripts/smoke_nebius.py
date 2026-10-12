import json
import sys
from time import perf_counter

import openai
from pydantic import BaseModel

from omnisim.config import Settings
from omnisim.llm.budget import get_usage_summary
from omnisim.llm.models import list_models, resolve_model
from omnisim.llm.nebius import LLMError, chat_json, get_client


class SmokeReply(BaseModel):
    ok: bool


def main() -> int:
    settings = Settings.from_env()
    if not settings.NEBIUS_API_KEY:
        print("NEBIUS_API_KEY is required")
        return 1

    try:
        client = get_client()
        models = list_models(client)
        print(f"models: {', '.join(models)}")
        for role in ("fast", "deep"):
            model = resolve_model(client, role)
            started = perf_counter()
            _, usage = chat_json(
                role,
                "Reply with a JSON object containing ok set to true.",
                "Return the requested JSON.",
                SmokeReply,
                temperature=0,
                max_tokens=256,
                run_id="smoke",
            )
            latency_s = perf_counter() - started
            print(
                f"{role}: model={model} latency_s={latency_s:.3f} "
                f"prompt_tokens={usage.prompt_tokens} "
                f"completion_tokens={usage.completion_tokens}"
            )
        print(json.dumps(get_usage_summary(), sort_keys=True))
    except (LLMError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 1
    except openai.OpenAIError as exc:
        print(type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
