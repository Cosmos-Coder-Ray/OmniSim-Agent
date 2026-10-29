# Provider integration feedback

The notes below describe behavior handled by the current integration and tests. `docs/PROGRESS.md` records that no live Nebius or Tavily campaign was run, so they should not be read as a report of a confirmed provider outage.

## Nemotron response content

**What we did**  
We send structured requests through the Nebius Token Factory OpenAI-compatible API and parse the assistant response as JSON (`src/omnisim/llm/nebius.py`).

**What happened**  
The response contract can place text in `reasoning_content` while `content` is empty. The parser also needs to handle JSON preceded by `<think>` material. These cases have unit coverage in `tests/unit/test_chat_json.py` and `tests/unit/test_json_utils.py`.

**What we expected / suggestion**  
It helps if clients can reliably distinguish the user-facing structured answer from internal reasoning, and if the structured answer remains available through the normal content field.

**How we handled it in OmniSim-Agent**  
We use `reasoning_content` as a fallback when `content` is empty and strip `<think>` blocks before extracting JSON (`src/omnisim/llm/json_utils.py`). Parsed values still pass Pydantic validation; reasoning text is not trusted as a validated answer.

## Truncated reasoning responses

**What we did**  
We set an output-token limit for each role request and inspect the completion finish reason (`src/omnisim/llm/nebius.py`).

**What happened**  
When reasoning uses the available output budget, the request can finish with `finish_reason="length"` before the JSON answer is complete. The behavior is covered by `test_length_finish_doubles_max_tokens` in `tests/unit/test_chat_json.py`.

**What we expected / suggestion**  
Structured-output callers need enough output budget for both reasoning and the final object, with a clear truncation signal when the response does not fit.

**How we handled it in OmniSim-Agent**  
On a `length` finish, the next output attempt doubles `max_tokens`, with an 8192-token cap. A separate repair attempt handles invalid or incomplete JSON; all completion attempts still pass token-budget admission (`docs/DECISIONS.md`, initial LLM decisions and Phase 10B).

## Model ID discovery

**What we did**  
At startup or before a completion, the model resolver calls `GET /v1/models` through the OpenAI-compatible client (`src/omnisim/llm/models.py`).

**What happened**  
Configured model IDs must be present in the returned model list and identify an NVIDIA Nemotron model. The call log has model-list records with no token charge (`data/llm_calls.jsonl`; `docs/DECISIONS.md`, initial LLM decisions).

**What we expected / suggestion**  
The model-list endpoint should expose stable IDs that match the IDs accepted by the completion endpoint. Clear model-list errors help distinguish unavailable configuration from an invalid ID.

**How we handled it in OmniSim-Agent**  
We select a configured Nemotron ID when it appears in the list; otherwise we search the available Nemotron IDs by role preference and raise `LLMUnavailable` if none match (`src/omnisim/llm/models.py`).

## Token admission and accounting

**What we did**  
We check a daily token cap before each completion attempt, record returned usage by role and UTC date, and log request attempts without prompts or credentials (`src/omnisim/llm/budget.py`, `src/omnisim/llm/call_log.py`).

**What happened**  
The repository records a successful fast-role response with 264 prompt tokens and 2,143 completion tokens, but its `run_id` is null. It cannot be attributed to a campaign (`data/llm_calls.jsonl`). Campaign profile admission thresholds remain labeled as unmeasured estimates (`docs/PROGRESS.md`, Phase 9; `src/omnisim/config.py`).

**What we expected / suggestion**  
Usage values should be returned consistently for successful and failed completion attempts, and model-listing requests should remain distinguishable from billable completion usage.

**How we handled it in OmniSim-Agent**  
The client records absent usage as zero, does not charge model-listing rows to the token budget, and reserves token admission for completion requests (`docs/DECISIONS.md`, initial LLM decisions). The current log has no campaign-level attribution for the cited fast-role call.

## Tavily search, extraction, and evidence checks

**What we did**  
We use advanced search restricted to the Panda manufacturer domains, then request extracted page text and keep operation results in a SHA-256 keyed disk cache (`src/omnisim/knowledge/tavily_agent.py`, `src/omnisim/knowledge/cache.py`).

**What happened**  
Evidence acceptance depends on a quote occurring in extracted text and containing the claimed value. Datasheet text can include different whitespace and Unicode punctuation; strict byte matching would reject some visually identical text. The verifier tests normalization for whitespace, Unicode minus, and non-breaking spaces (`tests/unit/test_knowledge_verifier.py`). The progress log says no live Tavily specification was fetched (`docs/PROGRESS.md`, Phase 5).

**What we expected / suggestion**  
Extracted text should preserve enough page context for exact claim review. It is useful to keep stable text alongside source URLs so clients can verify quotes without relying on layout-specific HTML.

**How we handled it in OmniSim-Agent**  
The verifier normalizes Unicode, whitespace, and punctuation before checking quote occurrence, then checks numeric or string value presence. Verified evidence is compared with URDF limits. Presence checks do not prove that a quote semantically supports a claim, so the app treats them as a gate and does not describe them as independent semantic verification (`src/omnisim/knowledge/verifier.py`, `src/omnisim/knowledge/compare.py`, `docs/DECISIONS.md`, Phase 5).
