import logging
import re
import time
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import requests
from pydantic import BaseModel, ConfigDict, TypeAdapter
from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    TavilyKeylessLimitError,
    UsageLimitExceededError,
)
from tavily.errors import TimeoutError as TavilyTimeoutError

from omnisim.config import Settings
from omnisim.knowledge.cache import CACHE_DIR, cache_file, read_cache, write_cache
from omnisim.knowledge.usage import CALL_LOG_PATH, log_call
from omnisim.knowledge.verifier import normalize_text, verified_only, verify_spec
from omnisim.llm.nebius import chat_json
from omnisim.schemas import ErrorMapping, Evidence, FailureKind, HardwareSpec

log = logging.getLogger(__name__)
ROBOT_DOMAINS = {"franka_panda": ("franka.de", "frankaemika.github.io", "docs.franka.de")}
ROBOT_NAMES = {"franka_panda": "Franka Emika Panda"}
MAX_PAGES = 12
MAX_PAGE_CHARS = 12_000
SEARCH_DEPTH = "advanced"
MAX_RESULTS = 20
EXTRACT_DEPTH = "advanced"
_KEYWORDS = ("torque", "velocity", "joint", "limit", "error", "fault", "collision", "safety")


class LimitDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_id: str
    quantity: Literal["torque", "velocity", "lower", "upper"]
    joint: int
    value: float
    unit: str
    quote: str


class LimitsDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[LimitDraft]


class ErrorDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_id: str
    claim: str
    value: float | str
    unit: str | None = None
    quote: str


class ErrorsDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[ErrorDraft]


class MappingDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: FailureKind
    evidence_id: str | None = None
    confidence: Literal["low", "medium", "high"] | None = None


class MappingsDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[MappingDraft]


def _robot(robot: str) -> tuple[str, tuple[str, ...]]:
    if robot not in ROBOT_DOMAINS:
        raise ValueError(f"unknown robot {robot!r}; supported: {', '.join(ROBOT_DOMAINS)}")
    return ROBOT_NAMES[robot], ROBOT_DOMAINS[robot]


def _allowed(url: str, domains: tuple[str, ...]) -> bool:
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def _client() -> TavilyClient:
    key = Settings.from_env().TAVILY_API_KEY
    if not key:
        raise ValueError("TAVILY_API_KEY is required on a cache miss")
    return TavilyClient(api_key=key)


def _get_client(op: str, run_id: str | None, call_log_path: Path) -> TavilyClient:
    try:
        return _client()
    except ValueError as exc:
        log_call(
            {
                "run_id": run_id,
                "op": op,
                "cache": "miss",
                "latency_s": 0.0,
                "n_results": 0,
                "ok": False,
                "error": type(exc).__name__,
            },
            path=call_log_path,
        )
        raise


def _run_live(
    op: str,
    run_id: str | None,
    call,
    *,
    call_log_path: Path,
) -> Any:
    retried = False
    while True:
        started = time.perf_counter()
        try:
            result = call()
        except (requests.Timeout, TavilyTimeoutError) as exc:
            latency = time.perf_counter() - started
            log_call(
                {
                    "run_id": run_id,
                    "op": op,
                    "cache": "miss",
                    "latency_s": latency,
                    "n_results": 0,
                    "ok": False,
                    "error": type(exc).__name__,
                },
                path=call_log_path,
            )
            if not retried:
                retried = True
                time.sleep(0.25)
                continue
            raise
        except requests.HTTPError as exc:
            latency = time.perf_counter() - started
            status = exc.response.status_code if exc.response is not None else None
            log_call(
                {
                    "run_id": run_id,
                    "op": op,
                    "cache": "miss",
                    "latency_s": latency,
                    "n_results": 0,
                    "ok": False,
                    "error": type(exc).__name__,
                },
                path=call_log_path,
            )
            if status is not None and status >= 500 and not retried:
                retried = True
                time.sleep(0.25)
                continue
            raise
        except requests.RequestException as exc:
            log_call(
                {
                    "run_id": run_id,
                    "op": op,
                    "cache": "miss",
                    "latency_s": time.perf_counter() - started,
                    "n_results": 0,
                    "ok": False,
                    "error": type(exc).__name__,
                },
                path=call_log_path,
            )
            raise
        except (
            BadRequestError,
            ForbiddenError,
            InvalidAPIKeyError,
            TavilyKeylessLimitError,
            UsageLimitExceededError,
        ) as exc:
            log_call(
                {
                    "run_id": run_id,
                    "op": op,
                    "cache": "miss",
                    "latency_s": time.perf_counter() - started,
                    "n_results": 0,
                    "ok": False,
                    "error": type(exc).__name__,
                },
                path=call_log_path,
            )
            raise
        result_count = len(result.get("results", [])) if isinstance(result, dict) else 0
        log_call(
            {
                "run_id": run_id,
                "op": op,
                "cache": "miss",
                "latency_s": time.perf_counter() - started,
                "n_results": result_count,
                "ok": True,
                "error": None,
            },
            path=call_log_path,
        )
        return result


def _search(
    query: str,
    domains: tuple[str, ...],
    *,
    run_id: str | None = None,
    cache_dir: Path = CACHE_DIR,
    call_log_path: Path = CALL_LOG_PATH,
    client: Any | None = None,
) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "depth": SEARCH_DEPTH,
        "max_results": MAX_RESULTS,
        "include_domains": list(domains),
        "extract_depth": EXTRACT_DEPTH,
    }
    path = cache_file("search", params, cache_dir=cache_dir)
    started = time.perf_counter()
    cached = read_cache(path)
    if cached is not None:
        results = cached.get("results", []) if isinstance(cached, dict) else []
        results = _filter_hits(results, domains)
        log_call(
            {
                "run_id": run_id,
                "op": "search",
                "cache": "hit",
                "latency_s": time.perf_counter() - started,
                "n_results": len(results),
                "ok": True,
            },
            path=call_log_path,
        )
        return results
    sdk = client if client is not None else _get_client("search", run_id, call_log_path)
    response = _run_live(
        "search",
        run_id,
        lambda: sdk.search(
            query=query,
            search_depth=SEARCH_DEPTH,
            max_results=MAX_RESULTS,
            include_domains=list(domains),
        ),
        call_log_path=call_log_path,
    )
    results = response.get("results", []) if isinstance(response, dict) else []
    results = _filter_hits(results, domains)
    results.sort(key=lambda hit: (-float(hit.get("score", 0) or 0), hit["url"]))
    write_cache(path, "search", params, {"results": results})
    return results


def _filter_hits(results: list[dict[str, Any]], domains: tuple[str, ...]) -> list[dict[str, Any]]:
    filtered = [
        {**hit, "url": _canonical_url(hit["url"])}
        for hit in results
        if isinstance(hit, dict)
        and isinstance(hit.get("url"), str)
        and _allowed(hit["url"], domains)
    ]
    filtered.sort(key=lambda hit: (-float(hit.get("score", 0) or 0), hit["url"]))
    return filtered


def _extract(
    urls: list[str],
    *,
    run_id: str | None = None,
    cache_dir: Path = CACHE_DIR,
    call_log_path: Path = CALL_LOG_PATH,
    client: Any | None = None,
) -> dict[str, str]:
    pages: dict[str, str] = {}
    missing: list[str] = []
    paths: dict[str, Path] = {}
    params_by_url: dict[str, dict[str, Any]] = {}
    for url in sorted(set(urls)):
        params = {
            "query": None,
            "depth": None,
            "max_results": None,
            "include_domains": None,
            "extract_depth": EXTRACT_DEPTH,
            "url": url,
        }
        path = cache_file("extract", params, cache_dir=cache_dir)
        paths[url] = path
        params_by_url[url] = params
        started = time.perf_counter()
        cached = read_cache(path)
        if isinstance(cached, dict) and isinstance(cached.get("content"), str):
            pages[url] = cached["content"]
            log_call(
                {
                    "run_id": run_id,
                    "op": "extract",
                    "cache": "hit",
                    "latency_s": time.perf_counter() - started,
                    "n_results": 1,
                    "ok": True,
                },
                path=call_log_path,
            )
        else:
            missing.append(url)
    if not missing:
        return pages
    sdk = client if client is not None else _get_client("extract", run_id, call_log_path)
    response = _run_live(
        "extract",
        run_id,
        lambda: sdk.extract(urls=missing, extract_depth=EXTRACT_DEPTH),
        call_log_path=call_log_path,
    )
    for row in response.get("results", []) if isinstance(response, dict) else []:
        if not isinstance(row, dict) or not isinstance(row.get("url"), str):
            continue
        url = _canonical_url(row["url"])
        if url not in paths or not isinstance(row.get("raw_content", row.get("content")), str):
            continue
        content = row.get("raw_content", row.get("content"))
        if not isinstance(content, str):
            continue
        pages[url] = content
        write_cache(paths[url], "extract", params_by_url[url], {"content": content})
    return pages


def _excerpt(text: str, cap: int = MAX_PAGE_CHARS) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= cap:
        return normalized
    radius = max(120, cap // max(2, len(_KEYWORDS) * 2))
    windows: dict[tuple[int, int], int] = {}
    for keyword in _KEYWORDS:
        for match in re.finditer(re.escape(keyword), normalized):
            start = max(0, match.start() - radius)
            end = min(len(normalized), match.end() + radius)
            windows[(start, end)] = windows.get((start, end), 0) + 1
    if not windows:
        return normalized[:cap]
    selected: list[tuple[int, int]] = []
    remaining = cap
    for (start, end), _ in sorted(windows.items(), key=lambda item: (-item[1], item[0][0])):
        if any(start <= old_end and end >= old_start for old_start, old_end in selected):
            continue
        window_len = min(end - start, remaining)
        selected.append((start, start + window_len))
        remaining -= window_len
        if remaining <= 0:
            break
    selected.sort()
    return " … ".join(normalized[start:end] for start, end in selected)[:cap]


def _page_bundle(
    results: list[dict[str, Any]], pages: dict[str, str]
) -> tuple[dict[str, str], dict[str, str], str]:
    ranked = sorted(
        (hit for hit in results if hit.get("url") in pages),
        key=lambda hit: (-float(hit.get("score", 0) or 0), hit["url"]),
    )[:MAX_PAGES]
    page_texts = {hit["url"]: pages[hit["url"]] for hit in ranked}
    page_urls = {f"P{i}": hit["url"] for i, hit in enumerate(ranked, 1)}
    prompt_pages = "\n\n".join(
        f'<page id="P{i}">\n{_excerpt(pages[hit["url"]])}\n</page>'
        for i, hit in enumerate(ranked, 1)
    )
    return page_texts, page_urls, prompt_pages


def _prompt(name: str) -> str:
    path = Path(__file__).resolve().parents[1] / "llm" / "prompts" / name
    return path.read_text(encoding="utf-8")


def _fetch_topic(
    query: str,
    domains: tuple[str, ...],
    *,
    run_id: str | None,
    cache_dir: Path,
    call_log_path: Path,
    client: Any | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    results = _search(
        query,
        domains,
        run_id=run_id,
        cache_dir=cache_dir,
        call_log_path=call_log_path,
        client=client,
    )
    urls = [hit["url"] for hit in results[:MAX_PAGES]]
    pages = _extract(
        urls,
        run_id=run_id,
        cache_dir=cache_dir,
        call_log_path=call_log_path,
        client=client,
    )
    return results, pages


def build_spec(robot: str, *, run_id: str | None = None) -> HardwareSpec:
    display_name, domains = _robot(robot)
    cache_dir, call_log_path = CACHE_DIR, CALL_LOG_PATH
    limits_results, limits_pages = _fetch_topic(
        f"{display_name} arm joint torque velocity position limits documentation",
        domains,
        run_id=run_id,
        cache_dir=cache_dir,
        call_log_path=call_log_path,
        client=None,
    )
    errors_results, errors_pages = _fetch_topic(
        f"{display_name} robot error codes fault taxonomy documentation",
        domains,
        run_id=run_id,
        cache_dir=cache_dir,
        call_log_path=call_log_path,
        client=None,
    )
    limit_page_texts, limit_urls, limit_prompt = _page_bundle(limits_results, limits_pages)
    error_page_texts, error_urls, error_prompt = _page_bundle(errors_results, errors_pages)
    limit_draft, _ = chat_json(
        "deep",
        _prompt("knowledge_limits.md"),
        f"Robot: {display_name} (slug: {robot})\nUntrusted source excerpts:\n{limit_prompt}",
        LimitsDraft,
        temperature=0,
        max_tokens=4096,
        run_id=run_id,
    )
    error_draft, _ = chat_json(
        "deep",
        _prompt("knowledge_errors.md"),
        f"Robot: {display_name} (slug: {robot})\nUntrusted source excerpts:\n{error_prompt}",
        ErrorsDraft,
        temperature=0,
        max_tokens=4096,
        run_id=run_id,
    )
    limits: dict[str, list[Evidence]] = {"torque": [], "velocity": [], "position": []}
    for item in limit_draft.items:
        url = limit_urls.get(item.page_id)
        if url is None:
            log.warning("unknown knowledge page id", extra={"page_id": item.page_id})
            continue
        if not 1 <= item.joint <= 7:
            log.warning("out of range documented joint", extra={"joint": item.joint})
            continue
        quantity = item.quantity
        ev = Evidence(
            claim=f"joint {item.joint} {quantity}",
            value=item.value,
            unit=item.unit,
            source_url=url,
            quote=item.quote,
            verified=False,
        )
        limits[
            "torque"
            if quantity == "torque"
            else "velocity"
            if quantity == "velocity"
            else "position"
        ].append(ev)
    errors: list[Evidence] = []
    for item in error_draft.items:
        url = error_urls.get(item.page_id)
        if url is None:
            log.warning("unknown knowledge page id", extra={"page_id": item.page_id})
            continue
        errors.append(
            Evidence(
                claim=item.claim,
                value=item.value,
                unit=item.unit,
                source_url=url,
                quote=item.quote,
                verified=False,
            )
        )
    pages = limit_page_texts | error_page_texts
    draft = HardwareSpec(
        robot=robot,
        joint_torque_limits_nm=limits["torque"],
        joint_velocity_limits=limits["velocity"],
        joint_position_limits=limits["position"],
        error_taxonomy=errors,
    )
    spec = verify_spec(draft, pages)
    for name, values in (
        ("torque", spec.joint_torque_limits_nm),
        ("velocity", spec.joint_velocity_limits),
        ("lower", spec.joint_position_limits),
        ("upper", spec.joint_position_limits),
    ):
        verified_joints = {
            int(match.group(1))
            for ev in values
            if ev.verified
            if (match := re.fullmatch(rf"joint ([1-7]) {name}", ev.claim))
        }
        log.info(
            "hardware spec coverage",
            extra={"quantity": name, "verified": len(verified_joints), "expected": 7},
        )
    return spec


_FAILURE_MEANINGS: dict[FailureKind, str] = {
    "torque_saturation": "measured joint torque reached its configured saturation threshold",
    "joint_limit": "measured joint position approached a configured joint bound",
    "velocity_limit": "measured joint velocity reached its configured limit",
    "tracking_error": "end-effector tracking error reached its configured threshold",
    "singularity": "the measured Jacobian condition number reached its threshold",
    "collision": "the simulation detected robot contact",
    "instability": "the simulation detected non-finite telemetry or divergent motion",
}


def map_failures_to_real_errors(
    failure_kinds: list[FailureKind], spec: HardwareSpec, *, run_id: str | None = None
) -> dict[FailureKind, ErrorMapping]:
    for kind in failure_kinds:
        TypeAdapter(FailureKind).validate_python(kind)
    candidates = verified_only(spec).error_taxonomy
    evidence_ids = {f"E{i}": ev for i, ev in enumerate(candidates, 1)}
    result: dict[FailureKind, ErrorMapping] = {
        kind: ErrorMapping(kind=kind) for kind in failure_kinds
    }
    if not candidates:
        return result
    evidence_text = "\n".join(
        f'<evidence id="{evidence_id}">{ev.claim}: {ev.value} {ev.unit or ""}. '
        f"Quote: {ev.quote}</evidence>"
        for evidence_id, ev in evidence_ids.items()
    )
    kinds = list(dict.fromkeys(failure_kinds))
    system = _prompt("knowledge_mapping.md")
    user = (
        "\n".join(f"{kind}: {_FAILURE_MEANINGS[kind]}" for kind in kinds)
        + f"\nVerified evidence candidates:\n{evidence_text}"
    )
    draft, _ = chat_json(
        "deep",
        system,
        user,
        MappingsDraft,
        temperature=0,
        max_tokens=2048,
        run_id=run_id,
    )
    for mapping in draft.items:
        if mapping.kind not in result:
            continue
        evidence = evidence_ids.get(mapping.evidence_id or "")
        if evidence is not None and mapping.confidence is not None:
            result[mapping.kind] = ErrorMapping(
                kind=mapping.kind,
                real_error_name=str(evidence.value),
                evidence=evidence,
                confidence=mapping.confidence,
            )
    return result


def search_known_issues(
    robot: str,
    failure_kind: FailureKind,
    *,
    run_id: str | None = None,
) -> list[dict[str, str]]:
    display_name, domains = _robot(robot)
    allowed = (*domains, "github.com")
    results = _search(
        f"{display_name} {failure_kind} known issue",
        allowed,
        run_id=run_id,
    )
    return [
        {
            "url": hit["url"],
            "title": str(hit.get("title", "")),
            "content": str(hit.get("content", "")),
        }
        for hit in results
    ]
