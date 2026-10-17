import re
import unicodedata

from omnisim.schemas import Evidence, HardwareSpec

MIN_QUOTE_CHARS = 15
_NUMBER = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u2212", "-").replace("\u00a0", " ")
    text = text.translate(_ZERO_WIDTH)
    text = text.translate(
        str.maketrans(
            {
                "‘": "'",
                "’": "'",
                "‚": "'",
                "‛": "'",
                "“": '"',
                "”": '"',
                "„": '"',
                "‟": '"',
                "‐": "-",
                "‑": "-",
                "‒": "-",
                "–": "-",
                "—": "-",
                "―": "-",
            }
        )
    )
    return " ".join(text.split()).casefold()


def _value_in_quote(value: float | str, quote: str) -> bool:
    normalized = normalize_text(quote)
    if isinstance(value, str):
        return bool(normalize_text(value)) and normalize_text(value) in normalized
    try:
        target = float(value)
    except (TypeError, ValueError):
        return False
    for match in _NUMBER.finditer(normalized):
        try:
            if float(match.group()) == target:
                return True
        except ValueError:
            continue
    return False


def verify_evidence(
    ev: Evidence, pages: dict[str, str], *, min_quote_chars: int = MIN_QUOTE_CHARS
) -> bool:
    page = pages.get(ev.source_url)
    quote = normalize_text(ev.quote)
    return (
        page is not None
        and len(quote) >= min_quote_chars
        and bool(quote)
        and quote in normalize_text(page)
        and _value_in_quote(ev.value, ev.quote)
    )


def verify_spec(
    spec: HardwareSpec,
    pages: dict[str, str],
    *,
    min_quote_chars: int = MIN_QUOTE_CHARS,
) -> HardwareSpec:
    return spec.model_copy(
        update={
            name: [
                ev.model_copy(
                    update={"verified": verify_evidence(ev, pages, min_quote_chars=min_quote_chars)}
                )
                for ev in getattr(spec, name)
            ]
            for name in (
                "joint_torque_limits_nm",
                "joint_velocity_limits",
                "joint_position_limits",
                "error_taxonomy",
            )
        }
    )


def verified_only(spec: HardwareSpec) -> HardwareSpec:
    return spec.model_copy(
        update={
            name: [ev for ev in getattr(spec, name) if ev.verified]
            for name in (
                "joint_torque_limits_nm",
                "joint_velocity_limits",
                "joint_position_limits",
                "error_taxonomy",
            )
        }
    )
