import re

from hypothesis import given
from hypothesis import strategies as st

from omnisim.knowledge.verifier import normalize_text, verified_only, verify_evidence, verify_spec
from omnisim.schemas import Evidence, HardwareSpec


def evidence(
    value: float | str = 2.175,
    *,
    quote: str = "maximum speed is 2.175 rad/s",
    url: str = "https://docs.test/page",
) -> Evidence:
    return Evidence(
        claim="joint 1 velocity",
        value=value,
        unit="rad/s",
        source_url=url,
        quote=quote,
        verified=False,
    )


def test_exact_quote_and_numeric_equivalence() -> None:
    ev = evidence(2.175, quote="maximum speed is 2.1750 rad/s")
    assert verify_evidence(ev, {ev.source_url: "The maximum speed is 2.1750 rad/s for joint 1."})


def test_normalization_handles_case_whitespace_unicode_minus_and_nbsp() -> None:
    ev = evidence(-2.175, quote="Lower limit is −2.175 rad/s")
    assert verify_evidence(ev, {ev.source_url: "lower   limit is -2.175 rad/s"})
    assert normalize_text("A\u200b  B") == "a b"


def test_numbers_must_be_standalone_tokens() -> None:
    for text in ("187 Nm", "870 Nm", "2087 Nm"):
        ev = evidence(87, quote=text)
        assert not verify_evidence(ev, {ev.source_url: text})


def test_short_quote_value_missing_page_and_empty_quote_fail() -> None:
    short = evidence(2.0, quote="2 rad/s")
    assert not verify_evidence(short, {short.source_url: short.quote})
    absent = evidence(2.0)
    assert not verify_evidence(absent, {})
    empty = evidence(2.0, quote="")
    assert not verify_evidence(empty, {empty.source_url: "2"})


def test_quote_must_contain_value_and_string_values_match() -> None:
    ev = evidence(9.81, quote="maximum speed is limited by the controller")
    assert not verify_evidence(ev, {ev.source_url: ev.quote})
    text = evidence("E42", quote="fault code E42 is a joint drive error")
    assert verify_evidence(text, {text.source_url: text.quote})


def test_hallucinated_quote_is_rejected() -> None:
    ev = evidence(2.175, quote="maximum speed is 2.175 rad/s")
    assert not verify_evidence(ev, {ev.source_url: "This page contains no such limit."})


def test_verify_spec_returns_copy_and_verified_only_drops_unverified() -> None:
    good = evidence()
    bad = evidence(3.0)
    spec = HardwareSpec(
        robot="franka_panda",
        joint_torque_limits_nm=[],
        joint_velocity_limits=[good, bad],
        joint_position_limits=[],
        error_taxonomy=[],
    )
    verified = verify_spec(spec, {good.source_url: good.quote})
    assert not spec.joint_velocity_limits[0].verified
    assert verified.joint_velocity_limits[0].verified
    assert not verified.joint_velocity_limits[1].verified
    assert len(verified_only(verified).joint_velocity_limits) == 1


@given(
    prefix=st.text(max_size=20),
    value=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    suffix=st.text(max_size=20),
)
def test_verifier_soundness(prefix: str, value: float, suffix: str) -> None:
    quote = f"{prefix} documented value {value!r} {suffix}"
    ev = evidence(value, quote=quote)
    page = f"before {quote} after"
    if verify_evidence(ev, {ev.source_url: page}):
        assert normalize_text(quote) in normalize_text(page)
        tokens = re.findall(
            r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])", quote
        )
        assert any(float(token) == ev.value for token in tokens)
