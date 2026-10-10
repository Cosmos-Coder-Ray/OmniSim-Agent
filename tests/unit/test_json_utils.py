import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnisim.llm.json_utils import extract_json


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('<think>draft {"bad": true}</think>{"ok": true}', {"ok": True}),
        ('```json\n{"ok": true}\n```', {"ok": True}),
        ('```\n{"ok": true}\n```', {"ok": True}),
        ('before this {"ok": true} after', {"ok": True}),
        ('{"outer": {"inner": 1}}', {"outer": {"inner": 1}}),
        ('{"text": "a } brace"}', {"text": "a } brace"}),
        ('{"text": "escaped \\" quote"}', {"text": 'escaped " quote'}),
        ('{"first": 1} {"second": 2}', {"first": 1}),
        ('prefix {"city": "東京"}', {"city": "東京"}),
        ('[{"inside": true}]', {"inside": True}),
    ],
)
def test_extract_json_returns_first_object(text: str, expected: dict) -> None:
    assert extract_json(text) == expected


@pytest.mark.parametrize(
    "text", ['{"unfinished":', '<think>unfinished {"bad": true}', "nothing here", "", "[]"]
)
def test_extract_json_rejects_missing_or_unbalanced_objects(text: str) -> None:
    with pytest.raises(ValueError, match="no balanced JSON object found"):
        extract_json(text)


json_value = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=10,
)
plain_text = st.text(alphabet=st.characters(exclude_characters="{}"))


@given(
    value=st.dictionaries(st.text(), json_value),
    prefix=plain_text,
    suffix=plain_text,
    fenced=st.booleans(),
)
def test_extract_json_round_trips_objects_through_prose(
    value: dict, prefix: str, suffix: str, fenced: bool
) -> None:
    encoded = json.dumps(value, ensure_ascii=False)
    payload = f"```json\n{encoded}\n```" if fenced else encoded
    assert extract_json(f"{prefix}{payload}{suffix}") == value
