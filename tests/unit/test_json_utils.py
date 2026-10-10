import json

import pytest
from hypothesis import given, strategies as st

from omnisim.llm.json_utils import extract_json


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('<think>draft {"bad": true}</think>{"ok": true}', {"ok": True}),
        ('<think>unfinished {"bad": true}', {}),
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


@pytest.mark.parametrize("text", ['{"unfinished":', "nothing here", "", "[]"])
def test_extract_json_rejects_missing_or_unbalanced_objects(text: str) -> None:
    with pytest.raises(ValueError, match="no balanced JSON object found"):
        extract_json(text)


@given(st.dictionaries(st.text(min_size=1), st.one_of(st.integers(), st.text(), st.booleans())))
def test_extract_json_round_trips_objects_through_prose(value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False)
    assert extract_json(f"Some context ```json\n{encoded}\n``` more context") == value
