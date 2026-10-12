import json
import re
from typing import Any


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)


def _objects(text: str) -> list[dict[str, Any]]:
    text = _strip_think(text)
    spans: list[str] = []
    starts: list[int] = []
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if depth == 0:
            if char == "{":
                starts.append(index)
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                spans.append(text[starts.pop() : index + 1])

    candidates: list[dict[str, Any]] = []
    for span in spans:
        try:
            value = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    return candidates


def extract_json(text: str) -> dict[str, Any]:
    """Return the first balanced JSON object in text."""
    candidates = _objects(text)
    if not candidates:
        raise ValueError("no balanced JSON object found")
    return candidates[0]
