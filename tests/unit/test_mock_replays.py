import json
from pathlib import Path

import pytest

MOCK_REPLAYS = Path(__file__).parents[2] / "web" / "public" / "mock"


def assert_constant_segment_lengths(replay: dict) -> None:
    links = [frame["links"] for frame in replay["frames"]]
    reference = [
        sum((links[0][index + 1][axis] - links[0][index][axis]) ** 2 for axis in range(3)) ** 0.5
        for index in range(7)
    ]
    for frame_links in links[1:]:
        for index, expected in enumerate(reference):
            actual = (
                sum(
                    (frame_links[index + 1][axis] - frame_links[index][axis]) ** 2
                    for axis in range(3)
                )
                ** 0.5
            )
            assert actual == pytest.approx(expected, abs=0.001)


@pytest.mark.parametrize("path", sorted(MOCK_REPLAYS.glob("episode_*.json")), ids=lambda p: p.name)
def test_mock_replay_segment_lengths_are_constant(path: Path) -> None:
    replay = json.loads(path.read_text(encoding="utf-8"))
    assert_constant_segment_lengths(replay)


def test_zero_length_segments_are_allowed() -> None:
    links = [[0.0, 0.0, 0.0] for _ in range(8)]
    assert_constant_segment_lengths({"frames": [{"links": links}, {"links": links}]})
