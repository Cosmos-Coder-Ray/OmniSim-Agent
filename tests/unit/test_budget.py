import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnisim.llm.budget import BudgetExceeded, check, get_usage_summary, record


def test_usage_persists_after_database_reopen(tmp_path) -> None:
    path = tmp_path / "usage.sqlite3"
    now = datetime(2026, 10, 10, 12, tzinfo=UTC)

    record("fast", 12, 8, db_path=path, now=now)
    summary = get_usage_summary(cap=100, db_path=path, now=now)

    assert summary["calls"] == 1
    assert summary["total"] == 20
    assert summary["by_role"]["fast"]["completion_tokens"] == 8


def test_usage_rolls_over_at_utc_midnight(tmp_path) -> None:
    path = tmp_path / "usage.sqlite3"
    before = datetime(2026, 10, 10, 23, 59, tzinfo=UTC)
    after = before + timedelta(minutes=2)
    record("deep", 5, 7, db_path=path, now=before)

    assert get_usage_summary(cap=100, db_path=path, now=after)["total"] == 0


def test_check_raises_when_daily_cap_is_reached(tmp_path) -> None:
    now = datetime(2026, 10, 10, tzinfo=UTC)
    path = tmp_path / "usage.sqlite3"
    record("fast", 4, 6, db_path=path, now=now)

    with pytest.raises(BudgetExceeded):
        check("deep", cap=10, db_path=path, now=now)


@given(
    cap=st.integers(min_value=1, max_value=1000),
    usages=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=100),
        )
    ),
)
def test_check_matches_total_recorded_usage(cap: int, usages: list[tuple[int, int]]) -> None:
    now = datetime(2026, 10, 10, tzinfo=UTC)
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as directory:
        path = Path(directory) / "usage.sqlite3"
        for prompt, completion in usages:
            record("fast", prompt, completion, db_path=path, now=now)

        if sum(prompt + completion for prompt, completion in usages) >= cap:
            with pytest.raises(BudgetExceeded):
                check("deep", cap=cap, db_path=path, now=now)
        else:
            check("deep", cap=cap, db_path=path, now=now)
