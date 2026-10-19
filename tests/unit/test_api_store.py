import sqlite3

from omnisim.api.store import ApiStore


def test_store_reopens_active_jobs_without_event_or_ip_columns(tmp_path) -> None:
    store = ApiStore(tmp_path / "data" / "api.sqlite")
    store.initialize()
    store.create("campaign_a", {"budget": 10}, created_at="2026-10-19T00:00:00+00:00")
    store.set_status("campaign_a", "running", at="2026-10-19T00:01:00+00:00")

    reopened = ApiStore(store.path)
    reopened.initialize()
    record = reopened.get("campaign_a")

    assert record is not None
    assert record.config == {"budget": 10}
    assert record.status == "running"
    assert record.started_at == "2026-10-19T00:01:00+00:00"
    assert [item.campaign_id for item in reopened.unfinished()] == ["campaign_a"]
    with sqlite3.connect(store.path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
    assert columns == {
        "campaign_id",
        "config_json",
        "status",
        "kind",
        "created_at",
        "started_at",
        "finished_at",
    }
