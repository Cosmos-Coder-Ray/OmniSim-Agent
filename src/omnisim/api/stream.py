from __future__ import annotations

from pathlib import Path

from omnisim.schemas import CampaignEvent


class EventTailer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self._pending = bytearray()

    def resume_after(self, event_id: int) -> None:
        self.offset = 0
        self._pending.clear()
        if event_id <= 0 or not self.path.is_file():
            return
        with self.path.open("rb") as stream:
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                try:
                    event = CampaignEvent.model_validate_json(line)
                except ValueError:
                    continue
                if event.seq == event_id:
                    self.offset = stream.tell()
                    return

    def read_complete(self) -> list[CampaignEvent]:
        try:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read()
        except FileNotFoundError:
            return []
        self.offset += len(chunk)
        self._pending.extend(chunk)
        events: list[CampaignEvent] = []
        while True:
            newline = self._pending.find(b"\n")
            if newline < 0:
                break
            line = bytes(self._pending[:newline])
            del self._pending[: newline + 1]
            if not line:
                continue
            try:
                event = CampaignEvent.model_validate_json(line)
            except ValueError:
                continue
            events.append(event)
        return events
