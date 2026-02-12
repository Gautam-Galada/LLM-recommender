from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FixtureConnector:
    def __init__(self, fixture_path: str | Path = "data/fixtures/fixture.json") -> None:
        self.fixture_path = Path(fixture_path)

    def fetch(self) -> list[dict[str, Any]]:
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
        if isinstance(payload, dict):
            records = payload.get("data") or payload.get("models") or []
            return [r for r in records if isinstance(r, dict)]
        return []
