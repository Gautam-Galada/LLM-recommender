from pathlib import Path

import pandas as pd
import pytest

from src import ingest


def test_choose_source_falls_back_to_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingConnector:
        def fetch(self):
            raise ingest.ArtificialAnalysisError("missing")

    class _Fixture:
        def fetch(self):
            return [{"model_name": "fixture-model"}]

    monkeypatch.setattr(ingest, "ArtificialAnalysisConnector", lambda: _FailingConnector())
    monkeypatch.setattr(ingest, "FixtureConnector", lambda: _Fixture())
    source, records = ingest.choose_source()
    assert source == "fixture"
    assert records[0]["model_name"] == "fixture-model"


def test_write_snapshot_creates_parquet(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"snapshot_ts": "2025-01-01T00:00:00+00:00", "source": "fixture"}])
    cwd = Path.cwd()
    try:
        # write to isolated path
        import os

        os.chdir(tmp_path)
        out = ingest.write_snapshot(frame, source="fixture", snapshot_ts="2025-01-01T00:00:00+00:00")
        assert out.exists()
        assert out.suffix == ".parquet"
    finally:
        os.chdir(cwd)
