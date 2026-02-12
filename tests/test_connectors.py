import json
from pathlib import Path

import pytest

from src.connectors.artificial_analysis import (
    ArtificialAnalysisConnector,
    ArtificialAnalysisError,
    _api_key_from_env_file,
)
from src.connectors.fixture import FixtureConnector


class _FakeResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_fixture_connector_fetches_models(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps({"models": [{"model_name": "x"}, {"model_name": "y"}]}), encoding="utf-8")
    records = FixtureConnector(fixture_path=fixture_path).fetch()
    assert len(records) == 2


def test_artificial_analysis_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AA_API_KEY", raising=False)
    with pytest.raises(ArtificialAnalysisError):
        ArtificialAnalysisConnector().fetch()


def test_artificial_analysis_fetch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AA_API_KEY", "token")

    def fake_urlopen(request, timeout=30):
        assert request.headers.get("X-api-key") == "token"
        return _FakeResponse('{"models": [{"model_name": "good-model"}]}')

    monkeypatch.setattr("src.connectors.artificial_analysis.urlopen", fake_urlopen)
    records = ArtificialAnalysisConnector().fetch()
    assert records[0]["model_name"] == "good-model"


def test_api_key_from_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AA_API_KEY=file-token\n", encoding="utf-8")
    assert _api_key_from_env_file(str(env_path)) == "file-token"
