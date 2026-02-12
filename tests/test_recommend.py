import json

import pandas as pd
import pytest

from src.ingest import run_ingest
from src.recommend import _extract_budget_from_text, _json_safe, parse_task_profile, recommend


def test_parse_task_profile_detects_coding_and_budget() -> None:
    profile = parse_task_profile(
        "Need python debugging with strong quality but low budget",
        max_price_per_1m=5.0,
        min_context=32000,
        provider_allowlist="meta,alibaba",
    )
    assert profile.task_type == "coding"
    assert profile.max_price_per_1m == 5.0
    assert profile.min_context == 32000
    assert profile.provider_allowlist == {"meta", "alibaba"}


def test_recommend_output_shape() -> None:
    run_ingest()
    profile = parse_task_profile(
        "I need a model for python debugging, prefer quality, budget $5/1M tokens",
        max_price_per_1m=5.0,
        min_context=None,
        provider_allowlist=None,
    )
    result = recommend(profile, topk=2)
    assert "recommendations" in result
    assert len(result["recommendations"]) <= 2
    if result["recommendations"]:
        first = result["recommendations"][0]
        assert "score" in first
        assert "justification" in first
        assert "snapshot_ts" in first


def test_recommend_hard_constraint_filter() -> None:
    run_ingest()
    profile = parse_task_profile(
        "general",
        max_price_per_1m=0.1,
        min_context=None,
        provider_allowlist=None,
    )
    result = recommend(profile, topk=3)
    assert result["recommendations"] == []


def test_json_safe_handles_pandas_na_and_sets() -> None:
    payload = {"x": pd.NA, "providers": {"meta", "openai"}}
    safe = _json_safe(payload)
    assert safe["x"] is None
    assert safe["providers"] == ["meta", "openai"]
    json.dumps(safe)


def test_extract_budget_accepts_without_dollar_sign() -> None:
    assert _extract_budget_from_text("budget 5 per 1M tokens") == 5.0


def test_recommend_warns_and_returns_empty_on_near_empty_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeConn:
        def execute(self, _query: str):
            class _Result:
                @staticmethod
                def fetch_df() -> pd.DataFrame:
                    return pd.DataFrame(
                        [
                            {
                                "snapshot_ts": "2026-02-12 21:44:48",
                                "canonical_model_key": "unknown::model-a",
                                "model_name": "Model A",
                                "provider": None,
                                "price_input_per_1m": pd.NA,
                                "output_tokens_per_s": pd.NA,
                                "context_window": pd.NA,
                                "ttft_s": pd.NA,
                                "coding_index": pd.NA,
                                "quality_index": pd.NA,
                                "reasoning_index": pd.NA,
                                "math_index": pd.NA,
                            }
                        ]
                    )

            return _Result()

        def close(self) -> None:
            return None

    monkeypatch.setattr("src.recommend.connect", lambda: _FakeConn())
    monkeypatch.setattr("src.recommend.init_warehouse", lambda _con: None)
    profile = parse_task_profile("python debugging", None, None, None)
    result = recommend(profile, topk=5)
    assert result["recommendations"] == []
    assert "warning" in result
