import json

import pandas as pd

from src.ingest import run_ingest
from src.recommend import _json_safe, parse_task_profile, recommend


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
