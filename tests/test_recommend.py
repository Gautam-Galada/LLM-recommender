from src.ingest import run_ingest
from src.recommend import parse_task_profile, recommend


def test_recommend_output_shape() -> None:
    run_ingest()
    profile = parse_task_profile(
        "I need a model for python debugging, prefer quality, budget $5/1M tokens",
        max_price_per_1m=None,
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
