from src.schema import canonical_model_key, normalize_records


def test_normalize_records_has_canonical_columns() -> None:
    records = [{"name": "Test Model", "vendor": "Acme", "overall_index": "66.5", "context_tokens": "4096"}]
    frame = normalize_records(records, source="fixture", snapshot_ts="2024-01-01T00:00:00+00:00")
    assert "canonical_model_key" in frame.columns
    assert frame.loc[0, "quality_index"] == 66.5
    assert frame.loc[0, "context_window"] == 4096


def test_canonical_model_key() -> None:
    assert canonical_model_key("Test Model", "Acme Labs") == "acme-labs::test-model"
