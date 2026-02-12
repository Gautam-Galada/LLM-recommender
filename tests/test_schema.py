from src.schema import canonical_model_key, normalize_records


def test_normalize_records_has_canonical_columns() -> None:
    records = [{"name": "Test Model", "vendor": "Acme", "overall_index": "66.5", "context_tokens": "4096"}]
    frame = normalize_records(records, source="fixture", snapshot_ts="2024-01-01T00:00:00+00:00")
    assert "canonical_model_key" in frame.columns
    assert frame.loc[0, "quality_index"] == 66.5
    assert frame.loc[0, "context_window"] == 4096


def test_normalize_records_bool_and_aliases() -> None:
    records = [{"model": "Foo", "lab": "Bar", "code_index": "88.2", "open_source": "yes"}]
    frame = normalize_records(records, source="fixture", snapshot_ts="2024-01-01T00:00:00+00:00")
    assert frame.loc[0, "coding_index"] == 88.2
    assert frame.loc[0, "is_open_source"] is True


def test_normalize_records_nested_and_camel_case_fields() -> None:
    records = [
        {
            "modelName": "Model Z",
            "organization": "OrgX",
            "intelligenceIndex": "77.7",
            "outputTokensPerSecond": "123.4",
            "pricing": {"inputPricePer1M": "4.2", "outputPricePer1M": "8.1"},
            "contextWindow": "200000",
        }
    ]
    frame = normalize_records(records, source="fixture", snapshot_ts="2024-01-01T00:00:00+00:00")
    assert frame.loc[0, "provider"] == "OrgX"
    assert frame.loc[0, "quality_index"] == 77.7
    assert frame.loc[0, "output_tokens_per_s"] == 123.4
    assert frame.loc[0, "price_input_per_1m"] == 4.2
    assert frame.loc[0, "context_window"] == 200000


def test_canonical_model_key() -> None:
    assert canonical_model_key("Test Model", "Acme Labs") == "acme-labs::test-model"
