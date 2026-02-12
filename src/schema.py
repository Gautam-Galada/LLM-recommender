from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

CANONICAL_COLUMNS = [
    "snapshot_ts",
    "source",
    "model_name",
    "provider",
    "quality_index",
    "coding_index",
    "math_index",
    "reasoning_index",
    "output_tokens_per_s",
    "ttft_s",
    "price_input_per_1m",
    "price_output_per_1m",
    "context_window",
    "is_open_source",
    "license",
    "canonical_model_key",
]


@dataclass(slots=True)
class TaskProfile:
    task_type: str
    weight_quality: float
    weight_speed: float
    weight_cost: float
    max_price_per_1m: float | None = None
    min_context: int | None = None
    provider_allowlist: set[str] | None = None


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _pick(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if "." in key:
            current: Any = record
            found = True
            for part in key.split("."):
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    found = False
                    break
            if found and current is not None:
                return current
        elif key in record and record[key] is not None:
            return record[key]
    return None


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def canonical_model_key(model_name: str, provider: str | None) -> str:
    name = (model_name or "unknown").strip().lower().replace(" ", "-")
    provider_part = (provider or "unknown").strip().lower().replace(" ", "-")
    return f"{provider_part}::{name}"


def normalize_records(records: list[dict[str, Any]], source: str, snapshot_ts: str) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for record in records:
        model_name = _pick(record, "model_name", "modelName", "name", "model") or "unknown-model"
        provider = _pick(
            record,
            "provider",
            "provider_name",
            "providerName",
            "vendor",
            "lab",
            "organization",
            "developer",
            "creator.name",
        )
        row = {
            "snapshot_ts": snapshot_ts,
            "source": source,
            "model_name": str(model_name),
            "provider": str(provider) if provider is not None else None,
            "quality_index": _to_float(
                _pick(record, "quality_index", "intelligence_index", "intelligenceIndex", "overall_index", "overall")
            ),
            "coding_index": _to_float(_pick(record, "coding_index", "codingIndex", "code_index")),
            "math_index": _to_float(_pick(record, "math_index", "mathIndex")),
            "reasoning_index": _to_float(_pick(record, "reasoning_index", "reasoningIndex", "reasoning")),
            "output_tokens_per_s": _to_float(
                _pick(
                    record,
                    "output_tokens_per_s",
                    "outputTokensPerSecond",
                    "tokens_per_second",
                    "tokensPerSecond",
                    "throughput",
                    "performance.output_tokens_per_s",
                )
            ),
            "ttft_s": _to_float(_pick(record, "ttft_s", "time_to_first_token", "timeToFirstToken", "latency_ttft_s")),
            "price_input_per_1m": _to_float(
                _pick(
                    record,
                    "price_input_per_1m",
                    "input_price_per_1m",
                    "inputPricePer1M",
                    "input_cost_per_million",
                    "pricing.input_price_per_1m",
                    "pricing.inputPricePer1M",
                )
            ),
            "price_output_per_1m": _to_float(
                _pick(
                    record,
                    "price_output_per_1m",
                    "output_price_per_1m",
                    "outputPricePer1M",
                    "output_cost_per_million",
                    "pricing.output_price_per_1m",
                    "pricing.outputPricePer1M",
                )
            ),
            "context_window": _to_int(_pick(record, "context_window", "contextWindow", "context_tokens", "max_context", "maxContext")),
            "is_open_source": _to_bool(_pick(record, "is_open_source", "open_source")),
            "license": _pick(record, "license", "license_type"),
        }
        row["canonical_model_key"] = canonical_model_key(row["model_name"], row["provider"])
        normalized.append(row)

    frame = pd.DataFrame(normalized)
    if frame.empty:
        frame = pd.DataFrame(columns=CANONICAL_COLUMNS)
    for col in CANONICAL_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    return frame[CANONICAL_COLUMNS]
