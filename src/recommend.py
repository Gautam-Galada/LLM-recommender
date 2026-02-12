from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.schema import TaskProfile
from src.warehouse import connect, init_warehouse

TASK_KEYWORDS = {
    "coding": ["code", "python", "debug", "program", "refactor"],
    "math": ["math", "algebra", "calculus", "theorem"],
    "reasoning": ["reason", "logic", "analysis", "decision"],
    "writing": ["write", "copy", "content", "email", "blog"],
    "rag": ["rag", "retrieval", "documents", "knowledge base"],
    "agent": ["agent", "tool use", "autonomous", "workflow"],
}


def _json_safe(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    return value


def parse_task_profile(
    task_text: str,
    max_price_per_1m: float | None,
    min_context: int | None,
    provider_allowlist: str | None,
) -> TaskProfile:
    text = task_text.lower()
    task_type = "general"
    for name, keywords in TASK_KEYWORDS.items():
        if any(k in text for k in keywords):
            task_type = name
            break

    weight_quality, weight_speed, weight_cost = 0.5, 0.25, 0.25
    if any(k in text for k in ["quality", "accuracy", "best"]):
        weight_quality += 0.2
        weight_speed -= 0.1
        weight_cost -= 0.1
    if any(k in text for k in ["fast", "latency", "real-time", "speed"]):
        weight_speed += 0.2
        weight_quality -= 0.1
        weight_cost -= 0.1
    if any(k in text for k in ["cheap", "budget", "$", "cost"]):
        weight_cost += 0.2
        weight_quality -= 0.1
        weight_speed -= 0.1

    total = max(weight_quality + weight_speed + weight_cost, 1e-9)
    providers = {p.strip().lower() for p in provider_allowlist.split(",")} if provider_allowlist else None
    return TaskProfile(
        task_type=task_type,
        weight_quality=weight_quality / total,
        weight_speed=weight_speed / total,
        weight_cost=weight_cost / total,
        max_price_per_1m=max_price_per_1m,
        min_context=min_context,
        provider_allowlist=providers,
    )


def _first_non_null(row: pd.Series, cols: list[str]) -> float:
    for col in cols:
        value = row.get(col)
        if pd.notna(value):
            return float(value)
    return 0.0


def _normalize_series(series: pd.Series, invert: bool = False, *, missing_policy: str = "neutral") -> pd.Series:
    """
    Normalize into [0,1].

    missing_policy:
      - "neutral": fill missing with median
      - "penalize": fill missing with worst-case for the metric
          * for non-inverted metrics (higher is better), missing -> min
          * for inverted metrics (lower is better), missing -> max
    """
    s = series.astype("float64")

    if s.dropna().empty:
        filled = s.fillna(0.0)
    else:
        if missing_policy == "neutral":
            filled = s.fillna(float(s.median()))
        elif missing_policy == "penalize":
            penalty = float(s.max()) if invert else float(s.min())
            filled = s.fillna(penalty)
        else:
            raise ValueError(f"unknown missing_policy={missing_policy!r}")

    min_val, max_val = float(filled.min()), float(filled.max())
    if abs(max_val - min_val) < 1e-9:
        norm = pd.Series([0.5] * len(filled), index=filled.index)
    else:
        norm = (filled - min_val) / (max_val - min_val)

    return 1 - norm if invert else norm


def _latest_snapshot_ts_utc() -> datetime | None:
    """
    Read max(snapshot_ts) from models_latest. Returns timezone-aware UTC datetime if available.
    If table doesn't exist / empty / error, returns None.
    """
    con = connect()
    try:
        init_warehouse(con)
        row = con.execute("SELECT max(snapshot_ts) AS max_ts FROM models_latest").fetchone()
        max_ts = row[0] if row else None
        if max_ts is None:
            return None

        # duckdb returns Python datetime (naive). Treat as UTC.
        if isinstance(max_ts, datetime):
            return max_ts.replace(tzinfo=timezone.utc) if max_ts.tzinfo is None else max_ts.astimezone(timezone.utc)

        # fallback: parse string
        parsed = datetime.fromisoformat(str(max_ts))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except Exception:
        return None
    finally:
        con.close()


def _maybe_refresh_warehouse(*, refresh: bool, max_age_hours: float) -> str | None:
    """
    Option A: auto-refresh on recommend.
    - If refresh=True: always run ingest.
    - Else: run ingest only if latest snapshot is older than max_age_hours or missing.

    Returns a warning string if refresh was attempted but failed; otherwise None.
    """
    now = datetime.now(tz=timezone.utc)
    latest = _latest_snapshot_ts_utc()

    needs_refresh = refresh or latest is None or (now - latest) > timedelta(hours=max_age_hours)
    if not needs_refresh:
        return None

    try:
        # Lazy import to avoid any circular import headaches at module import time.
        from src.ingest import run_ingest

        run_ingest()
        return None
    except Exception as exc:
        # If refresh fails, we still proceed using existing warehouse state.
        return f"Refresh failed; using existing warehouse snapshot. Error: {exc!s}"


def recommend(
    task_profile: TaskProfile,
    topk: int,
    *,
    missing_policy: str = "penalize",
    refresh: bool = False,
    max_age_hours: float = 24.0,
) -> dict:
    refresh_warning = _maybe_refresh_warehouse(refresh=refresh, max_age_hours=max_age_hours)

    con = connect()
    try:
        init_warehouse(con)
        df = con.execute("SELECT * FROM models_latest").fetch_df()
    finally:
        con.close()

    if df.empty:
        payload = {"task_profile": asdict(task_profile), "snapshot_ts": None, "recommendations": []}
        if refresh_warning:
            payload["warning"] = refresh_warning
        return _json_safe(payload)

    key_fields = ["provider", "price_input_per_1m", "output_tokens_per_s", "context_window"]
    null_rates = {field: float(df[field].isna().mean()) for field in key_fields if field in df.columns}
    data_quality_warning: str | None = None
    if null_rates and all(rate >= 0.98 for rate in null_rates.values()):
        data_quality_warning = (
            "models_latest has near-empty provider/price/speed/context fields; "
            "recommendations may be low-confidence due to source schema mismatch"
        )

    if task_profile.max_price_per_1m is not None:
        df = df[(df["price_input_per_1m"] <= task_profile.max_price_per_1m) | (df["price_input_per_1m"].isna())]
    if task_profile.min_context is not None:
        df = df[(df["context_window"] >= task_profile.min_context) | (df["context_window"].isna())]
    if task_profile.provider_allowlist:
        providers = df["provider"].fillna("").str.lower()
        df = df[providers.isin(task_profile.provider_allowlist)]

    if df.empty:
        payload = {"task_profile": asdict(task_profile), "snapshot_ts": None, "recommendations": []}
        if refresh_warning:
            payload["warning"] = refresh_warning
        return _json_safe(payload)

    quality_cols = {
        "coding": ["coding_index", "quality_index", "reasoning_index"],
        "math": ["math_index", "reasoning_index", "quality_index"],
        "reasoning": ["reasoning_index", "quality_index"],
        "writing": ["quality_index", "reasoning_index"],
        "rag": ["reasoning_index", "quality_index"],
        "agent": ["reasoning_index", "coding_index", "quality_index"],
        "general": ["quality_index", "reasoning_index", "coding_index", "math_index"],
    }
    selected_quality_cols = quality_cols[task_profile.task_type]
    if not any(df[col].notna().any() for col in selected_quality_cols):
        payload = {
            "task_profile": asdict(task_profile),
            "snapshot_ts": str(df["snapshot_ts"].max()),
            "recommendations": [],
            "warning": data_quality_warning
            or "No quality metrics are available for the selected task type in models_latest.",
        }
        if refresh_warning:
            payload["warning"] = f"{payload['warning']} | {refresh_warning}"
        return _json_safe(payload)

    df["quality_metric"] = df.apply(lambda row: _first_non_null(row, selected_quality_cols), axis=1)
    df["quality_norm"] = _normalize_series(df["quality_metric"], missing_policy="neutral")

    df["speed_norm"] = (
        _normalize_series(df["output_tokens_per_s"], missing_policy=missing_policy) * 0.7
        + _normalize_series(df["ttft_s"], invert=True, missing_policy=missing_policy) * 0.3
    )
    df["cost_norm"] = _normalize_series(df["price_input_per_1m"], invert=True, missing_policy=missing_policy)

    df["score"] = (
        task_profile.weight_quality * df["quality_norm"]
        + task_profile.weight_speed * df["speed_norm"]
        + task_profile.weight_cost * df["cost_norm"]
    )

    ranked = df.sort_values("score", ascending=False).head(topk)
    snapshot_ts = str(ranked["snapshot_ts"].max()) if not ranked.empty else None

    recs: list[dict[str, Any]] = []
    for _, row in ranked.iterrows():
        recs.append(
            {
                "canonical_model_key": row["canonical_model_key"],
                "model_name": row["model_name"],
                "provider": row.get("provider"),
                "score": round(float(row["score"]), 4),
                "metrics": {
                    "quality_metric": row.get("quality_metric"),
                    "output_tokens_per_s": row.get("output_tokens_per_s"),
                    "price_input_per_1m": row.get("price_input_per_1m"),
                    "context_window": row.get("context_window"),
                },
                "justification": (
                    f"Strong {task_profile.task_type} quality signal with normalized quality {row['quality_norm']:.2f}. "
                    f"Speed {row['speed_norm']:.2f} and cost-fit {row['cost_norm']:.2f} under your preferences."
                ),
                "snapshot_ts": str(row["snapshot_ts"]),
            }
        )

    payload: dict[str, Any] = {"task_profile": asdict(task_profile), "snapshot_ts": snapshot_ts, "recommendations": recs}

    warnings: list[str] = []
    if data_quality_warning:
        warnings.append(data_quality_warning)
    if refresh_warning:
        warnings.append(refresh_warning)
    if warnings:
        payload["warning"] = " | ".join(warnings)

    return _json_safe(payload)


def _extract_budget_from_text(task: str) -> float | None:
    """
    Extract budget like:
      "$5/1M", "$5 / 1m tokens", "5 per 1m", "5/1 million", "$5 per 1 million"
    Returns dollars per 1M tokens (input) as float.
    """
    text = task.lower()

    patterns = [
        r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/|per)\s*1\s*m\b",  # $5/1m
        r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/|per)\s*1\s*m\s*tokens?\b",  # $5/1m tokens
        r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/|per)\s*1\s*m\s*tok(?:ens?)?\b",  # $5/1m tok
        r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/|per)\s*1\s*million\b",  # $5/1 million
        r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/|per)\s*1\s*million\s*tokens?\b",  # $5/1 million tokens
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:/|per)\s*1\s*m\b",  # 5/1m
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:/|per)\s*1\s*million\b",  # 5/1 million
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend LLMs from latest warehouse snapshot")
    parser.add_argument("task_text", help="Natural language task description")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-price-per-1m", type=float, default=None)
    parser.add_argument("--min-context", type=int, default=None)
    parser.add_argument("--provider-allowlist", type=str, default=None, help="Comma-separated providers")
    parser.add_argument(
        "--missing-policy",
        type=str,
        default="penalize",
        choices=["neutral", "penalize"],
        help="How to treat missing speed/cost metrics during scoring",
    )

    # Option A: auto-refresh on recommend
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh by running ingestion before recommending (falls back to existing data if refresh fails).",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=24.0,
        help="If the latest snapshot is older than this, run ingestion before recommending (ignored if --refresh).",
    )

    args = parser.parse_args()

    max_price = args.max_price_per_1m if args.max_price_per_1m is not None else _extract_budget_from_text(args.task_text)
    profile = parse_task_profile(
        task_text=args.task_text,
        max_price_per_1m=max_price,
        min_context=args.min_context,
        provider_allowlist=args.provider_allowlist,
    )

    result = recommend(
        profile,
        topk=args.topk,
        missing_policy=args.missing_policy,
        refresh=args.refresh,
        max_age_hours=args.max_age_hours,
    )
    print(json.dumps(_json_safe(result), indent=2))


if __name__ == "__main__":
    main()