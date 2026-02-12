from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict

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


def parse_task_profile(task_text: str, max_price_per_1m: float | None, min_context: int | None, provider_allowlist: str | None) -> TaskProfile:
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


def _normalize_series(series: pd.Series, invert: bool = False) -> pd.Series:
    filled = series.fillna(series.median() if not series.dropna().empty else 0.0)
    min_val, max_val = float(filled.min()), float(filled.max())
    if abs(max_val - min_val) < 1e-9:
        norm = pd.Series([0.5] * len(filled), index=filled.index)
    else:
        norm = (filled - min_val) / (max_val - min_val)
    return 1 - norm if invert else norm


def recommend(task_profile: TaskProfile, topk: int) -> dict:
    con = connect()
    try:
        init_warehouse(con)
        df = con.execute("SELECT * FROM models_latest").fetch_df()
    finally:
        con.close()

    if df.empty:
        return {"task_profile": asdict(task_profile), "snapshot_ts": None, "recommendations": []}

    if task_profile.max_price_per_1m is not None:
        df = df[(df["price_input_per_1m"] <= task_profile.max_price_per_1m) | (df["price_input_per_1m"].isna())]
    if task_profile.min_context is not None:
        df = df[(df["context_window"] >= task_profile.min_context) | (df["context_window"].isna())]
    if task_profile.provider_allowlist:
        providers = df["provider"].fillna("").str.lower()
        df = df[providers.isin(task_profile.provider_allowlist)]

    if df.empty:
        return {"task_profile": asdict(task_profile), "snapshot_ts": None, "recommendations": []}

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
    df["quality_metric"] = df.apply(lambda row: _first_non_null(row, selected_quality_cols), axis=1)
    df["quality_norm"] = _normalize_series(df["quality_metric"])
    df["speed_norm"] = _normalize_series(df["output_tokens_per_s"]) * 0.7 + _normalize_series(df["ttft_s"], invert=True) * 0.3
    df["cost_norm"] = _normalize_series(df["price_input_per_1m"], invert=True)

    df["score"] = (
        task_profile.weight_quality * df["quality_norm"]
        + task_profile.weight_speed * df["speed_norm"]
        + task_profile.weight_cost * df["cost_norm"]
    )

    ranked = df.sort_values("score", ascending=False).head(topk)
    snapshot_ts = str(ranked["snapshot_ts"].max()) if not ranked.empty else None
    recs = []
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

    return {"task_profile": asdict(task_profile), "snapshot_ts": snapshot_ts, "recommendations": recs}


def _extract_budget_from_text(task: str) -> float | None:
    match = re.search(r"\$\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*1m", task.lower())
    if not match:
        return None
    return float(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend LLMs from latest warehouse snapshot")
    parser.add_argument("task_text", help="Natural language task description")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-price-per-1m", type=float, default=None)
    parser.add_argument("--min-context", type=int, default=None)
    parser.add_argument("--provider-allowlist", type=str, default=None, help="Comma-separated providers")
    args = parser.parse_args()

    max_price = args.max_price_per_1m if args.max_price_per_1m is not None else _extract_budget_from_text(args.task_text)
    profile = parse_task_profile(
        task_text=args.task_text,
        max_price_per_1m=max_price,
        min_context=args.min_context,
        provider_allowlist=args.provider_allowlist,
    )
    result = recommend(profile, topk=args.topk)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
