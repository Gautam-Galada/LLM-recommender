# LLM Recommender (Local Parquet + DuckDB)

A local, reproducible data system for ingesting LLM benchmark/price/perf data and recommending models for a user task.

## Stack
- Python 3.11+
- DuckDB
- pandas
- pyarrow

## Project layout

```text
src/
  connectors/
    artificial_analysis.py
    fixture.py
  schema.py
  warehouse.py
  ingest.py
  recommend.py
data/
  fixtures/fixture.json
  bronze/<source>/models_snapshot_<timestamp>.parquet
tests/
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install duckdb pandas pyarrow pytest
```

## Configure API key (optional but preferred)

```bash
export AA_API_KEY="your_key_here"
```

If `AA_API_KEY` is missing (or the API call fails), ingest falls back to `data/fixtures/fixture.json`.

## Ingest

```bash
python -m src.ingest
```

What happens:
1. Fetch records from Artificial Analysis API (`/api/v2/data/llms/models`) when key exists.
2. Normalize to canonical schema.
3. Write append-only Parquet snapshot to `data/bronze/<source>/`.
4. Upsert warehouse metadata by appending to DuckDB bronze table and refreshing views.

## Recommend

```bash
python -m src.recommend "I need a model for python debugging, prefer quality, budget $5/1M tokens" --topk 5
```

Optional flags:
- `--max-price-per-1m 5`
- `--min-context 32768`
- `--provider-allowlist "meta,alibaba"`

Output is JSON with:
- parsed task profile
- snapshot timestamp used
- ranked recommendations (score, metrics, short justification)

## Warehouse model

- **bronze_models** (table): append-only ingested snapshots with `snapshot_ts` + `source`
- **silver_models** (view): canonicalized types/columns
- **models_history** (view): all canonical rows
- **models_latest** (view): latest row per `canonical_model_key`

## Cron scheduling (example)

Run ingest every 6 hours:

```cron
0 */6 * * * cd /workspace/LLM-recommender && /usr/bin/env bash -lc 'source .venv/bin/activate && python -m src.ingest >> logs/ingest.log 2>&1'
```

Create `logs/` first if needed.

## Notes
- Attribution required when using Artificial Analysis data: https://artificialanalysis.ai/
- API keys are sensitive: keep in env vars, never commit.
