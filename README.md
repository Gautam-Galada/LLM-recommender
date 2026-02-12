# LLM Recommender (Local Parquet + DuckDB)

<p align="center">
  <img width="522" height="713" alt="image"
       src="https://github.com/user-attachments/assets/e985892f-b907-40df-9b1e-addad58d92a0" />
</p>



A local, reproducible data system for ingesting LLM benchmark, pricing, and performance data and ranking models for a given task using a multi-objective scoring engine.

---

## Stack

* Python 3.11+
* DuckDB
* pandas
* pyarrow

---

## Project Layout

```
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

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configure API Key (Optional but Recommended)

```bash
export AA_API_KEY="your_key_here"
```

If `AA_API_KEY` is missing or the API fails:

* The system falls back to `data/fixtures/fixture.json`

API keys should never be committed.

---

## Ingest

```bash
python -m src.ingest
```

### What Happens

1. Fetch records from Artificial Analysis API
   `/api/v2/data/llms/models`

2. Normalize into canonical schema:

   * quality_index
   * coding_index
   * math_index
   * output_tokens_per_s
   * ttft_s
   * price_input_per_1m
   * price_output_per_1m
   * provider
   * canonical_model_key

3. Write append-only Parquet snapshot to:

```
data/bronze/<source>/models_snapshot_<timestamp>.parquet
```

4. Load into DuckDB warehouse:

   * bronze_models (append-only table)
   * silver_models (clean view)
   * models_history (all records)
   * models_latest (latest row per model)

---

## Recommend

```bash
python -m src.recommend "I need a model for python debugging, prefer quality, budget $5/1M tokens" --topk 5
```

### Optional Flags

* `--max-price-per-1m 5`
* `--min-context 32768`
* `--provider-allowlist "openai,google"`
* `--missing-policy penalize|neutral`

Default missing policy = `penalize`

---

## How Recommendation Works

The system implements a **multi-objective weighted scoring model** over structured metrics.

It is deterministic and does not use embeddings or ML.

---

### Step 1 — Task Parsing

Natural language input is parsed into:

```python
TaskProfile(
  task_type,
  weight_quality,
  weight_speed,
  weight_cost,
  max_price_per_1m,
  min_context,
  provider_allowlist
)
```

Example:

```
"python debugging, budget $5/1M tokens"
```

→ task_type = coding  
→ cost weight increases  
→ budget parsed automatically from text

---

### Step 2 — Feature Selection (by task type)

For each task type, the system selects quality features:

| Task Type | Quality Columns Used                                        |
| --------- | ----------------------------------------------------------- |
| coding    | coding_index → quality_index → reasoning_index              |
| math      | math_index → reasoning_index → quality_index                |
| reasoning | reasoning_index → quality_index                             |
| writing   | quality_index → reasoning_index                             |
| agent     | reasoning_index → coding_index → quality_index              |
| general   | quality_index → reasoning_index → coding_index → math_index |

The first non-null metric is used as `quality_metric`.

---

### Step 3 — Features Used in Scoring

Each model is scored using:

#### 1️⃣ Quality

```
quality_norm = normalized(quality_metric)
```

Higher is better.

---

#### 2️⃣ Speed

Composite metric:

```
speed_norm =
    0.7 * normalized(output_tokens_per_s)
  + 0.3 * normalized_inverse(ttft_s)
```

Higher throughput → better  
Lower latency → better

---

#### 3️⃣ Cost

```
cost_norm = normalized_inverse(price_input_per_1m)
```

Lower price → higher score

---

### Step 4 — Missing Data Handling

Configurable via:

```
--missing-policy penalize | neutral
```

Default: penalize

If a model is missing:

* throughput
* latency
* price

Then it receives worst-case normalization for that metric.

This prevents incomplete data from appearing artificially competitive.

---

### Step 5 — Final Score

```
score =
  w_quality * quality_norm
+ w_speed   * speed_norm
+ w_cost    * cost_norm
```

Models are sorted by descending score.

---

## Output

Returns JSON:

```json
{
  "task_profile": {...},
  "snapshot_ts": "...",
  "recommendations": [
    {
      "model_name": "...",
      "provider": "...",
      "score": 0.85,
      "metrics": {...},
      "justification": "..."
    }
  ]
}
```

Justification reflects normalized feature values used in scoring.

---

## Auto-Refresh Behavior

The recommendation pipeline:

1. Checks warehouse
2. Auto-refreshes snapshot if needed
3. Uses latest snapshot timestamp
4. Ranks from models_latest view

This ensures results are based on the most recent ingestion.

---

## Warehouse Model

### bronze_models (table)

Append-only raw snapshots

### silver_models (view)

Canonical schema

### models_history (view)

All records across time

### models_latest (view)

Latest row per canonical_model_key

---

## Example Cron

Run ingest every 6 hours:

```cron
0 */6 * * * cd /workspace/LLM-recommender && /usr/bin/env bash -lc 'source .venv/bin/activate && python -m src.ingest >> logs/ingest.log 2>&1'
```

---

## Features Actually Being Read

From `models_latest`, the recommender reads:

* coding_index
* quality_index
* math_index
* reasoning_index
* output_tokens_per_s
* ttft_s
* price_input_per_1m
* context_window
* provider
* snapshot_ts
* canonical_model_key

These are the only inputs to the ranking function.

---

## Attribution

Artificial Analysis data requires attribution:  
[https://artificialanalysis.ai/](https://artificialanalysis.ai/)

---

## Current System Characteristics

### This is:

* Deterministic
* Linear multi-objective scoring
* Task-aware metric selection
* Missing-data robust
* Snapshot-based reproducible
* Offline-first

### Targetted towards enabling:

* Embedding-based
* Learned ranking
* Personalized
* Online adaptive

