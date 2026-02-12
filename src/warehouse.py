from __future__ import annotations

from pathlib import Path

import duckdb


def connect(db_path: str | Path = "data/warehouse.duckdb") -> duckdb.DuckDBPyConnection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    return con


def init_warehouse(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bronze_models (
            snapshot_ts TIMESTAMP,
            source VARCHAR,
            model_name VARCHAR,
            provider VARCHAR,
            quality_index DOUBLE,
            coding_index DOUBLE,
            math_index DOUBLE,
            reasoning_index DOUBLE,
            output_tokens_per_s DOUBLE,
            ttft_s DOUBLE,
            price_input_per_1m DOUBLE,
            price_output_per_1m DOUBLE,
            context_window BIGINT,
            is_open_source BOOLEAN,
            license VARCHAR,
            canonical_model_key VARCHAR
        );
        """
    )

    con.execute(
        """
        CREATE OR REPLACE VIEW silver_models AS
        SELECT
            snapshot_ts,
            source,
            model_name,
            provider,
            CAST(quality_index AS DOUBLE) AS quality_index,
            CAST(coding_index AS DOUBLE) AS coding_index,
            CAST(math_index AS DOUBLE) AS math_index,
            CAST(reasoning_index AS DOUBLE) AS reasoning_index,
            CAST(output_tokens_per_s AS DOUBLE) AS output_tokens_per_s,
            CAST(ttft_s AS DOUBLE) AS ttft_s,
            CAST(price_input_per_1m AS DOUBLE) AS price_input_per_1m,
            CAST(price_output_per_1m AS DOUBLE) AS price_output_per_1m,
            CAST(context_window AS BIGINT) AS context_window,
            CAST(is_open_source AS BOOLEAN) AS is_open_source,
            license,
            canonical_model_key
        FROM bronze_models;
        """
    )

    con.execute("CREATE OR REPLACE VIEW models_history AS SELECT * FROM silver_models;")

    con.execute(
        """
        CREATE OR REPLACE VIEW models_latest AS
        SELECT s.*
        FROM silver_models s
        INNER JOIN (
            SELECT canonical_model_key, MAX(snapshot_ts) AS max_snapshot_ts
            FROM silver_models
            GROUP BY canonical_model_key
        ) latest
        ON s.canonical_model_key = latest.canonical_model_key
        AND s.snapshot_ts = latest.max_snapshot_ts;
        """
    )
