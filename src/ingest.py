from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.connectors.artificial_analysis import ArtificialAnalysisConnector, ArtificialAnalysisError
from src.connectors.fixture import FixtureConnector
from src.schema import normalize_records, utc_now_iso
from src.warehouse import connect, init_warehouse


def choose_source() -> tuple[str, list[dict]]:
    connector = ArtificialAnalysisConnector()
    try:
        records = connector.fetch()
        return "artificial_analysis", records
    except ArtificialAnalysisError:
        fixture_records = FixtureConnector().fetch()
        return "fixture", fixture_records


def write_snapshot(frame: pd.DataFrame, source: str, snapshot_ts: str) -> Path:
    safe_ts = snapshot_ts.replace(":", "-")
    out_dir = Path("data") / "bronze" / source
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"models_snapshot_{safe_ts}.parquet"
    frame.to_parquet(out_path, index=False)
    return out_path


def load_snapshot_into_duckdb(snapshot_path: Path) -> None:
    con = connect()
    try:
        init_warehouse(con)
        con.execute("INSERT INTO bronze_models SELECT * FROM read_parquet(?)", [str(snapshot_path)])
    finally:
        con.close()


def run_ingest() -> Path:
    source, records = choose_source()
    snapshot_ts = utc_now_iso()
    normalized = normalize_records(records=records, source=source, snapshot_ts=snapshot_ts)
    snapshot_path = write_snapshot(normalized, source=source, snapshot_ts=snapshot_ts)
    load_snapshot_into_duckdb(snapshot_path)
    return snapshot_path


if __name__ == "__main__":
    path = run_ingest()
    print(f"Ingested snapshot: {path}")
