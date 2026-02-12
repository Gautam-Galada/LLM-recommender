from src.warehouse import connect, init_warehouse


def test_warehouse_views_exist(tmp_path) -> None:
    db_path = tmp_path / "test.duckdb"
    con = connect(db_path)
    try:
        init_warehouse(con)
        con.execute(
            """
            INSERT INTO bronze_models VALUES
            ('2025-01-01 00:00:00', 'fixture', 'M1', 'P1', 50, 40, 30, 20, 10, 1, 2, 3, 4096, true, 'apache-2.0', 'p1::m1'),
            ('2025-01-02 00:00:00', 'fixture', 'M1', 'P1', 60, 50, 40, 30, 11, 1, 2, 3, 4096, true, 'apache-2.0', 'p1::m1');
            """
        )
        latest_quality = con.execute("SELECT quality_index FROM models_latest WHERE canonical_model_key='p1::m1'").fetchone()[0]
        history_count = con.execute("SELECT COUNT(*) FROM models_history").fetchone()[0]
        assert latest_quality == 60
        assert history_count == 2
    finally:
        con.close()
