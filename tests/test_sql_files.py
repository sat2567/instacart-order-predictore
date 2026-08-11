"""Static checks on the SQL files (no database needed)."""
from reorder import config, db
from reorder.features import NUMERIC_COLS


def test_split_statements_ignores_comments():
    stmts = [s for s in db.split_statements("-- a; comment\nSELECT 1;\nSELECT 2;\n") if s.strip()]
    assert [s.strip() for s in stmts] == ["SELECT 1", "SELECT 2"]


def test_named_queries_parse():
    names = [n for n, _ in db.named_queries(config.SQL_DIR / "benchmark_queries.sql")]
    assert names == ["user_order_history", "single_order_basket", "user_reorder_rate"]


def test_candidates_view_exposes_every_model_feature():
    sql = (config.SQL_DIR / "03_features.sql").read_text()
    view = sql[sql.index("CREATE VIEW feat.candidates"):]
    missing = [c for c in NUMERIC_COLS + ["up_bitmask", "gaps_last20"] if c not in view]
    assert not missing, missing
