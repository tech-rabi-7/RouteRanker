"""
backend/database.py
===================
Database connection & DuckDB analytical query layer for RouteRanker v2.0.
Supports PostgreSQL, SQLite fallback, and direct fast Parquet querying via DuckDB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import duckdb
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import settings

# ---------------------------------------------------------------------------
# SQLAlchemy Relational Engine
# ---------------------------------------------------------------------------

Base = declarative_base()

def get_engine():
    try:
        engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
        # Test connection
        with engine.connect():
            return engine
    except Exception:
        if settings.ALLOW_SQLITE_FALLBACK:
            return create_engine(
                settings.SQLITE_FALLBACK_URL,
                connect_args={"check_same_thread": False},
            )
        raise

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# DuckDB Parquet Analytical Layer
# ---------------------------------------------------------------------------

class DuckDBRepository:
    """Provides high-performance analytical queries directly on processed Parquet files."""

    def __init__(self, processed_dir: Optional[Path] = None):
        self.processed_dir = processed_dir or settings.PROCESSED_DATA_DIR

    def query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        con = duckdb.connect(":memory:")
        # Register views for all available processed parquet files
        for f in self.processed_dir.glob("*.parquet"):
            tbl_name = f.stem
            con.execute(f"CREATE VIEW {tbl_name} AS SELECT * FROM read_parquet('{f}')")
        try:
            if params:
                df = con.execute(sql, params).df()
            else:
                df = con.execute(sql).df()
            return df
        finally:
            con.close()

    def get_table(self, table_name: str) -> Optional[pd.DataFrame]:
        path = self.processed_dir / f"{table_name}.parquet"
        if not path.exists():
            return None
        return duckdb.read_parquet(str(path)).df()


duckdb_repo = DuckDBRepository()
