from __future__ import annotations

from pathlib import Path

import pandas as pd

from clear_card.settings import DatabaseSettings, PROJECT_ROOT


def connect(database: str):
    try:
        import pyodbc
    except ImportError as exc:
        raise RuntimeError("Install the 'db' extra to use SQL Server support.") from exc
    settings = DatabaseSettings.from_env(database)
    return pyodbc.connect(settings.connection_string())


def check_connection(database: str) -> str:
    with connect(database) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT DB_NAME()")
        return str(cursor.fetchone()[0])


def extract_citation_edges(database: str, sql_path: Path | None = None) -> pd.DataFrame:
    query_path = sql_path or PROJECT_ROOT / "sql" / "extract_citation_edges.sql"
    query = query_path.read_text(encoding="utf-8")
    with connect(database) as connection:
        return pd.read_sql_query(query, connection)
