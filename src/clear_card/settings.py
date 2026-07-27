from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path | None = None) -> None:
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # Explicit process variables take precedence over convenience values in `.env`.
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class RepositoryPaths:
    root: Path = PROJECT_ROOT

    @property
    def processed_data(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def sample_data(self) -> Path:
        return self.root / "data" / "sample.xlsx"

    @property
    def journal_metrics(self) -> Path:
        return self.root / "data" / "journal_metrics.csv"

    @property
    def expected_data(self) -> Path:
        return self.root / "data" / "expected"

    @property
    def figures(self) -> Path:
        return self.root / "outputs" / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "outputs" / "tables"

    def create_outputs(self) -> None:
        self.figures.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DatabaseSettings:
    driver: str
    server: str
    port: int
    database: str
    username: str
    password: str
    encrypt: str
    trust_server_certificate: str

    @classmethod
    def from_env(cls, database: str) -> DatabaseSettings:
        load_env()
        database_key = "SCOPUS_DATABASE" if database == "scopus" else "ICITE_DATABASE"
        values = {
            "driver": os.getenv("MSSQL_DRIVER", "ODBC Driver 17 for SQL Server"),
            "server": os.getenv("MSSQL_SERVER", ""),
            "port": int(os.getenv("MSSQL_PORT", "1433")),
            "database": os.getenv(database_key, ""),
            "username": os.getenv("MSSQL_USERNAME", ""),
            "password": os.getenv("MSSQL_PASSWORD", ""),
            "encrypt": os.getenv("MSSQL_ENCRYPT", "yes"),
            "trust_server_certificate": os.getenv(
                "MSSQL_TRUST_SERVER_CERTIFICATE", "no"
            ),
        }
        missing = [
            key
            for key in ("server", "database", "username", "password")
            if not values[key]
        ]
        if missing:
            raise ValueError(f"Missing database settings: {', '.join(missing)}")
        return cls(**values)

    @staticmethod
    def _quote(value: object) -> str:
        return "{" + str(value).replace("}", "}}") + "}"

    def connection_string(self) -> str:
        fields = {
            "DRIVER": self.driver,
            "SERVER": f"{self.server},{self.port}",
            "DATABASE": self.database,
            "UID": self.username,
            "PWD": self.password,
            "Encrypt": self.encrypt,
            "TrustServerCertificate": self.trust_server_certificate,
        }
        return (
            ";".join(f"{key}={self._quote(value)}" for key, value in fields.items())
            + ";"
        )
