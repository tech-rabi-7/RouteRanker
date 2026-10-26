"""
backend/config.py
=================
Application settings and configuration for RouteRanker v2.0 Backend.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    # App
    PROJECT_NAME: str = "RouteRanker v2.0 API"
    VERSION: str = "2.0.0"
    API_PREFIX: str = "/api"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1")

    # Directories
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    MODELS_DIR: Path = BASE_DIR / "models"

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/routeranker",
    )
    SQLITE_FALLBACK_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'routeranker.db'}"
    ALLOW_SQLITE_FALLBACK: bool = os.getenv(
        "ALLOW_SQLITE_FALLBACK", "true"
    ).lower() in ("true", "1")

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",
    ]


settings = Settings()
