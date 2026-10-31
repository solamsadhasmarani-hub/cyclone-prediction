from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path = BASE_DIR
    backend_dir: Path = BASE_DIR / "backend"
    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    train_dir: Path = data_dir / "train"
    validation_dir: Path = data_dir / "validation"
    test_dir: Path = data_dir / "test"
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///" + str((BASE_DIR / "data" / "sih_demo.db")))
    secret_key: str = os.getenv("SECRET_KEY", "local-dev-secret")
    app_name: str = os.getenv("APP_NAME", "SIH26070")
    environment: str = os.getenv("ENVIRONMENT", "development")
    cors_origins: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    )


settings = Settings()
