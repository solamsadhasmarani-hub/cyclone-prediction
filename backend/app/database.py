from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.app.config import settings

engine = create_engine(settings.database_url, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db_session():
    """Provide a database session for later application layers."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Initialize database tables for the demo/local prototype."""
    from backend.app.models.database_models import Base as ModelBase

    ModelBase.metadata.create_all(bind=engine)
