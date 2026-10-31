from sqlalchemy import create_engine, inspect

from backend.app.database import init_db
from backend.app.models.database_models import Base


def test_database_models_metadata_is_valid():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    expected = {"cyclones", "satellite_images", "observations", "predictions", "track_points", "model_results"}
    assert expected.issubset(set(tables))


def test_database_init_creates_tables():
    init_db()
    from backend.app.database import engine

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "cyclones" in tables
    assert "satellite_images" in tables
