from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Cyclone(Base):
    __tablename__ = "cyclones"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    status = Column(String, default="demo")
    description = Column(Text, nullable=True)


class SatelliteImage(Base):
    __tablename__ = "satellite_images"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cyclone_id = Column(String, nullable=True, index=True)
    image_path = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    is_cyclone = Column(Integer, default=0)


class Observation(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cyclone_id = Column(String, nullable=True, index=True)
    timestamp = Column(DateTime, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    pressure = Column(Float, nullable=True)
    intensity = Column(Float, nullable=True)
    classification = Column(String, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cyclone_id = Column(String, nullable=True, index=True)
    prediction_type = Column(String, nullable=False)
    model_name = Column(String, nullable=True)
    value_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=True)


class TrackPoint(Base):
    __tablename__ = "track_points"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cyclone_id = Column(String, nullable=True, index=True)
    timestamp = Column(DateTime, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    prediction_horizon_hours = Column(Integer, nullable=True)


class ModelResult(Base):
    __tablename__ = "model_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=True)
    metric_json = Column(Text, nullable=True)
