# Database Design

This project uses SQLAlchemy in a local demo-friendly configuration.

## Tables

- cyclones
- satellite_images
- observations
- predictions
- track_points
- model_results

## Purpose

These tables support the planned pipeline for storing cyclone metadata, image references, observations, forecast outputs, track points, and model evaluation results.

## Phase 8 status

The current implementation creates the table schema and initializes it via a local SQLite-backed demo configuration. PostgreSQL-ready models are in place for future production deployment.
