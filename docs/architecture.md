# Architecture Overview

This project follows a modular architecture suitable for incremental AI/ML and software development.

## Layers

- Data layer: raw, processed, train, validation, and test metadata plus sample/demo images
- AI layer: preprocessing, detection, classification, intensity estimation, track prediction, explainability
- Backend layer: FastAPI-based APIs, SQLAlchemy models, and service logic
- Frontend layer: React + Vite dashboard for analysis and monitoring

## Design principles

- Start with a local demo dataset and simple baselines
- Prevent data leakage by grouping records by cyclone event
- Keep model and data flows modular and configurable
- Use environment variables to manage secrets and deployment settings

## Phase 1 status

The project currently includes only the initial data-management foundation and preprocessing utilities.
