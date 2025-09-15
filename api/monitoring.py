"""Monitoring setup for the FastAPI application.

This module exposes a simple helper to attach Prometheus instrumentation
to a FastAPI app using ``prometheus_fastapi_instrumentator``.  When
called, metrics will be available under the `/metrics` endpoint.
"""

from prometheus_fastapi_instrumentator import Instrumentator


def setup_monitoring(app) -> None:
    """Attach Prometheus instrumentation to a FastAPI app."""
    Instrumentator().instrument(app).expose(app)