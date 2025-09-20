"""Monitoring setup for the FastAPI application.

This module exposes a simple helper to attach Prometheus instrumentation
to a FastAPI app using ``prometheus_fastapi_instrumentator``.  When
called, metrics will be available under the `/metrics` endpoint.
"""

try:
    # Import the Prometheus instrumentator if available.  In environments
    # where the dependency is not installed (e.g. during unit tests), we
    # define a no‑op fallback that satisfies the same interface.
    from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore
except ImportError:  # pragma: no cover
    class Instrumentator:  # type: ignore[override]
        """Fallback instrumentator that performs no instrumentation."""

        def instrument(self, app):
            return self

        def expose(self, app):
            return self


def setup_monitoring(app) -> None:
    """Attach Prometheus instrumentation to a FastAPI app."""
    Instrumentator().instrument(app).expose(app)