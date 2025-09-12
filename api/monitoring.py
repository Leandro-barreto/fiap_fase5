"""
monitoring.py – integrate Prometheus instrumentation with FastAPI
----------------------------------------------------------------

This module wraps the Prometheus instrumentation provided by
``prometheus_fastapi_instrumentator`` into a function that can be
called during app startup.  See the original implementation in
``api/monitoring.py``【629603242625605†L0-L3】 for a minimal example.
"""

from prometheus_fastapi_instrumentator import Instrumentator


def setup_monitoring(app) -> None:
    """Register Prometheus instrumentation with the FastAPI app.

    This function exposes an endpoint at ``/metrics`` that Prometheus
    scrapes to collect metrics such as request latencies and counts.
    """
    Instrumentator().instrument(app).expose(app)
