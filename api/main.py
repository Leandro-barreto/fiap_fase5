"""
main.py – entry point for the FastAPI application
-------------------------------------------------

This module creates and configures a FastAPI application.  It includes
a simple home page rendered from an HTML template and registers the
prediction routes.  Monitoring via Prometheus is also enabled.  The
structure mirrors the example found in the original repository's
``api/main.py``【285454945483553†L0-L18】.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .routes.predict import router as predict_router
from .monitoring import setup_monitoring


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application.

    Returns
    -------
    FastAPI
        A configured FastAPI application ready to serve requests.
    """
    app = FastAPI(title="ML Prediction API")

    # register monitoring before other routes
    setup_monitoring(app)

    # configure template directory for the home page
    templates = Jinja2Templates(directory="new_api/static")

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        """Serve a simple home page with a form to trigger predictions.

        This endpoint renders an HTML template that allows users to
        specify a ticker and prediction date or upload a CSV file.  If
        you don't need a front‑end, you can remove this route and rely
        solely on the JSON/CSV prediction endpoints.
        """
        return templates.TemplateResponse("home.html", {"request": request})

    # include our prediction routes under the /api prefix
    app.include_router(predict_router, prefix="/api")

    return app


# if run as a script, start the application using uvicorn
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
