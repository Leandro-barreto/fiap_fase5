"""Entry point for the hiring prediction FastAPI application.

This module sets up a basic FastAPI app, configures monitoring and
registers a route for candidate hiring predictions. The API is
purposely simple: it exposes a single endpoint that accepts a CSV file
containing feature columns and returns binary predictions (0 = não
aprovado, 1 = aprovado).
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
    app = FastAPI(title="API de Predição de Contratação")
    # register monitoring before other routes
    setup_monitoring(app)
    # register prediction routes
    app.include_router(predict_router, prefix="/api")
    # configure template engine for the home page
    templates = Jinja2Templates(directory="api/static")

    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        """Serve the landing page for candidate prediction."""
        return templates.TemplateResponse("home.html", {"request": request})

    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)