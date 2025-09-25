#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======
Entry point da API FastAPI para predição de contratação.

"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from .routes.predict import router as predict_router
from .monitoring import setup_monitoring


def create_app() -> FastAPI:
    """Instancia e configura a aplicação FastAPI."""
    app = FastAPI(title="API de Predição de Contratação")

    # --- Monitoring (Prometheus / Grafana) ---
    # Ex.: expõe /metrics e adiciona middlewares de latência/contagem.
    setup_monitoring(app)

    # --- Static & Templates ---
    # monta a pasta /static para servir assets se necessário no futuro
    app.mount("/static", StaticFiles(directory="api/static"), name="static")
    templates = Jinja2Templates(directory="api/static")

    # --- Rotas de Predição ---
    # sem prefixo, pois o front consome /predict/... diretamente
    app.include_router(predict_router)

    # --- Home ---
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("home.html", {"request": request})

    # --- Healthcheck simples ---
    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


# Execução direta (útil em dev/local)
if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000, reload=True)
