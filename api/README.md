# 🌐 api/ - API FastAPI

Serve o modelo **LightGBM** salvo em `models/lgbm_model.joblib`. Página `static/home.html` com seções de **Vaga** e **Candidato**.

## Endpoints
- `GET /` – página inicial (form).
- `POST /api/predict/candidate` – predição via JSON ou CSV (multipart).

## Execução local
```bash
uvicorn api.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```
