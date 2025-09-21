# 📊 Projeto de Predição de Contratação

Pipeline de ML para estimar a probabilidade de contratação (modelo **LightGBM**), com API FastAPI e suporte a upload CSV.

## Pastas
- `api/` – aplicação FastAPI (rotas, HTML e carregamento do modelo).
- `src/` – scripts de features, treino e avaliação.
- `data/` – dados locais (não versionados).
- `docker/` – orquestração Prometheus/Grafana + API.
- `units/` – testes unitários (pytest).

## Como rodar
```bash
docker-compose up --build
# API: http://localhost:8000  |  Swagger: /docs
```
