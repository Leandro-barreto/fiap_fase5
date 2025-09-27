# 🚀 API – FastAPI Deployment

Esta pasta contém a **API** responsável por servir o modelo treinado em produção.  Ela utiliza o framework **FastAPI**, que oferece alta performance e geração automática de documentação para serviços REST.

## 📂 Estrutura

- **`main.py`** – ponto de entrada que constrói a aplicação via função factory e registra middlewares.
- **`routes/`** – módulo com os roteadores.  O arquivo `predict.py` implementa o endpoint `/api/predict/candidate` para processar entradas JSON ou arquivos CSV e retornar a probabilidade de match.
- **`model/`** – scripts utilitários para carregar o modelo serializado (por exemplo, `lgbm_model.joblib`).
- **`static/`** – contém arquivos estáticos como `home.html`, que apresenta um formulário web simples para preenchimento de vaga e candidato.
- **`monitoring.py`** – integra a API com **Prometheus**, expondo métricas no endpoint `/metrics`.

## ⚡ Rotas

- **`GET /`** – exibe a página inicial com a interface web (`home.html`).
- **`POST /api/predict/candidate`** – aceita dados de vaga e candidato (JSON ou arquivo CSV multipart) e devolve a pontuação de compatibilidade do modelo.
- **`GET /metrics`** – expõe métricas de desempenho no formato Prometheus.

A documentação interativa gerada automaticamente pelo FastAPI está disponível em **`/docs`** (Swagger UI) e **`/redoc`**.

## ⚙️ Sobre o FastAPI

**FastAPI** é um framework assíncrono para construção de APIs em Python.  Ele aproveita anotações de tipos para validar entradas e saídas, gerando automaticamente uma especificação OpenAPI e documentação interativa.  O resultado é um serviço de alta performance e com código conciso.

## ▶️ Execução local

Para iniciar a API localmente utilize o comando abaixo, que cria a aplicação e habilita o modo de recarregamento automático:

```bash
uvicorn api.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

Após iniciar, a API ficará acessível em http://localhost:8000 e a documentação interativa em http://localhost:8000/docs.
