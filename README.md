# 📊 Projeto de Predição de Contratação

Pipeline de ML para estimar a probabilidade de contratação (modelo **LightGBM**), com API FastAPI e suporte a upload CSV. Este repositório também inclui scripts de preparação de dados, engenharia de features, treinamento do modelo, inferência, dashboards de monitoramento e testes automatizados.

## 📁 Pastas Principais

- [`api/`](api/README.md) – contém a implementação da API FastAPI. O README dessa pasta descreve as rotas, estrutura de diretórios e como executar a API localmente com uvicorn.  
- [`data/`](data/README.md) – organização dos conjuntos de dados. Os dados brutos não são versionados; há um link para download via Google Drive.  
- [`docker/`](docker/README.md) – orquestração Docker Compose com três serviços: API, Prometheus e Grafana. Explica o papel de cada serviço, dashboards pré‑configurados e comandos úteis como ```bash docker ps``` e ```bash docker system prune```
- [`src/`](src/README.md) – código de preparação dos dados, extração de características (TF‑IDF, coeficiente de sobreposição, Jaccard etc.) e treinamento do modelo com LightGBM; também apresenta as métricas de desempenho e a curva ROC.  
- [`unit/`](unit/README.md) – testes unitários com pytest, incluindo o comando de execução e o relatório de cobertura de código.  

## ⚙️ Arquivo `main.py`

O arquivo main.py (na raiz) é um script de linha de comando que orquestra todas as etapas do pipeline:

1. ```bash --mode train``` – executa a preparação dos dados e a engenharia de features em src/data/, depois treina o modelo com LightGBM e salva os artefatos.

2. ```bash --mode predict``` – carrega o modelo treinado e executa inferência em um arquivo CSV especificado pelo parâmetro --inference-input

3. ```bash --mode all``` – executa a cadeia completa (treino e inferência) de ponta a ponta.

```bash
# Treino completo (prepare + feature engineering + treino)
python main.py --mode train

# Apenas predição (usando modelo treinado)
python main.py --mode predict --inference-input data/external/synthetic_inference_data.csv

# Pipeline completo (treino + predição)
python main.py --mode all
```

## 📦 Requisitos e Instalação

Crie um ambiente virtual e instale as dependências listadas em `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate    # Windows

pip install --upgrade pip
pip install -r requirements.txt
```
O arquivo requirements.txt lista dependências de machine learning (scikit‑learn, LightGBM, SHAP), da API (FastAPI e Uvicorn), ferramentas de monitoramento (Prometheus), testes (pytest, pytest‑cov) e utilidades diversas

Principais dependências: `scikit-learn`, `pandas`, `numpy`, `lightgbm`, `fastapi`, `uvicorn`, `pytest`, `prometheus-client`, entre outras.

## ▶️ Como Executar
Principais formas de rodar este projeto:

1. Via Docker Compose: a forma recomendada para ambiente de produção. Execute:
```bash
docker-compose up --build
```
Esse comando inicializa os containers da API, Prometheus e Grafana.

- API: [http://localhost:8000](http://localhost:8000)  
- Prometheus: [http://localhost:9090](http://localhost:9090)  
- Grafana: [http://localhost:3000](http://localhost:3000)


2. Localmente com Uvicorn: para desenvolvimento rápido. Navegue até a raiz do repositório, instale as dependências e rode a API diretamente:
```bash
uvicorn api.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```
A documentação interativa estará acessível em /docs e /redoc.
API disponível em: [http://localhost:8000](http://localhost:8000)  
Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

3. Executando o Pipeline: utilize o main.py conforme descrito acima para preparar os dados, treinar o modelo ou realizar inferência.
```bash
python main.py --mode all
```

4. Testes Unitários: navegue até a pasta unit/ e execute:
Entre na pasta `unit/` e rode:

```bash
cd unit
pytest -q -m "not slow" --cov=src --cov=api --cov-branch   --cov-report=term-missing:skip-covered --cov-report=xml
```
O relatório de cobertura mostra que o projeto atinge aproximadamente 81 % de cobertura
