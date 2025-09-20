# Dockerfile para empacotar a API de predição de contratação e o modelo
#
# Esta imagem utiliza Python 3.11 slim como base, instala as dependências
# do projeto a partir de requirements.txt e copia o código da API, os
# scripts de modelagem, o arquivo de modelo treinado e dados externos.
# O contêiner expõe a porta 8000 e executa o servidor Uvicorn para
# disponibilizar a API FastAPI.

FROM python:3.11-slim

# Diretório de trabalho dentro do contêiner
WORKDIR /app

# Instalar dependências de sistema necessárias para compilar algumas
# bibliotecas Python (por exemplo, pandas, shap)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da API, módulos auxiliares, modelo e dados externos
COPY api ./api
COPY src ./src
COPY models ./models
COPY data/external ./data/external

# Expor a porta usada pela API
EXPOSE 8000

# Comando de entrada do contêiner
CMD ["uvicorn", "api.main:create_app", "--host", "0.0.0.0", "--port", "8000"]