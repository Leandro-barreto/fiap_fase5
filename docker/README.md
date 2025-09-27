# 🐳 Docker e Observabilidade

Esta pasta contém os arquivos necessários para executar a API juntamente com um stack de observabilidade utilizando **Docker Compose**.  O `docker-compose.yml` define três serviços principais: a API, o servidor de métricas **Prometheus** e a ferramenta de visualização **Grafana**.

## 📦 O que está incluído

- **`docker-compose.yml`** – orquestra os containers de API, Prometheus e Grafana.  
- **Prometheus** – servidor de métricas open-source.  Ele coleta e armazena informações numéricas como séries temporais, identificadas por nomes de métricas e pares de chave/valor.  
- **Grafana** – plataforma de visualização de dados.  Permite consultar, analisar e alertar sobre métricas. Dashboards pré-configurados monitoram:
  - ⏱️ Latência de requisições
  - 📡 Número de chamadas ao endpoint `/api/predict/candidate`
  - ✅ Taxas de sucesso/erro

## 🚀 Como executar

Para iniciar todo o ambiente em containers basta executar:

```bash
docker-compose up --build
```

Comandos úteis:

```bash
# Listar containers em execução
docker ps

# Limpar recursos e imagens não utilizados
docker system prune -f
```

## 🌐 Endereços principais

- **API:** http://localhost:8000  
- **Prometheus:** http://localhost:9090  
- **Grafana:** http://localhost:3000 (login padrão: `admin/admin`)
