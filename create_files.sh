#!/usr/bin/env bash
set -euo pipefail

# Uso: ./bootstrap_ml_project.sh [nome-do-projeto]
PROJECT_ROOT="${1:-ml-eng-project}"

echo ">> Criando projeto em: ${PROJECT_ROOT}"

# Pastas
dirs=(
  "$PROJECT_ROOT/docker/prometheus"
  "$PROJECT_ROOT/docker/grafana/dashboards"
  "$PROJECT_ROOT/docker/grafana/provisioning/datasources"
  "$PROJECT_ROOT/docker/grafana/provisioning/dashboards"
  "$PROJECT_ROOT/src/data/raw"
  "$PROJECT_ROOT/src/data/processed"
  "$PROJECT_ROOT/src/data/external"
  "$PROJECT_ROOT/src/models/registry/latest"
  "$PROJECT_ROOT/src/api/routers"
  "$PROJECT_ROOT/src/utils"
  "$PROJECT_ROOT/src/monitoring"
  "$PROJECT_ROOT/tests/unit"
  "$PROJECT_ROOT/tests/integration"
  "$PROJECT_ROOT/scripts"
)

for d in "${dirs[@]}"; do
  mkdir -p "$d"
done

echo ">> Pastas criadas."

# Arquivos vazios na raiz
root_files=(
  "$PROJECT_ROOT/README.md"
  "$PROJECT_ROOT/.gitignore"
  "$PROJECT_ROOT/.env.example"
  "$PROJECT_ROOT/requirements.txt"
  "$PROJECT_ROOT/pyproject.toml"
  "$PROJECT_ROOT/Makefile"
  "$PROJECT_ROOT/docker-compose.yml"
)

# Dockerfiles e configs
docker_files=(
  "$PROJECT_ROOT/docker/Dockerfile.api"
  "$PROJECT_ROOT/docker/Dockerfile.train"
  "$PROJECT_ROOT/docker/prometheus/prometheus.yml"
  "$PROJECT_ROOT/docker/grafana/dashboards/api_metrics.json"
  "$PROJECT_ROOT/docker/grafana/provisioning/datasources/datasource.yml"
  "$PROJECT_ROOT/docker/grafana/provisioning/dashboards/dashboards.yml"
)

# Código fonte (vazios)
src_files=(
  "$PROJECT_ROOT/src/models/model.py"
  "$PROJECT_ROOT/src/models/train.py"
  "$PROJECT_ROOT/src/models/infer.py"
  "$PROJECT_ROOT/src/models/registry/latest/model.pkl"      # placeholder vazio
  "$PROJECT_ROOT/src/models/registry/latest/metadata.json"
  "$PROJECT_ROOT/src/api/main.py"
  "$PROJECT_ROOT/src/api/routers/predict.py"
  "$PROJECT_ROOT/src/api/schemas.py"
  "$PROJECT_ROOT/src/api/settings.py"
  "$PROJECT_ROOT/src/api/instrumentation.py"
  "$PROJECT_ROOT/src/api/logging_conf.py"
  "$PROJECT_ROOT/src/utils/io.py"
  "$PROJECT_ROOT/src/utils/metrics.py"
  "$PROJECT_ROOT/src/utils/seed.py"
  "$PROJECT_ROOT/src/monitoring/README.md"
)

# Testes (vazios)
test_files=(
  "$PROJECT_ROOT/tests/unit/test_model.py"
  "$PROJECT_ROOT/tests/unit/test_infer.py"
  "$PROJECT_ROOT/tests/unit/test_api.py"
  "$PROJECT_ROOT/tests/unit/test_utils.py"
  "$PROJECT_ROOT/tests/integration/test_api_integration.py"
  "$PROJECT_ROOT/tests/conftest.py"
)

# Scripts (vazios)
script_files=(
  "$PROJECT_ROOT/scripts/download_data.py"
  "$PROJECT_ROOT/scripts/prepare_data.py"
  "$PROJECT_ROOT/scripts/train_local.sh"
  "$PROJECT_ROOT/scripts/run_api_local.sh"
  "$PROJECT_ROOT/scripts/smoke_test.sh"
)

# Marcadores .gitkeep (vazios)
gitkeeps=(
  "$PROJECT_ROOT/src/data/raw/.gitkeep"
  "$PROJECT_ROOT/src/data/processed/.gitkeep"
  "$PROJECT_ROOT/src/data/external/.gitkeep"
)

# Criar todos os arquivos vazios
all_files=("${root_files[@]}" "${docker_files[@]}" "${src_files[@]}" "${test_files[@]}" "${script_files[@]}" "${gitkeeps[@]}")
for f in "${all_files[@]}"; do
  # Assegura diretório e cria vazio
  mkdir -p "$(dirname "$f")"
  : > "$f"
done

# Permissões de execução para scripts .sh
chmod +x "$PROJECT_ROOT/scripts/"*.sh 2>/dev/null || true

echo ">> Arquivos vazios criados."
echo ">> Pronto! Estrutura inicial em '${PROJECT_ROOT}'."
