#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_predict.py
===============
Testes unitários para o módulo predict.py
"""
import json
import os
import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytest_asyncio
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

# Configurar pytest para suportar async
pytestmark = pytest.mark.asyncio

from api.routes.predict import (
    _ensure_model_exists,
    _model_path,
    _to_label,
    health,
    predict_batch,
    predict_single,
    router
)


# Fixtures
@pytest.fixture
def mock_model_path():
    """Mock para o caminho do modelo"""
    with patch("api.routes.predict._model_path") as mock_path:
        mock_path.return_value = Path("/fake/model/path.joblib")
        yield mock_path


@pytest.fixture
def mock_infer_module():
    """Mock para o módulo infer"""
    with patch("api.routes.predict.infer_mod") as mock_infer:
        yield mock_infer


class TestUtilityFunctions:
    """Testes para funções utilitárias"""

    def test_model_path_default(self):
        """Testa _model_path com valor padrão"""
        with patch.dict(os.environ, {}, clear=True):
            result = _model_path()
            expected = Path("models/lgbm_model.joblib").resolve()
            assert result == expected

    def test_model_path_from_env(self):
        """Testa _model_path com variável de ambiente"""
        custom_path = "/custom/model/path.joblib"
        with patch.dict(os.environ, {"MODEL_PATH": custom_path}):
            result = _model_path()
            expected = Path(custom_path).resolve()
            assert result == expected

    def test_ensure_model_exists_success(self):
        """Testa _ensure_model_exists quando arquivo existe"""
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        
        # Não deve levantar exceção
        _ensure_model_exists(fake_path)

    def test_ensure_model_exists_failure(self):
        """Testa _ensure_model_exists quando arquivo não existe"""
        fake_path = MagicMock()
        fake_path.exists.return_value = False
        fake_path.__str__ = lambda self: "/fake/path.joblib"
        
        with pytest.raises(HTTPException) as exc_info:
            _ensure_model_exists(fake_path)
        
        assert exc_info.value.status_code == 404
        assert "Modelo não encontrado" in exc_info.value.detail

    @pytest.mark.parametrize("prob,threshold,expected", [
        (0.3, 0.5, 0),
        (0.5, 0.5, 1),
        (0.7, 0.5, 1),
        (0.4, 0.3, 1),
        (0.2, 0.3, 0),
    ])
    def test_to_label(self, prob, threshold, expected):
        """Testa conversão de probabilidade para label"""
        result = _to_label(prob, threshold)
        assert result == expected


class TestHealthEndpoint:
    """Testes para o endpoint /health"""

    def test_health_model_exists(self, mock_model_path):
        """Testa health quando modelo existe"""
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/model/path.joblib"
        mock_model_path.return_value = fake_path
        
        result = health()
        
        expected = {
            "status": "ok",
            "model_path": "/fake/model/path.joblib",
            "exists": True
        }
        assert result == expected

    def test_health_model_not_exists(self, mock_model_path):
        """Testa health quando modelo não existe"""
        fake_path = MagicMock()
        fake_path.exists.return_value = False
        fake_path.__str__ = lambda self: "/fake/model/path.joblib"
        mock_model_path.return_value = fake_path
        
        result = health()
        
        expected = {
            "status": "ok",
            "model_path": "/fake/model/path.joblib",
            "exists": False
        }
        assert result == expected


class TestPredictSingle:
    """Testes para o endpoint /predict/single"""

    @pytest.fixture
    def valid_payload(self):
        """Payload válido para testes"""
        return {
            "cand": {"idade": 25, "experiencia": 3},
            "vaga": {"salario": 5000, "nivel": "junior"},
            "top_k": 10
        }

    @pytest.fixture
    def mock_predict_one_result(self):
        """Mock do resultado de predict_one"""
        return {
            "prob_contratado": 0.75,
            "X_engineered": {"feature1": 1.0, "feature2": 0.5},
            "local_contributions": [("feature1", 0.3), ("feature2", -0.1)],
            "global_importance": [("feature1", 0.8), ("feature2", 0.2)]
        }

    @pytest.mark.asyncio
    async def test_predict_single_success(self, mock_model_path, mock_infer_module, valid_payload, mock_predict_one_result):
        """Testa predict_single com sucesso"""
        # Setup mocks
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        fake_path.__str__ = lambda self: "/fake/model/path.joblib"
        mock_model_path.return_value = fake_path
        mock_infer_module.predict_one.return_value = mock_predict_one_result
        
        result = await predict_single(valid_payload)
        
        expected = {
            "label": 1,
            "probability": 0.75,
            "features_engineered": {"feature1": 1.0, "feature2": 0.5},
            "top_local_contributions": [("feature1", 0.3), ("feature2", -0.1)],
            "global_importance": [("feature1", 0.8), ("feature2", 0.2)]
        }
        assert result == expected
        mock_infer_module.predict_one.assert_called_once_with(
            valid_payload["cand"],
            valid_payload["vaga"],
            model_or_path="/fake/model/path.joblib",
            top_k=10
        )

    @pytest.mark.asyncio
    async def test_predict_single_model_not_found(self, mock_model_path):
        """Testa predict_single quando modelo não existe"""
        fake_path = MagicMock()
        fake_path.exists.return_value = False
        mock_model_path.return_value = fake_path
        
        payload = {"cand": {}, "vaga": {}}
        
        with pytest.raises(HTTPException) as exc_info:
            await predict_single(payload)
        
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_predict_single_invalid_payload_not_dict(self, mock_model_path):
        """Testa predict_single com payload inválido (não é dict)"""
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        mock_model_path.return_value = fake_path
        
        with pytest.raises(HTTPException) as exc_info:
            await predict_single("invalid")
        
        assert exc_info.value.status_code == 400
        assert "objeto JSON" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_predict_single_missing_cand_or_vaga(self, mock_model_path):
        """Testa predict_single com cand ou vaga faltando"""
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        mock_model_path.return_value = fake_path
        
        payload = {"cand": {}}  # vaga faltando
        
        with pytest.raises(HTTPException) as exc_info:
            await predict_single(payload)
        
        assert exc_info.value.status_code == 400
        assert "'cand' e 'vaga'" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_predict_single_inference_error(self, mock_model_path, mock_infer_module, valid_payload):
        """Testa predict_single quando infer_mod.predict_one falha"""
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        mock_model_path.return_value = fake_path
        mock_infer_module.predict_one.side_effect = Exception("Inference error")
        
        with pytest.raises(HTTPException) as exc_info:
            await predict_single(valid_payload)
        
        assert exc_info.value.status_code == 500
        assert "Falha na predição" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_predict_single_default_top_k(self, mock_model_path, mock_infer_module, mock_predict_one_result):
        """Testa predict_single com top_k padrão"""
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        mock_model_path.return_value = fake_path
        mock_infer_module.predict_one.return_value = mock_predict_one_result
        
        payload = {
            "cand": {"idade": 25},
            "vaga": {"salario": 5000}
            # top_k não especificado
        }
        
        await predict_single(payload)
        
        mock_infer_module.predict_one.assert_called_once_with(
            payload["cand"],
            payload["vaga"],
            model_or_path="/fake/model/path.joblib",
            top_k=10  # valor padrão
        )


class TestPredictBatch:
    """Testes para o endpoint /predict/batch"""

    @pytest.fixture
    def mock_upload_file(self):
        """Mock de UploadFile"""
        mock_file = MagicMock(spec=UploadFile)
        mock_file.read.return_value = b"cand_idade,vaga_salario\n25,5000\n30,6000"
        return mock_file

    @pytest.fixture
    def mock_dataframe_result(self):
        """Mock do DataFrame retornado por predict_batch_from_csv"""
        data = {
            "prob_contratado": [0.75, 0.65],
            "top_features_json": ['["feature1", "feature2"]', '["feature1", "feature3"]'],
            "top_contribs_json": ['[0.3, -0.1]', '[0.2, 0.1]'],
            "cand_idade": [25, 30],
            "vaga_salario": [5000, 6000]
        }
        return pd.DataFrame(data)

    @pytest.mark.asyncio
    async def test_predict_batch_success(self, mock_model_path, mock_infer_module, mock_upload_file, mock_dataframe_result):
        """Testa predict_batch com sucesso"""
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        mock_model_path.return_value = fake_path
        mock_infer_module.predict_batch_from_csv.return_value = mock_dataframe_result
        
        with patch("tempfile.NamedTemporaryFile"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.unlink"):
            
            result = await predict_batch(mock_upload_file)
        
        expected = {
            "rows": [
                {
                    "probability": 0.75,
                    "label": 1,
                    "top_features": ["feature1", "feature2"],
                    "top_contribs": [0.3, -0.1]
                },
                {
                    "probability": 0.65,
                    "label": 1,
                    "top_features": ["feature1", "feature3"],
                    "top_contribs": [0.2, 0.1]
                }
            ],
            "count": 2
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_predict_batch_with_engineered_features(self, mock_model_path, mock_infer_module, mock_upload_file, mock_dataframe_result):
        """Testa predict_batch incluindo features engineered"""
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        mock_model_path.return_value = fake_path
        mock_infer_module.predict_batch_from_csv.return_value = mock_dataframe_result
        
        # Mock para build_engineered_from_raw
        mock_engineered_df = pd.DataFrame([{"eng_feature1": 1.5, "eng_feature2": 2.0}])
        mock_infer_module.build_engineered_from_raw.return_value = mock_engineered_df
        
        with patch("tempfile.NamedTemporaryFile"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.unlink"):
            
            result = await predict_batch(mock_upload_file, include_engineered=True)
        
        # Deve incluir a chave "engineered" nos items
        assert "engineered" in result["rows"][0]

    @pytest.mark.asyncio
    async def test_predict_batch_model_not_found(self, mock_model_path, mock_upload_file):
        """Testa predict_batch quando modelo não existe"""
        fake_path = MagicMock()
        fake_path.exists.return_value = False
        mock_model_path.return_value = fake_path
        
        with pytest.raises(HTTPException) as exc_info:
            await predict_batch(mock_upload_file)
        
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_predict_batch_inference_error(self, mock_model_path, mock_infer_module, mock_upload_file):
        """Testa predict_batch quando predict_batch_from_csv falha"""
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        mock_model_path.return_value = fake_path
        mock_infer_module.predict_batch_from_csv.side_effect = Exception("Batch inference error")
        
        with patch("tempfile.NamedTemporaryFile"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.unlink"):
            
            with pytest.raises(HTTPException) as exc_info:
                await predict_batch(mock_upload_file)
        
        assert exc_info.value.status_code == 500
        assert "Falha na predição em lote" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_predict_batch_json_parsing_error(self, mock_model_path, mock_infer_module, mock_upload_file):
        """Testa predict_batch com erro no parsing do JSON"""
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        mock_model_path.return_value = fake_path
        
        # DataFrame com JSON inválido
        invalid_data = {
            "prob_contratado": [0.75],
            "top_features_json": ['invalid json'],
            "top_contribs_json": ['also invalid'],
        }
        mock_infer_module.predict_batch_from_csv.return_value = pd.DataFrame(invalid_data)
        
        with patch("tempfile.NamedTemporaryFile"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.unlink"):
            
            result = await predict_batch(mock_upload_file)
        
        # Deve retornar listas vazias quando JSON é inválido
        assert result["rows"][0]["top_features"] == []
        assert result["rows"][0]["top_contribs"] == []

    @pytest.mark.asyncio
    async def test_predict_batch_cleanup_files(self, mock_model_path, mock_infer_module, mock_upload_file, mock_dataframe_result):
        """Testa se os arquivos temporários são limpos após o processamento"""
        fake_path = MagicMock()
        fake_path.exists.return_value = True
        mock_model_path.return_value = fake_path
        mock_infer_module.predict_batch_from_csv.return_value = mock_dataframe_result
        
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/fake_file.csv"
        
        mock_path_exists = MagicMock(return_value=True)
        mock_path_unlink = MagicMock()
        
        with patch("tempfile.NamedTemporaryFile", return_value=mock_temp_file), \
             patch("pathlib.Path.exists", mock_path_exists), \
             patch("pathlib.Path.unlink", mock_path_unlink):
            
            await predict_batch(mock_upload_file)
        
        # Verifica se unlink foi chamado para limpar arquivos temporários
        assert mock_path_unlink.call_count >= 1


class TestIntegration:
    """Testes de integração usando TestClient"""

    @pytest.fixture
    def client(self):
        """Cliente de teste FastAPI"""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_health_endpoint_integration(self, client):
        """Teste de integração para /health"""
        with patch("api.routes.predict._model_path") as mock_path:
            fake_path = MagicMock()
            fake_path.exists.return_value = True
            fake_path.__str__ = lambda self: "/test/model.joblib"
            mock_path.return_value = fake_path
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["exists"] is True