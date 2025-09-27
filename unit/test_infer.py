"""
Tests for the infer module.

These tests verify that the inference helpers behave correctly.  We
focus on two parts: that ``load_model`` properly delegates to
``joblib.load`` and validates the pipeline structure, and that
``build_engineered_from_raw`` constructs engineered features from
minimal candidate/vacancy dictionaries.  We do not load any real
model files; instead, we use mocks to stand in for the trained
pipeline returned by joblib.
"""

import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import pytest

from src.models import infer


class TestLoadModel:
    """Tests for load_model function"""

    def test_load_model_uses_joblib_load_and_validates_pipeline(self):
        """load_model should call joblib.load and ensure the returned object has expected steps."""
        # Create a fake pipeline with 'pre' and 'clf' steps
        fake_pipeline = mock.MagicMock(spec=Pipeline)
        fake_pipeline.named_steps = {"pre": mock.MagicMock(), "clf": mock.MagicMock()}
        
        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(infer, "joblib") as mock_joblib:
                mock_joblib.load.return_value = fake_pipeline
                result = infer.load_model("/path/to/model.joblib")
                # joblib.load should be called with a Path
                mock_joblib.load.assert_called_once()
                called_path = mock_joblib.load.call_args[0][0]
                assert isinstance(called_path, Path)
                assert str(called_path).endswith("model.joblib")
                assert result is fake_pipeline

    def test_load_model_file_not_found(self):
        """load_model should raise FileNotFoundError when model file doesn't exist"""
        with mock.patch.object(Path, "exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Modelo não encontrado"):
                infer.load_model("/path/to/nonexistent.joblib")

    def test_load_model_invalid_pipeline_structure(self):
        """load_model should raise ValueError for invalid pipeline structure"""
        bad_pipe = mock.MagicMock()
        bad_pipe.named_steps = {}
        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(infer, "joblib") as mock_joblib:
                mock_joblib.load.return_value = bad_pipe
                with pytest.raises(ValueError, match="não parece ser o Pipeline do treino"):
                    infer.load_model("/tmp/bad_model.joblib")

    def test_load_model_missing_pre_step(self):
        """load_model should raise ValueError when 'pre' step is missing"""
        bad_pipe = mock.MagicMock(spec=Pipeline)
        bad_pipe.named_steps = {"clf": mock.MagicMock()}  # missing 'pre'
        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(infer, "joblib") as mock_joblib:
                mock_joblib.load.return_value = bad_pipe
                with pytest.raises(ValueError):
                    infer.load_model("/tmp/bad_model.joblib")

    def test_load_model_missing_clf_step(self):
        """load_model should raise ValueError when 'clf' step is missing"""
        bad_pipe = mock.MagicMock(spec=Pipeline)
        bad_pipe.named_steps = {"pre": mock.MagicMock()}  # missing 'clf'
        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(infer, "joblib") as mock_joblib:
                mock_joblib.load.return_value = bad_pipe
                with pytest.raises(ValueError):
                    infer.load_model("/tmp/bad_model.joblib")


class TestTextBuilders:
    """Tests for text building helper functions"""

    def test_build_text_from_raw_cand(self):
        """Test building candidate text from raw data"""
        cand = {
            "conhecimentos_tecnicos": "Python SQL",
            "certificacoes": "AWS",
            "outras_certificacoes": "Docker",
            "titulo_profissional": "Engineer",
            "area_atuacao": "TI",
            "cv_text": "Experience with databases",
        }
        result = infer.build_text_from_raw_cand(cand)
        expected_parts = ["Python SQL", "AWS", "Docker", "Engineer", "TI", "Experience with databases"]
        assert all(part in result for part in expected_parts)

    def test_build_text_from_raw_cand_with_none_values(self):
        """Test building candidate text handles None values"""
        cand = {
            "conhecimentos_tecnicos": "Python SQL",
            "certificacoes": "",
            "outras_certificacoes": "",
            "titulo_profissional": "Engineer",
        }
        result = infer.build_text_from_raw_cand(cand)
        assert "Python SQL" in result
        assert "Engineer" in result
        # Should handle None gracefully

    def test_build_text_from_raw_vaga(self):
        """Test building vacancy text from raw data"""
        vaga = {
            "titulo_vaga": "Software Engineer",
            "principais_atividades": "Python development",
            "competencias": "SQL Python",
            "areas_atuacao": "TI",
            "demais_observacoes": "Remote work",
            "descricao": "Full stack role",
        }
        result = infer.build_text_from_raw_vaga(vaga)
        expected_parts = ["Software Engineer", "Python development", "SQL Python", "TI", "Remote work", "Full stack role"]
        assert all(part in result for part in expected_parts)


class TestHiringFlags:
    """Tests for hiring_flags function"""

    def test_hiring_flags_clt(self):
        """Test hiring flags for CLT"""
        clt, pj, est, cot = infer.hiring_flags("CLT")
        assert clt == 1
        assert pj == 0
        assert est == 0
        assert cot == 0

    def test_hiring_flags_pj(self):
        """Test hiring flags for PJ"""
        clt, pj, est, cot = infer.hiring_flags("PJ")
        assert clt == 0
        assert pj == 1
        assert est == 0
        assert cot == 0

    def test_hiring_flags_estagiario(self):
        """Test hiring flags for Estagiário"""
        clt, pj, est, cot = infer.hiring_flags("Estagiário")
        assert clt == 0
        assert pj == 0
        assert est == 1
        assert cot == 0

    def test_hiring_flags_cotas(self):
        """Test hiring flags for Cotas"""
        clt, pj, est, cot = infer.hiring_flags("Cotas")
        assert clt == 0
        assert pj == 0
        assert est == 0
        assert cot == 1

    def test_hiring_flags_multiple(self):
        """Test hiring flags with multiple types"""
        clt, pj, est, cot = infer.hiring_flags("CLT PJ")
        assert clt == 1
        assert pj == 1
        assert est == 0
        assert cot == 0

    def test_hiring_flags_none(self):
        """Test hiring flags with None input"""
        clt, pj, est, cot = infer.hiring_flags("")
        assert clt == 0
        assert pj == 0
        assert est == 0
        assert cot == 0

    def test_hiring_flags_empty(self):
        """Test hiring flags with empty string"""
        clt, pj, est, cot = infer.hiring_flags("")
        assert clt == 0
        assert pj == 0
        assert est == 0
        assert cot == 0


class TestSeniorityFlags:
    """Tests for seniority_flags function"""

    def test_seniority_flags_junior(self):
        """Test seniority flags for Junior"""
        jun, ple, sen = infer.seniority_flags("Junior")
        assert jun == 1
        assert ple == 0
        assert sen == 0

    def test_seniority_flags_junior_with_accent(self):
        """Test seniority flags for Júnior"""
        jun, ple, sen = infer.seniority_flags("Júnior")
        assert jun == 1
        assert ple == 0
        assert sen == 0

    def test_seniority_flags_pleno(self):
        """Test seniority flags for Pleno"""
        jun, ple, sen = infer.seniority_flags("Pleno")
        assert jun == 0
        assert ple == 1
        assert sen == 0

    def test_seniority_flags_senior(self):
        """Test seniority flags for Senior"""
        jun, ple, sen = infer.seniority_flags("Senior")
        assert jun == 0
        assert ple == 0
        assert sen == 1

    def test_seniority_flags_senior_with_accent(self):
        """Test seniority flags for Sênior"""
        jun, ple, sen = infer.seniority_flags("Sênior")
        assert jun == 0
        assert ple == 0
        assert sen == 1

    def test_seniority_flags_none(self):
        """Test seniority flags with None input"""
        jun, ple, sen = infer.seniority_flags("")
        assert jun == 0
        assert ple == 0
        assert sen == 0


class TestBuildEngineeredFromRaw:
    """Tests for build_engineered_from_raw function"""

    def test_build_engineered_from_raw(self):
        """build_engineered_from_raw should derive correct feature values from raw inputs."""
        cand = {
            "cidade": "São Paulo",
            "uf": "SP",
            "nivel_academico": "Ensino Superior Completo",
            "nivel_ingles": "Avançado",
            "nivel_espanhol": "Nenhum",
            "nivel_profissional": "Pleno",
            "conhecimentos_tecnicos": "Python SQL",
            "certificacoes": "AWS",
            "outras_certificacoes": "",
            "titulo_profissional": "Engenheiro de Dados",
            "area_atuacao": "TI",
            "cv_text": "Experiência com ETL e bancos de dados",
            "remuneracao": "R$ 12.000,00",
            "remuneracao_num": "",
        }
        vaga = {
            "cidade": "São Paulo",
            "uf": "SP",
            "nivel_academico": "Ensino Médio",
            "nivel_ingles": "Básico",
            "nivel_espanhol": "Nenhum",
            "nivel_profissional": "Pleno",
            "tipo_contratacao": "CLT",
            "titulo_vaga": "Engenheiro de Software",
            "principais_atividades": "Python SQL",
            "competencias": "SQL",
            "areas_atuacao": "TI",
            "demais_observacoes": "",
            "descricao": "",
        }
        df = infer.build_engineered_from_raw(cand, vaga)
        # Exactly one row returned
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        row = df.iloc[0]
        # Geography flags
        assert row["same_state"] == 1
        assert row["same_city"] == 1
        assert row["same_region"] == 1
        # Meeting education and language requirements: cand higher than vaga
        assert row["meets_academic"] == 1
        assert row["meets_english"] == 1
        assert row["meets_spanish"] == 1
        # Hiring type flags
        assert row["vaga_is_CLT"] == 1
        assert row["vaga_is_PJ"] == 0
        assert row["vaga_is_Estagiario"] == 0
        assert row["vaga_is_Cotas"] == 0
        # Seniority flags
        assert row["cand_is_Pleno"] == 1
        assert row["vaga_is_Pleno"] == 1
        # Remuneration should be parsed to float
        assert row["cand_remuneracao_num"] == 12000.0
        # Similarity measures should be positive when there is term overlap
        assert row["sim_tfidf"] > 0
        assert row["overlap_kw"] >= 1
        assert row["jaccard_kw"] > 0

    def test_build_engineered_different_states(self):
        """Test build_engineered_from_raw with different states"""
        cand = {"cidade": "São Paulo", "uf": "SP"}
        vaga = {"cidade": "Rio de Janeiro", "uf": "RJ"}
        
        df = infer.build_engineered_from_raw(cand, vaga)
        row = df.iloc[0]
        
        assert row["same_state"] == 0
        assert row["same_city"] == 0
        assert row["same_region"] == 1  # Both SP and RJ are in Southeast

    def test_build_engineered_missing_academic_levels(self):
        """Test build_engineered_from_raw with missing academic levels"""
        cand = {"nivel_academico": ""}
        vaga = {"nivel_academico": "Ensino Médio"}
        
        df = infer.build_engineered_from_raw(cand, vaga)
        row = df.iloc[0]
        
        assert row["meets_academic"] == 0

    def test_build_engineered_with_remuneracao_num(self):
        """Test build_engineered_from_raw when remuneracao_num is already provided"""
        cand = {"remuneracao_num": 15000.0, "remuneracao": "R$ 12.000,00"}
        vaga = {}
        
        df = infer.build_engineered_from_raw(cand, vaga)
        row = df.iloc[0]
        
        # Should use remuneracao_num when available
        assert row["cand_remuneracao_num"] == 15000.0


class TestOriginalCol:
    """Tests for _original_col helper function"""

    def test_original_col_with_cat_prefix(self):
        """Test _original_col with categorical encoding prefix"""
        result = infer._original_col("cat__feature_name_value")
        assert result == "feature"

    def test_original_col_with_num_prefix(self):
        """Test _original_col with numerical encoding prefix"""
        result = infer._original_col("num__feature_name")
        assert result == "feature_name"

    def test_original_col_no_prefix(self):
        """Test _original_col with no encoding prefix"""
        result = infer._original_col("regular_feature")
        assert result == "regular_feature"


class TestGlobalImportance:
    """Tests for global_importance function"""

    def test_global_importance(self):
        """Test global_importance function with mock pipeline"""
        # Create mock pipeline
        mock_pipe = mock.MagicMock()
        mock_pre = mock.MagicMock()
        mock_clf = mock.MagicMock()
        
        mock_pre.get_feature_names_out.return_value = ["cat__feature1_A", "num__feature2", "cat__feature1_B"]
        mock_clf.feature_importances_ = np.array([0.5, 0.3, 0.2])
        
        mock_pipe.named_steps = {"pre": mock_pre, "clf": mock_clf}
        
        result = infer.global_importance(mock_pipe)
        
        assert isinstance(result, pd.DataFrame)
        assert "feature_original" in result.columns
        assert "importance" in result.columns
        # Should aggregate feature1 (0.5 + 0.2 = 0.7) and feature2 (0.3)
        assert len(result) == 2


class TestLocalContributions:
    """Tests for local_contributions function"""

    def test_local_contributions(self):
        """Test local_contributions function"""
        mock_pipe = mock.MagicMock()
        mock_pre = mock.MagicMock()
        mock_clf = mock.MagicMock()
        
        # Mock preprocessing transformation
        mock_X_enc = np.array([[1, 2, 3]])
        mock_pre.transform.return_value = mock_X_enc
        
        # Mock classifier predict with contributions
        mock_contrib = np.array([[0.1, 0.2, 0.3, 0.1]])  # Last element is bias
        mock_clf.predict.return_value = mock_contrib
        
        mock_pipe.named_steps = {"pre": mock_pre, "clf": mock_clf}
        
        X = pd.DataFrame([[1, 2, 3]])
        result = infer.local_contributions(mock_pipe, X)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 4)  # 3 features + 1 bias
        np.testing.assert_array_equal(result, mock_contrib)


class TestPredictOne:
    """Tests for predict_one function"""

    def test_predict_one_with_pipeline_object(self):
        """Test predict_one when passed a pipeline object directly"""
        # Create comprehensive mock pipeline
        mock_pipe = mock.MagicMock()
        mock_pre = mock.MagicMock()
        mock_clf = mock.MagicMock()
        
        # Mock preprocessing
        mock_X_enc = np.array([[1, 2, 3]])
        mock_pre.transform.return_value = mock_X_enc
        mock_pre.get_feature_names_out.return_value = ["feat1", "feat2", "feat3"]
        
        # Mock prediction
        mock_clf.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_clf.feature_importances_ = np.array([0.5, 0.3, 0.2])
        mock_clf.predict.return_value = np.array([[0.1, 0.2, 0.3, 0.1]])  # contributions
        
        mock_pipe.named_steps = {"pre": mock_pre, "clf": mock_clf}
        
        cand = {"cidade": "São Paulo", "uf": "SP"}
        vaga = {"cidade": "São Paulo", "uf": "SP"}
        
        result = infer.predict_one(cand, vaga, mock_pipe, top_k=5)
        
        assert isinstance(result, dict)
        assert "prob_contratado" in result
        assert "global_importance" in result
        assert "local_contributions" in result
        assert "X_engineered" in result
        assert result["prob_contratado"] == 0.7

    def test_predict_one_with_model_path(self):
        """Test predict_one when passed a model path"""
        # Mock load_model to return our mock pipeline
        mock_pipe = mock.MagicMock()
        mock_pre = mock.MagicMock()
        mock_clf = mock.MagicMock()
        
        mock_X_enc = np.array([[1, 2, 3]])
        mock_pre.transform.return_value = mock_X_enc
        mock_pre.get_feature_names_out.return_value = ["feat1", "feat2", "feat3"]
        mock_clf.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_clf.feature_importances_ = np.array([0.4, 0.4, 0.2])
        mock_clf.predict.return_value = np.array([[0.1, 0.15, 0.05, 0.1]])
        
        mock_pipe.named_steps = {"pre": mock_pre, "clf": mock_clf}
        
        cand = {"uf": "RJ"}
        vaga = {"uf": "SP"}
        
        with mock.patch.object(infer, 'load_model', return_value=mock_pipe):
            result = infer.predict_one(cand, vaga, "/path/to/model.joblib")
            
            assert result["prob_contratado"] == 0.8


class TestRowToRawDicts:
    """Tests for _row_to_raw_dicts helper function"""

    def test_row_to_raw_dicts(self):
        """Test _row_to_raw_dicts function"""
        # Create a pandas Series with mixed cand_ and vaga_ columns
        data = {
            "cand_nome": "João",
            "cand_idade": 30,
            "vaga_titulo": "Developer", 
            "vaga_salario": 5000,
            "other_column": "ignored"
        }
        series = pd.Series(data)
        
        cand, vaga = infer._row_to_raw_dicts(series)
        
        assert cand == {"nome": "João", "idade": 30}
        assert vaga == {"titulo": "Developer", "salario": 5000}

    def test_row_to_raw_dicts_with_nan(self):
        """Test _row_to_raw_dicts with NaN values"""
        data = {
            "cand_nome": "João",
            "cand_idade": "",
            "vaga_titulo": "",
            "vaga_salario": 5000,
        }
        series = pd.Series(data)
        
        cand, vaga = infer._row_to_raw_dicts(series)
        
        assert cand == {"nome": "João", "idade": ""}
        assert vaga == {"titulo": "", "salario": 5000}


class TestPredictBatchFromCsv:
    """Tests for predict_batch_from_csv function"""

    def test_predict_batch_from_csv_basic(self):
        """Test basic predict_batch_from_csv functionality"""
        # Create mock pipeline
        mock_pipe = mock.MagicMock()
        mock_pre = mock.MagicMock()
        mock_clf = mock.MagicMock()
        
        mock_pre.transform.return_value = np.array([[1, 2], [3, 4]])
        mock_clf.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
        
        mock_pipe.named_steps = {"pre": mock_pre, "clf": mock_clf}
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_in:
            tmp_in.write("cand_nome,vaga_titulo\nJoão,Developer\nMaria,Analyst\n")
            input_csv = tmp_in.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_out:
            output_csv = tmp_out.name
        
        try:
            with mock.patch.object(infer, 'load_model', return_value=mock_pipe):
                result = infer.predict_batch_from_csv(
                    input_csv, "/fake/model.joblib", output_csv, include_local=False
                )
            
            assert isinstance(result, pd.DataFrame)
            assert "prob_contratado" in result.columns
            assert len(result) == 2
            np.testing.assert_array_almost_equal(result["prob_contratado"].values, [0.7, 0.4])
            
        finally:
            Path(input_csv).unlink(missing_ok=True)
            Path(output_csv).unlink(missing_ok=True)

    def test_predict_batch_from_csv_with_local_contributions(self):
        """Test predict_batch_from_csv with local contributions"""
        mock_pipe = mock.MagicMock()
        mock_pre = mock.MagicMock()
        mock_clf = mock.MagicMock()
        
        # Mock feature names and contributions
        mock_pre.get_feature_names_out.return_value = ["feat1", "feat2"]
        mock_pre.transform.return_value = np.array([[1, 2]])
        mock_clf.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        # Mock local contributions (2 features + 1 bias)
        mock_contrib = np.array([[0.1, 0.2, 0.05]])
        
        with mock.patch.object(infer, 'local_contributions', return_value=mock_contrib):
            mock_pipe.named_steps = {"pre": mock_pre, "clf": mock_clf}
            
            # Create temporary CSV
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_in:
                tmp_in.write("cand_nome,vaga_titulo\nJoão,Developer\n")
                input_csv = tmp_in.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_out:
                output_csv = tmp_out.name
            
            try:
                with mock.patch.object(infer, 'load_model', return_value=mock_pipe):
                    result = infer.predict_batch_from_csv(
                        input_csv, "/fake/model.joblib", output_csv, 
                        include_local=True, top_k=2
                    )
                
                assert "top_features_json" in result.columns
                assert "top_contribs_json" in result.columns
                
            finally:
                Path(input_csv).unlink(missing_ok=True)
                Path(output_csv).unlink(missing_ok=True)