"""
Tests for the feature_engineering module.

These tests verify that helper functions used to prepare the final
dataset behave as expected.  They do not execute the entire data
pipeline (which would require large input files) but instead focus
on deterministic, small pieces of functionality such as state/region
mappings, numeric conversions, token detection, text similarity and
overlap calculations, ranking functions, and seniority flag extraction.

The tests also include a basic check that the ``build_df_final``
function can consume minimal JSON inputs and produce a dataframe
containing the engineered columns.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import pytest

import importlib

try:
    # Prefer the package path used in the src/ layout
    fe = importlib.import_module("src.data.feature_engineering")  # type: ignore[attr-defined]
except Exception:
    # Fall back to a flat module if running outside the package
    fe = importlib.import_module("feature_engineering")  # type: ignore[attr-defined]

from src.data.prepare_data import flatten_applicants, flatten_prospects, flatten_vagas


def test_uf_to_region():
    """State codes should map to the correct Brazilian region."""
    assert fe.uf_to_region("SP") == "Sudeste"
    assert fe.uf_to_region("sc") == "Sul"
    assert fe.uf_to_region("DF") == "Centro-Oeste"
    # unknown or None yields None
    assert fe.uf_to_region("XX") is None
    assert fe.uf_to_region(None) is None


def test_extract_city_state_from_local():
    """Parsing of the local field should extract city, UF and state name."""
    # typical format "City, UF"
    assert fe.extract_city_state_from_local("São Paulo, SP") == (
        "São Paulo",
        "SP",
        None,
    )
    # delimiter with hyphen should also work
    assert fe.extract_city_state_from_local("Florianópolis-SC") == (
        "Florianópolis",
        "SC",
        None,
    )
    # single token that is a state name
    assert fe.extract_city_state_from_local("Rio de Janeiro") == (
        None,
        "RJ",
        "Rio de Janeiro",
    )
    # empty or invalid strings yield Nones
    assert fe.extract_city_state_from_local("") == (None, None, None)
    assert fe.extract_city_state_from_local(None) == (None, None, None)


def test_to_float_money():
    """Currency strings should be converted to floats or NaN."""
    assert fe.to_float_money("R$ 10.500,50") == pytest.approx(10500.50)
    assert fe.to_float_money("10500,50") == pytest.approx(10500.50)
    # non numeric values return NaN
    assert np.isnan(fe.to_float_money("não é um número"))
    assert np.isnan(fe.to_float_money(None))


def test_has_tokens_and_tfidf_sim():
    """Basic token detection and similarity should behave correctly."""
    a = "Python, SQL, Data"
    b = "SQL developer Python"
    assert fe.has_tokens(a)
    assert fe.has_tokens(b)
    sim = fe.tfidf_sim(a, b)
    # there are overlapping tokens so similarity should be > 0
    assert sim > 0
    # empty strings or no tokens should return zero similarity
    assert fe.tfidf_sim("", "abc") == 0.0
    # string shorter than token length threshold
    assert fe.tfidf_sim("ab cd", "efgh") == 0.0


def test_overlap_and_jaccard():
    """Overlap and Jaccard index should reflect shared keywords."""
    a = "Python SQL Java"
    b = "Python C# SQL"
    ov, jacc = fe.overlap_and_jaccard(a, b)
    # common tokens: Python and SQL
    assert ov == 2
    # union: python, sql, java, c# => 4 tokens
    assert pytest.approx(jacc) == 2 / 4
    # no valid tokens means zero overlap and zero jaccard
    assert fe.overlap_and_jaccard("", "text") == (0, 0.0)


def test_academic_and_lang_rank():
    """Ranking functions should return increasing values with higher education or language levels."""
    # academic ranks are ordered: Médio < Superior Incompleto < Superior Completo
    assert fe.academic_rank("Ensino Médio") < fe.academic_rank("Ensino Superior Incompleto") < fe.academic_rank("Ensino Superior Completo")
    # unknown labels return NaN
    assert np.isnan(fe.academic_rank("Desconhecido"))
    # language ranks: Básico < Intermediário < Avançado < Fluente
    assert fe.lang_rank("Básico") < fe.lang_rank("Intermediário") < fe.lang_rank("Avançado") < fe.lang_rank("Fluente")
    assert np.isnan(fe.lang_rank("Unknown"))


def test_seniority_flags():
    """Extract flags for junior, pleno and senior positions regardless of accents or case."""
    assert fe.seniority_flags("Junior") == (1, 0, 0)
    assert fe.seniority_flags("Pleno") == (0, 1, 0)
    assert fe.seniority_flags("Senior") == (0, 0, 1)
    assert fe.seniority_flags("Júnior") == (1, 0, 0)
    assert fe.seniority_flags("") == (0, 0, 0)
    assert fe.seniority_flags(None) == (0, 0, 0)


def test_build_df_final_with_minimal_input(tmp_path):
    """build_df_final should produce a dataframe with engineered columns for a minimal dataset."""
    # Construct minimal JSON structures
    applicants_data = {
        "1": {
            "infos_basicas": {"local": "São Paulo, SP"},
            "formacao_e_idiomas": {
                "nivel_academico": "Ensino Superior Completo",
                "nivel_ingles": "Avançado",
                "nivel_espanhol": "Nenhum",
            },
            "informacoes_profissionais": {
                "nivel_profissional": "Pleno",
                "conhecimentos_tecnicos": "Python SQL",
                "certificacoes": "",
                "outras_certificacoes": "",
                "titulo_profissional": "Desenvolvedor",
                "area_atuacao": "TI",
                "remuneracao": "R$ 10.000,00",
            },
        }
    }
    prospects_data = {
        "123": {
            "titulo": "Engenheiro de Dados",
            "modalidade": "Presencial",
            "prospects": [
                {
                    "nome": "Fulano",
                    "codigo": "1",
                    "situacao_candidato": "Contratado",
                }
            ],
        }
    }
    vagas_data = {
        "123": {
            "informacoes_basicas": {
                "titulo_vaga": "Engenheiro de Dados",
                "tipo_contratacao": "CLT",
            },
            "perfil_vaga": {
                "nivel_profissional": "Pleno",
                "nivel_academico": "Ensino Superior Completo",
                "nivel_ingles": "Intermediário",
                "nivel_espanhol": "Nenhum",
                "estado": "SP",
                "cidade": "São Paulo",
                "areas_atuacao": "TI",
                "principais_atividades": "Python SQL",
            },
            "beneficios": {},
        }
    }

    df_app = flatten_applicants(applicants_data)
    print(df_app.head())
    df_pro = flatten_prospects(prospects_data)
    print(df_pro.head())
    df_vag = flatten_vagas(vagas_data)
    print(df_vag.head())

    app_flat = tmp_path / "applicants_flat.json"
    pro_flat = tmp_path / "prospects_flat.json"
    vag_flat = tmp_path / "vagas_flat.json"

    df_app.to_json(app_flat)
    df_pro.to_json(pro_flat)
    df_vag.to_json(vag_flat)

    # Build dataframe
    df = fe.build_df_final(app_flat, pro_flat, vag_flat)
    # Should produce exactly one row
    assert len(df) == 1
    row = df.iloc[0]
    # Engineered columns should exist
    engineered_cols = [
        "cand_cidade",
        "cand_uf",
        "cand_regiao",
        "vaga_uf",
        "vaga_cidade_unif",
        "vaga_regiao",
        "same_state",
        "same_city",
        "same_region",
        "meets_academic",
        "meets_english",
        "meets_spanish",
        "sim_tfidf",
        "overlap_kw",
        "jaccard_kw",
        "cand_remuneracao_num",
        "vaga_is_CLT",
        "vaga_is_PJ",
        "vaga_is_Estagiario",
        "vaga_is_Cotas",
        "cand_is_Junior",
        "cand_is_Pleno",
        "cand_is_Senior",
        "vaga_is_Junior",
        "vaga_is_Pleno",
        "vaga_is_Senior",
        "label_contratado",
    ]
    for col in engineered_cols:
        assert col in df.columns, f"Missing engineered column: {col}"
    # Check some boolean engineered values for correctness
    assert row["same_state"] == True
    assert row["same_city"] == True
    assert row["same_region"] == True
    # Candidate has higher academic and English levels than required
    assert row["meets_academic"] == True
    assert row["meets_english"] == True
    # Both candidate and vacancy require no Spanish: meets_spanish should be True
    assert row["meets_spanish"] == True
    # Hiring type CLT flag should be set for vacancy
    assert row["vaga_is_CLT"] == 1
    assert row["vaga_is_PJ"] == 0
    assert row["vaga_is_Estagiario"] == 0
    assert row["vaga_is_Cotas"] == 0
    # Seniority flags: both candidate and vacancy are Pleno
    assert row["cand_is_Pleno"] == 1
    assert row["vaga_is_Pleno"] == 1
    # Label should be 1 because "Contratado" was provided
    assert row["label_contratado"] == 1