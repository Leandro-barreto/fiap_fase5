"""
Tests for the prepare_data module.

These tests exercise the flattening functions used to convert the raw JSON
dictionaries of applicants, prospects and vacancies into flat pandas
DataFrames.  The goal is to ensure that nested structures are
flattened correctly, types are normalized (e.g., codes to strings), and
aliases for important fields are populated.
"""

from pathlib import Path

import pandas as pd

from src.data import prepare_data as pdmod


def test_flatten_applicants_basic():
    """Simple applicant structures should be flattened with dot notation."""
    raw = {
        "1": {
            "infos_basicas": {"nome": "Fulano", "idade": 30},
            "informacoes_profissionais": {"cargo": "Dev"},
        }
    }
    df = pdmod.flatten_applicants(raw)
    # One row per applicant
    assert len(df) == 1
    row = df.iloc[0]
    # codigo_profissional should be taken from the dict key and cast to str
    assert row["codigo_profissional"] == "1"
    # Flattened keys combine block and field name
    assert row["infos_basicas.nome"] == "Fulano"
    assert row["informacoes_profissionais.cargo"] == "Dev"
    # nome_candidato should be set using the first available name field
    assert row["nome_candidato"] == "Fulano"


def test_flatten_prospects_basic():
    """Prospect lists should be exploded into separate rows with correct metadata."""
    raw = {
        "123": {
            "titulo": "Eng de Dados",
            "modalidade": "Presencial",
            "prospects": [
                {"nome": "Fulano", "codigo": "1", "situacao_candidato": "Contratado"},
                {"nome": "Beltrano", "codigo": "2", "situacao_candidato": "Reprovado"},
            ],
        }
    }
    df = pdmod.flatten_prospects(raw)
    # Should have as many rows as prospects
    assert len(df) == 2
    # Check that vacancy metadata is propagated
    assert set(df["vaga_id"]) == {"123"}
    assert set(df["vaga_titulo"]) == {"Eng de Dados"}
    assert set(df["vaga_modalidade"]) == {"Presencial"}
    # situacao_candidato should be the corrected column name
    assert "situacao_candidato" in df.columns
    # data_candidatura should be parsed to datetime if present
    raw2 = {
        "1": {
            "titulo": "",
            "modalidade": "",
            "prospects": [
                {"codigo": "1", "data_candidatura": "10-05-2023"},
            ],
        }
    }
    df2 = pdmod.flatten_prospects(raw2)
    assert pd.api.types.is_datetime64_any_dtype(df2["data_candidatura"]) if "data_candidatura" in df2.columns else True


def test_flatten_vagas_basic():
    """Vacancy structures should flatten nested keys and expose aliases."""
    raw = {
        "999": {
            "informacoes_basicas": {
                "titulo_vaga": "Dev", 
                "tipo_contratacao": "CLT", 
                "analista_responsavel": "Ana"
            },
            "perfil_vaga": {
                "estado": "SP",
                "cidade": "São Paulo",
                "nivel_profissional": "Pleno",
            },
            "beneficios": {},
        }
    }
    df = pdmod.flatten_vagas(raw)
    assert len(df) == 1
    row = df.iloc[0]
    # vaga_id should come from the dictionary key
    assert row["vaga_id"] == "999"
    # Nested fields flattened with dot notation
    assert row["informacoes_basicas.titulo_vaga"] == "Dev"
    # Aliases should be provided
    assert row["titulo_vaga"] == "Dev"
    assert row["estado"] == "SP"
    assert row["cidade"] == "São Paulo"
    assert row["tipo_contratacao"] == "CLT"
    assert row["analista_responsavel"] == "Ana"