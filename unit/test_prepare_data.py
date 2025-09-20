"""Unit tests for the data preparation module.

These tests cover the helper functions in ``src/data/prepare_data.py`` such as
``flatten_applicants``, ``flatten_prospects``, ``flatten_vagas``,
``parse_money`` and ``keyword_overlap``.  They also exercise the
``build_dataset`` function on a small toy dataset to ensure that it
produces the expected feature matrix, label vector and metadata.  Heavy
text vectorisation is bypassed by monkeypatching ``TfidfVectorizer`` to a
dummy implementation, keeping the tests fast and deterministic.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import prepare_data


def test_flatten_applicants() -> None:
    """Flattening applicants should expand nested keys and alias name."""
    raw = {
        "1": {
            "infos_basicas": {
                "nome": "Alice",
                "nivel_ingles": "Avançado",
            },
            "informacoes_profissionais": {
                "conhecimentos_tecnicos": "python pandas",
            },
        }
    }
    df = prepare_data.flatten_applicants(raw)
    # Expect columns for nested keys
    assert "infos_basicas.nome" in df.columns
    assert "informacoes_profissionais.conhecimentos_tecnicos" in df.columns
    # Alias nome_candidato should be present
    assert "nome_candidato" in df.columns
    assert df.loc[0, "nome_candidato"] == "Alice"


def test_flatten_prospects_normalisation() -> None:
    """Flattening prospects should normalise dates and fix typos."""
    raw = {
        "v1": {
            "titulo": "Data Scientist",
            "modalidade": "CLT",
            "prospects": [
                {
                    "codigo": "1",
                    "situacao_candidado": "Contratado",
                    "data_candidatura": "01-01-2024",
                    "ultima_atualizacao": "10-01-2024",
                }
            ],
        }
    }
    df = prepare_data.flatten_prospects(raw)
    # Typo should be corrected
    assert "situacao_candidato" in df.columns
    # Dates should be parsed into datetime
    assert pd.api.types.is_datetime64_any_dtype(df["data_candidatura"])
    assert pd.api.types.is_datetime64_any_dtype(df["ultima_atualizacao"])
    # Additional metadata columns
    assert df.loc[0, "vaga_titulo"] == "Data Scientist"
    assert df.loc[0, "vaga_modalidade"] == "CLT"


def test_flatten_vagas_aliases() -> None:
    """Flattening vagas should alias useful fields such as titulo, estado and tipo."""
    raw = {
        "v1": {
            "informacoes_basicas": {
                "titulo_vaga": "Engenheiro de Dados",
                "tipo_contratacao": "CLT",
                "analista_responsavel": "Maria",
            },
            "perfil_vaga": {
                "estado": "SP",
                "cidade": "São Paulo",
            },
        }
    }
    df = prepare_data.flatten_vagas(raw)
    # Aliased columns
    assert df.loc[0, "titulo_vaga"] == "Engenheiro de Dados"
    assert df.loc[0, "estado"] == "SP"
    assert df.loc[0, "cidade"] == "São Paulo"
    assert df.loc[0, "tipo_contratacao"] == "CLT"
    assert df.loc[0, "analista_responsavel"] == "Maria"


def test_parse_money_series() -> None:
    """Parsing monetary strings should normalise separators and return floats."""
    series = pd.Series(["1.234,56", "R$ 2.000", None, "abc"])
    out = prepare_data.parse_money(series)
    # First value: remove thousand separator and comma -> 1234.56
    assert np.isclose(out.iloc[0], 1234.56)
    # Second value: remove currency symbol -> 2000
    assert out.iloc[1] == 2000.0
    # None or non‑numeric should become NaN
    assert np.isnan(out.iloc[2])
    assert np.isnan(out.iloc[3])


def test_keyword_overlap_ratio() -> None:
    """Keyword overlap should compute intersection over union of matched keywords."""
    a = "Python pandas numpy"
    b = "Numpy sklearn"  # intersection {numpy}; union {python,pandas,numpy,sklearn}
    ratio = prepare_data.keyword_overlap(a, b)
    # intersection size 1, union size 4 -> 1/4 = 0.25
    assert pytest.approx(ratio, rel=1e-6) == 0.25
    # Non-string inputs should yield 0
    assert prepare_data.keyword_overlap(123, "abc") == 0.0


def test_build_dataset_with_toy_data(sample_data_dir, monkeypatch) -> None:
    """build_dataset should produce correct labels and metadata on toy data."""
    # Patch TfidfVectorizer to avoid heavy computation
    class DummyVec:
        def fit_transform(self, corpus):
            # return a simple sparse-like matrix with zeros
            import numpy as np
            # shape (n_docs, 1)
            return np.zeros((len(corpus), 1))

    # The TfidfVectorizer is imported inside the function from scikit-learn, so
    # patch the class at its fully qualified location.  This prevents heavy
    # computation when computing TF-IDF features.
    monkeypatch.setattr(
        "sklearn.feature_extraction.text.TfidfVectorizer",
        lambda *args, **kwargs: DummyVec(),
    )
    X, y, meta = prepare_data.build_dataset(sample_data_dir)
    # Expect two rows corresponding to two prospects
    assert len(X) == 2
    # Labels should reflect situacao_candidato: first Contratado -> 1, second Rejeitado -> 0
    assert list(y.tolist()) == [1, 0]
    # Numeric and categorical columns present
    assert set(meta["id_cols"]) == {"vaga_id", "codigo"}
    assert "sim_tfidf" in X.columns
    assert "overlap_kw" in X.columns
    assert all(col in X.columns for col in meta["num_cols"])
    assert all(col in X.columns for col in meta["cat_cols"])


def test_build_dataset_empty_prospects(empty_prospects_data_dir) -> None:
    """When prospects.json is empty, build_dataset should raise ValueError."""
    with pytest.raises(ValueError):
        prepare_data.build_dataset(empty_prospects_data_dir)