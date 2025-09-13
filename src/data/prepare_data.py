"""Prepare data for candidate hiring prediction.

This module handles reading JSON input files and transforming them
into a flattened pandas DataFrame suitable for feature engineering.
It also exposes a ``build_dataset`` function that merges the
applicants, prospects and vacancies (vagas) data into a single
dataset and derives a binary label indicating whether the candidate
was hired.  The implementation is adapted from the original
``prepare_data.py`` script in the `fiap_fase5` repository【86947837380131†L25-L69】【86947837380131†L140-L239】.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import json
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_string_dtype, is_sparse, is_integer_dtype

def load_json(path: Path) -> Dict:
    """Load a JSON file into a Python dict.

    Parameters
    ----------
    path : Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_applicants(raw: Dict) -> pd.DataFrame:
    """Flatten the nested applicants JSON into a DataFrame.

    Each candidate record may contain several nested objects, e.g.
    ``infos_basicas``.  This function expands those nested keys into
    dot‑delimited column names【86947837380131†L25-L42】.
    """
    rows: List[Dict] = []
    for codigo, dados in (raw or {}).items():
        flat: Dict[str, str] = {"codigo_profissional": str(codigo)}
        for bloco, conteudo in (dados or {}).items():
            if isinstance(conteudo, dict):
                for k, v in conteudo.items():
                    flat[f"{bloco}.{k}"] = v
            else:
                flat[bloco] = conteudo
        rows.append(flat)
    df = pd.DataFrame(rows)
    # alias for candidate name
    for col in ["infos_basicas.nome", "informacoes_pessoais.nome"]:
        if col in df.columns:
            df["nome_candidato"] = df[col]
            break
    return df

def flatten_prospects(raw: Dict) -> pd.DataFrame:
    """Flatten the prospects JSON into a DataFrame.

    Each vacancy (vaga) contains a list of prospects (applicants).  This
    function extracts prospect attributes and normalizes date columns【86947837380131†L44-L68】.
    """
    rows: List[Dict] = []
    for vaga_id, vaga in (raw or {}).items():
        title = None
        modality = None
        prospects: List[Dict] = []
        if isinstance(vaga, dict):
            title = vaga.get("titulo")
            modality = vaga.get("modalidade")
            prospects = vaga.get("prospects", []) or []
        for p in prospects:
            rec = dict(p)
            rec["vaga_id"] = str(vaga_id)
            rec["vaga_titulo"] = title
            rec["vaga_modalidade"] = modality
            if "codigo" in rec and pd.notna(rec["codigo"]):
                rec["codigo"] = str(rec["codigo"]).strip()
            rows.append(rec)
    df = pd.DataFrame(rows)
    # fix common typo
    if "situacao_candidado" in df.columns:
        df.rename(columns={"situacao_candidado": "situacao_candidato"}, inplace=True)
    for c in ["data_candidatura", "ultima_atualizacao"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%d-%m-%Y", errors="coerce")
    return df

def flatten_vagas(raw: Dict) -> pd.DataFrame:
    """Flatten the vacancies JSON into a DataFrame【86947837380131†L70-L97】."""
    rows: List[Dict] = []
    for vaga_id, dados in (raw or {}).items():
        flat = {"vaga_id": str(vaga_id)}
        for bloco, conteudo in (dados or {}).items():
            if isinstance(conteudo, dict):
                for k, v in conteudo.items():
                    flat[f"{bloco}.{k}"] = v
            else:
                flat[bloco] = conteudo
        rows.append(flat)
    df = pd.DataFrame(rows)
    # normalize dates and alias some useful fields
    for c in ["informacoes_basicas.data_requicisao", "informacoes_basicas.limite_esperado_para_contratacao"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%d-%m-%Y", errors="coerce")
    if "informacoes_basicas.titulo_vaga" in df.columns:
        df["titulo_vaga"] = df["informacoes_basicas.titulo_vaga"]
    if "perfil_vaga.estado" in df.columns:
        df["estado"] = df["perfil_vaga.estado"]
    if "perfil_vaga.cidade" in df.columns:
        df["cidade"] = df["perfil_vaga.cidade"]
    if "informacoes_basicas.tipo_contratacao" in df.columns:
        df["tipo_contratacao"] = df["informacoes_basicas.tipo_contratacao"]
    if "informacoes_basicas.analista_responsavel" in df.columns:
        df["analista_responsavel"] = df["informacoes_basicas.analista_responsavel"]
    return df

def parse_money(series: pd.Series) -> pd.Series:
    """Convert a salary string into a numeric value【86947837380131†L99-L104】."""
    s = (series.astype(str)
         .str.replace(r"[^0-9,]", "", regex=True)
         .str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")

TECHS = [
    "python","java","javascript","typescript","c#","c++","go","rust","ruby","php","scala",
    "sql","nosql","mysql","postgres","mongodb","spark","hadoop","kafka","airflow",
    "aws","gcp","azure","docker","kubernetes","terraform",
    "pandas","numpy","sklearn","pytorch","tensorflow","keras","xgboost","lightgbm",
    "fastapi","flask","django","react","vue","angular","excel","sap","power_bi","sql_server"
]

def keyword_overlap(a: str, b: str, keywords: List[str] = TECHS) -> float:
    """Compute the keyword overlap between two texts【86947837380131†L105-L112】."""
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    A = {kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", a.lower())}
    B = {kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", b.lower())}
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def clean_cat(s: pd.Series) -> pd.Series:
    """Normalize categorical strings by stripping and replacing common missing tokens【86947837380131†L114-L117】."""
    return (s.astype(str).str.strip()
            .replace(["", "nan", "NA", "None", "N/A", "vazio", "Vazio"], np.nan))

def build_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Construct the merged dataset and return features, label and metadata.

    Parameters
    ----------
    data_dir : Path
        Directory containing the raw JSON files (``applicants.json``,
        ``prospects.json`` and ``vagas.json``).

    Returns
    -------
    X : pd.DataFrame
        DataFrame of feature columns, including id columns.  Use
        ``meta`` to decide which columns to use for modelling.
    y : pd.Series
        Binary target (1 = contratado/hired, 0 = outro).
    meta : Dict
        Dictionary with lists of numeric, categorical and id columns.
    """
    apps = load_json(data_dir / "applicants.json")
    pros = load_json(data_dir / "prospects.json")
    vagas = load_json(data_dir / "vagas.json")

    df_app = flatten_applicants(apps)
    df_pro = flatten_prospects(pros)
    df_vag = flatten_vagas(vagas)

    if df_pro.empty:
        raise ValueError("prospects.json está vazio ou ausente.")

    # joins
    df_app["codigo_profissional"] = df_app.get("codigo_profissional", pd.Series(dtype=str)).astype(str)
    df_pro["codigo"] = df_pro["codigo"].astype(str)
    df_join = df_pro.merge(df_app, left_on="codigo", right_on="codigo_profissional", how="left", suffixes=("_pro","_app"))
    df_full = df_join.merge(df_vag, on="vaga_id", how="left", suffixes=("","_vaga")) if not df_vag.empty else df_join.copy()

    # label: hired indicator
    y = df_full["situacao_candidato"].astype(str).str.lower().str.contains("contratado").astype(int)

    # consolidate text fields
    cand_cols = [c for c in ["cv_pt","informacoes_profissionais.conhecimentos_tecnicos"] if c in df_full.columns]
    vaga_cols = [c for c in ["perfil_vaga.principais_atividades","perfil_vaga.competencia_tecnicas_e_comportamentais","titulo_vaga"] if c in df_full.columns]
    df_full["_cand_text"] = df_full[cand_cols].agg(lambda x: " ".join(x.fillna("").astype(str)), axis=1) if cand_cols else ""
    df_full["_vaga_text"] = df_full[vaga_cols].agg(lambda x: " ".join(x.fillna("").astype(str)), axis=1) if vaga_cols else ""

    # similarity (TF-IDF) and keyword overlap features
    corpus = pd.concat([df_full["_cand_text"], df_full["_vaga_text"]], axis=0).fillna("")
    if len(corpus) > 0:
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(min_df=3, max_features=40000)
        mat = tfidf.fit_transform(corpus.values)
        n = len(df_full)
        M_cand = mat[:n]
        M_vaga = mat[n:]
        sim_cosine = np.array((M_cand.multiply(M_vaga)).sum(axis=1)).ravel() / (
            np.sqrt((M_cand.power(2)).sum(axis=1)).A1 * np.sqrt((M_vaga.power(2)).sum(axis=1)).A1 + 1e-9
        )
        df_full["sim_tfidf"] = sim_cosine
    else:
        df_full["sim_tfidf"] = 0.0
    df_full["overlap_kw"] = [keyword_overlap(ct, vt) for ct, vt in zip(df_full["_cand_text"], df_full["_vaga_text"])]

    # numeric features
    df_full["remuneracao_num"] = parse_money(df_full.get("informacoes_profissionais.remuneracao", pd.Series(dtype=str))).clip(0, 50000)
    if {"data_candidatura","ultima_atualizacao"}.issubset(df_full.columns):
        df_full["tempo_processamento"] = (df_full["ultima_atualizacao"] - df_full["data_candidatura"]).dt.days.clip(0, 200)
    else:
        df_full["tempo_processamento"] = np.nan
    cand_cols_all = [c for c in df_app.columns if c not in {"cv_en"}]
    df_full["cand_missing_ratio"] = df_full[cand_cols_all].isna().mean(axis=1) if cand_cols_all else 0.0
    df_full["cand_text_len"] = df_full["_cand_text"].str.len()
    df_full["vaga_text_len"] = df_full["_vaga_text"].str.len()

    num_cols = [
        "sim_tfidf","overlap_kw",
        "remuneracao_num","tempo_processamento",
        "cand_missing_ratio","cand_text_len","vaga_text_len"
    ]

    # selected categorical columns
    cat_map = {
        "nivel_academico": "formacao_e_idiomas.nivel_academico",
        "nivel_ingles": "formacao_e_idiomas.nivel_ingles",
        "tipo_contratacao": "tipo_contratacao",
        "estado": "estado",
        "cidade": "cidade",
        "recrutador": "recrutador",
        "analista_responsavel": "analista_responsavel",
    }
    cat_cols: List[str] = []
    for out_col, src in cat_map.items():
        if src in df_full.columns:
            df_full[out_col] = clean_cat(df_full[src])
            cat_cols.append(out_col)

    id_cols = ["vaga_id","codigo"]
    X = df_full[id_cols + num_cols + cat_cols].copy()
    meta = {"num_cols": num_cols, "cat_cols": cat_cols, "id_cols": id_cols}
    return X, y, meta