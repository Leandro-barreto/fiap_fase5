from __future__ import annotations
from pathlib import Path
import json, re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_applicants(raw: dict) -> pd.DataFrame:
    rows = []
    for codigo, dados in (raw or {}).items():
        flat = {"codigo_profissional": str(codigo)}
        for bloco, conteudo in (dados or {}).items():
            if isinstance(conteudo, dict):
                for k, v in conteudo.items():
                    flat[f"{bloco}.{k}"] = v
            else:
                flat[bloco] = conteudo
        rows.append(flat)
    df = pd.DataFrame(rows); return df

def flatten_prospects(raw: dict) -> pd.DataFrame:
    rows = []
    for vaga_id, vaga in (raw or {}).items():
        title = None; modality = None; prospects = []
        if isinstance(vaga, dict):
            title = vaga.get("titulo"); modality = vaga.get("modalidade"); prospects = vaga.get("prospects", []) or []
        for p in prospects:
            rec = dict(p); rec["vaga_id"] = str(vaga_id); rec["vaga_titulo"] = title; rec["vaga_modalidade"] = modality
            if "codigo" in rec and rec["codigo"] is not None: rec["codigo"] = str(rec["codigo"]).strip()
            rows.append(rec)
    df = pd.DataFrame(rows)
    if "situacao_candidado" in df.columns: df.rename(columns={"situacao_candidado": "situacao_candidato"}, inplace=True)
    for c in ["data_candidatura","ultima_atualizacao"]:
        if c in df.columns: df[c] = pd.to_datetime(df[c], format="%d-%m-%Y", errors="coerce")
    return df

def flatten_vagas(raw: dict) -> pd.DataFrame:
    rows = []
    for vaga_id, dados in (raw or {}).items():
        flat = {"vaga_id": str(vaga_id)}
        for bloco, conteudo in (dados or {}).items():
            if isinstance(conteudo, dict):
                for k, v in conteudo.items():
                    flat[f"{bloco}.{k}"] = v
            else:
                flat[bloco] = conteudo
        rows.append(flat)
    df = pd.DataFrame(rows); df.columns = [c.replace(" ", "_") for c in df.columns]
    return df

def parse_money(series: pd.Series) -> pd.Series:
    s = (series.astype(str).str.replace(r"[^0-9,]", "", regex=True).str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


from pandas.api.types import is_categorical_dtype, is_string_dtype, is_sparse, is_integer_dtype

def sanitize_for_parquet(df):
    out = df.copy()
    for c in out.columns:
        try:
            if is_categorical_dtype(out[c]) or is_string_dtype(out[c]):
                out[c] = out[c].astype(object)
        except Exception:
            pass
        try:
            if is_integer_dtype(out[c]) and str(out[c].dtype).startswith(("Int","UInt")):
                out[c] = out[c].astype("float64")
        except Exception:
            pass
        try:
            if is_sparse(out[c]):
                out[c] = out[c].sparse.to_dense()
        except Exception:
            pass
        out[c] = out[c].apply(lambda x: str(x) if isinstance(x, (list, dict, set)) else x)
    return out

def build_dataset(data_dir: Path, out_dir: Path|None=None):
    data_dir = Path(data_dir)
    apps = load_json(data_dir / "applicants.json")
    pros = load_json(data_dir / "prospects.json")
    vagas = load_json(data_dir / "vagas.json")

    df_app = flatten_applicants(apps); df_pro = flatten_prospects(pros); df_vag = flatten_vagas(vagas)

    if df_pro.empty: raise ValueError("prospects.json está vazio ou ausente.")
    df_app["codigo_profissional"] = df_app.get("codigo_profissional", pd.Series(dtype=str)).astype(str)
    df_pro["codigo"] = df_pro["codigo"].astype(str)
    df_join = df_pro.merge(df_app, left_on="codigo", right_on="codigo_profissional", how="left", suffixes=("_pro","_app"))
    df_full = df_join.merge(df_vag, on="vaga_id", how="left", suffixes=("","_vaga")) if not df_vag.empty else df_join.copy()

    y = df_full["situacao_candidato"].astype(str).str.lower().str.contains("contratado").astype(int)

    # Minimal features (ids + few numerics to keep file simple here)
    def parse_money(series: pd.Series) -> pd.Series:
        s = (series.astype(str).str.replace(r"[^0-9,]", "", regex=True).str.replace(",", ".", regex=False))
        return pd.to_numeric(s, errors="coerce")

    df_full["remuneracao_num"] = parse_money(df_full.get("informacoes_profissionais.remuneracao", pd.Series(dtype=str))).clip(0, 50000)
    if {"data_candidatura","ultima_atualizacao"}.issubset(df_full.columns):
        df_full["tempo_processamento"] = (df_full["ultima_atualizacao"] - df_full["data_candidatura"]).dt.days.clip(0, 200)
    else:
        df_full["tempo_processamento"] = np.nan

    num_cols = ["remuneracao_num","tempo_processamento"]
    cat_cols = []
    id_cols = ["vaga_id","codigo"]

    X = df_full[id_cols + num_cols + cat_cols].copy()
    meta = {"num_cols": num_cols, "cat_cols": cat_cols, "id_cols": id_cols}

    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        X_save = sanitize_for_parquet(X.assign(label=y))
        try:
            X_save.to_parquet(out_dir / "train.parquet", index=False, engine="pyarrow")
        except Exception as e:
            print(f"[WARN] Falha ao salvar Parquet com pyarrow: {e}\nSalvando CSV como fallback.")
            X_save.to_csv(out_dir / "train.csv", index=False)
    return X, y, meta, df_full
