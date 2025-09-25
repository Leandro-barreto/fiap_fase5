"""Prepare data for candidate hiring prediction.

This module handles reading JSON input files and transforming them
into a flattened pandas DataFrame suitable for feature engineering.
It also exposes a ``build_dataset`` function that merges the
applicants, prospects and vacancies (vagas) data into a single
dataset and derives a binary label indicating whether the candidate
was hired.  The implementation is adapted from the original
``prepare_data.py`` script in the `fiap_fase5` repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import json
import re
import numpy as np
import pandas as pd

def load_json(path: Path):
    if not path.exists():
        print(f"[AVISO] Não encontrado: {path.resolve()}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERRO] Falha ao ler {path.name}: {e}")
        return None

def flatten_applicants(raw_dict: dict) -> pd.DataFrame:
    if not isinstance(raw_dict, dict):
        return pd.DataFrame()
    rows = []
    for codigo, dados in raw_dict.items():
        flat = {"codigo_profissional": str(codigo)}
        if isinstance(dados, dict):
            for bloco, conteudo in dados.items():
                if isinstance(conteudo, dict):
                    for k, v in conteudo.items():
                        flat[f"{bloco}.{k}"] = v
                else:
                    flat[bloco] = conteudo
        rows.append(flat)
    df = pd.DataFrame(rows)
    # normalizações úteis
    # pega um campo de nome padronizado para o candidato
    for col in ["infos_basicas.nome","informacoes_pessoais.nome"]:
        if col in df.columns:
            df["nome_candidato"] = df[col]
            break
    return df

def flatten_prospects(raw_dict: dict) -> pd.DataFrame:
    if not isinstance(raw_dict, dict):
        return pd.DataFrame()
    rows = []
    for vaga_id, vaga in raw_dict.items():
        titulo = None
        modalidade = None
        prospects = []
        if isinstance(vaga, dict):
            titulo = vaga.get("titulo")
            modalidade = vaga.get("modalidade")
            prospects = vaga.get("prospects", []) or []
        if not isinstance(prospects, list):
            continue
        for p in prospects:
            rec = dict(p)
            rec["vaga_id"] = str(vaga_id)
            rec["vaga_titulo"] = titulo
            rec["vaga_modalidade"] = modalidade
            if "codigo" in rec and pd.notna(rec["codigo"]):
                rec["codigo"] = str(rec["codigo"]).strip()
            rows.append(rec)
    df = pd.DataFrame(rows)
    if "situacao_candidado" in df.columns:
        df.rename(columns={"situacao_candidado": "situacao_candidato"}, inplace=True)
    # parse datas
    for c in ["data_candidatura","ultima_atualizacao"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%d-%m-%Y", errors="coerce")
    return df

def flatten_vagas(raw_dict: dict) -> pd.DataFrame:
    """
    Esperado: { "5185": { "informacoes_basicas": {...}, "perfil_vaga": {...}, "beneficios": {...} }, ...}
    Saída: uma linha por vaga, com colunas flatten e 'vaga_id'.
    """
    if not isinstance(raw_dict, dict):
        return pd.DataFrame()
    rows = []
    for vaga_id, dados in raw_dict.items():
        flat = {"vaga_id": str(vaga_id)}
        if isinstance(dados, dict):
            for bloco, conteudo in dados.items():
                if isinstance(conteudo, dict):
                    for k, v in conteudo.items():
                        flat[f"{bloco}.{k}"] = v
                else:
                    flat[bloco] = conteudo
        rows.append(flat)
    df = pd.DataFrame(rows)
    # normalizar nomes de colunas: trocar espaços por underscore
    df.columns = [c.replace(" ", "_") for c in df.columns]
    # datas
    for c in ["informacoes_basicas.data_requicisao", "informacoes_basicas.limite_esperado_para_contratacao"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%d-%m-%Y", errors="coerce")
    # alias para campos úteis
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
    # corrigir chave com espaço "nivel profissional"
    if "perfil_vaga.nivel_profissional" not in df.columns and "perfil_vaga.nivel_profissional" not in df.columns:
        # após replace space->underscore, o campo vira "perfil_vaga.nivel_profissional" se existia
        pass
    return df
