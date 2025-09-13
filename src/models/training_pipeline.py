"""
training_pipeline.py – Pipeline de treinamento para predição de contratação
==========================================================================

Este script define um pipeline de Machine Learning para prever se um
candidato será aprovado para uma vaga. O pipeline usa como base as
estruturas e funções de preparação de dados definidas no repositório
``fiap_fase5`` (vide ``src/data/prepare_data.py``). As funções aqui
copiadas e adaptadas fazem o flatten de múltiplos arquivos JSON
(``applicants.json``, ``prospects.json`` e ``vagas.json``), calculam
características numéricas e categóricas e retornam um conjunto de
features ``X`` e um vetor de rótulos ``y``【86947837380131†L140-L170】.  

Após o pré‑processamento, o script constrói um transformador de
colunas (``ColumnTransformer``) que aplica ``StandardScaler`` às
colunas numéricas e ``OneHotEncoder`` às categóricas. Em seguida,
treina um modelo de regressão logística para classificar o candidato
como "contratado" ou "não contratado". O desempenho é avaliado com
``accuracy`` e ``f1_score`` sobre um conjunto hold‑out.  

Para executar o pipeline utilize:

```
python training_pipeline.py --data-dir ./data/raw
```

As funções de flatten e construção do dataset são inspiradas diretamente
no código original do projeto, que define a leitura dos JSONs e
manipulação dos campos【86947837380131†L140-L239】.  
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from pandas.api.types import is_categorical_dtype, is_string_dtype, is_sparse, is_integer_dtype


# ---------------------------------------------------------------------------
# Funções de leitura e flatten de dados
# Estas funções foram extraídas e simplificadas de ``src/data/prepare_data.py``
# do repositório fiap_fase5【86947837380131†L25-L69】.  

def load_json(path: Path) -> Dict:
    """Carrega um arquivo JSON e retorna um dicionário.

    Parameters
    ----------
    path : Path
        Caminho até o arquivo JSON.

    Returns
    -------
    dict
        Conteúdo do arquivo JSON como dicionário.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_applicants(raw: Dict) -> pd.DataFrame:
    """Converte o dicionário de applicants em DataFrame flat.

    Cada candidato possui vários blocos (por exemplo, ``infos_basicas``),
    e este método expande chaves/valores para colunas do DataFrame【86947837380131†L25-L42】.
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
    # alias útil para nome do candidato
    for col in ["infos_basicas.nome", "informacoes_pessoais.nome"]:
        if col in df.columns:
            df["nome_candidato"] = df[col]
            break
    return df


def flatten_prospects(raw: Dict) -> pd.DataFrame:
    """Converte o dicionário de prospects em DataFrame flat.

    Cada vaga possui um array de prospects (candidatos aplicados). A
    função extrai atributos da vaga e do prospect, e normaliza datas
    para tipo ``datetime64[ns]``【86947837380131†L44-L68】.
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
            # normaliza código para string
            if "codigo" in rec and pd.notna(rec["codigo"]):
                rec["codigo"] = str(rec["codigo"]).strip()
            rows.append(rec)
    df = pd.DataFrame(rows)
    if "situacao_candidado" in df.columns:
        df.rename(columns={"situacao_candidado": "situacao_candidato"}, inplace=True)
    for c in ["data_candidatura", "ultima_atualizacao"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%d-%m-%Y", errors="coerce")
    return df


def flatten_vagas(raw: Dict) -> pd.DataFrame:
    """Converte o dicionário de vagas em DataFrame flat【86947837380131†L70-L97】."""
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
    # normaliza datas e aliases
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
    """Converte string de remuneração para número【86947837380131†L99-L104】."""
    s = (series.astype(str)
         .str.replace(r"[^0-9,]", "", regex=True)
         .str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


TECHS = [
    "python", "java", "javascript", "typescript", "c#", "c++", "go", "rust", "ruby", "php", "scala",
    "sql", "nosql", "mysql", "postgres", "mongodb", "spark", "hadoop", "kafka", "airflow",
    "aws", "gcp", "azure", "docker", "kubernetes", "terraform",
    "pandas", "numpy", "sklearn", "pytorch", "tensorflow", "keras", "xgboost", "lightgbm",
    "fastapi", "flask", "django", "react", "vue", "angular", "excel", "sap", "power_bi", "sql_server"
]


def keyword_overlap(a: str, b: str, keywords: List[str] = TECHS) -> float:
    """Calcula a sobreposição de palavras‑chave entre dois textos【86947837380131†L105-L112】.

    Retorna a proporção de keywords compartilhadas pelo conjunto união.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    A = {kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", a.lower())}
    B = {kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", b.lower())}
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def clean_cat(s: pd.Series) -> pd.Series:
    """Normaliza valores categóricos removendo strings vazias e placeholders【86947837380131†L114-L117】."""
    return (s.astype(str).str.strip()
            .replace(["", "nan", "NA", "None", "N/A", "vazio", "Vazio"], np.nan))


def build_dataset(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Constroi dataset a partir de arquivos JSON.

    Parameters
    ----------
    data_dir : Path
        Diretório contendo ``applicants.json``, ``prospects.json`` e ``vagas.json``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, Dict]
        ``X`` (DataFrame de features), ``y`` (Series de rótulos) e
        ``meta`` (dicionário com colunas numéricas, categóricas e de id)
        conforme definido no código original【86947837380131†L140-L239】.
    """
    apps = load_json(data_dir / "applicants.json")
    pros = load_json(data_dir / "prospects.json")
    vagas = load_json(data_dir / "vagas.json")

    df_app = flatten_applicants(apps)
    df_pro = flatten_prospects(pros)
    df_vag = flatten_vagas(vagas)

    if df_pro.empty:
        raise ValueError("prospects.json está vazio ou ausente.")

    # join das tabelas
    df_app["codigo_profissional"] = df_app.get("codigo_profissional", pd.Series(dtype=str)).astype(str)
    df_pro["codigo"] = df_pro["codigo"].astype(str)
    df_join = df_pro.merge(df_app, left_on="codigo", right_on="codigo_profissional", how="left", suffixes=("_pro","_app"))
    df_full = df_join.merge(df_vag, on="vaga_id", how="left", suffixes=("","_vaga")) if not df_vag.empty else df_join.copy()

    # label: 1 para contratado, 0 caso contrário
    y = df_full["situacao_candidato"].astype(str).str.lower().str.contains("contratado").astype(int)

    # texto consolidado
    cand_cols = [c for c in ["cv_pt", "informacoes_profissionais.conhecimentos_tecnicos"] if c in df_full.columns]
    vaga_cols = [c for c in ["perfil_vaga.principais_atividades", "perfil_vaga.competencia_tecnicas_e_comportamentais", "titulo_vaga"] if c in df_full.columns]
    df_full["_cand_text"] = df_full[cand_cols].agg(lambda x: " ".join(x.fillna("").astype(str)), axis=1) if cand_cols else ""
    df_full["_vaga_text"] = df_full[vaga_cols].agg(lambda x: " ".join(x.fillna("").astype(str)), axis=1) if vaga_cols else ""

    # similaridade TF-IDF
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

    # overlap de keywords
    df_full["overlap_kw"] = [keyword_overlap(ct, vt) for ct, vt in zip(df_full["_cand_text"], df_full["_vaga_text"])]

    # remuneração numérica
    df_full["remuneracao_num"] = parse_money(df_full.get("informacoes_profissionais.remuneracao", pd.Series(dtype=str))).clip(0, 50000)

    # tempo de processamento
    if {"data_candidatura", "ultima_atualizacao"}.issubset(df_full.columns):
        df_full["tempo_processamento"] = (df_full["ultima_atualizacao"] - df_full["data_candidatura"]).dt.days.clip(0, 200)
    else:
        df_full["tempo_processamento"] = np.nan

    # missing ratio e tamanhos de texto
    cand_cols_all = [c for c in df_app.columns if c not in {"cv_en"}]
    df_full["cand_missing_ratio"] = df_full[cand_cols_all].isna().mean(axis=1) if cand_cols_all else 0.0
    df_full["cand_text_len"] = df_full["_cand_text"].str.len()
    df_full["vaga_text_len"] = df_full["_vaga_text"].str.len()

    # colunas numéricas selecionadas
    num_cols = [
        "sim_tfidf", "overlap_kw",
        "remuneracao_num", "tempo_processamento",
        "cand_missing_ratio", "cand_text_len", "vaga_text_len"
    ]

    # categorias selecionadas
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

    id_cols = ["vaga_id", "codigo"]
    X = df_full[id_cols + num_cols + cat_cols].copy()
    meta = {"num_cols": num_cols, "cat_cols": cat_cols, "id_cols": id_cols}
    return X, y, meta


# ---------------------------------------------------------------------------
# Função principal de treinamento

def train_pipeline(data_dir: Path, test_size: float = 0.2, random_state: int = 42) -> None:
    """Executa o pipeline de treinamento e avaliação.

    Esta função carrega os dados do diretório ``data_dir``, constrói
    o dataset (``X``, ``y`` e metadados), divide em treino e teste,
    aplica pré‑processamento e treina um modelo de regressão logística.
    Por fim, imprime métricas de acurácia e F1 no conjunto de teste.

    Parameters
    ----------
    data_dir : Path
        Diretório contendo ``applicants.json``, ``prospects.json`` e
        ``vagas.json``.
    test_size : float, opcional
        Proporção de exemplos reservada para o teste. Padrão 0.2.
    random_state : int, opcional
        Semente para a divisão aleatória. Padrão 42.
    """
    # Carrega e prepara dados
    X, y, meta = build_dataset(data_dir)

    # Separa IDs, features numéricas e categóricas
    id_cols = meta["id_cols"]
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]

    # Remover colunas de ID do conjunto de treino
    X_feat = X.drop(columns=id_cols)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pré‑processamento: escala numéricos e codifica categorias
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop"
    )

    # Modelo de classificação
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)

    # Pipeline completo
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # Treina o modelo
    model.fit(X_train, y_train)

    # Predição e avaliação
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Acurácia: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de treinamento de predição de contratação")
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Diretório contendo applicants.json, prospects.json e vagas.json"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Proporção reservada para o conjunto de teste (padrão=0.2)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Semente para randomização (padrão=42)"
    )
    args = parser.parse_args()
    train_pipeline(Path(args.data_dir), test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":
    main()