#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer.py
---------
Inferência com modelo LightGBM treinado (pipeline salvo via train.py).
- Constrói as mesmas ENGINEERED features do treino a partir de dados RAW de candidato e vaga
  utilizando as funções do módulo `feature_engineering_simple.py`.
- Faz predição (probabilidade) e expõe importâncias: global (do modelo) e local (por amostra).

Funções principais (para importar em outros scripts):
- load_model(model_path) -> Pipeline
- build_engineered_from_raw(cand: dict, vaga: dict) -> pd.DataFrame
- predict_one(cand: dict, vaga: dict, model_or_path: Union[str, Pipeline], top_k:int=10) -> dict
- predict_batch_from_csv(input_csv, model_or_path, output_csv, include_local=False, top_k=5) -> pd.DataFrame

Uso via CLI:
  # caso único com JSONs
  python infer.py single \
      --model /mnt/data/model_lgbm.joblib \
      --cand-json /mnt/data/cand.json \
      --vaga-json /mnt/data/vaga.json \
      --out-json /mnt/data/pred_single.json

  # lote com CSV de entrada (colunas RAW prefixadas: cand_* e vaga_*)
  python infer.py batch \
      --model /mnt/data/model_lgbm.joblib \
      --input-csv /mnt/data/synthetic_batch_input.csv \
      --output-csv /mnt/data/synthetic_batch_predictions.csv \
      --include-local --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Union

import numpy as np
import pandas as pd
import joblib

# requisitado pelo pipeline salvo
try:
    import lightgbm as lgb  # noqa: F401
except Exception as e:
    raise SystemExit("LightGBM não está instalado. Instale com: pip install lightgbm") from e

# garante que conseguiremos importar o módulo simples
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import src.data.feature_engineering as fe
from sklearn.pipeline import Pipeline

# ----------------------------- Config --------------------------------

ENGINEERED = [
    # MESMO conjunto usado no treino (sem features vazias)
    "cand_cidade","cand_uf","cand_regiao","vaga_uf","vaga_cidade_unif","vaga_regiao",
    "same_state","same_city","same_region",
    "meets_academic","meets_english","meets_spanish",
    "sim_tfidf","overlap_kw","jaccard_kw",
    "cand_remuneracao_num",
    "vaga_is_CLT","vaga_is_PJ","vaga_is_Estagiario","vaga_is_Cotas",
    "cand_is_Junior","cand_is_Pleno","cand_is_Senior",
    "vaga_is_Junior","vaga_is_Pleno","vaga_is_Senior",
]

# -------------------------- Helpers de texto & flags ------------------

def build_text_from_raw_cand(cand: Dict[str, Any]) -> str:
    parts = [
        cand.get("conhecimentos_tecnicos",""),
        cand.get("certificacoes",""),
        cand.get("outras_certificacoes",""),
        cand.get("titulo_profissional",""),
        cand.get("area_atuacao",""),
        cand.get("cv_text",""),
    ]
    return " ".join([p for p in parts if isinstance(p, str)])

def build_text_from_raw_vaga(vaga: Dict[str, Any]) -> str:
    parts = [
        vaga.get("titulo_vaga",""),
        vaga.get("principais_atividades",""),
        vaga.get("competencias",""),
        vaga.get("areas_atuacao",""),
        vaga.get("demais_observacoes",""),
        vaga.get("descricao",""),
    ]
    return " ".join([p for p in parts if isinstance(p, str)])

def hiring_flags(tipo: str) -> Tuple[int,int,int,int]:
    t = (tipo or "").lower()
    return int("clt" in t), int("pj" in t), int("estag" in t), int("cota" in t)

def seniority_flags(s: str) -> Tuple[int,int,int]:
    s = (s or "").lower()
    return int(("junior" in s) or ("júnior" in s)), int("pleno" in s), int(("senior" in s) or ("sênior" in s))

# --------------------------- Feature Builder --------------------------

def build_engineered_from_raw(cand: Dict[str, Any], vaga: Dict[str, Any]) -> pd.DataFrame:
    """Gera 1 linha de features engineered a partir de dados RAW de candidato e vaga.
    Usa funções utilitárias do módulo feature_engineering_simple.py.
    """
    # Geografia
    cand_uf = cand.get("uf"); vaga_uf = vaga.get("uf")
    cand_city = cand.get("cidade"); vaga_city = vaga.get("cidade")
    out: Dict[str, Any] = {
        "cand_cidade": cand_city,
        "cand_uf": cand_uf,
        "cand_regiao": fe.uf_to_region(cand_uf),
        "vaga_uf": vaga_uf,
        "vaga_cidade_unif": vaga_city,
        "vaga_regiao": fe.uf_to_region(vaga_uf),
    }
    out["same_state"]  = int((out["cand_uf"] and out["vaga_uf"]) and (str(out["cand_uf"]).upper()==str(out["vaga_uf"]).upper()))
    out["same_city"]   = int(out["same_state"] and (str(out["cand_cidade"]).lower()==str(out["vaga_cidade_unif"]).lower()))
    out["same_region"] = int((out["cand_regiao"] is not None) and (out["vaga_regiao"] is not None) and (out["cand_regiao"]==out["vaga_regiao"]))

    # Meets (a partir de níveis RAW)
    cand_acad = fe.academic_rank(cand.get("nivel_academico"))
    vaga_acad = fe.academic_rank(vaga.get("nivel_academico"))
    out["meets_academic"] = int(pd.notna(cand_acad) and pd.notna(vaga_acad) and (cand_acad >= vaga_acad))

    cand_eng = fe.lang_rank(cand.get("nivel_ingles"))
    vaga_eng = fe.lang_rank(vaga.get("nivel_ingles"))
    out["meets_english"] = int(pd.notna(cand_eng) and pd.notna(vaga_eng) and (cand_eng >= vaga_eng))

    cand_esp = fe.lang_rank(cand.get("nivel_espanhol"))
    vaga_esp = fe.lang_rank(vaga.get("nivel_espanhol"))
    out["meets_spanish"] = int(pd.notna(cand_esp) and pd.notna(vaga_esp) and (cand_esp >= vaga_esp))

    # Texto -> TF-IDF e overlap
    cand_text = build_text_from_raw_cand(cand)
    vaga_text = build_text_from_raw_vaga(vaga)
    out["sim_tfidf"] = fe.tfidf_sim(cand_text, vaga_text)
    ov, jacc = fe.overlap_and_jaccard(cand_text, vaga_text)
    out["overlap_kw"] = ov
    out["jaccard_kw"] = jacc

    # Remuneração do candidato
    out["cand_remuneracao_num"] = cand.get("remuneracao_num")
    if out["cand_remuneracao_num"] is None:
        out["cand_remuneracao_num"] = fe.to_float_money(str(cand.get("remuneracao","")))

    # Tipo de contratação (vaga) e senioridade
    clt, pj, est, cot = hiring_flags(vaga.get("tipo_contratacao",""))
    out["vaga_is_CLT"] = clt; out["vaga_is_PJ"] = pj
    out["vaga_is_Estagiario"] = est; out["vaga_is_Cotas"] = cot

    cjun, cple, csen = seniority_flags(cand.get("nivel_profissional",""))
    vjun, vple, vsen = seniority_flags(vaga.get("nivel_profissional",""))
    out["cand_is_Junior"] = cjun; out["cand_is_Pleno"] = cple; out["cand_is_Senior"] = csen
    out["vaga_is_Junior"] = vjun; out["vaga_is_Pleno"]  = vple; out["vaga_is_Senior"] = vsen

    # Monta DF com colunas na ordem esperada
    return pd.DataFrame([{k: out.get(k) for k in ENGINEERED}])

# --------------------------- Modelo & Importâncias --------------------

def load_model(model_path: Union[str, Path]) -> Pipeline:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    pipe = joblib.load(model_path)
    # sanity check
    if not isinstance(pipe, Pipeline) or "pre" not in pipe.named_steps or "clf" not in pipe.named_steps:
        raise ValueError("O objeto carregado não parece ser o Pipeline do treino (precisa de steps 'pre' e 'clf').")
    return pipe

def _original_col(name: str) -> str:
    if name.startswith("cat__"):
        rest = name.split("cat__", 1)[1]
        return rest.split("_", 1)[0]
    if name.startswith("num__"):
        return name.split("num__", 1)[1]
    return name

def global_importance(pipe: Pipeline) -> pd.DataFrame:
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    feat_names = pre.get_feature_names_out()
    imps = np.asarray(getattr(clf, "feature_importances_", np.zeros(len(feat_names))), dtype=float)
    imp_df = pd.DataFrame({"feature_encoded": feat_names, "importance": imps})
    imp_df["feature_original"] = imp_df["feature_encoded"].apply(_original_col)
    agg = imp_df.groupby("feature_original", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    return agg

def local_contributions(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Retorna contribuições por feature codificada (n amostras x (n_features+1)).
    Para agregar por coluna original, use `aggregate_local_by_original(...)`.
    """
    pre = pipe.named_steps["pre"]; clf = pipe.named_steps["clf"]
    X_enc = pre.transform(X)
    X_dense = X_enc.toarray() if hasattr(X_enc, "toarray") else np.asarray(X_enc, dtype=float)
    contrib = clf.predict(X_dense, pred_contrib=True)
    return np.asarray(contrib, dtype=float)  # última coluna é o bias

def aggregate_local_by_original(pipe: Pipeline, contrib_row: np.ndarray) -> pd.DataFrame:
    """Agrupa as contribuições (sem o bias) por coluna original."""
    pre = pipe.named_steps["pre"]
    feat_names = pre.get_feature_names_out()
    contrib0 = np.asarray(contrib_row[:-1], dtype=float).ravel()
    df = pd.DataFrame({"feature_encoded": feat_names, "contribution": contrib0})
    df["feature_original"] = df["feature_encoded"].apply(_original_col)
    agg = df.groupby("feature_original", as_index=False)["contribution"].sum()
    agg["abs_contribution"] = agg["contribution"].abs()
    agg = agg.sort_values("abs_contribution", ascending=False, kind="mergesort").reset_index(drop=True)
    return agg

# ----------------------------- Predição -------------------------------

def predict_one(cand: Dict[str, Any], vaga: Dict[str, Any],
                model_or_path: Union[str, Path, Pipeline], top_k: int = 10) -> Dict[str, Any]:
    """Predição para um caso único + importâncias (global e local)."""
    pipe = load_model(model_or_path) if not isinstance(model_or_path, Pipeline) else model_or_path
    X = build_engineered_from_raw(cand, vaga)
    proba = float(pipe.predict_proba(X)[:, 1][0])

    # importâncias
    imp_global = global_importance(pipe)
    contrib = local_contributions(pipe, X)
    local_agg = aggregate_local_by_original(pipe, contrib[0])
    top_local = local_agg.head(top_k)

    return {
        "prob_contratado": proba,
        "global_importance": imp_global.to_dict(orient="records"),
        "local_contributions": top_local.to_dict(orient="records"),
        "X_engineered": X.to_dict(orient="records")[0],
    }

def _row_to_raw_dicts(row: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cand = {k.replace("cand_", ""): row[k] for k in row.index if k.startswith("cand_")}
    vaga = {k.replace("vaga_", ""): row[k] for k in row.index if k.startswith("vaga_")}
    # NaN -> None
    for d in (cand, vaga):
        for k, v in list(d.items()):
            if isinstance(v, float) and np.isnan(v):
                d[k] = None
    return cand, vaga

def predict_batch_from_csv(input_csv: Union[str, Path],
                           model_or_path: Union[str, Path, Pipeline],
                           output_csv: Union[str, Path],
                           include_local: bool = False,
                           top_k: int = 5) -> pd.DataFrame:
    """Predição em lote a partir de CSV (colunas RAW prefixadas cand_ e vaga_).
    Salva CSV com coluna prob_contratado.
    Se include_local=True, adiciona duas colunas: top_features_json e top_contribs_json (listas por linha).
    """
    pipe = load_model(model_or_path) if not isinstance(model_or_path, Pipeline) else model_or_path
    input_csv = Path(input_csv); output_csv = Path(output_csv)
    df_in = pd.read_csv(input_csv)

    engineered_list: List[pd.DataFrame] = []
    for _, row in df_in.iterrows():
        cand_raw, vaga_raw = _row_to_raw_dicts(row)
        Xi = build_engineered_from_raw(cand_raw, vaga_raw)
        engineered_list.append(Xi)
    X_batch = pd.concat(engineered_list, axis=0).reset_index(drop=True)

    proba = pipe.predict_proba(X_batch)[:, 1]
    df_out = df_in.copy()
    df_out["prob_contratado"] = proba

    if include_local:
        # calcula contribuições e pega top_k por linha
        contrib = local_contributions(pipe, X_batch)  # shape: (n, n_feat+1)
        # nomes originais por posição codificada
        pre = pipe.named_steps["pre"]
        encoded_names = list(pre.get_feature_names_out())
        # mapeamento de col enc->orig
        orig_names = [ _original_col(n) for n in encoded_names ]

        top_feats_col: List[str] = []
        top_vals_col: List[str] = []
        for i in range(X_batch.shape[0]):
            row = contrib[i][:-1].astype(float).ravel()  # sem bias
            df_row = pd.DataFrame({"enc": encoded_names, "orig": orig_names, "v": row})
            agg = df_row.groupby("orig", as_index=False)["v"].sum()
            agg["abs_v"] = agg["v"].abs()
            agg = agg.sort_values("abs_v", ascending=False).head(top_k)
            top_feats_col.append(json.dumps(list(agg["orig"])))
            top_vals_col.append(json.dumps([float(x) for x in agg["v"]]))
        df_out["top_features_json"] = top_feats_col
        df_out["top_contribs_json"] = top_vals_col

    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    return df_out

# ------------------------------- CLI ----------------------------------

def _load_json_arg(path_or_json: str) -> Dict[str, Any]:
    p = Path(path_or_json)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    # caso a string seja o JSON inline
    return json.loads(path_or_json)

def main():
    parser = argparse.ArgumentParser(description="Inferência com modelo LightGBM (pipeline) usando engineered features.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("single", help="Predição para um único par candidato/vaga (JSONs).")
    p1.add_argument("--model", required=True, type=str, help="Caminho para o modelo .joblib")
    p1.add_argument("--cand-json", required=True, type=str, help="Caminho para JSON do candidato (ou JSON inline).")
    p1.add_argument("--vaga-json", required=True, type=str, help="Caminho para JSON da vaga (ou JSON inline).")
    p1.add_argument("--out-json", required=False, type=str, default=None, help="Arquivo de saída .json com resultado.")

    p2 = sub.add_parser("batch", help="Predição em lote usando CSV com colunas RAW prefixadas (cand_*, vaga_*).")
    p2.add_argument("--model", required=True, type=str, help="Caminho para o modelo .joblib")
    p2.add_argument("--input-csv", required=True, type=str, help="CSV de entrada com campos RAW (cand_*, vaga_*).")
    p2.add_argument("--output-csv", required=True, type=str, help="CSV de saída com prob_contratado (e colunas extras se pedidas).")
    p2.add_argument("--include-local", action="store_true", help="Adicionar top_k contribuições locais por linha (JSONs).")
    p2.add_argument("--top-k", type=int, default=5, help="Quantas contribuições locais manter por linha.")

    args = parser.parse_args()

    if args.cmd == "single":
        pipe = load_model(args.model)
        cand = _load_json_arg(args.cand_json)
        vaga = _load_json_arg(args.vaga_json)
        res = predict_one(cand, vaga, pipe, top_k=10)
        if args.out_json:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"OK: resultado salvo em {out_path}")
        else:
            print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd == "batch":
        df_out = predict_batch_from_csv(args.input_csv, args.model, args.output_csv,
                                        include_local=args.include_local, top_k=args.top_k)
        print(f"OK: predições salvas em {args.output_csv} (linhas={len(df_out)})")

if __name__ == "__main__":
    main()
