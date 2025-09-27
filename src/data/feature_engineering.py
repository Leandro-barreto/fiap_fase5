#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_engineering.py

Gera o dataframe final com features engineered a partir dos arquivos brutos de applicants, prospects e vagas.
Sem análises/plots. Salva em CSV (e opcionalmente Parquet).

Uso:
    python feature_engineering.py \
        --applicants /caminho/applicants.json \
        --prospects  /caminho/prospects.json  \
        --vagas      /caminho/vagas.json      \
        --out-csv    /caminho/df_final.csv    \
        [--out-parquet /caminho/df_final.parquet]

Se algum caminho não for informado, usa valores padrão.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- Helpers --------------------

BR_STATE_TO_REGION = {
    "AC":"Norte","AM":"Norte","AP":"Norte","PA":"Norte","RO":"Norte","RR":"Norte","TO":"Norte",
    "AL":"Nordeste","BA":"Nordeste","CE":"Nordeste","MA":"Nordeste","PB":"Nordeste","PE":"Nordeste","PI":"Nordeste","RN":"Nordeste","SE":"Nordeste",
    "DF":"Centro-Oeste","GO":"Centro-Oeste","MT":"Centro-Oeste","MS":"Centro-Oeste",
    "ES":"Sudeste","MG":"Sudeste","RJ":"Sudeste","SP":"Sudeste",
    "PR":"Sul","RS":"Sul","SC":"Sul",
}
STATE_NAME_TO_UF = {
    "Acre":"AC","Amazonas":"AM","Amapá":"AP","Pará":"PA","Rondônia":"RO","Roraima":"RR","Tocantins":"TO",
    "Alagoas":"AL","Bahia":"BA","Ceará":"CE","Maranhão":"MA","Paraíba":"PB","Pernambuco":"PE","Piauí":"PI","Rio Grande do Norte":"RN","Sergipe":"SE",
    "Distrito Federal":"DF","Goiás":"GO","Mato Grosso":"MT","Mato Grosso do Sul":"MS",
    "Espírito Santo":"ES","Minas Gerais":"MG","Rio de Janeiro":"RJ","São Paulo":"SP",
    "Paraná":"PR","Rio Grande do Sul":"RS","Santa Catarina":"SC",
    "são paulo":"SP","sao paulo":"SP","sp":"SP"
}

def uf_to_region(uf: Optional[str]) -> Optional[str]:
    if not isinstance(uf, str) or not uf:
        return None
    return BR_STATE_TO_REGION.get(uf.upper(), None)

def robust_read_json(p: Path) -> pd.DataFrame:
    try:
        return pd.read_json(p, dtype=object)
    except Exception:
        with open(p, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))

def sanitize_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    mapping = {c: prefix + re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_") for c in df.columns}
    return df.rename(columns=mapping)


def extract_city_state_from_local(local: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not isinstance(local, str) or not local.strip():
        return (None, None, None)
    parts = re.split(r"[,-/|]+", local.strip())
    parts = [p.strip() for p in parts if p.strip()]
    city, uf, state_name = None, None, None
    if len(parts) == 1:
        p = parts[0]
        if len(p) == 2:
            uf = p.upper()
        else:
            state_name = p
            uf = STATE_NAME_TO_UF.get(p, None)
    else:
        city = parts[0]
        sec = parts[1]
        if len(sec) == 2:
            uf = sec.upper()
        else:
            state_name = sec
            uf = STATE_NAME_TO_UF.get(sec, None)
    if not uf and state_name:
        uf = STATE_NAME_TO_UF.get(state_name, None)
    return (city, uf, state_name)

def to_float_money(s):
    if not isinstance(s, str):
        return np.nan
    t = s.replace("R$", "").replace(".", "").replace(" ", "").replace("\u00a0", "")
    t = t.replace(",", ".")
    try:
        return float(t)
    except Exception:
        return np.nan

def has_tokens(s: str) -> bool:
    if not isinstance(s, str): return False
    toks = re.findall(r"\b\w{3,}\b", s.lower())
    return len(toks) > 0

def tfidf_sim(a: str, b: str) -> float:
    if not has_tokens(a) or not has_tokens(b): 
        return 0.0
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", min_df=1)
    X = vec.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0,0])

def tokenize_kw(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.lower()
    s = re.sub(r"[^a-záéíóúâêôãõç0-9#]+", " ", s)
    exceptions = {'c#', 'f#', 'go', 'r'}  # linguagens com nomes curtos
    return [t for t in s.split() if len(t) > 2 or t in exceptions]

def overlap_and_jaccard(a: str, b: str):
    A, B = set(tokenize_kw(a)), set(tokenize_kw(b))
    if not A or not B:
        return 0, 0.0
    inter = A & B
    jacc = len(inter) / len(A | B)
    return len(inter), jacc

def academic_rank(label: str):
    order = {"Nenhum":0,"Ensino Fundamental":1,"Ensino Médio":2,"Ensino Superior Incompleto":3,"Ensino Superior Completo":4,"Pós-Graduação":5,"Mestrado":6,"Doutorado":7}
    return order.get(label, np.nan)

def lang_rank(label: str):
    order = {"Nenhum":0,"Básico":1,"Intermediário":2,"Avançado":3,"Fluente":4}
    return order.get(label, np.nan)

def seniority_flags(s: Optional[str]):
    s = (s or "").lower() if isinstance(s, str) else ""
    return int("junior" in s or "júnior" in s), int("pleno" in s), int("senior" in s or "sênior" in s)

# -------------------- Core pipeline --------------------

def build_df_final(applicants_path: Path, prospects_path: Path, vagas_path: Path) -> pd.DataFrame:
    # Load and prefix
    applicants = sanitize_cols(robust_read_json(applicants_path), "cand_")
    prospects  = sanitize_cols(robust_read_json(prospects_path),  "prop_")
    vagas      = sanitize_cols(robust_read_json(vagas_path),      "vaga_")

    for col in ["prop_codigo", "prop_vaga_id"]:
        if col in prospects.columns:
            prospects[col] = prospects[col].astype(str)
    if "vaga_vaga_id" in vagas.columns:
        vagas["vaga_vaga_id"] = vagas["vaga_vaga_id"].astype(str)

    merged = prospects.merge(vagas, left_on="prop_vaga_id", right_on="vaga_vaga_id", how="left", suffixes=("", "_v"))

    cand_code_col = None
    for c in ["cand_infos_basicas_codigo_profissional", "cand_codigo_profissional"]:
        if c in applicants.columns:
            cand_code_col = c
            applicants[c] = applicants[c].astype(str)
            break
    if cand_code_col is None:
        applicants["cand_cod_join_fallback"] = ""
        cand_code_col = "cand_cod_join_fallback"

    merged = merged.merge(applicants, left_on="prop_codigo", right_on=cand_code_col, how="left", suffixes=("", "_c"))

    # Filter Technology
    mask_ti_areas = merged.get("vaga_perfil_vaga_areas_atuacao", "").astype(str).str.contains("TI", case=False, na=False)
    tech_keywords = r"(sap|sql|application|developer|desenvolvedor|analista|technical|architect|control|microfocus|quality|engineer|peoplesoft)"
    mask_titles = (
        merged.get("vaga_informacoes_basicas_titulo_vaga", "").astype(str).str.contains(tech_keywords, case=False, na=False) |
        merged.get("vaga_titulo_vaga", "").astype(str).str.contains(tech_keywords, case=False, na=False) |
        merged.get("prop_vaga_titulo", "").astype(str).str.contains(tech_keywords, case=False, na=False)
    )
    merged = merged[mask_ti_areas | mask_titles].copy()

    # Label
    categorias_nao_contratado = [
        "Não Aprovado pelo Cliente",
        "Desistiu",
        "Não Aprovado pelo RH",
        "Não Aprovado pelo Requisitante",
        "Não Aprovado pelo Requisitante",
        "Recusado",
    ]

    categorias_contratado = [
        "Contratado pela Decision",
        "Contratado como Hunting",
        "Aprovado",
        "Encaminhar Proposta",
        "Proposta Aceita",
        "Documentação PJ",
        "Documentação CLT",
        "Documentação Cooperado",
    ]

    keep_label = categorias_contratado + categorias_nao_contratado

    merged = merged[merged["prop_situacao_candidato"].isin(keep_label)]


    merged["label_contratado"] = merged["prop_situacao_candidato"].isin(categorias_contratado).astype(int)

    # Geography
    cand_local = merged.get("cand_infos_basicas_local", "")
    loc_parsed = cand_local.apply(extract_city_state_from_local)
    merged["cand_cidade"] = [t[0] for t in loc_parsed]
    merged["cand_uf"]     = [t[1] for t in loc_parsed]
    merged["cand_regiao"] = merged["cand_uf"].apply(uf_to_region)

    vaga_estado = merged.get("vaga_perfil_vaga_estado", merged.get("vaga_estado", None))
    vaga_cidade = merged.get("vaga_perfil_vaga_cidade", merged.get("vaga_cidade", None))

    def to_uf_from_state_or_uf(x):
        if not isinstance(x, str):
            return None
        x = x.strip()
        if len(x) == 2:
            return x.upper()
        return STATE_NAME_TO_UF.get(x, None)

    merged["vaga_uf"] = vaga_estado.apply(to_uf_from_state_or_uf) if isinstance(vaga_estado, pd.Series) else None
    merged["vaga_cidade_unif"] = vaga_cidade
    merged["vaga_regiao"] = merged["vaga_uf"].apply(uf_to_region)

    merged["same_state"]  = (merged["cand_uf"].notna()) & (merged["vaga_uf"].notna()) & (merged["cand_uf"].astype(str).str.upper() == merged["vaga_uf"].astype(str).str.upper())
    merged["same_city"]   = merged["same_state"] & (merged["cand_cidade"].astype(str).str.lower() == merged["vaga_cidade_unif"].astype(str).str.lower())
    merged["same_region"] = (merged["cand_regiao"].notna()) & (merged["vaga_regiao"].notna()) & (merged["cand_regiao"] == merged["vaga_regiao"])

    # Academic / Languages
    cand_acad = merged.get("cand_formacao_e_idiomas_nivel_academico", np.nan).map(academic_rank)
    vaga_acad = merged.get("vaga_perfil_vaga_nivel_academico", np.nan).map(academic_rank)
    merged["meets_academic"] = (cand_acad >= vaga_acad).fillna(False)

    cand_eng = merged.get("cand_formacao_e_idiomas_nivel_ingles", np.nan).map(lang_rank)
    vaga_eng = merged.get("vaga_perfil_vaga_nivel_ingles", np.nan).map(lang_rank)
    merged["meets_english"] = (cand_eng >= vaga_eng).fillna(False)

    cand_esp = merged.get("cand_formacao_e_idiomas_nivel_espanhol", np.nan).map(lang_rank)
    vaga_esp = merged.get("vaga_perfil_vaga_nivel_espanhol", np.nan).map(lang_rank)
    merged["meets_spanish"] = (cand_esp >= vaga_esp).fillna(False)

    # Text similarity & overlap
    def get_cand_text(row):
        parts = [
            row.get("cand_informacoes_profissionais_conhecimentos_tecnicos", ""),
            row.get("cand_informacoes_profissionais_certificacoes", ""),
            row.get("cand_informacoes_profissionais_outras_certificacoes", ""),
            row.get("cand_informacoes_profissionais_titulo_profissional", ""),
            row.get("cand_informacoes_profissionais_area_atuacao", ""),
            row.get("cand_cv_pt", ""),
        ]
        return " ".join([p for p in parts if isinstance(p, str)])

    def get_vaga_text(row):
        parts = [
            row.get("vaga_informacoes_basicas_titulo_vaga", "") or row.get("vaga_titulo_vaga", "") or row.get("prop_vaga_titulo", ""),
            row.get("vaga_perfil_vaga_principais_atividades", ""),
            row.get("vaga_perfil_vaga_competencia_tecnicas_e_comportamentais", ""),
            row.get("vaga_perfil_vaga_areas_atuacao", ""),
            row.get("vaga_perfil_vaga_demais_observacoes", ""),
        ]
        return " ".join([p for p in parts if isinstance(p, str)])

    merged["cand_text"] = merged.apply(get_cand_text, axis=1)
    merged["vaga_text"] = merged.apply(get_vaga_text, axis=1)

    merged["sim_tfidf"] = merged.apply(lambda r: tfidf_sim(r.get("cand_text",""), r.get("vaga_text","")), axis=1)
    ov = merged.apply(lambda r: overlap_and_jaccard(r.get("cand_text",""), r.get("vaga_text","")), axis=1)
    merged["overlap_kw"]  = [o[0] for o in ov]
    merged["jaccard_kw"]  = [o[1] for o in ov]

    # Remuneração & ratio
    merged["cand_remuneracao_num"] = merged.get("cand_informacoes_profissionais_remuneracao", "").apply(to_float_money)
    vaga_rem_cols = [c for c in merged.columns if c.startswith("vaga_") and ("remuneracao" in c.lower() or "salario" in c.lower())]
    if vaga_rem_cols:
        col = vaga_rem_cols[0]
        if merged[col].dtype == object:
            merged["vaga_remuneracao_num"] = merged[col].apply(to_float_money)
        else:
            merged["vaga_remuneracao_num"] = pd.to_numeric(merged[col], errors="coerce")
    else:
        merged["vaga_remuneracao_num"] = np.nan

    merged["ratio_remuneracao_cand_vaga"] = merged["cand_remuneracao_num"] / merged["vaga_remuneracao_num"]

    # Hiring type flags (vaga_)
    tipo_vaga = merged.get("vaga_informacoes_basicas_tipo_contratacao", merged.get("vaga_tipo_contratacao", ""))
    merged["vaga_is_CLT"]        = tipo_vaga.astype(str).str.contains("CLT", case=False, na=False).astype(int)
    merged["vaga_is_PJ"]         = tipo_vaga.astype(str).str.contains("PJ", case=False, na=False).astype(int)
    merged["vaga_is_Estagiario"] = tipo_vaga.astype(str).str.contains("Estagi", case=False, na=False).astype(int)
    merged["vaga_is_Cotas"]      = tipo_vaga.astype(str).str.contains("Cota", case=False, na=False).astype(int)

    # Seniority flags (vaga_ & cand_)
    def add_seniority_flags(series, prefix):
        j, p, s = [], [], []
        for v in series.astype(str):
            v = v or ""
            jj = int(("junior" in v.lower()) or ("júnior" in v.lower()))
            pp = int("pleno" in v.lower())
            ss = int(("senior" in v.lower()) or ("sênior" in v.lower()))
            j.append(jj); p.append(pp); s.append(ss)
        return pd.Series(j, name=f"{prefix}_is_Junior"), pd.Series(p, name=f"{prefix}_is_Pleno"), pd.Series(s, name=f"{prefix}_is_Senior")

    cand_lvl = merged.get("cand_informacoes_profissionais_nivel_profissional", "")
    vaga_lvl = merged.get("vaga_perfil_vaga_nivel_profissional", merged.get("vaga_nivel_profissional", ""))
    j1,p1,s1 = add_seniority_flags(cand_lvl, "cand")
    j2,p2,s2 = add_seniority_flags(vaga_lvl, "vaga")
    merged = pd.concat([merged, j1,p1,s1,j2,p2,s2], axis=1)

    # Final selection
    candidato_cols = [
        "cand_codigo_profissional",
        "cand_infos_basicas_local",
        "cand_infos_basicas_sabendo_de_nos_por",
        "cand_infos_basicas_codigo_profissional",
        "cand_informacoes_profissionais_titulo_profissional",
        "cand_informacoes_profissionais_area_atuacao",
        "cand_informacoes_profissionais_conhecimentos_tecnicos",
        "cand_informacoes_profissionais_certificacoes",
        "cand_informacoes_profissionais_outras_certificacoes",
        "cand_informacoes_profissionais_remuneracao",
        "cand_informacoes_profissionais_nivel_profissional",
        "cand_formacao_e_idiomas_nivel_academico",
        "cand_formacao_e_idiomas_nivel_ingles",
        "cand_formacao_e_idiomas_nivel_espanhol",
        "cand_formacao_e_idiomas_outro_idioma",
        "cand_formacao_e_idiomas_instituicao_ensino_superior",
        "cand_formacao_e_idiomas_cursos",
        "cand_formacao_e_idiomas_ano_conclusao",
        "cand_informacoes_profissionais_qualificacoes",
        "cand_informacoes_profissionais_experiencias",
        "cand_formacao_e_idiomas_outro_curso",
        "cand_cargo_atual_cargo_atual",
    ]
    candidato_cols = [c for c in candidato_cols if c in merged.columns]

    prospects_cols = ["prop_nome", "prop_codigo", "prop_situacao_candidato", "prop_vaga_id", "prop_vaga_titulo", "prop_vaga_modalidade"]
    prospects_cols = [c for c in prospects_cols if c in merged.columns]

    vagas_cols = [
        "vaga_vaga_id",
        "vaga_informacoes_basicas_data_requicisao",
        "vaga_informacoes_basicas_titulo_vaga",
        "vaga_informacoes_basicas_vaga_sap",
        "vaga_informacoes_basicas_tipo_contratacao",
        "vaga_informacoes_basicas_prazo_contratacao",
        "vaga_informacoes_basicas_objetivo_vaga",
        "vaga_informacoes_basicas_prioridade_vaga",
        "vaga_perfil_vaga_pais",
        "vaga_perfil_vaga_estado",
        "vaga_perfil_vaga_cidade",
        "vaga_perfil_vaga_bairro",
        "vaga_perfil_vaga_regiao",
        "vaga_perfil_vaga_local_trabalho",
        "vaga_perfil_vaga_vaga_especifica_para_pcd",
        "vaga_perfil_vaga_faixa_etaria",
        "vaga_perfil_vaga_horario_trabalho",
        "vaga_perfil_vaga_nivel_profissional",
        "vaga_perfil_vaga_nivel_academico",
        "vaga_perfil_vaga_nivel_ingles",
        "vaga_perfil_vaga_nivel_espanhol",
        "vaga_perfil_vaga_outro_idioma",
        "vaga_perfil_vaga_areas_atuacao",
        "vaga_perfil_vaga_principais_atividades",
        "vaga_perfil_vaga_competencia_tecnicas_e_comportamentais",
        "vaga_perfil_vaga_demais_observacoes",
        "vaga_perfil_vaga_viagens_requeridas",
        "vaga_perfil_vaga_equipamentos_necessarios",
        "vaga_beneficios_valor_venda",
        "vaga_beneficios_valor_compra_1",
        "vaga_beneficios_valor_compra_2",
        "vaga_informacoes_basicas_data_inicial",
        "vaga_informacoes_basicas_data_final",
        "vaga_perfil_vaga_habilidades_comportamentais_necessarias",
        "vaga_informacoes_basicas_nome_substituto",
        "vaga_titulo_vaga",
        "vaga_estado",
        "vaga_cidade",
        "vaga_tipo_contratacao",
        "vaga_analista_responsavel",
    ]
    vagas_cols = [c for c in vagas_cols if c in merged.columns]

    engineered = [
        "cand_cidade","cand_uf","cand_regiao","vaga_uf","vaga_cidade_unif","vaga_regiao",
        "same_state","same_city","same_region",
        "meets_academic","meets_english","meets_spanish",
        "sim_tfidf","overlap_kw","jaccard_kw",
        "cand_remuneracao_num","vaga_remuneracao_num","ratio_remuneracao_cand_vaga",
        "vaga_is_CLT","vaga_is_PJ","vaga_is_Estagiario","vaga_is_Cotas",
        "cand_is_Junior","cand_is_Pleno","cand_is_Senior",
        "vaga_is_Junior","vaga_is_Pleno","vaga_is_Senior",
        "label_contratado"
    ]

    final_cols = prospects_cols + vagas_cols + candidato_cols + engineered
    merged = merged[~merged.label_contratado.isna()].reset_index(drop=True)
    df_final = merged[final_cols].copy()
    return df_final

# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(description="Feature Engineering para vagas de tecnologia.")
    parser.add_argument("--applicants", type=str, default="/mnt/data/applicants_sample.json")
    parser.add_argument("--prospects",  type=str, default="/mnt/data/prospects_sample.json")
    parser.add_argument("--vagas",      type=str, default="/mnt/data/vagas_sample.json")
    parser.add_argument("--out-csv",    type=str, default="/mnt/data/df_final_engineered.csv")
    parser.add_argument("--out-parquet", type=str, default=None)
    args = parser.parse_args()

    df_final = build_df_final(Path(args.applicants), Path(args.prospects), Path(args.vagas))

    # Save
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_csv, index=False, encoding="utf-8")
    if args.out_parquet:
        try:
            df_final.to_parquet(args.out_parquet, index=False)
        except Exception as e:
            print("Aviso: falha ao salvar Parquet:", e)
    print(f"OK: df_final salvo em: {out_csv} (linhas={len(df_final)})")

if __name__ == "__main__":
    main()
