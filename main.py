#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
=======
Pipeline simplificado: prepare_data -> feature_engineering -> train -> (opcional) inferência.

Uso rápido:
- Treino completo:   python main.py --mode train
- Apenas predição:   python main.py --mode predict --inference-input data/external/synthetic_inference_data.csv
- Pipeline completo: python main.py --mode all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple
import pandas as pd

from src.data.prepare_data import load_json, flatten_applicants, flatten_prospects, flatten_vagas 
from src.data.feature_engineering import build_df_final
from src.models.train import train_model 
import src.models.infer as infer_mod  # type: ignore

def run_prepare_data(project_dir: Path) -> Tuple[Path, Path, Path]:
    raw_dir = project_dir / "data" / "raw"
    proc_dir = project_dir / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    app_json = raw_dir / "applicants.json"
    pro_json = raw_dir / "prospects.json"
    vag_json = raw_dir / "vagas.json"


    app_raw = load_json(app_json)
    pro_raw = load_json(pro_json)
    vag_raw = load_json(vag_json)

    df_app = flatten_applicants(app_raw) 
    df_pro = flatten_prospects(pro_raw)
    df_vag = flatten_vagas(vag_raw)

    print(df_app.shape, df_pro.shape, df_vag.shape)

    app_flat = proc_dir / "applicants_flat.json"
    pro_flat = proc_dir / "prospects_flat.json"
    vag_flat = proc_dir / "vagas_flat.json"

    # salvar em JSON (orient=records), como você mencionou
    if not df_app.empty: df_app.to_json(app_flat)
    if not df_pro.empty: df_pro.to_json(pro_flat)
    if not df_vag.empty: df_vag.to_json(vag_flat)

    print(f"[OK] Flats salvos em: {proc_dir}")
    return app_flat, pro_flat, vag_flat

def run_feature_engineering(project_dir: Path, app_json: Path, pro_json: Path, vag_json: Path) -> Path:
    df_final = build_df_final(app_json, pro_json, vag_json)
    out_csv = project_dir / "data" / "processed" / "df_final.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] df_final salvo em: {out_csv} (linhas={len(df_final)})")
    return df_final, out_csv

def run_predict(project_dir: Path, model_path: Path, input_csv: Path, output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    infer_mod.predict_batch_from_csv(input_csv, model_path, output_csv, include_local=True, top_k=5)
    print(f"[OK] Predições salvas em: {output_csv}")
    return output_csv

def main():
    parser = argparse.ArgumentParser(description="Pipeline simples: prepare_data -> feature_engineering -> train -> infer")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd(), help="Raiz do projeto")
    parser.add_argument("--mode", type=str, choices=["train", "predict", "all"], default="all",
                        help="Modo: train (prepara+FE+treino), predict (inferência), all (tudo).")
    parser.add_argument("--inference-input", type=Path, default=None,
                        help="CSV de entrada para inferência (default: data/external/synthetic_inference_data.csv)")

    args = parser.parse_args()

    project_dir = args.project_dir
    models_dir = project_dir / "models"
    assets_dir = models_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Caminhos padrão
    infer_input = args.inference_input or (project_dir / "data" / "external" / "synthetic_batch_input.csv")
    infer_output = assets_dir / "inference_predictions.csv"
    model_path = models_dir / "model_lgbm.joblib"

    if args.mode in ["train", "all"]:
        print("Etapa 1: Preparando os dados...")
        app_json, pro_json, vag_json = run_prepare_data(project_dir)

        print("Etapa 2: Feature engineering...")
        df_final_csv, _ = run_feature_engineering(project_dir, app_json, pro_json, vag_json)

        print("Etapa 3: Treinando o modelo...")
        train_model(df_final_csv, models_dir)

    if args.mode in ["predict", "all"]:
        print("Etapa 4: Fazendo inferência em lote...")
        if not model_path.exists():
            print(f"[ERRO] Modelo não encontrado em {model_path}. Rode: python main.py --mode train")
            raise SystemExit(2)
        run_predict(project_dir, model_path, infer_input, infer_output)

if __name__ == "__main__":
    main()
