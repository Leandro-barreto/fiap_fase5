#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py
==========
- /predict/single  -> recebe JSON com `cand` e `vaga` (raw) e retorna
  probabilidade, label (0/1), features **engineered** usadas e top contribuições locais.

- /predict/batch   -> recebe um CSV (UploadFile) com colunas RAW prefixadas
  `cand_*` e `vaga_*`, gera as mesmas features do treino e retorna as probabilidades
  por linha, além dos top fatores locais. Opcionalmente devolve as features engineered.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.models import infer as infer_mod


router = APIRouter()

def _model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", "models/model_lgbm.joblib")).resolve()

def _ensure_model_exists(p: Path) -> None:
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Modelo não encontrado em: {p}")

def _to_label(prob: float, threshold: float = 0.5) -> int:
    return int(prob >= threshold)

@router.get("/health")
def health() -> Dict[str, Any]:
    p = _model_path()
    return {"status": "ok", "model_path": str(p), "exists": p.exists()}

@router.post("/predict/single")
async def predict_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_path = _model_path()
    _ensure_model_exists(model_path)

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="O corpo da requisição deve ser um objeto JSON.")
    cand = payload.get("cand")
    vaga = payload.get("vaga")
    top_k = int(payload.get("top_k", 10))

    if not isinstance(cand, dict) or not isinstance(vaga, dict):
        raise HTTPException(status_code=400, detail="JSON deve conter objetos 'cand' e 'vaga'.")

    try:
        result = infer_mod.predict_one(cand, vaga, model_or_path=str(model_path), top_k=top_k)
        prob = float(result["prob_contratado"])
        label = _to_label(prob)
        return {
            "label": label,
            "probability": prob,
            "features_engineered": result["X_engineered"],
            "top_local_contributions": result["local_contributions"],
            "global_importance": result["global_importance"],
        }
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Falha na predição: {exc}")

@router.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    include_engineered: bool = False,
    top_k: int = 5,
) -> Dict[str, Any]:
    model_path = _model_path()
    _ensure_model_exists(model_path)

    try:
        with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as tmp_in:
            content = await file.read()
            tmp_in.write(content)
            tmp_in_path = Path(tmp_in.name)

        with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as tmp_out:
            tmp_out_path = Path(tmp_out.name)

        df_out = infer_mod.predict_batch_from_csv(
            input_csv=tmp_in_path,
            model_or_path=str(model_path),
            output_csv=tmp_out_path,
            include_local=True,
            top_k=int(top_k),
        )

        rows: List[Dict[str, Any]] = []
        need_engineered = bool(include_engineered)

        for _, row in df_out.iterrows():
            item: Dict[str, Any] = {
                "probability": float(row["prob_contratado"]),
                "label": _to_label(float(row["prob_contratado"])),
            }
            if "top_features_json" in df_out.columns and "top_contribs_json" in df_out.columns:
                try:
                    item["top_features"] = json.loads(row["top_features_json"]) if pd.notna(row["top_features_json"]) else []
                    item["top_contribs"] = json.loads(row["top_contribs_json"]) if pd.notna(row["top_contribs_json"]) else []
                except Exception:
                    item["top_features"] = []
                    item["top_contribs"] = []

            if need_engineered:
                cand_raw = {k.replace("cand_", ""): row[k] for k in df_out.columns if k.startswith("cand_")}
                vaga_raw = {k.replace("vaga_", ""): row[k] for k in df_out.columns if k.startswith("vaga_")}
                try:
                    Xi = infer_mod.build_engineered_from_raw(cand_raw, vaga_raw)
                    item["engineered"] = Xi.iloc[0].to_dict()
                except Exception:
                    item["engineered"] = {}

            rows.append(item)

        return {"rows": rows, "count": len(rows)}

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Falha na predição em lote: {exc}")
    finally:
        try:
            if 'tmp_in_path' in locals() and Path(tmp_in_path).exists():
                Path(tmp_in_path).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if 'tmp_out_path' in locals() and Path(tmp_out_path).exists():
                Path(tmp_out_path).unlink(missing_ok=True)
        except Exception:
            pass
