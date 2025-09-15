"""Prediction endpoints for the hiring API.

This module exposes a single endpoint capable of handling two types
of requests:

* **JSON body** – a user submits a JSON document describing a vacancy
  and a candidate.  The service constructs the same feature set used
  during training from the input fields (e.g. TF‑IDF similarity,
  keyword overlap, text lengths and categorical attributes) and
  returns the predicted class (0 = não aprovado, 1 = aprovado) along
  with the probability of contratação.

* **CSV upload** – a user uploads a CSV file containing one or more
  pre‑computed feature rows.  The file must contain the same
  columns used during model training (see the training pipeline for
  details).  The endpoint returns predictions and probabilities for
  each row.

By supporting both input formats the API can be used interactively
via the landing page form or programmatically by advanced users.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..model.loader import load_model

router = APIRouter()

# ---------------------------------------------------------------------------
# Helper functions

TECHS: List[str] = [
    "python",
    "java",
    "javascript",
    "typescript",
    "c#",
    "c++",
    "go",
    "rust",
    "ruby",
    "php",
    "scala",
    "sql",
    "nosql",
    "mysql",
    "postgres",
    "mongodb",
    "spark",
    "hadoop",
    "kafka",
    "airflow",
    "aws",
    "gcp",
    "azure",
    "docker",
    "kubernetes",
    "terraform",
    "pandas",
    "numpy",
    "sklearn",
    "pytorch",
    "tensorflow",
    "keras",
    "xgboost",
    "lightgbm",
    "fastapi",
    "flask",
    "django",
    "react",
    "vue",
    "angular",
    "excel",
    "sap",
    "power_bi",
    "sql_server",
]


def keyword_overlap(a: str, b: str, keywords: List[str] = TECHS) -> float:
    """Compute the keyword overlap between two texts.

    This helper replicates the behaviour used during training: it
    searches each keyword from ``keywords`` in both strings and
    returns the size of the intersection divided by the size of the
    union.  Texts are lower‑cased prior to matching.  If neither
    text contains any of the keywords the function returns zero.

    Parameters
    ----------
    a, b : str
        Input strings to compare.
    keywords : list of str
        List of keywords to match.

    Returns
    -------
    float
        Overlap ratio in the range [0, 1].
    """
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    a_lower, b_lower = a.lower(), b.lower()
    set_a = {kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", a_lower)}
    set_b = {kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", b_lower)}
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / max(1, len(union))


def parse_money(value: Any) -> float:
    """Parse a salary or monetary string into a float.

    Strings containing comma or dot separators are normalised to use
    ``.`` for the decimal point.  Non‑numeric input returns ``None``.

    Parameters
    ----------
    value : Any
        Value to parse.

    Returns
    -------
    float
        Parsed numeric value or ``None`` if not parseable.
    """
    if value is None:
        return None
    try:
        s = str(value)
        # remove any currency symbols and whitespace
        s = re.sub(r"[^0-9,.-]", "", s)
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def build_features_from_json(data: Dict[str, Any]) -> pd.DataFrame:
    """Construct a single‑row DataFrame of model features from user input.

    The keys of ``data`` correspond to the fields collected on the
    landing page (see ``static/home.html``).  Missing or empty values
    are imputed using the same conventions employed during training:

    * ``nivel_ingles`` – missing => ``"Nenhum"``
    * ``nivel_academico`` – missing => ``"Não informado"``
    * ``remuneracao_num`` – missing => ``-1``
    * other numeric features – missing => ``-1``
    * other categorical/textual features – missing => ``"N/A"``

    Parameters
    ----------
    data : dict
        A dictionary containing user‑submitted fields.

    Returns
    -------
    pandas.DataFrame
        A one‑row DataFrame with the same columns used in model training.
    """
    # extract candidate and vacancy free texts
    cand_text = str(data.get("cv", "") or "")
    job_title = str(data.get("job_title", "") or "")
    job_desc = str(data.get("job_description", "") or "")
    job_text = f"{job_title} {job_desc}".strip()
    # similarity via TF‑IDF
    sim = 0.0
    if cand_text or job_text:
        corpus = [cand_text, job_text]
        try:
            vec = TfidfVectorizer(min_df=1, max_features=10000)
            mat = vec.fit_transform(corpus)
            # compute cosine similarity between the two vectors
            sim = float(cosine_similarity(mat[0], mat[1])[0][0])
        except Exception:
            sim = 0.0
    # keyword overlap
    overlap = keyword_overlap(cand_text, job_text)
    # remuneration: expect candidate expectation or job salary
    remun_candidate = parse_money(data.get("expectativa_salario"))
    remun_job = parse_money(data.get("remuneracao"))
    remuneracao_num = remun_candidate if remun_candidate is not None else remun_job
    if remuneracao_num is None:
        remuneracao_num = -1.0
    # processing time is not available in the form; use sentinel
    tempo_processamento = -1.0
    # candidate missing ratio: proportion of missing candidate fields
    cand_fields = [cand_text, data.get("nivel_ingles"), data.get("nivel_academico")]
    missing_count = sum(1 for v in cand_fields if not (v and str(v).strip()))
    cand_missing_ratio = missing_count / len(cand_fields) if cand_fields else 0.0
    cand_text_len = len(cand_text)
    vaga_text_len = len(job_text)
    # categorical fields with default imputation
    nivel_academico = (data.get("nivel_academico") or "Não informado").strip()
    nivel_ingles = (data.get("nivel_ingles") or "Nenhum").strip()
    tipo_contratacao = (data.get("tipo_contratacao") or data.get("tipo") or "N/A").strip()
    estado = (data.get("estado") or "N/A").strip()
    cidade = (data.get("cidade") or "N/A").strip()
    recrutador = (data.get("recrutador") or "N/A").strip()
    analista = (data.get("analista_responsavel") or "N/A").strip()
    # assemble the feature dict with column names as in training
    features = {
        "sim_tfidf": [sim],
        "overlap_kw": [overlap],
        "remuneracao_num": [remuneracao_num],
        "tempo_processamento": [tempo_processamento],
        "cand_missing_ratio": [cand_missing_ratio],
        "cand_text_len": [cand_text_len],
        "vaga_text_len": [vaga_text_len],
        "nivel_academico": [nivel_academico],
        "nivel_ingles": [nivel_ingles],
        "tipo_contratacao": [tipo_contratacao],
        "estado": [estado],
        "cidade": [cidade],
        "recrutador": [recrutador],
        "analista_responsavel": [analista],
    }
    return pd.DataFrame(features)


@router.post("/predict/candidate")
async def predict_candidate(request: Request, file: UploadFile | None = File(None)) -> Dict[str, Any]:
    """Predict hiring outcome from JSON or CSV.

    This endpoint accepts either a JSON body describing a single candidate and
    vacancy or a CSV file containing multiple pre‑computed feature rows.  The
    model is loaded once per request and used to compute the predicted
    probabilities.  When called with JSON input, the response contains a
    single prediction and probability; when called with a CSV, the response
    includes lists of predictions and probabilities corresponding to each
    row.

    Parameters
    ----------
    request : Request
        Incoming HTTP request used to access the body if no file is provided.
    file : UploadFile, optional
        Optional CSV file containing feature rows.  If provided, the JSON
        body is ignored.

    Returns
    -------
    dict
        A dictionary with keys ``prediction``/``probability`` (for JSON) or
        ``predictions``/``probabilities`` (for CSV).
    """
    # Load the model pipeline
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # If a file was uploaded, process it as a batch prediction
    if file is not None:
        try:
            df = pd.read_csv(file.file)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Erro ao ler o arquivo CSV: {exc}")
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV não contém dados")
        try:
            preds = model.predict(df)
            probs = model.predict_proba(df)[:, 1]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Erro ao realizar a predição: {exc}")
        # include the feature rows for transparency
        features_data = df.to_dict(orient="records")
        return {
            "predictions": [int(p) for p in preds],
            "probabilities": [float(p) for p in probs],
            "features": features_data,
        }

    # Otherwise treat the body as JSON for a single prediction
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Requisição JSON inválida")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="O corpo JSON deve ser um objeto")
    try:
        features_df = build_features_from_json(payload)
        preds = model.predict(features_df)
        probs = model.predict_proba(features_df)[:, 1]
        # convert the single row to a dict for display
        features_dict: Dict[str, Any] = features_df.iloc[0].to_dict()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a entrada: {exc}")
    return {
        "prediction": int(preds[0]),
        "probability": float(probs[0]),
        "features": features_dict,
    }