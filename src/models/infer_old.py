"""Inference utilities for the hiring prediction model.

This module exposes functions to load a previously trained model
(including its preprocessing pipeline) and to perform predictions on
new data.  Because the model pipeline contains both the preprocessing
and the classifier, input data should have the same feature columns
used during training (excluding identifier columns).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # para ambientes sem display
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_shap(
    shap_values: np.ndarray,
    feature_names: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """Agrupa contribuições SHAP como no treinamento."""
    shap_vals = shap_values[:, :-1]
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    agg = pd.DataFrame()
    for col in num_cols:
        feat_name = f"num__{col}"
        agg[col] = shap_df.get(feat_name, 0.0)
    for col in cat_cols:
        prefix = f"cat__{col}_"
        cols = [c for c in shap_df.columns if c.startswith(prefix)]
        if cols:
            agg[col] = shap_df[cols].sum(axis=1)
        else:
            agg[col] = 0.0
    return agg


def plot_violin(shap_df: pd.DataFrame, title: str, output_path: Path) -> None:
    data_long = shap_df.melt(var_name="feature", value_name="contribution")
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data_long, x="contribution", y="feature", orient="h", cut=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_pipeline(model_path: Union[str, Path]):
    """Load a trained pipeline from disk.

    Parameters
    ----------
    model_path : str or Path
        Path to the ``joblib`` file containing the trained pipeline.

    Returns
    -------
    Any
        The loaded scikit‑learn pipeline.
    """
    return joblib.load(model_path)


def generate_shap(model, X: pd.DataFrame):
    
    # Colunas de entrada (devem coincidir com o treinamento)
    num_cols = [
        "sim_tfidf",
        "overlap_kw",
        "remuneracao_num",
        "tempo_processamento",
        "cand_missing_ratio",
        "cand_text_len",
        "vaga_text_len",
        "same_state",
        "same_region",
    ]
    cat_cols = [
        "nivel_academico",
        "nivel_ingles",
        "tipo_contratacao",
        "estado",
        "cidade",
        "recrutador",
        "analista_responsavel",
        "regiao_vaga",
        "regiao_candidato"
    ]
    
    X_trans = model.named_steps["preprocessor"].transform(X)
    booster = model.named_steps["classifier"].booster_
    shap_values = booster.predict(X_trans, pred_contrib=True)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    shap_df = aggregate_shap(shap_values, feature_names, num_cols, cat_cols)
    plot_violin(shap_df, "Contribuições por característica (inferência)", "models/assets/shap_violin.png")


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Predict labels for a batch of samples using the loaded model.

    Parameters
    ----------
    model : Any
        Trained scikit‑learn pipeline containing preprocessing and classifier.
    X : pd.DataFrame
        DataFrame of features (excluding id columns).  Must contain the
        same numeric and categorical columns used in training.

    Returns
    -------
    np.ndarray
        Array of predicted labels (0 or 1).
    """
    # Gerar contribuições SHAP aproximadas
    generate_shap(model, X)
    return model.predict(X)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Return prediction probabilities for a batch of samples.

    This function returns the probability of the positive class (1).

    Parameters
    ----------
    model : Any
        Trained scikit‑learn pipeline.
    X : pd.DataFrame
        DataFrame of features matching the training schema.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) with probabilities of the positive class.
    """
    generate_shap(model, X)
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.ndim == 2 else proba