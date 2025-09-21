"""
Script de inferência
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_shap(
    shap_values: np.ndarray,
    feature_names: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """Agrupa contribuições SHAP em features originais.

    Os valores de contribuição retornados por LightGBM via
    `pred_contrib=True` incluem uma coluna extra com o valor base.  Esta
    função descarta a última coluna, soma as contribuições das categorias
    one‑hot e devolve um DataFrame com uma coluna para cada variável
    original.
    """
    # descartar termo de base
    shap_vals = shap_values[:, :-1]
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    agg = pd.DataFrame()
    for col in num_cols:
        agg[col] = shap_df.get(f"num__{col}", 0.0)
    for col in cat_cols:
        prefix = f"cat__{col}_"
        cols = [c for c in shap_df.columns if c.startswith(prefix)]
        if cols:
            agg[col] = shap_df[cols].sum(axis=1)
        else:
            agg[col] = 0.0
    return agg


def plot_violin(shap_df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Gera e salva um gráfico violin das contribuições por feature."""
    data_long = shap_df.melt(var_name="feature", value_name="contribution")
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data_long, x="contribution", y="feature", orient="h", cut=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Retorna as probabilidades de contratação para cada amostra.

    `model` deve ser um pipeline treinado com LightGBM como classificador.
    A função retorna apenas a probabilidade da classe positiva (índice 1).
    """
    return model.predict_proba(X)[:, 1]

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


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Retorna a previsão binária (0 ou 1) de contratação.

    Usa um limiar fixo de 0,5 sobre a probabilidade de contratação.
    """
    proba = predict_proba(model, X)
    return (proba >= 0.5).astype(int)


def generate_shap(model, X: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """Calcula as contribuições SHAP aproximadas e retorna um DataFrame agregado.

    A transformação das features é feita utilizando o pré‑processador do pipeline.
    O DataFrame retornado tem uma coluna por feature original e uma linha por
    amostra, representando a contribuição daquela feature para a previsão.
    """
    booster = model.named_steps["classifier"].booster_
    X_trans = model.named_steps["preprocessor"].transform(X)
    shap_values = booster.predict(X_trans, pred_contrib=True)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    shap_df = aggregate_shap(shap_values, feature_names, num_cols, cat_cols)
    return shap_df


def run_inference(model_path: Path, input_csv: Path) -> None:
    """Executa a inferência em um conjunto de dados e grava resultados.

    Este método pode ser usado por outras aplicações (como um `main.py` ou
    endpoint da API) para obter as previsões e salvar o gráfico SHAP.
    """
    # Carregar modelo
    model = joblib.load(model_path)
    # Ler dados
    df = pd.read_csv(input_csv)
    # Definir colunas
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
        "regiao_candidato",
        "regiao_vaga",
    ]
    X = df[num_cols + cat_cols]
    # Previsões
    pred_proba = predict_proba(model, X)
    pred_class = predict(model, X)
    # Calcular SHAP
    shap_df = generate_shap(model, X, num_cols, cat_cols)

    # Salvar violin plot
    plot_violin(shap_df, "Contribuições por característica (inferência)", "models/assets/shap_violin.png")
    # Salvar previsões
    out_df = df.copy()
    out_df["pred_prob"] = pred_proba
    out_df["pred_class"] = pred_class
    out_df.to_csv("models/assets/inference_predictions.csv", index=False)
    # Imprimir prévia
    print("Pré-visualização das previsões (primeiras 5 linhas):")
    print(out_df[["pred_prob", "pred_class"]].head())
    print(f"Artefatos de inferência salvos")

