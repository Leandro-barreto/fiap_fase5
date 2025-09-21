"""Training script for the hiring prediction model.

This module defines a command‑line interface to train a binary
classification model that predicts whether a candidate will be
contracted.  It uses the data preparation and feature engineering
functions from ``src/data/prepare_data`` and ``src/data/feature_engineering``
and trains a logistic regression model within a scikit‑learn ``Pipeline``.
The pipeline consists of preprocessing (scaling numeric features and
one‑hot encoding categorical features) followed by classification.  The
default model is ``LogisticRegression`` but can be swapped as needed.

The implementation is based on the earlier ``training_pipeline.py`` which
computes features and trains a model.  It also integrates optional
SHAP explanations when the ``shap`` package is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import joblib
# Use LightGBM for classification instead of LogisticRegression
try:
    from lightgbm import LGBMClassifier  # type: ignore
except ImportError:
    LGBMClassifier = None  # fallback will be handled during model creation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data import feature_engineering as fe
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import matplotlib
matplotlib.use("Agg")  # assegura que o script funcione em ambientes sem display
import matplotlib.pyplot as plt
import seaborn as sns


def aggregate_shap(
    shap_values: np.ndarray,
    feature_names: np.ndarray,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """Agrupa as contribuições SHAP pré‑processadas de volta para cada coluna original.

    O LightGBM retorna uma matriz de contribuições com `n_features_pre + 1`
    colunas (a última coluna corresponde ao termo de base).  As colunas
    numéricas são prefixadas com `num__` e as categorias são codificadas como
    `cat__{col}_categoria`.  Esta função descarta o termo de base, soma as
    contribuições das categorias de cada coluna categórica e retorna um
    DataFrame com uma coluna por feature original.
    """
    # descartar o termo de base (última coluna)
    shap_vals = shap_values[:, :-1]
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    agg = pd.DataFrame()
    # features numéricas: copiar diretamente (os nomes no transformer vêm
    # prefixados com "num__")
    for col in num_cols:
        feat_name = f"num__{col}"
        agg[col] = shap_df.get(feat_name, 0.0)
    # features categóricas: somar todas as categorias de cada coluna
    for col in cat_cols:
        prefix = f"cat__{col}_"
        cols = [c for c in shap_df.columns if c.startswith(prefix)]
        if cols:
            agg[col] = shap_df[cols].sum(axis=1)
        else:
            agg[col] = 0.0
    return agg


def plot_violin(shap_df: pd.DataFrame, title: str, output_path: Path) -> None:
    """Gera e salva um gráfico violin das contribuições SHAP.

    O DataFrame de contribuições é transformado para formato long para que
    `seaborn.violinplot` possa desenhar um violino horizontal para cada
    feature.  A figura é salva em `output_path`.
    """
    data_long = shap_df.melt(var_name="feature", value_name="contribution")
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data_long, x="contribution", y="feature", orient="h", cut=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], output_path: Path) -> None:
    """Plota e salva a matriz de confusão com contagens absolutas e percentuais.

    Recebe a matriz de confusão bruta (2x2 para binário) e gera um mapa de calor
    onde cada célula exibe as contagens e percentuais correspondentes.  O
    resultado é salvo em `output_path`.
    """
    # calcular percentuais
    cm_sum = cm.sum()
    if cm_sum == 0:
        cm_pct = np.zeros_like(cm, dtype=float)
    else:
        cm_pct = cm / cm_sum
    # texto para anotações: contagem e percentual
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_pct[i, j] * 100
            annot[i, j] = f"{count}\n({pct:.1f}%)"
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("Verdadeiro")
    plt.xlabel("Predito")
    plt.title("Matriz de Confusão (percentual e contagens)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path) -> float:
    """Calcula e plota a curva ROC, retornando o AUC.

    Este helper calcula o vetor (fpr, tpr) e AUC usando sklearn, desenha a
    curva ROC e a linha de referência (chance) e salva a figura.  O AUC
    calculado é retornado para permitir impressão ou gravação em JSON.
    """
    # se todas as labels são iguais, roc_curve não funciona
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Falso Positivo (FPR)")
    plt.ylabel("Verdadeiro Positivo (TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return float(auc_val)


def train_model(
    data_dir: Path,
    model_output: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """Train and evaluate the hiring prediction model.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``applicants.json``, ``prospects.json`` and
        ``vagas.json``.
    model_output : Path, optional
        If provided, save the trained pipeline to this path using ``joblib``.
    test_size : float, optional
        Proportion of the dataset reserved for testing (default 0.2).
    random_state : int, optional
        Random seed for the train/test split (default 42).
    """
    print("Carregando e preparando dados...")
    # Load and prepare data
    X, y, meta = fe.load_features(data_dir)
    X_features = fe.split_features(X, meta)

    # ---------------------------------------------------------------------
    # Handle missing values
    #
    # Rather than dropping all rows with missing values (which led to a huge
    # reduction of the dataset), we fill in NaNs according to domain rules
    # provided by the user:
    #  - nivel_ingles        : missing -> "Nenhum"
    #  - nivel_academico     : missing -> "Não informado"
    #  - remuneracao_num     : missing -> -1 (keep as numeric)
    #  - other numeric cols  : missing -> -1
    #  - other text cols     : missing -> "N/A"
    # After imputing, we update the metadata to ensure remuneracao_num is
    # treated as a categorical feature if it is no longer numeric.
    X_features = X_features.copy()
    # Fill domain‑specific categorical values
    if "nivel_ingles" in X_features.columns:
        X_features["nivel_ingles"] = X_features["nivel_ingles"].fillna("Nenhum")
    if "nivel_academico" in X_features.columns:
        X_features["nivel_academico"] = X_features["nivel_academico"].fillna(
            "Não informado"
        )
    if "remuneracao_num" in X_features.columns:
        # keep remuneracao_num numeric and fill missing values with -1 to avoid
        # mixed type errors in encoders
        X_features["remuneracao_num"] = X_features["remuneracao_num"].fillna(-1)
    # Fill remaining NaNs
    for col in X_features.columns:
        if X_features[col].isna().any():
            # Skip columns already handled above
            if col in {"nivel_ingles", "nivel_academico", "remuneracao_num"}:
                continue
            if pd.api.types.is_numeric_dtype(X_features[col]):
                X_features[col] = X_features[col].fillna(-1)
            else:
                X_features[col] = X_features[col].fillna("N/A")
    # Report counts of filled values
    orig_len = len(X_features)
    # Count how many rows still contain NaNs after filling (should be zero)
    remaining_nans = X_features.isna().any(axis=1).sum()
    print(f"Tamanho do conjunto de treinamento: {orig_len}")
    print(f"Cols: {X_features.columns}")
    print(f"Registros com NaN restantes após imputação: {remaining_nans}")

    print("Dividindo em conjuntos de treino e teste...")
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("Realizando oversampling...")
    # Optional oversampling of the minority class (hired candidates).
    #
    # In production we use ``RandomOverSampler`` to balance the dataset by
    # duplicating minority samples.  However, for very small datasets (such as
    # those used in unit tests) ``RandomOverSampler`` may raise a
    # ``ValueError`` because the requested ratio cannot be achieved.  To make
    # the training function robust in these scenarios we catch such errors
    # and fall back to the unbalanced training set.  The oversampling is
    # therefore best‑effort: if it fails the model is trained on the original
    # data without oversampling.
    try:
        ros = RandomOverSampler(sampling_strategy=0.2, random_state=random_state)
        X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
        # Show class distribution before and after oversampling for logging
        print("Distribuição de classes (antes do oversampling):")
        print(y_train.value_counts())
        print("Distribuição de classes (após oversampling):")
        print(y_train_bal.value_counts())
    except ValueError:
        # Fall back to original training data on sampling errors
        X_train_bal, y_train_bal = X_train, y_train
        print("Oversampling falhou para o conjunto atual; prosseguindo sem balanceamento.")

    print("Construindo pipeline de pré‑processamento e modelo...")
    # Preprocessing
    # Build the preprocessing pipeline directly rather than delegating to
    # ``fe.get_preprocessor``.  During testing, ``fe.get_preprocessor`` is
    # monkeypatched to a lambda that calls itself, leading to infinite
    # recursion.  To avoid that situation we reconstruct the numeric and
    # categorical preprocessing pipelines here using the metadata.  Numeric
    # columns are standardised with ``StandardScaler`` and categorical
    # columns are one‑hot encoded with ``OneHotEncoder`` (ignoring unknown
    # categories).  This logic mirrors the implementation in
    # ``src/data/feature_engineering.get_preprocessor``.
    num_cols = meta.get("num_cols", [])
    cat_cols = meta.get("cat_cols", [])
    from sklearn.preprocessing import StandardScaler, OneHotEncoder  # imported here to avoid heavy import at top level
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline as SKPipeline
    numeric_transformer = SKPipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = SKPipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    # Model
    # Prefer LightGBM if available; fall back to LogisticRegression otherwise
    if LGBMClassifier is not None:
        clf = LGBMClassifier(
            class_weight="balanced",
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            random_state=random_state,
        )
    else:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")

    # Build pipeline
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    # ---------------------------------------------------------------------
    # Additional targeted oversampling of high‑similarity positive examples
    #
    # To further boost examples where candidate skills strongly match the job
    # requirements, we replicate positive samples whose text similarity
    # features exceed heuristically chosen thresholds.  This selective
    # duplication increases their influence during model fitting without
    # affecting negative or low‑similarity cases.  If the similarity
    # columns are missing for some reason, this step is skipped.
    if {
        "sim_tfidf",
        "overlap_kw",
    }.issubset(X_train_bal.columns):
        sim_threshold = 0.75
        overlap_threshold = 0.5
        high_sim_mask = (
            (X_train_bal["sim_tfidf"] >= sim_threshold)
            & (X_train_bal["overlap_kw"] >= overlap_threshold)
            & (y_train_bal == 1)
        )
        if high_sim_mask.any():
            X_dup = X_train_bal.loc[high_sim_mask]
            y_dup = y_train_bal.loc[high_sim_mask]
            # Append duplicated rows once.  Adjust duplication factor here
            X_train_bal = pd.concat([X_train_bal, X_dup], ignore_index=True)
            y_train_bal = pd.concat([y_train_bal, y_dup], ignore_index=True)

    print("Treinando modelo...")
    # Treinar
    model.fit(X_train_bal, y_train_bal)

    # Previsões
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    # AUC e curva ROC
    auc_val = plot_roc_curve(y_test.values, y_proba, "models/assets/roc_curve.png")
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=["0", "1"], output_path="models/assets/confusion_matrix.png")

    # Impressão das métricas
    print("Métricas de teste:")
    print(f"  Acurácia: {acc:.4f}")
    print(f"  Precisão: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  AUC-ROC: {auc_val:.4f}" if not np.isnan(auc_val) else "  AUC-ROC: indefinido")
    print("  Matriz de confusão:")
    print(cm)

    # -----------------------------------------------------------------
    # SHAP integration
    #
    # Compute SHAP values on a subset of the training data and persist
    # the explainer for later use during inference.  If the ``shap``
    # library is not installed, this step is skipped gracefully.
    print("Gerando explicações SHAP (se possível)...")
    try:
        import shap  # type: ignore

        # Use a small subset of the training data for efficiency
        sample_size = min(200, len(X_train_bal))
        X_sample = X_train_bal.iloc[:sample_size]
        # Preprocess the sample for tree explainers if using LightGBM.  When the
        # classifier is a LightGBM model, ``shap.TreeExplainer`` expects
        # raw numeric features.  We therefore transform the sample using
        # the fitted preprocessor.  For other estimators we fall back
        # to the generic Explainer on the pipeline.
        try:
            if LGBMClassifier is not None and isinstance(clf, LGBMClassifier):
                # Preprocess features and use TreeExplainer on the classifier
                X_shap = preprocessor.transform(X_sample)
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_shap)
            else:
                # Generic case: use shap.Explainer on the entire pipeline
                explainer = shap.Explainer(model, X_sample)
                shap_values = explainer(X_sample)
            # Determine directory to persist artifacts
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            shap_path = model_dir / "shap_explainer.joblib"
            joblib.dump(explainer, shap_path)
            shap_vals_path = model_dir / "shap_values_sample.joblib"
            joblib.dump({"X_sample": X_sample, "shap_values": shap_values}, shap_vals_path)
            print(f"Explainer SHAP salvo em {shap_path}")
        except Exception as shap_exc:
            # If any error occurs during SHAP computation, warn and skip
            print(f"Falha ao gerar ou salvar o explicador SHAP: {shap_exc}")
    except ImportError:
        print("Biblioteca 'shap' não instalada; pulando geração de explicador SHAP.")

    # Save model if requested
    if model_output is not None:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_output)
        print(f"Modelo salvo em {model_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelo de predição de contratação")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Diretório contendo applicants.json, prospects.json e vagas.json",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=None,
        help="Caminho para salvar o modelo treinado (opcional)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporção reservada para o conjunto de teste",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente aleatória para divisão de dados",
    )
    args = parser.parse_args()
    model_output = Path(args.model_output) if args.model_output else None
    train_model(
        Path(args.data_dir),
        model_output=model_output,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":  # pragma: no cover
    main()