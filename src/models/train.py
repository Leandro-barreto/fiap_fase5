"""Training script for the hiring prediction model.

This module defines a command‑line interface to train a binary
classification model that predicts whether a candidate will be
contracted.  It uses the data preparation and feature engineering
functions from ``src/data/prepare_data`` and
``src/data/feature_engineering`` and trains a logistic regression
model within a scikit‑learn ``Pipeline``.  The pipeline consists of
preprocessing (scaling numeric features and one‑hot encoding
categorical features) followed by classification.  The default
model is ``LogisticRegression`` but can be swapped as needed.

The implementation is based on the earlier ``training_pipeline.py``
which computes features and trains a model【86947837380131†L140-L239】.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data import feature_engineering as fe
import pandas as pd  # for dtype checking and filling NaNs

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
    # After imputing, we update the metadata to ensure remunaracao_num is
    # treated as a categorical feature if it is no longer numeric.
    X_features = X_features.copy()
    # Fill domain‑specific categorical values
    if "nivel_ingles" in X_features.columns:
        X_features["nivel_ingles"] = X_features["nivel_ingles"].fillna("Nenhum")
    if "nivel_academico" in X_features.columns:
        X_features["nivel_academico"] = X_features["nivel_academico"].fillna("Não informado")
    if "remuneracao_num" in X_features.columns:
        # keep remuneracao_num numeric and fill missing values with -1 to avoid
        # mixed type errors in encoders【86947837380131†L140-L239】
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
    print(f"Registros com NaN restantes após imputação: {remaining_nans}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Optional oversampling of the minority class (hired candidates)
    # This step balances the classes by duplicating minority samples.  It can
    # improve recall and F1 when the positive class is underrepresented.
    ros = RandomOverSampler(sampling_strategy=0.2, random_state=random_state)
    X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
    # Show class distribution before and after oversampling
    print("Distribuição de classes (antes do oversampling):")
    print(y_train.value_counts())
    print("Distribuição de classes (após oversampling):")
    print(y_train_bal.value_counts())

    # Preprocessing
    preprocessor = fe.get_preprocessor(meta)

    # Model
    # Use class_weight="balanced" to handle class imbalance
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced")

    # Build pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    # Train using the oversampled data
    model.fit(X_train_bal, y_train_bal)

    # Evaluate
    y_pred = model.predict(X_test)
    # Predict probabilities for AUC-ROC
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Acurácia: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    # Save model if requested
    if model_output is not None:
        model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_output)
        print(f"Modelo salvo em {model_output}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Treina modelo de predição de contratação")
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Diretório contendo applicants.json, prospects.json e vagas.json"
    )
    parser.add_argument(
        "--model-output", type=str, default=None,
        help="Caminho para salvar o modelo treinado (opcional)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Proporção reservada para o conjunto de teste"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Semente aleatória para divisão de dados"
    )
    args = parser.parse_args()
    model_output = Path(args.model_output) if args.model_output else None
    train_model(Path(args.data_dir), model_output=model_output, test_size=args.test_size, random_state=args.random_state)


if __name__ == "__main__":  # pragma: no cover
    main()