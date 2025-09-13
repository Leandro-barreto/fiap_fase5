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
from sklearn.metrics import accuracy_score, f1_score

from src.data import feature_engineering as fe

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

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocessing
    preprocessor = fe.get_preprocessor(meta)

    # Model
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)

    # Build pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")

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