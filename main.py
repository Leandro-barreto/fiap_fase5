"""Command‑line entry point for training and inference.

This script provides a simple interface to train the hiring prediction
model or perform batch inference using a trained pipeline.  It wraps
functions from :mod:`src.models.train` and :mod:`src.models.infer` into
a command‑line application.  Users can invoke training, inference, or
both by supplying the corresponding flags.

Example
-------

Train a model using data in ``./data/raw`` and save it to ``./models/model.joblib``::

    python main.py --train --data-dir data/raw --model-output models/model.joblib

Perform inference on a CSV file of features using a saved model::

    python main.py --infer --model-output models/model.joblib --input-csv samples.csv

If both ``--train`` and ``--infer`` are provided, the script will first
train the model and then perform inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.train import train_model
from src.models.infer import load_pipeline, predict, predict_proba


def run_training(data_dir: Path, model_output: Optional[Path]) -> None:
    """Execute the training pipeline.

    Parameters
    ----------
    data_dir : Path
        Directory containing the raw JSON files used for training.
    model_output : Optional[Path]
        Path to persist the trained model.  If ``None``, the model is
        trained but not saved to disk.
    """
    train_model(data_dir, model_output=model_output)


def run_inference(model_path: Path, input_csv: Path) -> None:
    """Run batch inference on a CSV of feature rows.

    Parameters
    ----------
    model_path : Path
        Path to the saved model pipeline (``joblib`` file).
    input_csv : Path
        CSV file containing feature rows.  The column names must match
        those expected by the training pipeline (excluding identifier
        columns).
    """
    # Load the model pipeline
    model = load_pipeline(model_path)
    # Read features from CSV
    X = pd.read_csv(input_csv)
    # Predict labels and probabilities
    preds = predict(model, X)
    probs = predict_proba(model, X)
    # Output results to stdout
    for i, (label, prob) in enumerate(zip(preds, probs)):
        print(f"Row {i}: prediction={int(label)}, probability={prob:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Train and/or run inference with the hiring model")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run model training.",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run inference on a CSV of features.  Requires --model-output and --input-csv.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing applicants.json, prospects.json and vagas.json for training.",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="models/contratacao_model.joblib",
        help="Path to save or load the trained model.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="CSV file with feature rows for inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Determine output model path
    model_path: Optional[Path] = Path(args.model_output) if args.model_output else None
    # Run training if requested
    if args.train:
        if args.data_dir is None:
            raise SystemExit("--data-dir must be specified when training")
        run_training(Path(args.data_dir), model_path)
    # Run inference if requested
    if args.infer:
        if model_path is None or args.input_csv is None:
            raise SystemExit("--model-output and --input-csv must be specified for inference")
        run_inference(model_path, Path(args.input_csv))
    # If neither flag is provided, print help
    if not args.train and not args.infer:
        print("No action specified. Use --train and/or --infer. See --help for details.")


if __name__ == "__main__":  # pragma: no cover
    main()