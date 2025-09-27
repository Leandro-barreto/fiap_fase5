"""
Tests for the training module.

These tests exercise the publicly exposed helpers and entry point in the
training script.  Because the training pipeline itself can be
computationally intensive, the tests here use very small synthetic
datasets to drive the code paths and focus on expected behaviours such
as error handling, file creation and metric reporting.

The test suite dynamically imports the training module from either
``src.models.train`` or a top‑level ``train`` module, depending on
where the code lives.  This allows the tests to run both in the
repository layout shipped to the user and in a flat module layout that
might be used in development.
"""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Attempt to import the training module from the package structure first.
train_mod = importlib.import_module("src.models.train")  # type: ignore[import]



def test_plot_confusion_matrix_creates_file(tmp_path: Path) -> None:
    """plot_confusion_matrix should write a PNG to the provided path."""
    # Construct a simple confusion matrix for a binary classifier
    cm = np.array([[3, 1], [2, 4]])
    out_file = tmp_path / "cm.png"
    train_mod.plot_confusion_matrix(cm, classes=["0", "1"], output_path=out_file)
    # The file should exist and contain some bytes
    assert out_file.is_file(), "Confusion matrix file was not created"
    assert out_file.stat().st_size > 0, "Confusion matrix file is empty"


def test_plot_roc_curve_nan_when_single_class(tmp_path: Path) -> None:
    """plot_roc_curve should return NaN when all labels are the same."""
    # All true labels are 0; AUC cannot be computed
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])
    out_file = tmp_path / "roc.png"
    auc = train_mod.plot_roc_curve(y_true, y_score, out_file)
    assert np.isnan(auc), "AUC should be NaN when only one class is present"


def test_main_raises_when_target_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI should raise a ValueError if the input CSV lacks a 'target' column."""
    # Create a small dataset without the required 'target' column
    df = pd.DataFrame({
        'cand_cidade': {0: np.nan, 1: 'Rio de Janeiro'},
        'cand_uf': {0: np.nan, 1: 'RJ'},
        'cand_regiao': {0: np.nan, 1: 'Sudeste'},
        'vaga_uf': {0: 'RJ', 1: 'RJ'},
        'vaga_cidade_unif': {0: 'Rio de Janeiro', 1: 'Rio de Janeiro'},
        'vaga_regiao': {0: 'Sudeste', 1: 'Sudeste'},
        'same_state': {0: False, 1: True},
        'same_city': {0: False, 1: True},
        'same_region': {0: False, 1: True},
        'meets_academic': {0: False, 1: True},
        'meets_english': {0: False, 1: True},
        'meets_spanish': {0: False, 1: True},
        'sim_tfidf': {0: 0.4434254923760287, 1: 0.3664345600275163},
        'overlap_kw': {0: 7.0, 1: 6.0},
        'jaccard_kw': {0: 0.0191780821917808, 1: 0.02803738317757},
        'cand_remuneracao_num': {0: np.nan, 1: np.nan},
        'vaga_is_CLT': {0: 0.0, 1: 0.0},
        'vaga_is_PJ': {0: 1.0, 1: 1.0},
        'vaga_is_Estagiario': {0: 0.0, 1: 0.0},
        'vaga_is_Cotas': {0: 0.0, 1: 0.0},
        'cand_is_Junior': {0: 0.0, 1: 0.0},
        'cand_is_Pleno': {0: 0.0, 1: 0.0},
        'cand_is_Senior': {0: 0.0, 1: 0.0},
        'vaga_is_Junior': {0: 0.0, 1: 0.0},
        'vaga_is_Pleno': {0: 1.0, 1: 1.0},
        'vaga_is_Senior': {0: 0.0, 1: 0.0}
    })
    csv_path = tmp_path / "data_no_target.csv"
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / "out_no_target"
    # Temporarily override sys.argv for the CLI
    argv_backup = sys.argv.copy()
    sys.argv = ["train.py", "--input-csv", str(csv_path), "--output-dir", str(out_dir)]
    with pytest.raises(ValueError):
        train_mod.main()
    # Restore argv
    sys.argv = argv_backup


def test_main_trains_and_saves_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Running the CLI on a tiny dataset should produce model and metric files."""
    # Construct a minimal training dataset with both classes present
    df = pd.read_csv("artifacts/minimal_pd_train.csv")
    csv_path = tmp_path / "train_data.csv"
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / "out_train"
    # Prepare sys.argv
    argv_backup = sys.argv.copy()
    sys.argv = ["train.py", "--input-csv", str(csv_path), "--output-dir", str(out_dir)]
    # Execute the training CLI
    train_mod.main()
    # Restore argv
    sys.argv = argv_backup
    # The output directory should now contain the trained model and metrics
    model_file = out_dir / "trained_model.joblib"
    metrics_file = out_dir / "metrics.json"
    roc_file = out_dir / "roc_curve.png"
    cm_file = out_dir / "confusion_matrix.png"
    assert model_file.is_file(), "Model file not found"
    assert metrics_file.is_file(), "Metrics JSON not found"
    assert roc_file.is_file(), "ROC curve image not found"
    assert cm_file.is_file(), "Confusion matrix image not found"
    # Load and inspect metrics
    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    # Basic keys should be present
    for key in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]:
        assert key in metrics, f"Missing metric: {key}"