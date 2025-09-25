"""Pipeline de treinamento.

Este script treina um modelo LightGBM sobre um conjunto de dados tabular com variáveis
numéricas e categóricas. Ele imprime métricas de avaliação completas e gera gráficos
básicos como a curva ROC e a matriz de confusão. As métricas calculadas incluem
acurácia, precisão, recall, F1‑score e a área sob a curva ROC (AUC). A curva ROC e a
matriz de confusão são salvas como arquivos PNG no diretório de saída.

Para usar este script:

```
python train.py --input-csv caminho/para/dados_com_target.csv --output-dir saida/
```

Um arquivo `trained_model.joblib` será salvo no diretório de saída, bem como `metrics.json`,
`roc_curve.png` e `confusion_matrix.png`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

import matplotlib

# Use backend Agg for environments without display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


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


def train_model(df: pd.DataFrame, output_dir: Path) -> None:
    # Ler dados
    if "label_contratado" not in df.columns:
        raise ValueError("Arquivo de entrada deve conter uma coluna 'target'")
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_SPLITS = 3  # folds do GridSearchCV
    ENGINEERED = [
        "cand_cidade","cand_uf","cand_regiao","vaga_uf","vaga_cidade_unif","vaga_regiao",
        "same_state","same_city","same_region",
        "meets_academic","meets_english","meets_spanish",
        "sim_tfidf","overlap_kw","jaccard_kw",
        "cand_remuneracao_num",
        "vaga_is_CLT","vaga_is_PJ","vaga_is_Estagiario","vaga_is_Cotas",
        "cand_is_Junior","cand_is_Pleno","cand_is_Senior",
        "vaga_is_Junior","vaga_is_Pleno","vaga_is_Senior",
    ]
    TARGET = "label_contratado"

    CATEGORICAL = ["cand_cidade","cand_uf","cand_regiao","vaga_uf","vaga_cidade_unif","vaga_regiao"]
    NUMERIC = [c for c in ENGINEERED if c not in CATEGORICAL]

    missing = [c for c in ENGINEERED + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    df = df[~df.label_contratado.isna()].reset_index(drop=True)
    X = df[ENGINEERED].copy()
    y = df[TARGET].astype(int).copy()

    print("Formato X:", X.shape, "| Formato y:", y.shape)
    print("Distribuição de y:")
    print(y.value_counts(dropna=False))

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = n_neg / max(n_pos, 1)

    # Pré-processador: padronização para numéricas e one-hot para categóricas
    preproc_lr = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", Pipeline(steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False))
        ]), NUMERIC),
    ], remainder="drop", sparse_threshold=0.3)

    preproc_tree = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", SimpleImputer(strategy="median"), NUMERIC),
    ], remainder="drop", sparse_threshold=0.3)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    lgbm = LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        metric="auc",
        scale_pos_weight=pos_weight
    )
    pipe_lgbm = Pipeline(steps=[("pre", preproc_tree), ("clf", lgbm)])

    param_grid_lgbm = {
        "clf__n_estimators":[300, 600],
        "clf__learning_rate":[0.05, 0.1],
        "clf__num_leaves":[31, 63],
    }

    gs_lgbm = GridSearchCV(
        estimator=pipe_lgbm,
        param_grid=param_grid_lgbm,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    gs_lgbm.fit(X_train, y_train)
    print("Melhor params LGBM:", gs_lgbm.best_params_)
    print("Melhor CV AUC (LGBM):", gs_lgbm.best_score_)

    best_lgbm = gs_lgbm.best_estimator_

    # Previsões
    y_pred = best_lgbm.predict(X_test)
    y_proba = best_lgbm.predict_proba(X_test)[:, 1]

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # AUC e curva ROC
    auc_val = plot_roc_curve(y_test.values, y_proba, output_dir / "assets/roc_curve.png")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=["0", "1"], output_path=output_dir / "assets/confusion_matrix.png")

    # Impressão das métricas
    print("Métricas de teste:")
    print(f"  Acurácia: {acc:.4f}")
    print(f"  Precisão: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  AUC-ROC: {auc_val:.4f}" if not np.isnan(auc_val) else "  AUC-ROC: indefinido")
    print("  Matriz de confusão:")
    print(cm)

    # Preparar diretório de saída
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salvar modelo
    import joblib

    joblib.dump(best_lgbm, output_dir / "model_lgbm.joblib")

    # Salvar métricas em JSON
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": None if np.isnan(auc_val) else float(auc_val),
        "confusion_matrix": cm.tolist(),
    }
    with open(output_dir / "assets/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Modelo e artefatos salvos em {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar modelo com métricas e gráficos básicos")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Caminho para o arquivo CSV de treinamento (com coluna 'target')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Diretório de saída para modelo, métricas e gráficos",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    train_model(df, args.output_dir)


if __name__ == "__main__":
    main()