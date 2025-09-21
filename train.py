"""
Pipeline de treinamento atualizado.

Este script treina um modelo LightGBM sobre um conjunto de dados tabular com variáveis
numéricas e categóricas, imprime métricas de avaliação completas e gera gráficos
interpretáveis.  Em particular, além de acurácia e F1‑score, são calculados precisão,
recall e a área sob a curva ROC (AUC).  A curva ROC é plotada e salva em um arquivo
PNG.  A matriz de confusão também é exibida como um mapa de calor contendo tanto
contagens absolutas quanto proporções percentuais para cada célula.  As contribuições
de cada feature para o modelo são derivadas via `pred_contrib` do LightGBM e
agregadas para as variáveis originais, sendo depois apresentadas em um gráfico
violin.

Para usar este script:

```
python train.py --input-csv caminho/para/dados_com_target.csv --output-dir saida/
```

Um arquivo `trained_model.joblib` será salvo no diretório de saída, bem como
`metrics.json`, `shap_violin.png`, `roc_curve.png` e `confusion_matrix.png`.
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar modelo com gráficos e métricas completos")
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

    # Ler dados
    df = pd.read_csv(args.input_csv)
    if "target" not in df.columns:
        raise ValueError("Arquivo de entrada deve conter uma coluna 'target'")

    # Colunas numéricas e categóricas conforme projeto original
    num_cols = [
        "sim_tfidf",
        "overlap_kw",
        "remuneracao_num",
        "tempo_processamento",
        "cand_missing_ratio",
        "cand_text_len",
        "vaga_text_len",
    ]
    cat_cols = [
        "nivel_academico",
        "nivel_ingles",
        "tipo_contratacao",
        "estado",
        "cidade",
        "recrutador",
        "analista_responsavel",
    ]

    X = df[num_cols + cat_cols]
    y = df["target"]

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pré-processador: padronização para numéricas e one-hot para categóricas
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Classificador LightGBM
    clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
    )

    # Pipeline completo
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    # Treinar
    model.fit(X_train, y_train)

    # Previsões
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    # AUC e curva ROC
    auc_val = plot_roc_curve(y_test.values, y_proba, args.output_dir / "roc_curve.png")
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=["0", "1"], output_path=args.output_dir / "confusion_matrix.png")

    # Impressão das métricas
    print("Métricas de teste:")
    print(f"  Acurácia: {acc:.4f}")
    print(f"  Precisão: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  AUC-ROC: {auc_val:.4f}" if not np.isnan(auc_val) else "  AUC-ROC: indefinido")
    print("  Matriz de confusão:")
    print(cm)

    # Contribuições SHAP aproximadas
    booster = model.named_steps["classifier"].booster_
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    X_test_pre = model.named_steps["preprocessor"].transform(X_test)
    shap_values = booster.predict(X_test_pre, pred_contrib=True)
    shap_df = aggregate_shap(shap_values, feature_names, num_cols, cat_cols)

    # Preparar diretório de saída
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Salvar modelo
    import joblib
    joblib.dump(model, args.output_dir / "trained_model.joblib")

    # Salvar métricas em JSON
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": None if np.isnan(auc_val) else float(auc_val),
        "confusion_matrix": cm.tolist(),
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Salvar gráfico violin
    plot_violin(shap_df, "Contribuições por característica (teste)", args.output_dir / "shap_violin.png")

    print(f"Modelo e artefatos salvos em {args.output_dir}")


if __name__ == "__main__":
    main()