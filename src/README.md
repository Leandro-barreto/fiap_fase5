# 🧠 src/ – Pré-processamento, Engenharia de Features e Treinamento

Este diretório contém o código responsável por preparar os dados, extrair características e treinar o modelo utilizado pela API.

## 📂 Estrutura
A modelagem segue os seguintes passos:
- **data/prepare_data** Prepara os arquivos json para um formato mais amigável para gerar as features 
- **data/feature_engineering** – Faz as manipulações para a criação das features que serão usadas no modelo e também a escolha dos labels
- **models/train** – Treinamento do modelo e impressão e salvamento das métricas
- **models/infer** – Realiza as predições;


## 📌 TF-IDF e Similaridades de Texto

- **TF-IDF** (*Term Frequency–Inverse Document Frequency*) mede a importância de uma palavra dentro de um documento em relação a uma coleção de documentos.  
- **overlap_kw (Coeficiente de Sobreposição)** – proporção de palavras em comum em relação ao menor conjunto.  
- **jaccard_kw (Índice de Jaccard)** – compara interseção sobre união de conjuntos de palavras.

## 🔍 Conjunto de Features

As features finais utilizadas foram:

```
['cand_cidade', 'cand_uf', 'cand_regiao', 'vaga_uf', 'vaga_cidade_unif',
 'vaga_regiao', 'same_state', 'same_city', 'same_region',
 'meets_academic', 'meets_english', 'meets_spanish', 'sim_tfidf',
 'overlap_kw', 'jaccard_kw', 'cand_remuneracao_num', 'vaga_is_CLT',
 'vaga_is_PJ', 'vaga_is_Estagiario', 'vaga_is_Cotas', 'cand_is_Junior',
 'cand_is_Pleno', 'cand_is_Senior', 'vaga_is_Junior', 'vaga_is_Pleno',
 'vaga_is_Senior']
```

## 🌳 Modelo: LGBMClassifier

Foi utilizado o **LightGBM (LGBMClassifier)**, um algoritmo baseado em árvores de decisão com boosting gradiente. É altamente eficiente e escalável para dados estruturados.

## 📈 Métricas de Desempenho

Do arquivo `models/assets/metrics.json`:

- Accuracy: 0.694  
- Precision: 0.451  
- Recall: 0.541  
- F1 Score: 0.492  
- AUC: 0.708  

Matriz de confusão:

```
[[1230, 406],
 [ 283, 334]]
```

Curva ROC (AUC=0.708):

![Curva ROC](https://raw.githubusercontent.com/Leandro-barreto/fiap_fase5/main/models/assets/roc_curve.png)

## Saídas esperadas
- `models/lgbm_model.joblib`
- métricas e gráficos
