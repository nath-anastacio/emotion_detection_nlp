[us English](README.md) | br Português

# Detecção de Emoções em Texto

Um projeto de machine learning que classifica textos em 6 emoções - **raiva, medo, alegria, amor, tristeza e surpresa** - usando pré-processamento de NLP e modelos clássicos de machine learning. Esse projeto inclui comparação de modelos, análise de erros e uma demo interativa em Streamlit.

**Demo:** []
**Dataset** [Kaggle - Emotion Detection on Text Dataset](https://www.kaggle.com/datasets/abhrajaiswal/emotions-detection-text-dataset)

---

## Visão Geral

O dataset contém aproximadamente 16000 amostras de texto em inglês, rotuladas em 6 categorias de emoção. O objetivo é construir um pipeline de classificação de texto - do texto bruto até uma demo funcional - documentando as decisões técnicas e trade-offs tomados ao longo do caminho.

| | |
|---|---|
| **Tarefa** | Classificação multi-classe de texto (6 emoções) |
| **Melhor modelo** | Linear SVM (features TF-IDF) |
| **Acurácia no teste** | 86% |
| **F1 (macro) no teste** | 0.83 |
| **Demo** | App Streamlit |

---

## Pipeline do projeto

1. **Carregamento dos dados:** leitura do dataset (~16000 amostras de texto rotuladas)
2. **Análise exploratória (EDA):** Distribuição de classes, tamanho dos textos, valores nulos e duplicatas
3. **Limpeza de texto:** Lowercase, remoção de URLs/menções/hashtags/pontuação, tokenização
4. **POS tagging & Lematização:** Classificação gramatical (POS tagging) *antes* da lematização, para evitar que o lematizador trate toda palavra como substantivo por padrão 
5. **Remoçao de stopwords:** Aplicada *depois* do POS tagging e da lematização, para preservar a estrutura/contexto da frase durante a classificação gramatical
6. **Codificação dos rótulos:** Conversão das emoções para formato numérico
7. **Divisão treino/validação/teste:** Split estratificado 70/15/15, preservando o balanceamento de classes nos três conjuntos
8. **Engenharia de features (TF-IDF):** Vetorização do texto limpo; ajustado apenas nos dados de treino, para evitar vazamento de dados
9. **Treinamento e comparação de modelos:** Logistic Regression, Linear SVM, Multinomial Naive Bayes e Random Forest
10. **Seleção do modelo:** Linear SVM escolhido com base no F1-macro e veocidade de treino
11. **Análise de erros:** Investigação das classificações incorretas, comparação entre as previsões do SVM e da Logistic Regressio, e inspeção dos coeficientes do modelo para explicar padrões específicos de erro
12. **Avaliação final:** Modelo selecionado avaliado uma única vez no conjunto de teste
13. **Exportação do modelo:** Modelo, vetorizador e label encoder salvos com `joblib`
14. **Demo interativa:** App Streamlit para predição de emoções em tempo real (`app.py`)

---

## Decisões Técnicas

### POS tagging antes da lematização

O `WordNetLemmatizer` do NLTK assume, por padrão, que toda palavra é um substantivo, o que gera lematizações incorretas em verbos e adjetivos. Para corrigir isso, a classificação gramaticall (`nltk.pos_tag`) é aplicada antes da lematização, mapeando cada palavra para sua categorial gramatical real.

### Stopwords removidas *depois* do POS tagging

O POS tagger depende do contexto da frase para esclarecer o papel gramatical de uma palavra. Remover as stopwords primeiro retiraria esse contexto (ex: perdendo palavras como "as", "is", "was", que ajudam a identificar se uma palavra é verbo ou substantivo), reduzindo a precisão da classificação. As stopwprds só são removidas como etapa final, depois que a lematização já foi concluída.

### Compatibilizando POS tagging e lematização

`pos_tag()` e `WordNetLemmatizer` vês de subsistemas diferentes dentro do NLTK e usam formatos de tags incompatíveis entre si: o `pos_tag()` retorna tags detalhadas no formato Penn Treebank (ex: `VBG`, `JJ`, `NNS`), enquanto o `WordNetLemmatizer` só reconhece quatro categorias genéricar (`VERB`, `NOUN`, `ADJ`, `ADV`). Uma função de mapeamento (`get_wordnet_pos`) converte entre os dois formatos, já que o NLTK não fornece essa tradução nativamente.

### TF-IDF ajustado apenas nos dados de treino

O vocabulário e os pesos IDF do vetorizador são aprendidos exclusivamente a partir de `X_train`. Os conjuntos de validação e teste apenas passam por *transform*, nunca são usados para ajustar o vetorizador - isso evita vazamento de dados e simula como o modelo encontraria texto genuinamente novo em produção.

### F1-macro ao invés de acurácia

O dataset é desbalanceado (`joy` e `sadness` são super-representados, `surprise` é sub-representada). A acurácia isolada mascararia um desempenho ruim nas classes minoritárias, então o F1-macro (que pondera todas as classes igualmente) foi usado como métrica principal de seleção de modelo, junto com `class_weight = 'balanced'` durante o treinamento.

---

## Comparação de Modelos

Quatro classificadores foram treinados com as mesmas features TF-IDF e avaliados no conjunto de validação:

| Modelo | Acurácia | F1 (macro) | F1 (weighted) | Tempo de treino |
|---|---|---|---|---|
| **Linear SVM** | **0.866** | **0.826** | **0.867** | 0.16s |
| Logistic Regression | 0.854 | 0.821 | 0.857 | ~0s |
| Random Forest | 0.833 | 0.799 | 0.834 | 4.40s |
| Multinomial Naive Bayes | 0.706 | 0.490 | 0.657 | 0.01s |

O **Linear SVM** foi escolhido: melhor F1-macro no conjunto de validação, tempo de treinamento desprezível e uma fronteira de decisão que se mostrou mais robusta a palavras-chave isoladas e enganosas (ver Análise de Erros abaixo). O Naive Bayes teve desempenho siginificativamente pior - sua suposição de independência entre features não se sustenta bem em linguagem natural e ele não tem suporte nativo a ponderação de classes, o que provavelmente prejudicou o desempenho nas classes minoritárias.

---

## resultados Finais (Conjunto de Teste)

Avalido uma única vez, após a seleção do modelo, no conjunto de teste (nunca tocado antes):

| Emoção | Precisão | Recall | F1-score | Suporte |
|---|---|---|---|---|
| anger | 0.88 | 0.83 | 0.85 | 324 |
| fear | 0.83 | 0.86 | 0.84 | 290 |
| joy | 0.91 | 0.86 | 0.88 | 804 |
| love | 0.68 | 0.78 | 0.72 | 196 |
| sadness | 0.90 | 0.92 | 0.91 | 700 |
| surprise | 0.73 | 0.79 | 0.76 | 86 |
| **Acurácia** | | | **0.86** | 2400 |
| **Média macro** | 0.82 | 0.84 | **0.83** | 2400 |

Os resultados no conjunto de teste ficaram muito próximos dos resultados de validação (F1-macro 0.83 vs 0.826), indicando que o modelo generaliza bem e não sofreu overfitting durante a etapa de seleção na validação.

---

## Análise de Erros

### Confusão entre Joy <-> Love e Fear <-> Surprise

As classificações incorretas mais comuns ocorrem entre emoções semanticamente próximas: `joy` é frequentemente confundida com `love`, e `fear` com `surprise`. Isso é esperado - o TF-IDF captura frequência de palavras isoladas, sem entendimento semântico ou contextual mais profundo, então emoções que compartilham vocabulário (ex: "amazing", "shocked", "loved") são mais difíceis de separar.

### SVM vs. Logistic Regression: Sensibilidade a palavras-chave

Comparando os casos em que o SVM acertou e a Logistic Regression errou, um padrão consistente emergiu: frases contendo uma palavra lexicalmente associada a uma emoção, mas cujo significado geral transmite outra (ex: "love" aparecendo em uma frase sobre reclamação, na verdade expressando `sadness`), confundiam mais a Logistic Regression do que o SVM. Isso sugere que a margem de decisão do SVM é mais robusta a esse tipo de ruído lexical, embora nenhum dos dois modelos capture o contexto real da frase.

### Estudo de caso: "amaze" dominando predições

Testes manuais fora do dataset revelaram uma limitação: a frase *"happy today, feel amaze"* foi classificada como `surprise`, apesar de conter três palavras associadas a `joy` (`happy`, `today`, `feel`) contra apenas uma associada a `surprise` (`amaze`). A inspeção dos coeficientes do SVM para a classe `surprise` confirmou a causa - palavras como `impressed` (peso 4.95), `curious` (4.94), `shock` (4.16) e `amaze` (4.08) têm peso desproporcionalmente alto para essa classe, permitindo que uma única palavra forte sobreponha múltiplos sinais mais fracos e contrários.

Isso é uma consequência direta da naturaza bag-of-words do TF-IDF: cada palavra contribui de forma independente para a predição, sem noção da frase como um todo. Arquiteturas com maior sensibilidade a contexto (ex: modelos baseados em transformers) tenderiam a lidar melhor com esse tipo de ambiguidade.

---

## Limitações

1. **Sem entendimento contextual:** o TF-IDF trata cada palavra de forma independente; negação e contexto em nível de frase não são capturados.
2. **Sensibilidade a palavras dominantes:** como demonstrado acima, uma única palavra de peso alto pode sobrepor o sentimento geral de uma frase
3. **Desbalanceamento de classes:** `surprise` (86 exemplos no teste) tem uma amostra menor e estatisticamente menos robusta do que `joy` ou `sadness` (700+ exemplos cada).
4. **Sem validação cruzada:** os resultados se baseiam em um único split estratificado; validação cruzada (k-fold) forneceria estimativas mais confiáveis estatisticamente, especialmente para classes minoritárias.
5. **Abordagem de ML clássico:** nenhum modelo de embeddings ou baseado em transformers foi utilizado, o que provavelmente limita o desempenho em linguagem mais ambígua.

---

## Demo

Um app interativo em Streamlit permite predição de emoções em tempo real a partir de texto inserido pelo usuário, usando o mesmo pipeline de limpeza e vetorização utilizado no treinamento.

**Experimente aqui:** []

Para rodar localmente:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Estrutura do projeto

```
emotion-detection/
├── data/
│   └── emotions.xlsx          # não versionado no git — ver seção Dataset
├── model/
│   ├── emotion_svm_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── notebooks/
│   └── emotion_detection.ipynb
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── README.pt-BR.md
```

---

## Dataset

Este projeto utiliza o [Emotion Detection Text Dataset](https://www.kaggle.com/datasets/abhrajaiswal/emotions-detection-text-dataset) do Kaggle (~16000 amostras rotuladas em 6 emoções).

Para reproduzir este projeto:
1. Baixe o dataset no Kaggle (ou use o trecho com `kagglehub` incluído no notebook)
2. Coloque o arquivo em `data/emotions.xlsx`
3. Execute o notebook em `notebooks/`

---
 
## Stack Técnica
 
- **Python**, **pandas**, **NumPy**
- **NLTK** — tokenização, POS tagging, lematização, stopwords
- **scikit-learn** — TF-IDF, treinamento e avaliação de modelos
- **matplotlib**, **seaborn** — visualização
- **Streamlit** — demo interativa
- **joblib** — persistência do modelo

---