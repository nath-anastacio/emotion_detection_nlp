US English | [BR Portuguese](README.pt-BR.md)

# Emotion Detection from Text

A machine learning projtect that classifies text into 6 emotions - **anger, fear, joy, love, sadness and surprise** - using NLP and classical machine learning models. Including model comparison, error analysis, and an interactive Streamlit demo.

**Live demo:** []
**Dataset:** [Kaggle - Emotion Detection Text Dataset](https://www.kaggle.com/datasets/abhrajaiswal/emotions-detection-text-dataset)

---

## Overview

The dataset contains ~16000 labeled English text samples across 6 emotion categories.
The goal is to build a text classification pipeline - from raw text to a working demo - while documenting the technical decisions and trede-ofs made along the way, not just the final metric.

| | |
|---|---|
| **Task** | Multi-class text classification (6 emotions) |
| **Best model** | Linear SVM (TF-IDF features) |
| **Test accuracy** | 86% |
| **Test F1 (macro)** | 0.83 |
| **Demo** | Streamlit app |

---

## Project Pipeline 

1. **Data Loading:** Load the dataset (~1600 labeled text samples) from kaggle;
2. **Exploratory Data Analysis (EDA):** Inspect class distribuition, text lenght, missing values and duplicates;
3. **Text Cleaning:** Lowercase, remove URLs/mentions/hashtags/punctuation, tokenize;
4. **POS_Tagging & Lemmatization:** Tag part-of-speech before lemmatizing to improve accuracy;
5. **Stopword Removal:** Applied after lemmatization to preserve sentence context during POS tagging;
6. **Label Encoding:** Convert emotion labels to numeric format;
7. **Train/Validation/Test Split:** Stratified 70/15/15 split to preserve class balance;
8. **Feature Engineering (TF-IDF):** Vectorize cleaned text, fit only on training data to avoid data leakage;
9. **Model Training and Comparison:** Train and compare Logistic Regression, Linear SVM, Naive Bayes and Random Forest;
10. **Model Selection:** Choose Linear SVM based on F1 macro score (best performance/speed trade-off);
11. **Error Analysis:** Investigate missclassifications, compare SVM vs Logistic Regression predicions, inspect model coefficients to undestand specific failure patterns;
12. **Final Evaluation:** Evaluate the selected model once on the held-out test set;
13. **Model Export:** Save the trained model, vectorizer and label encoder for deployment;
14. **Interactive Demo:** Streamlit app for real-time emotion prediction ('app.py').

---

## Technical Decisions

### POS tagging before lemmatization

During the preprocessing, it was observed that the NLTKs WordNetLemmatizer assumes every word is a noun by default, which produces incorect lemmas for verbs and adjectives. To address this, I added a POS tagging step (`nltk.pos_tag`) prior to lemmatization, mapping each word to its actual part-of-speech (noun, verb, adjective or adverb). This change increases the pipeline's computational cost, as the POS tagger must process each token individually, but the gain in text quality makes it worthwhile. Since the dataset is of moderate size, preprocessing runs only once, and the extra time does not pose a real constraint for the project.

### Stop words removed *after* POS tagging

The POS tagger relies on sentence context to disambiguate a word's grammatical role. Removing stopwords first would strip that context (e.g. losing words like 'as', 'is', 'was' that help identify wether a word is a verb or noun), reducing tagging accuracy. Stopwords are only removed as the final step, after lemmatization is already complete.

### Bridging POS tagging and lemmatization

`pos_tag()`and `WordNetLemmatizer` come from different sub-systems withing NLTK and use incompatible tag formats: `pos_tag` returns detailed Penn Treebank tags (e.g. `VBG`, `JJ`, `NNS`), while `WordNetLemmatizer` only recognizes four generic categories (`VERB`, `NOUN`, `ADJ`, `ADV`). A mapping funtion (`get_wordnet_pos`) converts between the two, since NLTK does not provide this translation natively.

### TF-IDF is fit only on training data

The vectorizer's vocabulary and IDF weights are learned exclusively from `X_train`. Validation and test sets are only *transformed*, never used to fit the vectorizer - this avoids data leakage and simulates how the model would encounter genuinely unseen text in production. 

### F1-macro over accuracy

The dataset is imbalanced (`joy` and `sadness` are overrepresented, `surprise` is underrepresented). Accuracy alone would mask poor performance on minority classes, so F1-macro (which weights all classes equally) was used as the primary model-selection metric, alongside `class_weight = 'balanced'` during training.

---

## Model comparison

Four classifiers were trained on the same TF-IDF features and evaluated on the validation set:

| Model | Accuracy | F1 (macro) | F1 (weighted) | Train time |
|---|---|---|---|---|
| **Linear SVM** | **0.866** | **0.826** | **0.867** | 0.16s |
| Logistic Regression | 0.854 | 0.821 | 0.857 | ~0s |
| Random Forest | 0.833 | 0.799 | 0.834 | 4.40s |
| Multinomial Naive Bayes | 0.706 | 0.490 | 0.657 | 0.01s |

**Linear SVM** was selected: best F1-macro on the validation set, negligible training time, and a decision boundary that proved more robust to misleading individual keywords (see error analysis below). Naive Bayes underperformed significantly - its independence assumption between features doesn't hold well for natural language, and it lacks native class-weighting support, wich likely hurt performance on minority classes.

---

## Final Results (Test Set)

Evaluated once, after model selection, on the held-out test set:

| Emotion | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| anger | 0.88 | 0.83 | 0.85 | 324 |
| fear | 0.83 | 0.86 | 0.84 | 290 |
| joy | 0.91 | 0.86 | 0.88 | 804 |
| love | 0.68 | 0.78 | 0.72 | 196 |
| sadness | 0.90 | 0.92 | 0.91 | 700 |
| surprise | 0.73 | 0.79 | 0.76 | 86 |
| **Accuracy** | | | **0.86** | 2400 |
| **Macro avg** | 0.82 | 0.84 | **0.83** | 2400 |

Test set results closely match validation set results (F1-macro 0.83 vs 0.826), indicating the model generalizes well and wasn't overfit to the validation set during model selection.

---

## Error Analysis

### Joy <-> Love and Fear <-> Surprise confusion

The most comon misclassifications occur between semmantically related emotions: `joy` is frequently confused with `love` and `fear` with `surprise`. This is expected - TF-IDF captures word frequency in isolation, without deeper semantic or contextual understanding, so emotions that share vocabulary (e.g. "amazing", "shocked", "loved") are harder to separate. 

### SVM vs Logistic Regression: keyword sensitivity

Comparing cases where SVM was correct and Logistic Regression was wrong revealed a consistent pattern: sentences containing a word lexically associated with one emotion, but whose overall meaning conveys another (e.g. "love" appearing in a sentence about complaining, actually expressing `sadness`), confused Logistic Regression more than SVM. This suggests SVM's decision margin is more robust to this kind of lexical noise, though neither model captures true sentence-level context.

### Case study: "amaze" dominating predictions

Manual testing with out-of-dataset sentences uncovered a concreate limitation: the sentence *"happy today, feel amaze* was classified as `surprise`, despite containing three words associated with `joy` (`happy`, `today`, `feel`) against only one associated with `surprise` (`amaze`). Inspecting the SVM's coefficients for the `surprise` class confirmed the cause - words like `impressed` (weight 4.95), `curious` (4.94), `shock` (4.16), and `amaze` (4.08) carry disproportionately high weight for that class, allowing a single strong word to override multiple weaker, contrary signals.

This is a direct consequence of the bag-of-words nature of TF-IDF: each word contributes independently to the prediction, with no awareness of the sentence as a whole. Context-aware architectures (e.g. transformer-based models would be expectec to handle this kind of ambiguity better).

## Limitations

1. **No contextual understanding:** TF-IDF treats each word independently; negation ("I don't love this") and sentence-level context are not captured.
2. **Sensitive to dominant keywords:** as shown above, a single high-weight word can override the overall sentiment of a sentence.
3. **Class imbalance:** `surprise` (86 text examples) has a smaller, less statistically robust sample than `joy` or `sadness` (700+ examples each).
4. **No cross-validation:** results are based on a single stratified split; k-fold cross-validation would provide more statistically reliable estimates, especially for minority classes.
5. **Classical ML approach:** no embeddings or transformer-based models were used, which likely limits performance on nuanced or ambiguous language.

## Demo

An interactive Streamlit app allows real-time emotion from user-provided text, using the same cleaning and vectorization pipeline as the training process.

**Try it here:**[]

To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
emotion-detection/
├── data/
│   └── emotions.xlsx          # not tracked in git — see Dataset section
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

## Dataset

This project uses the [Emotion Detection Dataset](https://www.kaggle.com/datasets/abhrajaiswal/emotions-detection-text-dataset) from kaggle (~16000 labeled samples across 6 emotions)

To reproduce this project:
1. Download the dataset from Kaggle (or use the `kagglehub` snippet included in the notebook)
2. Place the file as `data/emotions.xlsx`
3. Run the notebook in `notebooks/`

---

## Tech Stack

1. **Python**, **pandas**, **Numpy**
2. **NLTK:** tokenization, POS tagging, lemmatizatioon, stopwords
3. **scikit-learn:** TF-IDF, model training and evaluation
4. **matplotlib, seaborn:** visualization
5. **Streamlit:** interactive model
6. **joblib:** model persistence

---


