"""
Baseline clássico: TF-IDF + Regressão Logística na tarefa binária
(ofensivo/não-ofensivo). Roda nos dois splits (aleatório e por conta) pra
estabelecer o número de referência antes do fine-tuning do BERTimbau.

Uso: python src/baseline.py   (requer data/processed/*, ver src/split_data.py)
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

PROCESSED = "data/processed"


def run_split(train_path: str, test_path: str, label: str) -> None:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train["comentario"].astype(str))
    X_test = vectorizer.transform(test["comentario"].astype(str))

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, train["label_final"])

    preds = clf.predict(X_test)
    print(f"\n=== Baseline TF-IDF + LogReg — split: {label} ===")
    print(classification_report(test["label_final"], preds, target_names=["não-ofensivo", "ofensivo"]))


def main():
    run_split(f"{PROCESSED}/random_train.csv", f"{PROCESSED}/random_test.csv", "aleatório (padrão)")
    run_split(f"{PROCESSED}/account_train.csv", f"{PROCESSED}/account_test.csv", "por conta (generalização)")


if __name__ == "__main__":
    main()
