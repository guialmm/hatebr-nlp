"""
Gera os splits de treino/validação/teste a partir de data/raw/HateBR.csv.

Dois splits, salvos em data/processed/:
1. split aleatório estratificado (padrão) — random_{train,val,test}.csv
2. split por conta (5 contas pro treino, 1 de fora pro teste) — account_{train,test}.csv,
   usado como teste extra de generalização (ver data/README.md)

Uso: python src/split_data.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW = "data/raw/HateBR.csv"
OUT = "data/processed"

RANDOM_STATE = 42
# Conta deixada de fora do treino no split de generalização — a de menor
# volume (834 comentários), pra não desperdiçar dado demais no teste.
HOLDOUT_ACCOUNT = "Gleisi Hoffmann"


def make_random_split(df: pd.DataFrame) -> None:
    train, temp = train_test_split(
        df, test_size=0.3, stratify=df["label_final"], random_state=RANDOM_STATE
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp["label_final"], random_state=RANDOM_STATE
    )
    train.to_csv(f"{OUT}/random_train.csv", index=False)
    val.to_csv(f"{OUT}/random_val.csv", index=False)
    test.to_csv(f"{OUT}/random_test.csv", index=False)
    print(f"split aleatório: train={len(train)}, val={len(val)}, test={len(test)}")


def make_account_split(df: pd.DataFrame) -> None:
    train = df[df["account_post"] != HOLDOUT_ACCOUNT]
    test = df[df["account_post"] == HOLDOUT_ACCOUNT]
    train.to_csv(f"{OUT}/account_train.csv", index=False)
    test.to_csv(f"{OUT}/account_test.csv", index=False)
    print(f"split por conta (holdout={HOLDOUT_ACCOUNT!r}): train={len(train)}, test={len(test)}")


def main():
    df = pd.read_csv(RAW)
    make_random_split(df)
    make_account_split(df)


if __name__ == "__main__":
    main()
