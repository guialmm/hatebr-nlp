"""
Análise exploratória do HateBR — roda antes de qualquer modelo pra entender
o dado: distribuição de classes, qualidade da anotação, tamanho dos textos
e se faz sentido separar treino/teste por conta (generalização) além do
split aleatório padrão.

Uso: python src/explore.py
"""

import pandas as pd

RAW = "data/raw/HateBR.csv"


def main():
    df = pd.read_csv(RAW)

    print(f"Total de comentários: {len(df)}")
    print(f"Distribuição de classes:\n{df['label_final'].value_counts()}\n")

    dup = df["comentario"].duplicated().sum()
    nulls = df.isnull().sum().sum()
    print(f"Duplicatas: {dup} | Valores nulos: {nulls}\n")

    lens = df["comentario"].astype(str).str.len()
    print(f"Tamanho do comentário (caracteres): média={lens.mean():.1f}, "
          f"mediana={lens.median():.0f}, min={lens.min()}, max={lens.max()}\n")

    # Quantos comentários tiveram concordância unânime entre os 3 anotadores —
    # mede o quão "óbvios" (vs. ambíguos) são os exemplos do dataset.
    unanimous = (df["anotator1"] == df["anotator2"]) & (df["anotator2"] == df["anotator3"])
    print(f"Concordância unânime entre anotadores: {unanimous.mean()*100:.1f}%\n")

    print("Comentários por conta/post de origem:")
    print(df["account_post"].value_counts())
    print(
        "\nComo os comentários vêm de só 6 contas, um split aleatório padrão "
        "deixa a mesma conta presente em treino e teste — o modelo pode "
        "aprender a 'reconhecer o autor do post' em vez do padrão de "
        "linguagem ofensiva em si. Por isso, além do split estratificado "
        "padrão, vale medir também a performance num split por conta "
        "(treina em 5 contas, testa na 6ª) — mostra se o modelo generaliza "
        "de verdade pra um contexto político novo."
    )


if __name__ == "__main__":
    main()
