# Dataset — HateBR

O dataset **não é versionado neste repositório** (licença acadêmica/pesquisa,
sem redistribuição comercial — ver [LICENSE](../LICENSE)). Baixe direto do
repositório oficial:

```bash
mkdir -p data/raw
curl -sL "https://raw.githubusercontent.com/franciellevargas/HateBR/main/dataset/HateBR.csv" -o data/raw/HateBR.csv
curl -sL "https://raw.githubusercontent.com/franciellevargas/HateBR/main/dataset/HateBRXplain.csv" -o data/raw/HateBRXplain.csv
```

## Arquivos

- **`HateBR.csv`** (7.000 linhas) — comentários do Instagram + rótulo binário
  final (`label_final`: 0 = não-ofensivo, 1 = ofensivo) e o voto individual dos
  3 anotadores especialistas (`anotator1/2/3`)
- **`HateBRXplain.csv`** (7.000 linhas) — mesmos comentários + rótulo binário +
  trechos de texto anotados por humanos justificando o rótulo ofensivo
  (`rationales_annotator1/2`) — usado na Fase 2 (interpretabilidade)

## Sobre as camadas extras (nível de ofensividade e grupo-alvo)

O paper original ([Vargas et al., 2022](https://aclanthology.org/2022.lrec-1.777))
descreve 3 camadas de anotação: binário, nível de ofensividade (leve/moderado/
alto) e grupo-alvo do discurso de ódio (9 categorias, multi-label). **Só a
camada binária está pública** (dataset HateBR 2.0, no repositório oficial). As
outras duas ficaram só na versão 1.0, que não é distribuída publicamente — é
preciso pedir acesso direto à autora (franciellealvargas@gmail.com).

Por isso o escopo multi-classe/multi-label da Fase 2 do projeto (ver README
principal) depende dessa resposta — se não vier a tempo, a Fase 2 foca em
comparação fine-tuning vs. LoRA e interpretabilidade via HateBRXplain, que já
estão garantidos com o dado público.

## Notas da exploração inicial

Ver [`src/explore.py`](../src/explore.py). Resumo: 7.000 comentários,
perfeitamente balanceado (3.500/3.500), sem duplicatas nem nulos, 81,2% de
concordância unânime entre os 3 anotadores. Vêm de só 6 contas/posts de
políticos de espectros opostos — o que motiva medir a performance também num
split por conta (generalização), não só no split aleatório padrão.
