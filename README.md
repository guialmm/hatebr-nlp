# HateBR NLP — Detecção de Discurso de Ódio em Português

> **Status: em desenvolvimento (Fase 1).** Este README documenta o escopo completo
> do projeto antes da implementação — serve de plano/roadmap por enquanto, e vai
> sendo atualizado com resultados reais conforme cada etapa é concluída.

Projeto de portfólio/estudo em NLP: fine-tuning de um transformer pré-treinado em
português (BERTimbau) para detectar discurso de ódio e linguagem ofensiva em
comentários reais, usando o [HateBR](https://github.com/franciellevargas/HateBR)
— corpus acadêmico com 7.000 comentários do Instagram brasileiro, anotados por
especialistas em 3 camadas simultâneas (ver [Dataset](#dataset) abaixo).

## Motivação

Meu portfólio até aqui é forte em visão computacional (DrowsyGuard, Eye Tracking,
Facial Landmark, VirtualCalculator) mas não tem nenhum projeto de NLP — e
fine-tuning de transformer é, junto com RAG, a habilidade mais citada em vagas de
ML/IA em 2026. Este projeto fecha essa lacuna com um problema real (moderação de
conteúdo), não um dataset de brinquedo.

## Dataset

**HateBR** ([Vargas et al., LREC 2022](https://arxiv.org/abs/2103.14972)) — corpus
de comentários reais do Instagram brasileiro direcionados a políticos, anotado por
3 especialistas por comentário (alta concordância entre anotadores). Três camadas
de rótulo no mesmo dataset:

1. **Binário**: ofensivo vs. não-ofensivo (3.500 / 3.500, balanceado)
2. **Nível de ofensividade**: levemente / moderadamente / altamente ofensivo
3. **Grupo-alvo** (multi-label): xenofobia, racismo, homofobia, machismo,
   intolerância religiosa, partidarismo, apologia à ditadura, antissemitismo,
   gordofobia

**Licença**: uso acadêmico/pesquisa apenas (sem uso comercial sem autorização por
escrito dos autores) — mesmo padrão do dataset da Olist no meu outro projeto: o
dataset **não é versionado neste repositório**, só o código (MIT). Instruções de
download em [`data/README.md`](data/README.md).

Fontes: [repositório oficial](https://github.com/franciellevargas/HateBR) ou
[Hugging Face (`ruanchaves/hatebr`)](https://huggingface.co/datasets/ruanchaves/hatebr).

> **Atualização (exploração inicial):** só a camada binária (ofensivo/
> não-ofensivo) está pública — as camadas de nível de ofensividade e
> grupo-alvo (as extensões multi-classe/multi-label da Fase 2) só existem na
> versão 1.0 do dataset, que exige pedir acesso direto à autora. A Fase 2 foi
> ajustada: ver [`data/README.md`](data/README.md) pra detalhes.

## Escopo completo

### Fase 1 — núcleo do projeto

- [x] Setup do repositório (estrutura, ambiente, dependências, licença)
- [x] Download + exploração do dataset (distribuição de classes, tamanho médio de
      texto, checagem de duplicatas/qualidade) — ver [`src/explore.py`](src/explore.py)
- [x] Pré-processamento: split treino/val/teste estratificado (70/15/15) +
      split por conta (5 contas treino, 1 de fora teste) — ver
      [`src/split_data.py`](src/split_data.py)
- [x] **Baseline clássico**: TF-IDF + Regressão Logística na tarefa binária —
      ver [`src/baseline.py`](src/baseline.py). Resultado: **F1 0.82** no
      split aleatório, **F1 0.77** no split por conta — a queda confirma que
      o split aleatório infla um pouco o número (o modelo aprende sinais
      específicos da conta, não só linguagem ofensiva em geral). É o número
      de referência que o BERTimbau precisa superar de verdade.
- [x] **Fine-tuning do BERTimbau** (`neuralmind/bert-base-portuguese-cased`),
      loop de treino manual em PyTorch (não `transformers.Trainer` — ver nota
      técnica abaixo) — ver [`src/finetune_bertimbau.py`](src/finetune_bertimbau.py).
      3 épocas, ~9-10 min por split em GPU (MPS, Apple Silicon).
- [x] Avaliação rigorosa: F1 por classe, precisão/recall, nos dois splits —
      não só acurácia.

**Resultado final — baseline vs. fine-tuning, nos dois splits:**

| Modelo | F1 macro (aleatório) | F1 macro (por conta) | queda |
|---|---|---|---|
| TF-IDF + Regressão Logística | 0.82 | 0.77 | -0.05 |
| **BERTimbau (fine-tuned)** | **0.92** | **0.90** | **-0.02** |

O fine-tuning não só supera o baseline nos dois splits por larga margem
(+10 e +13 pontos de F1), como **generaliza muito melhor** pra uma conta
política que nunca viu no treino — a queda de performance é 2.5x menor que a
do TF-IDF. Faz sentido: o BERTimbau entende semântica da frase, não só
vocabulário específico de uma conta.

Nota sobre o treino: o F1 de validação caiu levemente entre as épocas em
ambos os splits (época 1 foi a melhor ou quase) enquanto o loss de treino ia
a quase zero — sinal de leve overfitting a partir da época 2. O modelo final
(época 3) ainda assim generaliza bem no teste, mas early stopping na época 1
provavelmente teria um resultado igual ou melhor.
- [ ] API (FastAPI) servindo o modelo treinado + interface simples pra testar
      (cola um texto, recebe a classificação)
- [ ] Deploy público (mesmo padrão dos outros projetos: live demo + link no
      portfólio)
- [ ] README final com resultados reais, comparação baseline vs. transformer, e
      discussão de limitações

### Fase 2 — o que diferencia o projeto

- [ ] _(condicional — depende de acesso à versão 1.0, pedido à autora)_
      Extensão multi-classe (nível de ofensividade) e multi-label (9
      grupos-alvo)
- [ ] Fine-tuning completo vs. **LoRA/PEFT** — comparar custo computacional,
      tempo de treino e acurácia entre as duas abordagens
- [ ] Interpretabilidade: gerar explicação do modelo (LIME) e comparar contra
      os trechos que humanos de fato marcaram como justificativa
      (`HateBRXplain` tem essas anotações) — mais rigoroso que só eyeball
      qualitativo
- [ ] Discussão de viés/limitações: o dataset é de comentários em posts
      políticos — até que ponto o modelo generaliza pra outros contextos?

## Notas técnicas

**Por que loop de treino manual em vez de `transformers.Trainer`?** Nesta
máquina o Python do pyenv foi compilado sem suporte a `lzma`, o que quebra o
import da lib `datasets` — e `Trainer` importa `datasets` incondicionalmente,
mesmo sem usar nenhum recurso dela. Em vez de recompilar o Python do sistema,
escrevi o loop de treino direto com `torch.optim.AdamW` +
`get_linear_schedule_with_warmup` — o que também deixa explícito cada passo
(forward, backward, clip de gradiente, scheduler) em vez de esconder atrás de
`.fit()`.

## Stack (planejada)

- **Python**, **PyTorch**, **Hugging Face Transformers** (fine-tuning do
  BERTimbau)
- **scikit-learn** (baseline TF-IDF + Regressão Logística, métricas)
- **FastAPI** (API de inferência)
- **pandas** (pré-processamento e análise exploratória)
- Fase 2: **PEFT** (LoRA), **LIME** ou similar (interpretabilidade)

## Setup

_A preencher conforme a Fase 1 avança — ainda não há código._

## Licença

Código sob MIT (a adicionar). Dataset sob licença acadêmica/pesquisa do HateBR
(ver [Dataset](#dataset) acima) — não versionado neste repositório.
