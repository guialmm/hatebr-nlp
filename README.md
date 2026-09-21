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
download em [`data/README.md`](data/README.md) (a criar).

Fontes: [repositório oficial](https://github.com/franciellevargas/HateBR) ou
[Hugging Face (`ruanchaves/hatebr`)](https://huggingface.co/datasets/ruanchaves/hatebr).

## Escopo completo

### Fase 1 — núcleo do projeto

- [ ] Setup do repositório (estrutura, ambiente, dependências, licença)
- [ ] Download + exploração do dataset (distribuição de classes, tamanho médio de
      texto, checagem de duplicatas/qualidade)
- [ ] Pré-processamento (limpeza de texto, tokenização, split treino/val/teste
      estratificado)
- [ ] **Baseline clássico**: TF-IDF + Regressão Logística na tarefa binária —
      estabelece um número de referência antes do modelo pesado
- [ ] **Fine-tuning do BERTimbau** (`neuralmind/bert-base-portuguese-cased`) via
      Hugging Face Transformers, na tarefa binária (ofensivo/não-ofensivo)
- [ ] Avaliação rigorosa: F1 por classe, precisão/recall, matriz de confusão —
      não só acurácia (dataset tem nuance em "moderadamente ofensivo")
- [ ] API (FastAPI) servindo o modelo treinado + interface simples pra testar
      (cola um texto, recebe a classificação)
- [ ] Deploy público (mesmo padrão dos outros projetos: live demo + link no
      portfólio)
- [ ] README final com resultados reais, comparação baseline vs. transformer, e
      discussão de limitações

### Fase 2 — o que diferencia o projeto

- [ ] Extensão multi-classe: nível de ofensividade (leve/moderado/alto)
- [ ] Extensão multi-label: os 9 grupos-alvo simultaneamente
- [ ] Fine-tuning completo vs. **LoRA/PEFT** — comparar custo computacional,
      tempo de treino e acurácia entre as duas abordagens
- [ ] Interpretabilidade (LIME ou attention weights) — mostrar quais palavras
      pesaram na decisão do modelo pra cada predição
- [ ] Discussão de viés/limitações: o dataset é de comentários em posts
      políticos — até que ponto o modelo generaliza pra outros contextos?

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
