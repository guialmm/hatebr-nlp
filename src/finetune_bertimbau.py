"""
Fine-tuning do BERTimbau (neuralmind/bert-base-portuguese-cased) pra
classificação binária ofensivo/não-ofensivo.

Loop de treino manual em PyTorch (não usa transformers.Trainer — nesta
máquina a lib `datasets`, dependência do Trainer, não importa por falta de
suporte a lzma no Python do pyenv; e escrever o loop na mão também deixa
explícito o que está acontecendo em cada passo).

Roda nos dois splits (aleatório e por conta) igual o baseline, pra comparar
diretamente contra o TF-IDF + Regressão Logística (ver src/baseline.py).

Uso: python src/finetune_bertimbau.py [--split random|account] [--epochs 3]
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
PROCESSED = "data/processed"


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_split(split: str):
    if split == "random":
        train = pd.read_csv(f"{PROCESSED}/random_train.csv")
        val = pd.read_csv(f"{PROCESSED}/random_val.csv")
        test = pd.read_csv(f"{PROCESSED}/random_test.csv")
    else:
        train = pd.read_csv(f"{PROCESSED}/account_train.csv")
        val = test = pd.read_csv(f"{PROCESSED}/account_test.csv")
    return train, val, test


class HateBRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            df["comentario"].astype(str).tolist(), truncation=True, max_length=max_length
        )
        self.labels = df["label_final"].tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


@torch.no_grad()
def evaluate(model, loader, dev) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(dev) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def train(model, train_loader, val_loader, dev, epochs: int):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += outputs.loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch+1} step {step}/{len(train_loader)} loss={outputs.loss.item():.4f}")

        val_labels, val_preds = evaluate(model, val_loader, dev)
        val_f1 = classification_report(val_labels, val_preds, output_dict=True)["macro avg"]["f1-score"]
        print(
            f"epoch {epoch+1}/{epochs} — loss médio {running_loss/len(train_loader):.4f} — "
            f"val F1 macro {val_f1:.4f} — {time.time()-t0:.0f}s"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "account"], default="random")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    dev = device()
    print(f"Device: {dev}")

    train_df, val_df, test_df = load_split(args.split)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(dev)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        HateBRDataset(train_df, tokenizer), batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(HateBRDataset(val_df, tokenizer), batch_size=32, collate_fn=collator)
    test_loader = DataLoader(HateBRDataset(test_df, tokenizer), batch_size=32, collate_fn=collator)

    train(model, train_loader, val_loader, dev, args.epochs)

    print(f"\n=== BERTimbau fine-tuned — split: {args.split} — avaliação no teste ===")
    test_labels, test_preds = evaluate(model, test_loader, dev)
    print(classification_report(test_labels, test_preds, target_names=["não-ofensivo", "ofensivo"]))

    out_dir = f"models/bertimbau-{args.split}"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Modelo salvo em {out_dir}")


if __name__ == "__main__":
    main()
