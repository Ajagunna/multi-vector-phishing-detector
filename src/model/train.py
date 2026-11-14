"""
MeAJOR offline sheets are text+label; use src.features.extractor.extract_full_features
for the live multi-feature schema (VT, sandbox, SPF/DKIM proxies) when retraining
on richer IMAP-derived corpora.

Train the Multi-Vector classifier on MeAJOR (text + engineered metadata).
Compact model for sandbox memory limits.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.model.architecture import MultiVectorPhishingClassifier, NUM_METADATA_FEATURES, MAX_LENGTH
from src.features.extractor import (
    extract_metadata_vector,
    prepare_text_for_model,
    simple_tokenize,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "MeAJOR Dataset"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhishingDataset(Dataset):
    def __init__(self, texts, labels, max_len=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.metas = [extract_metadata_vector(t) for t in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = prepare_text_for_model(self.texts[idx], max_chars=800)
        ids, mask = simple_tokenize(text, self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "metadata": torch.tensor(self.metas[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_split(name: str, max_rows: int = None) -> tuple:
    path = DATA_DIR / f"{name}.xlsx"
    print(f"[DATA] Loading {path} ...")
    df = pd.read_excel(path)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    print(f"       {name}: {len(texts)} samples, phishing ratio={sum(labels)/len(labels):.3f}")
    return texts, labels


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            meta = batch["metadata"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(ids, mask, meta)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, y in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                if p == 1 and y == 1: tp += 1
                elif p == 1 and y == 0: fp += 1
                elif p == 0 and y == 0: tn += 1
                else: fn += 1
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    acc = correct / max(total, 1)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def train(
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_train: int = 2000,
    max_val: int = 500,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Device: {DEVICE}")

    train_texts, train_labels = load_split("train", max_rows=max_train)
    val_texts, val_labels = load_split("validation", max_rows=max_val)

    train_ds = PhishingDataset(train_texts, train_labels)
    val_ds = PhishingDataset(val_texts, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MultiVectorPhishingClassifier(use_surrogate=True).to(DEVICE)
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    weight = torch.tensor([1.0, max(n_neg / max(n_pos, 1), 0.5)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_f1 = 0.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        for batch in train_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            meta = batch["metadata"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            logits = model(ids, mask, meta)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        train_loss = running / max(len(train_loader), 1)
        metrics = evaluate(model, val_loader, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, **{k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}})
        print(
            f"Epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_f1={metrics['f1']:.4f}  "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f}  "
            f"({time.time()-t0:.1f}s)"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            ckpt = MODEL_DIR / "best_model.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "meta_dim": NUM_METADATA_FEATURES,
                    "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
                    "epoch": epoch,
                },
                ckpt,
            )
            print(f"         -> saved best model (F1={best_f1:.4f})")

    # quick test
    test_texts, test_labels = load_split("test", max_rows=800)
    test_ds = PhishingDataset(test_texts, test_labels)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    test_metrics = evaluate(model, test_loader, criterion)
    print("\n[TEST] Held-out metrics:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    results = {
        "history": history,
        "test": {k: float(v) if isinstance(v, (int, float)) else v for k, v in test_metrics.items()},
        "best_val_f1": best_f1,
    }
    with open(MODEL_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] Results -> {MODEL_DIR / 'training_results.json'}")
    return model, results


if __name__ == "__main__":
    train(epochs=3, batch_size=32, max_train=2000, max_val=500)
