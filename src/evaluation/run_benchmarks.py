"""
Offline benchmark: classical baselines + surrogate multi-vector + DistilBERT multi-vector
on MeAJOR train/test splits. Writes models/benchmark_results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.extractor import extract_metadata_vector, prepare_text_for_model, simple_tokenize
from src.model.architecture import MultiVectorPhishingClassifier, NUM_METADATA_FEATURES, MAX_LENGTH

DATA_DIR = ROOT / "data" / "MeAJOR Dataset"
MODEL_DIR = ROOT / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_xy(split: str, max_rows: int | None = None):
    df = pd.read_excel(DATA_DIR / f"{split}.xlsx")
    if max_rows and len(df) > max_rows:
        # stratified sample
        parts = []
        for lab in (0, 1):
            sub = df[df["label"] == lab]
            n = min(len(sub), max_rows // 2)
            parts.append(sub.sample(n=n, random_state=42) if len(sub) > n else sub)
        df = pd.concat(parts).sample(frac=1, random_state=42)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def prf1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run_baselines(train_texts, train_y, test_texts, test_y):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import FeatureUnion

    results = {}
    # pure TF-IDF text baselines
    configs = [
        ("Naive_Bayes_TFIDF", Pipeline([
            ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)),
            ("clf", MultinomialNB(alpha=0.5)),
        ])),
        ("Linear_SVM_TFIDF", Pipeline([
            ("tfidf", TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)),
            ("clf", LinearSVC(C=1.0, max_iter=3000, dual="auto")),
        ])),
        ("Random_Forest_TFIDF", Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)),
            ("clf", RandomForestClassifier(n_estimators=120, max_depth=25, n_jobs=-1, random_state=42)),
        ])),
    ]
    for name, pipe in configs:
        t0 = time.time()
        pipe.fit(train_texts, train_y)
        pred = pipe.predict(test_texts)
        m = prf1(test_y, pred)
        m["train_seconds"] = round(time.time() - t0, 2)
        results[name] = m
        print(f"[BASELINE] {name:22} F1={m['f1']:.4f}  P={m['precision']:.3f} R={m['recall']:.3f}  ({m['train_seconds']}s)")
    return results


class MetaTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_len=128, use_hf=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_hf = use_hf
        self.metas = [extract_metadata_vector(t) for t in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = prepare_text_for_model(self.texts[idx], max_chars=1200)
        if self.use_hf:
            enc = self.tokenizer(
                text, truncation=True, padding="max_length",
                max_length=self.max_len, return_tensors="pt",
            )
            ids = enc["input_ids"].squeeze(0)
            mask = enc["attention_mask"].squeeze(0)
        else:
            tok = simple_tokenize(text, self.max_len)
            ids = torch.tensor(tok["input_ids"], dtype=torch.long)
            mask = torch.tensor(tok["attention_mask"], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": mask,
            "metadata": torch.tensor(self.metas[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


@torch.no_grad()
def eval_torch(model, loader):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        meta = batch["metadata"].to(DEVICE)
        logits = model(ids, mask, meta)
        pred = logits.argmax(dim=1).cpu().tolist()
        ys.extend(batch["label"].tolist())
        ps.extend(pred)
    return prf1(ys, ps)


def train_torch_model(model, train_loader, val_loader, epochs=2, lr=2e-5, freeze_backbone=False):
    if freeze_backbone and not model.use_surrogate and hasattr(model, "transformer"):
        for p in model.transformer.parameters():
            p.requires_grad = False
        # unfreeze last layer lightly
        for p in model.transformer.transformer.layer[-1].parameters():
            p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    # class weights from first epoch batch sizes approximate
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            meta = batch["metadata"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(ids, mask, meta), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item()
            n += 1
        val = eval_torch(model, val_loader)
        print(f"  epoch {ep}: train_loss={total/max(n,1):.4f}  val_f1={val['f1']:.4f}")
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def run_surrogate(train_texts, train_y, test_texts, test_y, val_texts, val_y):
    print("[SURROGATE] Evaluating existing checkpoint / quick retrain…")
    ckpt = MODEL_DIR / "best_model.pt"
    model = MultiVectorPhishingClassifier(use_surrogate=True).to(DEVICE)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model_state"])
        print("  loaded best_model.pt")
    train_ds = MetaTextDataset(train_texts, train_y, max_len=MAX_LENGTH, use_hf=False)
    val_ds = MetaTextDataset(val_texts, val_y, max_len=MAX_LENGTH, use_hf=False)
    test_ds = MetaTextDataset(test_texts, test_y, max_len=MAX_LENGTH, use_hf=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)
    # light fine-tune on larger data for better numbers
    model = train_torch_model(model, train_loader, val_loader, epochs=2, lr=1e-3)
    torch.save({"model_state": model.state_dict(), "use_surrogate": True}, MODEL_DIR / "best_model.pt")
    m = eval_torch(model, test_loader)
    print(f"[SURROGATE] test F1={m['f1']:.4f}")
    return m


def run_distilbert(train_texts, train_y, test_texts, test_y, val_texts, val_y):
    print("[DISTILBERT] Fine-tuning multi-vector DistilBERT (partial freeze)…")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = MultiVectorPhishingClassifier(use_surrogate=False).to(DEVICE)
    # smaller subset for speed if huge
    max_tr = min(len(train_texts), 4000)
    max_va = min(len(val_texts), 800)
    max_te = min(len(test_texts), 1500)
    tr_t, tr_y = train_texts[:max_tr], train_y[:max_tr]
    va_t, va_y = val_texts[:max_va], val_y[:max_va]
    te_t, te_y = test_texts[:max_te], test_y[:max_te]

    train_ds = MetaTextDataset(tr_t, tr_y, tokenizer=tok, max_len=128, use_hf=True)
    val_ds = MetaTextDataset(va_t, va_y, tokenizer=tok, max_len=128, use_hf=True)
    test_ds = MetaTextDataset(te_t, te_y, tokenizer=tok, max_len=128, use_hf=True)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    t0 = time.time()
    model = train_torch_model(model, train_loader, val_loader, epochs=2, lr=2e-5, freeze_backbone=True)
    train_sec = round(time.time() - t0, 1)
    m = eval_torch(model, test_loader)
    m["train_seconds"] = train_sec
    m["note"] = f"DistilBERT multi-vector; freeze backbone except last layer; train n={max_tr}, test n={max_te}"
    # save checkpoint
    torch.save(
        {
            "model_state": model.state_dict(),
            "use_surrogate": False,
            "metrics": m,
        },
        MODEL_DIR / "distilbert_multivector.pt",
    )
    print(f"[DISTILBERT] test F1={m['f1']:.4f}  ({train_sec}s)")
    return m


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print("Loading MeAJOR splits…")
    train_texts, train_y = load_xy("train", max_rows=8000)
    val_texts, val_y = load_xy("validation", max_rows=2000)
    test_texts, test_y = load_xy("test", max_rows=3000)
    print(f"  train={len(train_y)}  val={len(val_y)}  test={len(test_y)}")
    print(f"  train pos={sum(train_y)} neg={len(train_y)-sum(train_y)}")

    results = {
        "dataset": {
            "name": "MeAJOR",
            "train_n": len(train_y),
            "val_n": len(val_y),
            "test_n": len(test_y),
            "train_phishing": int(sum(train_y)),
            "test_phishing": int(sum(test_y)),
        },
        "models": {},
    }

    results["models"].update(run_baselines(train_texts, train_y, test_texts, test_y))
    results["models"]["Surrogate_MultiVector"] = run_surrogate(
        train_texts, train_y, test_texts, test_y, val_texts, val_y
    )
    try:
        results["models"]["DistilBERT_MultiVector"] = run_distilbert(
            train_texts, train_y, test_texts, test_y, val_texts, val_y
        )
    except Exception as e:
        print("[DISTILBERT] failed:", e)
        results["models"]["DistilBERT_MultiVector"] = {"error": str(e)}

    out = MODEL_DIR / "benchmark_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Wrote {out}")

    # markdown table
    print("\n| Model | Precision | Recall | F1 | Accuracy |")
    print("|-------|-----------|--------|----|----------|")
    for name, m in results["models"].items():
        if "error" in m:
            print(f"| {name} | error | | | |")
        else:
            print(f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['accuracy']:.4f} |")


if __name__ == "__main__":
    main()
