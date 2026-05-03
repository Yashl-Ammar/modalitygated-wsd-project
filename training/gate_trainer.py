"""
training/gate_trainer.py

Train the learned gating network on SemCor gate labels.

The gate predicts whether multimodal (CLIP + fusion) will outperform
fine-tuned BERT for a given instance, using only BERT's output statistics
(max softmax probability, margin, p1/p2 ratio, entropy) and POS.

Usage:
    python training/gate_trainer.py
    python training/gate_trainer.py --epochs 30 --lr 0.0005
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.gate_model import (
    GatingNetwork,
    SCALAR_FEATURES,
    POS_ORDER,
    pos_to_onehot,
)


# ── Feature extraction ────────────────────────────────────────────────────────

def load_and_split(path, seed=42, val_fraction=0.10):
    """
    Load gate label records, shuffle with fixed seed, split 90/10.
    Returns (train_records, val_records).
    """
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    random.seed(seed)
    random.shuffle(records)
    split = int((1.0 - val_fraction) * len(records))
    return records[:split], records[split:]


def compute_norm_stats(records):
    """Compute mean and std for each scalar feature over a set of records."""
    import statistics
    stats = {}
    for feat in SCALAR_FEATURES:
        vals   = [r[feat] for r in records]
        mean   = statistics.mean(vals)
        stdev  = statistics.stdev(vals) if len(vals) > 1 else 1.0
        stats[feat] = {"mean": mean, "std": max(stdev, 1e-8)}
    return stats


def records_to_tensors(records, norm_stats):
    """Convert records to (X, y) float tensors."""
    X_rows = []
    y_vals = []

    mean = {f: norm_stats[f]["mean"] for f in SCALAR_FEATURES}
    std  = {f: norm_stats[f]["std"]  for f in SCALAR_FEATURES}

    for r in records:
        scalars = [
            (r[feat] - mean[feat]) / std[feat]
            for feat in SCALAR_FEATURES
        ]
        onehot = pos_to_onehot(r.get("pos"))
        X_rows.append(scalars + onehot)
        y_vals.append(float(r["gate_label"]))

    X = torch.tensor(X_rows, dtype=torch.float32)
    y = torch.tensor(y_vals, dtype=torch.float32)
    return X, y


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_val_f1(model, X_val, y_val, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_val.to(device))
        preds  = (torch.sigmoid(logits) > 0.5).long().cpu()
        labels = y_val.long()

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)
    return f1, precision, recall, tp, fp, fn


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load + split ──────────────────────────────────────────────────────────
    print(f"Loading gate labels from: {args.labels}")
    train_records, val_records = load_and_split(args.labels, seed=42)
    print(f"  Train: {len(train_records):,}  |  Val: {len(val_records):,}")

    n_pos = sum(r["gate_label"] for r in train_records)
    n_neg = len(train_records) - n_pos
    print(f"  Train label dist — label=1: {n_pos:,} ({100*n_pos/len(train_records):.1f}%)  "
          f"label=0: {n_neg:,}")

    # ── Normalisation stats (train only) ──────────────────────────────────────
    print("Computing normalisation statistics from train split...")
    norm_stats_raw = compute_norm_stats(train_records)

    # flatten for JSON serialisation
    norm_stats_json = {
        "mean": {f: norm_stats_raw[f]["mean"] for f in SCALAR_FEATURES},
        "std" : {f: norm_stats_raw[f]["std"]  for f in SCALAR_FEATURES},
    }
    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats_json, f, indent=2)
    print(f"  Saved: {out_dir / 'norm_stats.json'}")

    # ── Build tensors ─────────────────────────────────────────────────────────
    X_train, y_train = records_to_tensors(train_records, norm_stats_raw)
    X_val,   y_val   = records_to_tensors(val_records,   norm_stats_raw)
    print(f"  Feature matrix: train {tuple(X_train.shape)}  val {tuple(X_val.shape)}")

    # ── Model + loss ──────────────────────────────────────────────────────────
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    model     = GatingNetwork().to(device)
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-8)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Gate network parameters: {n_params:,}")
    print(f"pos_weight: {pos_weight.item():.2f}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_ds     = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_f1   = -1.0
    best_epoch    = 0

    print(f"\nTraining for {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"{'─'*70}")
    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val F1':>8}  "
          f"{'Prec':>7}  {'Recall':>7}  {'TP':>6}  {'FP':>6}  {'FN':>6}")
    print(f"{'─'*70}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        t0         = time.time()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_f1, prec, rec, tp, fp, fn = compute_val_f1(model, X_val, y_val, device)

        marker = " *" if val_f1 > best_val_f1 else ""
        print(f"  {epoch:>5}  {avg_loss:>11.4f}  {val_f1:>8.4f}  "
              f"{prec:>7.4f}  {rec:>7.4f}  {tp:>6,}  {fp:>6,}  {fn:>6,}{marker}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), out_dir / "best.pt")

    print(f"{'─'*70}")
    print(f"\nBest val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {out_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",     default="dataset/gate_labels/semcor_gate_labels.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/gate")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
