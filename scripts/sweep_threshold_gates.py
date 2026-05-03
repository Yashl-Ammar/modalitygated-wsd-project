"""
scripts/sweep_threshold_gates.py

Sweep confidence and entropy thresholds on the SemCor gate-label validation split
to find the threshold that maximises F1 for each gate type.

Gate label = 1 means "multimodal is better than text-only for this instance".

Gate decision rules:
    confidence gate : gate = 1  if  max_prob < t    (BERT uncertain → use image)
    entropy gate    : gate = 1  if  entropy  > t    (BERT uncertain → use image)

Usage:
    python scripts/sweep_threshold_gates.py
"""

import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def binary_f1(y_true, y_pred):
    """F1 for the positive class (label = 1)."""
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    return 2 * precision * recall / (precision + recall + 1e-10)


def sweep(values, labels, thresholds, decision_fn, gate_name):
    """
    Sweep thresholds, print curve, return (best_t, best_f1).

    decision_fn(value, t) -> 0 or 1
    """
    print(f"\n{'─'*50}")
    print(f"  {gate_name} sweep")
    print(f"{'─'*50}")
    print(f"  {'threshold':>12}  {'F1':>8}  {'TP':>7}  {'FP':>7}  {'FN':>7}")

    best_t  = None
    best_f1 = -1.0
    results = {}

    for t in thresholds:
        preds = [decision_fn(v, t) for v in values]

        tp = sum(tl == 1 and p == 1 for tl, p in zip(labels, preds))
        fp = sum(tl == 0 and p == 1 for tl, p in zip(labels, preds))
        fn = sum(tl == 1 and p == 0 for tl, p in zip(labels, preds))

        precision = tp / (tp + fp + 1e-10)
        recall    = tp / (tp + fn + 1e-10)
        f1        = 2 * precision * recall / (precision + recall + 1e-10)

        results[round(float(t), 4)] = round(f1, 4)
        print(f"  {t:>12.2f}  {f1:>8.4f}  {tp:>7,}  {fp:>7,}  {fn:>7,}")

        if f1 > best_f1:
            best_f1 = f1
            best_t  = float(t)

    print(f"\n  Best threshold : {best_t:.2f}")
    print(f"  Best F1        : {best_f1:.4f}")
    return best_t, best_f1, results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    labels_path = Path("dataset/gate_labels/semcor_gate_labels.jsonl")
    out_path    = Path("results/gate_thresholds.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading gate labels from: {labels_path}")
    records = []
    with open(labels_path) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"  {len(records):,} records loaded")

    # ── Shuffle + split ───────────────────────────────────────────────────────
    random.seed(42)
    random.shuffle(records)
    split     = int(0.9 * len(records))
    val_data  = records[split:]
    print(f"  Train : {split:,}  |  Val : {len(val_data):,}")

    labels    = [r["gate_label"] for r in val_data]
    max_probs = [r["max_prob"]   for r in val_data]
    entropies = [r["entropy"]    for r in val_data]

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"\n  Validation label distribution:")
    print(f"    label=1 (use image) : {n_pos:,}  ({100*n_pos/len(labels):.1f}%)")
    print(f"    label=0 (text-only) : {n_neg:,}  ({100*n_neg/len(labels):.1f}%)")

    # ── Feature ranges ────────────────────────────────────────────────────────
    print(f"\n  max_prob range : [{min(max_probs):.4f}, {max(max_probs):.4f}]")
    print(f"  entropy  range : [{min(entropies):.4f}, {max(entropies):.4f}]")
    print(f"  entropy  p50   : {float(np.percentile(entropies, 50)):.4f}")
    print(f"  entropy  p90   : {float(np.percentile(entropies, 90)):.4f}")

    # ── Confidence sweep: 0.05 → 0.95, step 0.05 ─────────────────────────────
    conf_thresholds = np.arange(0.05, 1.00, 0.05)

    best_conf_t, best_conf_f1, conf_curve = sweep(
        values      = max_probs,
        labels      = labels,
        thresholds  = conf_thresholds,
        decision_fn = lambda v, t: 1 if v < t else 0,
        gate_name   = "CONFIDENCE GATE  (gate=1 if max_prob < t)",
    )

    # ── Entropy sweep: 0.05 → 5.00, step 0.05 ────────────────────────────────
    ent_max       = max(5.00, float(np.percentile(entropies, 99)) + 0.5)
    ent_thresholds = np.arange(0.05, ent_max, 0.05)

    best_ent_t, best_ent_f1, ent_curve = sweep(
        values      = entropies,
        labels      = labels,
        thresholds  = ent_thresholds,
        decision_fn = lambda v, t: 1 if v > t else 0,
        gate_name   = "ENTROPY GATE  (gate=1 if entropy > t)",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        "confidence_threshold" : round(best_conf_t, 4),
        "entropy_threshold"    : round(best_ent_t,  4),
        "confidence_best_f1"   : round(best_conf_f1, 4),
        "entropy_best_f1"      : round(best_ent_f1,  4),
        "val_size"             : len(val_data),
        "sweep_curves": {
            "confidence": conf_curve,
            "entropy"   : ent_curve,
        },
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  CONFIDENCE threshold : {best_conf_t:.2f}  (F1 = {best_conf_f1:.4f})")
    print(f"  ENTROPY    threshold : {best_ent_t:.2f}  (F1 = {best_ent_f1:.4f})")
    print(f"\n  Saved to: {out_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
