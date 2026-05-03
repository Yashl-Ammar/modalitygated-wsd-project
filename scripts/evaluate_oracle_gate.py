"""
scripts/evaluate_oracle_gate.py

Oracle gate: retroactively picks whichever branch (BERT or multimodal) produced
the correct prediction for each instance. This is the theoretical upper bound —
perfect routing with full knowledge of which branch is correct.

Routing rule:
    multi only correct  → use multimodal prediction
    bert only correct   → use BERT prediction
    both correct        → use BERT (tie)
    neither correct     → use BERT (neither)

Evaluated across five conditions:
    helpful / irrelevant / mixed 30/70 / mixed 50/50 / mixed 70/30

Output:
    results/oracle_gate_helpful.json
    results/oracle_gate_irrelevant.json
    results/oracle_gate_mixed_30_70.json
    results/oracle_gate_mixed_50_50.json
    results/oracle_gate_mixed_70_30.json

Usage:
    python scripts/evaluate_oracle_gate.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from encoders.bert_encoder import BERTEncoder
from encoders.clip_encoder import CLIPEncoder
from fusion.cross_attention import CrossAttentionFusion
from models.bert_model import BERTWSDModel
from models.multimodal_model import MultimodalWSDModel
from evaluation.evaluator import compute_metrics, POS_LABELS
from data.loader import load_jsonl
from utils.wordnet import sensekey_to_synset, pred_to_synset


# ── Constants ─────────────────────────────────────────────────────────────────

DATASETS = {
    "senseval2"  : "dataset/augmented_semeval/senseval2.jsonl",
    "senseval3"  : "dataset/augmented_semeval/senseval3.jsonl",
    "semeval2007": "dataset/augmented_semeval/semeval2007.jsonl",
    "semeval2013": "dataset/augmented_semeval/semeval2013.jsonl",
    "semeval2015": "dataset/augmented_semeval/semeval2015.jsonl",
}

# (display_label, image_mode, helpful_prob_or_None, out_tag)
EVAL_CONDITIONS = [
    ("Helpful",    "helpful",    None, "helpful"),
    ("Irrelevant", "irrelevant", None, "irrelevant"),
    ("Mixed 30/70","mixed",      0.30, "mixed_30_70"),
    ("Mixed 50/50","mixed",      0.50, "mixed_50_50"),
    ("Mixed 70/30","mixed",      0.70, "mixed_70_30"),
]

BERT_F1 = 0.7158
SEED    = 42


# ── Counter helpers ───────────────────────────────────────────────────────────

def empty_counters():
    c = {"overall": {"correct": 0, "total": 0, "skipped": 0}}
    for pos in POS_LABELS:
        c[pos] = {"correct": 0, "total": 0, "skipped": 0}
    return c


def update_counters(counters, pos, is_correct):
    counters["overall"]["total"]   += 1
    counters["overall"]["correct"] += int(is_correct)
    key = (pos or "UNKNOWN").upper()
    if key in counters:
        counters[key]["total"]   += 1
        counters[key]["correct"] += int(is_correct)


def counters_to_result(counters, elapsed):
    overall = compute_metrics(**counters["overall"])
    pos_bd  = {
        pos: compute_metrics(**counters[pos])
        for pos in POS_LABELS
        if counters[pos]["total"] > 0
    }
    return {
        "overall"      : overall,
        "pos_breakdown": pos_bd,
        "time_s"       : round(elapsed, 2),
        "speed"        : round(overall["n_total"] / (elapsed + 1e-8), 1),
    }


def aggregate_results(ds_results):
    total_correct = total_total = total_skipped = 0
    pos_agg = {p: {"correct": 0, "total": 0, "skipped": 0} for p in POS_LABELS}
    for r in ds_results.values():
        o = r["overall"]
        total_correct += o["n_correct"]
        total_total   += o["n_total"]
        total_skipped += o["n_skipped"]
        for pos in POS_LABELS:
            if pos in r["pos_breakdown"]:
                pm = r["pos_breakdown"][pos]
                pos_agg[pos]["correct"] += pm["n_correct"]
                pos_agg[pos]["total"]   += pm["n_total"]
    overall = compute_metrics(total_correct, total_total, total_skipped)
    pos_bd  = {
        pos: compute_metrics(pos_agg[pos]["correct"], pos_agg[pos]["total"], 0)
        for pos in POS_LABELS
        if pos_agg[pos]["total"] > 0
    }
    return {"overall": overall, "pos_breakdown": pos_bd}


# ── Per-dataset oracle evaluation ─────────────────────────────────────────────

def evaluate_dataset(ds_name, path, image_mode, helpful_prob,
                     bert_model, multi_model, device, rng):
    data     = load_jsonl(path)
    counters = empty_counters()

    n_bert_only  = 0   # oracle chose bert  (bert correct, multi wrong)
    n_multi_only = 0   # oracle chose multi (multi correct, bert wrong)
    n_both       = 0   # both correct — bert used (tie)
    n_neither    = 0   # neither correct — bert used
    start        = time.time()

    for item in tqdm(data, desc=f"  {ds_name}", leave=False):
        gold_syn = sensekey_to_synset(item["sense"])
        if gold_syn is None:
            counters["overall"]["skipped"] += 1
            continue

        sentence = item["sentence"]
        word     = item["word"]
        pos      = item.get("pos")

        # ── Image path ─────────────────────────────────────────────────────
        if image_mode == "mixed":
            chosen     = "helpful" if rng.random() < helpful_prob else "irrelevant"
            image_path = item.get("images", {}).get(chosen)
        else:
            image_path = item.get("images", {}).get(image_mode)

        # ── BERT inference ─────────────────────────────────────────────────
        pred_bert = bert_model.predict(sentence, word, pos)

        # ── Multimodal inference (always runs — oracle needs both) ─────────
        pred_multi = multi_model.predict(sentence, word, pos, image_path=image_path)

        # ── Correctness ────────────────────────────────────────────────────
        bert_syn  = pred_to_synset(pred_bert)  if pred_bert  else None
        multi_syn = pred_to_synset(pred_multi) if pred_multi else None

        bert_correct  = bert_syn  is not None and bert_syn  == gold_syn
        multi_correct = multi_syn is not None and multi_syn == gold_syn

        # ── Oracle routing ─────────────────────────────────────────────────
        if multi_correct and not bert_correct:
            oracle_pred = pred_multi
            n_multi_only += 1
        else:
            oracle_pred = pred_bert
            if bert_correct and not multi_correct:
                n_bert_only += 1
            elif bert_correct and multi_correct:
                n_both += 1
            else:
                n_neither += 1

        oracle_syn = pred_to_synset(oracle_pred) if oracle_pred else None
        update_counters(counters, pos, oracle_syn is not None and oracle_syn == gold_syn)

    elapsed = time.time() - start
    result  = counters_to_result(counters, elapsed)
    o       = result["overall"]

    print(f"\n  ── {ds_name}")
    print(f"     Oracle F1: {o['f1']:.4f}  |  {o['n_correct']:,}/{o['n_total']:,}")
    print(f"     routing → bert_only: {n_bert_only:,}  multi_only: {n_multi_only:,}  "
          f"both: {n_both:,}  neither: {n_neither:,}")

    return result


# ── Per-condition evaluation ──────────────────────────────────────────────────

def evaluate_condition(label, image_mode, helpful_prob, out_tag,
                       bert_model, multi_model, device, output_dir):
    rng = random.Random(SEED) if image_mode == "mixed" else None

    print(f"\n{'='*58}")
    print(f"  ORACLE GATE — {label.upper()}")
    print(f"{'='*58}")

    ds_results = {}
    start      = time.time()

    for ds_name, path in DATASETS.items():
        if not Path(path).exists():
            print(f"  [WARN] Not found: {path}")
            continue
        ds_results[ds_name] = evaluate_dataset(
            ds_name, path, image_mode, helpful_prob,
            bert_model, multi_model, device, rng,
        )

    total_time = time.time() - start
    ds_results["aggregate"] = aggregate_results(ds_results)

    record = {
        "condition"      : "oracle_gate",
        "image_condition": out_tag,
        "total_time_s"   : round(total_time, 2),
        "datasets"       : ds_results,
    }
    out_path = output_dir / f"oracle_gate_{out_tag}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    agg_f1 = ds_results["aggregate"]["overall"]["f1"]
    print(f"\n  Aggregate F1: {agg_f1:.4f}  |  {total_time:.1f}s")
    print(f"  Saved: {out_path}")
    return agg_f1


# ── Summary table helpers ─────────────────────────────────────────────────────

def load_f1_from_file(path, *keys):
    """Safely navigate a result JSON to datasets.aggregate.overall.f1."""
    try:
        d = json.load(open(path))
        for k in keys:
            d = d[k]
        return d["aggregate"]["overall"]["f1"]
    except Exception:
        return None


def fmt(v):
    return f"{v:.2%}" if v is not None else "   ?"


def print_summary_table(oracle_f1s):
    existing = {
        "Helpful"   : {
            "clip": load_f1_from_file("results/clip_multimodal_helpful.json",         "datasets"),
            "ens" : load_f1_from_file("results/learned_gate_ensemble_helpful.json",   "datasets"),
        },
        "Irrelevant": {
            "clip": load_f1_from_file("results/clip_multimodal_irrelevant.json",      "datasets"),
            "ens" : load_f1_from_file("results/learned_gate_ensemble_irrelevant.json","datasets"),
        },
        "Mixed 30/70": {
            "clip": load_f1_from_file("results/mixed_ratio_30_70.json", "conditions", "clip_multimodal",       "datasets"),
            "ens" : load_f1_from_file("results/mixed_ratio_30_70.json", "conditions", "learned_gate_ensemble", "datasets"),
        },
        "Mixed 50/50": {
            "clip": load_f1_from_file("results/mixed_ratio_50_50.json", "conditions", "clip_multimodal",       "datasets"),
            "ens" : load_f1_from_file("results/mixed_ratio_50_50.json", "conditions", "learned_gate_ensemble", "datasets"),
        },
        "Mixed 70/30": {
            "clip": load_f1_from_file("results/mixed_ratio_70_30.json", "conditions", "clip_multimodal",       "datasets"),
            "ens" : load_f1_from_file("results/mixed_ratio_70_30.json", "conditions", "learned_gate_ensemble", "datasets"),
        },
    }

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Condition':<14}  {'BERT':>8}  {'CLIP':>8}  {'Ensemble':>8}  {'Oracle':>8}")
    print(f"  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    for label, _, _, out_tag in EVAL_CONDITIONS:
        clip_f1   = existing.get(label, {}).get("clip")
        ens_f1    = existing.get(label, {}).get("ens")
        oracle_f1 = oracle_f1s.get(label)
        print(f"  {label:<14}  {fmt(BERT_F1):>8}  {fmt(clip_f1):>8}  "
              f"{fmt(ens_f1):>8}  {fmt(oracle_f1):>8}")

    print(f"{'='*70}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading BERT...")
    bert_encoder = BERTEncoder.load("checkpoints/bert_wsd/best", device=device)
    bert_model   = BERTWSDModel(bert_encoder, use_token_level=True)

    clip_encoder = CLIPEncoder(device=device)
    fusion = CrossAttentionFusion(embed_dim=512, num_heads=8, dropout=0.1, ff_dim=2048).to(device)
    fusion.load_state_dict(torch.load("checkpoints/fusion/best.pt", map_location=device))
    fusion.eval()
    multi_model = MultimodalWSDModel(clip_encoder, fusion)
    print("Models loaded.\n")

    oracle_f1s = {}

    for label, image_mode, helpful_prob, out_tag in EVAL_CONDITIONS:
        oracle_f1s[label] = evaluate_condition(
            label, image_mode, helpful_prob, out_tag,
            bert_model, multi_model, device, output_dir,
        )

    print_summary_table(oracle_f1s)


if __name__ == "__main__":
    main()
