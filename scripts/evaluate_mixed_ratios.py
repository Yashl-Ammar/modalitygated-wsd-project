"""
scripts/evaluate_mixed_ratios.py

Evaluate CLIP multimodal, learned gate, and learned gate ensemble across three
helpful/irrelevant image ratios on all five SemEval benchmarks.

Ratios:  30/70  |  50/50  |  70/30  (helpful/irrelevant probability)

For each ratio:
  - A fresh random.Random(42) is used, so draws are at identical RNG positions
    across all three ratios. An instance that draws helpful in 30/70 also draws
    helpful in 50/50 and 70/30 (subset relationship).
  - CLIP multimodal always runs (needed as always-use-image baseline).
  - Gate and ensemble reuse the same CLIP forward pass — no double inference.

Output:
    results/mixed_ratio_30_70.json
    results/mixed_ratio_50_50.json
    results/mixed_ratio_70_30.json

    Each file contains clip_multimodal, learned_gate, and learned_gate_ensemble
    results in standard condition-JSON format.

Usage:
    python scripts/evaluate_mixed_ratios.py
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
from models.gate_model import GatingNetwork, build_feature_vector
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

RATIOS = [
    (0.30, "30_70"),
    (0.50, "50_50"),
    (0.70, "70_30"),
]

BERT_F1 = 0.7158   # bert_finetuned aggregate F1, text-only (no re-evaluation needed)
SEED    = 42


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_all_features(probs: torch.Tensor):
    p = probs.cpu().float()
    sorted_p, _ = torch.sort(p, descending=True)
    p1 = sorted_p[0].item()
    p2 = sorted_p[1].item() if len(sorted_p) > 1 else 0.0
    return (
        p1,
        p1 - p2,
        min(p1 / p2, 999.0) if p2 > 0.0 else 999.0,
        -float(torch.sum(p * torch.log(p + 1e-10))),
    )


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


# ── Per-dataset evaluation (single pass, three systems) ───────────────────────

def evaluate_dataset(ds_name, path, helpful_prob, bert_model, multi_model,
                     gate_model, norm_stats, device, rng):
    """
    Single pass tracking clip_multimodal, learned_gate, learned_gate_ensemble.

    CLIP always runs (needed for the baseline). Gate and ensemble reuse the same
    forward pass result.

    Returns (clip_result, gate_result, ens_result).
    """
    data           = load_jsonl(path)
    clip_counters  = empty_counters()
    gate_counters  = empty_counters()
    ens_counters   = empty_counters()

    n_gate_use = 0
    n_bert_win = 0
    n_clip_win = 0
    start      = time.time()

    gate_model.eval()

    for item in tqdm(data, desc=f"  {ds_name}", leave=False):
        gold_syn = sensekey_to_synset(item["sense"])
        if gold_syn is None:
            for c in (clip_counters, gate_counters, ens_counters):
                c["overall"]["skipped"] += 1
            continue

        sentence = item["sentence"]
        word     = item["word"]
        pos      = item.get("pos")

        # ── Image assignment for this ratio ────────────────────────────────
        chosen     = "helpful" if rng.random() < helpful_prob else "irrelevant"
        image_path = item.get("images", {}).get(chosen)

        # ── BERT inference ─────────────────────────────────────────────────
        pred_bert, probs, _ = bert_model.predict_with_scores(sentence, word, pos)

        if probs is not None:
            max_prob, margin, p1_p2_ratio, entropy = extract_all_features(probs)
            feat_vec = build_feature_vector(
                max_prob, margin, p1_p2_ratio, entropy, pos, norm_stats
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                gate_opens = (torch.sigmoid(gate_model(feat_vec)) > 0.5).item()
        else:
            max_prob   = 0.0
            gate_opens = False

        # ── CLIP multimodal (always run) ───────────────────────────────────
        pred_multi, clip_max_sim = (None, None)
        if image_path is not None:
            pred_multi, clip_max_sim = multi_model.predict_with_max_similarity(
                sentence, word, pos, image_path=image_path
            )
            if gate_opens:
                n_gate_use += 1

        # ── clip_multimodal: always use image ──────────────────────────────
        pred_clip = pred_multi if image_path is not None else pred_bert
        clip_syn  = pred_to_synset(pred_clip) if pred_clip else None
        update_counters(clip_counters, pos, clip_syn is not None and clip_syn == gold_syn)

        # ── learned_gate: use CLIP only when gate opens ────────────────────
        pred_gate = pred_multi if (gate_opens and image_path is not None) else pred_bert
        gate_syn  = pred_to_synset(pred_gate) if pred_gate else None
        update_counters(gate_counters, pos, gate_syn is not None and gate_syn == gold_syn)

        # ── learned_gate_ensemble: confidence tiebreak on gate=1 ──────────
        if gate_opens and clip_max_sim is not None:
            if max_prob > clip_max_sim:
                pred_ens = pred_bert
                n_bert_win += 1
            else:
                pred_ens = pred_multi
                n_clip_win += 1
        else:
            pred_ens = pred_bert
        ens_syn = pred_to_synset(pred_ens) if pred_ens else None
        update_counters(ens_counters, pos, ens_syn is not None and ens_syn == gold_syn)

    elapsed = time.time() - start

    o        = compute_metrics(**gate_counters["overall"])
    gate_pct = 100 * n_gate_use / max(o["n_total"], 1)
    print(f"\n  ── {ds_name}")
    print(f"     clip  F1: {counters_to_result(clip_counters, elapsed)['overall']['f1']:.4f}  "
          f"|  gate  F1: {o['f1']:.4f} (routed {n_gate_use:,}, {gate_pct:.0f}%)  "
          f"|  ens   F1: {counters_to_result(ens_counters, elapsed)['overall']['f1']:.4f}  "
          f"[bert_win {n_bert_win:,} / clip_win {n_clip_win:,}]")

    return (
        counters_to_result(clip_counters, elapsed),
        counters_to_result(gate_counters, elapsed),
        counters_to_result(ens_counters,  elapsed),
    )


# ── Per-ratio evaluation ──────────────────────────────────────────────────────

def evaluate_ratio(helpful_prob, ratio_tag, bert_model, multi_model, gate_model,
                   norm_stats, device, output_dir):
    rng = random.Random(SEED)

    clip_ds = {}
    gate_ds = {}
    ens_ds  = {}
    start   = time.time()

    for ds_name, path in DATASETS.items():
        if not Path(path).exists():
            print(f"  [WARN] Not found: {path}")
            continue
        clip_r, gate_r, ens_r = evaluate_dataset(
            ds_name, path, helpful_prob,
            bert_model, multi_model, gate_model, norm_stats, device, rng,
        )
        clip_ds[ds_name] = clip_r
        gate_ds[ds_name] = gate_r
        ens_ds[ds_name]  = ens_r

    total_time = time.time() - start

    clip_ds["aggregate"] = aggregate_results(clip_ds)
    gate_ds["aggregate"] = aggregate_results(gate_ds)
    ens_ds ["aggregate"] = aggregate_results(ens_ds)

    record = {
        "ratio"           : ratio_tag,
        "helpful_prob"    : helpful_prob,
        "irrelevant_prob" : round(1.0 - helpful_prob, 2),
        "seed"            : SEED,
        "total_time_s"    : round(total_time, 2),
        "conditions": {
            "clip_multimodal"        : {"condition": "clip_multimodal",         "datasets": clip_ds},
            "learned_gate"           : {"condition": "learned_gate",            "datasets": gate_ds},
            "learned_gate_ensemble"  : {"condition": "learned_gate_ensemble",   "datasets": ens_ds},
        },
    }

    out_path = Path(output_dir) / f"mixed_ratio_{ratio_tag}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return (
        clip_ds["aggregate"]["overall"]["f1"],
        gate_ds["aggregate"]["overall"]["f1"],
        ens_ds ["aggregate"]["overall"]["f1"],
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading BERT...")
    bert_encoder = BERTEncoder.load("checkpoints/bert_wsd/best", device=device)
    bert_model   = BERTWSDModel(bert_encoder, use_token_level=True)

    clip_encoder = CLIPEncoder(device=device)
    fusion = CrossAttentionFusion(embed_dim=512, num_heads=8, dropout=0.1, ff_dim=2048).to(device)
    fusion.load_state_dict(torch.load("checkpoints/fusion/best.pt", map_location=device))
    fusion.eval()
    multi_model = MultimodalWSDModel(clip_encoder, fusion)
    print("Fusion loaded.")

    gate_model = GatingNetwork().to(device)
    gate_model.load_state_dict(torch.load("checkpoints/gate/best.pt", map_location=device))
    gate_model.eval()
    print("Gate network loaded.")

    with open("checkpoints/gate/norm_stats.json") as f:
        norm_stats = json.load(f)
    print("Norm stats loaded.\n")

    # ── Evaluate each ratio ───────────────────────────────────────────────────
    summary = []   # [(ratio_label, clip_f1, gate_f1, ens_f1)]

    for helpful_prob, ratio_tag in RATIOS:
        label = ratio_tag.replace("_", "/")
        print(f"\n{'='*60}")
        print(f"  RATIO {label}  (helpful={helpful_prob:.0%}  irrelevant={1-helpful_prob:.0%})")
        print(f"{'='*60}")

        clip_f1, gate_f1, ens_f1 = evaluate_ratio(
            helpful_prob, ratio_tag,
            bert_model, multi_model, gate_model, norm_stats, device, output_dir,
        )
        summary.append((label, clip_f1, gate_f1, ens_f1))

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  AGGREGATE F1 SUMMARY  (seed={SEED})")
    print(f"{'='*60}")
    print(f"  {'Ratio':<10}  {'BERT':>8}  {'CLIP':>8}  {'Gate':>8}  {'Ensemble':>8}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for label, clip_f1, gate_f1, ens_f1 in summary:
        print(f"  {label:<10}  {BERT_F1:>7.2%}  {clip_f1:>7.2%}  {gate_f1:>7.2%}  {ens_f1:>7.2%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
