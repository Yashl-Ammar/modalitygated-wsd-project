"""
scripts/evaluate_learned_gate.py

Evaluate the learned gating network on all five SemEval benchmarks across
three image conditions (helpful / misleading / irrelevant).

For each instance:
    1. Run fine-tuned BERT → softmax probs → max_prob, margin, p1_p2_ratio, entropy
    2. Normalise scalars using checkpoints/gate/norm_stats.json
    3. One-hot encode POS
    4. Run learned gate → sigmoid > 0.5 → binary decision
    5. gate=1 → invoke multimodal; gate=0 → keep BERT prediction

Output files:
    results/learned_gate_helpful.json
    results/learned_gate_misleading.json
    results/learned_gate_irrelevant.json

Usage:
    python scripts/evaluate_learned_gate.py
    python scripts/evaluate_learned_gate.py --dataset semeval2013
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


DATASETS = {
    "senseval2"  : "dataset/augmented_semeval/senseval2.jsonl",
    "senseval3"  : "dataset/augmented_semeval/senseval3.jsonl",
    "semeval2007": "dataset/augmented_semeval/semeval2007.jsonl",
    "semeval2013": "dataset/augmented_semeval/semeval2013.jsonl",
    "semeval2015": "dataset/augmented_semeval/semeval2015.jsonl",
}

IMAGE_CONDITIONS    = ["helpful", "misleading", "irrelevant", "mixed"]
ENSEMBLE_CONDITIONS = {"helpful", "irrelevant", "mixed"}


# ── Feature extraction (mirrors generate_gate_labels.py) ─────────────────────

def extract_all_features(probs: torch.Tensor):
    """
    Extract max_prob, margin, p1_p2_ratio, entropy from a softmax distribution.
    Returns (max_prob, margin, p1_p2_ratio, entropy) as Python floats.
    """
    p = probs.cpu().float()
    sorted_p, _ = torch.sort(p, descending=True)

    p1 = sorted_p[0].item()
    p2 = sorted_p[1].item() if len(sorted_p) > 1 else 0.0

    max_prob    = p1
    margin      = p1 - p2
    p1_p2_ratio = min(p1 / p2, 999.0) if p2 > 0.0 else 999.0
    entropy     = -float(torch.sum(p * torch.log(p + 1e-10)))
    return max_prob, margin, p1_p2_ratio, entropy


# ── Counters helpers (shared with evaluate_threshold_gates.py pattern) ────────

def empty_counters():
    c = {"overall": {"correct": 0, "total": 0, "skipped": 0}}
    for pos in POS_LABELS:
        c[pos] = {"correct": 0, "total": 0, "skipped": 0}
    return c


def update_counters(counters, pos, is_correct):
    counters["overall"]["total"]   += 1
    counters["overall"]["correct"] += int(is_correct)
    pos_key = (pos or "UNKNOWN").upper()
    if pos_key in counters:
        counters[pos_key]["total"]   += 1
        counters[pos_key]["correct"] += int(is_correct)


def counters_to_result(counters, elapsed):
    overall = compute_metrics(**counters["overall"])
    pos_bd  = {}
    for pos in POS_LABELS:
        c = counters[pos]
        if c["total"] > 0:
            pos_bd[pos] = compute_metrics(**c)
    return {
        "overall"      : overall,
        "pos_breakdown": pos_bd,
        "time_s"       : round(elapsed, 2),
        "speed"        : round(overall["n_total"] / (elapsed + 1e-8), 1),
    }


def aggregate_results(all_results):
    total_correct = total_total = total_skipped = 0
    pos_agg = {p: {"correct": 0, "total": 0, "skipped": 0} for p in POS_LABELS}
    for ds_result in all_results.values():
        o = ds_result["overall"]
        total_correct += o["n_correct"]
        total_total   += o["n_total"]
        total_skipped += o["n_skipped"]
        for pos in POS_LABELS:
            if pos in ds_result["pos_breakdown"]:
                pm = ds_result["pos_breakdown"][pos]
                pos_agg[pos]["correct"] += pm["n_correct"]
                pos_agg[pos]["total"]   += pm["n_total"]
    overall = compute_metrics(total_correct, total_total, total_skipped)
    pos_bd  = {}
    for pos in POS_LABELS:
        c = pos_agg[pos]
        if c["total"] > 0:
            pos_bd[pos] = compute_metrics(c["correct"], c["total"], 0)
    return {"overall": overall, "pos_breakdown": pos_bd}


# ── Image path resolution ─────────────────────────────────────────────────────

def get_image_path(item, image_condition, rng=None):
    """
    Return the image path for the given condition.
    'mixed' randomly selects between helpful and irrelevant (requires rng).
    """
    if image_condition == "mixed":
        chosen = rng.choice(["helpful", "irrelevant"])
        return item.get("images", {}).get(chosen)
    return item.get("images", {}).get(image_condition)


# ── Per-dataset evaluation ────────────────────────────────────────────────────

def evaluate_dataset(ds_name, path, bert_model, multi_model, gate_model,
                     norm_stats, device, image_condition, include_ensemble,
                     include_clip_baseline, rng, verbose):
    """
    Single-pass evaluation producing up to three result dicts simultaneously:
        gate_result:  learned_gate routing
        ens_result:   learned_gate_ensemble routing  (None if not include_ensemble)
        clip_result:  always-use-image CLIP baseline (None if not include_clip_baseline)

    include_clip_baseline=True forces multimodal to run on every instance regardless
    of the gate decision. Used for the 'mixed' condition only.
    rng is a seeded random.Random used for mixed image selection.
    """
    data          = load_jsonl(path)
    gate_counters = empty_counters()
    ens_counters  = empty_counters() if include_ensemble        else None
    clip_counters = empty_counters() if include_clip_baseline   else None

    n_gate_use = 0
    n_bert_win = 0
    n_clip_win = 0
    start      = time.time()

    gate_model.eval()

    for item in tqdm(data, desc=f"  {ds_name}", leave=False):
        gold_syn = sensekey_to_synset(item["sense"])
        if gold_syn is None:
            gate_counters["overall"]["skipped"] += 1
            if include_ensemble:      ens_counters ["overall"]["skipped"] += 1
            if include_clip_baseline: clip_counters["overall"]["skipped"] += 1
            continue

        sentence   = item["sentence"]
        word       = item["word"]
        pos        = item.get("pos")
        image_path = get_image_path(item, image_condition, rng)

        # ── BERT inference ─────────────────────────────────────────────────
        pred_bert, probs, _ = bert_model.predict_with_scores(sentence, word, pos)

        if probs is not None:
            max_prob, margin, p1_p2_ratio, entropy = extract_all_features(probs)
            feat_vec = build_feature_vector(
                max_prob, margin, p1_p2_ratio, entropy, pos, norm_stats
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                logit      = gate_model(feat_vec)
                gate_opens = (torch.sigmoid(logit) > 0.5).item()
        else:
            max_prob   = 0.0
            gate_opens = False

        # ── Multimodal inference ───────────────────────────────────────────
        # Run when gate opens OR when we need the clip baseline (mixed condition).
        pred_multi   = None
        clip_max_sim = None

        need_multi = (gate_opens and image_path is not None) or \
                     (include_clip_baseline and image_path is not None)

        if need_multi:
            pred_multi, clip_max_sim = multi_model.predict_with_max_similarity(
                sentence, word, pos, image_path=image_path
            )
            if gate_opens and image_path is not None:
                n_gate_use += 1

        # ── learned_gate routing ───────────────────────────────────────────
        pred_gate = pred_multi if (gate_opens and image_path is not None) else pred_bert
        gate_syn  = pred_to_synset(pred_gate) if pred_gate else None
        update_counters(gate_counters, pos, gate_syn is not None and gate_syn == gold_syn)

        # ── learned_gate_ensemble routing ──────────────────────────────────
        if include_ensemble:
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

        # ── clip_multimodal baseline (always use image) ────────────────────
        if include_clip_baseline:
            pred_clip = pred_multi if image_path is not None else pred_bert
            clip_syn  = pred_to_synset(pred_clip) if pred_clip else None
            update_counters(clip_counters, pos, clip_syn is not None and clip_syn == gold_syn)

    elapsed    = time.time() - start
    gate_result = counters_to_result(gate_counters, elapsed)
    ens_result  = counters_to_result(ens_counters,  elapsed) if include_ensemble      else None
    clip_result = counters_to_result(clip_counters, elapsed) if include_clip_baseline else None

    if verbose:
        o        = gate_result["overall"]
        gate_pct = 100 * n_gate_use / max(o["n_total"], 1)
        print(f"\n  ── {ds_name} {'─'*(38-len(ds_name))}")
        print(f"     gate     F1: {o['f1']:.4f}  |  {o['n_correct']:,}/{o['n_total']:,}  "
              f"|  gate routed: {n_gate_use:,} ({gate_pct:.1f}%)")
        if include_ensemble:
            eo = ens_result["overall"]
            print(f"     ensemble F1: {eo['f1']:.4f}  |  {eo['n_correct']:,}/{eo['n_total']:,}  "
                  f"|  bert_win: {n_bert_win:,}  clip_win: {n_clip_win:,}")
        if include_clip_baseline:
            co = clip_result["overall"]
            print(f"     clip base F1: {co['f1']:.4f}  |  {co['n_correct']:,}/{co['n_total']:,}")

    return gate_result, ens_result, clip_result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate_checkpoint",  default="checkpoints/gate/best.pt")
    parser.add_argument("--norm_stats",       default="checkpoints/gate/norm_stats.json")
    parser.add_argument("--bert_checkpoint",  default="checkpoints/bert_wsd/best")
    parser.add_argument("--fusion_checkpoint",default="checkpoints/fusion")
    parser.add_argument("--output_dir",       default="results")
    parser.add_argument("--dataset",          default=None)
    parser.add_argument("--image_condition",  default=None,
                        choices=IMAGE_CONDITIONS,
                        help="Run only this image condition (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load norm stats ───────────────────────────────────────────────────────
    with open(args.norm_stats) as f:
        norm_stats = json.load(f)
    print(f"Norm stats loaded from: {args.norm_stats}")

    # ── Load models ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading BERT from: {args.bert_checkpoint}")
    bert_encoder = BERTEncoder.load(args.bert_checkpoint, device=device)
    bert_model   = BERTWSDModel(bert_encoder, use_token_level=True)

    clip_encoder = CLIPEncoder(device=device)
    fusion       = CrossAttentionFusion(embed_dim=512, num_heads=8, dropout=0.1, ff_dim=2048).to(device)
    ckpt_path    = Path(args.fusion_checkpoint) / "best.pt"
    fusion.load_state_dict(torch.load(ckpt_path, map_location=device))
    fusion.eval()
    multi_model  = MultimodalWSDModel(clip_encoder, fusion)
    print(f"Fusion loaded from: {ckpt_path}")

    gate_model = GatingNetwork().to(device)
    gate_model.load_state_dict(torch.load(args.gate_checkpoint, map_location=device))
    gate_model.eval()
    print(f"Gate network loaded from: {args.gate_checkpoint}")

    target_datasets = {k: v for k, v in DATASETS.items()
                       if args.dataset is None or k == args.dataset}
    if not target_datasets:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        sys.exit(1)

    # ── Evaluate across all image conditions ──────────────────────────────────
    run_conditions = [args.image_condition] if args.image_condition else IMAGE_CONDITIONS
    for image_condition in run_conditions:
        include_ensemble      = image_condition in ENSEMBLE_CONDITIONS
        include_clip_baseline = (image_condition == "mixed")
        rng                   = random.Random(42) if image_condition == "mixed" else None

        label = image_condition.upper()
        if include_clip_baseline: label += "  [+ ensemble + clip baseline]"
        elif include_ensemble:    label += "  [+ ensemble]"
        print(f"\n{'='*52}")
        print(f"  IMAGE CONDITION: {label}")
        print(f"{'='*52}")

        ds_gate_results = {}
        ds_ens_results  = {} if include_ensemble      else None
        ds_clip_results = {} if include_clip_baseline else None
        start_cond      = time.time()

        for ds_name, path in target_datasets.items():
            if not Path(path).exists():
                print(f"[WARN] Not found: {path}")
                continue
            gate_result, ens_result, clip_result = evaluate_dataset(
                ds_name, path, bert_model, multi_model, gate_model,
                norm_stats, device, image_condition,
                include_ensemble=include_ensemble,
                include_clip_baseline=include_clip_baseline,
                rng=rng, verbose=True,
            )
            ds_gate_results[ds_name] = gate_result
            if include_ensemble:      ds_ens_results [ds_name] = ens_result
            if include_clip_baseline: ds_clip_results[ds_name] = clip_result

        total_time = time.time() - start_cond

        def save_record(condition_name, ds_results):
            ds_results["aggregate"] = aggregate_results(
                {k: v for k, v in ds_results.items() if k != "aggregate"}
            )
            record = {
                "condition"      : condition_name,
                "image_condition": image_condition,
                "total_time_s"   : round(total_time, 2),
                "datasets"       : ds_results,
            }
            out_path = output_dir / f"{condition_name}_{image_condition}.json"
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            agg = ds_results["aggregate"]["overall"]
            print(f"  {condition_name} | {image_condition}  —  "
                  f"F1: {agg['f1']:.4f}  |  {agg['n_total']:,} instances  |  {total_time:.1f}s")
            print(f"  Saved: {out_path}")

        save_record("learned_gate", ds_gate_results)
        if include_ensemble:      save_record("learned_gate_ensemble", ds_ens_results)
        if include_clip_baseline: save_record("clip_multimodal",       ds_clip_results)

    print(f"\nDone. Results written to: {output_dir}/")


if __name__ == "__main__":
    main()
