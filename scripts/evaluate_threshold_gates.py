"""
scripts/evaluate_threshold_gates.py

Evaluate confidence-gate and entropy-gate WSD on all five SemEval benchmarks
across three image conditions (helpful / misleading / irrelevant).

For each instance:
    1. Run fine-tuned BERT → softmax probs → max_prob, entropy
    2. Confidence gate: if max_prob < conf_t  → multimodal, else BERT
    3. Entropy gate:    if entropy  > ent_t   → multimodal, else BERT

Both gates are evaluated in a single forward-pass loop per condition so BERT
runs exactly once per instance.

Output (6 files):
    results/confidence_gate_{helpful,misleading,irrelevant}.json
    results/entropy_gate_{helpful,misleading,irrelevant}.json

Usage:
    python scripts/evaluate_threshold_gates.py
    python scripts/evaluate_threshold_gates.py --dataset semeval2013  # single dataset
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from collections import defaultdict
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


DATASETS = {
    "senseval2"  : "dataset/augmented_semeval/senseval2.jsonl",
    "senseval3"  : "dataset/augmented_semeval/senseval3.jsonl",
    "semeval2007": "dataset/augmented_semeval/semeval2007.jsonl",
    "semeval2013": "dataset/augmented_semeval/semeval2013.jsonl",
    "semeval2015": "dataset/augmented_semeval/semeval2015.jsonl",
}

IMAGE_CONDITIONS = ["helpful", "misleading", "irrelevant"]


# ── Feature extraction (mirrors generate_gate_labels.py) ─────────────────────

def extract_gate_features(probs: torch.Tensor):
    p        = probs.cpu().float()
    max_prob = float(p.max())
    entropy  = -float(torch.sum(p * torch.log(p + 1e-10)))
    return max_prob, entropy


# ── Counters helpers ──────────────────────────────────────────────────────────

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


# ── Per-dataset evaluation (both gates in one pass) ───────────────────────────

def evaluate_dataset(ds_name, path, bert_model, multi_model, conf_t, ent_t,
                     image_condition, verbose):
    data = load_jsonl(path)

    conf_counters = empty_counters()
    ent_counters  = empty_counters()
    start         = time.time()

    for item in tqdm(data, desc=f"  {ds_name}", leave=False):
        gold_syn = sensekey_to_synset(item["sense"])
        if gold_syn is None:
            conf_counters["overall"]["skipped"] += 1
            ent_counters ["overall"]["skipped"] += 1
            continue

        sentence   = item["sentence"]
        word       = item["word"]
        pos        = item.get("pos")
        image_path = item.get("images", {}).get(image_condition)

        # ── BERT inference (one forward pass) ─────────────────────────────
        pred_bert, probs, _ = bert_model.predict_with_scores(sentence, word, pos)

        if probs is not None:
            max_prob, entropy = extract_gate_features(probs)
        else:
            # no senses found — both gates default to BERT (also None → miss)
            max_prob, entropy = 1.0, 0.0

        # ── Gate decisions ─────────────────────────────────────────────────
        use_multi_conf = (probs is not None) and (max_prob < conf_t) and (image_path is not None)
        use_multi_ent  = (probs is not None) and (entropy  > ent_t)  and (image_path is not None)

        # Call multimodal at most once per instance
        pred_multi = None
        if use_multi_conf or use_multi_ent:
            pred_multi = multi_model.predict(sentence, word, pos, image_path=image_path)

        pred_conf = pred_multi if use_multi_conf else pred_bert
        pred_ent  = pred_multi if use_multi_ent  else pred_bert

        # ── Correctness ────────────────────────────────────────────────────
        conf_syn = pred_to_synset(pred_conf) if pred_conf else None
        ent_syn  = pred_to_synset(pred_ent)  if pred_ent  else None

        update_counters(conf_counters, pos, conf_syn is not None and conf_syn == gold_syn)
        update_counters(ent_counters,  pos, ent_syn  is not None and ent_syn  == gold_syn)

    elapsed = time.time() - start
    conf_result = counters_to_result(conf_counters, elapsed)
    ent_result  = counters_to_result(ent_counters,  elapsed)

    if verbose:
        for gate_name, result in [("confidence_gate", conf_result), ("entropy_gate", ent_result)]:
            o = result["overall"]
            print(f"\n  ── {ds_name} [{gate_name}] {'─'*(28-len(ds_name))}")
            print(f"     F1: {o['f1']:.4f}  |  {o['n_correct']:,}/{o['n_total']:,}  "
                  f"|  {result['speed']:.1f} samp/s")

    return conf_result, ent_result


# ── Save result file ──────────────────────────────────────────────────────────

def save_result(condition_name, image_condition, datasets_results, total_time, output_dir):
    record = {
        "condition"      : condition_name,
        "image_condition": image_condition,
        "total_time_s"   : round(total_time, 2),
        "datasets"       : datasets_results,
    }
    out_path = output_dir / f"{condition_name}_{image_condition}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Saved: {out_path}")
    return record


def print_summary(condition_name, image_condition, record):
    agg = record["datasets"].get("aggregate", {}).get("overall", {})
    print(f"\n{'='*52}")
    print(f"  CONDITION : {condition_name}")
    print(f"  IMAGE     : {image_condition}")
    print(f"  Aggregate F1 : {agg.get('f1', 0):.4f}")
    print(f"  Total        : {agg.get('n_total', 0):,} instances")
    print(f"  Time         : {record['total_time_s']:.1f}s")
    print(f"{'='*52}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds",       default="results/gate_thresholds.json")
    parser.add_argument("--bert_checkpoint",  default="checkpoints/bert_wsd/best")
    parser.add_argument("--fusion_checkpoint",default="checkpoints/fusion")
    parser.add_argument("--output_dir",       default="results")
    parser.add_argument("--dataset",          default=None,
                        help="Evaluate on a single dataset (e.g. semeval2013)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load thresholds ───────────────────────────────────────────────────────
    with open(args.thresholds) as f:
        thresholds = json.load(f)

    conf_t = thresholds["confidence_threshold"]
    ent_t  = thresholds["entropy_threshold"]
    print(f"Thresholds loaded:")
    print(f"  Confidence threshold : {conf_t}  (F1={thresholds['confidence_best_f1']})")
    print(f"  Entropy    threshold : {ent_t}   (F1={thresholds['entropy_best_f1']})")

    # ── Load models ───────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print(f"Loading BERT from: {args.bert_checkpoint}")
    bert_encoder = BERTEncoder.load(args.bert_checkpoint, device=device)
    bert_model   = BERTWSDModel(bert_encoder, use_token_level=True)
    print("BERT ready.")

    clip_encoder = CLIPEncoder(device=device)
    fusion = CrossAttentionFusion(embed_dim=512, num_heads=8, dropout=0.1, ff_dim=2048).to(device)
    ckpt_path = Path(args.fusion_checkpoint) / "best.pt"
    fusion.load_state_dict(torch.load(ckpt_path, map_location=device))
    fusion.eval()
    multi_model = MultimodalWSDModel(clip_encoder, fusion)
    print(f"Fusion checkpoint loaded from: {ckpt_path}")

    target_datasets = {k: v for k, v in DATASETS.items()
                       if args.dataset is None or k == args.dataset}
    if not target_datasets:
        print(f"[ERROR] Unknown dataset: {args.dataset}")
        sys.exit(1)

    # ── Evaluate across all image conditions ──────────────────────────────────
    for image_condition in IMAGE_CONDITIONS:
        print(f"\n{'='*52}")
        print(f"  IMAGE CONDITION: {image_condition.upper()}")
        print(f"{'='*52}")

        conf_ds_results = {}
        ent_ds_results  = {}
        start_cond      = time.time()

        for ds_name, path in target_datasets.items():
            if not Path(path).exists():
                print(f"[WARN] Not found: {path}")
                continue
            conf_result, ent_result = evaluate_dataset(
                ds_name, path, bert_model, multi_model,
                conf_t, ent_t, image_condition, verbose=True,
            )
            conf_ds_results[ds_name] = conf_result
            ent_ds_results[ds_name]  = ent_result

        conf_ds_results["aggregate"] = aggregate_results(conf_ds_results)
        ent_ds_results ["aggregate"] = aggregate_results(ent_ds_results)

        total_time = time.time() - start_cond

        conf_record = save_result(
            "confidence_gate", image_condition, conf_ds_results, total_time, output_dir
        )
        ent_record = save_result(
            "entropy_gate", image_condition, ent_ds_results, total_time, output_dir
        )

        print_summary("confidence_gate", image_condition, conf_record)
        print_summary("entropy_gate",    image_condition, ent_record)

    print(f"\nDone. Results written to: {output_dir}/")


if __name__ == "__main__":
    main()
