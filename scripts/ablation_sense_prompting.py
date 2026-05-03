"""
scripts/ablation_sense_prompting.py

Ablation: evaluate CLIP text-only and CLIP multimodal with sense prompting
disabled. Raw sentences and raw WordNet glosses are passed directly to CLIP
instead of the POS-aware caption templates.

Output:
    results/clip_text_no_prompt.json
    results/clip_multimodal_helpful_no_prompt.json
    results/clip_multimodal_irrelevant_no_prompt.json

Usage:
    python scripts/ablation_sense_prompting.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from encoders.clip_encoder import CLIPEncoder
from fusion.cross_attention import CrossAttentionFusion
from models.clip_model import CLIPTextWSDModel
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

# (condition_name, image_condition, output_filename)
ABLATION_CONDITIONS = [
    ("clip_text_no_prompt",          None,         "clip_text_no_prompt.json"),
    ("clip_multimodal_no_prompt",    "helpful",    "clip_multimodal_helpful_no_prompt.json"),
    ("clip_multimodal_no_prompt",    "irrelevant", "clip_multimodal_irrelevant_no_prompt.json"),
]


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


# ── Per-dataset evaluation ────────────────────────────────────────────────────

def evaluate_dataset(ds_name, path, model, image_condition):
    data     = load_jsonl(path)
    counters = empty_counters()
    start    = time.time()

    for item in tqdm(data, desc=f"  {ds_name}", leave=False):
        gold_syn = sensekey_to_synset(item["sense"])
        if gold_syn is None:
            counters["overall"]["skipped"] += 1
            continue

        sentence   = item["sentence"]
        word       = item["word"]
        pos        = item.get("pos")
        image_path = item.get("images", {}).get(image_condition) if image_condition else None

        if image_path is not None:
            pred = model.predict(sentence, word, pos, image_path=image_path)
        else:
            pred = model.predict(sentence, word, pos)

        pred_syn   = pred_to_synset(pred) if pred else None
        is_correct = pred_syn is not None and pred_syn == gold_syn
        update_counters(counters, pos, is_correct)

    elapsed = time.time() - start
    return counters_to_result(counters, elapsed)


# ── Full condition evaluation ─────────────────────────────────────────────────

def evaluate_condition(condition_name, image_condition, out_filename,
                       model, output_dir):
    label = condition_name + (f" [{image_condition}]" if image_condition else "")
    print(f"\n{'='*56}")
    print(f"  {label}  (no sense prompting)")
    print(f"{'='*56}")

    ds_results = {}
    start      = time.time()

    for ds_name, path in DATASETS.items():
        if not Path(path).exists():
            print(f"  [WARN] Not found: {path}")
            continue
        ds_results[ds_name] = evaluate_dataset(ds_name, path, model, image_condition)
        o = ds_results[ds_name]["overall"]
        print(f"  {ds_name:<14}  F1: {o['f1']:.4f}  "
              f"({o['n_correct']:,}/{o['n_total']:,})")

    total_time = time.time() - start
    ds_results["aggregate"] = aggregate_results(ds_results)

    record = {
        "condition"       : condition_name,
        "image_condition" : image_condition,
        "sense_prompting" : False,
        "total_time_s"    : round(total_time, 2),
        "datasets"        : ds_results,
    }
    out_path = output_dir / out_filename
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    agg_f1 = ds_results["aggregate"]["overall"]["f1"]
    print(f"\n  Aggregate F1: {agg_f1:.4f}  |  {total_time:.1f}s")
    print(f"  Saved: {out_path}")
    return agg_f1


# ── Summary table ─────────────────────────────────────────────────────────────

def load_agg_f1(path, *keys):
    try:
        d = json.load(open(path))
        for k in keys:
            d = d[k]
        return d["aggregate"]["overall"]["f1"]
    except Exception:
        return None


def fmt(v):
    return f"{v:.2%}" if v is not None else "     ?"


def print_summary(no_prompt_f1s):
    with_prompt = {
        "CLIP Text-only"             : load_agg_f1("results/clip_text_text_only.json",         "datasets"),
        "CLIP Multimodal Helpful"    : load_agg_f1("results/clip_multimodal_helpful.json",     "datasets"),
        "CLIP Multimodal Irrelevant" : load_agg_f1("results/clip_multimodal_irrelevant.json",  "datasets"),
    }

    rows = [
        ("CLIP Text-only",            "clip_text_no_prompt.json"),
        ("CLIP Multimodal Helpful",   "clip_multimodal_helpful_no_prompt.json"),
        ("CLIP Multimodal Irrelevant","clip_multimodal_irrelevant_no_prompt.json"),
    ]

    print(f"\n{'='*68}")
    print(f"  SENSE PROMPTING ABLATION")
    print(f"{'='*68}")
    print(f"  {'Condition':<28}  {'With':>8}  {'Without':>8}  {'Delta':>8}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*8}  {'─'*8}")

    for label, filename in rows:
        w = with_prompt.get(label)
        wo = no_prompt_f1s.get(filename)
        delta = (wo - w) if (w is not None and wo is not None) else None
        delta_str = f"{delta:+.2%}" if delta is not None else "     ?"
        print(f"  {label:<28}  {fmt(w):>8}  {fmt(wo):>8}  {delta_str:>8}")

    print(f"{'='*68}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Initialising CLIP (use_sense_prompting=False)...")

    clip_encoder = CLIPEncoder(device=device, use_sense_prompting=False)

    clip_text_model = CLIPTextWSDModel(clip_encoder)

    fusion = CrossAttentionFusion(embed_dim=512, num_heads=8, dropout=0.1, ff_dim=2048).to(device)
    fusion.load_state_dict(torch.load("checkpoints/fusion/best.pt", map_location=device))
    fusion.eval()
    clip_multi_model = MultimodalWSDModel(clip_encoder, fusion)
    print("Models ready.\n")

    no_prompt_f1s = {}

    for condition_name, image_condition, out_filename in ABLATION_CONDITIONS:
        model = clip_text_model if image_condition is None else clip_multi_model
        # Clear sense cache between conditions so cached embeddings don't leak
        model.sense_cache.clear()

        f1 = evaluate_condition(
            condition_name, image_condition, out_filename, model, output_dir
        )
        no_prompt_f1s[out_filename] = f1

    print_summary(no_prompt_f1s)


if __name__ == "__main__":
    main()
