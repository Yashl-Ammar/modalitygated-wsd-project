"""
scripts/generate_gate_labels.py

For each SemCor instance, runs fine-tuned BERT and CLIP multimodal, compares
predictions against the gold synset, and emits gate training labels.

Gate label rule:
    gate_label = 1 if multimodal correct AND BERT not correct, else 0
    (ties always → 0: prefer text-only when equal)

Features extracted from BERT softmax probabilities:
    max_prob    — max softmax probability
    margin      — p1 - p2
    p1_p2_ratio — p1 / p2  (capped at 999 when p2 == 0)
    entropy     — -sum(p * log(p + 1e-10))

Usage:
    python scripts/generate_gate_labels.py
    python scripts/generate_gate_labels.py --max_samples 100  # quick test
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from encoders.bert_encoder import BERTEncoder
from encoders.clip_encoder import CLIPEncoder
from fusion.cross_attention import CrossAttentionFusion
from models.bert_model import BERTWSDModel
from models.multimodal_model import MultimodalWSDModel
from utils.wordnet import sensekey_to_synset, pred_to_synset


def synset_to_id(synset) -> str:
    return f"{synset.pos()}{synset.offset():08d}"


def extract_features(probs: torch.Tensor) -> dict:
    p = probs.cpu().float()
    sorted_p, _ = torch.sort(p, descending=True)

    p1 = sorted_p[0].item()
    p2 = sorted_p[1].item() if len(sorted_p) > 1 else 0.0

    return {
        "max_prob"    : round(p1, 6),
        "margin"      : round(p1 - p2, 6),
        "p1_p2_ratio" : round(min(p1 / p2, 999.0) if p2 > 0.0 else 999.0, 6),
        "entropy"     : round(-float(torch.sum(p * torch.log(p + 1e-10)).item()), 6),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate gate training labels from SemCor."
    )
    parser.add_argument("--semcor",             default="dataset/semcor/SemCor.jsonl")
    parser.add_argument("--image_index",        default="data/image_index.json")
    parser.add_argument("--bert_checkpoint",    default="checkpoints/bert_wsd/best")
    parser.add_argument("--fusion_checkpoint",  default="checkpoints/fusion")
    parser.add_argument("--output",             default="dataset/gate_labels/semcor_gate_labels.jsonl")
    parser.add_argument("--max_samples",        type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────

    print(f"Loading SemCor from: {args.semcor}")
    with open(args.semcor) as f:
        data = [json.loads(line) for line in f]
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"  Limited to {args.max_samples:,} samples")
    else:
        print(f"  {len(data):,} instances")

    print(f"Loading image index from: {args.image_index}")
    with open(args.image_index) as f:
        image_index = json.load(f)
    print(f"  {len(image_index):,} synsets with images")

    # ── Load models ────────────────────────────────────────────────────────────

    print(f"\nLoading BERT from: {args.bert_checkpoint}")
    bert_model = BERTWSDModel(
        BERTEncoder.load(args.bert_checkpoint, device=device),
        use_token_level=True,
    )
    print("BERT ready.")

    clip_encoder = CLIPEncoder(device=device)
    fusion = CrossAttentionFusion(embed_dim=512, num_heads=8, dropout=0.1, ff_dim=2048).to(device)
    ckpt_path = Path(args.fusion_checkpoint) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint not found: {ckpt_path}\nRun: python train_fusion.py")
    fusion.load_state_dict(torch.load(ckpt_path, map_location=device))
    fusion.eval()
    print(f"Fusion checkpoint loaded from: {ckpt_path}")
    multi_model = MultimodalWSDModel(clip_encoder, fusion)

    # ── Generate labels ────────────────────────────────────────────────────────

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_label_1 = 0
    skip_reasons = {"invalid_sense_key": 0, "no_image": 0, "no_bert_probs": 0}

    with open(out_path, "w") as out_f:
        for item in tqdm(data, desc="Generating gate labels"):

            gold_syn = sensekey_to_synset(item["sense"])
            if gold_syn is None:
                skip_reasons["invalid_sense_key"] += 1
                continue

            img_list = image_index.get(synset_to_id(gold_syn), [])
            if not img_list:
                skip_reasons["no_image"] += 1
                continue
            image_path = img_list[0]

            sentence = item["sentence"]
            word     = item["word"]
            pos      = item.get("pos")

            pred_bert, probs, _ = bert_model.predict_with_scores(sentence, word, pos)
            if probs is None:
                skip_reasons["no_bert_probs"] += 1
                continue

            pred_multi = multi_model.predict(sentence, word, pos, image_path=image_path)

            bert_syn  = pred_to_synset(pred_bert)  if pred_bert  else None
            multi_syn = pred_to_synset(pred_multi) if pred_multi else None

            bert_correct  = int(bert_syn  is not None and bert_syn  == gold_syn)
            multi_correct = int(multi_syn is not None and multi_syn == gold_syn)
            gate_label    = 1 if multi_correct > bert_correct else 0

            record = {
                **item,
                "synset"       : gold_syn.name(),
                "gate_label"   : gate_label,
                "bert_correct" : bert_correct,
                "multi_correct": multi_correct,
                **extract_features(probs),
            }
            out_f.write(json.dumps(record) + "\n")
            n_written += 1
            n_label_1 += gate_label

    # ── Summary ────────────────────────────────────────────────────────────────

    n_skipped  = sum(skip_reasons.values())
    n_label_0  = n_written - n_label_1
    pct1 = 100.0 * n_label_1 / n_written if n_written else 0.0

    print(f"\n{'='*52}")
    print(f"  Processed : {n_written + n_skipped:>8,}")
    print(f"  Written   : {n_written:>8,}")
    print(f"  Skipped   : {n_skipped:>8,}")
    print(f"    invalid sense key : {skip_reasons['invalid_sense_key']:>6,}")
    print(f"    no image          : {skip_reasons['no_image']:>6,}")
    print(f"    no BERT probs     : {skip_reasons['no_bert_probs']:>6,}")
    print(f"\n  Label distribution:")
    print(f"    label=1 (use image) : {n_label_1:>7,}  ({pct1:.1f}%)")
    print(f"    label=0 (text-only) : {n_label_0:>7,}  ({100-pct1:.1f}%)")
    print(f"\n  Output: {out_path}")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
