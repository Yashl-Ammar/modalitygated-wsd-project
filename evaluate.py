"""
evaluate.py

Entry point for running WSD evaluations.

Supported conditions:
    bert_zero_shot     — zero-shot BERT (CLS, no fine-tuning)
    bert_finetuned     — fine-tuned BERT (token-level grounding)
    clip_text          — CLIP text-only with sense prompting
    clip_multimodal    — CLIP text + image via cross-attention fusion
                         runs all three image conditions automatically:
                         helpful / misleading / irrelevant
    all_text           — runs all text-only conditions
    all                — runs everything

Usage:
    python evaluate.py --condition bert_zero_shot
    python evaluate.py --condition bert_finetuned
    python evaluate.py --condition clip_text
    python evaluate.py --condition clip_multimodal
    python evaluate.py --condition clip_multimodal --dataset semeval2013
    python evaluate.py --condition all_text
"""

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch

from encoders.bert_encoder import BERTEncoder
from encoders.clip_encoder import CLIPEncoder
from fusion.cross_attention import CrossAttentionFusion
from models.bert_model import BERTWSDModel
from models.clip_model import CLIPTextWSDModel
from models.multimodal_model import MultimodalWSDModel
from evaluation.evaluator import WSDEvaluator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ── BERT conditions ───────────────────────────────────────────────────────────

def run_bert_zero_shot(args):
    print("\n── Zero-shot BERT ───────────────────────────────")
    encoder = BERTEncoder(model_name="bert-base-uncased", device=get_device())
    model   = BERTWSDModel(encoder, use_token_level=False)
    WSDEvaluator(model, output_dir=args.output_dir).run(
        condition_name  = "bert_zero_shot",
        image_condition = None,
        datasets        = [args.dataset] if args.dataset else None,
    )


def run_bert_finetuned(args):
    print("\n── Fine-tuned BERT ──────────────────────────────")
    encoder = BERTEncoder.load(args.checkpoint, device=get_device())
    model   = BERTWSDModel(encoder, use_token_level=True)
    WSDEvaluator(model, output_dir=args.output_dir).run(
        condition_name  = "bert_finetuned",
        image_condition = None,
        datasets        = [args.dataset] if args.dataset else None,
    )


# ── CLIP text-only ────────────────────────────────────────────────────────────

def run_clip_text(args):
    print("\n── CLIP text-only ───────────────────────────────")
    encoder = CLIPEncoder(device=get_device())
    model   = CLIPTextWSDModel(encoder)
    WSDEvaluator(model, output_dir=args.output_dir).run(
        condition_name  = "clip_text",
        image_condition = None,
        datasets        = [args.dataset] if args.dataset else None,
    )


# ── CLIP multimodal ───────────────────────────────────────────────────────────

def _load_multimodal_model(args):
    """Load CLIP encoder + fusion layer."""
    device       = get_device()
    clip_encoder = CLIPEncoder(device=device)

    fusion = CrossAttentionFusion(
        embed_dim = 512,
        num_heads = 8,
        dropout   = 0.1,
        ff_dim    = 2048,
    ).to(device)

    checkpoint_path = Path(args.fusion_checkpoint) / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Fusion checkpoint not found: {checkpoint_path}\n"
            f"Run: python train_fusion.py"
        )

    fusion.load_state_dict(torch.load(checkpoint_path, map_location=device))
    fusion.eval()
    print(f"Fusion checkpoint loaded from: {checkpoint_path}")

    return MultimodalWSDModel(clip_encoder, fusion)


def run_clip_multimodal(args):
    """Run CLIP multimodal across image conditions (default: helpful/misleading/irrelevant)."""
    print("\n── CLIP multimodal ──────────────────────────────")
    model    = _load_multimodal_model(args)
    datasets = [args.dataset] if args.dataset else None

    conditions = (
        [args.image_condition]
        if args.image_condition
        else ["helpful", "misleading", "irrelevant"]
    )

    for image_condition in conditions:
        data_dir = (
            "dataset/augmented_semeval_mixed"
            if image_condition == "mixed"
            else None
        )
        print(f"\n  Image condition: {image_condition}"
              + (" (mixed dataset)" if data_dir else ""))
        evaluator = WSDEvaluator(model, output_dir=args.output_dir, data_dir=data_dir)
        evaluator.run(
            condition_name  = "clip_multimodal",
            image_condition = image_condition,
            datasets        = datasets,
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WSD Evaluation")
    parser.add_argument(
        "--condition",
        choices=[
            "bert_zero_shot",
            "bert_finetuned",
            "clip_text",
            "clip_multimodal",
            "all_text",
            "all",
        ],
        required=True,
    )
    parser.add_argument(
        "--dataset",
        choices=["senseval2", "senseval3", "semeval2007", "semeval2013", "semeval2015"],
        default=None,
    )
    parser.add_argument("--image_condition",
                        choices=["helpful", "misleading", "irrelevant", "mixed"],
                        default=None,
                        help="Override image condition for clip_multimodal "
                             "(default: run all three standard conditions)")
    parser.add_argument("--checkpoint",
                        default="checkpoints/bert_wsd/best")
    parser.add_argument("--fusion_checkpoint",
                        default="checkpoints/fusion")
    parser.add_argument("--output_dir",
                        default="results")
    args = parser.parse_args()

    set_seed()
    print(f"Device: {get_device()}")

    if args.condition == "bert_zero_shot":
        run_bert_zero_shot(args)

    elif args.condition == "bert_finetuned":
        run_bert_finetuned(args)

    elif args.condition == "clip_text":
        run_clip_text(args)

    elif args.condition == "clip_multimodal":
        run_clip_multimodal(args)

    elif args.condition == "all_text":
        run_bert_zero_shot(args)
        run_bert_finetuned(args)
        run_clip_text(args)

    elif args.condition == "all":
        run_bert_zero_shot(args)
        run_bert_finetuned(args)
        run_clip_text(args)
        run_clip_multimodal(args)


if __name__ == "__main__":
    main()