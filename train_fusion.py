"""
train_fusion.py

Entry point for training the cross-attention fusion layer.

Usage:
    python train_fusion.py

    # Quick sanity check with small subset
    python train_fusion.py --max_samples 500
"""

import argparse
import json
import random
import numpy as np
import torch
from pathlib import Path

from encoders.clip_encoder import CLIPEncoder
from fusion.cross_attention import CrossAttentionFusion
from training.fusion_trainer import FusionTrainer
from data.loader import load_jsonl


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training data (for sanity checks)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", default="checkpoints/fusion")
    args = parser.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading SemCor...")
    data = load_jsonl("dataset/semcor/SemCor.jsonl")
    if args.max_samples:
        data = data[:args.max_samples]
        print(f"  Limited to {args.max_samples:,} samples (sanity check mode)")
    else:
        print(f"  Loaded {len(data):,} instances")

    print("Loading image index...")
    with open("data/image_index.json") as f:
        image_index = json.load(f)
    print(f"  {len(image_index):,} synsets with images")

    # ── Models ────────────────────────────────────────────────────────────────
    print("\nLoading CLIP encoder...")
    clip_encoder = CLIPEncoder(device=device)

    print("Initialising CrossAttentionFusion...")
    fusion = CrossAttentionFusion(
        embed_dim = 512,
        num_heads = 8,
        dropout   = 0.1,
        ff_dim    = 2048,
    ).to(device)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = FusionTrainer(
        clip_encoder = clip_encoder,
        fusion       = fusion,
        semcor_data  = data,
        image_index  = image_index,
        output_dir   = args.output_dir,
        epochs       = args.epochs,
        lr           = args.lr,
    )

    trainer.train()


if __name__ == "__main__":
    main()
