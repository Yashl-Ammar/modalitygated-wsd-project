import random
import numpy as np
import torch

from data.loader import load_jsonl
from encoders.bert_encoder import BERTEncoder
from training.bert_trainer import BERTTrainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading SemCor...")
    data = load_jsonl("dataset/semcor/SemCor.jsonl")
    print(f"Loaded {len(data)} instances")

    # ── Encoder ──────────────────────────────────────────────────────────────
    encoder = BERTEncoder(model_name="bert-base-uncased", device=device)

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = BERTTrainer(
        encoder         = encoder,
        data            = data,
        output_dir      = "checkpoints/bert_wsd",
        val_split       = 0.05,
        epochs          = 3,
        batch_size      = 16,
        lr              = 2e-5,
        warmup_ratio    = 0.06,
        temperature     = 0.1,
        frozen_layers   = 6,
        grad_accum_steps= 1,
        max_senses      = 20,
    )

    trainer.train()


if __name__ == "__main__":
    main()