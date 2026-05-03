"""
models/gate_model.py

Learned gating network for multimodal WSD.

Inputs (8-dim):
    4 scalar features (z-score normalised):
        max_prob, margin, p1_p2_ratio, entropy
    4 POS one-hot:
        NOUN, VERB, ADJ, ADV  (all-zero for unknown POS)

Output:
    scalar logit (apply sigmoid + threshold 0.5 for binary decision)
"""

import torch
import torch.nn as nn

POS_ORDER = ["NOUN", "VERB", "ADJ", "ADV"]
SCALAR_FEATURES = ["max_prob", "margin", "p1_p2_ratio", "entropy"]


class GatingNetwork(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # raw logits, shape (batch,)


def pos_to_onehot(pos: str) -> list:
    """Return a 4-element one-hot list for NOUN/VERB/ADJ/ADV, all-zero otherwise."""
    vec = [0.0] * len(POS_ORDER)
    key = (pos or "").upper()
    if key in POS_ORDER:
        vec[POS_ORDER.index(key)] = 1.0
    return vec


def build_feature_vector(max_prob, margin, p1_p2_ratio, entropy, pos, norm_stats):
    """
    Build a normalised 8-dim feature tensor for a single instance.

    norm_stats: {"mean": {feat: val}, "std": {feat: val}}
    """
    mean = norm_stats["mean"]
    std  = norm_stats["std"]

    scalars = [
        (max_prob    - mean["max_prob"])    / (std["max_prob"]    + 1e-8),
        (margin      - mean["margin"])      / (std["margin"]      + 1e-8),
        (p1_p2_ratio - mean["p1_p2_ratio"]) / (std["p1_p2_ratio"] + 1e-8),
        (entropy     - mean["entropy"])     / (std["entropy"]     + 1e-8),
    ]
    onehot = pos_to_onehot(pos)
    return torch.tensor(scalars + onehot, dtype=torch.float32)
