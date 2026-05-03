"""
fusion/cross_attention.py

Shallow cross-attention fusion layer for multimodal WSD.

Fuses CLIP text and image embeddings:
    query  = CLIP text embedding  (512-dim)
    keys   = CLIP image embedding (512-dim)
    values = CLIP image embedding (512-dim)

Architecture:
    - 1-layer multi-head cross-attention (8 heads, 64-dim per head)
    - Residual connection: fused = crossattn(text, image) + text
    - Layer norm after residual
    - Feed-forward projection (512 → 2048 → 512) with residual
    - Final layer norm

The residual connection is critical — it lets the model fall back to
text-only when the image is irrelevant or misleading.

Only this module is trained. CLIP encoders remain frozen throughout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Single-layer cross-attention fusion module.

    Args:
        embed_dim:   embedding dimension (512 to match CLIP)
        num_heads:   number of attention heads (8)
        dropout:     dropout rate (0.1)
        ff_dim:      feed-forward hidden dimension (2048)
    """

    def __init__(
        self,
        embed_dim = 512,
        num_heads = 8,
        dropout   = 0.1,
        ff_dim    = 2048,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # ── Cross-attention ───────────────────────────────────────────────────
        # query from text, keys/values from image
        self.cross_attn = nn.MultiheadAttention(
            embed_dim    = embed_dim,
            num_heads    = num_heads,
            dropout      = dropout,
            batch_first  = True,   # (batch, seq, dim) format
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # ── Feed-forward ──────────────────────────────────────────────────────
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, text_emb, image_emb):
        """
        Fuse text and image embeddings via cross-attention with residual.

        Args:
            text_emb:  (batch, 512) or (512,) — CLIP text embedding
            image_emb: (batch, 512) or (512,) — CLIP image embedding

        Returns:
            fused: (batch, 512) or (512,) — fused embedding, same shape as input
        """
        # handle unbatched inputs
        squeeze = False
        if text_emb.dim() == 1:
            text_emb  = text_emb.unsqueeze(0)
            image_emb = image_emb.unsqueeze(0)
            squeeze   = True

        # reshape to (batch, seq=1, dim) for MultiheadAttention
        q = text_emb.unsqueeze(1)    # (batch, 1, 512)
        k = image_emb.unsqueeze(1)   # (batch, 1, 512)
        v = image_emb.unsqueeze(1)   # (batch, 1, 512)

        # cross-attention + residual + norm
        attn_out, _ = self.cross_attn(q, k, v)   # (batch, 1, 512)
        x = self.norm1(text_emb + attn_out.squeeze(1))   # residual

        # feed-forward + residual + norm
        x = self.norm2(x + self.ff(x))

        if squeeze:
            x = x.squeeze(0)

        return x

    def fuse(self, text_emb, image_emb):
        """
        Inference-time fusion. Normalises the output to unit length
        so it's directly comparable to sense embeddings via cosine similarity.
        """
        with torch.no_grad():
            fused = self.forward(text_emb, image_emb)
        return F.normalize(fused, dim=-1)
