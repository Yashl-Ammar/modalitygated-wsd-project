"""
models/multimodal_model.py

Multimodal WSD model using CLIP text + image via cross-attention fusion.

Pipeline:
    1. Encode sentence → CLIP text embedding
    2. Encode image    → CLIP image embedding
    3. Fuse via cross-attention + residual
    4. Compare fused embedding against sense embeddings
    5. Return highest-scoring sense

Falls back to text-only if image path is None or image fails to load.
"""

import torch
import torch.nn.functional as F

from nltk.corpus import wordnet as wn
from senses.wordnet import get_senses


class MultimodalWSDModel:
    """
    CLIP text + image WSD model with cross-attention fusion.

    Args:
        clip_encoder: CLIPEncoder (frozen)
        fusion:       CrossAttentionFusion (trained)
    """

    def __init__(self, clip_encoder, fusion):
        self.encoder     = clip_encoder
        self.fusion      = fusion
        self.sense_cache = {}

    def _get_sense_embeddings(self, word, pos, senses):
        """Cache CLIP sense embeddings per (word, pos) pair."""
        key = (word, pos)
        if key in self.sense_cache:
            return self.sense_cache[key]

        lemmas  = []
        glosses = []
        poses   = []
        for sense_name, gloss in senses:
            try:
                syn   = wn.synset(sense_name)
                lemma = syn.lemma_names()[0]
                wpos  = syn.pos()
            except Exception:
                lemma = word
                wpos  = "n"
            lemmas.append(lemma)
            glosses.append(gloss)
            poses.append(wpos)

        embs = self.encoder.encode_sense_batch(lemmas, glosses, poses)
        self.sense_cache[key] = embs
        return embs

    @torch.no_grad()
    def predict(self, sentence, word, pos=None, image_path=None):
        """
        Predict sense. Uses image if provided, falls back to text-only if not.

        Returns synset name string or None.
        """
        senses = get_senses(word, pos)
        if not senses:
            return None

        sense_embs = self._get_sense_embeddings(word, pos, senses)
        text_emb   = self.encoder.encode_sentence(sentence, word)

        # try to fuse with image
        if image_path is not None:
            image_emb = self.encoder.encode_image(image_path)
            if image_emb is not None:
                fused = self.fusion.fuse(text_emb, image_emb)
            else:
                fused = F.normalize(text_emb, dim=-1)
        else:
            fused = F.normalize(text_emb, dim=-1)

        scores   = F.cosine_similarity(
            fused.unsqueeze(0).expand(len(senses), -1),
            sense_embs
        )
        best_idx = torch.argmax(scores).item()
        return senses[best_idx][0]

    @torch.no_grad()
    def predict_with_max_similarity(self, sentence, word, pos=None, image_path=None):
        """
        Same as predict but also returns the highest cosine similarity score
        across all candidate senses after fusion. Used by the ensemble gate to
        compare CLIP confidence against BERT max_prob.

        Returns:
            pred_sense:     synset name string or None
            max_similarity: float — highest raw cosine similarity score
        """
        senses = get_senses(word, pos)
        if not senses:
            return None, 0.0

        sense_embs = self._get_sense_embeddings(word, pos, senses)
        text_emb   = self.encoder.encode_sentence(sentence, word)

        if image_path is not None:
            image_emb = self.encoder.encode_image(image_path)
            if image_emb is not None:
                fused = self.fusion.fuse(text_emb, image_emb)
            else:
                fused = F.normalize(text_emb, dim=-1)
        else:
            fused = F.normalize(text_emb, dim=-1)

        scores   = F.cosine_similarity(
            fused.unsqueeze(0).expand(len(senses), -1),
            sense_embs
        )
        best_idx = torch.argmax(scores).item()
        return senses[best_idx][0], scores.max().item()

    @torch.no_grad()
    def predict_with_scores(self, sentence, word, pos=None, image_path=None):
        """
        Returns (pred_sense, probs, senses) for use by the modality gate.
        """
        senses = get_senses(word, pos)
        if not senses:
            return None, None, None

        sense_embs = self._get_sense_embeddings(word, pos, senses)
        text_emb   = self.encoder.encode_sentence(sentence, word)

        if image_path is not None:
            image_emb = self.encoder.encode_image(image_path)
            if image_emb is not None:
                fused = self.fusion.fuse(text_emb, image_emb)
            else:
                fused = F.normalize(text_emb, dim=-1)
        else:
            fused = F.normalize(text_emb, dim=-1)

        scores   = F.cosine_similarity(
            fused.unsqueeze(0).expand(len(senses), -1),
            sense_embs
        )
        probs    = F.softmax(scores, dim=0)
        best_idx = torch.argmax(scores).item()
        return senses[best_idx][0], probs, senses
