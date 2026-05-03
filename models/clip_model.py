"""
models/clip_model.py

CLIP-based WSD model using sense prompting.

Two variants:
    CLIPTextWSDModel    — text-only, uses sentence + sense prompts
    CLIPMultimodalWSDModel — text + image, fuses via cross-attention (built later)

CLIPTextWSDModel usage:
    encoder = CLIPEncoder()
    model   = CLIPTextWSDModel(encoder)
    pred    = model.predict(sentence, word, pos)
"""

import torch
import torch.nn.functional as F

from nltk.corpus import wordnet as wn
from senses.wordnet import get_senses


class CLIPTextWSDModel:
    """
    Text-only WSD using CLIP's text encoder with sense prompting.

    For each instance:
        1. Encode sentence as "a photo of {word} in context: {sentence}"
        2. Encode each sense gloss using POS-aware sense prompts
        3. Predict via cosine similarity

    Both sentence and sense embeddings live in CLIP's shared text space.
    Sense embeddings are cached per (word, pos) pair.
    """

    def __init__(self, encoder):
        self.encoder     = encoder
        self.sense_cache = {}   # (word, pos) → (senses, sense_embs)

    def _get_sense_embeddings(self, word, pos):
        """
        Look up or compute sense embeddings for a word/POS pair.
        Caches results to avoid redundant CLIP calls.
        """
        cache_key = (word, pos)
        if cache_key in self.sense_cache:
            return self.sense_cache[cache_key]

        senses = get_senses(word, pos)
        if not senses:
            return None, None

        # build sense prompts
        lemmas = []
        glosses = []
        poses  = []

        for sense_name, gloss in senses:
            try:
                syn = wn.synset(sense_name)
                lemma = syn.lemma_names()[0]
                wn_pos = syn.pos()
            except Exception:
                lemma = word
                wn_pos = pos[0].lower() if pos else "n"

            lemmas.append(lemma)
            glosses.append(gloss)
            poses.append(wn_pos)

        sense_embs = self.encoder.encode_sense_batch(lemmas, glosses, poses)
        self.sense_cache[cache_key] = (senses, sense_embs)
        return senses, sense_embs

    @torch.no_grad()
    def predict(self, sentence, word, pos=None, image_path=None):
        """
        Predict the sense of `word` in `sentence`.
        image_path is accepted but ignored (text-only model).

        Returns synset name string or None.
        """
        senses, sense_embs = self._get_sense_embeddings(word, pos)
        if senses is None:
            return None

        sent_emb = self.encoder.encode_sentence(sentence, word)
        sent_emb = sent_emb.unsqueeze(0).expand(len(senses), -1)

        scores   = F.cosine_similarity(sent_emb, sense_embs)
        best_idx = torch.argmax(scores).item()
        return senses[best_idx][0]

    @torch.no_grad()
    def predict_with_scores(self, sentence, word, pos=None, image_path=None):
        """
        Returns (pred_sense, probs, senses) for use by the modality gate.
        """
        senses, sense_embs = self._get_sense_embeddings(word, pos)
        if senses is None:
            return None, None, None

        sent_emb = self.encoder.encode_sentence(sentence, word)
        sent_emb = sent_emb.unsqueeze(0).expand(len(senses), -1)

        scores   = F.cosine_similarity(sent_emb, sense_embs)
        probs    = F.softmax(scores, dim=0)
        best_idx = torch.argmax(scores).item()
        return senses[best_idx][0], probs, senses
