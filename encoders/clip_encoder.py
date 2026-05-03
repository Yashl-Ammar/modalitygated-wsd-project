"""
encoders/clip_encoder.py

CLIP encoder for multimodal WSD.

Provides:
    - Text encoding with sense prompting (sentence context + sense glosses)
    - Image encoding from file paths
    - Shared embedding space for text-image similarity

Model: openai/clip-vit-base-patch32
    - 77 token context window
    - 512-dimensional embedding space
    - Text and image embeddings are aligned by contrastive training

Sense prompting:
    Sentences are reformatted as captions CLIP understands:
        "a photo of {word} in context: {sentence}"

    Sense glosses are reformatted to match:
        noun  → "a photo of {lemma}, {gloss}"
        verb  → "a photo of a person {lemma}ing, {gloss}"
        adj   → "a photo of something {lemma}, {gloss}"
        adv   → "a photo of an action performed {lemma}, {gloss}"
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


MODEL_ID = "openai/clip-vit-base-patch32"


class CLIPEncoder:
    def __init__(self, model_name=MODEL_ID, device=None, use_sense_prompting=True):
        self.device              = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name          = model_name
        self.use_sense_prompting = use_sense_prompting

        print(f"Loading CLIP: {model_name}...")
        self.model     = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print("CLIP ready.")

    # ── Sentence encoding ─────────────────────────────────────────────────────

    def encode_sentence(self, sentence, word):
        """
        Encode a sentence with the target word foregrounded.
        Uses sense prompting to match CLIP's training distribution.

        Format: "a photo of {word} in context: {sentence}"
        When use_sense_prompting=False: passes the raw sentence directly.

        Args:
            sentence: full input sentence
            word:     target ambiguous word

        Returns:
            Tensor of shape (512,)
        """
        if self.use_sense_prompting:
            text = self._sentence_prompt(sentence, word)
        else:
            text = sentence.strip()
        return self._encode_text(text)

    def encode_sentence_batch(self, sentences, words):
        """
        Batch version of encode_sentence.

        Returns:
            Tensor of shape (batch, 512)
        """
        if self.use_sense_prompting:
            texts = [self._sentence_prompt(s, w) for s, w in zip(sentences, words)]
        else:
            texts = [s.strip() for s in sentences]
        return self._encode_text_batch(texts)

    # ── Sense encoding ────────────────────────────────────────────────────────

    def encode_sense(self, lemma, gloss, pos):
        """
        Encode a WordNet sense using sense prompting.
        When use_sense_prompting=False: passes the raw gloss directly.

        Args:
            lemma: primary lemma name (e.g. 'bank')
            gloss: WordNet definition string
            pos:   WordNet POS ('n', 'v', 'a', 's', 'r')

        Returns:
            Tensor of shape (512,)
        """
        if self.use_sense_prompting:
            text = self._sense_prompt(lemma, gloss, pos)
        else:
            text = gloss.strip()
        return self._encode_text(text)

    def encode_sense_batch(self, lemmas, glosses, poses):
        """
        Batch version of encode_sense.

        Returns:
            Tensor of shape (batch, 512)
        """
        if self.use_sense_prompting:
            texts = [self._sense_prompt(l, g, p) for l, g, p in zip(lemmas, glosses, poses)]
        else:
            texts = [g.strip() for g in glosses]
        return self._encode_text_batch(texts)

    # ── Image encoding ────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, image_path):
        """
        Encode an image from a file path.

        Args:
            image_path: str or Path to a JPEG/PNG image

        Returns:
            Tensor of shape (512,) or None if image loading fails
        """
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to load image {image_path}: {e}")
            return None

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.image_embeds if hasattr(features, "image_embeds") else features.pooler_output
        return F.normalize(features.squeeze(0), dim=-1)

    @torch.no_grad()
    def encode_image_batch(self, image_paths):
        """
        Encode a batch of images. Skips failed loads and returns
        a tensor of shape (n_successful, 512) plus a list of
        successfully loaded indices.
        """
        images  = []
        indices = []

        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                indices.append(i)
            except Exception:
                continue

        if not images:
            return None, []

        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        features = self.model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.image_embeds if hasattr(features, "image_embeds") else features.pooler_output
        return F.normalize(features, dim=-1), indices

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _sentence_prompt(self, sentence, word):
        """
        Convert a WSD sentence into a CLIP-friendly caption.
        Truncates sentence to 200 chars to stay within CLIP's 77 token limit.
        """
        sentence = sentence.strip()
        if len(sentence) > 200:
            sentence = sentence[:200]
        return f"a photo of {word} in context: {sentence}"

    def _sense_prompt(self, lemma, gloss, pos):
        """
        Convert a WordNet sense into a CLIP-friendly caption.
        POS-aware templates match the style used for SD image generation
        so text and image embeddings are stylistically aligned.
        """
        lemma = lemma.replace("_", " ")
        gloss = gloss.strip()

        if pos == "n":
            return f"a photo of {lemma}, {gloss}"

        elif pos == "v":
            root = lemma[:-1] if lemma.endswith("e") else lemma
            return f"a photo of a person {root}ing, {gloss}"

        elif pos in ("a", "s"):
            return f"a photo of something {lemma}, {gloss}"

        else:   # adverb
            return f"a photo of an action performed {lemma}, {gloss}"

    # ── Internal text encoding ────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_text(self, text):
        """Encode a single text string. Returns normalised (512,) tensor."""
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        features = self.model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.text_embeds if hasattr(features, "text_embeds") else features.pooler_output
        return F.normalize(features.squeeze(0), dim=-1)

    @torch.no_grad()
    def _encode_text_batch(self, texts):
        """Encode a list of texts. Returns normalised (batch, 512) tensor."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        features = self.model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.text_embeds if hasattr(features, "text_embeds") else features.pooler_output
        return F.normalize(features, dim=-1)