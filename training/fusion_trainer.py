"""
training/fusion_trainer.py

Trains the CrossAttentionFusion layer on SemCor with augmented images.

Training objective:
    query  = fused embedding (crossattn(text, image) + text)
    keys   = CLIP sense gloss embeddings
    scores = cosine_similarity(query, keys) / temperature
    loss   = CrossEntropy(scores, gold_sense_index)

Only the CrossAttentionFusion parameters are trained.
Both CLIP encoders remain frozen throughout.

Training uses helpful images only — the model learns to fuse
correctly when the image is relevant. Misleading/irrelevant
conditions are evaluation-only.
"""

import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from data.loader import load_jsonl
from senses.wordnet import get_senses
from utils.wordnet import sensekey_to_synset
from nltk.corpus import wordnet as wn


class FusionTrainer:
    """
    Trains CrossAttentionFusion on SemCor + augmented images.

    Args:
        clip_encoder:   CLIPEncoder instance (frozen)
        fusion:         CrossAttentionFusion instance (trained)
        semcor_data:    list of SemCor dicts
        image_index:    dict mapping synset_id → list of image paths
        output_dir:     where to save checkpoints
    """

    def __init__(
        self,
        clip_encoder,
        fusion,
        semcor_data,
        image_index,
        output_dir      = "checkpoints/fusion",
        val_split       = 0.05,
        epochs          = 3,
        lr              = 1e-4,
        warmup_ratio    = 0.06,
        temperature     = 0.07,
        max_senses      = 20,
        seed            = 42,
    ):
        self.encoder     = clip_encoder
        self.fusion      = fusion
        self.device      = clip_encoder.device
        self.temperature = temperature
        self.epochs      = epochs
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        random.seed(seed)

        # ── Build training samples ────────────────────────────────────────────
        print("Building fusion training dataset...")
        self.train_samples, self.val_samples = self._build_dataset(
            semcor_data, image_index, max_senses, val_split, seed
        )
        print(f"Train: {len(self.train_samples):,} | Val: {len(self.val_samples):,}")

        # ── Optimiser + scheduler ─────────────────────────────────────────────
        trainable = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
        print(f"Trainable fusion params: {trainable:,}")

        self.optimizer = AdamW(fusion.parameters(), lr=lr, weight_decay=0.01)

        total_steps  = len(self.train_samples) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps  = warmup_steps,
            num_training_steps= total_steps,
        )
        self.scaler = torch.amp.GradScaler('cuda')
        print(f"Total steps: {total_steps:,} | Warmup: {warmup_steps:,}")

        # ── Sense embedding cache ─────────────────────────────────────────────
        self._sense_cache = {}

    # ─────────────────────────────────────────────────────────────────────────

    def _build_dataset(self, data, image_index, max_senses, val_split, seed):
        """
        Build training samples from SemCor.
        Each sample needs: sentence, word, pos, gold sense, and an image path.
        Only instances where the gold synset has an image are included.
        """
        rng     = random.Random(seed)
        samples = []
        skipped = 0

        for item in tqdm(data, desc="  Scanning SemCor", leave=False):
            gold_syn = sensekey_to_synset(item["sense"])
            if gold_syn is None:
                skipped += 1
                continue

            # get image for gold synset
            gold_id    = f"{gold_syn.pos()}{gold_syn.offset():08d}"
            img_paths  = image_index.get(gold_id, [])
            if not img_paths:
                skipped += 1
                continue

            # get senses and find gold index
            senses = get_senses(item["word"], item.get("pos"))
            if not senses or len(senses) > max_senses:
                skipped += 1
                continue

            sense_names = [name for name, _ in senses]
            if gold_syn.name() not in sense_names:
                skipped += 1
                continue

            gold_idx = sense_names.index(gold_syn.name())

            samples.append({
                "sentence"  : item["sentence"],
                "word"      : item["word"],
                "pos"       : item.get("pos"),
                "image_path": rng.choice(img_paths),
                "gold_idx"  : gold_idx,
                "senses"    : senses,
                "synset_key": (item["word"], item.get("pos")),
            })

        print(f"  Valid: {len(samples):,}  Skipped: {skipped:,}")

        # split
        rng.shuffle(samples)
        val_size   = int(len(samples) * val_split)
        val        = samples[:val_size]
        train      = samples[val_size:]
        return train, val

    # ─────────────────────────────────────────────────────────────────────────

    def _get_sense_embeddings(self, senses, word, pos):
        """Cache CLIP sense embeddings per (word, pos) pair."""
        key = (word, pos)
        if key in self._sense_cache:
            return self._sense_cache[key]

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

        with torch.no_grad():
            embs = self.encoder.encode_sense_batch(lemmas, glosses, poses)

        self._sense_cache[key] = embs
        return embs

    # ─────────────────────────────────────────────────────────────────────────

    def _forward(self, item):
        """
        Forward pass for one training instance.
        Returns (loss, correct: bool).
        """
        sentence   = item["sentence"]
        word       = item["word"]
        pos        = item["pos"]
        image_path = item["image_path"]
        gold_idx   = item["gold_idx"]
        senses     = item["senses"]

        # encode text (frozen)
        with torch.no_grad():
            text_emb  = self.encoder.encode_sentence(sentence, word)
            image_emb = self.encoder.encode_image(image_path)

        if image_emb is None:
            # image failed to load — skip by returning zero loss
            return None, False

        # fuse (trained)
        with torch.amp.autocast('cuda'):
            fused = self.fusion(text_emb, image_emb)   # (512,)
            fused = F.normalize(fused, dim=-1)

            # sense embeddings (frozen, cached)
            sense_embs = self._get_sense_embeddings(senses, word, pos)

            # scores
            scores = F.cosine_similarity(
                fused.unsqueeze(0).expand(len(senses), -1),
                sense_embs
            ) / self.temperature

            gold_tensor = torch.tensor([gold_idx], device=self.device)
            loss        = F.cross_entropy(scores.unsqueeze(0), gold_tensor)
            correct     = scores.argmax().item() == gold_idx

        return loss, correct

    # ─────────────────────────────────────────────────────────────────────────

    def _run_epoch(self, samples, train=True):
        self.fusion.train(train)
        if train:
            random.shuffle(samples)

        total_loss    = 0.0
        total_correct = 0
        total_valid   = 0
        step          = 0

        pbar = tqdm(samples, desc="Train" if train else "Val", leave=False)

        for item in pbar:
            loss, correct = self._forward(item)

            if loss is None:
                continue

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.fusion.parameters(), max_norm=1.0
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss    += loss.item()
            total_correct += int(correct)
            total_valid   += 1
            step          += 1

            if step % 100 == 0:
                pbar.set_postfix({
                    "loss": f"{total_loss / step:.4f}",
                    "acc" : f"{total_correct / total_valid:.4f}",
                })

        avg_loss = total_loss / max(step, 1)
        accuracy = total_correct / max(total_valid, 1)
        return avg_loss, accuracy

    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        best_val_acc = 0.0

        print(f"\nStarting fusion training for {self.epochs} epoch(s)...")
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(self.train_samples, train=True)

            with torch.no_grad():
                val_loss, val_acc = self._run_epoch(self.val_samples, train=False)

            elapsed = time.time() - t0
            print(
                f"\nEpoch {epoch}/{self.epochs} | "
                f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
                f"Time: {elapsed:.0f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save("best")
                print(f"  ✓ New best val acc: {val_acc:.4f} — saved")

            self._save("latest")

        print(f"\nFusion training complete. Best val acc: {best_val_acc:.4f}")
        print(f"Checkpoints: {self.output_dir}")

    def _save(self, tag):
        path = self.output_dir / f"{tag}.pt"
        torch.save(self.fusion.state_dict(), path)
