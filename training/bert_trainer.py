import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from data.dataset import WSDDataset, wsd_collate_fn


class BERTTrainer:
    """
    Fine-tunes a SentenceEncoder on SemCor using a cosine similarity +
    cross-entropy objective over WordNet sense glosses.

    Training objective (per instance):
        query  = target word embedding from BERT (token-level)
        keys   = BERT embeddings of all candidate sense glosses
        scores = cosine_similarity(query, keys) / temperature
        loss   = CrossEntropy(scores, gold_sense_index)
    """

    def __init__(
        self,
        encoder,
        data,
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
        seed            = 42,
    ):
        self.encoder         = encoder
        self.device          = encoder.device
        self.output_dir      = output_dir
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.temperature     = temperature
        self.grad_accum_steps= grad_accum_steps

        os.makedirs(output_dir, exist_ok=True)

        torch.manual_seed(seed)

        # ── Dataset split ────────────────────────────────────────────────────
        print("Building dataset (sense lookup + gold resolution)...")
        full_dataset = WSDDataset(data, max_senses=max_senses)

        val_size   = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        self.train_set, self.val_set = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        print(f"Train: {train_size} | Val: {val_size}")

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=wsd_collate_fn,
            num_workers=0      # keep simple on Windows
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=wsd_collate_fn,
            num_workers=0
        )

        # ── Encoder setup ────────────────────────────────────────────────────
        encoder.freeze_bottom_layers(frozen_layers)
        trainable = sum(p.numel() for p in encoder.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in encoder.model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,}")

        # ── Optimizer + scheduler ────────────────────────────────────────────
        self.optimizer = AdamW(
            [p for p in encoder.model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01
        )

        total_steps   = (len(self.train_loader) // grad_accum_steps) * epochs
        warmup_steps  = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # ── Mixed precision ──────────────────────────────────────────────────
        self.scaler = torch.amp.GradScaler('cuda')

        print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")

    # ─────────────────────────────────────────────────────────────────────────

    def _forward_instance(self, item):
        """
        Compute cross-entropy loss for a single WSD instance.
        Returns (loss, correct: bool)
        """
        sentence = item["sentence"]
        word     = item["word"]
        glosses  = item["glosses"]
        gold_idx = item["gold_idx"]

        # query: target word contextual embedding  (hidden,)
        query = self.encoder.encode_word_in_context(sentence, word, no_grad=False)
        query = query.unsqueeze(0)   # (1, hidden)

        # keys: sense gloss embeddings  (n_senses, hidden)
        # encode_batch runs under no_grad but we need grad for fine-tuning
        inputs = self.encoder.tokenizer(
            glosses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        outputs  = self.encoder.model(**inputs)
        keys     = outputs.last_hidden_state[:, 0, :]   # CLS per gloss

        # normalise + cosine similarity
        query = F.normalize(query, dim=-1)
        keys  = F.normalize(keys,  dim=-1)
        scores = (query @ keys.T).squeeze(0) / self.temperature  # (n_senses,)

        gold_tensor = torch.tensor([gold_idx], device=self.device)
        loss    = F.cross_entropy(scores.unsqueeze(0), gold_tensor)
        correct = scores.argmax().item() == gold_idx

        return loss, correct

    # ─────────────────────────────────────────────────────────────────────────

    def _run_epoch(self, loader, train=True):
        self.encoder.model.train(train)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        step = 0

        pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)

        for batch in pbar:
            batch_loss    = torch.tensor(0.0, device=self.device)
            batch_correct = 0

            for item in batch:
                with torch.amp.autocast('cuda'):
                    loss, correct = self._forward_instance(item)
                batch_loss    = batch_loss + loss / len(batch)
                batch_correct += int(correct)

            if train:
                self.scaler.scale(batch_loss / self.grad_accum_steps).backward()
                step += 1

                if step % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.encoder.model.parameters(), max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                pass   # val: no backward

            total_loss    += batch_loss.item()
            total_correct += batch_correct
            total_samples += len(batch)

            pbar.set_postfix({
                "loss": f"{total_loss / (total_samples / self.batch_size + 1e-8):.4f}",
                "acc" : f"{total_correct / total_samples:.4f}"
            })

        avg_loss = total_loss / (total_samples / self.batch_size + 1e-8)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        best_val_acc = 0.0

        print(f"\nStarting fine-tuning for {self.epochs} epoch(s)...")
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)

            with torch.no_grad():
                val_loss, val_acc = self._run_epoch(self.val_loader, train=False)

            elapsed = time.time() - t0
            print(
                f"\nEpoch {epoch}/{self.epochs} | "
                f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f} | "
                f"Time: {elapsed:.0f}s"
            )

            # save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.encoder.save(os.path.join(self.output_dir, "best"))
                print(f"  ✓ New best val acc: {val_acc:.4f} — checkpoint saved")

            # save latest regardless
            self.encoder.save(os.path.join(self.output_dir, "latest"))

        print(f"\nFine-tuning complete. Best val acc: {best_val_acc:.4f}")
        print(f"Checkpoints saved to: {self.output_dir}")