import torch
from transformers import AutoTokenizer, AutoModel


class BERTEncoder:
    def __init__(self, model_name="bert-base-uncased", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        self.model.eval()

    def freeze_bottom_layers(self, num_frozen=6):
        """Freeze bottom N transformer layers to prevent catastrophic forgetting."""
        modules_to_freeze = [
            self.model.embeddings,
            *self.model.encoder.layer[:num_frozen]
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def encode_batch(self, texts):
        """Encode a batch of texts using CLS token. Used for sense gloss encoding."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128     # glosses are short
        ).to(self.device)

        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    @torch.no_grad()
    def encode(self, text):
        return self.encode_batch([text]).squeeze(0)

    def encode_word_in_context(self, sentence, word, no_grad=False):
        """
        Extract a contextual embedding for the target word by mean-pooling
        its subword tokens. Falls back to CLS if the word is not found.

        Args:
            sentence: full input sentence
            word: the target ambiguous word (must appear in sentence)
            no_grad: if True, runs under torch.no_grad() (inference mode)

        Returns:
            Tensor of shape (hidden_size,)
        """
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            offset_mapping = inputs.pop("offset_mapping")[0]   # (seq_len, 2)

            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state[0]              # (seq_len, hidden)

            # locate target word's character span (case-insensitive)
            sent_lower = sentence.lower()
            word_lower = word.lower()
            char_start = sent_lower.find(word_lower)

            if char_start == -1:
                # word not found verbatim — fall back to CLS
                return hidden[0]

            char_end = char_start + len(word)

            # collect subword token indices that fall inside the char span
            token_indices = [
                i for i, (s, e) in enumerate(offset_mapping.tolist())
                if s >= char_start and e <= char_end and e > s
            ]

            if not token_indices:
                return hidden[0]

            # mean-pool subword tokens
            return hidden[token_indices].mean(dim=0)

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Encoder saved to {path}")

    @classmethod
    def load(cls, path, device=None):
        instance = cls.__new__(cls)
        instance.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        instance.model_name = path
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModel.from_pretrained(path).to(instance.device)
        instance.model.eval()
        return instance