import torch
import torch.nn.functional as F
from senses.wordnet import get_senses


class BERTWSDModel:
    """
    WSD model using BERT embeddings and cosine similarity over WordNet sense glosses.

    Supports two encoding modes:
        use_token_level=False  — CLS embedding (zero-shot BERT baseline)
        use_token_level=True   — token-level grounding (fine-tuned BERT)

    Usage:
        # Zero-shot
        encoder = BERTEncoder(model_name="bert-base-uncased", device=device)
        model = BERTWSDModel(encoder, use_token_level=False)

        # Fine-tuned
        encoder = BERTEncoder.load("checkpoints/bert_wsd/best", device=device)
        model = BERTWSDModel(encoder, use_token_level=True)
    """

    def __init__(self, encoder, use_token_level=False):
        self.encoder       = encoder
        self.use_token_level = use_token_level
        self.sense_cache   = {}

    def _encode_sentence(self, sentence, word):
        """
        Encode the sentence using either CLS or token-level grounding
        depending on use_token_level.
        """
        if self.use_token_level:
            return self.encoder.encode_word_in_context(sentence, word, no_grad=True)
        else:
            return self.encoder.encode(sentence)

    @torch.no_grad()
    def predict(self, sentence, word, pos=None):
        """
        Predict the sense of `word` in `sentence`.

        Returns the synset name string (e.g. 'bank.n.01') or None if
        no senses are found for the word.
        """
        cache_key = (word, pos)

        if cache_key not in self.sense_cache:
            senses = get_senses(word, pos)
            if len(senses) == 0:
                return None

            glosses   = [g for _, g in senses]
            sense_embs = self.encoder.encode_batch(glosses)
            self.sense_cache[cache_key] = (senses, sense_embs)

        senses, sense_embs = self.sense_cache[cache_key]

        sent_emb = self._encode_sentence(sentence, word)
        sent_emb = sent_emb.unsqueeze(0).expand(len(senses), -1)

        sent_emb   = F.normalize(sent_emb,   dim=-1)
        sense_embs = F.normalize(sense_embs, dim=-1)

        scores   = F.cosine_similarity(sent_emb, sense_embs)
        best_idx = torch.argmax(scores).item()

        return senses[best_idx][0]

    def predict_with_scores(self, sentence, word, pos=None):
        """
        Same as predict but also returns the full softmax probability
        distribution over senses. Used by the modality gate.

        Returns:
            pred_sense: synset name string or None
            probs:      tensor of shape (n_senses,) — softmax probabilities
            senses:     list of (synset_name, gloss) tuples
        """
        cache_key = (word, pos)

        if cache_key not in self.sense_cache:
            senses = get_senses(word, pos)
            if len(senses) == 0:
                return None, None, None

            glosses    = [g for _, g in senses]
            sense_embs = self.encoder.encode_batch(glosses)
            self.sense_cache[cache_key] = (senses, sense_embs)

        senses, sense_embs = self.sense_cache[cache_key]

        sent_emb = self._encode_sentence(sentence, word)
        sent_emb = sent_emb.unsqueeze(0).expand(len(senses), -1)

        sent_emb   = F.normalize(sent_emb,   dim=-1)
        sense_embs = F.normalize(sense_embs, dim=-1)

        scores   = F.cosine_similarity(sent_emb, sense_embs)
        probs    = F.softmax(scores, dim=0)
        best_idx = torch.argmax(scores).item()

        return senses[best_idx][0], probs, senses
