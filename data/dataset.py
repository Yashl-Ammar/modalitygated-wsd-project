from torch.utils.data import Dataset
from senses.wordnet import get_senses
from utils.wordnet import sensekey_to_synset


class WSDDataset(Dataset):
    """
    PyTorch Dataset for WSD fine-tuning on SemCor.

    Each item returns the raw data needed for the training loop.
    Sense lookups are cached at construction time to avoid repeated
    WordNet calls during training.
    """

    def __init__(self, data, max_senses=20):
        """
        Args:
            data: list of dicts with keys: sentence, word, pos, sense
            max_senses: discard instances where the word has more than
                        this many senses (rare edge cases, keeps batching clean)
        """
        self.samples = []
        self._sense_cache = {}

        skipped = 0
        for item in data:
            word = item["word"]
            pos  = item.get("pos")
            key  = (word, pos)

            if key not in self._sense_cache:
                senses = get_senses(word, pos)
                self._sense_cache[key] = senses

            senses = self._sense_cache[key]

            if len(senses) == 0:
                skipped += 1
                continue

            # find the gold sense index in the candidate list
            gold_synset_name = item["sense"]   # this is a sense key — resolved later
            sense_names = [name for name, _ in senses]
            glosses     = [gloss for _, gloss in senses]

            # resolve gold sense key → synset name
            
            gold_syn = sensekey_to_synset(gold_synset_name)
            if gold_syn is None:
                skipped += 1
                continue

            gold_synset_str = gold_syn.name()
            if gold_synset_str not in sense_names:
                skipped += 1
                continue

            if len(senses) > max_senses:
                skipped += 1
                continue

            gold_idx = sense_names.index(gold_synset_str)

            self.samples.append({
                "sentence" : item["sentence"],
                "word"     : word,
                "pos"      : pos,
                "glosses"  : glosses,
                "gold_idx" : gold_idx,
            })

        print(f"Dataset built: {len(self.samples)} valid samples, {skipped} skipped")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def wsd_collate_fn(batch):
    """
    Custom collate — keeps samples as a list of dicts since sense counts
    differ per instance. The trainer iterates over items individually.
    """
    return batch