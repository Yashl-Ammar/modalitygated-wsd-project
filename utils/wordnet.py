# utils/wordnet.py
from nltk.corpus import wordnet as wn

def map_pos(pos):
    if pos is None:
        return None

    pos = pos.upper()

    mapping = {
        "NOUN": "n",
        "VERB": "v",
        "ADJ": "a",
        "ADV": "r",

        # safety variants
        "ADJECTIVE": "a",
        "ADVERB": "r",
    }

    return mapping.get(pos, None)



def sensekey_to_synset(sense_key):
    try:
        # WordNet expects exactly one %
        if sense_key.count("%") != 1:
            return None

        lemma = wn.lemma_from_key(sense_key)
        return lemma.synset()

    except Exception:
        return None

def pred_to_synset(pred):
    try:
        return wn.synset(pred)
    except:
        return None