# senses/wordnet.py
from nltk.corpus import wordnet as wn
from utils.wordnet import map_pos

def get_senses(word, pos=None):
    pos = map_pos(pos)

    synsets = wn.synsets(word, pos=pos) if pos else wn.synsets(word)

    return [
        (s.name(), s.definition())
        for s in synsets
    ]