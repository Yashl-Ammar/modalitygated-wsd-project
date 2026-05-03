"""
scripts/analyse_semcor_synsets.py

Counts unique synsets in SemCor and checks how many already have
images in the existing image index (from the SemEval generation run).

Run from project root:
    python scripts/analyse_semcor_synsets.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import defaultdict
from pathlib import Path

from nltk.corpus import wordnet as wn
from tqdm import tqdm

from data.loader import load_jsonl
from utils.wordnet import sensekey_to_synset


IMAGE_INDEX_PATH = Path("data/image_index.json")


def synset_to_id(synset):
    return f"{synset.pos()}{synset.offset():08d}"


def main():
    print("Loading SemCor...")
    data = load_jsonl("dataset/semcor/SemCor.jsonl")
    print(f"  {len(data):,} instances")

    print("Loading image index...")
    with open(IMAGE_INDEX_PATH) as f:
        image_index = json.load(f)
    print(f"  {len(image_index):,} synsets already have images")

    # collect unique synsets
    unique_synsets = set()
    skipped        = 0
    pos_counts     = defaultdict(int)

    for item in tqdm(data, desc="Scanning", leave=False):
        gold_syn = sensekey_to_synset(item["sense"])
        if gold_syn is None:
            skipped += 1
            continue
        sid = synset_to_id(gold_syn)
        unique_synsets.add(sid)
        pos_counts[gold_syn.pos()] += 1

    already_have  = {sid for sid in unique_synsets if image_index.get(sid)}
    need_generate = unique_synsets - already_have

    print(f"\n{'='*50}")
    print(f"SemCor unique synsets   : {len(unique_synsets):,}")
    print(f"Already have images     : {len(already_have):,}")
    print(f"Need to generate        : {len(need_generate):,}")
    print(f"Skipped (invalid gold)  : {skipped:,}")
    print(f"\nPOS breakdown (instances):")
    pos_map = {"n": "NOUN", "v": "VERB", "a": "ADJ", "s": "ADJ_SAT", "r": "ADV"}
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        print(f"  {pos_map.get(pos, pos):<10} {count:>7,} instances")

    # estimate generation time
    secs_per_image = 2.0   # ~2s per image on RTX 3070 fp16
    total_mins     = len(need_generate) * secs_per_image / 60
    print(f"\nEstimated generation time: {total_mins:.0f} min  ({total_mins/60:.1f} hrs)")

    # save list of synsets to generate
    out_path = Path("data/semcor_synsets_to_generate.json")
    with open(out_path, "w") as f:
        json.dump(list(need_generate), f)
    print(f"\nSynset list saved to: {out_path}")
    print("Pass this to generate_semcor_images.py")


if __name__ == "__main__":
    main()
