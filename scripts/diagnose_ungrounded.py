"""
diagnose_ungrounded.py

Scans dataset/augmented/all.jsonl for ungrounded instances and classifies
each one into the reason it couldn't be grounded:

  - monosemous        : word has only 1 sense in WordNet → no misleading image possible
  - sd_fail_helpful   : gold synset image generation failed (image_index has [])
  - sd_fail_misleading: all competing synsets failed generation
  - sd_fail_irrelevant: all irrelevant pool synsets failed (very unlikely)
  - mixed             : more than one of the above

Run from project root:
    python scripts/diagnose_ungrounded.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import defaultdict
from pathlib import Path

from nltk.corpus import wordnet as wn
from tqdm import tqdm

from senses.wordnet import get_senses
from utils.wordnet import sensekey_to_synset


AUGMENTED_ALL  = Path("dataset/augmented_semeval/all.jsonl")
IMAGE_INDEX    = Path("data/image_index.json")


def synset_to_id(synset):
    return f"{synset.pos()}{synset.offset():08d}"


def main():
    print("Loading image index...")
    with open(IMAGE_INDEX) as f:
        image_index = json.load(f)

    print("Loading augmented dataset...")
    with open(AUGMENTED_ALL) as f:
        instances = [json.loads(l) for l in f]

    ungrounded = [i for i in instances if not i["has_visual_grounding"]]
    print(f"\nTotal instances    : {len(instances):,}")
    print(f"Ungrounded         : {len(ungrounded):,} ({len(ungrounded)/len(instances):.1%})\n")

    reasons = defaultdict(list)

    for inst in tqdm(ungrounded, desc="Diagnosing"):
        gold_syn = sensekey_to_synset(inst["sense"])

        # ── reason 1: invalid sense key ──────────────────────────────────────
        if gold_syn is None:
            reasons["invalid_sense_key"].append(inst)
            continue

        gold_id = synset_to_id(gold_syn)

        # ── reason 2: helpful image missing ──────────────────────────────────
        helpful_missing = not image_index.get(gold_id)

        # ── reason 3: monosemous / misleading missing ─────────────────────────
        senses = get_senses(inst["word"], inst.get("pos"))
        competing_ids = []
        for sense_name, _ in senses:
            try:
                syn = wn.synset(sense_name)
                if syn == gold_syn:
                    continue
                competing_ids.append(synset_to_id(syn))
            except Exception:
                continue

        monosemous = len(competing_ids) == 0
        misleading_missing = (
            not monosemous and
            not any(image_index.get(cid) for cid in competing_ids)
        )

        # ── reason 4: irrelevant missing ─────────────────────────────────────
        irrelevant_sid = inst["imagenet_synsets"].get("irrelevant")
        irrelevant_missing = irrelevant_sid and not image_index.get(irrelevant_sid)

        # ── classify ─────────────────────────────────────────────────────────
        if monosemous:
            reasons["monosemous"].append(inst)
        elif helpful_missing and not misleading_missing and not irrelevant_missing:
            reasons["sd_fail_helpful_only"].append(inst)
        elif misleading_missing and not helpful_missing and not irrelevant_missing:
            reasons["sd_fail_misleading_only"].append(inst)
        elif helpful_missing and misleading_missing:
            reasons["sd_fail_both"].append(inst)
        elif irrelevant_missing:
            reasons["sd_fail_irrelevant"].append(inst)
        else:
            reasons["unknown"].append(inst)

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"{'REASON':<30} {'COUNT':>6}  {'%ungrounded':>12}")
    print(f"{'-'*55}")
    total_ung = len(ungrounded)
    for reason, items in sorted(reasons.items(), key=lambda x: -len(x[1])):
        print(f"  {reason:<28} {len(items):>6}  {len(items)/total_ung:>11.1%}")
    print(f"{'='*55}")

    # ── per-POS breakdown for monosemous ─────────────────────────────────────
    if reasons["monosemous"]:
        print(f"\nMonosemous by POS:")
        pos_counts = defaultdict(int)
        for inst in reasons["monosemous"]:
            pos_counts[(inst.get("pos") or "UNKNOWN").upper()] += 1
        for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
            print(f"  {pos:<10} {count:>5}")

    # ── fixable vs unfixable summary ─────────────────────────────────────────
    fixable = (
        len(reasons["sd_fail_helpful_only"]) +
        len(reasons["sd_fail_misleading_only"]) +
        len(reasons["sd_fail_both"])
    )
    unfixable = len(reasons["monosemous"]) + len(reasons["invalid_sense_key"])

    print(f"\nFixable by retrying SD   : {fixable:,}")
    print(f"Unfixable (monosemous)   : {unfixable:,}")
    print(f"Unknown                  : {len(reasons['unknown']):,}")

    # ── sample monosemous words ───────────────────────────────────────────────
    if reasons["monosemous"]:
        print(f"\nSample monosemous instances (first 10):")
        seen = set()
        for inst in reasons["monosemous"]:
            key = (inst["word"], inst.get("pos"))
            if key in seen:
                continue
            seen.add(key)
            syn = sensekey_to_synset(inst["sense"])
            gloss = syn.definition() if syn else "?"
            print(f"  {inst['word']:<20} {(inst.get('pos') or '?'):<6}  {gloss[:60]}")
            if len(seen) >= 10:
                break


if __name__ == "__main__":
    main()
