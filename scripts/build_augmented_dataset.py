"""
build_augmented_dataset.py

Reads the local image index (built by generate_images.py) and
assembles the final augmented JSONL files — one per dataset — where
each instance has helpful, misleading, and irrelevant image fields.

Instances without full image coverage are included with
has_visual_grounding=False so the same file works for text-only
and multimodal experiments.

Run from project root:
    python scripts/build_augmented_dataset.py

Output:
    dataset/augmented/senseval2.jsonl
    dataset/augmented/senseval3.jsonl
    dataset/augmented/semeval2007.jsonl
    dataset/augmented/semeval2013.jsonl
    dataset/augmented/semeval2015.jsonl
    dataset/augmented/all.jsonl          ← concatenation of all above
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from collections import defaultdict
from pathlib import Path

from nltk.corpus import wordnet as wn
from tqdm import tqdm

from data.loader import load_jsonl
from senses.wordnet import get_senses
from utils.wordnet import sensekey_to_synset


# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_INDEX_PATH = Path("data/image_index.json")
SYNSET_META_PATH = Path("data/synset_meta.json")
OUTPUT_DIR       = Path("dataset/augmented")
SEED             = 42

DATASETS = {
    "senseval2"  : "dataset/senseval2.jsonl",
    "senseval3"  : "dataset/senseval3.jsonl",
    "semeval2007": "dataset/semeval2007.jsonl",
    "semeval2013": "dataset/semeval2013.jsonl",
    "semeval2015": "dataset/semeval2015.jsonl",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def synset_to_id(synset):
    return f"{synset.pos()}{synset.offset():08d}"


def pick_image(image_index, synset_id, used_paths, rng):
    """
    Pick one unused image for a synset.
    Returns the path string or None if no images available.
    """
    candidates = [
        p for p in image_index.get(synset_id, [])
        if p not in used_paths
    ]
    if not candidates:
        # relax constraint and allow reuse if nothing else available
        candidates = image_index.get(synset_id, [])
    if not candidates:
        return None
    chosen = rng.choice(candidates)
    used_paths.add(chosen)
    return chosen


def build_instance(item, dataset_name, image_index, irrelevant_pool, rng):
    """
    Build one augmented instance dict.
    Always includes all original fields plus image fields.
    """
    gold_syn = sensekey_to_synset(item["sense"])
    if gold_syn is None:
        return _no_grounding(item, dataset_name, reason="invalid_gold")

    gold_id = synset_to_id(gold_syn)

    # competing synsets
    senses = get_senses(item["word"], item.get("pos"))
    competing_ids = []
    for sense_name, _ in senses:
        try:
            syn = wn.synset(sense_name)
            if syn == gold_syn:
                continue
            competing_ids.append(synset_to_id(syn))
        except Exception:
            continue

    used_paths = set()

    # helpful image — from gold synset
    helpful_path = pick_image(image_index, gold_id, used_paths, rng)

    # misleading image — from a competing synset that has images
    misleading_path = None
    competing_with_images = [
        cid for cid in competing_ids
        if image_index.get(cid)
    ]
    if competing_with_images:
        chosen_competing = rng.choice(competing_with_images)
        misleading_path = pick_image(image_index, chosen_competing, used_paths, rng)
    else:
        chosen_competing = None

    # irrelevant image — from the pre-sampled irrelevant pool
    irrelevant_path = None
    irrelevant_candidates = [
        sid for sid in irrelevant_pool
        if image_index.get(sid)
    ]
    if irrelevant_candidates:
        chosen_irrelevant = rng.choice(irrelevant_candidates)
        irrelevant_path = pick_image(image_index, chosen_irrelevant, used_paths, rng)
    else:
        chosen_irrelevant = None

    has_visual_grounding = (
        helpful_path is not None and
        misleading_path is not None and
        irrelevant_path is not None
    )

    return {
        # ── original fields (unchanged) ──────────────────────────────────
        "sentence" : item["sentence"],
        "word"     : item["word"],
        "pos"      : item.get("pos"),
        "sense"    : item["sense"],
        "dataset"  : dataset_name,

        # ── synset metadata ───────────────────────────────────────────────
        "synset"   : gold_syn.name(),

        # ── visual grounding ─────────────────────────────────────────────
        "has_visual_grounding" : has_visual_grounding,

        "imagenet_synsets" : {
            "gold"       : gold_id,
            "competing"  : [chosen_competing] if chosen_competing else [],
            "irrelevant" : chosen_irrelevant,
        },

        "images" : {
            "helpful"    : helpful_path,
            "misleading" : misleading_path,
            "irrelevant" : irrelevant_path,
        },

        # ── set at eval time, null at rest ────────────────────────────────
        "image_condition" : None,
    }


def _no_grounding(item, dataset_name, reason="unknown"):
    return {
        "sentence"             : item["sentence"],
        "word"                 : item["word"],
        "pos"                  : item.get("pos"),
        "sense"                : item["sense"],
        "dataset"              : dataset_name,
        "synset"               : None,
        "has_visual_grounding" : False,
        "imagenet_synsets"     : {"gold": None, "competing": [], "irrelevant": None},
        "images"               : {"helpful": None, "misleading": None, "irrelevant": None},
        "image_condition"      : None,
        "_skip_reason"         : reason,
    }


# ── Per-dataset stats ─────────────────────────────────────────────────────────

def print_stats(name, instances):
    total = len(instances)
    pos_counts = defaultdict(int)

    for inst in instances:
        pos_counts[(inst.get("pos") or "UNKNOWN").upper()] += 1

    print(f"\n── {name} ────────────────────────────────────")
    print(f"  Total     : {total:,}")
    print("  POS breakdown:")
    for pos, count in sorted(pos_counts.items()):
        print(f"    {pos:<10} {count:>5}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)

    # load image index
    if not IMAGE_INDEX_PATH.exists():
        print(f"ERROR: {IMAGE_INDEX_PATH} not found.")
        print("Run scripts/generate_images.py first.")
        return

    print("Loading image index...")
    with open(IMAGE_INDEX_PATH) as f:
        image_index = json.load(f)
    print(f"  Synsets with images: {len(image_index):,}")

    # load irrelevant pool
    if not SYNSET_META_PATH.exists():
        print(f"ERROR: {SYNSET_META_PATH} not found.")
        print("Run scripts/generate_images.py first.")
        return

    with open(SYNSET_META_PATH) as f:
        meta = json.load(f)
    irrelevant_pool = meta["irrelevant_pool"]
    print(f"  Irrelevant pool    : {len(irrelevant_pool):,} synsets")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_instances = []

    for ds_name, path in DATASETS.items():
        data = load_jsonl(path)
        instances = []

        for item in tqdm(data, desc=f"Building {ds_name}", leave=False):
            inst = build_instance(item, ds_name, image_index, irrelevant_pool, rng)
            if not inst["has_visual_grounding"]:
                continue
            instances.append(inst)

        # write per-dataset file
        out_path = OUTPUT_DIR / f"{ds_name}.jsonl"
        with open(out_path, "w") as f:
            for inst in instances:
                f.write(json.dumps(inst) + "\n")

        print_stats(ds_name, instances)
        all_instances.extend(instances)

    # write combined file
    all_path = OUTPUT_DIR / "all.jsonl"
    with open(all_path, "w") as f:
        for inst in all_instances:
            f.write(json.dumps(inst) + "\n")

    # aggregate summary
    total = len(all_instances)
    print(f"\n══ AGGREGATE ═══════════════════════════════════")
    print(f"  Total instances    : {total:,}  (all visually grounded)")
    print(f"\nAugmented datasets written to: {OUTPUT_DIR}/")
    print(f"Combined file: {all_path}")


if __name__ == "__main__":
    main()
