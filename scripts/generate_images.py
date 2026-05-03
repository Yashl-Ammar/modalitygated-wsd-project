"""
generate_images.py

Generates images for all synsets needed across SemEval/Senseval datasets
using Stable Diffusion 1.5, conditioned on WordNet gloss definitions.

For each synset the prompt is built from its POS, lemma names, and gloss:
    noun  → "a photograph of {lemma}, {gloss}"
    verb  → "a photograph of a person {lemma}ing, {gloss}"
    adj   → "a photograph of something {lemma}, {gloss}"
    adv   → "a photograph of an action done {lemma}, {gloss}"

Each image uses a seed derived from its synset ID, so images are
reproducible regardless of run order.

Checkpoints every 50 synsets so runs can be safely interrupted and resumed.

Run from project root:
    python scripts/generate_images.py

Requirements:
    pip install diffusers transformers accelerate torch

Output:
    data/images/{synset_id}/img_0000.jpg
    data/image_index.json
    data/synset_meta.json
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from nltk.corpus import wordnet as wn
from tqdm import tqdm

from data.loader import load_jsonl
from senses.wordnet import get_senses
from utils.wordnet import sensekey_to_synset


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID         = "runwayml/stable-diffusion-v1-5"
IMAGE_DIR        = Path("data/images")
IMAGE_INDEX_PATH = Path("data/image_index.json")
SYNSET_META_PATH = Path("data/synset_meta.json")
IMAGE_SIZE       = 512
INFERENCE_STEPS  = 25
GUIDANCE_SCALE   = 7.5
SEED             = 42

DATASETS = {
    "senseval2"  : "dataset/senseval2.jsonl",
    "senseval3"  : "dataset/senseval3.jsonl",
    "semeval2007": "dataset/semeval2007.jsonl",
    "semeval2013": "dataset/semeval2013.jsonl",
    "semeval2015": "dataset/semeval2015.jsonl",
}

NEGATIVE_PROMPT = (
    "cartoon, illustration, drawing, painting, sketch, text, watermark, "
    "logo, blurry, low quality, abstract, collage, multiple images, grid"
)


# ── Synset helpers ────────────────────────────────────────────────────────────

def synset_to_id(synset):
    return f"{synset.pos()}{synset.offset():08d}"


def synset_seed(synset_id):
    """Deterministic seed from synset ID so images reproduce independently of run order."""
    return int(hashlib.md5(synset_id.encode()).hexdigest(), 16) % (2 ** 32)


def build_prompt(synset):
    """
    POS-aware prompt templates so SD receives a visually grounded instruction
    regardless of whether the synset is a noun, verb, adjective, or adverb.
    """
    gloss  = synset.definition()
    lemmas = ", ".join(synset.lemma_names()[:3]).replace("_", " ")
    pos    = synset.pos()

    if pos == "n":
        prompt = f"a photograph of {lemmas}, {gloss}"
    elif pos == "v":
        primary = synset.lemma_names()[0].replace("_", " ")
        root    = primary[:-1] if primary.endswith("e") else primary
        prompt  = f"a photograph of a person {root}ing, {gloss}"
    elif pos == "a" or pos == "s":   # adjective / adjective satellite
        primary = synset.lemma_names()[0].replace("_", " ")
        prompt  = f"a photograph of something {primary}, {gloss}"
    else:                             # adverb
        primary = synset.lemma_names()[0].replace("_", " ")
        prompt  = f"a photograph of an action performed {primary}, {gloss}"

    return prompt


# ── Step 1: collect needed synsets ────────────────────────────────────────────

def collect_needed_synsets():
    random.seed(SEED)

    all_gold_ids      = set()
    all_competing_ids = set()
    instance_map      = {}

    for ds_name, path in DATASETS.items():
        data = load_jsonl(path)
        for idx, item in enumerate(tqdm(data, desc=f"Scanning {ds_name}", leave=False)):
            gold_syn = sensekey_to_synset(item["sense"])
            if gold_syn is None:
                continue

            gold_id = synset_to_id(gold_syn)
            all_gold_ids.add(gold_id)

            senses = get_senses(item["word"], item.get("pos"))
            competing_ids = []
            for sense_name, _ in senses:
                try:
                    syn = wn.synset(sense_name)
                    if syn == gold_syn:
                        continue
                    cid = synset_to_id(syn)
                    competing_ids.append(cid)
                    all_competing_ids.add(cid)
                except Exception:
                    continue

            instance_map[(ds_name, idx)] = {
                "gold_id"      : gold_id,
                "competing_ids": competing_ids,
            }

    dataset_synsets      = all_gold_ids | all_competing_ids
    candidate_irrelevant = [
        synset_to_id(s)
        for s in wn.all_synsets(pos="n")
        if synset_to_id(s) not in dataset_synsets
    ]
    irrelevant_pool = random.sample(candidate_irrelevant, min(500, len(candidate_irrelevant)))
    needed_synsets  = all_gold_ids | all_competing_ids | set(irrelevant_pool)

    print(f"\nGold synsets        : {len(all_gold_ids):,}")
    print(f"Competing synsets   : {len(all_competing_ids):,}")
    print(f"Irrelevant pool     : {len(irrelevant_pool):,}")
    print(f"Total to generate   : {len(needed_synsets):,} images")

    meta = {
        "irrelevant_pool": irrelevant_pool,
        "instance_map"   : {f"{k[0]}__{k[1]}": v for k, v in instance_map.items()},
    }
    SYNSET_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYNSET_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved {SYNSET_META_PATH}")

    return needed_synsets


# ── Step 2: load pipeline ─────────────────────────────────────────────────────

def load_pipeline():
    from diffusers import StableDiffusionPipeline

    print(f"\nLoading {MODEL_ID}...")
    print("(First run downloads ~4GB — cached after that)")
    print("Using Stable Diffusion 1.5")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers enabled.")
    except Exception:
        pipe.enable_attention_slicing()
        print("Attention slicing enabled.")

    print("Pipeline ready.")
    return pipe


# ── Step 3: generate images ───────────────────────────────────────────────────

def generate_images(needed_synsets, pipe):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    image_index = {}
    if IMAGE_INDEX_PATH.exists():
        with open(IMAGE_INDEX_PATH) as f:
            image_index = json.load(f)
        print(f"Resuming — {len(image_index):,} synsets already done")

    # sorted for deterministic order
    remaining = sorted(sid for sid in needed_synsets if sid not in image_index)
    print(f"Synsets to generate : {len(remaining):,}\n")

    if not remaining:
        print("All images already generated.")
        return image_index

    start_time = time.time()
    failed_synsets = []

    for i, synset_id in enumerate(tqdm(remaining, desc="Generating")):
        try:
            pos    = synset_id[0]
            offset = int(synset_id[1:])
            syn    = wn.synset_from_pos_and_offset(pos, offset)
        except Exception as e:
            print(f"\n[SKIP] {synset_id}: could not resolve synset — {e}")
            image_index[synset_id] = []
            failed_synsets.append((synset_id, "synset_lookup", str(e)))
            continue

        prompt    = build_prompt(syn)
        generator = torch.Generator("cuda").manual_seed(synset_seed(synset_id))

        synset_dir = IMAGE_DIR / synset_id
        synset_dir.mkdir(parents=True, exist_ok=True)
        img_path = synset_dir / "img_0000.jpg"

        try:
            result = pipe(
                prompt             = prompt,
                negative_prompt    = NEGATIVE_PROMPT,
                height             = IMAGE_SIZE,
                width              = IMAGE_SIZE,
                num_inference_steps= INFERENCE_STEPS,
                guidance_scale     = GUIDANCE_SCALE,
                generator          = generator,
            )
            result.images[0].save(img_path, "JPEG", quality=92)
            image_index[synset_id] = [str(img_path)]

        except Exception as e:
            print(f"\n[FAIL] {synset_id} ({syn.name()}): {e}")
            image_index[synset_id] = []
            failed_synsets.append((synset_id, syn.name(), str(e)))
            continue

        if (i + 1) % 50 == 0:
            _save_index(image_index)
            elapsed  = time.time() - start_time
            per_img  = elapsed / (i + 1)
            eta_secs = per_img * (len(remaining) - i - 1)
            done     = sum(1 for v in image_index.values() if v)
            print(f"\n[{i+1}/{len(remaining)}] {done:,} saved  "
                  f"| {per_img:.1f}s/image  "
                  f"| ETA {eta_secs/3600:.1f}h  "
                  f"| {len(failed_synsets)} failed")

    _save_index(image_index)
    done = sum(1 for v in image_index.values() if v)
    print(f"\nDone: {done:,} / {len(needed_synsets):,} images generated")

    if failed_synsets:
        print(f"\n{'='*50}")
        print(f"FAILED ({len(failed_synsets)} synsets):")
        for sid, name, err in failed_synsets[:20]:
            print(f"  {sid}  {name}  —  {err}")
        if len(failed_synsets) > 20:
            print(f"  ... and {len(failed_synsets) - 20} more")

    return image_index


def _save_index(image_index):
    IMAGE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(IMAGE_INDEX_PATH, "w") as f:
        json.dump(image_index, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, metavar="N",
                        help="Sanity check: generate only N images and exit")
    args = parser.parse_args()

    print("Step 1: Scanning datasets for needed synsets...")
    needed_synsets = collect_needed_synsets()

    if args.test:
        needed_synsets = set(sorted(needed_synsets)[:args.test])
        print(f"\n[test mode] Limiting to {args.test} synsets")

    print("\nStep 2: Loading Stable Diffusion 1.5 pipeline...")
    pipe = load_pipeline()

    print("\nStep 3: Generating images...")
    generate_images(needed_synsets, pipe)

    if not args.test:
        print("\nNext: python scripts/build_augmented_dataset.py")


if __name__ == "__main__":
    main()
