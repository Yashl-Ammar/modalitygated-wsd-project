"""
scripts/generate_semcor_images.py

Generates Stable Diffusion images for SemCor synsets that don't yet
have images in the existing image index.

Reads the list of needed synsets from data/semcor_synsets_to_generate.json
(produced by analyse_semcor_synsets.py) and generates one image per synset.

Already-generated synsets are skipped automatically so the run is safely
resumable if interrupted.

Run from project root:
    python scripts/generate_semcor_images.py

    # Sanity check — generate only 10 images
    python scripts/generate_semcor_images.py --test 10

Requirements:
    pip install diffusers transformers accelerate torch Pillow
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from nltk.corpus import wordnet as wn
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID          = "runwayml/stable-diffusion-v1-5"
IMAGE_DIR         = Path("data/images")
IMAGE_INDEX_PATH  = Path("data/image_index.json")
SYNSETS_TO_GEN    = Path("data/semcor_synsets_to_generate.json")
IMAGE_SIZE        = 512
INFERENCE_STEPS   = 25
GUIDANCE_SCALE    = 7.5
SEED              = 42

NEGATIVE_PROMPT = (
    "cartoon, illustration, drawing, painting, sketch, text, watermark, "
    "logo, blurry, low quality, abstract, collage, multiple images, grid"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def synset_seed(synset_id):
    return int(hashlib.md5(synset_id.encode()).hexdigest(), 16) % (2 ** 32)


def build_prompt(synset):
    gloss  = synset.definition()
    lemmas = ", ".join(synset.lemma_names()[:3]).replace("_", " ")
    pos    = synset.pos()

    if pos == "n":
        return f"a photograph of {lemmas}, {gloss}"
    elif pos == "v":
        primary = synset.lemma_names()[0].replace("_", " ")
        root    = primary[:-1] if primary.endswith("e") else primary
        return f"a photograph of a person {root}ing, {gloss}"
    elif pos in ("a", "s"):
        primary = synset.lemma_names()[0].replace("_", " ")
        return f"a photograph of something {primary}, {gloss}"
    else:
        primary = synset.lemma_names()[0].replace("_", " ")
        return f"a photograph of an action performed {primary}, {gloss}"


def _save_index(image_index):
    IMAGE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(IMAGE_INDEX_PATH, "w") as f:
        json.dump(image_index, f, indent=2)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, metavar="N",
                        help="Generate only N images (sanity check)")
    args = parser.parse_args()

    # load synsets to generate
    if not SYNSETS_TO_GEN.exists():
        print(f"ERROR: {SYNSETS_TO_GEN} not found.")
        print("Run: python scripts/analyse_semcor_synsets.py first.")
        return

    with open(SYNSETS_TO_GEN) as f:
        needed_synsets = json.load(f)
    print(f"Synsets to generate: {len(needed_synsets):,}")

    # load existing image index
    image_index = {}
    if IMAGE_INDEX_PATH.exists():
        with open(IMAGE_INDEX_PATH) as f:
            image_index = json.load(f)
        print(f"Existing index     : {len(image_index):,} synsets")

    # filter already done
    remaining = sorted(sid for sid in needed_synsets if sid not in image_index)
    print(f"Still needed       : {len(remaining):,}")

    if not remaining:
        print("All SemCor images already generated.")
        return

    # test mode
    if args.test:
        remaining = remaining[:args.test]
        print(f"\n[test mode] Limiting to {args.test} synsets")

    # load pipeline
    print(f"\nLoading {MODEL_ID}...")
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype             = torch.float16,
        safety_checker          = None,
        requires_safety_checker = False,
    ).to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers enabled.")
    except Exception:
        pipe.enable_attention_slicing()
        print("Attention slicing enabled.")

    # estimate time
    secs_per = 2.0
    eta_hrs  = len(remaining) * secs_per / 3600
    print(f"Estimated time: {eta_hrs:.1f} hrs")
    print("Starting generation...\n")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    start_time   = time.time()
    failed       = []

    for i, synset_id in enumerate(tqdm(remaining, desc="Generating")):
        try:
            pos    = synset_id[0]
            offset = int(synset_id[1:])
            syn    = wn.synset_from_pos_and_offset(pos, offset)
        except Exception as e:
            image_index[synset_id] = []
            failed.append((synset_id, str(e)))
            continue

        prompt    = build_prompt(syn)
        generator = torch.Generator("cuda").manual_seed(synset_seed(synset_id))

        synset_dir = IMAGE_DIR / synset_id
        synset_dir.mkdir(parents=True, exist_ok=True)
        img_path = synset_dir / "img_0000.jpg"

        try:
            result = pipe(
                prompt              = prompt,
                negative_prompt     = NEGATIVE_PROMPT,
                height              = IMAGE_SIZE,
                width               = IMAGE_SIZE,
                num_inference_steps = INFERENCE_STEPS,
                guidance_scale      = GUIDANCE_SCALE,
                generator           = generator,
            )
            result.images[0].save(img_path, "JPEG", quality=92)
            image_index[synset_id] = [str(img_path)]
        except Exception as e:
            image_index[synset_id] = []
            failed.append((synset_id, str(e)))
            continue

        # checkpoint every 100 synsets
        if (i + 1) % 100 == 0:
            _save_index(image_index)
            elapsed  = time.time() - start_time
            per_img  = elapsed / (i + 1)
            eta_secs = per_img * (len(remaining) - i - 1)
            done     = sum(1 for v in image_index.values() if v)
            print(f"\n[{i+1}/{len(remaining)}] {done:,} total images"
                  f" | {per_img:.1f}s/img"
                  f" | ETA {eta_secs/3600:.1f}h"
                  f" | {len(failed)} failed")

    _save_index(image_index)

    done = sum(1 for v in image_index.values() if v)
    print(f"\nDone: {done:,} total images in index")
    print(f"New SemCor images: {done - (len(image_index) - len(remaining)):,}")

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for sid, err in failed[:10]:
            print(f"  {sid}: {err}")


if __name__ == "__main__":
    main()
