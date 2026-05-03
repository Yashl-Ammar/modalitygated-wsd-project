"""
scripts/create_mixed_dataset.py

Build dataset/augmented_semeval_mixed/ from dataset/augmented_semeval/.

For each SemEval instance, randomly assigns either its helpful or irrelevant
image with 50/50 probability (seed=42) and writes the result as a new
`images["mixed"]` field.  Every other field is preserved unchanged.

The output files are read by:
    python evaluate.py --condition clip_multimodal --image_condition mixed
    python scripts/evaluate_learned_gate.py --image_condition mixed

Usage:
    python scripts/create_mixed_dataset.py
    python scripts/create_mixed_dataset.py --seed 123
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
from pathlib import Path


DATASETS = [
    "senseval2",
    "senseval3",
    "semeval2007",
    "semeval2013",
    "semeval2015",
]

SRC_DIR = Path("dataset/augmented_semeval")
DST_DIR = Path("dataset/augmented_semeval_mixed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DST_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    total_helpful    = 0
    total_irrelevant = 0

    print(f"Source : {SRC_DIR}")
    print(f"Output : {DST_DIR}")
    print(f"Seed   : {args.seed}\n")
    print(f"  {'Dataset':<14}  {'Helpful':>9}  {'Irrelevant':>11}  {'Total':>7}")
    print(f"  {'─'*14}  {'─'*9}  {'─'*11}  {'─'*7}")

    for ds_name in DATASETS:
        src_path = SRC_DIR / f"{ds_name}.jsonl"
        dst_path = DST_DIR / f"{ds_name}.jsonl"

        if not src_path.exists():
            print(f"  [WARN] Not found: {src_path}")
            continue

        n_helpful    = 0
        n_irrelevant = 0
        out_lines    = []

        with open(src_path) as f:
            for line in f:
                record = json.loads(line)
                chosen = rng.choice(["helpful", "irrelevant"])
                record["images"]["mixed"] = record["images"].get(chosen)

                if chosen == "helpful":
                    n_helpful += 1
                else:
                    n_irrelevant += 1

                out_lines.append(json.dumps(record))

        with open(dst_path, "w") as f:
            f.write("\n".join(out_lines) + "\n")

        total_helpful    += n_helpful
        total_irrelevant += n_irrelevant
        n_total           = n_helpful + n_irrelevant

        print(f"  {ds_name:<14}  {n_helpful:>9,}  {n_irrelevant:>11,}  {n_total:>7,}")

    grand_total = total_helpful + total_irrelevant
    print(f"  {'─'*14}  {'─'*9}  {'─'*11}  {'─'*7}")
    print(f"  {'aggregate':<14}  {total_helpful:>9,}  {total_irrelevant:>11,}  {grand_total:>7,}")
    print(f"\nDone. Files written to: {DST_DIR}/")


if __name__ == "__main__":
    main()
