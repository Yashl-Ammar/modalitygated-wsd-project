"""
evaluation/evaluator.py

Shared evaluation framework for all WSD model conditions.

Handles:
    - Per-dataset and aggregate F1 / accuracy
    - Per-POS breakdown (NOUN, VERB, ADJ, ADV)
    - Image condition routing (helpful / misleading / irrelevant / None)
    - Structured JSON result output
    - Console summary

Usage:
    from evaluation.evaluator import WSDEvaluator
    evaluator = WSDEvaluator(model, output_dir="results")
    evaluator.run(condition_name="bert_finetuned", image_condition=None)
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from data.loader import load_jsonl
from utils.wordnet import sensekey_to_synset, pred_to_synset


# ── Dataset registry ──────────────────────────────────────────────────────────

DATASETS = {
    "senseval2"  : "dataset/augmented_semeval/senseval2.jsonl",
    "senseval3"  : "dataset/augmented_semeval/senseval3.jsonl",
    "semeval2007": "dataset/augmented_semeval/semeval2007.jsonl",
    "semeval2013": "dataset/augmented_semeval/semeval2013.jsonl",
    "semeval2015": "dataset/augmented_semeval/semeval2015.jsonl",
}

POS_LABELS = ["NOUN", "VERB", "ADJ", "ADV"]


# ── Metrics helpers ───────────────────────────────────────────────────────────

def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def compute_metrics(correct, total, skipped):
    """
    For WSD with full coverage (every instance gets a prediction),
    precision == recall == F1 == accuracy. We report all for completeness.
    """
    if total == 0:
        return {"f1": 0.0, "accuracy": 0.0, "precision": 0.0,
                "recall": 0.0, "n_correct": 0, "n_total": 0, "n_skipped": skipped}

    accuracy          = correct / total
    precision, recall, f1 = compute_f1(correct, total - correct, total - correct)

    return {
        "f1"        : round(f1, 4),
        "accuracy"  : round(accuracy, 4),
        "precision" : round(precision, 4),
        "recall"    : round(recall, 4),
        "n_correct" : correct,
        "n_total"   : total,
        "n_skipped" : skipped,
    }


# ── Core evaluation loop ──────────────────────────────────────────────────────

class WSDEvaluator:
    """
    Evaluates a WSD model across all augmented SemEval datasets.

    Args:
        model:        Any model with a .predict(sentence, word, pos, image_path=None) method
        output_dir:   Directory to write JSON result files
    """

    def __init__(self, model, output_dir="results", data_dir=None):
        self.model      = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if data_dir is not None:
            base          = Path(data_dir)
            self.datasets = {k: str(base / Path(v).name) for k, v in DATASETS.items()}
        else:
            self.datasets = DATASETS

    def run(
        self,
        condition_name,
        image_condition=None,
        datasets=None,
        verbose=True,
    ):
        """
        Run evaluation across all datasets.

        Args:
            condition_name:  identifier for this run e.g. 'bert_finetuned'
            image_condition: one of None / 'helpful' / 'misleading' / 'irrelevant'
                             None means text-only (image is never passed to model)
            datasets:        subset of dataset names to evaluate, or None for all
            verbose:         print per-dataset summaries to console

        Returns:
            results dict with per-dataset and aggregate metrics
        """
        target_datasets = datasets or list(self.datasets.keys())
        all_results     = {}
        start_total     = time.time()

        for ds_name in target_datasets:
            path = self.datasets.get(ds_name)
            if path is None:
                print(f"[WARN] Unknown dataset: {ds_name}")
                continue
            if not Path(path).exists():
                print(f"[WARN] File not found: {path}")
                continue

            ds_result = self._evaluate_dataset(
                ds_name, path, image_condition, verbose
            )
            all_results[ds_name] = ds_result

        # ── aggregate across all datasets ─────────────────────────────────
        aggregate = self._aggregate(all_results)
        all_results["aggregate"] = aggregate

        total_time = time.time() - start_total

        # ── build full result record ──────────────────────────────────────
        record = {
            "condition"      : condition_name,
            "image_condition": image_condition,
            "total_time_s"   : round(total_time, 2),
            "datasets"       : all_results,
        }

        # ── save to file ──────────────────────────────────────────────────
        img_tag  = f"_{image_condition}" if image_condition else "_text_only"
        out_path = self.output_dir / f"{condition_name}{img_tag}.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        if verbose:
            self._print_summary(record)
            print(f"\nResults saved to: {out_path}")

        return record

    # ─────────────────────────────────────────────────────────────────────────

    def _evaluate_dataset(self, ds_name, path, image_condition, verbose):
        """Evaluate on a single dataset file."""
        data = load_jsonl(path)

        # per-POS counters
        counters = {
            "overall": {"correct": 0, "total": 0, "skipped": 0},
        }
        for pos in POS_LABELS:
            counters[pos] = {"correct": 0, "total": 0, "skipped": 0}

        start = time.time()

        for item in tqdm(data, desc=f"  {ds_name}", leave=False):
            gold_syn = sensekey_to_synset(item["sense"])
            if gold_syn is None:
                counters["overall"]["skipped"] += 1
                continue

            # get image path if multimodal condition
            image_path = None
            if image_condition is not None:
                image_path = item.get("images", {}).get(image_condition)

            # get prediction
            pred_sense = self._predict(item, image_path)
            pred_syn   = pred_to_synset(pred_sense) if pred_sense else None

            pos = (item.get("pos") or "UNKNOWN").upper()
            is_correct = (pred_syn is not None and pred_syn == gold_syn)

            # update overall
            counters["overall"]["total"]   += 1
            counters["overall"]["correct"] += int(is_correct)

            # update per-POS
            if pos in counters:
                counters[pos]["total"]   += 1
                counters[pos]["correct"] += int(is_correct)

        elapsed = time.time() - start

        # compute metrics
        overall_metrics = compute_metrics(**counters["overall"])
        pos_metrics     = {}
        for pos in POS_LABELS:
            c = counters[pos]
            if c["total"] > 0:
                pos_metrics[pos] = compute_metrics(**c)

        result = {
            "overall"     : overall_metrics,
            "pos_breakdown": pos_metrics,
            "time_s"      : round(elapsed, 2),
            "speed"       : round(overall_metrics["n_total"] / (elapsed + 1e-8), 1),
        }

        if verbose:
            self._print_dataset_result(ds_name, result)

        return result

    def _predict(self, item, image_path=None):
        """
        Call the model's predict method.
        Passes image_path if provided, otherwise text-only.
        """
        sentence = item["sentence"]
        word     = item["word"]
        pos      = item.get("pos")

        if image_path is not None:
            return self.model.predict(sentence, word, pos, image_path=image_path)
        else:
            return self.model.predict(sentence, word, pos)

    # ─────────────────────────────────────────────────────────────────────────

    def _aggregate(self, all_results):
        """Aggregate metrics across all datasets."""
        total_correct = 0
        total_total   = 0
        total_skipped = 0
        pos_agg       = {pos: {"correct": 0, "total": 0, "skipped": 0}
                         for pos in POS_LABELS}

        for ds_name, result in all_results.items():
            if ds_name == "aggregate":
                continue
            o = result["overall"]
            total_correct += o["n_correct"]
            total_total   += o["n_total"]
            total_skipped += o["n_skipped"]

            for pos in POS_LABELS:
                if pos in result["pos_breakdown"]:
                    pm = result["pos_breakdown"][pos]
                    pos_agg[pos]["correct"] += pm["n_correct"]
                    pos_agg[pos]["total"]   += pm["n_total"]

        overall_metrics = compute_metrics(total_correct, total_total, total_skipped)
        pos_metrics     = {}
        for pos in POS_LABELS:
            c = pos_agg[pos]
            if c["total"] > 0:
                pos_metrics[pos] = compute_metrics(
                    c["correct"], c["total"], 0
                )

        return {
            "overall"      : overall_metrics,
            "pos_breakdown": pos_metrics,
        }

    # ─────────────────────────────────────────────────────────────────────────

    def _print_dataset_result(self, ds_name, result):
        o = result["overall"]
        print(f"\n  ── {ds_name} {'─'*(35-len(ds_name))}")
        print(f"     F1       : {o['f1']:.4f}")
        print(f"     Accuracy : {o['accuracy']:.4f}")
        print(f"     Correct  : {o['n_correct']:,} / {o['n_total']:,}")
        print(f"     Speed    : {result['speed']:.1f} samples/sec")
        if result["pos_breakdown"]:
            print(f"     POS breakdown:")
            for pos, pm in result["pos_breakdown"].items():
                print(f"       {pos:<6}  F1: {pm['f1']:.4f}  "
                      f"({pm['n_correct']}/{pm['n_total']})")

    def _print_summary(self, record):
        agg = record["datasets"].get("aggregate", {}).get("overall", {})
        print(f"\n{'='*50}")
        print(f"CONDITION : {record['condition']}")
        print(f"IMAGE     : {record['image_condition'] or 'text-only'}")
        print(f"{'='*50}")
        print(f"  Aggregate F1       : {agg.get('f1', 0):.4f}")
        print(f"  Aggregate Accuracy : {agg.get('accuracy', 0):.4f}")
        print(f"  Total instances    : {agg.get('n_total', 0):,}")
        print(f"  Total time         : {record['total_time_s']:.1f}s")
        print(f"{'='*50}")

        # aggregate POS breakdown
        pos_bd = record["datasets"].get("aggregate", {}).get("pos_breakdown", {})
        if pos_bd:
            print(f"  POS breakdown (aggregate):")
            for pos, pm in pos_bd.items():
                print(f"    {pos:<6}  F1: {pm['f1']:.4f}  "
                      f"({pm['n_correct']}/{pm['n_total']})")