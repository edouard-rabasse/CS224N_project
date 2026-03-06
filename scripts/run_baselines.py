"""
run_baselines.py — Evaluate and compare baseline models on STS + transfer + probing tasks.

Runs SentEval evaluation on:
  1. bert-base-uncased          (plain BERT, CLS pooling)
  2. princeton-nlp/unsup-simcse-bert-base-uncased  (SimCSE, CLS pooling)
  3. (optional) our trained SyntaxBertModel checkpoint

Outputs a side-by-side table and saves results to JSON.

Usage:
    # Baselines only
    uv run python scripts/run_baselines.py --mode test --task-set sts

    # Include our model
    uv run python scripts/run_baselines.py \
        --mode test --task-set sts \
        --our-model outputs/multi_loss

    # Probing tasks
    uv run python scripts/run_baselines.py --mode test --task-set probing

    # All tasks
    uv run python scripts/run_baselines.py --mode test --task-set full

    # Fast dev run
    uv run python scripts/run_baselines.py --mode dev --task-set sts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from prettytable import PrettyTable
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# SentEval lives inside the SimCSE submodule
PROJECT_ROOT = Path(__file__).parent.parent
SENTEVAL_PATH = PROJECT_ROOT / "SimCSE" / "SentEval"
SENTEVAL_DATA = SENTEVAL_PATH / "data"

STS_TASKS = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"]
TRANSFER_TASKS = ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]
PROBING_TASKS = ['Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline comparison via SentEval.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "test", "fasttest"],
        default="test",
        help="dev: fast, dev-set; test: full, test-set; fasttest: fast mode, test-set.",
    )
    parser.add_argument(
        "--task-set",
        type=str,
        choices=["sts", "transfer", "probing", "full"],
        default="sts",
        help="sts: STS tasks; transfer: classification tasks; probing: linguistic probing tasks; full: all three.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Explicit task list, overrides --task-set. E.g. --tasks STSBenchmark STS12",
    )
    parser.add_argument(
        "--our-model",
        type=str,
        default=None,
        help="Path to our trained SyntaxBertModel checkpoint (optional).",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        default=False,
        help="Skip BERT and SimCSE baselines; evaluate only --our-model.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/baseline_comparison.json",
        help="Where to save the JSON results.",
    )
    return parser.parse_args()


# =============================================================================
# SentEval runner
# =============================================================================


def make_senteval_params(mode: str) -> dict:
    if mode in ("dev", "fasttest"):
        return {
            "task_path": str(SENTEVAL_DATA),
            "usepytorch": True,
            "kfold": 5,
            "classifier": {"nhid": 0, "optim": "rmsprop", "batch_size": 128, "tenacity": 3, "epoch_size": 2},
        }
    return {
        "task_path": str(SENTEVAL_DATA),
        "usepytorch": True,
        "kfold": 10,
        "classifier": {"nhid": 0, "optim": "adam", "batch_size": 64, "tenacity": 5, "epoch_size": 4},
    }


def make_batcher(model, tokenizer, device: torch.device, pooler: str = "cls_before_pooler"):
    """Build a SentEval-compatible batcher for a plain HuggingFace AutoModel."""

    def batcher(params: dict, batch: list[list[str]], max_length: int | None = None) -> np.ndarray:
        if batch and batch[0] and isinstance(batch[0][0], bytes):
            batch = [[w.decode("utf-8") for w in s] for s in batch]
        sentences = [" ".join(s) for s in batch]

        encoded = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            **({"max_length": max_length} if max_length else {}),
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)

        last_hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"]

        if pooler == "cls":
            emb = outputs.pooler_output
        elif pooler == "cls_before_pooler":
            emb = last_hidden[:, 0]
        elif pooler == "avg":
            emb = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
        elif pooler in ("avg_top2", "avg_first_last"):
            hidden_states = outputs.hidden_states
            a, b = (hidden_states[-2], hidden_states[-1]) if pooler == "avg_top2" else (hidden_states[1], hidden_states[-1])
            pooled = (a + b) / 2.0
            emb = (pooled * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
        else:
            raise NotImplementedError(f"Unknown pooler: {pooler}")

        return emb.cpu().float().numpy()

    return batcher


def run_senteval(batcher, tasks: list[str], mode: str) -> dict[str, Any]:
    """Run SentEval for a list of tasks and return raw results."""
    import senteval  # imported after sys.path is set

    params = make_senteval_params(mode)
    results = {}
    for task in tasks:
        logger.info(f"  Evaluating {task}...")
        se = senteval.engine.SE(params, batcher, prepare=lambda p, s: None)
        results[task] = se.eval(task)
    return results


# =============================================================================
# Score extraction
# =============================================================================


def extract_scores(results: dict, tasks: list[str], mode: str) -> dict[str, float]:
    """Pull the single scalar score per task from SentEval raw results."""
    scores = {}
    for task in tasks:
        if task not in results:
            scores[task] = float("nan")
            continue
        r = results[task]
        if task in ("STSBenchmark", "SICKRelatedness"):
            key = "dev" if mode == "dev" else "test"
            scores[task] = r[key]["spearman"].correlation * 100 if mode != "dev" else r["dev"]["spearman"][0] * 100
        elif task in STS_TASKS:
            if mode == "dev":
                scores[task] = r["dev"]["spearman"][0] * 100
            else:
                scores[task] = r["all"]["spearman"]["all"] * 100
        elif task in TRANSFER_TASKS:
            scores[task] = r.get("devacc" if mode == "dev" else "acc", float("nan"))
        elif task in PROBING_TASKS:
            # Probing tasks report accuracy under "devacc" / "acc"
            scores[task] = r.get("devacc" if mode == "dev" else "acc", float("nan"))
        else:
            scores[task] = r.get("devacc" if mode == "dev" else "acc", float("nan"))
    return scores


# =============================================================================
# Pretty printing
# =============================================================================


def print_comparison_table(all_scores: dict[str, dict[str, float]], tasks: list[str], section: str) -> None:
    model_names = list(all_scores.keys())
    task_subset = [t for t in tasks if any(t in s for s in all_scores.values())]

    tb = PrettyTable()
    tb.field_names = ["Task"] + model_names
    tb.align["Task"] = "l"

    row_scores: dict[str, list[float]] = {name: [] for name in model_names}
    for task in task_subset:
        row = [task]
        for name in model_names:
            score = all_scores[name].get(task, float("nan"))
            row.append(f"{score:.2f}" if not np.isnan(score) else "-")
            if not np.isnan(score):
                row_scores[name].append(score)
        tb.add_row(row)

    # Average row
    avg_row = ["Avg."]
    for name in model_names:
        vals = row_scores[name]
        avg_row.append(f"{sum(vals)/len(vals):.2f}" if vals else "-")
    tb.add_row(avg_row)

    print(f"\n{'='*60}")
    print(f"  {section}")
    print(f"{'='*60}")
    print(tb)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    if not SENTEVAL_PATH.exists():
        logger.error(f"SentEval not found at {SENTEVAL_PATH}. Run: git submodule update --init")
        sys.exit(1)
    sys.path.insert(0, str(SENTEVAL_PATH))

    if args.tasks is not None:
        tasks = args.tasks
    elif args.task_set == "sts":
        tasks = STS_TASKS
    elif args.task_set == "transfer":
        tasks = TRANSFER_TASKS
    elif args.task_set == "probing":
        tasks = PROBING_TASKS
    else:
        tasks = STS_TASKS + TRANSFER_TASKS + PROBING_TASKS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Models to evaluate ----
    baselines: list[tuple[str, str, str]] = [
        # (display_name, hf_model_id, pooler)
        # BERT-base: avg_first_last matches the SimCSE paper's reported baseline (~59 on STS-B)
        # cls_before_pooler gives ~20 (CLS token without NSP head is anisotropic and poorly calibrated)
        ("BERT-base", "bert-base-uncased", "avg_first_last"),
        # SimCSE uses cls_before_pooler at inference (MLP head is discarded after training)
        ("SimCSE-BERT", "princeton-nlp/unsup-simcse-bert-base-uncased", "cls_before_pooler"),
    ]

    all_scores: dict[str, dict[str, float]] = {}
    all_raw: dict[str, dict] = {}

    if args.no_baselines and not args.our_model:
        logger.error("--no-baselines requires --our-model to be set.")
        sys.exit(1)

    for display_name, model_id, pooler in ([] if args.no_baselines else baselines):
        logger.info(f"\n{'='*50}")
        logger.info(f"  Evaluating: {display_name}  ({model_id})")
        logger.info(f"{'='*50}")

        model = AutoModel.from_pretrained(model_id).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        batcher = make_batcher(model, tokenizer, device, pooler=pooler)

        raw = run_senteval(batcher, tasks, args.mode)
        all_raw[display_name] = raw
        all_scores[display_name] = extract_scores(raw, tasks, args.mode)

        # Free GPU memory between models
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Our model (optional) ----
    if args.our_model:
        from src.evaluate import _load_syntax_bert_model, _resolve_bert_path

        display_name = "Ours"
        logger.info(f"\n{'='*50}")
        logger.info(f"  Evaluating: {display_name}  ({args.our_model})")
        logger.info(f"{'='*50}")

        our_model = _load_syntax_bert_model(args.our_model, device)
        tokenizer = AutoTokenizer.from_pretrained(_resolve_bert_path(args.our_model), use_fast=True)

        def our_batcher(params, batch, max_length=None):
            if batch and batch[0] and isinstance(batch[0][0], bytes):
                batch = [[w.decode("utf-8") for w in s] for s in batch]
            sentences = [" ".join(s) for s in batch]
            encoded = tokenizer(
                sentences, return_tensors="pt", padding=True, truncation=True,
                **({"max_length": max_length} if max_length else {}),
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                emb = our_model.encode_sentences(**encoded)
            return emb.cpu().float().numpy()

        raw = run_senteval(our_batcher, tasks, args.mode)
        all_raw[display_name] = raw
        all_scores[display_name] = extract_scores(raw, tasks, args.mode)

    # ---- Print results ----
    sts_tasks_present = [t for t in STS_TASKS if t in tasks]
    transfer_tasks_present = [t for t in TRANSFER_TASKS if t in tasks]
    probing_tasks_present = [t for t in PROBING_TASKS if t in tasks]

    if sts_tasks_present:
        print_comparison_table(all_scores, sts_tasks_present, f"STS Results ({args.mode})")
    if transfer_tasks_present:
        print_comparison_table(all_scores, transfer_tasks_present, f"Transfer Results ({args.mode})")
    if probing_tasks_present:
        print_comparison_table(all_scores, probing_tasks_present, f"Probing Results ({args.mode})")

    # ---- Save JSON ----
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"mode": args.mode, "task_set": args.task_set, "scores": all_scores}, f, indent=2)
    logger.info(f"\nScores saved to {out_path}")


if __name__ == "__main__":
    main()
