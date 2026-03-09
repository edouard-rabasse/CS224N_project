"""
evaluate.py — STS and transfer-task evaluation for SyntaxBertModel.

Adapts SimCSE's evaluation.py to use SyntaxBertModel.encode_sentences(),
which runs only the BERT branch (GNN discarded after training).

Prerequisites:
    1. Clone SentEval and download its data:
           git clone https://github.com/facebookresearch/SentEval.git
           cd SentEval/data/downstream && bash download_dataset.sh
    2. Install prettytable (already in pyproject.toml via SimCSE deps).

Usage:
    # Evaluate an output directory produced by SyntaxCLTrainer
    uv run python src/evaluate.py \\
        --model-path outputs/multi_loss \\
        --mode test \\
        --task-set sts

    # Fast dev-set run for iteration
    uv run python src/evaluate.py \\
        --model-path outputs/multi_loss \\
        --mode dev \\
        --task-set sts

    # Evaluate a plain HuggingFace model (for BERT baseline)
    uv run python src/evaluate.py \\
        --model-path bert-base-uncased \\
        --baseline \\
        --mode fasttest \\
        --task-set sts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from prettytable import PrettyTable
from transformers import AutoConfig, AutoTokenizer

logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SyntaxBertModel on STS / transfer tasks via SentEval."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a SyntaxBertModel checkpoint directory (or HuggingFace model name "
        "when --baseline is set).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="If set, load a plain HuggingFace AutoModel instead of SyntaxBertModel "
        "(useful for BERT / SimCSE baseline comparisons).",
    )
    parser.add_argument(
        "--pooler",
        type=str,
        choices=["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"],
        default="cls",
        help="Pooling strategy used when --baseline is set (ignored for SyntaxBertModel).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dev", "test", "fasttest"],
        default="test",
        help="dev: fast, dev-set results; test: full, test-set results; "
        "fasttest: fast mode but test-set results.",
    )
    parser.add_argument(
        "--task-set",
        type=str,
        choices=["sts", "transfer", "probing", "full", "na"],
        default="sts",
        help="sts: STS12-16 + STSBenchmark + SICKRelatedness; "
        "transfer: classification tasks; probing: linguistic probing tasks; full: all three.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Explicit task list (overrides --task-set when provided).",
    )
    parser.add_argument(
        "--senteval-path",
        type=str,
        default="./SimCSE/SentEval",
        help="Path to the SentEval repository root.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If provided, write a JSON summary of all scores to this path.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenisation length for the batcher.",
    )
    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def _print_table(task_names: list[str], scores: list[str]) -> None:
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)


def _load_syntax_bert_model(model_path: str, device: torch.device):
    """Load a SyntaxBertModel from a checkpoint directory.

    The checkpoint must contain:
      - ``bert_weights/``  — HuggingFace BERT weights (saved by save_bert_only)
        OR  the full ``syntax_bert_state.pt`` produced by the Trainer.

    Falls back to plain BERT if neither is found (treats model_path as
    a HuggingFace model name or a BERT-only checkpoint).
    """
    from src.models.wrapper import SyntaxBertModel

    ckpt_dir = Path(model_path)

    # Try full checkpoint first (contains GNN weights + config)
    # save_checkpoint writes: bert_weights/, gnn_state.pt, model_config.json
    gnn_ckpt = ckpt_dir / "gnn_state.pt"
    model_cfg_path = ckpt_dir / "model_config.json"
    bert_only_dir = ckpt_dir / "bert_weights"
    bert_inference_dir = ckpt_dir / "bert_inference"

    if gnn_ckpt.exists() and model_cfg_path.exists() and bert_only_dir.exists():
        logger.info(f"Loading full SyntaxBertModel checkpoint from {ckpt_dir}")
        return SyntaxBertModel.from_checkpoint(str(ckpt_dir), device=device)

    if bert_only_dir.exists():
        logger.info(f"Loading BERT-only weights from {bert_only_dir} (inference mode)")
        # GNN state is not needed at inference — build a minimal SyntaxBertModel
        # with random GNN weights (GNN is unused at inference anyway)
        return _load_bert_only_wrapper(str(bert_only_dir), device)

    if bert_inference_dir.exists():
        logger.info(f"Loading BERT-only weights from {bert_inference_dir} (inference mode)")
        return _load_bert_only_wrapper(str(bert_inference_dir), device)

    # Fallback: treat model_path as a plain HuggingFace model / BERT-only dir
    logger.warning(
        f"No SyntaxBertModel checkpoint found at {ckpt_dir}. "
        "Loading as plain HuggingFace BERT (no GNN)."
    )
    return _load_bert_only_wrapper(model_path, device)


def _load_bert_only_wrapper(model_path: str, device: torch.device):
    """Wrap a HuggingFace BERT as a minimal SyntaxBertModel (inference only).

    Uses ``add_pooling_layer=False`` so inference falls back to
    ``last_hidden_state[:, 0]`` (raw CLS token), matching SimCSE's
    ``pooler_type="cls"`` inference behaviour.  ``from_pretrained`` handles
    old-style ``beta``/``gamma`` → ``weight``/``bias`` renaming automatically.
    """
    from transformers import BertModel

    from src.models.wrapper import SyntaxBertModel

    # add_pooling_layer=False: SimCSE's BertForCL creates its inner BertModel
    # without a pooler.  At inference (pooler_type="cls", mlp_only_train=True)
    # it returns last_hidden_state[:, 0] directly.  wrapper.py's non-SimCSE
    # path does the same when pooler_output is None.
    bert_model = BertModel.from_pretrained(model_path, add_pooling_layer=False)
    wrapper = SyntaxBertModel(
        bert_model=bert_model,
        gnn_config={"conv_type": "gat", "num_layers": 2, "hidden_dim": bert_model.config.hidden_size},
        alignment_config={},
    )
    wrapper.eval()
    return wrapper.to(device)


# =============================================================================
# SentEval batcher
# =============================================================================


def build_batcher(model, tokenizer, device: torch.device, args: argparse.Namespace, is_baseline: bool):
    """Return a SentEval-compatible batcher function.

    For SyntaxBertModel, calls ``encode_sentences()`` (BERT-only inference).
    For baseline AutoModel, applies the pooler specified by ``--pooler``.
    """

    def batcher(params: dict[str, Any], batch: list[list[str]], max_length: int | None = None) -> "np.ndarray":  # noqa: F821
        import numpy as np

        # SentEval passes word-tokenised sentences; rejoin to strings
        if batch and batch[0] and isinstance(batch[0][0], bytes):
            batch = [[w.decode("utf-8") for w in s] for s in batch]
        sentences = [" ".join(s) for s in batch]

        enc_kwargs: dict[str, Any] = dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length or args.max_length,
        )
        encoded = tokenizer(sentences, **enc_kwargs)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            if is_baseline:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                embeddings = _apply_pooler(outputs, attention_mask, args.pooler)
            else:
                # SyntaxBertModel: BERT-only inference
                embeddings = model.encode_sentences(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

        return embeddings.cpu().float().numpy()

    return batcher


def _apply_pooler(outputs, attention_mask: torch.Tensor, pooler: str) -> torch.Tensor:
    """Apply pooling strategy to HuggingFace model outputs."""
    last_hidden = outputs.last_hidden_state
    if pooler == "cls":
        return outputs.pooler_output
    elif pooler == "cls_before_pooler":
        return last_hidden[:, 0]
    elif pooler == "avg":
        return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
    elif pooler in ("avg_top2", "avg_first_last"):
        hidden_states = outputs.hidden_states
        if pooler == "avg_first_last":
            first, last = hidden_states[1], hidden_states[-1]
        else:
            first, last = hidden_states[-2], hidden_states[-1]
        pooled = (first + last) / 2.0
        return (pooled * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
    else:
        raise NotImplementedError(f"Unknown pooler: {pooler}")


# =============================================================================
# Result printing
# =============================================================================


def print_sts_results(results: dict, mode: str) -> dict[str, float]:
    """Print STS results table and return a flat score dict."""
    sts_tasks = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"]
    task_names, scores = [], []

    for task in sts_tasks:
        if task not in results:
            continue
        task_names.append(task)
        if task in ("STSBenchmark", "SICKRelatedness"):
            # These tasks have dev/test splits
            split = "dev" if mode == "dev" else "test"
            scores.append(results[task][split]["spearman"].correlation * 100)
        else:
            # STS12–16: no dev split, always use "all"
            scores.append(results[task]["all"]["spearman"]["all"] * 100)

    if task_names:
        avg = sum(scores) / len(scores)
        task_names.append("Avg.")
        scores.append(avg)
        print(f"\n------ STS ({mode}) ------")
        _print_table(task_names, [f"{s:.2f}" for s in scores])

    return {n: s for n, s in zip(task_names, scores)}


def print_transfer_results(results: dict, mode: str) -> dict[str, float]:
    """Print transfer task results table and return a flat score dict."""
    transfer_tasks = ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]
    task_names, scores = [], []

    for task in transfer_tasks:
        if task not in results:
            continue
        task_names.append(task)
        if mode == "dev":
            scores.append(results[task].get("devacc", 0.0))
        else:
            scores.append(results[task].get("acc", 0.0))

    if task_names:
        avg = sum(scores) / len(scores)
        task_names.append("Avg.")
        scores.append(avg)
        print(f"\n------ Transfer ({mode}) ------")
        _print_table(task_names, [f"{s:.2f}" for s in scores])

    return {n: s for n, s in zip(task_names, scores)}


def print_probing_results(results: dict, mode: str) -> dict[str, float]:
    """Print probing task results table and return a flat score dict."""
    probing_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift',
                     'Tense', 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    task_names, scores = [], []

    for task in probing_tasks:
        if task not in results:
            continue
        task_names.append(task)
        if mode == "dev":
            scores.append(results[task].get("devacc", 0.0))
        else:
            scores.append(results[task].get("acc", 0.0))

    if task_names:
        avg = sum(scores) / len(scores)
        task_names.append("Avg.")
        scores.append(avg)
        print(f"\n------ Probing ({mode}) ------")
        _print_table(task_names, [f"{s:.2f}" for s in scores])

    return {n: s for n, s in zip(task_names, scores)}


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    # --- SentEval path ---
    senteval_path = Path(args.senteval_path)
    if not senteval_path.exists():
        logger.error(
            f"SentEval not found at {senteval_path}. "
            "Clone it with: git clone https://github.com/facebookresearch/SentEval.git"
        )
        sys.exit(1)
    sys.path.insert(0, str(senteval_path))
    import senteval  # type: ignore[import]  # noqa: PLC0415

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Model ---
    if args.baseline:
        from transformers import AutoModel

        logger.info(f"Loading baseline model: {args.model_path}")
        model = AutoModel.from_pretrained(args.model_path, output_hidden_states=True)
        model.eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        is_baseline = True
    else:
        logger.info(f"Loading SyntaxBertModel from: {args.model_path}")
        model = _load_syntax_bert_model(args.model_path, device)
        tokenizer = AutoTokenizer.from_pretrained(
            _resolve_bert_path(args.model_path), use_fast=True
        )
        is_baseline = False

    # --- Task list ---
    if args.tasks is not None:
        tasks = args.tasks
    elif args.task_set == "sts":
        tasks = ["STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness"]
    elif args.task_set == "transfer":
        tasks = ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC"]
    elif args.task_set == "probing":
        tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                 'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
    elif args.task_set == "full":
        tasks = [
            "STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICKRelatedness",
            "MR", "CR", "MPQA", "SUBJ", "SST2", "TREC", "MRPC",
            'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
            'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion',
        ]
    else:
        tasks = []

    # --- SentEval params ---
    if args.mode in ("dev", "fasttest"):
        params = {
            "task_path": str(senteval_path / "data"),
            "usepytorch": True,
            "kfold": 5,
            "classifier": {"nhid": 0, "optim": "rmsprop", "batch_size": 128, "tenacity": 3, "epoch_size": 2},
        }
    else:
        params = {
            "task_path": str(senteval_path / "data"),
            "usepytorch": True,
            "kfold": 10,
            "classifier": {"nhid": 0, "optim": "adam", "batch_size": 64, "tenacity": 5, "epoch_size": 4},
        }

    batcher = build_batcher(model, tokenizer, device, args, is_baseline)

    # --- Run evaluation ---
    raw_results: dict[str, Any] = {}
    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare=lambda p, s: None)
        raw_results[task] = se.eval(task)

    # --- Print & collect scores ---
    flat_scores: dict[str, float] = {}
    flat_scores.update(print_sts_results(raw_results, args.mode))
    flat_scores.update(print_transfer_results(raw_results, args.mode))
    flat_scores.update(print_probing_results(raw_results, args.mode))

    # --- Save JSON summary ---
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"mode": args.mode, "task_set": args.task_set, "scores": flat_scores}, f, indent=2)
        logger.info(f"Scores saved to {out_path}")


def _resolve_bert_path(model_path: str) -> str:
    """Return the BERT tokenizer path from a SyntaxBertModel checkpoint directory.

    Looks for ``bert_weights/`` or ``bert_inference/`` sub-directories first,
    then falls back to the checkpoint directory itself (which may be a raw
    HuggingFace model name).

    If the resolved directory has no tokenizer files, falls back to the
    ``model_name_or_path`` from the BERT config (e.g. ``bert-base-uncased``).
    """
    ckpt_dir = Path(model_path)
    for subdir_name in ("bert_weights", "bert_inference"):
        subdir = ckpt_dir / subdir_name
        if subdir.exists():
            # Check if tokenizer files exist here
            if (subdir / "tokenizer.json").exists() or (subdir / "vocab.txt").exists():
                return str(subdir)
            # No tokenizer files — fall through to config-based fallback below
            break

    # Check the base directory for tokenizer files
    if (ckpt_dir / "tokenizer.json").exists() or (ckpt_dir / "vocab.txt").exists():
        return str(ckpt_dir)

    # No tokenizer files found — try to read model_type from config.json and
    # resolve to a known HuggingFace hub model (e.g. bert-base-uncased)
    config_path = ckpt_dir / "config.json"
    for candidate in (ckpt_dir, ckpt_dir / "bert_weights", ckpt_dir / "bert_inference"):
        cfg_file = candidate / "config.json"
        if cfg_file.exists():
            config_path = cfg_file
            break

    if config_path.exists():
        import json as _json
        with open(config_path) as f:
            cfg = _json.load(f)
        # Use _name_or_path if available (set by save_pretrained),
        # otherwise fall back to model_type-base-uncased
        name = cfg.get("_name_or_path", "")
        if name and not name.startswith("/"):
            logger.info(f"No tokenizer files at {ckpt_dir}; using tokenizer from '{name}'")
            return name
        # Last resort: infer from model_type (e.g. "bert" → "bert-base-uncased")
        model_type = cfg.get("model_type", "")
        if model_type:
            fallback = f"{model_type}-base-uncased"
            logger.info(f"No tokenizer files at {ckpt_dir}; falling back to '{fallback}'")
            return fallback

    return model_path


if __name__ == "__main__":
    main()
