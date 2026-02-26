"""
parse_graphs.py — Batch parse dependency graphs from the training corpus.

Runs Stanza's dependency parser on wiki1m_for_simcse.txt (or any text file)
and saves chunked .jsonl output to data/parsed_graphs/, ready for the
SyntaxCLTrainer.  Chunk files are named parsed_XXXXX.jsonl; already-existing
chunks are skipped, so the job is safely resumable.

Resource guide
--------------
  Single CPU  : ~8-12 h for 1M sentences
  4 CPU workers: ~2-3 h  (each worker needs ~2 GB RAM for Stanza)
  GPU          : 5-10× faster than CPU — use --gpu when available

Usage
-----
    # Parse everything (defaults to 1M sentences):
    uv run python scripts/parse_graphs.py

    # Quick test with a small subset:
    uv run python scripts/parse_graphs.py --max-sentences 5000

    # Use GPU (recommended on a compute node):
    uv run python scripts/parse_graphs.py --gpu

    # Multiple CPU workers (4 workers × ~2 GB RAM each):
    uv run python scripts/parse_graphs.py --num-workers 4

    # Custom paths:
    uv run python scripts/parse_graphs.py \\
        --input data/wiki1m_for_simcse.txt \\
        --output data/parsed_graphs/ \\
        --chunk-size 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Absolute path to the project root (parent of scripts/)
_PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Worker helpers (must be top-level for multiprocessing pickling on Windows)
# ---------------------------------------------------------------------------

_worker_parser = None  # module-level so each worker process holds its own


def _worker_init(use_gpu: bool) -> None:
    """Initialize one Stanza pipeline per worker process (called once)."""
    global _worker_parser
    from src.processing.syntax_parser import StanzaSyntaxParser

    _worker_parser = StanzaSyntaxParser(use_gpu=use_gpu)


def _parse_and_save_chunk(task: tuple[list[str], Path]) -> tuple[Path, int, int]:
    """Parse a list of sentences and write them to a single .jsonl chunk file.

    Args:
        task: (sentences, output_path) pair.

    Returns:
        (output_path, n_parsed, n_failed) tuple.
    """
    sentences, out_path = task
    global _worker_parser

    results: list[dict] = []
    n_failed = 0

    for sent in sentences:
        sent = sent.strip()
        try:
            parse = _worker_parser.parse_sentence(sent)  # type: ignore[union-attr]
            parse["sentence"] = sent
            results.append(parse)
        except Exception as exc:
            n_failed += 1
            logger.warning(f"Parse failed: {sent[:60]!r} — {exc}")
            # Fallback: whitespace-split tokens, no edges
            words = sent.split()
            results.append(
                {
                    "tokens": words,
                    "edges_src": [],
                    "edges_dst": [],
                    "deprels": [],
                    "pos_tags": [],
                    "num_tokens": max(len(words), 1),
                    "sentence": sent,
                    "parse_error": str(exc),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return out_path, len(results), n_failed


# ---------------------------------------------------------------------------
# Main parsing orchestrator
# ---------------------------------------------------------------------------


def parse_corpus(
    input_file: Path,
    output_dir: Path,
    max_sentences: int | None,
    chunk_size: int,
    num_workers: int,
    use_gpu: bool,
) -> None:
    # ---- Load sentences ----
    logger.info(f"Loading sentences from {input_file}")
    with open(input_file, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    if max_sentences is not None:
        sentences = sentences[:max_sentences]
    logger.info(f"Sentences to parse: {len(sentences):,}")

    # ---- Build chunk tasks, skipping already-completed chunks ----
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[list[str], Path]] = []
    total_chunks = (len(sentences) + chunk_size - 1) // chunk_size

    for chunk_idx in range(total_chunks):
        chunk_path = output_dir / f"parsed_{chunk_idx:05d}.jsonl"
        if chunk_path.exists():
            logger.info(f"  Chunk {chunk_idx:05d} already exists — skipping")
            continue
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(sentences))
        tasks.append((sentences[start:end], chunk_path))

    if not tasks:
        logger.info("All chunks already parsed. Nothing to do.")
        _write_metadata(output_dir, len(sentences), total_chunks, chunk_size)
        return

    logger.info(f"Chunks to parse: {len(tasks)} of {total_chunks} total")

    # ---- Parse ----
    t0 = time.time()
    total_parsed = 0
    total_failed = 0

    if num_workers <= 1:
        # Single-process path — simpler, less memory overhead
        _worker_init(use_gpu)
        for task in tqdm(tasks, desc="Parsing chunks", unit="chunk"):
            out_path, n_parsed, n_failed = _parse_and_save_chunk(task)
            total_parsed += n_parsed
            total_failed += n_failed
            logger.info(f"  Saved {out_path.name}: {n_parsed} sentences, {n_failed} failures")
    else:
        # Multi-process path — one Stanza pipeline per worker
        logger.info(
            f"Spawning {num_workers} worker processes "
            f"(~{num_workers * 2} GB RAM required)"
        )
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(use_gpu,),
        ) as pool:
            for out_path, n_parsed, n_failed in tqdm(
                pool.imap_unordered(_parse_and_save_chunk, tasks),
                total=len(tasks),
                desc="Parsing chunks",
                unit="chunk",
            ):
                total_parsed += n_parsed
                total_failed += n_failed
                logger.info(
                    f"  Saved {out_path.name}: {n_parsed} sentences, {n_failed} failures"
                )

    elapsed = time.time() - t0
    logger.info(
        f"Parsing complete in {elapsed / 60:.1f} min — "
        f"{total_parsed:,} parsed, {total_failed:,} failed"
    )

    _write_metadata(output_dir, len(sentences), total_chunks, chunk_size)


def _write_metadata(output_dir: Path, total: int, num_chunks: int, chunk_size: int) -> None:
    meta = {"total_sentences": total, "num_chunks": num_chunks, "chunk_size": chunk_size}
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata written to {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse dependency graphs from the training corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_PROJECT_ROOT / "data" / "wiki1m_for_simcse.txt",
        help="Input text file (one sentence per line). Default: data/wiki1m_for_simcse.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "data" / "parsed_graphs",
        help="Output directory for parsed .jsonl chunks. Default: data/parsed_graphs/",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        metavar="N",
        help="Parse only the first N sentences (useful for quick tests).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        metavar="N",
        help="Sentences per chunk file (default: 10 000).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel Stanza processes (default: 1). "
            "Each worker uses ~2 GB RAM. Avoid exceeding available memory."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run Stanza on GPU (much faster when available).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            "Download it first:  uv run python scripts/download_wiki.py"
        )

    parse_corpus(
        input_file=args.input,
        output_dir=args.output,
        max_sentences=args.max_sentences,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
