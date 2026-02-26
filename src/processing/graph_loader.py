"""
graph_loader.py — Loading and converting pre-parsed dependency graphs.

Provides:
- load_parsed_graphs: reads cached .jsonl chunk files from disk
- parse_to_pyg_data: converts a parse result dict to a PyG Data object
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def load_parsed_graphs(parsed_dir: str) -> list[dict]:
    """Load pre-parsed dependency graphs from cached .jsonl files.

    Args:
        parsed_dir: Directory containing parsed_XXXXX.jsonl chunk files.

    Returns:
        List of parse result dicts (one per sentence).
    """
    parsed_path = Path(parsed_dir)
    if not parsed_path.exists():
        logger.warning(f"Parsed graphs directory not found: {parsed_dir}")
        logger.warning(
            "Run: uv run python -m src.processing.syntax_parser "
            "--input data/wiki1m_for_simcse.txt --output data/parsed_graphs/"
        )
        return []

    graphs = []
    chunk_files = sorted(parsed_path.glob("parsed_*.jsonl"))
    logger.info(f"Loading parsed graphs from {len(chunk_files)} chunk files...")

    for chunk_file in chunk_files:
        with open(chunk_file, encoding="utf-8") as f:
            for line in f:
                graphs.append(json.loads(line))

    logger.info(f"Loaded {len(graphs):,} parsed graphs")
    return graphs


def parse_to_pyg_data(parse_result: dict) -> Data:
    """Convert a cached parse result dict to a PyG Data object.

    Node features (x) are left unset — they will be populated from
    BERT hidden states at training time.

    Args:
        parse_result: Dict with keys: edges_src, edges_dst, num_tokens.

    Returns:
        PyG Data with edge_index and num_nodes.
    """
    from src.processing.syntax_parser import StanzaSyntaxParser

    return StanzaSyntaxParser.to_pyg_data(parse_result)
