"""
syntax_parser.py — Syntactic Dependency Parsing Pipeline

Converts raw sentences into PyTorch Geometric graph objects using Stanza's
dependency parser. Handles the critical subword-to-word alignment between
BERT's WordPiece tokenization and Stanza's word-level tokenization.

Pipeline:
    1. Parse sentence with Stanza → dependency tree (head, deprel per token)
    2. Build edge_index from dependency arcs (optionally bidirectional)
    3. Align BERT subword tokens to Stanza words (first-subword / mean-pool)
    4. Return a PyG Data object ready for the GNN encoder

Usage:
    # Single sentence
    parser = StanzaSyntaxParser()
    parse = parser.parse_sentence("The cat sat on the mat.")
    data = parser.to_pyg_data(parse)

    # Batch processing for training
    python -m src.processing.syntax_parser \\
        --input data/wiki1m_for_simcse.txt \\
        --output data/parsed_graphs/ \\
        --num-workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StanzaSyntaxParser:
    """Dependency parser using Stanza for constructing syntactic graphs.

    Attributes:
        nlp: Stanza NLP pipeline with dependency parsing.
        add_reverse_edges: Whether to add reverse edges (child → parent)
                           for bidirectional message passing.
    """

    def __init__(
        self,
        lang: str = "en",
        add_reverse_edges: bool = True,
        use_gpu: bool = False,
    ) -> None:
        import stanza

        self.add_reverse_edges = add_reverse_edges
        self.nlp = stanza.Pipeline(
            lang=lang,
            processors="tokenize,pos,lemma,depparse",
            use_gpu=use_gpu,
            logging_level="WARN",
        )
        logger.info(f"Stanza pipeline initialized (lang={lang}, gpu={use_gpu})")

    def parse_sentence(self, sentence: str) -> dict[str, Any]:
        """Parse a single sentence into a structured dependency representation.

        Args:
            sentence: Raw text sentence.

        Returns:
            Dict with keys:
                - "tokens": list of word strings
                - "edges_src": list of source node indices (0-indexed)
                - "edges_dst": list of destination node indices (0-indexed)
                - "deprels": list of dependency relation labels
                - "pos_tags": list of POS tags
                - "num_tokens": number of tokens
        """
        doc = self.nlp(sentence)

        tokens = []
        edges_src = []
        edges_dst = []
        deprels = []
        pos_tags = []

        for sent in doc.sentences:
            for word in sent.words:
                tokens.append(word.text)
                pos_tags.append(word.upos)

                if word.head > 0:  # head=0 is ROOT (no real parent node)
                    parent_idx = word.head - 1  # convert to 0-indexed
                    child_idx = word.id - 1  # convert to 0-indexed
                    # Directed edge: parent → child
                    edges_src.append(parent_idx)
                    edges_dst.append(child_idx)
                    deprels.append(word.deprel)

                    # Optional reverse edge: child → parent
                    if self.add_reverse_edges:
                        edges_src.append(child_idx)
                        edges_dst.append(parent_idx)
                        deprels.append(f"rev_{word.deprel}")

        return {
            "tokens": tokens,
            "edges_src": edges_src,
            "edges_dst": edges_dst,
            "deprels": deprels,
            "pos_tags": pos_tags,
            "num_tokens": len(tokens),
        }

    @staticmethod
    def to_pyg_data(
        parse_result: dict[str, Any],
        token_embeddings: torch.Tensor | None = None,
    ) -> Data:
        """Convert a parse result to a PyG Data object.

        Args:
            parse_result: Output of parse_sentence().
            token_embeddings: Optional (num_tokens, hidden_dim) tensor.
                If None, node features are left unset (will be populated
                from BERT hidden states at training time).

        Returns:
            PyG Data object with edge_index and optionally x.
        """
        num_nodes = parse_result["num_tokens"]

        if len(parse_result["edges_src"]) > 0:
            edge_index = torch.tensor(
                [parse_result["edges_src"], parse_result["edges_dst"]],
                dtype=torch.long,
            )
        else:
            # Isolated node (single-word sentence or parse failure)
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(
            edge_index=edge_index,
            num_nodes=num_nodes,
        )

        if token_embeddings is not None:
            assert token_embeddings.size(0) == num_nodes, (
                f"Expected {num_nodes} node features, got {token_embeddings.size(0)}"
            )
            data.x = token_embeddings

        return data

    @staticmethod
    def align_subwords(
        stanza_tokens: list[str],
        bert_tokens: list[str],
    ) -> list[int]:
        """Map each BERT subword token to its corresponding Stanza word index.

        Uses greedy character-span matching: tracks character positions consumed
        by each Stanza word and matches BERT subwords to the active Stanza word.

        Special tokens ([CLS], [SEP], [PAD]) are mapped to -1.

        Args:
            stanza_tokens: Word-level tokens from Stanza (e.g., ["The", "cat", "sat"]).
            bert_tokens: Subword tokens from BERT tokenizer (e.g., ["[CLS]", "the", "cat", "sat", "[SEP]"]).

        Returns:
            List of length len(bert_tokens), where each element is the Stanza word
            index (0-based) or -1 for special tokens.
        """
        alignment = []
        stanza_idx = 0
        stanza_chars_consumed = 0
        stanza_word_lower = stanza_tokens[0].lower() if stanza_tokens else ""

        for bt in bert_tokens:
            # Skip special tokens
            if bt in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                alignment.append(-1)
                continue

            # Remove WordPiece prefix
            clean_bt = bt[2:] if bt.startswith("##") else bt

            if stanza_idx < len(stanza_tokens):
                alignment.append(stanza_idx)
                stanza_chars_consumed += len(clean_bt)

                # Check if we've consumed the full Stanza word
                if stanza_chars_consumed >= len(stanza_word_lower):
                    stanza_idx += 1
                    stanza_chars_consumed = 0
                    if stanza_idx < len(stanza_tokens):
                        stanza_word_lower = stanza_tokens[stanza_idx].lower()
            else:
                # Overflow: more BERT tokens than Stanza words
                alignment.append(-1)

        return alignment

    @staticmethod
    def aggregate_subword_embeddings(
        bert_hidden: torch.Tensor,
        alignment: list[int],
        num_words: int,
    ) -> torch.Tensor:
        """Mean-pool BERT subword embeddings per Stanza word.

        Args:
            bert_hidden: (seq_len, hidden_dim) — BERT last hidden state for one sentence.
            alignment: Output of align_subwords(), length = seq_len.
            num_words: Number of Stanza words.

        Returns:
            (num_words, hidden_dim) — word-level embeddings.
        """
        hidden_dim = bert_hidden.size(-1)
        word_feats = torch.zeros(num_words, hidden_dim, device=bert_hidden.device, dtype=bert_hidden.dtype)
        word_counts = torch.zeros(num_words, device=bert_hidden.device)

        for subword_idx, word_idx in enumerate(alignment):
            if word_idx >= 0:
                word_feats[word_idx] += bert_hidden[subword_idx]
                word_counts[word_idx] += 1

        # Avoid division by zero for unmatched words
        word_counts = word_counts.clamp(min=1)
        word_feats = word_feats / word_counts.unsqueeze(-1)
        return word_feats

    def batch_parse(
        self,
        sentences: list[str],
        output_dir: str,
        chunk_size: int = 10_000,
    ) -> None:
        """Parse a large list of sentences and cache results as .jsonl chunks.

        Saves parse results (tokens, edges, deprels, POS) as JSON lines files
        for efficient loading during training. The PyG Data objects are
        constructed at training time from these cached parses.

        Args:
            sentences: List of raw text sentences.
            output_dir: Directory to save parsed chunks.
            chunk_size: Number of sentences per chunk file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total = len(sentences)
        num_chunks = (total + chunk_size - 1) // chunk_size

        logger.info(f"Parsing {total:,} sentences into {num_chunks} chunks (chunk_size={chunk_size:,})")

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total)
            chunk_sentences = sentences[start:end]

            chunk_file = output_path / f"parsed_{chunk_idx:05d}.jsonl"
            if chunk_file.exists():
                logger.info(f"  Chunk {chunk_idx} already exists, skipping")
                continue

            results = []
            for sent in tqdm(
                chunk_sentences,
                desc=f"Chunk {chunk_idx}/{num_chunks}",
                leave=False,
            ):
                try:
                    parse = self.parse_sentence(sent.strip())
                    parse["sentence"] = sent.strip()
                    results.append(parse)
                except Exception as e:
                    logger.warning(f"Failed to parse: {sent[:80]}... — {e}")
                    # Store a fallback: single node, no edges
                    results.append(
                        {
                            "tokens": sent.strip().split(),
                            "edges_src": [],
                            "edges_dst": [],
                            "deprels": [],
                            "pos_tags": [],
                            "num_tokens": len(sent.strip().split()),
                            "sentence": sent.strip(),
                            "parse_error": str(e),
                        }
                    )

            with open(chunk_file, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            logger.info(f"  Saved chunk {chunk_idx}: {len(results):,} parses → {chunk_file}")

        # Save metadata
        meta = {
            "total_sentences": total,
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Parsing complete. Metadata saved to {output_path / 'metadata.json'}")


def main() -> None:
    """CLI entry point for batch parsing."""
    parser = argparse.ArgumentParser(description="Parse sentences from a text file into dependency graphs")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input text file (one sentence per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for parsed graph chunks",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Sentences per chunk file (default: 10000)",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Maximum number of sentences to parse (default: all)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for Stanza parsing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load sentences
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading sentences from {input_path}")
    with open(input_path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    if args.max_sentences:
        sentences = sentences[: args.max_sentences]
        logger.info(f"Truncated to {len(sentences):,} sentences")
    else:
        logger.info(f"Loaded {len(sentences):,} sentences")

    # Parse
    syntax_parser = StanzaSyntaxParser(use_gpu=args.gpu)
    syntax_parser.batch_parse(sentences, args.output, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
