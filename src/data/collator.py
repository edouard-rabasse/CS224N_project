"""
collator.py — Graph-aware dataset and data collator for GNN-Syntax-BERT.

SyntaxGraphDataset:
    Pairs raw sentences with pre-parsed dependency graphs (loaded from
    StanzaSyntaxParser.batch_parse() output).

SyntaxGraphCollator:
    Builds SimCSE-style text batches — shape (batch_size, 2, seq_len) — paired
    with PyG Batch objects and subword-to-word alignment maps, ready to be
    consumed by SyntaxBertModel.forward().
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.processing.syntax_parser import StanzaSyntaxParser

logger = logging.getLogger(__name__)


class SyntaxGraphDataset(Dataset):
    """Dataset that pairs sentences with their pre-parsed dependency graphs.

    Expects the output of StanzaSyntaxParser.batch_parse() as the
    ``parsed_graphs`` argument: a list of dicts with keys
    ``tokens``, ``edges_src``, ``edges_dst``, ``num_tokens``.

    Args:
        sentences: Raw text sentences (one per example).
        parsed_graphs: Parse result dicts, aligned 1-to-1 with ``sentences``.
        max_samples: Cap on the dataset size. ``None`` = use all.
    """

    def __init__(
        self,
        sentences: list[str],
        parsed_graphs: list[dict],
        max_samples: int | None = None,
    ) -> None:
        n = min(len(sentences), len(parsed_graphs))
        if len(sentences) != len(parsed_graphs):
            logger.warning(
                f"Sentence count ({len(sentences)}) != parsed graph count ({len(parsed_graphs)}). "
                f"Using the first {n} from each."
            )
        if max_samples is not None:
            n = min(n, max_samples)

        self.sentences = sentences[:n]
        self.parsed_graphs = parsed_graphs[:n]
        logger.info(f"SyntaxGraphDataset: {n:,} examples")

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "sentence": self.sentences[idx],
            "parse_result": self.parsed_graphs[idx],
            "idx": idx,
        }


class SyntaxGraphCollator:
    """Data collator that builds SimCSE-style batches paired with PyG graph batches.

    For each batch of sentences this collator:

    1. Tokenises sentences and stacks to ``(batch_size, 2, seq_len)``
       (same tokens repeated twice so SimCSE's dropout masks create two views).
    2. Converts each parse result to a PyG ``Data`` object and forms a
       ``Batch`` for the whole batch.
    3. Computes BERT-subword → Stanza-word alignment maps used to aggregate
       BERT hidden states into per-word node features for the GNN.
    4. Optionally builds an edge-dropout-augmented graph view for the GNN
       auxiliary contrastive loss (``L_GNN``).

    Args:
        tokenizer: HuggingFace tokenizer matching the BERT model in use.
        max_seq_length: Token sequence length (padding + truncation target).
        edge_drop_rate: Fraction of edges randomly dropped for GNN augmentation.
            Set to 0.0 to disable augmented views (saves compute when
            ``mu_gnn == 0``).
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 32,
        edge_drop_rate: float = 0.1,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.edge_drop_rate = edge_drop_rate

    def __call__(self, batch_items: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a list of dataset items into a model-ready batch.

        Args:
            batch_items: List of dicts from ``SyntaxGraphDataset.__getitem__``.

        Returns:
            Dict with keys:

            - ``input_ids``:         ``(bs, 2, seq_len)`` — same tokens twice (SimCSE)
            - ``attention_mask``:    ``(bs, 2, seq_len)``
            - ``graph_batch``:       PyG ``Batch`` of ``bs`` dependency graphs
            - ``graph_batch_aug``:   PyG ``Batch`` with edge dropout, or ``None``
            - ``token_to_word_maps``: ``list[list[int]]`` length ``bs``
        """
        sentences = [item["sentence"] for item in batch_items]
        parse_results = [item["parse_result"] for item in batch_items]

        # ---- Tokenise ------------------------------------------------
        encoded = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = encoded["input_ids"]      # (bs, seq_len)
        attention_mask: torch.Tensor = encoded["attention_mask"]  # (bs, seq_len)

        # SimCSE expects (bs, 2, seq_len) — same input, different dropout views
        input_ids_2x = torch.stack([input_ids, input_ids], dim=1)          # (bs, 2, seq_len)
        attention_mask_2x = torch.stack([attention_mask, attention_mask], dim=1)

        # ---- Subword → word alignment maps ---------------------------
        token_to_word_maps: list[list[int]] = []
        for i, parse in enumerate(parse_results):
            bert_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
            stanza_tokens = parse.get("tokens", [])
            alignment = StanzaSyntaxParser.align_subwords(stanza_tokens, bert_tokens)
            token_to_word_maps.append(alignment)

        # ---- Build PyG graphs ----------------------------------------
        graphs = [self._parse_to_data(p) for p in parse_results]
        graph_batch = Batch.from_data_list(graphs)

        # ---- Augmented graph view (edge dropout for L_GNN) -----------
        graph_batch_aug: Batch | None = None
        if self.edge_drop_rate > 0:
            from src.alignment.losses import GNNContrastiveLoss

            aug_graphs = []
            for g in graphs:
                aug_ei = GNNContrastiveLoss.drop_edges(g.edge_index, self.edge_drop_rate)
                aug_graphs.append(Data(edge_index=aug_ei, num_nodes=g.num_nodes))
            graph_batch_aug = Batch.from_data_list(aug_graphs)

        return {
            "input_ids": input_ids_2x,
            "attention_mask": attention_mask_2x,
            "graph_batch": graph_batch,
            "graph_batch_aug": graph_batch_aug,
            "token_to_word_maps": token_to_word_maps,
        }

    @staticmethod
    def _parse_to_data(parse_result: dict) -> Data:
        """Convert a cached parse result dict to a PyG Data (no node features).

        Node features are populated later from BERT hidden states during
        ``SyntaxBertModel._compute_gnn_embeddings()``.
        """
        num_nodes = parse_result.get("num_tokens", len(parse_result.get("tokens", [])))
        num_nodes = max(num_nodes, 1)  # guard against empty parses
        edges_src = parse_result.get("edges_src", [])
        edges_dst = parse_result.get("edges_dst", [])

        if edges_src:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(edge_index=edge_index, num_nodes=num_nodes)
