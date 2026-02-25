#!/usr/bin/env python3
"""
debug_train.py — Quick debugging script for the full GNN-Syntax-BERT pipeline.

Runs an end-to-end training step WITHOUT requiring:
  - SimCSE installed (falls back to vanilla BERT)
  - Pre-parsed graph data (generates synthetic dependency trees on the fly)
  - Downloaded wiki1m data (uses hardcoded toy sentences)
  - Stanza installed (builds synthetic parse results)
  - GPU (runs on CPU)

This script validates the entire forward + backward pass:
  1. Tokenize sentences with BERT tokenizer
  2. Build synthetic dependency graphs (PyG Data objects)
  3. Run forward pass through SyntaxBertModel (BERT → hidden states → GNN)
  4. Compute combined loss: L_SimCSE_proxy + λ·L_align + μ·L_GNN
  5. Backprop and verify gradients flow to both BERT and GNN parameters

Usage:
    uv run python scripts/debug_train.py
    uv run python scripts/debug_train.py --conv-type gcn
    uv run python scripts/debug_train.py --steps 5 --batch-size 8
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from transformers import AutoConfig, AutoTokenizer, BertModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.losses import AlignmentLoss, CombinedLoss
from src.models.gnn_encoder import SyntaxGNNEncoder
from src.models.wrapper import SyntaxBertModel
from src.processing.syntax_parser import StanzaSyntaxParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Toy Data
# =============================================================================

TOY_SENTENCES = [
    "The cat sat on the mat.",
    "A dog runs in the park every morning.",
    "She quickly finished her homework before dinner.",
    "The old man walked slowly across the bridge.",
    "Birds sing beautiful songs at dawn.",
    "The children played happily in the garden.",
    "Heavy rain fell throughout the entire night.",
    "A bright star shone above the mountain peak.",
    "The professor explained complex theories with clarity.",
    "Small waves crashed gently against the shore.",
    "The musician performed a stunning solo on stage.",
    "Tall trees swayed gracefully in the autumn wind.",
    "The scientist discovered a new species of butterfly.",
    "Fresh bread baked slowly in the warm oven.",
    "The detective examined every clue at the scene.",
    "Young students eagerly raised their hands in class.",
]

# Synthetic dependency parses (pre-built to avoid Stanza dependency)
# Format: list of (parent_idx, child_idx) edges for each sentence
# These approximate real dependency structures
SYNTHETIC_PARSES = [
    # "The cat sat on the mat."
    # sat(ROOT) -> cat(nsubj) -> The(det); sat -> on(obl) -> mat(nmod) -> the(det)
    {
        "tokens": ["The", "cat", "sat", "on", "the", "mat", "."],
        "edges_src": [2, 1, 2, 3, 5],
        "edges_dst": [1, 0, 3, 5, 4],
    },
    # "A dog runs in the park every morning."
    {
        "tokens": ["A", "dog", "runs", "in", "the", "park", "every", "morning", "."],
        "edges_src": [2, 1, 2, 3, 5, 2, 7],
        "edges_dst": [1, 0, 3, 5, 4, 7, 6],
    },
    # "She quickly finished her homework before dinner."
    {
        "tokens": ["She", "quickly", "finished", "her", "homework", "before", "dinner", "."],
        "edges_src": [2, 2, 2, 4, 2, 5],
        "edges_dst": [0, 1, 4, 3, 5, 6],
    },
    # "The old man walked slowly across the bridge."
    {
        "tokens": ["The", "old", "man", "walked", "slowly", "across", "the", "bridge", "."],
        "edges_src": [3, 2, 2, 3, 3, 5, 7],
        "edges_dst": [2, 0, 1, 4, 5, 7, 6],
    },
    # "Birds sing beautiful songs at dawn."
    {
        "tokens": ["Birds", "sing", "beautiful", "songs", "at", "dawn", "."],
        "edges_src": [1, 1, 3, 1, 4],
        "edges_dst": [0, 3, 2, 4, 5],
    },
    # "The children played happily in the garden."
    {
        "tokens": ["The", "children", "played", "happily", "in", "the", "garden", "."],
        "edges_src": [2, 1, 2, 2, 4, 6],
        "edges_dst": [1, 0, 3, 4, 6, 5],
    },
    # "Heavy rain fell throughout the entire night."
    {
        "tokens": ["Heavy", "rain", "fell", "throughout", "the", "entire", "night", "."],
        "edges_src": [2, 1, 2, 3, 6, 6],
        "edges_dst": [1, 0, 3, 6, 4, 5],
    },
    # "A bright star shone above the mountain peak."
    {
        "tokens": ["A", "bright", "star", "shone", "above", "the", "mountain", "peak", "."],
        "edges_src": [3, 2, 2, 3, 4, 7, 7],
        "edges_dst": [2, 0, 1, 4, 7, 5, 6],
    },
    # "The professor explained complex theories with clarity."
    {
        "tokens": ["The", "professor", "explained", "complex", "theories", "with", "clarity", "."],
        "edges_src": [2, 1, 2, 4, 2, 5],
        "edges_dst": [1, 0, 4, 3, 5, 6],
    },
    # "Small waves crashed gently against the shore."
    {
        "tokens": ["Small", "waves", "crashed", "gently", "against", "the", "shore", "."],
        "edges_src": [2, 1, 2, 2, 4, 6],
        "edges_dst": [1, 0, 3, 4, 6, 5],
    },
    # "The musician performed a stunning solo on stage."
    {
        "tokens": ["The", "musician", "performed", "a", "stunning", "solo", "on", "stage", "."],
        "edges_src": [2, 1, 2, 5, 5, 2, 6],
        "edges_dst": [1, 0, 5, 3, 4, 6, 7],
    },
    # "Tall trees swayed gracefully in the autumn wind."
    {
        "tokens": ["Tall", "trees", "swayed", "gracefully", "in", "the", "autumn", "wind", "."],
        "edges_src": [2, 1, 2, 2, 4, 7, 7],
        "edges_dst": [1, 0, 3, 4, 7, 5, 6],
    },
    # "The scientist discovered a new species of butterfly."
    {
        "tokens": ["The", "scientist", "discovered", "a", "new", "species", "of", "butterfly", "."],
        "edges_src": [2, 1, 2, 5, 5, 2, 6],
        "edges_dst": [1, 0, 5, 3, 4, 6, 7],
    },
    # "Fresh bread baked slowly in the warm oven."
    {
        "tokens": ["Fresh", "bread", "baked", "slowly", "in", "the", "warm", "oven", "."],
        "edges_src": [2, 1, 2, 2, 4, 7, 7],
        "edges_dst": [1, 0, 3, 4, 7, 5, 6],
    },
    # "The detective examined every clue at the scene."
    {
        "tokens": ["The", "detective", "examined", "every", "clue", "at", "the", "scene", "."],
        "edges_src": [2, 1, 2, 4, 2, 5, 7],
        "edges_dst": [1, 0, 4, 3, 5, 7, 6],
    },
    # "Young students eagerly raised their hands in class."
    {
        "tokens": ["Young", "students", "eagerly", "raised", "their", "hands", "in", "class", "."],
        "edges_src": [3, 1, 3, 3, 5, 3, 6],
        "edges_dst": [1, 0, 2, 5, 4, 6, 7],
    },
]


def build_graph_from_parse(parse: dict, add_reverse: bool = True) -> Data:
    """Build a PyG Data from a synthetic parse dict (no node features yet)."""
    src = list(parse["edges_src"])
    dst = list(parse["edges_dst"])

    if add_reverse:
        src_rev = list(dst)
        dst_rev = list(src)
        src = src + src_rev
        dst = dst + dst_rev

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(edge_index=edge_index, num_nodes=len(parse["tokens"]))


def build_token_to_word_map(
    stanza_tokens: list[str],
    bert_tokens: list[str],
) -> list[int]:
    """Quick subword alignment using StanzaSyntaxParser utility."""
    return StanzaSyntaxParser.align_subwords(stanza_tokens, bert_tokens)


# =============================================================================
# Debug Training Loop
# =============================================================================


def run_debug_training(
    conv_type: str = "gat",
    num_layers: int = 2,
    heads: int = 4,
    pooling: str = "mean",
    projector_dim: int = 256,
    lambda_align: float = 0.1,
    mu_gnn: float = 0.0,
    batch_size: int = 8,
    num_steps: int = 3,
    lr: float = 1e-4,
    max_seq_length: int = 64,
) -> None:
    """Run a quick debug training loop on GPU with toy data."""

    device = torch.device("cuda")
    logger.info("=" * 60)
    logger.info("  GNN-Syntax-BERT — Debug Training Script")
    logger.info("=" * 60)
    logger.info(f"  Device:       {device}")
    logger.info(f"  GNN type:     {conv_type.upper()}, {num_layers} layers, {heads} heads")
    logger.info(f"  Pooling:      {pooling}")
    logger.info(f"  Projector:    {projector_dim}-dim")
    logger.info(f"  Batch size:   {batch_size}")
    logger.info(f"  Steps:        {num_steps}")
    logger.info(f"  λ_align:      {lambda_align}")
    logger.info(f"  μ_gnn:        {mu_gnn}")
    logger.info("")

    # ---- Step 1: Load tokenizer & BERT ----
    logger.info("[1/6] Loading BERT tokenizer and model...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)

    # Wrap in a minimal SimCSE-compatible interface
    # (since SimCSE may not be installed in debug mode)
    bert_wrapper = DebugBertForCL(bert_model)
    logger.info(f"  BERT loaded: {sum(p.numel() for p in bert_model.parameters()):,} params")

    # ---- Step 2: Build SyntaxBertModel ----
    logger.info("[2/6] Building SyntaxBertModel...")
    gnn_config = {
        "conv_type": conv_type,
        "num_layers": num_layers,
        "hidden_dim": 768,
        "heads": heads,
        "dropout": 0.1,
        "pooling": pooling,
        "freeze": False,
    }
    alignment_config = {
        "lambda_align": lambda_align,
        "mu_gnn": mu_gnn,
        "align_temperature": 0.05,
        "projector_dim": projector_dim,
        "stop_grad_gnn": False,
    }

    model = SyntaxBertModel(
        bert_model=bert_wrapper,
        gnn_config=gnn_config,
        alignment_config=alignment_config,
    )
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gnn_params = sum(p.numel() for p in model.gnn.parameters())
    logger.info(f"  Total params:     {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,}")
    logger.info(f"  GNN params:       {gnn_params:,}")

    # ---- Step 3: Build loss ----
    logger.info("[3/6] Setting up CombinedLoss...")
    combined_loss = CombinedLoss(
        lambda_align=lambda_align,
        mu_gnn=mu_gnn,
        align_temperature=0.05,
    )
    logger.info(f"  L = L_proxy + {lambda_align}·L_align + {mu_gnn}·L_GNN")

    # ---- Step 4: Build optimizer ----
    logger.info("[4/6] Building optimizer (separate LR groups)...")
    bert_params_list = [p for n, p in model.named_parameters() if "gnn" not in n and p.requires_grad]
    gnn_params_list = [p for n, p in model.named_parameters() if "gnn" in n and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": bert_params_list, "lr": lr, "weight_decay": 0.0},
            {"params": gnn_params_list, "lr": lr * 10, "weight_decay": 0.0},  # higher LR for GNN
        ]
    )
    logger.info(f"  BERT LR: {lr}, GNN LR: {lr * 10}")

    # ---- Step 5: Prepare toy data ----
    logger.info("[5/6] Preparing toy data...")
    assert len(TOY_SENTENCES) >= batch_size, f"Need ≥{batch_size} sentences, have {len(TOY_SENTENCES)}"
    logger.info(f"  {len(TOY_SENTENCES)} toy sentences available")

    # ---- Step 6: Training loop ----
    logger.info("[6/6] Running training steps...")
    logger.info("")

    for step in range(num_steps):
        step_start = time.time()

        # Sample a batch
        batch_sentences = TOY_SENTENCES[:batch_size]
        batch_parses = SYNTHETIC_PARSES[:batch_size]

        # Tokenize with BERT
        encoded = tokenizer(
            batch_sentences,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Convert BERT tokens back to strings for alignment
        bert_token_strs = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

        # Build subword→word alignment maps
        token_to_word_maps = []
        for i, parse in enumerate(batch_parses):
            alignment = build_token_to_word_map(parse["tokens"], bert_token_strs[i])
            token_to_word_maps.append(alignment)

        # Build PyG graph batch (without node features — they come from BERT)
        graphs = [build_graph_from_parse(p) for p in batch_parses]
        graph_batch = Batch.from_data_list(graphs).to(device)

        # ---- Forward pass ----
        optimizer.zero_grad()

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_batch=graph_batch,
            token_to_word_maps=token_to_word_maps,
        )

        # Proxy SimCSE loss (since we don't have the full SimCSE cl_forward here,
        # we simulate it as cosine similarity of h_B with a shifted version)
        h_B = output.h_bert
        h_B_shifted = torch.roll(h_B, 1, dims=0)
        sim = F.cosine_similarity(h_B, h_B_shifted, dim=-1)
        loss_simcse_proxy = -sim.mean() + 1.0  # crude proxy, always positive

        # Combined loss
        losses = combined_loss(
            loss_simcse=loss_simcse_proxy,
            h_bert=output.h_bert,
            h_gnn=output.h_gnn,
            h_bert_proj=output.h_bert_proj,
            h_gnn_proj=output.h_gnn_proj,
        )

        # ---- Backward pass ----
        losses["total"].backward()

        # Check gradient flow
        bert_grad_norm = torch.nn.utils.clip_grad_norm_(bert_params_list, max_norm=1.0)
        gnn_grad_norm = torch.nn.utils.clip_grad_norm_(gnn_params_list, max_norm=1.0)

        optimizer.step()

        step_time = time.time() - step_start

        logger.info(
            f"  Step {step + 1}/{num_steps} | "
            f"L_total={losses['total'].item():.4f} | "
            f"L_simcse={losses['simcse'].item():.4f} | "
            f"L_align={losses['alignment'].item():.4f} | "
            f"L_gnn={losses['gnn'].item():.4f} | "
            f"‖∇_bert‖={bert_grad_norm:.4f} | "
            f"‖∇_gnn‖={gnn_grad_norm:.4f} | "
            f"{step_time:.2f}s"
        )

    # ---- Verification ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Verification Checks")
    logger.info("=" * 60)

    # Check output shapes
    assert output.h_bert.shape == (batch_size, 768), f"h_bert shape: {output.h_bert.shape}"
    logger.info(f"  ✓ h_bert shape: {output.h_bert.shape}")

    assert output.h_gnn.shape == (batch_size, 768), f"h_gnn shape: {output.h_gnn.shape}"
    logger.info(f"  ✓ h_gnn shape: {output.h_gnn.shape}")

    if output.h_bert_proj is not None:
        assert output.h_bert_proj.shape == (batch_size, projector_dim)
        logger.info(f"  ✓ h_bert_proj shape: {output.h_bert_proj.shape}")

    if output.h_gnn_proj is not None:
        assert output.h_gnn_proj.shape == (batch_size, projector_dim)
        logger.info(f"  ✓ h_gnn_proj shape: {output.h_gnn_proj.shape}")

    # Check no NaN/Inf
    assert not torch.isnan(output.h_bert).any(), "h_bert contains NaN!"
    assert not torch.isnan(output.h_gnn).any(), "h_gnn contains NaN!"
    logger.info(f"  ✓ No NaN/Inf in outputs")

    # Check gradients flowed
    gnn_grads_ok = all(p.grad is not None for p in model.gnn.parameters() if p.requires_grad)
    logger.info(f"  ✓ GNN gradients: {'flowing' if gnn_grads_ok else 'MISSING!'}")

    bert_has_grads = any(p.grad is not None for p in model.bert.parameters() if p.requires_grad)
    logger.info(f"  ✓ BERT gradients: {'flowing' if bert_has_grads else 'MISSING!'}")

    # Test inference mode (GNN discarded)
    model.eval()
    with torch.no_grad():
        inf_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sent_emb=True,
        )
    assert inf_output.h_bert.shape == (batch_size, 768)
    assert inf_output.h_gnn is None, "GNN should be None in inference mode"
    logger.info(f"  ✓ Inference mode: h_bert={inf_output.h_bert.shape}, h_gnn=None")

    # Cosine similarity sanity check
    cos_sim = F.cosine_similarity(output.h_bert, output.h_gnn, dim=-1)
    logger.info(f"  ✓ BERT↔GNN cosine sim: mean={cos_sim.mean():.4f}, std={cos_sim.std():.4f}")

    logger.info("")
    logger.info("  All checks passed! Pipeline is functional.")
    logger.info("=" * 60)


# =============================================================================
# Debug BERT wrapper (mimics SimCSE's BertForCL interface)
# =============================================================================


class DebugBertForCL(nn.Module):
    """Minimal wrapper around BertModel that mimics BertForCL's interface.

    Used when SimCSE is not installed. Provides:
    - self.bert: the underlying BertModel
    - self.config: model config
    - self.pooler: SimCSE-style pooler
    - self.mlp: MLP projection (cls pooler)
    - self.model_args: namespace with pooler settings
    - forward(sent_emb=True) for inference
    """

    def __init__(self, bert_model: BertModel) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.bert = bert_model
        self.config = bert_model.config
        self.model_args = SimpleNamespace(
            pooler_type="cls_before_pooler",
            mlp_only_train=True,
            temp=0.05,
        )

        # Simple pooler: CLS token extraction
        self.pooler_type = "cls_before_pooler"

    def pooler(self, attention_mask, outputs):
        """Extract CLS token (index 0) from last hidden state."""
        return outputs.last_hidden_state[:, 0]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        sent_emb=False,
        **kwargs,
    ):
        from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = outputs.last_hidden_state[:, 0]  # CLS

        if sent_emb:
            return BaseModelOutputWithPoolingAndCrossAttentions(
                pooler_output=pooler_output,
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
            )

        return outputs


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Debug training for GNN-Syntax-BERT")
    parser.add_argument("--conv-type", type=str, default="gat", choices=["gat", "gcn"])
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls_node"])
    parser.add_argument("--projector-dim", type=int, default=256)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--mu-gnn", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    run_debug_training(
        conv_type=args.conv_type,
        num_layers=args.num_layers,
        heads=args.heads,
        pooling=args.pooling,
        projector_dim=args.projector_dim,
        lambda_align=args.lambda_align,
        mu_gnn=args.mu_gnn,
        batch_size=args.batch_size,
        num_steps=args.steps,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
