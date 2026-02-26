"""
test_integration.py — End-to-end integration tests for GNN-Syntax-BERT.

Tests the full pipeline without requiring SimCSE, Stanza, or downloaded data:
  1. SyntaxGraphDataset + SyntaxGraphCollator (data pipeline)
  2. SyntaxBertModel.forward() with (bs, 2, seq_len) SimCSE format
  3. CombinedLoss backward through BERT + GNN
  4. SyntaxGraphCollator edge-dropout augmentation
  5. Inference mode (GNN discarded)

Uses the same DebugBertForCL helper from scripts/debug_train.py so that
SimCSE is not required in the test environment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from transformers import AutoConfig, AutoTokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

# Make sure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Minimal SimCSE-compatible BERT wrapper (no SimCSE required)
# =============================================================================


class _DebugBertForCL(nn.Module):
    """Minimal BertForCL shim for testing (mirrors scripts/debug_train.py)."""

    def __init__(self, bert_model: BertModel) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.bert = bert_model
        self.config = bert_model.config
        self.pooler_type = "cls_before_pooler"
        self.model_args = SimpleNamespace(
            pooler_type="cls_before_pooler",
            mlp_only_train=True,
            temp=0.05,
        )

    def pooler(self, attention_mask, outputs):
        return outputs.last_hidden_state[:, 0]  # CLS token

    def forward(self, input_ids=None, attention_mask=None, sent_emb=False, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        pooler_output = outputs.last_hidden_state[:, 0]
        if sent_emb:
            return BaseModelOutputWithPoolingAndCrossAttentions(
                pooler_output=pooler_output,
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
            )
        return outputs


# =============================================================================
# Shared fixtures
# =============================================================================

MODEL_NAME = "bert-base-uncased"
HIDDEN_DIM = 768
BATCH_SIZE = 4
SEQ_LEN = 32


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def syntax_bert_model():
    """Instantiate SyntaxBertModel with DebugBertForCL (no SimCSE needed)."""
    from src.models.wrapper import SyntaxBertModel

    bert = BertModel.from_pretrained(MODEL_NAME)
    bert_wrapper = _DebugBertForCL(bert)

    model = SyntaxBertModel(
        bert_model=bert_wrapper,
        gnn_config={
            "conv_type": "gat",
            "num_layers": 2,
            "hidden_dim": HIDDEN_DIM,
            "heads": 4,
            "dropout": 0.1,
            "pooling": "mean",
            "freeze": False,
        },
        alignment_config={
            "lambda_align": 0.1,
            "mu_gnn": 0.05,
            "align_temperature": 0.05,
            "projector_dim": 128,
            "stop_grad_gnn": False,
        },
    )
    model.eval()
    return model


def _make_toy_parse(num_tokens: int = 6) -> dict:
    """Build a minimal parse result dict (no Stanza needed)."""
    tokens = [f"word{i}" for i in range(num_tokens)]
    # Chain: 0→1→2→…→(n-1)
    edges_src = list(range(num_tokens - 1))
    edges_dst = list(range(1, num_tokens))
    # Add reverse edges
    edges_src += list(range(1, num_tokens))
    edges_dst += list(range(num_tokens - 1))
    return {
        "tokens": tokens,
        "edges_src": edges_src,
        "edges_dst": edges_dst,
        "num_tokens": num_tokens,
        "sentence": " ".join(tokens),
    }


# =============================================================================
# Data pipeline tests
# =============================================================================


class TestSyntaxGraphDataset:
    def test_len_and_getitem(self):
        from src.data.collator import SyntaxGraphDataset

        sentences = ["hello world", "foo bar baz"]
        parses = [_make_toy_parse(2), _make_toy_parse(3)]
        ds = SyntaxGraphDataset(sentences, parses)

        assert len(ds) == 2
        item = ds[0]
        assert item["sentence"] == "hello world"
        assert "parse_result" in item
        assert item["idx"] == 0

    def test_max_samples(self):
        from src.data.collator import SyntaxGraphDataset

        sentences = [f"sentence {i}" for i in range(10)]
        parses = [_make_toy_parse() for _ in range(10)]
        ds = SyntaxGraphDataset(sentences, parses, max_samples=5)

        assert len(ds) == 5

    def test_mismatched_lengths_uses_min(self):
        from src.data.collator import SyntaxGraphDataset

        sentences = ["a", "b", "c"]
        parses = [_make_toy_parse(), _make_toy_parse()]  # only 2
        ds = SyntaxGraphDataset(sentences, parses)

        assert len(ds) == 2


class TestSyntaxGraphCollator:
    def test_output_shapes(self, tokenizer):
        from src.data.collator import SyntaxGraphCollator, SyntaxGraphDataset

        parses = [_make_toy_parse(5) for _ in range(BATCH_SIZE)]
        sentences = [p["sentence"] for p in parses]
        ds = SyntaxGraphDataset(sentences, parses)
        collator = SyntaxGraphCollator(tokenizer, max_seq_length=SEQ_LEN, edge_drop_rate=0.0)

        batch = collator([ds[i] for i in range(BATCH_SIZE)])

        assert batch["input_ids"].shape == (BATCH_SIZE, 2, SEQ_LEN)
        assert batch["attention_mask"].shape == (BATCH_SIZE, 2, SEQ_LEN)
        assert isinstance(batch["graph_batch"], Batch)
        assert batch["graph_batch_aug"] is None  # edge_drop_rate=0
        assert len(batch["token_to_word_maps"]) == BATCH_SIZE

    def test_graph_batch_has_correct_node_count(self, tokenizer):
        from src.data.collator import SyntaxGraphCollator, SyntaxGraphDataset

        num_tokens = 6
        parses = [_make_toy_parse(num_tokens) for _ in range(BATCH_SIZE)]
        sentences = [p["sentence"] for p in parses]
        ds = SyntaxGraphDataset(sentences, parses)
        collator = SyntaxGraphCollator(tokenizer, max_seq_length=SEQ_LEN, edge_drop_rate=0.0)

        batch = collator([ds[i] for i in range(BATCH_SIZE)])

        # Total nodes across batch = num_tokens * batch_size
        assert batch["graph_batch"].num_nodes == num_tokens * BATCH_SIZE

    def test_edge_dropout_augmentation(self, tokenizer):
        from src.data.collator import SyntaxGraphCollator, SyntaxGraphDataset

        parses = [_make_toy_parse(8) for _ in range(BATCH_SIZE)]
        sentences = [p["sentence"] for p in parses]
        ds = SyntaxGraphDataset(sentences, parses)
        collator = SyntaxGraphCollator(tokenizer, max_seq_length=SEQ_LEN, edge_drop_rate=0.3)

        batch = collator([ds[i] for i in range(BATCH_SIZE)])

        assert batch["graph_batch_aug"] is not None
        # Augmented graph should have ≤ edges of original
        orig_edges = batch["graph_batch"].edge_index.shape[1]
        aug_edges = batch["graph_batch_aug"].edge_index.shape[1]
        assert aug_edges <= orig_edges

    def test_alignment_maps_length(self, tokenizer):
        from src.data.collator import SyntaxGraphCollator, SyntaxGraphDataset

        parses = [_make_toy_parse(4) for _ in range(BATCH_SIZE)]
        sentences = [p["sentence"] for p in parses]
        ds = SyntaxGraphDataset(sentences, parses)
        collator = SyntaxGraphCollator(tokenizer, max_seq_length=SEQ_LEN, edge_drop_rate=0.0)

        batch = collator([ds[i] for i in range(BATCH_SIZE)])

        for alignment in batch["token_to_word_maps"]:
            # Alignment length should match padded sequence length
            assert len(alignment) == SEQ_LEN


# =============================================================================
# SyntaxBertModel forward pass tests
# =============================================================================


class TestSyntaxBertModelForward:
    def _make_batch(self, tokenizer, bs: int = BATCH_SIZE):
        """Build a (bs, 2, seq_len) input batch with toy graphs."""
        parses = [_make_toy_parse(5) for _ in range(bs)]
        sentences = [p["sentence"] for p in parses]

        encoded = tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=SEQ_LEN,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        # SimCSE format
        input_ids_2x = torch.stack([input_ids, input_ids], dim=1)
        attention_mask_2x = torch.stack([attention_mask, attention_mask], dim=1)

        # Alignment maps
        token_to_word_maps = []
        for i, parse in enumerate(parses):
            bert_tokens = tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
            from src.processing.syntax_parser import StanzaSyntaxParser
            alignment = StanzaSyntaxParser.align_subwords(parse["tokens"], bert_tokens)
            token_to_word_maps.append(alignment)

        # PyG batch
        from src.data.collator import SyntaxGraphCollator
        graphs = [SyntaxGraphCollator._parse_to_data(p) for p in parses]
        graph_batch = Batch.from_data_list(graphs)

        return input_ids_2x, attention_mask_2x, graph_batch, token_to_word_maps

    def test_training_output_shapes(self, syntax_bert_model, tokenizer):
        input_ids, attn_mask, graph_batch, tok_maps = self._make_batch(tokenizer)
        syntax_bert_model.train()

        with torch.no_grad():
            out = syntax_bert_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                graph_batch=graph_batch,
                token_to_word_maps=tok_maps,
            )

        assert out.h_bert.shape == (BATCH_SIZE, HIDDEN_DIM), f"h_bert: {out.h_bert.shape}"
        assert out.h_gnn.shape == (BATCH_SIZE, HIDDEN_DIM), f"h_gnn: {out.h_gnn.shape}"
        assert out.h_bert_proj.shape == (BATCH_SIZE, 128), f"h_bert_proj: {out.h_bert_proj.shape}"
        assert out.h_gnn_proj.shape == (BATCH_SIZE, 128), f"h_gnn_proj: {out.h_gnn_proj.shape}"
        assert out.bert_hidden_states.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)

    def test_simcse_loss_is_computed(self, syntax_bert_model, tokenizer):
        input_ids, attn_mask, graph_batch, tok_maps = self._make_batch(tokenizer)
        syntax_bert_model.train()

        with torch.no_grad():
            out = syntax_bert_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                graph_batch=graph_batch,
                token_to_word_maps=tok_maps,
            )

        assert out.simcse_loss is not None, "simcse_loss should be set in training mode"
        assert out.simcse_loss.ndim == 0, "simcse_loss should be a scalar"
        assert not torch.isnan(out.simcse_loss), "simcse_loss is NaN"
        assert out.simcse_loss.item() > 0, "simcse_loss should be positive"

    def test_inference_mode_no_gnn(self, syntax_bert_model, tokenizer):
        """Inference mode: GNN is discarded; h_gnn and simcse_loss are None."""
        encoded = tokenizer(
            ["hello world"] * BATCH_SIZE,
            padding="max_length",
            truncation=True,
            max_length=SEQ_LEN,
            return_tensors="pt",
        )
        syntax_bert_model.eval()

        with torch.no_grad():
            out = syntax_bert_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                sent_emb=True,
            )

        assert out.h_bert.shape == (BATCH_SIZE, HIDDEN_DIM)
        assert out.h_gnn is None, "h_gnn should be None in inference mode"
        assert out.simcse_loss is None, "simcse_loss should be None in inference mode"

    def test_no_nan_in_outputs(self, syntax_bert_model, tokenizer):
        input_ids, attn_mask, graph_batch, tok_maps = self._make_batch(tokenizer)
        syntax_bert_model.train()

        with torch.no_grad():
            out = syntax_bert_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                graph_batch=graph_batch,
                token_to_word_maps=tok_maps,
            )

        assert not torch.isnan(out.h_bert).any(), "h_bert contains NaN"
        assert not torch.isnan(out.h_gnn).any(), "h_gnn contains NaN"
        assert not torch.isinf(out.h_bert).any(), "h_bert contains Inf"
        assert not torch.isinf(out.h_gnn).any(), "h_gnn contains Inf"


# =============================================================================
# Full forward + backward test
# =============================================================================


class TestFullPipeline:
    def test_backward_gradients_flow(self, tokenizer):
        """Full forward + backward with combined loss; verify BERT and GNN get grads."""
        from src.alignment.losses import CombinedLoss
        from src.data.collator import SyntaxGraphCollator, SyntaxGraphDataset
        from src.models.wrapper import SyntaxBertModel

        bs = 4
        parses = [_make_toy_parse(5) for _ in range(bs)]
        sentences = [p["sentence"] for p in parses]
        ds = SyntaxGraphDataset(sentences, parses)
        collator = SyntaxGraphCollator(tokenizer, max_seq_length=SEQ_LEN, edge_drop_rate=0.0)
        batch = collator([ds[i] for i in range(bs)])

        bert = BertModel.from_pretrained(MODEL_NAME)
        bert_wrapper = _DebugBertForCL(bert)
        model = SyntaxBertModel(
            bert_model=bert_wrapper,
            gnn_config={"conv_type": "gat", "num_layers": 2, "hidden_dim": HIDDEN_DIM, "heads": 4,
                        "dropout": 0.0, "pooling": "mean", "freeze": False},
            alignment_config={"lambda_align": 0.1, "mu_gnn": 0.0, "align_temperature": 0.05,
                              "projector_dim": 0, "stop_grad_gnn": False},
        )
        model.train()

        combined_loss = CombinedLoss(lambda_align=0.1, mu_gnn=0.0)

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            graph_batch=batch["graph_batch"],
            token_to_word_maps=batch["token_to_word_maps"],
        )

        loss_simcse = out.simcse_loss if out.simcse_loss is not None else torch.tensor(1.0)
        losses = combined_loss(loss_simcse=loss_simcse, h_bert=out.h_bert, h_gnn=out.h_gnn)

        losses["total"].backward()

        # BERT should receive gradients
        bert_has_grads = any(
            p.grad is not None for p in model.bert.parameters() if p.requires_grad
        )
        assert bert_has_grads, "No gradients flowed to BERT parameters"

        # GNN should receive gradients
        gnn_has_grads = any(
            p.grad is not None for p in model.gnn.parameters() if p.requires_grad
        )
        assert gnn_has_grads, "No gradients flowed to GNN parameters"

    def test_stop_grad_freezes_gnn_gradient(self, tokenizer):
        """With stop_grad_gnn=True, the alignment loss should NOT update the GNN."""
        from src.alignment.losses import CombinedLoss
        from src.data.collator import SyntaxGraphCollator, SyntaxGraphDataset
        from src.models.wrapper import SyntaxBertModel

        bs = 4
        parses = [_make_toy_parse(5) for _ in range(bs)]
        sentences = [p["sentence"] for p in parses]
        ds = SyntaxGraphDataset(sentences, parses)
        collator = SyntaxGraphCollator(tokenizer, max_seq_length=SEQ_LEN, edge_drop_rate=0.0)
        batch = collator([ds[i] for i in range(bs)])

        bert = BertModel.from_pretrained(MODEL_NAME)
        bert_wrapper = _DebugBertForCL(bert)
        model = SyntaxBertModel(
            bert_model=bert_wrapper,
            gnn_config={"conv_type": "gat", "num_layers": 2, "hidden_dim": HIDDEN_DIM, "heads": 4,
                        "dropout": 0.0, "pooling": "mean", "freeze": False},
            alignment_config={"lambda_align": 0.1, "mu_gnn": 0.0, "align_temperature": 0.05,
                              "projector_dim": 0, "stop_grad_gnn": True},  # <-- stop grad
        )
        model.train()
        combined_loss = CombinedLoss(lambda_align=0.1, mu_gnn=0.0)

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            graph_batch=batch["graph_batch"],
            token_to_word_maps=batch["token_to_word_maps"],
        )

        # Only alignment loss (BERT → GNN direction is stopped)
        losses = combined_loss(
            loss_simcse=torch.tensor(0.0),
            h_bert=out.h_bert,
            h_gnn=out.h_gnn,
        )
        losses["total"].backward()

        # GNN parameters should have NO gradients (h_gnn was detached)
        gnn_grads = [p.grad for p in model.gnn.parameters() if p.requires_grad]
        assert all(g is None for g in gnn_grads), "GNN received gradients despite stop_grad_gnn=True"
