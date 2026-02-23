"""
test_gnn_encoder.py — Smoke tests for the SyntaxGNNEncoder and losses.

Verifies that:
1. GAT and GCN encoders produce correct output shapes
2. Different pooling strategies work
3. Batching works correctly
4. AlignmentLoss computes without errors
5. CombinedLoss aggregates correctly
6. Subword alignment produces valid mappings
"""

import pytest
import torch
from torch_geometric.data import Batch, Data


# =============================================================================
# GNN Encoder Tests
# =============================================================================


def make_random_graph(num_nodes: int = 8, num_edges: int = 14, hidden_dim: int = 768) -> Data:
    """Create a random graph with node features for testing."""
    x = torch.randn(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def make_dependency_tree(hidden_dim: int = 768) -> Data:
    """Create a realistic dependency tree for 'The cat sat on the mat'.

    Tree structure (ROOT=sat):
        sat → cat (nsubj)
        cat → The (det)
        sat → on (obl)
        on → mat (nmod)
        mat → the (det)

    Bidirectional edges are included.
    """
    # Tokens: [The, cat, sat, on, the, mat]
    num_nodes = 6
    x = torch.randn(num_nodes, hidden_dim)

    # Directed edges (parent → child) + reverse
    edges_src = [2, 1, 2, 3, 5,   1, 0, 3, 5, 4]  # reverse included
    edges_dst = [1, 0, 3, 5, 4,   2, 1, 2, 3, 5]

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


class TestSyntaxGNNEncoder:
    """Tests for SyntaxGNNEncoder forward pass."""

    def test_gat_forward_shape(self):
        from src.models.gnn_encoder import SyntaxGNNEncoder

        encoder = SyntaxGNNEncoder(
            in_dim=768, hidden_dim=768, num_layers=2,
            conv_type="gat", heads=4, pooling="mean",
        )
        graph = make_random_graph()
        batch = Batch.from_data_list([graph])

        h_G = encoder(batch)

        assert h_G.shape == (1, 768), f"Expected (1, 768), got {h_G.shape}"

    def test_gcn_forward_shape(self):
        from src.models.gnn_encoder import SyntaxGNNEncoder

        encoder = SyntaxGNNEncoder(
            in_dim=768, hidden_dim=768, num_layers=2,
            conv_type="gcn", pooling="mean",
        )
        graph = make_random_graph()
        batch = Batch.from_data_list([graph])

        h_G = encoder(batch)

        assert h_G.shape == (1, 768), f"Expected (1, 768), got {h_G.shape}"

    def test_batch_of_graphs(self):
        from src.models.gnn_encoder import SyntaxGNNEncoder

        encoder = SyntaxGNNEncoder(
            in_dim=768, hidden_dim=768, num_layers=2,
            conv_type="gat", heads=4, pooling="mean",
        )

        graphs = [make_random_graph(num_nodes=n) for n in [5, 8, 3, 10]]
        batch = Batch.from_data_list(graphs)

        h_G = encoder(batch)

        assert h_G.shape == (4, 768), f"Expected (4, 768), got {h_G.shape}"

    def test_dependency_tree(self):
        from src.models.gnn_encoder import SyntaxGNNEncoder

        encoder = SyntaxGNNEncoder(
            in_dim=768, hidden_dim=768, num_layers=2,
            conv_type="gat", heads=4, pooling="mean",
        )
        tree = make_dependency_tree()
        batch = Batch.from_data_list([tree])

        h_G = encoder(batch)

        assert h_G.shape == (1, 768)
        assert not torch.isnan(h_G).any(), "Output contains NaN"
        assert not torch.isinf(h_G).any(), "Output contains Inf"

    @pytest.mark.parametrize("pooling", ["mean", "max", "cls_node"])
    def test_pooling_strategies(self, pooling: str):
        from src.models.gnn_encoder import SyntaxGNNEncoder

        encoder = SyntaxGNNEncoder(
            in_dim=768, hidden_dim=768, num_layers=2,
            conv_type="gat", heads=4, pooling=pooling,
        )
        graph = make_random_graph()
        batch = Batch.from_data_list([graph])

        h_G = encoder(batch)

        assert h_G.shape == (1, 768)

    def test_gradient_flow(self):
        from src.models.gnn_encoder import SyntaxGNNEncoder

        encoder = SyntaxGNNEncoder(
            in_dim=768, hidden_dim=768, num_layers=2,
            conv_type="gat", heads=4, pooling="mean",
        )
        graph = make_random_graph()
        batch = Batch.from_data_list([graph])

        h_G = encoder(batch)
        loss = h_G.sum()
        loss.backward()

        # Check that gradients flowed to all conv layers
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_invalid_conv_type_raises(self):
        from src.models.gnn_encoder import SyntaxGNNEncoder

        with pytest.raises(ValueError, match="conv_type"):
            SyntaxGNNEncoder(conv_type="invalid")

    def test_invalid_pooling_raises(self):
        from src.models.gnn_encoder import SyntaxGNNEncoder

        with pytest.raises(ValueError, match="pooling"):
            SyntaxGNNEncoder(pooling="invalid")


# =============================================================================
# Loss Tests
# =============================================================================


class TestAlignmentLoss:
    """Tests for the AlignmentLoss (NT-Xent)."""

    def test_identical_embeddings_low_loss(self):
        from src.alignment.losses import AlignmentLoss

        loss_fn = AlignmentLoss(temperature=0.05)
        h = torch.randn(16, 256)

        loss = loss_fn(h, h)

        # Loss should be near 0 when embeddings are identical
        assert loss.item() < 1.0, f"Expected low loss for identical embeddings, got {loss.item()}"

    def test_random_embeddings_higher_loss(self):
        from src.alignment.losses import AlignmentLoss

        loss_fn = AlignmentLoss(temperature=0.05)
        h_bert = torch.randn(16, 256)
        h_gnn = torch.randn(16, 256)

        loss = loss_fn(h_bert, h_gnn)

        # Loss should be positive
        assert loss.item() > 0, "Loss should be positive for random embeddings"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_gradient_flows(self):
        from src.alignment.losses import AlignmentLoss

        loss_fn = AlignmentLoss(temperature=0.05)
        h_bert = torch.randn(8, 256, requires_grad=True)
        h_gnn = torch.randn(8, 256, requires_grad=True)

        loss = loss_fn(h_bert, h_gnn)
        loss.backward()

        assert h_bert.grad is not None
        assert h_gnn.grad is not None


class TestCombinedLoss:
    """Tests for the CombinedLoss aggregation."""

    def test_combined_loss_weights(self):
        from src.alignment.losses import CombinedLoss

        combined = CombinedLoss(lambda_align=0.5, mu_gnn=0.3)

        loss_simcse = torch.tensor(1.0)
        h_bert = torch.randn(8, 256)
        h_gnn = torch.randn(8, 256)

        result = combined(loss_simcse, h_bert, h_gnn)

        assert "total" in result
        assert "simcse" in result
        assert "alignment" in result
        assert "gnn" in result
        assert result["total"].item() > 0

    def test_zero_weights(self):
        from src.alignment.losses import CombinedLoss

        combined = CombinedLoss(lambda_align=0.0, mu_gnn=0.0)

        loss_simcse = torch.tensor(2.5)
        h_bert = torch.randn(8, 256)
        h_gnn = torch.randn(8, 256)

        result = combined(loss_simcse, h_bert, h_gnn)

        # Total should equal SimCSE loss when weights are 0
        assert abs(result["total"].item() - 2.5) < 0.1


# =============================================================================
# Subword Alignment Tests
# =============================================================================


class TestSubwordAlignment:
    """Tests for BERT subword → Stanza word alignment."""

    def test_simple_alignment(self):
        from src.processing.syntax_parser import StanzaSyntaxParser

        stanza_tokens = ["The", "cat", "sat"]
        bert_tokens = ["[CLS]", "the", "cat", "sat", "[SEP]"]

        alignment = StanzaSyntaxParser.align_subwords(stanza_tokens, bert_tokens)

        assert alignment == [-1, 0, 1, 2, -1]

    def test_subword_alignment(self):
        from src.processing.syntax_parser import StanzaSyntaxParser

        stanza_tokens = ["unbelievable", "cat"]
        bert_tokens = ["[CLS]", "un", "##bel", "##ie", "##va", "##ble", "cat", "[SEP]"]

        alignment = StanzaSyntaxParser.align_subwords(stanza_tokens, bert_tokens)

        # All "un##bel##ie##va##ble" subwords should map to stanza word 0
        assert alignment[0] == -1   # [CLS]
        assert alignment[1] == 0    # un
        assert alignment[2] == 0    # ##bel
        assert alignment[3] == 0    # ##ie
        assert alignment[4] == 0    # ##va
        assert alignment[5] == 0    # ##ble
        assert alignment[6] == 1    # cat
        assert alignment[7] == -1   # [SEP]

    def test_aggregate_subword_embeddings(self):
        from src.processing.syntax_parser import StanzaSyntaxParser

        bert_hidden = torch.randn(5, 768)  # [CLS], w1_sub1, w1_sub2, w2, [SEP]
        alignment = [-1, 0, 0, 1, -1]
        num_words = 2

        word_embs = StanzaSyntaxParser.aggregate_subword_embeddings(
            bert_hidden, alignment, num_words
        )

        assert word_embs.shape == (2, 768)
        # First word should be mean of positions 1 and 2
        expected_w0 = (bert_hidden[1] + bert_hidden[2]) / 2
        assert torch.allclose(word_embs[0], expected_w0, atol=1e-5)
