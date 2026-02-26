# TODO — Remaining Work Items

## Critical Path (Training Loop)

### `src/train.py` — Complete the SyntaxCLTrainer

- [x] **Integrate HuggingFace Trainer properly**: `SyntaxCLTrainer` now subclasses
      `transformers.Trainer` and overrides `compute_loss()` to call `CombinedLoss`
      with graph batch data alongside text inputs. (`src/trainer.py`)
- [x] **Graph-aware DataLoader**: `SyntaxGraphDataset` + `SyntaxGraphCollator` in
      `src/data/collator.py`. Loads pre-parsed graphs, builds `PyG.Batch`, computes
      subword-to-word alignment, returns `(bs, 2, seq_len)` + graph data to the loop.
- [x] **Freeze-then-align phase transitions**: `FreezeThawCallback` in `src/trainer.py`
      switches from Phase 1 (GNN-only, BERT frozen) to Phase 2 (joint) at the
      configured epoch boundary.
- [x] **GNN augmented view for L_GNN**: Edge dropout augmentation implemented in
      `SyntaxGraphCollator` (controlled by `edge_drop_rate`); second PyG Batch
      returned as `graph_batch_aug`.
- [ ] **Logging**: Add per-component loss logging (SimCSE, alignment, GNN) to
      TensorBoard/WandB. (`_last_loss_components` is stored in the trainer but
      not yet wired to a logging backend.)

### `src/models/wrapper.py` — BertForCL Integration

- [x] **SimCSE forward pass interplay**: `SyntaxBertModel.forward()` now accepts
      `(bs, 2, seq_len)` SimCSE format. BERT runs once on the flattened input,
      view-1 hidden states feed the GNN, and the NT-Xent SimCSE loss is computed
      from `z1`/`z2` pool outputs — no double BERT pass needed.

## Evaluation

- [ ] **STS Evaluation Integration**: Adapt SimCSE's `evaluation.py` to work with
      `SyntaxBertModel` — needs to load the wrapper but only use the BERT branch
      (`encode_sentences()`).
- [ ] **Probing Tasks**: Implement syntactic probing benchmarks to measure whether
      BERT has actually internalized syntactic structure (tree depth prediction,
      top constituent classification, etc.).
- [ ] **Ablation Configs**: Create Hydra configs for key ablations:
  - GCN vs GAT
  - Number of GNN layers (1, 2, 3)
  - λ and μ sweep
  - With/without projection head
  - Pooling strategy comparison

## Data Pipeline

- [ ] **Multiprocessing for parsing**: The `batch_parse()` method is single-process.
      For 1M sentences, parallelize across multiple Stanza instances (be careful
      with memory — each Stanza pipeline uses ~2GB).
- [ ] **Pre-tokenized caching**: Cache BERT tokenization alongside dependency parses
      to avoid redundant tokenization during training.
- [ ] **Alignment validation**: Add a utility to spot-check subword alignment quality
      on sampled sentences and log mismatches.

## Infrastructure

- [ ] **Multi-GPU / DDP**: Ensure PyG batch tensors are properly handled under
      `DistributedDataParallel`. PyG's `DataLoader` has its own distributed sampler.
- [ ] **Checkpointing**: Save/load both BERT and GNN parameters. For inference
      checkpoints, save only BERT weights.
- [ ] **Experiment tracking**: Wire `_last_loss_components` in `SyntaxCLTrainer` to
      WandB or MLflow for tracking experiments across strategies.

## Testing

- [x] **Integration test**: End-to-end test with toy data in `tests/test_integration.py`
      — covers data pipeline, (bs, 2, seq_len) forward, backward gradients, inference
      mode, and stop-grad behaviour.
- [x] **Wrapper forward pass test**: `TestSyntaxBertModelForward` tests `SyntaxBertModel`
      with `_DebugBertForCL` and real graph data.
- [x] **Loss backward test**: `TestFullPipeline.test_backward_gradients_flow` verifies
      gradient flow through the full pipeline (BERT → GNN → CombinedLoss → backward).
