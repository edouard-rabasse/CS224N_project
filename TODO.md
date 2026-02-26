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
- [x] **Logging**: Per-component loss logging wired via `SyntaxCLTrainer.log()`
      override. `loss_simcse`, `loss_alignment`, `loss_gnn` are running-averaged
      over `logging_steps` and injected into every TensorBoard/WandB log event.

### `src/models/wrapper.py` — BertForCL Integration

- [x] **SimCSE forward pass interplay**: `SyntaxBertModel.forward()` now accepts
      `(bs, 2, seq_len)` SimCSE format. BERT runs once on the flattened input,
      view-1 hidden states feed the GNN, and the NT-Xent SimCSE loss is computed
      from `z1`/`z2` pool outputs — no double BERT pass needed.

## Evaluation

- [x] **STS Evaluation Integration**: `src/evaluate.py` — SentEval wrapper around
      `SyntaxBertModel.encode_sentences()`. Supports dev/test/fasttest modes,
      STS + transfer task sets, JSON score export, and a `--baseline` flag for
      plain HuggingFace models.
- [ ] **Probing Tasks**: Implement syntactic probing benchmarks to measure whether
      BERT has actually internalized syntactic structure (tree depth prediction,
      top constituent classification, etc.).
- [x] **Ablation Configs**: Created under `configs/experiment/ablation/`:
  - `gcn.yaml` — GCN vs GAT
  - `gnn_1layer.yaml`, `gnn_3layer.yaml` — depth sweep
  - `lambda_low.yaml`, `lambda_high.yaml` — λ sweep (0.01, 0.5)
  - `mu_zero.yaml`, `mu_high.yaml` — μ sweep (0.0, 0.2)
  - `no_projection.yaml` — disable projection heads
  - `max_pooling.yaml`, `cls_pooling.yaml` — pooling strategy comparison

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
- [x] **Checkpointing**: `SyntaxBertModel.save_checkpoint()` saves `bert_weights/`,
      `gnn_state.pt`, and `model_config.json`. `save_bert_only()` writes a lean
      inference checkpoint. `from_checkpoint()` restores both branches.
      `SyntaxCLTrainer._save_checkpoint()` and `save_model()` call these
      automatically at every HF checkpoint interval and at training end.
- [x] **Experiment tracking**: `_last_loss_components` averaged via running sums
      and injected into TensorBoard/WandB through `SyntaxCLTrainer.log()` override.

## Testing

- [x] **Integration test**: End-to-end test with toy data in `tests/test_integration.py`
      — covers data pipeline, (bs, 2, seq_len) forward, backward gradients, inference
      mode, and stop-grad behaviour.
- [x] **Wrapper forward pass test**: `TestSyntaxBertModelForward` tests `SyntaxBertModel`
      with `_DebugBertForCL` and real graph data.
- [x] **Loss backward test**: `TestFullPipeline.test_backward_gradients_flow` verifies
      gradient flow through the full pipeline (BERT → GNN → CombinedLoss → backward).
