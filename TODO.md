# TODO — Remaining Work Items

## Critical Path (Training Loop)

### `src/train.py` — Complete the SyntaxCLTrainer
- [ ] **Integrate HuggingFace Trainer properly**: Subclass `transformers.Trainer` and override `compute_loss()` to call `CombinedLoss` with graph batch data alongside text inputs.
- [ ] **Graph-aware DataLoader**: Create a custom `DataCollator` that, given a batch of sentence indices, loads the corresponding pre-parsed dependency graphs and constructs a `torch_geometric.data.Batch` alongside the tokenized text tensors. This requires:
  - Mapping sentence index → parsed graph (from `.jsonl` cache files)
  - Building the subword-to-word alignment per sentence on the fly
  - Returning both HuggingFace `BatchEncoding` and PyG `Batch`
- [ ] **Freeze-then-align phase transitions**: Implement the epoch callback that switches from Phase 1 (GNN-only) to Phase 2 (joint) at the configured epoch boundary. This means dynamically freezing/unfreezing parameters and swapping loss functions mid-training.
- [ ] **GNN augmented view for L_GNN**: Implement edge dropout augmentation in the data collator to create two views of each graph for the GNN contrastive loss.
- [ ] **Logging**: Add loss component logging (SimCSE, alignment, GNN) to TensorBoard/WandB.

### `src/models/wrapper.py` — BertForCL Integration
- [ ] **SimCSE forward pass interplay**: Currently the wrapper calls `self.bert.bert(...)` directly to get hidden states, bypassing SimCSE's `cl_forward()`. Need to carefully integrate so that SimCSE's contrastive loss is computed as usual, AND we simultaneously extract hidden states for the GNN. Options:
  - Run BERT twice (inefficient but clean)
  - Hook into SimCSE's `cl_forward()` to extract hidden states as a side-effect
  - Refactor `cl_forward()` to return hidden states alongside the loss

## Evaluation

- [ ] **STS Evaluation Integration**: Adapt SimCSE's `evaluation.py` to work with `SyntaxBertModel` — needs to load the wrapper but only use the BERT branch (`encode_sentences()`).
- [ ] **Probing Tasks**: Implement syntactic probing benchmarks to measure whether BERT has actually internalized syntactic structure (tree depth prediction, top constituent classification, etc.).
- [ ] **Ablation Configs**: Create Hydra configs for key ablations:
  - GCN vs GAT
  - Number of GNN layers (1, 2, 3)
  - λ and μ sweep
  - With/without projection head
  - Pooling strategy comparison

## Data Pipeline

- [ ] **Multiprocessing for parsing**: The `batch_parse()` method is single-process. For 1M sentences, parallelize across multiple Stanza instances (be careful with memory — each Stanza pipeline uses ~2GB).
- [ ] **Pre-tokenized caching**: Cache BERT tokenization alongside dependency parses to avoid redundant tokenization during training.
- [ ] **Alignment validation**: Add a utility to spot-check subword alignment quality on sampled sentences and log mismatches.

## Infrastructure

- [ ] **Multi-GPU / DDP**: Ensure PyG batch tensors are properly handled under `DistributedDataParallel`. PyG's `DataLoader` has its own distributed sampler.
- [ ] **Checkpointing**: Save/load both BERT and GNN parameters. For inference checkpoints, save only BERT weights.
- [ ] **Experiment tracking**: WandB or MLflow integration for tracking experiments across strategies.

## Testing

- [ ] **Integration test**: End-to-end test with a tiny dataset (10 sentences), parsing, graph construction, and one training step.
- [ ] **Wrapper forward pass test**: Test `SyntaxBertModel.forward()` with mock BertForCL and real graph data.
- [ ] **Loss backward test**: Verify gradient flow through the full pipeline (BERT → GNN → CombinedLoss → backward).
