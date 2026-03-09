"""
trainer.py — SyntaxCLTrainer: HuggingFace Trainer subclass for GNN-Syntax-BERT.

Extends ``transformers.Trainer`` with:
  - Graph-aware data loading via SyntaxGraphDataset + SyntaxGraphCollator
  - Combined loss: L_SimCSE + λ·L_align + μ·L_GNN  (overrides compute_loss)
  - Separate AdamW LR groups for BERT and GNN      (overrides create_optimizer)
  - Freeze-then-align phase transitions via TrainerCallback

Training strategies (set by experiment config):
  multi_loss       — joint training from epoch 0 with all three loss terms
  stop_grad        — GNN frozen; alignment gradients only update BERT
  freeze_then_align — Phase 1: GNN warmup (BERT frozen); Phase 2: joint
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.alignment.losses import CombinedLoss
from src.data.collator import SyntaxGraphCollator, SyntaxGraphDataset

logger = logging.getLogger(__name__)


# =============================================================================
# Phase-transition callback (freeze-then-align)
# =============================================================================


class LambdaWarmupCallback(TrainerCallback):
    """Linearly ramp ``lambda_align`` from 0 to its target value over
    ``warmup_steps`` training steps.

    This prevents the alignment loss from applying gradient pressure on BERT
    toward a randomly-initialized GNN at the start of training — the GNN gets
    ``warmup_steps`` steps to learn meaningful representations first.

    Args:
        loss_fn: The ``CombinedLoss`` instance whose ``lambda_align`` to schedule.
        target_lambda: The final ``lambda_align`` value to reach at step ``warmup_steps``.
        warmup_steps: Number of steps over which to linearly ramp λ from 0 to target.
    """

    def __init__(self, loss_fn: CombinedLoss, target_lambda: float, warmup_steps: int) -> None:
        self.loss_fn = loss_fn
        self.target_lambda = target_lambda
        self.warmup_steps = warmup_steps
        # Start at 0
        self.loss_fn.lambda_align = 0.0

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.global_step < self.warmup_steps:
            self.loss_fn.lambda_align = self.target_lambda * state.global_step / self.warmup_steps
        else:
            self.loss_fn.lambda_align = self.target_lambda


class FreezeThawCallback(TrainerCallback):
    """TrainerCallback that transitions from Phase 1 → Phase 2 at a set epoch.

    Phase 1 (epochs 0 … phase1_epochs-1): GNN trains alone (BERT frozen).
    Phase 2 (epochs phase1_epochs … end):  Joint training (BERT unfrozen).

    Args:
        model: The ``SyntaxBertModel`` instance being trained.
        phase1_epochs: Number of epochs to keep BERT frozen.
    """

    def __init__(self, model: torch.nn.Module, phase1_epochs: int) -> None:
        self.model = model
        self.phase1_epochs = phase1_epochs
        self._phase2_started = False

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if state.epoch >= self.phase1_epochs and not self._phase2_started:
            logger.info(
                f"[FreezeThawCallback] Epoch {state.epoch:.0f}: "
                f"Phase 2 — unfreezing BERT for joint training."
            )
            self.model._unfreeze_bert()  # type: ignore[attr-defined]
            self._phase2_started = True


# =============================================================================
# Main Trainer
# =============================================================================


class SyntaxCLTrainer(Trainer):
    """HuggingFace Trainer subclass for GNN-Syntax-BERT.

    Overrides:
      - ``compute_loss``:      combined objective (SimCSE + alignment + GNN)
      - ``create_optimizer``:  AdamW with separate LR groups (BERT vs GNN)

    The dataset, collator, tokenizer and callbacks are built in ``__init__``
    from the Hydra config and pre-parsed graph list.

    Args:
        model: A ``SyntaxBertModel`` instance.
        cfg: Hydra ``DictConfig`` (root config).
        model_args: SimpleNamespace with ``model_name_or_path``, ``pooler_type``, etc.
        data_args: SimpleNamespace with ``train_file``, ``max_seq_length``, etc.
        training_args: HuggingFace ``TrainingArguments``.
        parsed_graphs: Pre-parsed dependency graphs from ``load_parsed_graphs()``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DictConfig,
        model_args,
        data_args,
        training_args: TrainingArguments,
        parsed_graphs: list[dict],
    ) -> None:
        self.syntax_cfg = cfg
        self.syntax_model_args = model_args
        self.syntax_data_args = data_args

        # ---- Loss hyperparameters ----
        align_cfg: dict = (
            OmegaConf.to_container(cfg.model.alignment, resolve=True)
            if "model" in cfg and "alignment" in cfg.model
            else {}
        )
        self._combined_loss = CombinedLoss(
            lambda_align=align_cfg.get("lambda_align", 0.1),
            mu_gnn=align_cfg.get("mu_gnn", 0.05),
            align_temperature=align_cfg.get("align_temperature", 0.05),
        )

        # ---- Dataset & collator ----
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        tokenizer = self.tokenizer
        sentences = self._load_sentences(data_args)

        dataset = SyntaxGraphDataset(
            sentences=sentences,
            parsed_graphs=parsed_graphs,
            max_samples=data_args.max_train_samples,
        )

        training_cfg: dict = (
            OmegaConf.to_container(cfg.training, resolve=True) if "training" in cfg else {}
        )
        # Enable edge-dropout augmentation only when L_GNN is active
        edge_drop = 0.1 if align_cfg.get("mu_gnn", 0.0) > 0 else 0.0
        collator = SyntaxGraphCollator(
            tokenizer=tokenizer,
            max_seq_length=data_args.max_seq_length,
            edge_drop_rate=edge_drop,
        )

        # ---- Phase callbacks ----
        callbacks: list[TrainerCallback] = []

        # Lambda warmup: ramp λ from 0 to target over warmup_steps steps
        lambda_warmup_steps = align_cfg.get("lambda_warmup_steps", 0)
        if lambda_warmup_steps > 0:
            target_lambda = align_cfg.get("lambda_align", 0.1)
            logger.info(f"Lambda warmup: λ will ramp 0 → {target_lambda} over {lambda_warmup_steps} steps.")
            callbacks.append(LambdaWarmupCallback(
                loss_fn=self._combined_loss,
                target_lambda=target_lambda,
                warmup_steps=lambda_warmup_steps,
            ))

        experiment = cfg.get("experiment_name", "multi_loss")
        if experiment == "freeze_then_align":
            phase1_epochs = training_cfg.get("phase1_epochs", 5)
            logger.info(f"freeze_then_align: freezing BERT for {phase1_epochs} epochs (Phase 1).")
            model._freeze_bert()  # type: ignore[attr-defined]
            callbacks.append(FreezeThawCallback(model=model, phase1_epochs=phase1_epochs))

        super().__init__(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=callbacks if callbacks else None,
        )

        # Running sums for per-component loss logging (reset each logging interval)
        self._loss_sums: dict[str, float] = {}
        self._loss_count: int = 0

    # ------------------------------------------------------------------
    # compute_loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """Compute the combined training objective.

        Extracts graph data from ``inputs``, runs the SyntaxBertModel forward
        pass, then combines SimCSE + alignment + GNN losses.

        Args:
            model: The ``SyntaxBertModel``.
            inputs: Batch dict from ``SyntaxGraphCollator``:
                - ``input_ids``:         ``(bs, 2, seq_len)``
                - ``attention_mask``:    ``(bs, 2, seq_len)``
                - ``graph_batch``:       PyG Batch (on CPU, moved here)
                - ``graph_batch_aug``:   PyG Batch or None
                - ``token_to_word_maps``: list of alignment maps
            return_outputs: If ``True``, return ``(loss, outputs)`` tuple.
            num_items_in_batch: Ignored (HF Trainer compat).

        Returns:
            Scalar total loss (or ``(loss, outputs)`` if ``return_outputs``).
        """
        # Extract non-tensor items (HF Trainer's _prepare_inputs skips these)
        graph_batch: Batch = inputs["graph_batch"]
        graph_batch_aug: Batch | None = inputs.get("graph_batch_aug")
        token_to_word_maps: list[list[int]] = inputs["token_to_word_maps"]
        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]

        # Move graph data to the same device as the text tensors
        device = input_ids.device
        graph_batch = graph_batch.to(device)
        if graph_batch_aug is not None:
            graph_batch_aug = graph_batch_aug.to(device)

        # ---- Forward pass ----
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_batch=graph_batch,
            token_to_word_maps=token_to_word_maps,
        )

        # SimCSE loss might be None if single-view input was used
        if output.simcse_loss is None:
            loss_simcse = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss_simcse = output.simcse_loss

        # ---- GNN augmented view for L_GNN auxiliary loss ----
        h_gnn_aug: torch.Tensor | None = None
        if (
            graph_batch_aug is not None
            and output.bert_hidden_states is not None
            and self._combined_loss.mu_gnn > 0
        ):
            h_gnn_aug = model._compute_gnn_embeddings(  # type: ignore[attr-defined]
                output.bert_hidden_states, graph_batch_aug, token_to_word_maps
            )

        # ---- Combined loss ----
        losses = self._combined_loss(
            loss_simcse=loss_simcse,
            h_bert=output.h_bert,
            h_gnn=output.h_gnn,
            h_bert_proj=output.h_bert_proj,
            h_gnn_proj=output.h_gnn_proj,
            h_gnn_aug=h_gnn_aug,
        )

        # Accumulate per-component losses for averaged logging
        components = {
            "loss_simcse": losses["simcse"].item(),
            "loss_alignment": losses["alignment"].item(),
            "loss_gnn": losses["gnn"].item(),
        }
        self._last_loss_components = components
        for k, v in components.items():
            self._loss_sums[k] = self._loss_sums.get(k, 0.0) + v
        self._loss_count += 1

        return (losses["total"], output) if return_outputs else losses["total"]

    # ------------------------------------------------------------------
    # create_optimizer — separate LR for BERT and GNN
    # ------------------------------------------------------------------

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW with distinct learning rates for BERT and GNN parameter groups.

        BERT parameters use ``training.learning_rate`` (typically 3e-5).
        GNN parameters use ``training.gnn_learning_rate`` (typically 1e-4),
        since the GNN is trained from scratch while BERT is fine-tuned.

        Returns:
            Configured ``AdamW`` optimizer.
        """
        if self.optimizer is not None:
            return self.optimizer

        training_cfg: dict = (
            OmegaConf.to_container(self.syntax_cfg.training, resolve=True)
            if "training" in self.syntax_cfg
            else {}
        )
        bert_lr = training_cfg.get("learning_rate", 3e-5)
        gnn_lr = training_cfg.get("gnn_learning_rate", 1e-4)
        weight_decay = training_cfg.get("weight_decay", 0.0)

        bert_params: list[torch.nn.Parameter] = []
        gnn_params: list[torch.nn.Parameter] = []
        other_params: list[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "gnn" in name:
                gnn_params.append(param)
            elif "bert" in name:
                bert_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": bert_params, "lr": bert_lr, "weight_decay": weight_decay},
                {"params": gnn_params, "lr": gnn_lr, "weight_decay": weight_decay},
                {"params": other_params, "lr": bert_lr, "weight_decay": weight_decay},
            ]
        )

        logger.info(
            f"Optimizer: AdamW | BERT lr={bert_lr} ({len(bert_params)} tensors) | "
            f"GNN lr={gnn_lr} ({len(gnn_params)} tensors) | "
            f"Other lr={bert_lr} ({len(other_params)} tensors)"
        )
        self.optimizer = optimizer
        return optimizer

    # ------------------------------------------------------------------
    # _save_checkpoint — persist BERT + GNN weights
    # ------------------------------------------------------------------

    def _save_checkpoint(self, model: torch.nn.Module, trial, **kwargs) -> None:
        """Override HF Trainer checkpoint saving to persist both branches.

        In addition to the standard HF checkpoint (optimizer, scheduler, RNG
        state), this writes:
          - ``bert_weights/``   — HuggingFace-format BERT weights
          - ``gnn_state.pt``    — GNN + projection-head weights
          - ``model_config.json`` — hyper-parameters for reconstruction

        At the end of training, ``save_bert_only()`` produces a lean
        ``bert_inference/`` directory containing only BERT weights for
        deployment without any GNN dependency.

        Args:
            model: The ``SyntaxBertModel`` being trained.
            trial: Optuna trial (passed through from HF Trainer, may be None).
            **kwargs: Forwarded to super (signature varies across transformers versions).
        """
        # Let HF Trainer handle optimizer / scheduler / RNG / tokenizer state
        super()._save_checkpoint(model, trial, **kwargs)

        # Write BERT + GNN weights alongside the HF checkpoint
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        output_dir = Path(self.args.output_dir) / checkpoint_folder
        if hasattr(model, "save_checkpoint"):
            model.save_checkpoint(str(output_dir), tokenizer=self.tokenizer)  # type: ignore[union-attr]
            logger.info(f"[SyntaxCLTrainer] Saved BERT+GNN checkpoint → {output_dir}")

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """Save the final model after training completes.

        Writes both the full ``SyntaxBertModel`` checkpoint and a lightweight
        BERT-only inference checkpoint under ``<output_dir>/bert_inference/``.

        Args:
            output_dir: Destination directory. Defaults to ``args.output_dir``.
            _internal_call: Internal HF Trainer flag (forwarded to super).
        """
        out = output_dir or self.args.output_dir
        super().save_model(out, _internal_call=_internal_call)

        model = self.model
        if hasattr(model, "save_checkpoint"):
            model.save_checkpoint(out, tokenizer=self.tokenizer)  # type: ignore[union-attr]
            logger.info(f"[SyntaxCLTrainer] Full checkpoint saved → {out}")

        if hasattr(model, "save_bert_only"):
            bert_inference_dir = str(Path(out) / "bert_inference")
            model.save_bert_only(bert_inference_dir, tokenizer=self.tokenizer)  # type: ignore[union-attr]
            logger.info(
                f"[SyntaxCLTrainer] BERT-only inference checkpoint saved → {bert_inference_dir}"
            )

    # ------------------------------------------------------------------
    # log — inject per-component loss averages into metrics
    # ------------------------------------------------------------------

    def log(self, logs: dict[str, float], start_time: float | None = None, **kwargs: Any) -> None:
        """Override Trainer.log to add averaged per-component loss metrics.

        Injects ``loss_simcse``, ``loss_alignment``, and ``loss_gnn`` into every
        log event, then resets the running accumulators.

        Args:
            logs: Metric dict assembled by HuggingFace Trainer.
            start_time: Passed positionally by transformers ≥4.46.
            **kwargs: Extra keyword args forwarded to the super.
        """
        if self._loss_count > 0:
            for k, v in self._loss_sums.items():
                logs[k] = round(v / self._loss_count, 6)
            self._loss_sums = {}
            self._loss_count = 0
        if start_time is not None:
            super().log(logs, start_time, **kwargs)
        else:
            super().log(logs, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_sentences(data_args) -> list[str]:
        """Load raw sentences from the training text file."""
        train_path = Path(data_args.train_file)
        if not train_path.exists():
            logger.error(f"Training file not found: {train_path}")
            logger.error("Download it with:  uv run python scripts/download_wiki.py")
            return []
        with open(train_path, encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(sentences):,} sentences from {train_path}")
        return sentences
