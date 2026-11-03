"""Trainer with explicit fresh-training vs checkpoint-resume behavior.

Default behavior is a fresh run: stale checkpoints are removed before training,
and existing checkpoints are never resumed implicitly.

To resume deliberately, instantiate Trainer(..., resume=True) and call fit().
This implementation preserves the checkpoint helper API used by the original code:
    save_checkpoint(model=..., optimizer=..., epoch=..., train_loss=...,
                    val_loss=..., checkpoint_file=..., split_file=...,
                    vocab_file=..., dataset_file=..., config=...)
    load_checkpoint(path, model, optimizer)
"""

import os
from pathlib import Path

from tqdm import tqdm

from src.checkpoint import save_checkpoint, load_checkpoint
from src.Evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        checkpoint_path,
        best_model_path,
        patience=10,
        resume=False,
        clean_checkpoints_on_fresh_start=True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.checkpoint_path = str(checkpoint_path)
        self.best_model_path = str(best_model_path)
        self.patience = int(patience)
        self.resume = bool(resume)
        self.clean_checkpoints_on_fresh_start = bool(
            clean_checkpoints_on_fresh_start
        )

        self.evaluator = Evaluator(
            model=self.model,
            criterion=self.criterion,
            device=self.device,
        )

        self._reset_training_state()

    def _reset_training_state(self):
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        split_file="",
        vocab_file="",
        dataset_file="",
        config="",
    ):
        """Train the model.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Total target epoch count, not additional epochs.
            resume: Set on Trainer construction. False (default) means fresh run.
        """
        if len(train_loader) == 0:
            raise ValueError("train_loader is empty; cannot train the model.")
        if len(val_loader) == 0:
            raise ValueError("val_loader is empty; cannot select a best checkpoint.")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}.")

        if self.resume:
            self._resume()
        else:
            self._start_fresh_run()

        if self.start_epoch >= epochs:
            print(
                f"Checkpoint already reached epoch {self.start_epoch}; "
                f"target epochs={epochs}. No training epochs remain."
            )