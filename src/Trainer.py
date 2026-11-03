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
        """Train for up to `epochs` total epochs."""
        if len(train_loader) == 0:
            raise ValueError("train_loader is empty.")
        if len(val_loader) == 0:
            raise ValueError("val_loader is empty.")
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
            return

        for epoch in range(self.start_epoch, epochs):
            train_loss = self._train_epoch(
                epoch, epochs, train_loader
            )

            result = self.evaluator.evaluate(val_loader)
            val_loss = result["loss"]

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Train={train_loss:.4f} Val={val_loss:.4f}"
            )

            self._save_checkpoint(
                epoch, train_loss, val_loss,
                self.checkpoint_path,
                split_file, vocab_file, dataset_file, config,
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                self._save_checkpoint(
                    epoch, train_loss, val_loss,
                    self.best_model_path,
                    split_file, vocab_file, dataset_file, config,
                )
                print(f"New best model (val={val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(
                    f"No improvement "
                    f"({self.patience_counter}/{self.patience})"
                )

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


    def _resume(self):
        """Resume from the latest checkpoint when explicitly requested."""
        if not os.path.exists(self.checkpoint_path):
            print("No latest checkpoint found; starting fresh.")
            return

        checkpoint = load_checkpoint(
            self.checkpoint_path, self.model, self.optimizer
        )

        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_val_loss = float(checkpoint["val_loss"])

        print(
            f"Resuming from epoch {self.start_epoch}; "
            f"checkpoint={self.checkpoint_path}"
        )


    def _start_fresh_run(self):
        """Reset training state and remove stale checkpoints."""
        self._reset_training_state()

        if self.clean_checkpoints_on_fresh_start:
            for path in (self.checkpoint_path, self.best_model_path):
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Removed old checkpoint: {path}")

        print("Starting fresh training run.")


    def _save_checkpoint(
        self,
        epoch,
        train_loss,
        val_loss,
        checkpoint_file,
        split_file,
        vocab_file,
        dataset_file,
        config,
    ):
        os.makedirs(os.path.dirname(checkpoint_file) or ".", exist_ok=True)

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            checkpoint_file=checkpoint_file,
            split_file=split_file,
            vocab_file=vocab_file,
            dataset_file=dataset_file,
            config=config,
        )


    def load_best_model(self):
        """Restore the best validation checkpoint before testing."""
        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError(
                f"Best model checkpoint not found: {self.best_model_path}"
            )

        checkpoint = load_checkpoint(
            self.best_model_path,
            self.model,
            self.optimizer,
        )

        print(f"Loaded best model: {self.best_model_path}")
        return checkpoint   