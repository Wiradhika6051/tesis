
"""Training loop for graph-based vulnerability classification.

Fresh training is the default. Set resume=True to resume explicitly.

The latest checkpoint supports continuation. The best checkpoint is used
for final evaluation.
"""

import os

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

        self.checkpoint_path = os.fspath(checkpoint_path)
        self.best_model_path = os.fspath(best_model_path)

        self.patience = int(patience)
        self.resume = bool(resume)
        self.clean_checkpoints_on_fresh_start = bool(
            clean_checkpoints_on_fresh_start
        )

        if self.patience < 0:
            raise ValueError(
                f"patience must be >= 0, got {self.patience}"
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

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(
                f"epochs must be a positive integer, got {epochs}"
            )

        if self.resume:
            self._resume()
        else:
            self._start_fresh_run()

        if self.start_epoch >= epochs:
            print(
                f"Checkpoint already reached epoch "
                f"{self.start_epoch}; target is {epochs}. "
                "No training epochs remain."
            )
            return

        for epoch in range(self.start_epoch, epochs):
            train_loss = self._train_epoch(
                epoch,
                epochs,
                train_loader,
            )

            result = self.evaluator.evaluate(val_loader)
            val_loss = float(result["loss"])

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Train={train_loss:.4f} "
                f"Val={val_loss:.4f}"
            )

            improved = val_loss < self.best_val_loss

            if improved:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                self._save_checkpoint(
                    epoch,
                    train_loss,
                    val_loss,
                    self.best_model_path,
                    split_file,
                    vocab_file,
                    dataset_file,
                    config,
                )

                print(
                    f"New best model (val={val_loss:.4f})"
                )
            else:
                self.patience_counter += 1

                print(
                    f"No improvement "
                    f"({self.patience_counter}/{self.patience})"
                )

            # Save after updating best-loss and patience state.
            self._save_checkpoint(
                epoch,
                train_loss,
                val_loss,
                self.checkpoint_path,
                split_file,
                vocab_file,
                dataset_file,
                config,
            )

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    def _train_epoch(self, epoch, epochs, loader):
        """Run one training epoch and return mean batch loss."""

        self.model.train()

        total_loss = 0.0
        batch_count = 0

        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for batch in progress:
            # The first two elements are graph and labels.
            graph = batch[0]
            labels = batch[1]
            sample_id = batch[3] if len(batch) > 3 else None

            graph = graph.to(self.device)
            labels = labels.to(self.device)

            validate_graph(graph, sample_id=sample_id)

            self.optimizer.zero_grad()

            logits = self.model(graph)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        if batch_count == 0:
            raise ValueError(
                "Training loader produced no batches."
            )

        return total_loss / batch_count

    def _resume(self):
        """Restore the latest checkpoint and training state."""

        if not os.path.isfile(self.checkpoint_path):
            print("No latest checkpoint found; starting fresh.")
            self._reset_training_state()
            return

        checkpoint = load_checkpoint(
            self.checkpoint_path,
            self.model,
            self.optimizer,
        )

        self.start_epoch = int(checkpoint["epoch"]) + 1

        # New checkpoints preserve the historical best loss.
        # Older checkpoints only contain the latest validation loss.
        self.best_val_loss = float(
            checkpoint.get(
                "best_val_loss",
                checkpoint["val_loss"],
            )
        )

        self.patience_counter = int(
            checkpoint.get("patience_counter", 0)
        )

        print(
            f"Resuming at epoch {self.start_epoch} "
            f"from {self.checkpoint_path}"
        )

    def _start_fresh_run(self):
        """Reset state and optionally remove stale checkpoints."""

        self._reset_training_state()

        if self.clean_checkpoints_on_fresh_start:
            paths = {
                self.checkpoint_path,
                self.best_model_path,
            }

            for path in paths:
                if os.path.isfile(path):
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
        """Save model state, optimizer state, and trainer state."""

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
            trainer_state={
                "best_val_loss": self.best_val_loss,
                "patience_counter": self.patience_counter,
            },
        )

    def load_best_model(self):
        """Load the best validation checkpoint before testing."""

        if not os.path.isfile(self.best_model_path):
            raise FileNotFoundError(
                "Best model checkpoint not found: "
                f"{self.best_model_path}"
            )

        checkpoint = load_checkpoint(
            self.best_model_path,
            self.model,
            optimizer=None,
        )

        print(
            "Loaded best model checkpoint: "
            f"{self.best_model_path}"
        )

        return checkpoint


def validate_graph(graph, sample_id=None):
    """Validate graph attributes and edge-index bounds."""

    prefix = f"[GRAPH VALIDATION] sample={sample_id}"

    edge_index = getattr(graph, "edge_index", None)
    node_types = getattr(graph, "node_types", None)
    node_token_vectors = getattr(
        graph,
        "node_token_vectors",
        None,
    )

    if edge_index is None:
        raise ValueError(f"{prefix}: edge_index is None")

    if node_types is None:
        raise ValueError(f"{prefix}: node_types is None")

    if node_token_vectors is None:
        raise ValueError(
            f"{prefix}: node_token_vectors is None"
        )

    num_nodes = len(node_token_vectors)

    if node_types.numel() != num_nodes:
        raise ValueError(
            f"{prefix}: node_types has {node_types.numel()} "
            f"entries, but node_token_vectors has {num_nodes}"
        )

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"{prefix}: edge_index must have shape "
            f"[2, num_edges], got {tuple(edge_index.shape)}"
        )

    if edge_index.numel() > 0:
        min_idx = int(edge_index.min().item())
        max_idx = int(edge_index.max().item())

        if min_idx < 0:
            raise ValueError(
                f"{prefix}: negative edge index {min_idx}"
            )

        if max_idx >= num_nodes:
            raise ValueError(
                f"{prefix}: edge index {max_idx} exceeds "
                f"valid node range [0, {num_nodes - 1}]"
            )

    return True
