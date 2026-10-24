import os

from tqdm import tqdm

from src.checkpoint import (
    save_checkpoint,
    load_checkpoint
)
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
        patience=10
    ):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.checkpoint_path = checkpoint_path
        self.best_model_path = best_model_path

        self.patience = patience
        self.evaluator = Evaluator(
            model=self.model,
            criterion=self.criterion,
            device=self.device
        )
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
        config=""
    ):

        self._resume()

        for epoch in range(
            self.start_epoch,
            epochs
        ):

            train_loss = self._train_epoch(
                epoch,
                epochs,
                train_loader
            )

            result = self.evaluator.evaluate(
                val_loader
            )

            val_loss = result["loss"]

            print(
                f"Epoch {epoch + 1} "
                f"Train={train_loss:.4f} "
                f"Val={val_loss:.4f}"
            )

            self._save_checkpoint(
                epoch,
                train_loss,
                val_loss,
                self.checkpoint_path,
                split_file,
                vocab_file,
                dataset_file,
                config
            )

            improved = (
                val_loss < self.best_val_loss
            )

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
                    config
                )

                print(
                    f"✓ New best model "
                    f"(val={val_loss:.4f})"
                )

            else:

                self.patience_counter += 1

                print(
                    f"No improvement "
                    f"({self.patience_counter}/"
                    f"{self.patience})"
                )

            if (
                self.patience_counter
                >=
                self.patience
            ):

                print(
                    f"Early stopping "
                    f"at epoch {epoch+1}"
                )

                break

    def _train_epoch(
        self,
        epoch,
        epochs,
        loader
    ):
    
        self.model.train()
    
        total_loss = 0
    
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch {epoch+1}/{epochs}"
        )
    
        for batch in progress:
        
            #
            # First two elements are always
            # graph and labels.
            #
            graph = batch[0]
    
            labels = batch[1]

            sample_id = batch[3]
    
            graph = graph.to(
                self.device
            )
    
            labels = labels.to(
                self.device
            )
    
            #
            # Reset gradients
            #
            self.optimizer.zero_grad()
    
            #
            # Forward pass
            #
            validate_graph(graph, sample_id=sample_id)
            logits = self.model(
                graph
            )
    
            #
            # Compute loss
            #
            loss = self.criterion(
                logits,
                labels
            )
    
            #
            # Backpropagation
            #
            loss.backward()
    
            #
            # Update parameters
            #
            self.optimizer.step()
    
            total_loss += loss.item()
    
            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )
    
        return (
            total_loss
            /
            len(loader)
        )

    def _resume(self):

        if not os.path.exists(
            self.checkpoint_path
        ):
            return

        checkpoint = load_checkpoint(
            self.checkpoint_path,
            self.model,
            self.optimizer
        )

        self.start_epoch = (
            checkpoint["epoch"] + 1
        )

        self.best_val_loss = (
            checkpoint["val_loss"]
        )

        print(
            f"Resume from epoch "
            f"{self.start_epoch}"
        )

    def _save_checkpoint(
        self,
        epoch,
        train_loss,
        val_loss,
        checkpoint_file,
        split_file,
        vocab_file,
        dataset_file,
        config
    ):

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
            config=config
        )

    def load_best_model(self):
        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError(
                f"Best model checkpoint not found: "
                f"{self.best_model_path}"
            )

        checkpoint = load_checkpoint(
            self.best_model_path,
            self.model,
            self.optimizer,
        )

        print(
            "Loaded best model checkpoint: "
            f"{self.best_model_path}"
        )

        return checkpoint
        
import torch

def validate_graph(graph, sample_id=None):
    x = graph.x
    edge_index = graph.edge_index

    prefix = f"[GRAPH VALIDATION] sample={sample_id}"

    assert x is not None, f"{prefix}: x is None"
    assert edge_index is not None, f"{prefix}: edge_index is None"

    assert x.ndim == 2, (
        f"{prefix}: x must be 2D, got {tuple(x.shape)}"
    )

    num_nodes = x.size(0)

    assert edge_index.ndim == 2, (
        f"{prefix}: edge_index must be 2D, "
        f"got {tuple(edge_index.shape)}"
    )
    assert edge_index.size(0) == 2, (
        f"{prefix}: expected edge_index shape [2, E], "
        f"got {tuple(edge_index.shape)}"
    )
    assert edge_index.dtype == torch.long, (
        f"{prefix}: edge_index must be torch.long, "
        f"got {edge_index.dtype}"
    )

    # Inspect indices on CPU to avoid hiding the original error
    # behind another CUDA operation.
    edges = edge_index.detach().cpu()

    if edges.numel() > 0:
        min_idx = edges.min().item()
        max_idx = edges.max().item()

        assert min_idx >= 0, (
            f"{prefix}: negative node index {min_idx}"
        )
        assert max_idx < num_nodes, (
            f"{prefix}: edge index {max_idx} exceeds "
            f"num_nodes={num_nodes}"
        )

    # Check feature values on CPU too.
    features = x.detach().cpu()

    assert torch.isfinite(features).all().item(), (
        f"{prefix}: x contains NaN or Inf"
    )

    print(
        f"{prefix}: OK | "
        f"nodes={num_nodes}, "
        f"edges={edge_index.size(1)}, "
        f"features={tuple(x.shape)}, "
        f"device={x.device}"
    )