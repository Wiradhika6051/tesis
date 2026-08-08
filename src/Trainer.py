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

            val_loss = self.evaluator.evaluate(
                val_loader
            )

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

        for graph, labels in progress:
        
            print("1")
        
            graph = graph.to(self.device)
        
            print("2")
        
            labels = labels.to(self.device)
        
            print("3")
        
            logits = self.model(graph)
            print("logits:", logits.shape)
            print("labels:", labels.shape)
            print("4")
        
            loss = self.criterion(
                logits,
                labels
            )
        
            print("5")
        
            loss.backward()
        
            print("6")

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