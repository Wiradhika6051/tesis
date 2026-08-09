import torch

from tqdm import tqdm


class Evaluator:

    def __init__(
        self,
        model,
        criterion,
        device
    ):

        self.model = model
        self.criterion = criterion
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        loader
    ):

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_predictions = []
        all_labels = []

        progress = tqdm(
            loader,
            total=len(loader),
            desc="Validation"
        )

        for graph, labels in progress:

            graph = graph.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            #
            # Forward pass
            #
            logits = self.model(
                graph
            )

            #
            # Loss
            #
            loss = self.criterion(
                logits,
                labels
            )

            #
            # Predictions
            #
            predictions = torch.argmax(
                logits,
                dim=1
            )

            #
            # Batch statistics
            #
            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                *
                batch_size
            )

            total_samples += batch_size

            #
            # Store predictions and labels
            #
            all_predictions.append(
                predictions.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        #
        # Empty validation set
        #
        if total_samples == 0:

            return {
                "loss": float("inf"),
                "accuracy": 0.0,
                "predictions": torch.empty(
                    0,
                    dtype=torch.long
                ),
                "labels": torch.empty(
                    0,
                    dtype=torch.long
                )
            }

        #
        # Average validation loss
        #
        average_loss = (
            total_loss
            /
            total_samples
        )

        #
        # Combine batches
        #
        predictions = torch.cat(
            all_predictions
        )

        labels = torch.cat(
            all_labels
        )

        #
        # Accuracy
        #
        accuracy = (
            predictions == labels
        ).float().mean().item()

        #
        # Restore training mode
        #
        self.model.train()

        print(
            f"Validation "
            f"Loss={average_loss:.4f} "
            f"Accuracy={accuracy:.4f}"
        )

        return {

            "loss":
                average_loss,

            "accuracy":
                accuracy,

            "predictions":
                predictions,

            "labels":
                labels
        }