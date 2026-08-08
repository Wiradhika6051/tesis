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
        correct = 0

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
            # Statistics
            #
            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            correct += (
                predictions == labels
            ).sum().item()

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
        # Avoid division by zero
        #
        if total_samples == 0:

            return float("inf")

        average_loss = (
            total_loss
            /
            total_samples
        )

        accuracy = (
            correct
            /
            total_samples
        )

        #
        # Restore training mode
        #
        self.model.train()

        print(
            f"Validation "
            f"Loss={average_loss:.4f} "
            f"Accuracy={accuracy:.4f}"
        )

        return average_loss