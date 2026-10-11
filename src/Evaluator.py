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
    def evaluate(self, loader):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_pair_ids = []
        all_sample_ids = []

        progress = tqdm(
            loader,
            total=len(loader),
            desc="Evaluation",
        )

        for graph, labels, pair_ids, sample_ids in progress:
            graph = graph.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(graph)
            loss = self.criterion(logits, labels)

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())

            all_pair_ids.extend(pair_ids)
            all_sample_ids.extend(sample_ids)

            progress.set_postfix(loss=f"{loss.item():.4f}")

        if total_samples == 0:
            return {
                "loss": float("inf"),
                "accuracy": 0.0,
                "predictions": torch.empty(0, dtype=torch.long),
                "labels": torch.empty(0, dtype=torch.long),
                "probabilities": torch.empty((0, 2)),
                "pair_ids": [],
                "sample_ids": [],
                "records": [],
            }

        predictions = torch.cat(all_predictions)
        labels = torch.cat(all_labels)
        probabilities = torch.cat(all_probabilities)

        average_loss = total_loss / total_samples
        accuracy = (predictions == labels).float().mean().item()

        records = []

        for i in range(total_samples):
            actual = int(labels[i].item())
            predicted = int(predictions[i].item())

            if actual == 1 and predicted == 1:
                outcome = "TP"
            elif actual == 0 and predicted == 0:
                outcome = "TN"
            elif actual == 0 and predicted == 1:
                outcome = "FP"
            else:
                outcome = "FN"

            records.append({
                "sample_id": all_sample_ids[i],
                "pair_id": all_pair_ids[i],
                "label": actual,
                "predicted": predicted,
                "score_0": float(probabilities[i, 0].item()),
                "score_1": float(probabilities[i, 1].item()),
                "outcome": outcome,
            })

        print(
            f"Evaluation "
            f"Loss={average_loss:.4f} "
            f"Accuracy={accuracy:.4f}"
        )

        return {
            "loss": average_loss,
            "accuracy": accuracy,
            "predictions": predictions,
            "labels": labels,
            "probabilities": probabilities,
            "pair_ids": all_pair_ids,
            "sample_ids": all_sample_ids,
            "records": records,
        }



    @torch.no_grad()
    def evaluate_with_samples(
        self,
        loader,
        samples
    ):
        """
        Evaluate model and preserve the mapping
        between predictions and original samples.

        IMPORTANT:
        loader must use shuffle=False.
        """

        self.model.eval()

        prediction_results = []

        sample_index = 0

        progress = tqdm(
            loader,
            total=len(loader),
            desc="Evaluation"
        )

        for graph, labels in progress:

            graph = graph.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            #
            # Forward pass.
            #
            logits = self.model(
                graph
            )

            #
            # Convert logits to probabilities.
            #
            probabilities = torch.softmax(
                logits,
                dim=1
            )

            #
            # Predicted class.
            #
            predictions = torch.argmax(
                logits,
                dim=1
            )

            #
            # Move results to CPU.
            #
            predictions = predictions.cpu()

            labels = labels.cpu()

            probabilities = probabilities.cpu()

            batch_size = labels.size(0)

            #
            # Match each prediction
            # to its original sample.
            #
            for i in range(batch_size):

                sample = samples[
                    sample_index
                ]

                pair_id = (

                    f"{sample.repo}"
                    f"|{sample.file_path}"
                    f"|{sample.parent_commit}"

                )

                prediction_results.append({

                    #
                    # Pair identifier.
                    #
                    "pair_id":
                        pair_id,

                    #
                    # Sample information.
                    #
                    "repo":
                        sample.repo,

                    "file":
                        sample.file_path,

                    "parent_commit":
                        sample.parent_commit,

                    #
                    # Actual and predicted label.
                    #
                    "label":
                        labels[i].item(),

                    "predicted":
                        predictions[i].item(),

                    #
                    # Model confidence.
                    #
                    "score_0":
                        probabilities[i][0].item(),

                    "score_1":
                        probabilities[i][1].item()

                })

                sample_index += 1

        self.model.train()

        #
        # Safety check.
        #
        if sample_index != len(samples):

            print()

            print(
                "WARNING: Sample count mismatch"
            )

            print(
                "Predictions:",
                sample_index
            )

            print(
                "Samples:",
                len(samples)
            )

        return prediction_results