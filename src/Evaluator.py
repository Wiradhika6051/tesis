import torch
from tqdm import tqdm


class Evaluator:
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device

    @torch.no_grad()
    def evaluate(self, loader):
        was_training = self.model.training
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_pair_ids = []
        all_sample_uids = []
        all_metadata = []

        progress = tqdm(
            loader,
            total=len(loader),
            desc="Evaluation",
        )

        for (
            graph,
            labels,
            pair_ids,
            sample_uids,
            metadata,
        ) in progress:
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
            all_sample_uids.extend(sample_uids)
            all_metadata.extend(metadata)

            progress.set_postfix(loss=f"{loss.item():.4f}")

        if total_samples == 0:
            self.model.train(was_training)
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
        accuracy = (
            predictions == labels
        ).float().mean().item()

        records = []

        for i, metadata in enumerate(all_metadata):
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

            record = dict(metadata)
            record.update({
                "sample_uid": all_sample_uids[i],
                "pair_id": list(all_pair_ids[i]),
                "label": actual,
                "predicted": predicted,
                "score_0": float(probabilities[i, 0].item()),
                "score_1": float(probabilities[i, 1].item()),
                "outcome": outcome,
            })

            records.append(record)

        self.model.train(was_training)

        print(
            f"Evaluation Loss={average_loss:.4f} "
            f"Accuracy={accuracy:.4f}"
        )

        return {
            "loss": average_loss,
            "accuracy": accuracy,
            "predictions": predictions,
            "labels": labels,
            "probabilities": probabilities,
            "pair_ids": all_pair_ids,
            "sample_ids": all_sample_uids,
            "records": records,
        }