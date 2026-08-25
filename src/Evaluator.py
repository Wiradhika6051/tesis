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

        all_pair_ids = []

        all_sample_ids = []

        progress = tqdm(

            loader,

            total=len(loader),

            desc="Validation"

        )

        for (

            graph,

            labels,

            pair_ids,

            sample_ids

        ) in progress:

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
            # Loss.
            #
            loss = self.criterion(

                logits,

                labels

            )

            #
            # Predictions.
            #
            predictions = torch.argmax(

                logits,

                dim=1

            )

            #
            # Batch statistics.
            #
            batch_size = labels.size(
                0
            )

            total_loss += (

                loss.item()

                *
                batch_size

            )

            total_samples += batch_size

            #
            # Store results.
            #
            all_predictions.append(

                predictions.cpu()

            )

            all_labels.append(

                labels.cpu()

            )

            #
            # Metadata stays as Python objects.
            #
            all_pair_ids.extend(
                pair_ids
            )

            all_sample_ids.extend(
                sample_ids
            )

            progress.set_postfix(

                loss=f"{loss.item():.4f}"

            )

        #
        # Empty validation set.
        #
        if total_samples == 0:

            return {

                "loss":
                    float("inf"),

                "accuracy":
                    0.0,

                "predictions":
                    torch.empty(
                        0,
                        dtype=torch.long
                    ),

                "labels":
                    torch.empty(
                        0,
                        dtype=torch.long
                    ),

                "pair_ids":
                    [],

                "sample_ids":
                    []

            }

        #
        # Average loss.
        #
        average_loss = (

            total_loss

            /

            total_samples

        )

        predictions = torch.cat(
            all_predictions
        )

        labels = torch.cat(
            all_labels
        )

        accuracy = (

            predictions == labels

        ).float().mean().item()

        #
        # Restore training mode.
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
                labels,

            "pair_ids":
                all_pair_ids,

            "sample_ids":
                all_sample_ids

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