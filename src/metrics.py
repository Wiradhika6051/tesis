def compute_metrics(
    labels,
    predictions
):
    """
    Compute binary classification metrics.

    Args:
        labels:
            Ground-truth labels.

        predictions:
            Model predictions.

    Returns:
        Dictionary containing classification metrics.
    """

    #
    # Convert tensors to Python lists if necessary.
    #
    if hasattr(labels, "tolist"):
        labels = labels.tolist()

    if hasattr(predictions, "tolist"):
        predictions = predictions.tolist()

    if len(labels) != len(predictions):
        raise ValueError(
            "Labels and predictions must "
            "have the same length."
        )

    if len(labels) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0
        }

    #
    # Confusion matrix.
    #
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for label, prediction in zip(
        labels,
        predictions
    ):

        if label == 1 and prediction == 1:
            true_positive += 1

        elif label == 0 and prediction == 0:
            true_negative += 1

        elif label == 0 and prediction == 1:
            false_positive += 1

        elif label == 1 and prediction == 0:
            false_negative += 1

    #
    # Accuracy.
    #
    total = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )

    accuracy = (
        (true_positive + true_negative)
        /
        total
    )

    #
    # Precision.
    #
    precision_denominator = (
        true_positive
        + false_positive
    )

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = (
            true_positive
            /
            precision_denominator
        )

    #
    # Recall.
    #
    recall_denominator = (
        true_positive
        + false_negative
    )

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = (
            true_positive
            /
            recall_denominator
        )

    #
    # F1 score.
    #
    f1_denominator = (
        precision
        + recall
    )

    if f1_denominator == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            *
            precision
            *
            recall
            /
            f1_denominator
        )

    return {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "true_positive":
            true_positive,

        "true_negative":
            true_negative,

        "false_positive":
            false_positive,

        "false_negative":
            false_negative
    }


def print_metrics(
    metrics
):
    """
    Print classification metrics.
    """

    print()
    print("=" * 50)
    print("Classification Metrics")
    print("=" * 50)

    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision          : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall             : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score           : "
        f"{metrics['f1']:.4f}"
    )

    print()
    print("Confusion Matrix")
    print("-" * 50)

    print(
        f"True Positive      : "
        f"{metrics['true_positive']}"
    )

    print(
        f"True Negative      : "
        f"{metrics['true_negative']}"
    )

    print(
        f"False Positive     : "
        f"{metrics['false_positive']}"
    )

    print(
        f"False Negative     : "
        f"{metrics['false_negative']}"
    )

    print("=" * 50)