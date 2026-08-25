def build_model_outcomes(
    evaluation_results
):

    predictions = (

        evaluation_results[
            "predictions"
        ].tolist()

    )

    labels = (

        evaluation_results[
            "labels"
        ].tolist()

    )

    pair_ids = (

        evaluation_results[
            "pair_ids"
        ]

    )

    sample_ids = (

        evaluation_results[
            "sample_ids"
        ]

    )

    results = []

    for (

        prediction,
        label,
        pair_id,
        sample_id

    ) in zip(

        predictions,

        labels,

        pair_ids,

        sample_ids

    ):

        #
        # Determine classification outcome.
        #
        if label == 1 and prediction == 1:

            outcome = "TP"

        elif label == 0 and prediction == 0:

            outcome = "TN"

        elif label == 0 and prediction == 1:

            outcome = "FP"

        elif label == 1 and prediction == 0:

            outcome = "FN"

        else:

            outcome = "UNKNOWN"

        results.append({

            "pair_id":
                pair_id,

            "sample_id":
                sample_id,

            "label":
                label,

            "prediction":
                prediction,

            "outcome":
                outcome

        })

    return results