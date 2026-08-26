from collections import Counter


def compare_directional_predictions(

    forward_results,

    backward_results

):

    """
    Compare predictions from forward and backward
    slice models for the same samples.

    Returns a record for every sample that exists
    in both evaluation results.
    """

    #
    # --------------------------------------------------
    # Build lookup for forward results.
    # --------------------------------------------------
    #

    forward_lookup = {}

    for (

        sample_id,
        prediction,
        label

    ) in zip(

        forward_results[
            "sample_ids"
        ],

        forward_results[
            "predictions"
        ].tolist(),

        forward_results[
            "labels"
        ].tolist()

    ):

        forward_lookup[
            sample_id
        ] = {

            "prediction":
                prediction,

            "label":
                label

        }

    #
    # --------------------------------------------------
    # Build lookup for backward results.
    # --------------------------------------------------
    #

    backward_lookup = {}

    for (

        sample_id,
        prediction,
        label

    ) in zip(

        backward_results[
            "sample_ids"
        ],

        backward_results[
            "predictions"
        ].tolist(),

        backward_results[
            "labels"
        ].tolist()

    ):

        backward_lookup[
            sample_id
        ] = {

            "prediction":
                prediction,

            "label":
                label

        }

    #
    # --------------------------------------------------
    # Compare only samples present in both.
    # --------------------------------------------------
    #

    results = []

    forward_only = []

    backward_only = []

    for sample_id, forward_data in forward_lookup.items():

        backward_data = (
            backward_lookup.get(
                sample_id
            )
        )

        if backward_data is None:

            forward_only.append(
                sample_id
            )

            continue

        forward_prediction = (
            forward_data[
                "prediction"
            ]
        )

        backward_prediction = (
            backward_data[
                "prediction"
            ]
        )

        forward_label = (
            forward_data[
                "label"
            ]
        )

        backward_label = (
            backward_data[
                "label"
            ]
        )

        #
        # Sanity check.
        #
        if forward_label != backward_label:

            print()

            print(
                "WARNING: LABEL MISMATCH"
            )

            print(
                "Sample:",
                sample_id
            )

            print(
                "Forward label:",
                forward_label
            )

            print(
                "Backward label:",
                backward_label
            )

            continue

        label = forward_label

        forward_correct = (

            forward_prediction
            ==
            label

        )

        backward_correct = (

            backward_prediction
            ==
            label

        )

        #
        # Categorize outcome.
        #
        if (

            forward_correct
            and
            backward_correct

        ):

            category = (
                "both_correct"
            )

        elif (

            forward_correct
            and
            not backward_correct

        ):

            category = (
                "forward_wins"
            )

        elif (

            not forward_correct
            and
            backward_correct

        ):

            category = (
                "backward_wins"
            )

        else:

            category = (
                "both_wrong"
            )

        results.append({

            "sample_id":
                sample_id,

            "label":
                label,

            "forward_prediction":
                forward_prediction,

            "backward_prediction":
                backward_prediction,

            "forward_correct":
                forward_correct,

            "backward_correct":
                backward_correct,

            "category":
                category

        })

    #
    # Samples present only in backward results.
    #
    for sample_id in backward_lookup:

        if sample_id not in forward_lookup:

            backward_only.append(
                sample_id
            )

    return {

        "results":
            results,

        "forward_only":
            forward_only,

        "backward_only":
            backward_only

    }


def print_directional_prediction_comparison(
    analysis
):

    results = analysis[
        "results"
    ]

    categories = Counter(

        result[
            "category"
        ]

        for result in results

    )

    total = len(
        results
    )

    print()

    print(
        "=" * 80
    )

    print(
        "FORWARD VS BACKWARD PREDICTION COMPARISON"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Samples compared:",
        total
    )

    print()

    ordered_categories = [

        "both_correct",

        "forward_wins",

        "backward_wins",

        "both_wrong"

    ]

    for category in ordered_categories:

        count = categories.get(
            category,
            0
        )

        percentage = (

            count
            /
            total
            *
            100

            if total > 0

            else 0.0

        )

        print(

            f"{category:20s}: "
            f"{count:4d} "
            f"({percentage:6.2f}%)"

        )

    print()

    print(
        "Forward-only IDs:",
        len(
            analysis[
                "forward_only"
            ]
        )
    )

    print(
        "Backward-only IDs:",
        len(
            analysis[
                "backward_only"
            ]
        )
    )

    #
    # Prediction disagreement.
    #
    disagreements = [

        result

        for result in results

        if (

            result[
                "forward_prediction"
            ]

            !=

            result[
                "backward_prediction"
            ]

        )

    ]

    print()

    print(
        "Prediction disagreements:",
        len(disagreements)
    )

    if total > 0:

        print(

            "Disagreement ratio:",
            f"{len(disagreements) / total * 100:.2f}%"

        )

    return analysis