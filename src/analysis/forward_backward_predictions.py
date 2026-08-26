from collections import Counter


def compare_forward_backward_predictions(
    forward_results,
    backward_results
):

    #
    # --------------------------------------------------
    # Extract predictions and metadata.
    # --------------------------------------------------
    #

    forward_predictions = (
        forward_results[
            "predictions"
        ].tolist()
    )

    forward_labels = (
        forward_results[
            "labels"
        ].tolist()
    )

    forward_sample_ids = (
        forward_results[
            "sample_ids"
        ]
    )

    backward_predictions = (
        backward_results[
            "predictions"
        ].tolist()
    )

    backward_labels = (
        backward_results[
            "labels"
        ].tolist()
    )

    backward_sample_ids = (
        backward_results[
            "sample_ids"
        ]
    )

    #
    # --------------------------------------------------
    # Build lookup for forward model.
    # --------------------------------------------------
    #

    forward_lookup = {}

    for (
        prediction,
        label,
        sample_id

    ) in zip(

        forward_predictions,
        forward_labels,
        forward_sample_ids

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
    # Build lookup for backward model.
    # --------------------------------------------------
    #

    backward_lookup = {}

    for (
        prediction,
        label,
        sample_id

    ) in zip(

        backward_predictions,
        backward_labels,
        backward_sample_ids

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
    # Compare only samples that exist in both results.
    # --------------------------------------------------
    #

    forward_ids = set(
        forward_lookup.keys()
    )

    backward_ids = set(
        backward_lookup.keys()
    )

    common_ids = (
        forward_ids
        &
        backward_ids
    )

    forward_only_ids = (
        forward_ids
        -
        backward_ids
    )

    backward_only_ids = (
        backward_ids
        -
        forward_ids
    )

    #
    # --------------------------------------------------
    # Store sample-by-sample comparison.
    # --------------------------------------------------
    #

    comparisons = []

    outcome_counter = Counter()

    for sample_id in sorted(
        common_ids,
        key=str
    ):

        forward = (
            forward_lookup[
                sample_id
            ]
        )

        backward = (
            backward_lookup[
                sample_id
            ]
        )

        forward_prediction = (
            forward[
                "prediction"
            ]
        )

        backward_prediction = (
            backward[
                "prediction"
            ]
        )

        label = (
            forward[
                "label"
            ]
        )

        #
        # Sanity check.
        #
        if label != backward["label"]:

            comparison_type = (
                "LABEL_MISMATCH"
            )

        else:

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
            # Both models predict correctly.
            #
            if (
                forward_correct
                and
                backward_correct
            ):

                comparison_type = (
                    "BOTH_CORRECT"
                )

            #
            # Both models predict incorrectly.
            #
            elif (
                not forward_correct
                and
                not backward_correct
            ):

                comparison_type = (
                    "BOTH_WRONG"
                )

            #
            # Forward is correct,
            # backward is wrong.
            #
            elif (
                forward_correct
                and
                not backward_correct
            ):

                comparison_type = (
                    "FORWARD_CORRECT"
                )

            #
            # Backward is correct,
            # forward is wrong.
            #
            elif (
                not forward_correct
                and
                backward_correct
            ):

                comparison_type = (
                    "BACKWARD_CORRECT"
                )

            else:

                comparison_type = (
                    "UNKNOWN"
                )

        #
        # Did the two models predict
        # the same class?
        #
        same_prediction = (

            forward_prediction
            ==
            backward_prediction

        )

        comparisons.append({

            "sample_id":
                sample_id,

            "label":
                label,

            "forward_prediction":
                forward_prediction,

            "backward_prediction":
                backward_prediction,

            "forward_correct":
                forward_prediction
                ==
                label,

            "backward_correct":
                backward_prediction
                ==
                label,

            "same_prediction":
                same_prediction,

            "comparison":
                comparison_type

        })

        outcome_counter[
            comparison_type
        ] += 1

    #
    # --------------------------------------------------
    # Summary.
    # --------------------------------------------------
    #

    total_common = len(
        common_ids
    )

    different_predictions = sum(

        1

        for comparison in comparisons

        if not comparison[
            "same_prediction"
        ]

    )

    same_predictions = (
        total_common
        -
        different_predictions
    )

    return {

        "comparisons":
            comparisons,

        "summary": {

            "forward_samples":
                len(
                    forward_ids
                ),

            "backward_samples":
                len(
                    backward_ids
                ),

            "common_samples":
                total_common,

            "forward_only":
                len(
                    forward_only_ids
                ),

            "backward_only":
                len(
                    backward_only_ids
                ),

            "same_predictions":
                same_predictions,

            "different_predictions":
                different_predictions,

            "both_correct":
                outcome_counter[
                    "BOTH_CORRECT"
                ],

            "both_wrong":
                outcome_counter[
                    "BOTH_WRONG"
                ],

            "forward_correct":
                outcome_counter[
                    "FORWARD_CORRECT"
                ],

            "backward_correct":
                outcome_counter[
                    "BACKWARD_CORRECT"
                ],

            "label_mismatch":
                outcome_counter[
                    "LABEL_MISMATCH"
                ]

        }

    }


def print_forward_backward_predictions(
    results
):

    summary = (
        results[
            "summary"
        ]
    )

    comparisons = (
        results[
            "comparisons"
        ]
    )

    print()

    print(
        "=" * 80
    )

    print(
        "FORWARD VS BACKWARD MODEL PREDICTIONS"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Forward samples :",
        summary[
            "forward_samples"
        ]
    )

    print(
        "Backward samples:",
        summary[
            "backward_samples"
        ]
    )

    print(
        "Common samples  :",
        summary[
            "common_samples"
        ]
    )

    print()

    print(
        "Forward only:",
        summary[
            "forward_only"
        ]
    )

    print(
        "Backward only:",
        summary[
            "backward_only"
        ]
    )

    print()

    print(
        "-" * 60
    )

    print(
        "PREDICTION AGREEMENT"
    )

    print(
        "-" * 60
    )

    total = (
        summary[
            "common_samples"
        ]
    )

    if total > 0:

        same_ratio = (

            summary[
                "same_predictions"
            ]
            /
            total

        )

        different_ratio = (

            summary[
                "different_predictions"
            ]
            /
            total

        )

    else:

        same_ratio = 0.0
        different_ratio = 0.0

    print(
        f"Same predictions      : "
        f"{summary['same_predictions']} "
        f"({same_ratio:.2%})"
    )

    print(
        f"Different predictions : "
        f"{summary['different_predictions']} "
        f"({different_ratio:.2%})"
    )

    print()

    print(
        "-" * 60
    )

    print(
        "WHEN THE TWO DIRECTIONS DIFFER"
    )

    print(
        "-" * 60
    )

    print(
        "Both correct     :",
        summary[
            "both_correct"
        ]
    )

    print(
        "Both wrong       :",
        summary[
            "both_wrong"
        ]
    )

    print(
        "Forward correct  :",
        summary[
            "forward_correct"
        ]
    )

    print(
        "Backward correct :",
        summary[
            "backward_correct"
        ]
    )

    print(
        "Label mismatch   :",
        summary[
            "label_mismatch"
        ]
    )

    #
    # --------------------------------------------------
    # Show disagreement examples.
    # --------------------------------------------------
    #

    disagreements = [

        comparison

        for comparison in comparisons

        if not comparison[
            "same_prediction"
        ]

    ]

    print()

    print(
        "=" * 80
    )

    print(
        "PREDICTION DISAGREEMENTS"
    )

    print(
        "=" * 80
    )

    for comparison in disagreements:

        print()

        print(
            "Sample ID:"
        )

        print(
            comparison[
                "sample_id"
            ]
        )

        print(
            "Label:",
            comparison[
                "label"
            ]
        )

        print(
            "Forward prediction:",
            comparison[
                "forward_prediction"
            ]
        )

        print(
            "Backward prediction:",
            comparison[
                "backward_prediction"
            ]
        )

        print(
            "Result:",
            comparison[
                "comparison"
            ]
        )