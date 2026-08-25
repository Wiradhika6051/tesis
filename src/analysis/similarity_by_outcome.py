from collections import defaultdict

import statistics


def analyze_similarity_by_outcome(

    similarity_results,

    model_outcomes

):

    #
    # --------------------------------------------------
    # Build outcome lookup.
    # --------------------------------------------------
    #

    outcome_lookup = {}

    for outcome in model_outcomes:

        sample_id = (
            outcome["sample_id"]
        )

        outcome_lookup[
            sample_id
        ] = outcome

    #
    # --------------------------------------------------
    # Get paired similarity records.
    # --------------------------------------------------
    #

    pair_results = (
        similarity_results[
            "pairs"
        ]
    )

    grouped = defaultdict(
        list
    )

    unmatched = []

    #
    # --------------------------------------------------
    # Match each vulnerable/fixed pair
    # with model outcomes.
    # --------------------------------------------------
    #

    for result in pair_results:

        pair_id = (

            result["repo"],

            result["parent_commit"],

            result["file"]

        )

        #
        # Sample IDs must match the IDs produced
        # by your evaluator / dataset.
        #
        vulnerable_id = (

            result["repo"],

            result["parent_commit"],

            result["file"],

            1

        )

        fixed_id = (

            result["repo"],

            result["parent_commit"],

            result["file"],

            0

        )

        vulnerable_outcome = (
            outcome_lookup.get(
                vulnerable_id
            )
        )

        fixed_outcome = (
            outcome_lookup.get(
                fixed_id
            )
        )

        #
        # --------------------------------------------------
        # Record vulnerable sample.
        # --------------------------------------------------
        #

        if vulnerable_outcome is None:

            unmatched.append({

                "pair_id":
                    pair_id,

                "version":
                    "vulnerable"

            })

        else:

            grouped[
                vulnerable_outcome[
                    "outcome"
                ]
            ].append({

                "pair_id":
                    pair_id,

                "version":
                    "vulnerable",

                "label":
                    1,

                "prediction":
                    vulnerable_outcome[
                        "prediction"
                    ],

                "forward_similarity":
                    result[
                        "forward_similarity"
                    ],

                "backward_similarity":
                    result[
                        "backward_similarity"
                    ],

                "similarity_gap":
                    result[
                        "similarity_gap"
                    ]

            })

        #
        # --------------------------------------------------
        # Record fixed sample.
        # --------------------------------------------------
        #

        if fixed_outcome is None:

            unmatched.append({

                "pair_id":
                    pair_id,

                "version":
                    "fixed"

            })

        else:

            grouped[
                fixed_outcome[
                    "outcome"
                ]
            ].append({

                "pair_id":
                    pair_id,

                "version":
                    "fixed",

                "label":
                    0,

                "prediction":
                    fixed_outcome[
                        "prediction"
                    ],

                "forward_similarity":
                    result[
                        "forward_similarity"
                    ],

                "backward_similarity":
                    result[
                        "backward_similarity"
                    ],

                "similarity_gap":
                    result[
                        "similarity_gap"
                    ]

            })

    return {

        "grouped":
            grouped,

        "unmatched":
            unmatched

    }

def print_similarity_by_outcome(

    analysis

):

    grouped = (
        analysis[
            "grouped"
        ]
    )

    unmatched = (
        analysis[
            "unmatched"
        ]
    )

    print()

    print(
        "=" * 80
    )

    print(
        "PAIRED SLICE SIMILARITY BY MODEL OUTCOME"
    )

    print(
        "=" * 80
    )

    outcomes = [

        "TP",

        "TN",

        "FP",

        "FN"

    ]

    for outcome in outcomes:

        records = (
            grouped.get(
                outcome,
                []
            )
        )

        print()

        print(
            outcome
        )

        print(
            "-" * 50
        )

        print(
            "Samples:",
            len(records)
        )

        if not records:

            continue

        forward_values = [

            record[
                "forward_similarity"
            ]

            for record in records

        ]

        backward_values = [

            record[
                "backward_similarity"
            ]

            for record in records

        ]

        gap_values = [

            record[
                "similarity_gap"
            ]

            for record in records

        ]

        print()

        print(
            "Forward similarity"
        )

        print(
            f"Average : "
            f"{statistics.mean(forward_values):.4f}"
        )

        print(
            f"Median  : "
            f"{statistics.median(forward_values):.4f}"
        )

        print()

        print(
            "Backward similarity"
        )

        print(
            f"Average : "
            f"{statistics.mean(backward_values):.4f}"
        )

        print(
            f"Median  : "
            f"{statistics.median(backward_values):.4f}"
        )

        print()

        print(
            "Similarity gap"
        )

        print(
            f"Average : "
            f"{statistics.mean(gap_values):.4f}"
        )

        print(
            f"Median  : "
            f"{statistics.median(gap_values):.4f}"
        )

        print(
            f"Minimum : "
            f"{min(gap_values):.4f}"
        )

        print(
            f"Maximum : "
            f"{max(gap_values):.4f}"
        )

    print()

    print(
        "=" * 80
    )

    print(
        "UNMATCHED SAMPLE RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        len(unmatched)
    )

    for item in unmatched[:20]:

        print(

            item

        )