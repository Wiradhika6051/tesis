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

            outcome[
                "sample_id"
            ]

        )

        outcome_lookup[
            sample_id
        ] = outcome

    #
    # --------------------------------------------------
    # Group similarity by classification outcome.
    # --------------------------------------------------
    #

    grouped = defaultdict(
        list
    )

    unmatched = []

    for result in similarity_results:

        #
        # Your paired similarity result should contain
        # vulnerable and fixed samples or their metadata.
        #
        pair_id = result.get(
            "pair_id"
        )

        if pair_id is None:

            #
            # Fallback if your current structure stores
            # repo/file/etc. separately.
            #
            pair_id = (

                result.get("repo"),

                result.get("parent_commit"),

                result.get("file")

            )

        #
        # Extract similarity.
        #
        similarity = result.get(
            "similarity"
        )

        if similarity is None:

            #
            # Adjust this depending on your existing
            # paired similarity structure.
            #
            similarity = result.get(
                "jaccard_similarity"
            )

        #
        # Vulnerable sample ID.
        #
        vulnerable_id = (

            pair_id[0],

            pair_id[1],

            pair_id[2],

            1

        )

        #
        # Fixed sample ID.
        #
        fixed_id = (

            pair_id[0],

            pair_id[1],

            pair_id[2],

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

                "similarity":
                    similarity,

                "label":
                    1,

                "prediction":
                    vulnerable_outcome[
                        "prediction"
                    ]

            })

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

                "similarity":
                    similarity,

                "label":
                    0,

                "prediction":
                    fixed_outcome[
                        "prediction"
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

    grouped = analysis[
        "grouped"
    ]

    unmatched = analysis[
        "unmatched"
    ]

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

        records = grouped.get(
            outcome,
            []
        )

        print()

        print(
            outcome
        )

        print(
            "-" * 40
        )

        print(

            "Samples:",

            len(records)

        )

        if not records:

            continue

        similarities = [

            r["similarity"]

            for r in records

            if r["similarity"] is not None

        ]

        if not similarities:

            print(
                "No similarity values."
            )

            continue

        print(

            f"Average similarity : "
            f"{statistics.mean(similarities):.4f}"

        )

        print(

            f"Median similarity  : "
            f"{statistics.median(similarities):.4f}"

        )

        print(

            f"Minimum similarity : "
            f"{min(similarities):.4f}"

        )

        print(

            f"Maximum similarity : "
            f"{max(similarities):.4f}"

        )

    print()

    print(
        "UNMATCHED RESULTS"
    )

    print(
        "-" * 40
    )

    print(
        len(unmatched)
    )