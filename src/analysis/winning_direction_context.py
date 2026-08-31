from collections import defaultdict
import statistics

from src.analysis.paired_slice_similarity import normalize_text



def _sample_id(sample):

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def _node_signature(node):

    return (
        node.node_type,
        normalize_text(
            node.text
        )
    )


def _pruned_signatures(sample):

    if sample.pruned_cfg is None:
        return set()

    return {

        _node_signature(node)

        for node in sample.pruned_cfg["nodes"]

    }


def analyze_winning_direction_context(

    directional_results,

    forward_samples,

    backward_samples

):

    #
    # --------------------------------------------------------
    # Build sample lookup.
    # --------------------------------------------------------

    forward_lookup = {

        _sample_id(sample):
            sample

        for sample in forward_samples

    }

    backward_lookup = {

        _sample_id(sample):
            sample

        for sample in backward_samples

    }

    records = []

    #
    # --------------------------------------------------------
    # Only analyze prediction disagreements.
    # --------------------------------------------------------

    for result in directional_results:

        outcome = result.get(
            "outcome"
        )

        if outcome not in (

            "FORWARD_CORRECT",

            "BACKWARD_CORRECT"

        ):

            continue

        sample_id = result.get(
            "sample_id"
        )

        forward_sample = forward_lookup.get(
            sample_id
        )

        backward_sample = backward_lookup.get(
            sample_id
        )

        #
        # Both directions must have this SAME sample.
        #

        if (
            forward_sample is None
            or
            backward_sample is None
        ):

            continue

        #
        # Get semantic slices.
        #

        forward_slice = _pruned_signatures(
            forward_sample
        )

        backward_slice = _pruned_signatures(
            backward_sample
        )

        #
        # Direction-specific context.
        #

        forward_only = (

            forward_slice
            -
            backward_slice

        )

        backward_only = (

            backward_slice
            -
            forward_slice

        )

        overlap = (

            forward_slice
            &
            backward_slice

        )

        #
        # Determine winner.
        #

        if outcome == "FORWARD_CORRECT":

            winner = "FORWARD"

            winner_context = forward_only

            loser_context = backward_only

        else:

            winner = "BACKWARD"

            winner_context = backward_only

            loser_context = forward_only

        #
        # ----------------------------------------------------
        # Record.
        # ----------------------------------------------------

        records.append({

            "sample_id":
                sample_id,

            "outcome":
                outcome,

            "winner":
                winner,

            #
            # Predictions.
            #

            "forward_prediction":
                result.get(
                    "forward_prediction"
                ),

            "backward_prediction":
                result.get(
                    "backward_prediction"
                ),

            #
            # Slice sizes.
            #

            "forward_size":
                len(
                    forward_slice
                ),

            "backward_size":
                len(
                    backward_slice
                ),

            "overlap":
                len(
                    overlap
                ),

            #
            # Directional context.
            #

            "forward_only":
                len(
                    forward_only
                ),

            "backward_only":
                len(
                    backward_only
                ),

            #
            # Winner context.
            #

            "winner_context":
                len(
                    winner_context
                ),

            "loser_context":
                len(
                    loser_context
                ),

            "winner_advantage":
                (
                    len(winner_context)
                    -
                    len(loser_context)
                ),

            #
            # Context shares.
            #

            "forward_context_share":
                (
                    len(forward_only)
                    /
                    len(forward_slice)
                    if forward_slice
                    else 0.0
                ),

            "backward_context_share":
                (
                    len(backward_only)
                    /
                    len(backward_slice)
                    if backward_slice
                    else 0.0
                ),

            #
            # Winner context ratio.
            #

            "winner_context_ratio":
                (
                    len(winner_context)
                    /
                    len(loser_context)
                    if loser_context
                    else float("inf")
                )

        })

    return records


def print_winning_direction_context(
    records
):

    print()

    print(
        "=" * 100
    )

    print(
        "WINNING DIRECTION VS LOSING DIRECTION CONTEXT"
    )

    print(
        "=" * 100
    )

    #
    # --------------------------------------------------------
    # Split outcomes.
    # --------------------------------------------------------

    forward_wins = [

        r

        for r in records

        if r["outcome"]
        ==
        "FORWARD_CORRECT"

    ]

    backward_wins = [

        r

        for r in records

        if r["outcome"]
        ==
        "BACKWARD_CORRECT"

    ]

    #
    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    print()

    print(
        f"{'Metric':<40}"
        f"{'Forward correct':>20}"
        f"{'Backward correct':>20}"
    )

    print(
        "-" * 80
    )

    metrics = [

        (
            "Samples",
            "count"
        ),

        (
            "Forward slice",
            "forward_size"
        ),

        (
            "Backward slice",
            "backward_size"
        ),

        (
            "Overlap",
            "overlap"
        ),

        (
            "Forward-only",
            "forward_only"
        ),

        (
            "Backward-only",
            "backward_only"
        ),

        (
            "Winner context",
            "winner_context"
        ),

        (
            "Loser context",
            "loser_context"
        ),

        (
            "Winner advantage",
            "winner_advantage"
        ),

        (
            "Winner context ratio",
            "winner_context_ratio"
        )

    ]

    for label, key in metrics:

        if key == "count":

            forward_value = len(
                forward_wins
            )

            backward_value = len(
                backward_wins
            )

        else:

            forward_values = [

                r[key]

                for r in forward_wins

            ]

            backward_values = [

                r[key]

                for r in backward_wins

            ]

            forward_value = (

                statistics.mean(
                    forward_values
                )

                if forward_values
                else 0.0

            )

            backward_value = (

                statistics.mean(
                    backward_values
                )

                if backward_values
                else 0.0

            )

        if key == "winner_context_ratio":

            def fmt(value):

                if value == float("inf"):

                    return "∞"

                return f"{value:.2f}"

            print(

                f"{label:<40}"
                f"{fmt(forward_value):>20}"
                f"{fmt(backward_value):>20}"

            )

        else:

            print(

                f"{label:<40}"
                f"{forward_value:>20.2f}"
                f"{backward_value:>20.2f}"

            )

    #
    # --------------------------------------------------------
    # Compactness test.
    # --------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "WINNING DIRECTION VS CONTEXT SIZE"
    )

    print(
        "=" * 100
    )

    for name, subset in [

        (
            "FORWARD_CORRECT",
            forward_wins
        ),

        (
            "BACKWARD_CORRECT",
            backward_wins
        )

    ]:

        smaller = 0
        larger = 0
        equal = 0

        for r in subset:

            if (
                r["winner_context"]
                <
                r["loser_context"]
            ):

                smaller += 1

            elif (
                r["winner_context"]
                >
                r["loser_context"]
            ):

                larger += 1

            else:

                equal += 1

        print()

        print(name)

        print(
            "-" * 60
        )

        print(
            "Samples:",
            len(subset)
        )

        print(
            "Winner context smaller:",
            smaller
        )

        print(
            "Winner context larger:",
            larger
        )

        print(
            "Equal:",
            equal
        )

    #
    # --------------------------------------------------------
    # Per-sample results.
    # --------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "PER-SAMPLE WINNING CONTEXT"
    )

    print(
        "=" * 100
    )

    for r in records:

        print()

        print(
            "Outcome:",
            r["outcome"]
        )

        print(
            "Sample:",
            r["sample_id"]
        )

        print(
            "Forward slice:",
            r["forward_size"]
        )

        print(
            "Backward slice:",
            r["backward_size"]
        )

        print(
            "Overlap:",
            r["overlap"]
        )

        print(
            "Forward-only:",
            r["forward_only"]
        )

        print(
            "Backward-only:",
            r["backward_only"]
        )

        print(
            "Winner:",
            r["winner"]
        )

        print(
            "Winner context:",
            r["winner_context"]
        )

        print(
            "Loser context:",
            r["loser_context"]
        )

        print(
            "Winner advantage:",
            r["winner_advantage"]
        )