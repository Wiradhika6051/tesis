from collections import Counter, defaultdict
import statistics


# ============================================================
# Helpers
# ============================================================

def _sample_id(sample):

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def _normalize_text(text):

    if text is None:
        return ""

    return " ".join(
        text.strip().split()
    )


def _node_signature(node):

    return (
        node.node_type,
        _normalize_text(node.text)
    )


def _get_slice_signatures(
    sample,
    forward
):

    if sample.cfg is None:
        return set()

    if not sample.seed_nodes:
        return set()

    if not sample.function_nodes:
        return set()

    from src.analysis.slice_analysis import (
        get_slice_nodes
    )

    node_ids = get_slice_nodes(

        sample.cfg,

        set(sample.seed_nodes),

        set(sample.function_nodes),

        forward=forward

    )

    node_lookup = {

        node.node_id: node
        for node in sample.cfg["nodes"]

    }

    return {

        _node_signature(
            node_lookup[node_id]
        )

        for node_id in node_ids

        if node_id in node_lookup

    }


def _get_directional_type_counts(
    forward_sample,
    backward_sample
):

    forward = _get_slice_signatures(
        forward_sample,
        forward=True
    )

    backward = _get_slice_signatures(
        backward_sample,
        forward=False
    )

    forward_only = (
        forward - backward
    )

    backward_only = (
        backward - forward
    )

    forward_types = Counter(

        node_type

        for node_type, _ in forward_only

    )

    backward_types = Counter(

        node_type

        for node_type, _ in backward_only

    )

    return (
        forward_types,
        backward_types
    )


# ============================================================
# Main analysis
# ============================================================

def analyze_winner_node_type_enrichment(
    directional_results,
    forward_samples,
    backward_samples
):

    sample_counts = Counter()
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

    winner_counts = defaultdict(
        Counter
    )

    loser_counts = defaultdict(
        Counter
    )

    sample_winner_counts = defaultdict(
        Counter
    )

    sample_loser_counts = defaultdict(
        Counter
    )

    processed = set()

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

        if sample_id in processed:
            continue

        forward_sample = forward_lookup.get(
            sample_id
        )

        backward_sample = backward_lookup.get(
            sample_id
        )

        if (
            forward_sample is None
            or
            backward_sample is None
        ):
            continue

        (
            forward_types,
            backward_types
        ) = _get_directional_type_counts(

            forward_sample,

            backward_sample

        )

        if outcome == "FORWARD_CORRECT":

            winner = forward_types
            loser = backward_types

        else:

            winner = backward_types
            loser = forward_types

        winner_counts[outcome].update(
            winner
        )

        loser_counts[outcome].update(
            loser
        )
        sample_counts[outcome] += 1
        #
        # Per-sample enrichment.
        #
        node_types = set(
            winner
        ) | set(
            loser
        )

        for node_type in node_types:

            winner_value = winner.get(
                node_type,
                0
            )

            loser_value = loser.get(
                node_type,
                0
            )

            if winner_value > loser_value:

                sample_winner_counts[
                    outcome
                ][
                    node_type
                ] += 1

            elif loser_value > winner_value:

                sample_loser_counts[
                    outcome
                ][
                    node_type
                ] += 1

        processed.add(
            sample_id
        )

    return {

        "winner_counts":
            dict(winner_counts),

        "loser_counts":
            dict(loser_counts),

        "sample_winner_counts":
            dict(sample_winner_counts),

        "sample_loser_counts":
            dict(sample_loser_counts),
        "sample_counts": dict(sample_counts)

    }


# ============================================================
# Printing
# ============================================================

def _print_table(
    winner_counts,
    loser_counts,
    sample_winner_counts,
    sample_loser_counts,
    sample_count
):

    node_types = (

        set(winner_counts)
        |
        set(loser_counts)

    )

    rows = []

    for node_type in node_types:

        winner_total = winner_counts.get(
            node_type,
            0
        )

        loser_total = loser_counts.get(
            node_type,
            0
        )

        winner_samples = sample_winner_counts.get(
            node_type,
            0
        )

        loser_samples = sample_loser_counts.get(
            node_type,
            0
        )

        winner_frequency = (

            winner_samples
            / sample_count
            * 100

            if sample_count
            else 0

        )

        loser_frequency = (

            loser_samples
            / sample_count
            * 100

            if sample_count
            else 0

        )

        difference = (

            winner_frequency
            -
            loser_frequency

        )

        rows.append(

            (
                node_type,
                winner_total,
                loser_total,
                winner_frequency,
                loser_frequency,
                difference

            )

        )

    rows.sort(

        key=lambda row:
            abs(row[5]),

        reverse=True

    )

    print(
        f"{'Node type':<25}"
        f"{'Winner':>10}"
        f"{'Loser':>10}"
        f"{'Winner %':>12}"
        f"{'Loser %':>12}"
        f"{'Diff':>12}"
    )

    print(
        "-" * 81
    )

    for row in rows:

        print(

            f"{row[0]:<25}"
            f"{row[1]:>10}"
            f"{row[2]:>10}"
            f"{row[3]:>11.2f}%"
            f"{row[4]:>11.2f}%"
            f"{row[5]:>11.2f}%"

        )


def print_winner_node_type_enrichment(
    results
):

    print()
    print(
        "=" * 100
    )
    print(
        "WINNER VS LOSER NODE-TYPE ENRICHMENT"
    )
    print(
        "=" * 100
    )

    for outcome in (
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT"
    ):

        winner_counts = Counter(
            results[
                "winner_counts"
            ].get(
                outcome,
                {}
            )
        )

        loser_counts = Counter(
            results[
                "loser_counts"
            ].get(
                outcome,
                {}
            )
        )

        sample_winner_counts = Counter(
            results[
                "sample_winner_counts"
            ].get(
                outcome,
                {}
            )
        )

        sample_loser_counts = Counter(
            results[
                "sample_loser_counts"
            ].get(
                outcome,
                {}
            )
        )
        sample_count = results[
            "sample_counts"
        ].get(
            outcome,
            0
        )

        print()
        print(
            outcome
        )

        print(
            "-" * 100
        )

        print(
            "Winner = correctly predicting direction"
        )

        print()

        _print_table(

            winner_counts,

            loser_counts,

            sample_winner_counts,

            sample_loser_counts,

            sample_count

        )