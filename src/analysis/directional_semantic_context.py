from collections import Counter, defaultdict

from src.analysis.paired_slice_similarity import (
    get_slice_signatures,
    normalize_text
)


# ============================================================
# Helpers
# ============================================================

def _get_sample_id(sample):
    """
    Sample ID used to match directional results.
    """

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def _signature_to_text(signature):
    """
    Convert:

        (node_type, normalized_text)

    into readable text.
    """

    node_type, text = signature

    return f"{node_type}: {text}"


def _get_pruned_signatures(
    sample
):

    if sample.pruned_cfg is None:
        return set()

    signatures = set()

    for node in sample.pruned_cfg["nodes"]:

        signatures.add(

            (
                node.node_type,
                normalize_text(
                    node.text
                )
            )

        )

    return signatures

# ============================================================
# Build lookup
# ============================================================

def _build_sample_lookup(
    forward_samples,
    backward_samples
):

    forward_lookup = {

        _get_sample_id(sample):
            sample

        for sample in forward_samples

    }

    backward_lookup = {

        _get_sample_id(sample):
            sample

        for sample in backward_samples

    }

    return (
        forward_lookup,
        backward_lookup
    )


# ============================================================
# Main analysis
# ============================================================

def analyze_directional_semantic_context(

    directional_results,

    forward_samples,

    backward_samples

):

    (
        forward_lookup,
        backward_lookup
    ) = _build_sample_lookup(

        forward_samples,
        backward_samples

    )

    grouped = {

        "BOTH_CORRECT": {

            "forward_only":
                Counter(),

            "backward_only":
                Counter()

        },

        "BOTH_WRONG": {

            "forward_only":
                Counter(),

            "backward_only":
                Counter()

        },

        "FORWARD_CORRECT": {

            "forward_only":
                Counter(),

            "backward_only":
                Counter()

        },

        "BACKWARD_CORRECT": {

            "forward_only":
                Counter(),

            "backward_only":
                Counter()

        }

    }

    sample_counts = defaultdict(int)

    unmatched = []

    #
    # --------------------------------------------------
    # Analyze every directional result.
    # --------------------------------------------------
    #

    for result in directional_results:

        sample_id = result.get(
            "sample_id"
        )

        outcome = result.get(
            "outcome"
        )

        if outcome not in grouped:
            continue

        forward_sample = (
            forward_lookup.get(
                sample_id
            )
        )

        backward_sample = (
            backward_lookup.get(
                sample_id
            )
        )

        if (
            forward_sample is None
            or
            backward_sample is None
        ):

            unmatched.append(
                sample_id
            )

            continue

        #
        # IMPORTANT:
        #
        # We need the directional representation
        # from the corresponding directional sample.
        #
        forward_context = _get_pruned_signatures(
            forward_sample
        )

        backward_context = _get_pruned_signatures(
            backward_sample
        )

        #
        # Forward model's forward-only context.
        #
        # Backward model's backward-only context.
        #
        #
        # Since each sample has its own pruned CFG,
        # we analyze the actual node signatures in
        # each directional representation.
        #

        grouped[
            outcome
        ][
            "forward_only"
        ].update(

            forward_context[
                "forward_only"
            ]

        )

        grouped[
            outcome
        ][
            "backward_only"
        ].update(

            backward_context[
                "backward_only"
            ]

        )

        sample_counts[
            outcome
        ] += 1

    return {

        "grouped":
            grouped,

        "sample_counts":
            dict(sample_counts),

        "unmatched":
            unmatched

    }


# ============================================================
# Print semantic context
# ============================================================

def print_directional_semantic_context(

    analysis,

    top_n=20

):

    grouped = analysis[
        "grouped"
    ]

    sample_counts = analysis[
        "sample_counts"
    ]

    print()

    print(
        "=" * 100
    )

    print(
        "DIRECTIONAL SEMANTIC CONTEXT"
    )

    print(
        "=" * 100
    )

    outcomes = [

        "BOTH_CORRECT",

        "BOTH_WRONG",

        "FORWARD_CORRECT",

        "BACKWARD_CORRECT"

    ]

    for outcome in outcomes:

        print()

        print(
            outcome
        )

        print(
            "-" * 60
        )

        print(
            "Samples:",
            sample_counts.get(
                outcome,
                0
            )
        )

        #
        # Forward-only.
        #

        print()

        print(
            "FORWARD-ONLY NODE SIGNATURES"
        )

        print(
            "-" * 60
        )

        counter = grouped[
            outcome
        ][
            "forward_only"
        ]

        if not counter:

            print(
                "None"
            )

        else:

            for signature, count in counter.most_common(
                top_n
            ):

                print(

                    f"{count:5d}  "
                    f"{_signature_to_text(signature)}"

                )

        #
        # Backward-only.
        #

        print()

        print(
            "BACKWARD-ONLY NODE SIGNATURES"
        )

        print(
            "-" * 60
        )

        counter = grouped[
            outcome
        ][
            "backward_only"
        ]

        if not counter:

            print(
                "None"
            )

        else:

            for signature, count in counter.most_common(
                top_n
            ):

                print(

                    f"{count:5d}  "
                    f"{_signature_to_text(signature)}"

                )


# ============================================================
# Compare winning directions
# ============================================================

def compare_directional_semantic_signatures(
    analysis,
    top_n=30
):

    grouped = analysis[
        "grouped"
    ]

    forward_counter = Counter()

    backward_counter = Counter()

    #
    # Forward-correct:
    #
    # forward-only signatures are evidence available
    # to the forward direction when it wins.
    #
    forward_counter.update(

        grouped[
            "FORWARD_CORRECT"
        ][
            "forward_only"
        ]

    )

    #
    # Backward-correct:
    #
    # backward-only signatures are evidence available
    # to the backward direction when it wins.
    #
    backward_counter.update(

        grouped[
            "BACKWARD_CORRECT"
        ][
            "backward_only"
        ]

    )

    all_signatures = (

        set(
            forward_counter
        )
        |
        set(
            backward_counter
        )

    )

    comparison = []

    for signature in all_signatures:

        forward_count = (
            forward_counter.get(
                signature,
                0
            )
        )

        backward_count = (
            backward_counter.get(
                signature,
                0
            )
        )

        comparison.append({

            "signature":
                signature,

            "forward_correct_count":
                forward_count,

            "backward_correct_count":
                backward_count,

            "difference":
                forward_count
                -
                backward_count

        })

    comparison.sort(

        key=lambda x:
            abs(
                x["difference"]
            ),

        reverse=True

    )

    print()

    print(
        "=" * 100
    )

    print(
        "DIRECTIONAL SEMANTIC SIGNATURE COMPARISON"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"{'Node signature':60}"
        f"{'Forward':>10}"
        f"{'Backward':>10}"
        f"{'Difference':>12}"
    )

    print(
        "-" * 94
    )

    for item in comparison[
        :top_n
    ]:

        print(

            f"{_signature_to_text(item['signature'])[:60]:60}"
            f"{item['forward_correct_count']:10d}"
            f"{item['backward_correct_count']:10d}"
            f"{item['difference']:12d}"

        )

    return comparison