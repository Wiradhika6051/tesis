from collections import Counter
from collections import defaultdict

from src.analysis.paired_slice_similarity import normalize_text


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


def _node_signature(node):

    return (
        node.node_type,
        normalize_text(
            node.text
        )
    )


def _pruned_nodes(sample):

    if sample.pruned_cfg is None:
        return []

    return sample.pruned_cfg["nodes"]


def _pruned_signatures(sample):

    return {

        _node_signature(node)

        for node in _pruned_nodes(sample)

    }


def _node_type_counts(
    signatures
):

    return Counter(

        signature[0]

        for signature in signatures

    )


# ============================================================
# Main analysis
# ============================================================

def analyze_directional_context_composition(

    directional_results,

    forward_samples,

    backward_samples

):

    #
    # --------------------------------------------------------
    # Build sample lookup.
    # --------------------------------------------------------
    #

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
    # Aggregate node types separately for each outcome.
    #

    aggregate = {

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

    #
    # --------------------------------------------------------
    # Process prediction disagreements.
    # --------------------------------------------------------
    #

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

        if sample_id is None:
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

        #
        # Semantic node sets.
        #

        forward_nodes = _pruned_signatures(
            forward_sample
        )

        backward_nodes = _pruned_signatures(
            backward_sample
        )

        #
        # Direction-specific context.
        #

        forward_only = (

            forward_nodes
            -
            backward_nodes

        )

        backward_only = (

            backward_nodes
            -
            forward_nodes

        )

        overlap = (

            forward_nodes
            &
            backward_nodes

        )

        #
        # Node type composition.
        #

        forward_types = _node_type_counts(
            forward_only
        )

        backward_types = _node_type_counts(
            backward_only
        )

        #
        # Aggregate.
        #

        aggregate[
            outcome
        ][
            "forward_only"
        ].update(
            forward_types
        )

        aggregate[
            outcome
        ][
            "backward_only"
        ].update(
            backward_types
        )

        #
        # ----------------------------------------------------
        # Compare context ratios.
        # ----------------------------------------------------
        #

        overlap_size = len(
            overlap
        )

        forward_ratio = (

            len(forward_only)
            /
            overlap_size

            if overlap_size
            else float("inf")

        )

        backward_ratio = (

            len(backward_only)
            /
            overlap_size

            if overlap_size
            else float("inf")

        )

        #
        # ----------------------------------------------------
        # Store record.
        # ----------------------------------------------------
        #

        records.append({

            "sample_id":
                sample_id,

            "outcome":
                outcome,

            "forward_size":
                len(forward_nodes),

            "backward_size":
                len(backward_nodes),

            "overlap":
                overlap_size,

            "forward_only":
                len(forward_only),

            "backward_only":
                len(backward_only),

            "forward_context_ratio":
                forward_ratio,

            "backward_context_ratio":
                backward_ratio,

            "forward_only_types":
                dict(forward_types),

            "backward_only_types":
                dict(backward_types)

        })

    return {

        "records":
            records,

        "aggregate":
            aggregate

    }


# ============================================================
# Node type comparison
# ============================================================

def print_directional_context_composition(
    analysis
):

    aggregate = analysis[
        "aggregate"
    ]

    print()

    print(
        "=" * 100
    )

    print(
        "DIRECTIONAL-ONLY CONTEXT COMPOSITION"
    )

    print(
        "=" * 100
    )

    for outcome in (

        "FORWARD_CORRECT",

        "BACKWARD_CORRECT"

    ):

        print()

        print(
            f"Outcome: {outcome}"
        )

        print(
            "-" * 60
        )

        records = [

            r

            for r in analysis["records"]

            if r["outcome"] == outcome

        ]

        print(
            "Samples:",
            len(records)
        )

        #
        # ----------------------------------------------------
        # Forward-only.
        # ----------------------------------------------------
        #

        forward_counts = aggregate[
            outcome
        ][
            "forward_only"
        ]

        backward_counts = aggregate[
            outcome
        ][
            "backward_only"
        ]

        print()

        print(
            "FORWARD-ONLY NODE TYPES"
        )

        print(
            "-" * 60
        )

        _print_counter(
            forward_counts
        )

        print()

        print(
            "BACKWARD-ONLY NODE TYPES"
        )

        print(
            "-" * 60
        )

        _print_counter(
            backward_counts
        )


def _print_counter(
    counter
):

    total = sum(
        counter.values()
    )

    if total == 0:

        print(
            "No directional-only nodes."
        )

        return

    for node_type, count in counter.most_common():

        percentage = (

            count
            /
            total
            *
            100

        )

        print(

            f"{node_type:<25}"
            f"{count:>8}"
            f"{percentage:>10.2f}%"

        )


# ============================================================
# Winner vs loser context
# ============================================================

def print_winner_loser_context_types(
    analysis
):

    records = analysis[
        "records"
    ]

    print()

    print(
        "=" * 100
    )

    print(
        "WINNER VS LOSER CONTEXT COMPOSITION"
    )

    print(
        "=" * 100
    )

    for outcome in (

        "FORWARD_CORRECT",

        "BACKWARD_CORRECT"

    ):

        print()

        print(
            outcome
        )

        print(
            "-" * 60
        )

        subset = [

            r

            for r in records

            if r["outcome"] == outcome

        ]

        if not subset:

            print(
                "No samples."
            )

            continue

        winner_counter = Counter()
        loser_counter = Counter()

        for r in subset:

            if outcome == "FORWARD_CORRECT":

                winner_counter.update(

                    r[
                        "forward_only_types"
                    ]

                )

                loser_counter.update(

                    r[
                        "backward_only_types"
                    ]

                )

            else:

                winner_counter.update(

                    r[
                        "backward_only_types"
                    ]

                )

                loser_counter.update(

                    r[
                        "forward_only_types"
                    ]

                )

        print()

        print(
            "WINNING DIRECTION"
        )

        print(
            "-" * 40
        )

        _print_counter(
            winner_counter
        )

        print()

        print(
            "LOSING DIRECTION"
        )

        print(
            "-" * 40
        )

        _print_counter(
            loser_counter
        )


# ============================================================
# Per-sample suspicious context
# ============================================================

def print_extreme_context_composition(
    analysis,
    limit=10
):

    records = analysis[
        "records"
    ]

    #
    # Find samples where directional context differs
    # substantially.
    #

    records = sorted(

        records,

        key=lambda r:
            abs(
                r["forward_only"]
                -
                r["backward_only"]
            ),

        reverse=True

    )

    print()

    print(
        "=" * 100
    )

    print(
        "EXTREME DIRECTIONAL CONTEXT COMPOSITION"
    )

    print(
        "=" * 100
    )

    for r in records[:limit]:

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

        print()

        print(
            "Forward-only types:"
        )

        if r["forward_only_types"]:

            for node_type, count in sorted(

                r["forward_only_types"].items(),

                key=lambda x:
                    x[1],

                reverse=True

            ):

                print(
                    f"  {node_type}: {count}"
                )

        else:

            print(
                "  None"
            )

        print()

        print(
            "Backward-only types:"
        )

        if r["backward_only_types"]:

            for node_type, count in sorted(

                r["backward_only_types"].items(),

                key=lambda x:
                    x[1],

                reverse=True

            ):

                print(
                    f"  {node_type}: {count}"
                )

        else:

            print(
                "  None"
            )


# ============================================================
# Node category analysis
# ============================================================

CONTROL_FLOW_TYPES = {

    "If",
    "For",
    "While",
    "Try",
    "With",
    "AsyncFor",
    "Break",
    "Continue",
    "Raise",
    "Return"
}


DATA_OPERATION_TYPES = {

    "Assign",
    "AnnAssign",
    "AugAssign",
    "Expr",
    "NamedExpr"
}


STRUCTURAL_TYPES = {

    "FunctionDef",
    "AsyncFunctionDef",
    "ClassDef",
    "ENTRY",
    "EXIT",
    "MERGE",
    "LOOP_EXIT"
}


def _categorize_node_type(
    node_type
):

    if node_type in CONTROL_FLOW_TYPES:

        return "CONTROL_FLOW"

    if node_type in DATA_OPERATION_TYPES:

        return "DATA_OPERATION"

    if node_type in STRUCTURAL_TYPES:

        return "STRUCTURAL"

    return "OTHER"


def print_context_categories(
    analysis
):

    records = analysis[
        "records"
    ]

    print()

    print(
        "=" * 100
    )

    print(
        "DIRECTIONAL CONTEXT CATEGORIES"
    )

    print(
        "=" * 100
    )

    for outcome in (

        "FORWARD_CORRECT",

        "BACKWARD_CORRECT"

    ):

        subset = [

            r

            for r in records

            if r["outcome"] == outcome

        ]

        forward_categories = Counter()
        backward_categories = Counter()

        for r in subset:

            for node_type, count in r[
                "forward_only_types"
            ].items():

                category = _categorize_node_type(
                    node_type
                )

                forward_categories[
                    category
                ] += count

            for node_type, count in r[
                "backward_only_types"
            ].items():

                category = _categorize_node_type(
                    node_type
                )

                backward_categories[
                    category
                ] += count

        print()

        print(
            outcome
        )

        print(
            "-" * 60
        )

        print(
            "Forward-only categories"
        )

        _print_counter(
            forward_categories
        )

        print()

        print(
            "Backward-only categories"
        )

        _print_counter(
            backward_categories
        )