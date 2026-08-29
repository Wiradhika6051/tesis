from collections import Counter, defaultdict


from src.analysis.slice_analysis import get_slice_nodes


def get_sample_id(sample):

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def build_sample_lookup(samples):

    lookup = {}

    for sample in samples:

        if sample.graph is None:
            continue

        lookup[
            sample.graph.sample_id
        ] = sample

    return lookup


def get_slice_sets(sample):

    seed_nodes = set(
        sample.seed_nodes
    )

    function_nodes = set(
        sample.function_nodes
    )

    forward_nodes = get_slice_nodes(

        sample.cfg,

        seed_nodes,

        function_nodes,

        forward=True

    )

    backward_nodes = get_slice_nodes(

        sample.cfg,

        seed_nodes,

        function_nodes,

        forward=False

    )

    return (

        seed_nodes,

        forward_nodes,

        backward_nodes

    )


def classify_disagreement(
    comparison
):

    label = comparison[
        "label"
    ]

    forward_prediction = comparison[
        "forward_prediction"
    ]

    backward_prediction = comparison[
        "backward_prediction"
    ]

    forward_correct = (
        forward_prediction == label
    )

    backward_correct = (
        backward_prediction == label
    )

    if (
        forward_correct
        and
        not backward_correct
    ):

        return "FORWARD_CORRECT"

    if (
        backward_correct
        and
        not forward_correct
    ):

        return "BACKWARD_CORRECT"

    return None


def analyze_disagreement_slices(
    comparison_results,
    forward_samples,
    backward_samples
):

    forward_lookup = build_sample_lookup(
        forward_samples
    )

    backward_lookup = build_sample_lookup(
        backward_samples
    )

    records = []

    comparisons = comparison_results[
        "comparisons"
    ]

    for comparison in comparisons:

        outcome = classify_disagreement(
            comparison
        )

        #
        # We only care about cases where
        # the directions disagree in correctness.
        #
        if outcome is None:
            continue

        sample_id = comparison[
            "sample_id"
        ]

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
            continue

        (
            forward_seeds,
            forward_nodes,
            forward_backward_nodes
        ) = get_slice_sets(
            forward_sample
        )

        (
            backward_seeds,
            backward_forward_nodes,
            backward_nodes
        ) = get_slice_sets(
            backward_sample
        )

        #
        # The two sets should contain the same
        # conceptual directional slices.
        #
        forward_slice = forward_nodes

        backward_slice = backward_nodes

        overlap = (
            forward_slice
            &
            backward_slice
        )

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

        records.append({

            "sample_id":
                sample_id,

            "label":
                comparison["label"],

            "forward_prediction":
                comparison[
                    "forward_prediction"
                ],

            "backward_prediction":
                comparison[
                    "backward_prediction"
                ],

            "outcome":
                outcome,

            "seed_lines":
                list(
                    forward_sample.seed_lines
                ),

            "forward_seeds":
                forward_seeds,

            "backward_seeds":
                backward_seeds,

            "forward_nodes":
                forward_slice,

            "backward_nodes":
                backward_slice,

            "overlap":
                overlap,

            "forward_only":
                forward_only,

            "backward_only":
                backward_only

        })

    return records

def node_type_distribution(
    sample,
    node_ids
):

    lookup = {

        node.node_id:
            node

        for node in sample.cfg["nodes"]

    }

    counter = Counter()

    for node_id in node_ids:

        node = lookup.get(
            node_id
        )

        if node is None:
            continue

        counter[
            node.node_type
        ] += 1

    return counter

def positional_distribution(
    sample,
    node_ids
):

    lookup = {

        node.node_id:
            node

        for node in sample.cfg["nodes"]

    }

    seed_lines = [

        line
        for line in sample.seed_lines
        if line is not None

    ]

    before = 0
    after = 0
    same = 0
    unknown = 0

    for node_id in node_ids:

        node = lookup.get(
            node_id
        )

        if node is None:
            unknown += 1
            continue

        if node.lineno < 0:
            unknown += 1
            continue

        if not seed_lines:
            unknown += 1
            continue

        #
        # Distance to closest changed line.
        #
        distance = min(

            node.lineno - seed_line
            for seed_line in seed_lines

        )

        if distance < 0:

            before += 1

        elif distance > 0:

            after += 1

        else:

            same += 1

    return {

        "before":
            before,

        "after":
            after,

        "same":
            same,

        "unknown":
            unknown

    }

def print_disagreement_slice_analysis(
    records,
    forward_samples,
    backward_samples
):

    forward_lookup = build_sample_lookup(
        forward_samples
    )

    backward_lookup = build_sample_lookup(
        backward_samples
    )

    print()
    print("=" * 80)
    print("FORWARD VS BACKWARD DISAGREEMENT SLICE ANALYSIS")
    print("=" * 80)

    grouped = defaultdict(list)

    for record in records:

        grouped[
            record["outcome"]
        ].append(
            record
        )

    print()

    print(
        "FORWARD_CORRECT :",
        len(
            grouped[
                "FORWARD_CORRECT"
            ]
        )
    )

    print(
        "BACKWARD_CORRECT:",
        len(
            grouped[
                "BACKWARD_CORRECT"
            ]
        )
    )

    #
    # --------------------------------------------------
    # Aggregate structural statistics.
    # --------------------------------------------------
    #

    for outcome in [
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT"
    ]:

        group = grouped[
            outcome
        ]

        if not group:
            continue

        print()
        print("=" * 80)
        print(outcome)
        print("=" * 80)

        forward_sizes = [
            len(r["forward_nodes"])
            for r in group
        ]

        backward_sizes = [
            len(r["backward_nodes"])
            for r in group
        ]

        overlap_sizes = [
            len(r["overlap"])
            for r in group
        ]

        forward_only_sizes = [
            len(r["forward_only"])
            for r in group
        ]

        backward_only_sizes = [
            len(r["backward_only"])
            for r in group
        ]

        print()
        print(
            "Average forward slice :",
            sum(forward_sizes)
            / len(forward_sizes)
        )

        print(
            "Average backward slice:",
            sum(backward_sizes)
            / len(backward_sizes)
        )

        print(
            "Average overlap       :",
            sum(overlap_sizes)
            / len(overlap_sizes)
        )

        print(
            "Average forward-only  :",
            sum(forward_only_sizes)
            / len(forward_only_sizes)
        )

        print(
            "Average backward-only :",
            sum(backward_only_sizes)
            / len(backward_only_sizes)
        )

        #
        # --------------------------------------------------
        # Node type composition.
        # --------------------------------------------------
        #

        forward_types = Counter()
        backward_types = Counter()

        for record in group:

            sample = forward_lookup.get(
                record["sample_id"]
            )

            if sample is None:
                continue

            forward_types.update(
                node_type_distribution(

                    sample,

                    record[
                        "forward_only"
                    ]

                )
            )

            backward_types.update(
                node_type_distribution(

                    sample,

                    record[
                        "backward_only"
                    ]

                )
            )

        print()
        print(
            "Forward-only node types:"
        )

        for node_type, count in (
            forward_types.most_common()
        ):

            print(
                f"  {node_type:<20} {count}"
            )

        print()
        print(
            "Backward-only node types:"
        )

        for node_type, count in (
            backward_types.most_common()
        ):

            print(
                f"  {node_type:<20} {count}"
            )

        #
        # --------------------------------------------------
        # Position relative to changed lines.
        # --------------------------------------------------
        #

        forward_position = Counter()
        backward_position = Counter()

        for record in group:

            sample = forward_lookup.get(
                record["sample_id"]
            )

            if sample is None:
                continue

            forward_position.update(
                positional_distribution(

                    sample,

                    record[
                        "forward_only"
                    ]

                )
            )

            backward_position.update(
                positional_distribution(

                    sample,

                    record[
                        "backward_only"
                    ]

                )
            )

        print()
        print(
            "Forward-only node position:"
        )

        print(
            "  Before:",
            forward_position["before"]
        )

        print(
            "  After :",
            forward_position["after"]
        )

        print(
            "  Same  :",
            forward_position["same"]
        )

        print(
            "  Unknown:",
            forward_position["unknown"]
        )

        print()
        print(
            "Backward-only node position:"
        )

        print(
            "  Before:",
            backward_position["before"]
        )

        print(
            "  After :",
            backward_position["after"]
        )

        print(
            "  Same  :",
            backward_position["same"]
        )

        print(
            "  Unknown:",
            backward_position["unknown"]
        )

    #
    # --------------------------------------------------
    # Detailed sample inspection.
    # --------------------------------------------------
    #

    print()
    print("=" * 80)
    print("INDIVIDUAL DISAGREEMENT CASES")
    print("=" * 80)

    for index, record in enumerate(
        records,
        start=1
    ):

        sample = forward_lookup.get(
            record["sample_id"]
        )

        if sample is None:
            continue

        node_lookup = {

            node.node_id:
                node

            for node in sample.cfg["nodes"]

        }

        print()
        print("-" * 80)

        print(
            f"Case {index}"
        )

        print(
            "Outcome:",
            record["outcome"]
        )

        print(
            "Sample ID:",
            record["sample_id"]
        )

        print(
            "Label:",
            record["label"]
        )

        print(
            "Forward prediction:",
            record[
                "forward_prediction"
            ]
        )

        print(
            "Backward prediction:",
            record[
                "backward_prediction"
            ]
        )

        print(
            "Seed lines:",
            record["seed_lines"]
        )

        print()

        print(
            "Forward-only nodes:"
        )

        for node_id in sorted(
            record["forward_only"]
        ):

            node = node_lookup.get(
                node_id
            )

            if node is None:
                continue

            print(
                f"  [{node.node_id}] "
                f"Line {node.lineno} "
                f"{node.node_type}: "
                f"{node.text}"
            )

        print()

        print(
            "Backward-only nodes:"
        )

        for node_id in sorted(
            record["backward_only"]
        ):

            node = node_lookup.get(
                node_id
            )

            if node is None:
                continue

            print(
                f"  [{node.node_id}] "
                f"Line {node.lineno} "
                f"{node.node_type}: "
                f"{node.text}"
            )

        print()

        print(
            "Overlap:",
            len(
                record["overlap"]
            )
        )