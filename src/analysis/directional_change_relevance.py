from collections import Counter

from src.analysis.paired_slice_similarity import normalize_text


def _sample_id(sample):

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def _signature(node):

    return (
        node.node_type,
        normalize_text(
            node.text
        )
    )


def _signatures(sample):

    if sample.pruned_cfg is None:
        return set()

    return {

        _signature(node)

        for node in sample.pruned_cfg["nodes"]

    }


def _changed_nodes(
    vulnerable,
    fixed
):

    vulnerable_nodes = _signatures(
        vulnerable
    )

    fixed_nodes = _signatures(
        fixed
    )

    return {

        "vulnerable_only":
            vulnerable_nodes - fixed_nodes,

        "fixed_only":
            fixed_nodes - vulnerable_nodes,

        "common":
            vulnerable_nodes & fixed_nodes

    }


def _node_types(
    nodes
):

    return Counter(

        node_type

        for node_type, _ in nodes

    )


def analyze_directional_change_relevance(

    directional_results,

    forward_samples,
    backward_samples

):

    #
    # --------------------------------------------------------
    # Lookups
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

    #
    # --------------------------------------------------------
    # Group vulnerable/fixed samples.
    # --------------------------------------------------------
    #

    pairs = {}

    for sample in forward_samples:

        pair_id = (

            sample.repo,
            sample.parent_commit,
            sample.file_path

        )

        pairs.setdefault(
            pair_id,
            {}
        )["forward"] = sample

    for sample in backward_samples:

        pair_id = (

            sample.repo,
            sample.parent_commit,
            sample.file_path

        )

        pairs.setdefault(
            pair_id,
            {}
        )["backward"] = sample

    #
    # --------------------------------------------------------
    # Results.
    # --------------------------------------------------------
    #

    results = []

    for directional in directional_results:

        outcome = directional.get(
            "outcome"
        )

        if outcome not in (

            "FORWARD_CORRECT",
            "BACKWARD_CORRECT"

        ):

            continue

        sample_id = directional.get(
            "sample_id"
        )

        if sample_id is None:
            continue

        pair_id = sample_id[:3]

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
        # Need vulnerable/fixed versions.
        #

        pair = pairs.get(
            pair_id
        )

        if pair is None:
            continue

        #
        # Find vulnerable/fixed sample.
        #

        vulnerable_forward = None
        fixed_forward = None

        vulnerable_backward = None
        fixed_backward = None

        #
        # Search explicitly because sample_id contains label.
        #

        for sample in forward_samples:

            if (

                sample.repo,
                sample.parent_commit,
                sample.file_path

            ) != pair_id:

                continue

            if sample.label == 1:

                vulnerable_forward = sample

            else:

                fixed_forward = sample

        for sample in backward_samples:

            if (

                sample.repo,
                sample.parent_commit,
                sample.file_path

            ) != pair_id:

                continue

            if sample.label == 1:

                vulnerable_backward = sample

            else:

                fixed_backward = sample

        if (
            vulnerable_forward is None
            or
            fixed_forward is None
            or
            vulnerable_backward is None
            or
            fixed_backward is None
        ):

            continue

        #
        # ----------------------------------------------------
        # Semantic changes.
        # ----------------------------------------------------
        #

        forward_change = _changed_nodes(

            vulnerable_forward,
            fixed_forward

        )

        backward_change = _changed_nodes(

            vulnerable_backward,
            fixed_backward

        )

        #
        # ----------------------------------------------------
        # Directional context.
        # ----------------------------------------------------
        #

        forward_nodes = _signatures(
            forward_sample
        )

        backward_nodes = _signatures(
            backward_sample
        )

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

        #
        # Which directional nodes are actually
        # related to the vulnerable/fixed change?
        #
        forward_changed = (
            forward_only
            &
            (
                forward_change[
                    "vulnerable_only"
                ]
                |
                forward_change[
                    "fixed_only"
                ]
            )
        )

        backward_changed = (
            backward_only
            &
            (
                backward_change[
                    "vulnerable_only"
                ]
                |
                backward_change[
                    "fixed_only"
                ]
            )
        )

        #
        # ----------------------------------------------------
        # Store.
        # ----------------------------------------------------
        #

        results.append({

            "sample_id":
                sample_id,

            "outcome":
                outcome,

            "forward_only":
                len(forward_only),

            "backward_only":
                len(backward_only),

            "forward_changed":
                len(forward_changed),

            "backward_changed":
                len(backward_changed),

            "forward_change_relevance":
                (
                    len(forward_changed)
                    /
                    len(forward_only)
                    if forward_only
                    else 0.0
                ),

            "backward_change_relevance":
                (
                    len(backward_changed)
                    /
                    len(backward_only)
                    if backward_only
                    else 0.0
                ),

            "forward_changed_types":
                dict(
                    _node_types(
                        forward_changed
                    )
                ),

            "backward_changed_types":
                dict(
                    _node_types(
                        backward_changed
                    )
                )

        })

    return results

def print_directional_change_relevance(
    results
):

    print()

    print(
        "=" * 100
    )

    print(
        "DIRECTIONAL CONTEXT CHANGE RELEVANCE"
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

            for r in results

            if r["outcome"] == outcome

        ]

        print()

        print(
            outcome
        )

        print(
            "-" * 60
        )

        print(
            "Samples:",
            len(subset)
        )

        if not subset:
            continue

        forward_relevance = [

            r["forward_change_relevance"]

            for r in subset

        ]

        backward_relevance = [

            r["backward_change_relevance"]

            for r in subset

        ]

        print()

        print(
            "Forward-only context"
        )

        print(
            "  Average change relevance:",
            f"{sum(forward_relevance) / len(forward_relevance):.2%}"
        )

        print()

        print(
            "Backward-only context"
        )

        print(
            "  Average change relevance:",
            f"{sum(backward_relevance) / len(backward_relevance):.2%}"
        )

        print()

        print(
            "Per-sample:"
        )

        for r in subset:

            print()

            print(
                "Sample:",
                r["sample_id"]
            )

            print(
                "  Forward-only:",
                r["forward_only"]
            )

            print(
                "  Forward changed:",
                r["forward_changed"]
            )

            print(
                "  Forward relevance:",
                f"{r['forward_change_relevance']:.2%}"
            )

            print(
                "  Backward-only:",
                r["backward_only"]
            )

            print(
                "  Backward changed:",
                r["backward_changed"]
            )

            print(
                "  Backward relevance:",
                f"{r['backward_change_relevance']:.2%}"
            )

            print(
                "  Forward changed types:",
                r["forward_changed_types"]
            )

            print(
                "  Backward changed types:",
                r["backward_changed_types"]
            )