from collections import Counter

from src.analysis.paired_slice_similarity import normalize_text


def _pair_id(sample):

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path
    )


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


def _build_pair_lookup(samples):

    lookup = {}

    for sample in samples:

        pair_id = _pair_id(
            sample
        )

        lookup.setdefault(
            pair_id,
            {}
        )[sample.label] = sample

    return lookup


def analyze_directional_change_relevance(

    directional_results,
    forward_samples,
    backward_samples

):

    #
    # --------------------------------------------------
    # Build pair-level lookups.
    # --------------------------------------------------
    #

    forward_pairs = _build_pair_lookup(
        forward_samples
    )

    backward_pairs = _build_pair_lookup(
        backward_samples
    )

    results = []

    unmatched = []

    #
    # --------------------------------------------------
    # Process directional prediction results.
    # --------------------------------------------------
    #

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

        #
        # sample_id:
        #
        # (
        #     repo,
        #     parent_commit,
        #     file_path,
        #     label
        # )
        #

        pair_id = sample_id[:3]

        #
        # --------------------------------------------------
        # Get both vulnerable/fixed samples.
        # --------------------------------------------------
        #

        forward_pair = forward_pairs.get(
            pair_id
        )

        backward_pair = backward_pairs.get(
            pair_id
        )

        if (
            forward_pair is None
            or
            backward_pair is None
        ):

            unmatched.append({

                "sample_id":
                    sample_id,

                "reason":
                    "missing pair"

            })

            continue

        vulnerable_forward = forward_pair.get(
            1
        )

        fixed_forward = forward_pair.get(
            0
        )

        vulnerable_backward = backward_pair.get(
            1
        )

        fixed_backward = backward_pair.get(
            0
        )

        if (
            vulnerable_forward is None
            or
            fixed_forward is None
            or
            vulnerable_backward is None
            or
            fixed_backward is None
        ):

            unmatched.append({

                "sample_id":
                    sample_id,

                "reason":
                    "incomplete vulnerable/fixed pair"

            })

            continue

        #
        # --------------------------------------------------
        # Determine which sample won.
        # --------------------------------------------------
        #

        winning_label = sample_id[3]

        #
        # The directional sample corresponding to
        # the prediction result.
        #

        if winning_label == 1:

            forward_prediction_sample = (
                vulnerable_forward
            )

            backward_prediction_sample = (
                vulnerable_backward
            )

        else:

            forward_prediction_sample = (
                fixed_forward
            )

            backward_prediction_sample = (
                fixed_backward
            )

        #
        # --------------------------------------------------
        # Directional contexts.
        # --------------------------------------------------
        #

        forward_context = _signatures(
            forward_prediction_sample
        )

        backward_context = _signatures(
            backward_prediction_sample
        )

        forward_only = (
            forward_context
            -
            backward_context
        )

        backward_only = (
            backward_context
            -
            forward_context
        )

        #
        # --------------------------------------------------
        # Semantic change.
        #
        # We compare vulnerable vs fixed within the
        # SAME direction.
        # --------------------------------------------------
        #

        vulnerable_forward_nodes = _signatures(
            vulnerable_forward
        )

        fixed_forward_nodes = _signatures(
            fixed_forward
        )

        vulnerable_backward_nodes = _signatures(
            vulnerable_backward
        )

        fixed_backward_nodes = _signatures(
            fixed_backward
        )

        #
        # Nodes that actually changed between
        # vulnerable and fixed.
        #

        forward_changed = (

            vulnerable_forward_nodes
            ^
            fixed_forward_nodes

        )

        backward_changed = (

            vulnerable_backward_nodes
            ^
            fixed_backward_nodes

        )

        #
        # Directional-only context that is also
        # part of the semantic change.
        #

        forward_changed_context = (

            forward_only
            &
            forward_changed

        )

        backward_changed_context = (

            backward_only
            &
            backward_changed

        )

        #
        # --------------------------------------------------
        # Relevance.
        # --------------------------------------------------
        #

        forward_relevance = (

            len(
                forward_changed_context
            )
            /
            len(
                forward_only
            )

            if forward_only

            else 0.0

        )

        backward_relevance = (

            len(
                backward_changed_context
            )
            /
            len(
                backward_only
            )

            if backward_only

            else 0.0

        )

        #
        # --------------------------------------------------
        # Store.
        # --------------------------------------------------
        #

        results.append({

            "sample_id":
                sample_id,

            "pair_id":
                pair_id,

            "outcome":
                outcome,

            "forward_context":
                len(
                    forward_context
                ),

            "backward_context":
                len(
                    backward_context
                ),

            "forward_only":
                len(
                    forward_only
                ),

            "backward_only":
                len(
                    backward_only
                ),

            "forward_changed":
                len(
                    forward_changed
                ),

            "backward_changed":
                len(
                    backward_changed
                ),

            "forward_changed_context":
                len(
                    forward_changed_context
                ),

            "backward_changed_context":
                len(
                    backward_changed_context
                ),

            "forward_relevance":
                forward_relevance,

            "backward_relevance":
                backward_relevance,

            "forward_changed_types":
                dict(
                    Counter(
                        node_type
                        for node_type, _ in
                        forward_changed_context
                    )
                ),

            "backward_changed_types":
                dict(
                    Counter(
                        node_type
                        for node_type, _ in
                        backward_changed_context
                    )
                )

        })

    return {

        "results":
            results,

        "unmatched":
            unmatched

    }

def print_directional_change_relevance(
    analysis
):

    results = analysis[
        "results"
    ]

    unmatched = analysis[
        "unmatched"
    ]

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

            r["forward_relevance"]

            for r in subset

        ]

        backward_relevance = [

            r["backward_relevance"]

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
                "Forward context:",
                r["forward_context"]
            )

            print(
                "Backward context:",
                r["backward_context"]
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
                "Forward changed:",
                r["forward_changed"]
            )

            print(
                "Backward changed:",
                r["backward_changed"]
            )

            print(
                "Forward changed context:",
                r["forward_changed_context"]
            )

            print(
                "Backward changed context:",
                r["backward_changed_context"]
            )

            print(
                "Forward relevance:",
                f"{r['forward_relevance']:.2%}"
            )

            print(
                "Backward relevance:",
                f"{r['backward_relevance']:.2%}"
            )

            print(
                "Forward changed types:",
                r["forward_changed_types"]
            )

            print(
                "Backward changed types:",
                r["backward_changed_types"]
            )

    print()

    print(
        "=" * 100
    )

    print(
        "UNMATCHED"
    )

    print(
        "=" * 100
    )

    print(
        "Count:",
        len(unmatched)
    )