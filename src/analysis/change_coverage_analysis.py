from collections import defaultdict
import statistics

from src.analysis.paired_slice_similarity import normalize_text
from src.analysis.slice_analysis import get_slice_nodes


def _node_signature(node):
    """
    Create the same semantic signature used by
    directional slice analysis.
    """

    return (
        node.node_type,
        normalize_text(
            node.text
        )
    )

def _get_sample_id(sample):
    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )

def _cfg_signatures(sample):
    """
    Return semantic signatures for every node
    in the original function CFG.
    """

    if sample.cfg is None:
        return set()

    if not sample.cfg.get("nodes"):
        return set()

    return {
        _node_signature(node)
        for node in sample.cfg["nodes"]
    }


def _pruned_signatures(sample):
    """
    Return semantic signatures contained in the
    pruned CFG.
    """

    if sample.pruned_cfg is None:
        return set()

    if not sample.pruned_cfg.get("nodes"):
        return set()

    return {
        _node_signature(node)
        for node in sample.pruned_cfg["nodes"]
    }


def _slice_signatures(
    sample,
    forward
):
    """
    Return semantic signatures belonging to the
    directional slice.

    Uses the existing get_slice_signatures()
    implementation so that this analysis remains
    consistent with the other analyses.
    """

    from src.analysis.paired_slice_similarity import (
        get_slice_signatures
    )

    return get_slice_signatures(
        sample,
        forward=forward
    )


def _pair_key(sample):
    """
    Vulnerable/fixed pair identifier.
    """

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path
    )


def _build_pairs(samples):
    """
    Group samples into vulnerable/fixed pairs.
    """

    pairs = defaultdict(dict)

    for sample in samples:

        key = _pair_key(sample)

        pairs[key][
            sample.label
        ] = sample

    return pairs


def _change_signatures(
    vulnerable,
    fixed
):
    """
    Determine semantic signatures that differ
    between vulnerable and fixed versions.

    Returns:

        vulnerable_only
        fixed_only
        changed
    """

    vulnerable_signatures = _cfg_signatures(
        vulnerable
    )

    fixed_signatures = _cfg_signatures(
        fixed
    )

    vulnerable_only = (
        vulnerable_signatures
        -
        fixed_signatures
    )

    fixed_only = (
        fixed_signatures
        -
        vulnerable_signatures
    )

    changed = (
        vulnerable_only
        |
        fixed_only
    )

    return (
        vulnerable_only,
        fixed_only,
        changed
    )


def _coverage(
    slice_signatures,
    target_signatures
):
    """
    Calculate how much of target_signatures is
    represented by the directional slice.
    """

    if not target_signatures:
        return 0.0

    return (
        len(
            slice_signatures
            &
            target_signatures
        )
        /
        len(target_signatures)
    )


def _change_capture(
    vulnerable,
    fixed,
    forward_signatures,
    backward_signatures
):
    """
    Calculate directional coverage of the
    vulnerable/fixed semantic change.
    """

    (
        vulnerable_only,
        fixed_only,
        changed
    ) = _change_signatures(
        vulnerable,
        fixed
    )

    return {

        "changed_count":
            len(changed),

        "vulnerable_change_count":
            len(vulnerable_only),

        "fixed_change_count":
            len(fixed_only),

        #
        # Overall change coverage.
        #
        "forward_change_coverage":
            _coverage(
                forward_signatures,
                changed
            ),

        "backward_change_coverage":
            _coverage(
                backward_signatures,
                changed
            ),

        #
        # Vulnerable-side change coverage.
        #
        "forward_vulnerable_coverage":
            _coverage(
                forward_signatures,
                vulnerable_only
            ),

        "backward_vulnerable_coverage":
            _coverage(
                backward_signatures,
                vulnerable_only
            ),

        #
        # Fixed-side change coverage.
        #
        "forward_fixed_coverage":
            _coverage(
                forward_signatures,
                fixed_only
            ),

        "backward_fixed_coverage":
            _coverage(
                backward_signatures,
                fixed_only
            )
    }
def analyze_change_coverage(
    directional_results,
    forward_samples,
    backward_samples
):

    forward_lookup = _build_sample_lookup(
        forward_samples
    )

    backward_lookup = _build_sample_lookup(
        backward_samples
    )

    #
    # Build pair lookup.
    #
    forward_pairs = {}

    for sample in forward_samples:

        pair_id = _pair_id(sample)

        forward_pairs.setdefault(
            pair_id,
            {}
        )[sample.label] = sample

    backward_pairs = {}

    for sample in backward_samples:

        pair_id = _pair_id(sample)

        backward_pairs.setdefault(
            pair_id,
            {}
        )[sample.label] = sample

    results = []

    for directional in directional_results:

        sample_id = directional.get(
            "sample_id"
        )

        if sample_id is None:
            continue

        outcome = directional.get(
            "outcome"
        )

        pair_id = sample_id[:3]

        #
        # Locate pair.
        #
        forward_pair = forward_pairs.get(
            pair_id,
            {}
        )

        backward_pair = backward_pairs.get(
            pair_id,
            {}
        )

        forward_vulnerable = (
            forward_pair.get(1)
        )

        forward_fixed = (
            forward_pair.get(0)
        )

        backward_vulnerable = (
            backward_pair.get(1)
        )

        backward_fixed = (
            backward_pair.get(0)
        )

        #
        # Require complete pairs.
        #
        if (
            forward_vulnerable is None
            or
            forward_fixed is None
            or
            backward_vulnerable is None
            or
            backward_fixed is None
        ):
            continue

        #
        # Semantic changes.
        #
        forward_vulnerable_signatures = (
            _get_pruned_signatures(
                forward_vulnerable
            )
        )

        forward_fixed_signatures = (
            _get_pruned_signatures(
                forward_fixed
            )
        )

        backward_vulnerable_signatures = (
            _get_pruned_signatures(
                backward_vulnerable
            )
        )

        backward_fixed_signatures = (
            _get_pruned_signatures(
                backward_fixed
            )
        )

        #
        # What changed between vulnerable
        # and fixed versions?
        #
        forward_vulnerable_only = (
            forward_vulnerable_signatures
            -
            forward_fixed_signatures
        )

        forward_fixed_only = (
            forward_fixed_signatures
            -
            forward_vulnerable_signatures
        )

        backward_vulnerable_only = (
            backward_vulnerable_signatures
            -
            backward_fixed_signatures
        )

        backward_fixed_only = (
            backward_fixed_signatures
            -
            backward_vulnerable_signatures
        )

        forward_change = (
            forward_vulnerable_only
            |
            forward_fixed_only
        )

        backward_change = (
            backward_vulnerable_only
            |
            backward_fixed_only
        )

        #
        # Directional slice for the exact
        # sample being classified.
        #
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

        forward_slice = (
            _get_directional_signatures(
                forward_sample,
                forward=True
            )
        )

        backward_slice = (
            _get_directional_signatures(
                backward_sample,
                forward=False
            )
        )

        #
        # Coverage of semantic change.
        #
        forward_captured = (
            forward_slice
            &
            forward_change
        )

        backward_captured = (
            backward_slice
            &
            backward_change
        )

        if forward_change:

            forward_coverage = (
                len(forward_captured)
                /
                len(forward_change)
            )

        else:

            forward_coverage = 0.0

        if backward_change:

            backward_coverage = (
                len(backward_captured)
                /
                len(backward_change)
            )

        else:

            backward_coverage = 0.0

        results.append({

            "sample_id":
                sample_id,

            "outcome":
                outcome,

            "forward_change_size":
                len(forward_change),

            "backward_change_size":
                len(backward_change),

            "forward_captured":
                len(forward_captured),

            "backward_captured":
                len(backward_captured),

            "forward_change_coverage":
                forward_coverage,

            "backward_change_coverage":
                backward_coverage,

            "change_coverage_difference":
                (
                    forward_coverage
                    -
                    backward_coverage
                )

        })

    return results
def _mean(
    records,
    key
):
    values = [

        r[key]

        for r in records

        if key in r

    ]

    if not values:
        return 0.0

    return statistics.mean(
        values
    )


def _median(
    records,
    key
):
    values = [

        r[key]

        for r in records

        if key in r

    ]

    if not values:
        return 0.0

    return statistics.median(
        values
    )


def print_change_coverage(
    results
):

    outcomes = [

        "BOTH_CORRECT",

        "BOTH_WRONG",

        "FORWARD_CORRECT",

        "BACKWARD_CORRECT"

    ]

    print()

    print(
        "=" * 100
    )

    print(
        "SEMANTIC CHANGE COVERAGE BY PREDICTION OUTCOME"
    )

    print(
        "=" * 100
    )

    print()

    header = (

        f"{'Metric':35}"
        f"{'Both correct':>16}"
        f"{'Both wrong':>16}"
        f"{'Forward correct':>18}"
        f"{'Backward correct':>20}"

    )

    print(
        header
    )

    print(
        "-" * 100
    )

    metrics = [

        (
            "Forward change coverage",
            "forward_change_coverage"
        ),

        (
            "Backward change coverage",
            "backward_change_coverage"
        ),

        (
            "Forward vulnerable coverage",
            "forward_vulnerable_coverage"
        ),

        (
            "Backward vulnerable coverage",
            "backward_vulnerable_coverage"
        ),

        (
            "Forward fixed coverage",
            "forward_fixed_coverage"
        ),

        (
            "Backward fixed coverage",
            "backward_fixed_coverage"
        ),

        (
            "Change coverage difference",
            "change_coverage_difference"
        )

    ]

    for label, key in metrics:

        values = []

        for outcome in outcomes:

            records = [

                r

                for r in results

                if r.get("outcome")
                == outcome

            ]

            values.append(
                _mean(
                    records,
                    key
                )
            )

        print(

            f"{label:35}"
            f"{values[0]:15.2%}"
            f"{values[1]:15.2%}"
            f"{values[2]:17.2%}"
            f"{values[3]:19.2%}"

        )

    #
    # --------------------------------------------------
    # Winning-direction analysis.
    # --------------------------------------------------
    #

    print()

    print(
        "=" * 100
    )

    print(
        "WINNING DIRECTION VS CHANGE COVERAGE"
    )

    print(
        "=" * 100
    )

    print()

    forward_wins = [

        r

        for r in results

        if r.get("outcome")
        == "FORWARD_CORRECT"

    ]

    backward_wins = [

        r

        for r in results

        if r.get("outcome")
        == "BACKWARD_CORRECT"

    ]

    print(
        "Forward-correct samples:",
        len(forward_wins)
    )

    print(

        "  Forward captures more change:",

        sum(

            r[
                "change_coverage_difference"
            ] > 0

            for r in forward_wins

        )

    )

    print(

        "  Backward captures more change:",

        sum(

            r[
                "change_coverage_difference"
            ] < 0

            for r in forward_wins

        )

    )

    print()

    print(
        "Backward-correct samples:",
        len(backward_wins)
    )

    print(

        "  Backward captures more change:",

        sum(

            r[
                "change_coverage_difference"
            ] < 0

            for r in backward_wins

        )

    )

    print(

        "  Forward captures more change:",

        sum(

            r[
                "change_coverage_difference"
            ] > 0

            for r in backward_wins

        )

    )

    #
    # --------------------------------------------------
    # Detailed winning-direction averages.
    # --------------------------------------------------
    #

    print()

    print(
        "AVERAGE CHANGE COVERAGE"
    )

    print(
        "-" * 70
    )

    print(

        f"{'Metric':35}"
        f"{'Forward correct':>18}"
        f"{'Backward correct':>20}"

    )

    print(
        "-" * 70
    )

    winning_metrics = [

        (
            "Forward change coverage",
            "forward_change_coverage"
        ),

        (
            "Backward change coverage",
            "backward_change_coverage"
        ),

        (
            "Coverage difference",
            "change_coverage_difference"
        )

    ]

    for label, key in winning_metrics:

        print(

            f"{label:35}"
            f"{_mean(forward_wins, key):17.2%}"
            f"{_mean(backward_wins, key):19.2%}"

        )

def _sample_id(sample):

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def _pair_id(sample):

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path
    )

def _build_sample_lookup(samples):

    lookup = {}

    for sample in samples:

        sample_id = _sample_id(sample)

        lookup[sample_id] = sample

    return lookup

def _get_pruned_signatures(sample):

    if sample.pruned_cfg is None:
        return set()

    signatures = set()

    for node in sample.pruned_cfg["nodes"]:

        signatures.add(
            (
                node.node_type,
                normalize_text(node.text)
            )
        )

    return signatures

def _get_directional_signatures(
    sample,
    forward
):

    if sample.cfg is None:
        return set()

    if not sample.seed_nodes:
        return set()

    if not sample.function_nodes:
        return set()

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

    signatures = set()

    for node_id in node_ids:

        node = node_lookup.get(
            node_id
        )

        if node is None:
            continue

        signatures.add(

            (
                node.node_type,
                normalize_text(node.text)
            )

        )

    return signatures