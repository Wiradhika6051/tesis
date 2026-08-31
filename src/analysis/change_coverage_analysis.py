from collections import defaultdict
import statistics

from src.analysis.paired_slice_similarity import normalize_text

def _get_sample_id(sample):
    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )

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

    #
    # Build sample lookup.
    #
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

    results = []

    for result in directional_results:

        sample_id = result.get(
            "sample_id"
        )

        if sample_id is None:
            continue

        #
        # Remove label to obtain pair identity.
        #
        pair_key = sample_id[:3]

        vulnerable_id = (
            pair_key[0],
            pair_key[1],
            pair_key[2],
            1
        )

        fixed_id = (
            pair_key[0],
            pair_key[1],
            pair_key[2],
            0
        )

        #
        # Locate vulnerable/fixed samples.
        #
        forward_vulnerable = (
            forward_lookup.get(
                vulnerable_id
            )
        )

        forward_fixed = (
            forward_lookup.get(
                fixed_id
            )
        )

        #
        # Locate the exact sample being analyzed.
        #
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
            forward_vulnerable is None
            or
            forward_fixed is None
            or
            forward_sample is None
            or
            backward_sample is None
        ):
            continue

        #
        # Get directional slice signatures.
        #
        forward_signatures = _slice_signatures(
            forward_sample,
            forward=True
        )

        backward_signatures = _slice_signatures(
            backward_sample,
            forward=False
        )

        #
        # Calculate change coverage.
        #
        coverage = _change_capture(

            forward_vulnerable,
            forward_fixed,

            forward_signatures,
            backward_signatures

        )

        record = dict(
            result
        )

        record.update(
            coverage
        )

        record[
            "change_coverage_difference"
        ] = (

            coverage[
                "forward_change_coverage"
            ]

            -

            coverage[
                "backward_change_coverage"
            ]

        )

        results.append(
            record
        )

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