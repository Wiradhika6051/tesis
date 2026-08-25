from collections import defaultdict
import re
from src.analysis.slice_analysis import get_slice_nodes


def normalize_text(text):
    """
    Normalize node text so trivial formatting
    differences do not dominate similarity.
    """

    if text is None:
        return ""

    text = text.strip()

    #
    # Collapse whitespace.
    #
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def get_node_signature(node):
    """
    Create a representation that can be compared
    across vulnerable and fixed versions.

    Node IDs and line numbers are intentionally
    excluded because they may differ between versions.
    """

    return (

        node.node_type,

        normalize_text(
            node.text
        )

    )


def get_slice_signatures(
    sample,
    forward
):
    """
    Return node signatures for either the
    forward or backward slice.
    """

    if sample.cfg is None:
        return set()

    if not sample.seed_nodes:
        return set()

    if not sample.function_nodes:
        return set()

    node_ids = get_slice_nodes(

        sample.cfg,

        set(
            sample.seed_nodes
        ),

        set(
            sample.function_nodes
        ),

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

            get_node_signature(
                node
            )

        )

    return signatures


def jaccard_similarity(
    set_a,
    set_b
):
    """
    Jaccard similarity:

        intersection
        ------------
        union
    """

    if not set_a and not set_b:
        return 1.0

    union = set_a | set_b

    if not union:
        return 0.0

    return (

        len(set_a & set_b)
        /
        len(union)

    )


def get_pair_key(
    sample
):
    """
    Pair vulnerable and fixed samples.

    Both samples created from the same GitChange
    should share:

        repo
        parent_commit
        file_path
    """

    return (

        sample.repo,

        sample.parent_commit,

        sample.file_path

    )


def compare_paired_slices(
    samples
):
    """
    Compare vulnerable and fixed samples
    originating from the same change.

    Returns:

        {
            "pairs": [...],
            "missing_pairs": [...],
            "summary": {...}
        }
    """

    grouped = defaultdict(
        dict
    )

    #
    # Group samples into pairs.
    #
    for sample in samples:

        key = get_pair_key(
            sample
        )

        grouped[key][
            sample.label
        ] = sample

    results = []

    missing_pairs = []

    for key, group in grouped.items():

        vulnerable = group.get(
            1
        )

        fixed = group.get(
            0
        )

        #
        # Skip incomplete pairs.
        #
        if vulnerable is None or fixed is None:

            missing_pairs.append({

                "repo":
                    key[0],

                "parent_commit":
                    key[1],

                "file":
                    key[2],

                "has_vulnerable":
                    vulnerable is not None,

                "has_fixed":
                    fixed is not None

            })

            continue

        #
        # Forward slices.
        #
        vulnerable_forward = (
            get_slice_signatures(

                vulnerable,

                forward=True

            )
        )

        fixed_forward = (
            get_slice_signatures(

                fixed,

                forward=True

            )
        )

        #
        # Backward slices.
        #
        vulnerable_backward = (
            get_slice_signatures(

                vulnerable,

                forward=False

            )
        )

        fixed_backward = (
            get_slice_signatures(

                fixed,

                forward=False

            )
        )

        #
        # Calculate similarity.
        #
        forward_similarity = (
            jaccard_similarity(

                vulnerable_forward,

                fixed_forward

            )
        )

        backward_similarity = (
            jaccard_similarity(

                vulnerable_backward,

                fixed_backward

            )
        )

        #
        # Information unique to vulnerable version.
        #
        forward_vulnerable_only = (

            vulnerable_forward
            -
            fixed_forward

        )

        backward_vulnerable_only = (

            vulnerable_backward
            -
            fixed_backward

        )

        #
        # Information unique to fixed version.
        #
        forward_fixed_only = (

            fixed_forward
            -
            vulnerable_forward

        )

        backward_fixed_only = (

            fixed_backward
            -
            vulnerable_backward

        )

        results.append({
            "pair_id":
                key,
            "repo":
                key[0],

            "parent_commit":
                key[1],

            "file":
                key[2],

            #
            # Similarity.
            #
            "forward_similarity":
                forward_similarity,

            "backward_similarity":
                backward_similarity,

            "similarity_gap":
                forward_similarity
                -
                backward_similarity,

            #
            # Slice sizes.
            #
            "vulnerable_forward_size":
                len(
                    vulnerable_forward
                ),

            "fixed_forward_size":
                len(
                    fixed_forward
                ),

            "vulnerable_backward_size":
                len(
                    vulnerable_backward
                ),

            "fixed_backward_size":
                len(
                    fixed_backward
                ),

            #
            # Unique information.
            #
            "forward_vulnerable_only":
                forward_vulnerable_only,

            "forward_fixed_only":
                forward_fixed_only,

            "backward_vulnerable_only":
                backward_vulnerable_only,

            "backward_fixed_only":
                backward_fixed_only

        })

    return {

        "pairs":
            results,

        "missing_pairs":
            missing_pairs

    }

def print_paired_slice_similarity(
    results
):

    pairs = results[
        "pairs"
    ]

    missing_pairs = results[
        "missing_pairs"
    ]

    print()
    print("=" * 80)
    print(
        "PAIRED VULNERABLE VS FIXED "
        "SLICE SIMILARITY"
    )
    print("=" * 80)

    print()

    print(
        f"Valid pairs   : {len(pairs)}"
    )

    print(
        f"Missing pairs : "
        f"{len(missing_pairs)}"
    )

    if not pairs:

        print(
            "\nNo valid pairs found."
        )

        return

    #
    # Similarity.
    #
    forward_similarities = [

        pair[
            "forward_similarity"
        ]

        for pair in pairs

    ]

    backward_similarities = [

        pair[
            "backward_similarity"
        ]

        for pair in pairs

    ]

    gaps = [

        pair[
            "similarity_gap"
        ]

        for pair in pairs

    ]

    print()
    print(
        "## Similarity"
    )
    print()

    print(
        f"Average Forward Similarity  : "
        f"{sum(forward_similarities) / len(pairs):.4f}"
    )

    print(
        f"Average Backward Similarity : "
        f"{sum(backward_similarities) / len(pairs):.4f}"
    )

    print(
        f"Average Similarity Gap      : "
        f"{sum(gaps) / len(pairs):.4f}"
    )

    #
    # Slice sizes.
    #
    print()
    print(
        "## Average Slice Size"
    )
    print()

    for key, title in [

        (
            "vulnerable_forward_size",
            "Vulnerable Forward"
        ),

        (
            "fixed_forward_size",
            "Fixed Forward"
        ),

        (
            "vulnerable_backward_size",
            "Vulnerable Backward"
        ),

        (
            "fixed_backward_size",
            "Fixed Backward"
        )

    ]:

        average = (

            sum(

                pair[key]

                for pair in pairs

            )

            /

            len(pairs)

        )

        print(
            f"{title:<25}: "
            f"{average:.2f}"
        )

    #
    # Distribution.
    #
    print()
    print(
        "## Similarity Distribution"
    )
    print()

    categories = {

        "Nearly identical (>= 0.90)": 0,

        "High similarity (0.70-0.90)": 0,

        "Moderate similarity (0.40-0.70)": 0,

        "Low similarity (< 0.40)": 0

    }

    for similarity in forward_similarities:

        if similarity >= 0.90:

            categories[
                "Nearly identical (>= 0.90)"
            ] += 1

        elif similarity >= 0.70:

            categories[
                "High similarity (0.70-0.90)"
            ] += 1

        elif similarity >= 0.40:

            categories[
                "Moderate similarity (0.40-0.70)"
            ] += 1

        else:

            categories[
                "Low similarity (< 0.40)"
            ] += 1

    print(
        "Forward"
    )

    for category, count in categories.items():

        percentage = (

            count
            /
            len(pairs)
            *
            100

        )

        print(

            f"  {category:<35} "
            f"{count:>4} "
            f"({percentage:.2f}%)"

        )

    #
    # Recalculate for backward.
    #
    categories = {

        "Nearly identical (>= 0.90)": 0,

        "High similarity (0.70-0.90)": 0,

        "Moderate similarity (0.40-0.70)": 0,

        "Low similarity (< 0.40)": 0

    }

    for similarity in backward_similarities:

        if similarity >= 0.90:

            categories[
                "Nearly identical (>= 0.90)"
            ] += 1

        elif similarity >= 0.70:

            categories[
                "High similarity (0.70-0.90)"
            ] += 1

        elif similarity >= 0.40:

            categories[
                "Moderate similarity (0.40-0.70)"
            ] += 1

        else:

            categories[
                "Low similarity (< 0.40)"
            ] += 1

    print()
    print(
        "Backward"
    )

    for category, count in categories.items():

        percentage = (

            count
            /
            len(pairs)
            *
            100

        )

        print(

            f"  {category:<35} "
            f"{count:>4} "
            f"({percentage:.2f}%)"

        )

def print_extreme_similarity_pairs(
    results,
    limit=10
):

    pairs = results[
        "pairs"
    ]

    if not pairs:

        return

    print()
    print("=" * 80)
    print(
        "LARGEST FORWARD VS BACKWARD "
        "SIMILARITY DIFFERENCES"
    )
    print("=" * 80)

    sorted_pairs = sorted(

        pairs,

        key=lambda pair: abs(
            pair[
                "similarity_gap"
            ]
        ),

        reverse=True

    )

    for pair in sorted_pairs[:limit]:

        print()
        print("-" * 80)

        print(
            "Repo:",
            pair["repo"]
        )

        print(
            "File:",
            pair["file"]
        )

        print()

        print(
            f"Forward similarity : "
            f"{pair['forward_similarity']:.4f}"
        )

        print(
            f"Backward similarity: "
            f"{pair['backward_similarity']:.4f}"
        )

        print(
            f"Difference         : "
            f"{pair['similarity_gap']:.4f}"
        )

        print()

        print(
            "Forward vulnerable-only:"
        )

        for signature in list(
            pair[
                "forward_vulnerable_only"
            ]
        )[:10]:

            print(
                " ",
                signature
            )

        print()

        print(
            "Forward fixed-only:"
        )

        for signature in list(
            pair[
                "forward_fixed_only"
            ]
        )[:10]:

            print(
                " ",
                signature
            )

        print()

        print(
            "Backward vulnerable-only:"
        )

        for signature in list(
            pair[
                "backward_vulnerable_only"
            ]
        )[:10]:

            print(
                " ",
                signature
            )

        print()

        print(
            "Backward fixed-only:"
        )

        for signature in list(
            pair[
                "backward_fixed_only"
            ]
        )[:10]:

            print(
                " ",
                signature
            )