from collections import Counter
from statistics import mean, median

from scipy.stats import wilcoxon


# ============================================================
# NODE CATEGORY MAPPING
# ============================================================

NODE_CATEGORIES = {
    # Data / execution operations
    "Assign": "DATA_OPERATION",
    "AnnAssign": "DATA_OPERATION",
    "AugAssign": "DATA_OPERATION",
    "Expr": "DATA_OPERATION",
    "Return": "DATA_OPERATION",
    "Raise": "DATA_OPERATION",

    # Control flow
    "If": "CONTROL_FLOW",
    "For": "CONTROL_FLOW",
    "AsyncFor": "CONTROL_FLOW",
    "While": "CONTROL_FLOW",
    "Try": "CONTROL_FLOW",
    "With": "CONTROL_FLOW",
    "AsyncWith": "CONTROL_FLOW",
    "Continue": "CONTROL_FLOW",
    "Break": "CONTROL_FLOW",
    "Pass": "CONTROL_FLOW",

    # Structural
    "FunctionDef": "STRUCTURAL",
    "AsyncFunctionDef": "STRUCTURAL",
    "ClassDef": "STRUCTURAL",
    "Import": "STRUCTURAL",
    "ImportFrom": "STRUCTURAL",

    # CFG structural nodes
    "ENTRY": "STRUCTURAL",
    "EXIT": "STRUCTURAL",
    "MERGE": "STRUCTURAL",
    "LOOP_EXIT": "STRUCTURAL",
}


# ============================================================
# CONTEXT NORMALIZATION
# ============================================================

def _normalize_context(context):
    """
    Normalize context into a flat list of node dictionaries.

    Supports:
        - list[dict]
        - tuple[list[dict], list[dict]]
    """

    if context is None:
        return []

    if isinstance(context, tuple):
        flattened = []

        for part in context:
            if isinstance(part, list):
                flattened.extend(part)

        return flattened

    if isinstance(context, list):
        return context

    raise TypeError(
        f"Unsupported context type: {type(context)}"
    )


# ============================================================
# DISTANCE HELPERS
# ============================================================

def _is_local(distance_bucket):
    """
    Local context means CFG distance 1 to 3.

    Distance 0 is intentionally excluded because
    it refers to the seed itself rather than
    additional contextual nodes.
    """

    return distance_bucket in {
        "1",
        "2-3",
    }


def _is_distant(distance_bucket):
    """
    Distant context means CFG distance >5.
    """

    return distance_bucket == ">5"


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def _extract_features(context):

    context = _normalize_context(context)

    features = Counter()

    # Total unique contextual nodes
    features["context_size"] = len(context)

    for item in context:

        node_type = item.get("node_type")

        position = item.get(
            "position",
            "UNKNOWN"
        )

        distance_bucket = item.get(
            "distance_bucket",
            "UNKNOWN"
        )

        # ----------------------------------------------------
        # Position
        # ----------------------------------------------------

        if position == "BEFORE":
            features["before"] += 1

        elif position == "AFTER":
            features["after"] += 1

        elif position == "WITHIN_SEED_RANGE":
            features["within_seed_range"] += 1

        # ----------------------------------------------------
        # CFG distance
        # ----------------------------------------------------

        if _is_local(distance_bucket):
            features["local_cfg"] += 1

        if _is_distant(distance_bucket):
            features["distant_cfg"] += 1

        # ----------------------------------------------------
        # Node category
        # ----------------------------------------------------

        category = NODE_CATEGORIES.get(
            node_type
        )

        if category == "DATA_OPERATION":
            features["data_operation"] += 1

        elif category == "CONTROL_FLOW":
            features["control_flow"] += 1

        elif category == "STRUCTURAL":
            features["structural"] += 1

    return features


# ============================================================
# SAFE WILCOXON
# ============================================================

def _paired_wilcoxon(
    winner_values,
    loser_values,
):

    differences = [
        winner - loser
        for winner, loser in zip(
            winner_values,
            loser_values
        )
    ]

    nonzero_differences = [
        difference
        for difference in differences
        if difference != 0
    ]

    # No differences at all
    if not nonzero_differences:

        return {
            "statistic": None,
            "p_value": None,
            "n_nonzero": 0,
        }

    try:

        statistic, p_value = wilcoxon(
            winner_values,
            loser_values,
            zero_method="wilcox",
            alternative="two-sided",
        )

        return {
            "statistic": statistic,
            "p_value": p_value,
            "n_nonzero": len(
                nonzero_differences
            ),
        }

    except ValueError:

        return {
            "statistic": None,
            "p_value": None,
            "n_nonzero": len(
                nonzero_differences
            ),
        }


# ============================================================
# EFFECT SIZE
# ============================================================

def _rank_biserial_effect_size(
    winner_values,
    loser_values,
):
    """
    Simple paired directional effect estimate.

    Range:
        -1 = loser consistently larger
         0 = no directional tendency
        +1 = winner consistently larger
    """

    differences = [
        winner - loser
        for winner, loser in zip(
            winner_values,
            loser_values
        )
    ]

    nonzero = [
        difference
        for difference in differences
        if difference != 0
    ]

    if not nonzero:
        return 0.0

    positive = sum(
        difference > 0
        for difference in nonzero
    )

    negative = sum(
        difference < 0
        for difference in nonzero
    )

    return (
        positive - negative
    ) / len(nonzero)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_paired_directional_feature_difference(
    winner_context_results,
):
    """
    Compare winner and loser directional contexts.

    Expected input structure:

    {
        "FORWARD_CORRECT": [
            {
                "sample_id": ...,
                "winner": "FORWARD",
                "winner_context": [...],
                "loser_context": [...],
            },
            ...
        ],
        "BACKWARD_CORRECT": [
            ...
        ],
    }
    """

    results = {
        "FORWARD_CORRECT": [],
        "BACKWARD_CORRECT": [],
    }

    for outcome, records in winner_context_results.items():

        for record in records:

            winner_context = record.get(
                "winner_context",
                []
            )

            loser_context = record.get(
                "loser_context",
                []
            )

            sample_id = record.get(
                "sample_id"
            )

            winner = record.get(
                "winner"
            )

            # --------------------------------------------------
            # Skip samples where both contexts are empty
            # --------------------------------------------------

            if not winner_context and not loser_context:
                continue

            # --------------------------------------------------
            # Calculate features
            # --------------------------------------------------

            winner_features = extract_context_features(
                winner_context
            )

            loser_features = extract_context_features(
                loser_context
            )

            # --------------------------------------------------
            # Calculate paired differences
            # --------------------------------------------------

            differences = {}

            for feature_name in winner_features:

                winner_value = winner_features.get(
                    feature_name,
                    0
                )

                loser_value = loser_features.get(
                    feature_name,
                    0
                )

                differences[
                    feature_name
                ] = (
                    winner_value
                    - loser_value
                )

            results[outcome].append(
                {
                    "sample_id": sample_id,
                    "winner": winner,
                    "winner_features": winner_features,
                    "loser_features": loser_features,
                    "differences": differences,
                }
            )

    return results

# ============================================================
# PRINT RESULTS
# ============================================================

def print_paired_directional_feature_difference(
    analysis,
):

    print()
    print("=" * 100)
    print(
        "PAIRED WINNER VS LOSER FEATURE DIFFERENCE"
    )
    print("=" * 100)

    for outcome, outcome_result in (
        analysis.items()
    ):

        print()
        print(outcome)

        print("-" * 100)

        print(
            f"Samples: "
            f"{outcome_result['sample_count']}"
        )

        print()

        header = (
            f"{'Feature':<24}"
            f"{'Winner':>10}"
            f"{'Loser':>10}"
            f"{'Mean Δ':>10}"
            f"{'Median Δ':>12}"
            f"{'W > L':>8}"
            f"{'L > W':>8}"
            f"{'Equal':>8}"
            f"{'p-value':>12}"
            f"{'Effect':>10}"
        )

        print(header)

        print("-" * 100)

        for feature, result in (
            outcome_result[
                "features"
            ].items()
        ):

            p_value = result[
                "p_value"
            ]

            p_value_text = (
                f"{p_value:.4f}"
                if p_value is not None
                else "N/A"
            )

            print(
                f"{feature:<24}"
                f"{result['winner_mean']:>10.2f}"
                f"{result['loser_mean']:>10.2f}"
                f"{result['mean_difference']:>10.2f}"
                f"{result['median_difference']:>12.2f}"
                f"{result['winner_greater']:>8}"
                f"{result['loser_greater']:>8}"
                f"{result['equal']:>8}"
                f"{p_value_text:>12}"
                f"{result['effect_size']:>10.2f}"
            )

    print()
    print("=" * 100)
    print(
        "INTERPRETATION GUIDE"
    )
    print("=" * 100)

    print()
    print(
        "Mean Δ > 0  : winner contains more of the feature"
    )

    print(
        "Mean Δ < 0  : loser contains more of the feature"
    )

    print(
        "Effect > 0  : winner more frequently larger"
    )

    print(
        "Effect < 0  : loser more frequently larger"
    )

    print(
        "p < 0.05    : paired difference is statistically significant"
    )