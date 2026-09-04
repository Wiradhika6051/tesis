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
from collections import defaultdict
import statistics


def analyze_paired_directional_feature_difference(
    winner_context_results,
):
    """
    Compare features of winner and loser contexts
    within the same sample.

    Only samples where BOTH winner and loser contexts
    are non-empty are included in the paired comparison.
    """

    results = {}

    for outcome, records in winner_context_results.items():

        paired_records = []

        excluded_empty = 0

        for record in records:

            winner_context = record.get(
                "winner_context",
                []
            )

            loser_context = record.get(
                "loser_context",
                []
            )

            # --------------------------------------------------
            # Paired comparison requires both contexts
            # --------------------------------------------------

            if (
                not winner_context
                or not loser_context
            ):
                excluded_empty += 1
                continue

            # --------------------------------------------------
            # Extract features
            # --------------------------------------------------

            winner_features = (
                extract_context_features(
                    winner_context
                )
            )

            loser_features = (
                extract_context_features(
                    loser_context
                )
            )

            # --------------------------------------------------
            # Compare every feature from either context
            # --------------------------------------------------

            all_features = (
                set(
                    winner_features.keys()
                )
                |
                set(
                    loser_features.keys()
                )
            )

            differences = {}

            for feature_name in all_features:

                winner_value = (
                    winner_features.get(
                        feature_name,
                        0
                    )
                )

                loser_value = (
                    loser_features.get(
                        feature_name,
                        0
                    )
                )

                differences[
                    feature_name
                ] = (
                    winner_value
                    - loser_value
                )

            paired_records.append(
                {
                    "sample_id": record.get(
                        "sample_id"
                    ),

                    "winner": record.get(
                        "winner"
                    ),

                    "winner_features": (
                        winner_features
                    ),

                    "loser_features": (
                        loser_features
                    ),

                    "differences": (
                        differences
                    ),
                }
            )

        # ------------------------------------------------------
        # Aggregate feature differences
        # ------------------------------------------------------

        feature_values = defaultdict(
            list
        )

        for record in paired_records:

            for (
                feature_name,
                difference,
            ) in record[
                "differences"
            ].items():

                feature_values[
                    feature_name
                ].append(
                        difference
                    )

        feature_summary = {}

        for (
            feature_name,
            values,
        ) in feature_values.items():

            positive_count = sum(
                value > 0
                for value in values
            )

            negative_count = sum(
                value < 0
                for value in values
            )

            zero_count = sum(
                value == 0
                for value in values
            )

            # --------------------------------------------------
            # Statistical test
            #
            # Differences are:
            #
            # winner - loser
            #
            # Wilcoxon tests whether the paired differences
            # are systematically different from zero.
            # --------------------------------------------------

            non_zero_values = [
                value
                for value in values
                if value != 0
            ]

            if len(non_zero_values) >= 2:
            
                try:
                
                    statistic, p_value = wilcoxon(
                        values,
                        alternative="two-sided",
                        zero_method="wilcox",
                    )

                except ValueError:
                
                    statistic = None
                    p_value = None

            else:
            
                statistic = None
                p_value = None

            # --------------------------------------------------
            # Effect direction
            # --------------------------------------------------

            mean_difference = statistics.mean(
                values
            )

            median_difference = statistics.median(
                values
            )

            if mean_difference > 0:
            
                direction = "WINNER_HIGHER"

            elif mean_difference < 0:
            
                direction = "LOSER_HIGHER"

            else:
            
                direction = "EQUAL"

            # --------------------------------------------------
            # Store feature result
            # --------------------------------------------------

            feature_summary[
                feature_name
            ] = {
                "mean_difference": (
                    mean_difference
                ),
                "significant": (
                    p_value is not None
                    and p_value < 0.05
            ),

                "median_difference": (
                    median_difference
                ),

                "winner_higher_count": (
                    positive_count
                ),

                "loser_higher_count": (
                    negative_count
                ),

                "equal_count": (
                    zero_count
                ),

                "sample_count": (
                    len(values)
                ),

                "winner_higher_ratio": (
                    positive_count
                    / len(values)
                    if values
                    else 0.0
                ),

                "loser_higher_ratio": (
                    negative_count
                    / len(values)
                    if values
                    else 0.0
                ),

                "wilcoxon_statistic": (
                    statistic
                ),

                "p_value": (
                    p_value
                ),

                "direction": (
                    direction
                ),
            }

        # ------------------------------------------------------
        # Store outcome result
        # ------------------------------------------------------

        results[outcome] = {
            "total_samples": (
                len(records)
            ),

            "sample_count": (
                len(paired_records)
            ),

            "excluded_samples": (
                excluded_empty
            ),

            "records": (
                paired_records
            ),

            "features": (
                feature_summary
            ),
        }

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
        print(
            f"Total directional samples: "
            f"{outcome_result['total_samples']}"
        )

        print(
            f"Paired usable samples: "
            f"{outcome_result['sample_count']}"
        )

        print(
            f"Excluded due to missing "
            f"winner or loser context: "
            f"{outcome_result['excluded_samples']}"
        )
        for feature, result in (
            outcome_result[
                "features"
            ].items()
        ):

            if not isinstance(
                result,
                dict,
            ):
                print(
                    f"{feature:<24}"
                    f"{'N/A':>10}"
                    f"{'N/A':>10}"
                    f"{'N/A':>10}"
                    f"{'N/A':>12}"
                )

                continue
            
            winner_mean = _get_feature_value(
                result,
                "winner_mean",
                "winner_avg",
                "mean_winner",
            )

            loser_mean = _get_feature_value(
                result,
                "loser_mean",
                "loser_avg",
                "mean_loser",
            )

            mean_difference = _get_feature_value(
                result,
                "mean_difference",
                "difference",
                "mean_diff",
            )

            p_value = _get_feature_value(
                result,
                "p_value",
                "pvalue",
            )

            print(
                f"{feature:<24}"
                f"{_format_float(winner_mean)}"
                f"{_format_float(loser_mean)}"
                f"{_format_float(mean_difference)}"
                f"{_format_p_value(p_value):>12}"
            )

            print(
                f"{result['median_difference']:>12.2f}"
                f"{result['winner_greater']:>8}"
                f"{result['loser_greater']:>8}"
                f"{result['equal']:>8}"
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

from collections import Counter


def extract_context_features(
    context,
):
    """
    Extract numerical features from a directional context.

    Parameters
    ----------
    context : list[dict]

        Each item is expected to contain:

        {
            "node_type": str,
            "position": str,
            "distance": int | None,
            "distance_bucket": str,
        }

    Returns
    -------
    dict

        Flat dictionary of numerical features.
    """

    # --------------------------------------------------
    # Basic context size
    # --------------------------------------------------

    features = {}

    features["context_size"] = len(
        context
    )

    # --------------------------------------------------
    # Node type counts
    # --------------------------------------------------

    node_types = Counter(
        item.get(
            "node_type",
            "UNKNOWN"
        )
        for item in context
    )

    for node_type, count in node_types.items():

        features[
            f"node_type::{node_type}"
        ] = count

    # --------------------------------------------------
    # Position counts
    # --------------------------------------------------

    positions = Counter(
        item.get(
            "position",
            "UNKNOWN"
        )
        for item in context
    )

    for position, count in positions.items():

        features[
            f"position::{position}"
        ] = count

    # --------------------------------------------------
    # CFG distance bucket counts
    # --------------------------------------------------

    distance_buckets = Counter(
        item.get(
            "distance_bucket",
            "UNKNOWN"
        )
        for item in context
    )

    for bucket, count in distance_buckets.items():

        features[
            f"distance::{bucket}"
        ] = count

    # --------------------------------------------------
    # Explicit locality features
    # --------------------------------------------------

    local_count = 0
    distant_count = 0

    for item in context:

        bucket = item.get(
            "distance_bucket",
            "UNKNOWN"
        )

        if bucket in {
            "1",
            "2-3",
        }:

            local_count += 1

        elif bucket in {
            "4-5",
            ">5",
        }:

            distant_count += 1

    features["local_context"] = (
        local_count
    )

    features["distant_context"] = (
        distant_count
    )

    # --------------------------------------------------
    # Node category counts
    # --------------------------------------------------

    category_counts = Counter()

    for item in context:

        node_type = item.get(
            "node_type",
            "UNKNOWN"
        )

        category = categorize_node_type(
            node_type
        )

        category_counts[
            category
        ] += 1

    for category, count in category_counts.items():

        features[
            f"category::{category}"
        ] = count
    # --------------------------------------------------
    # Normalized composition features
    # --------------------------------------------------

    context_size = len(
        context
    )

    if context_size > 0:

        for node_type, count in node_types.items():

            features[
                f"node_type_ratio::{node_type}"
            ] = (
                count / context_size
            )

        for position, count in positions.items():

            features[
                f"position_ratio::{position}"
            ] = (
                count / context_size
            )

        for bucket, count in distance_buckets.items():

            features[
                f"distance_ratio::{bucket}"
            ] = (
                count / context_size
            )

        for category, count in category_counts.items():

            features[
                f"category_ratio::{category}"
            ] = (
                count / context_size
            )

        features[
            "local_context_ratio"
        ] = (
            local_count
            / context_size
        )

        features[
            "distant_context_ratio"
        ] = (
            distant_count
            / context_size
        )

    else:

        features[
            "local_context_ratio"
        ] = 0.0

        features[
            "distant_context_ratio"
        ] = 0.0

    return features

def categorize_node_type(
    node_type,
):
    """
    Group CFG / AST node types into
    broader program-context categories.
    """

    data_operation_types = {
        "Assign",
        "AnnAssign",
        "AugAssign",
        "Expr",
        "Return",
        "Raise",
    }

    control_flow_types = {
        "If",
        "For",
        "AsyncFor",
        "While",
        "Try",
        "With",
        "AsyncWith",
        "Continue",
        "Break",
        "Pass",
        "Assert",
        "LOOP_EXIT",
    }

    structural_types = {
        "FunctionDef",
        "AsyncFunctionDef",
        "ClassDef",
        "Import",
        "ImportFrom",
        "ENTRY",
        "EXIT",
        "MERGE",
    }

    if node_type in data_operation_types:

        return "DATA_OPERATION"

    if node_type in control_flow_types:

        return "CONTROL_FLOW"

    if node_type in structural_types:

        return "STRUCTURAL"

    return "OTHER"

def _get_feature_value(result, *keys, default=None):
    """
    Return the first available value from a set of
    possible result keys.
    """

    for key in keys:

        if key in result:
            return result[key]

    return default


def _format_float(value, width=10):

    if value is None:
        return f"{'N/A':>{width}}"

    try:
        return f"{float(value):>{width}.2f}"

    except (
        TypeError,
        ValueError,
    ):
        return f"{str(value):>{width}}"


def _format_p_value(value):

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"

    except (
        TypeError,
        ValueError,
    ):
        return str(value)