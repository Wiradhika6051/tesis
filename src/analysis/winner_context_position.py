from collections import Counter, defaultdict
import statistics


# ============================================================
# BASIC HELPERS
# ============================================================

def _sample_id(sample):
    """
    Build the same sample identifier used by directional_results.
    """

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label,
    )


def _normalize_text(text):

    if text is None:
        return ""

    return " ".join(
        text.strip().split()
    )


def _node_signature(node):

    return (
        node.node_type,
        _normalize_text(node.text),
    )


# ============================================================
# NODE LOOKUP
# ============================================================

def _build_node_lookup(sample):

    if sample.cfg is None:
        return {}

    return {
        node.node_id: node
        for node in sample.cfg["nodes"]
    }


def _get_slice_nodes(
    sample,
    forward,
):
    """
    Return actual CFG nodes belonging to the directional slice.
    """

    if sample.cfg is None:
        return []

    if not sample.seed_nodes:
        return []

    if not sample.function_nodes:
        return []

    from src.analysis.slice_analysis import (
        get_slice_nodes,
    )

    node_ids = get_slice_nodes(
        sample.cfg,
        set(sample.seed_nodes),
        set(sample.function_nodes),
        forward=forward,
    )

    lookup = _build_node_lookup(
        sample
    )

    return [
        lookup[node_id]
        for node_id in node_ids
        if node_id in lookup
    ]


# ============================================================
# CFG DISTANCE
# ============================================================
from collections import defaultdict, deque


def _build_undirected_adjacency(cfg):
    """
    Build an undirected adjacency map from the CFG edges.

    The CFG stores nodes and edges separately:

        cfg["nodes"]
        cfg["edges"]

    Edges are expected to contain source/destination
    node IDs.
    """

    adjacency = defaultdict(set)

    for edge in cfg["edges"]:

        #
        # Handle tuple/list representation:
        #
        #     (source, destination)
        #
        if isinstance(edge, (tuple, list)):

            if len(edge) < 2:
                continue

            source = edge[0]
            target = edge[1]

        #
        # Handle object representation if needed.
        #
        else:

            source = getattr(
                edge,
                "source",
                None
            )

            target = getattr(
                edge,
                "target",
                None
            )

            if source is None:

                source = getattr(
                    edge,
                    "src",
                    None
                )

            if target is None:

                target = getattr(
                    edge,
                    "dst",
                    None
                )

        #
        # Convert CFGNode objects to node IDs.
        #
        if hasattr(source, "node_id"):
            source = source.node_id

        if hasattr(target, "node_id"):
            target = target.node_id

        if source is None or target is None:
            continue

        adjacency[source].add(
            target
        )

        adjacency[target].add(
            source
        )

    return adjacency

def _shortest_distances(
    adjacency,
    seeds,
):
    """
    Compute shortest undirected CFG distance from
    every seed node to every reachable node.
    """

    distances = {}

    queue = deque()

    for seed in seeds:

        #
        # Seeds may occasionally be CFGNode objects.
        #
        if hasattr(seed, "node_id"):
            seed = seed.node_id

        if seed in distances:
            continue

        distances[seed] = 0

        queue.append(
            seed
        )

    while queue:

        current = queue.popleft()

        current_distance = (
            distances[current]
        )

        for neighbor in adjacency.get(
            current,
            ()
        ):

            if neighbor in distances:
                continue

            distances[neighbor] = (
                current_distance + 1
            )

            queue.append(
                neighbor
            )

    return distances

# ============================================================
# POSITION RELATIVE TO SEED
# ============================================================

def _get_line_number(node):

    #
    # Different CFG implementations may
    # expose line information differently.
    #

    for attr in (
        "lineno",
        "line",
        "start_line",
    ):

        value = getattr(
            node,
            attr,
            None,
        )

        if value is not None:
            return value

    return None


def _get_seed_line_range(
    sample,
):

    lookup = _build_node_lookup(
        sample
    )

    lines = []

    for seed_id in sample.seed_nodes:

        node = lookup.get(
            seed_id
        )

        if node is None:
            continue

        line = _get_line_number(
            node
        )

        if line is not None:
            lines.append(
                line
            )

    if not lines:
        return None, None

    return (
        min(lines),
        max(lines),
    )

def _position_relative_to_seed(
    node,
    seed_node_ids,
    seed_min,
    seed_max,
):
    """
    Classify a node relative to the actual seed.

    Priority:

        actual seed node -> AT_SEED

        source line before seed -> BEFORE

        source line after seed -> AFTER

        otherwise -> UNKNOWN
    """

    #
    # Actual seed membership comes first.
    #
    if node.node_id in seed_node_ids:

        return "AT_SEED"

    line = _get_line_number(
        node
    )

    if line is None:
        return "UNKNOWN"

    if line < seed_min:
        return "BEFORE"

    if line > seed_max:
        return "AFTER"

    #
    # Node lies inside the seed's source-line
    # range but isn't itself a seed node.
    #
    return "WITHIN_SEED_RANGE"

# ============================================================
# DISTANCE BUCKET
# ============================================================

def _distance_bucket(distance):

    if distance is None:
        return "UNKNOWN"

    if distance == 0:
        return "0"

    if distance == 1:
        return "1"

    if distance <= 3:
        return "2-3"

    if distance <= 5:
        return "4-5"

    return ">5"


# ============================================================
# EXTRACT DIRECTIONAL-ONLY CONTEXT
# ============================================================

def _directional_context(
    sample,
):

    forward_nodes = _get_slice_nodes(
        sample,
        forward=True,
    )

    backward_nodes = _get_slice_nodes(
        sample,
        forward=False,
    )

    forward_by_signature = {
        _node_signature(node): node
        for node in forward_nodes
    }

    backward_by_signature = {
        _node_signature(node): node
        for node in backward_nodes
    }

    forward_only_signatures = (
        set(forward_by_signature)
        -
        set(backward_by_signature)
    )

    backward_only_signatures = (
        set(backward_by_signature)
        -
        set(forward_by_signature)
    )

    adjacency = _build_undirected_adjacency(
        sample.cfg
    )

    distances = _shortest_distances(
        adjacency,
        set(sample.seed_nodes),
    )

    seed_min, seed_max = (
        _get_seed_line_range(
            sample
        )
    )

    forward_context = []

    for signature in forward_only_signatures:

        node = forward_by_signature[
            signature
        ]

        distance = distances.get(
            node.node_id
        )

        position = (
            _position_relative_to_seed(
                node,
                set(sample.seed_nodes),
                seed_min,
                seed_max,
            )
            if seed_min is not None
            else "UNKNOWN"
        )

        forward_context.append({

            "node_type":
                node.node_type,

            "distance":
                distance,

            "distance_bucket":
                _distance_bucket(
                    distance
                ),

            "position":
                position,

            "signature":
                signature,

        })

    backward_context = []

    for signature in backward_only_signatures:

        node = backward_by_signature[
            signature
        ]

        distance = distances.get(
            node.node_id
        )

        position = (
            _position_relative_to_seed(
                node,
                set(sample.seed_nodes),
                seed_min,
                seed_max,
            )
            if seed_min is not None
            else "UNKNOWN"
        )

        backward_context.append({

            "node_type":
                node.node_type,

            "distance":
                distance,

            "distance_bucket":
                _distance_bucket(
                    distance
                ),

            "position":
                position,

            "signature":
                signature,

        })

    return (
        forward_context,
        backward_context,
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_winner_context_position(
    directional_results,
    forward_samples,
    backward_samples,
):

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

    outcomes = {
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT",
    }

    results = {
        "FORWARD_CORRECT": [],
        "BACKWARD_CORRECT": [],
    }

    processed = set()

    for record in directional_results:

        outcome = record.get(
            "outcome"
        )

        if outcome not in outcomes:
            continue

        sample_id = record.get(
            "sample_id"
        )

        if sample_id in processed:
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
            continue

        #
        # The two samples should represent
        # the same underlying example.
        #
        forward_context, backward_context = (
            _directional_context(
            forward_sample
        )
)

        if outcome == "FORWARD_CORRECT":

            winner = forward_context
            loser = backward_context

        else:

            winner = backward_context
            loser = forward_context

        results[outcome].append({

            "sample_id":
                sample_id,

            "winner":
                (
                    "FORWARD"
                    if outcome == "FORWARD_CORRECT"
                    else "BACKWARD"
                ),

            "winner_context":
                winner,

            "loser_context":
                loser,

        })

        processed.add(
            sample_id
        )

    return results


# ============================================================
# AGGREGATION
# ============================================================
def _aggregate_dimension(
    records,
    context_key,
    dimension,
):
    """
    Aggregate one dimension from winner/loser context.

    Handles both:
        context = [dict, dict, ...]
    and accidentally nested:
        context = [[dict, dict, ...], ...]
    """

    counter = Counter()

    def flatten(items):

        if isinstance(items, dict):
            yield items
            return

        if isinstance(items, (list, tuple, set)):

            for item in items:

                yield from flatten(
                    item
                )

    for record in records:

        context = record.get(
            context_key,
            []
        )

        for item in flatten(context):

            if not isinstance(
                item,
                dict
            ):
                continue

            value = item.get(
                dimension
            )

            if value is None:
                value = "UNKNOWN"

            counter[value] += 1

    return counter

def _aggregate_type_position(
    records,
    context_key,
):

    counter = Counter()

    for record in records:

        for item in record[
            context_key
        ]:

            counter[
                (
                    item["node_type"],
                    item["position"],
                )
            ] += 1

    return counter


def _aggregate_type_distance(
    records,
    context_key,
):

    counter = Counter()

    for record in records:

        for item in record[
            context_key
        ]:

            counter[
                (
                    item["node_type"],
                    item["distance_bucket"],
                )
            ] += 1

    return counter


# ============================================================
# PRINT MAIN RESULT
# ============================================================

def print_winner_context_position(
    results,
):

    print()
    print(
        "=" * 100
    )

    print(
        "WINNER VS LOSER CONTEXT: "
        "NODE TYPE × POSITION × DISTANCE"
    )

    print(
        "=" * 100
    )

    for outcome in (
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT",
    ):

        records = results.get(
            outcome,
            []
        )

        print()
        print(
            outcome
        )

        print(
            "-" * 100
        )

        print(
            f"Samples: {len(records)}"
        )

        if not records:
            continue

        #
        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------
        #

        print()
        print(
            "POSITION"
        )

        print(
            "-" * 60
        )

        winner_position = (
            _aggregate_dimension(
                records,
                "winner_context",
                "position",
            )
        )

        loser_position = (
            _aggregate_dimension(
                records,
                "loser_context",
                "position",
            )
        )

        positions = (
            set(winner_position)
            |
            set(loser_position)
        )

        print(
            f"{'Position':<15}"
            f"{'Winner':>10}"
            f"{'Loser':>10}"
        )

        print(
            "-" * 35
        )

        for position in sorted(
            positions
        ):

            print(
                f"{position:<15}"
                f"{winner_position.get(position, 0):>10}"
                f"{loser_position.get(position, 0):>10}"
            )

        #
        # ----------------------------------------------------
        # DISTANCE
        # ----------------------------------------------------
        #

        print()
        print(
            "CFG DISTANCE"
        )

        print(
            "-" * 60
        )

        winner_distance = (
            _aggregate_dimension(
                records,
                "winner_context",
                "distance_bucket",
            )
        )

        loser_distance = (
            _aggregate_dimension(
                records,
                "loser_context",
                "distance_bucket",
            )
        )

        buckets = [
            "0",
            "1",
            "2-3",
            "4-5",
            ">5",
            "UNKNOWN",
        ]

        print(
            f"{'Distance':<15}"
            f"{'Winner':>10}"
            f"{'Loser':>10}"
        )

        print(
            "-" * 35
        )

        for bucket in buckets:

            print(
                f"{bucket:<15}"
                f"{winner_distance.get(bucket, 0):>10}"
                f"{loser_distance.get(bucket, 0):>10}"
            )

        #
        # ----------------------------------------------------
        # NODE TYPE × POSITION
        # ----------------------------------------------------
        #

        print()
        print(
            "NODE TYPE × POSITION"
        )

        print(
            "-" * 80
        )

        winner_type_position = (
            _aggregate_type_position(
                records,
                "winner_context",
            )
        )

        loser_type_position = (
            _aggregate_type_position(
                records,
                "loser_context",
            )
        )

        combinations = (
            set(winner_type_position)
            |
            set(loser_type_position)
        )

        rows = []

        for (
            node_type,
            position,
        ) in combinations:

            winner_count = (
                winner_type_position.get(
                    (
                        node_type,
                        position,
                    ),
                    0,
                )
            )

            loser_count = (
                loser_type_position.get(
                    (
                        node_type,
                        position,
                    ),
                    0,
                )
            )

            rows.append(
                (
                    node_type,
                    position,
                    winner_count,
                    loser_count,
                    winner_count - loser_count,
                )
            )

        rows.sort(
            key=lambda row:
                abs(row[4]),
            reverse=True,
        )

        print(
            f"{'Node type':<25}"
            f"{'Position':<12}"
            f"{'Winner':>10}"
            f"{'Loser':>10}"
            f"{'Diff':>10}"
        )

        print(
            "-" * 70
        )

        for row in rows:

            print(
                f"{row[0]:<25}"
                f"{row[1]:<12}"
                f"{row[2]:>10}"
                f"{row[3]:>10}"
                f"{row[4]:>10}"
            )

        #
        # ----------------------------------------------------
        # NODE TYPE × DISTANCE
        # ----------------------------------------------------
        #

        print()
        print(
            "NODE TYPE × CFG DISTANCE"
        )

        print(
            "-" * 80
        )

        winner_type_distance = (
            _aggregate_type_distance(
                records,
                "winner_context",
            )
        )

        loser_type_distance = (
            _aggregate_type_distance(
                records,
                "loser_context",
            )
        )

        combinations = (
            set(winner_type_distance)
            |
            set(loser_type_distance)
        )

        rows = []

        for (
            node_type,
            distance,
        ) in combinations:

            winner_count = (
                winner_type_distance.get(
                    (
                        node_type,
                        distance,
                    ),
                    0,
                )
            )

            loser_count = (
                loser_type_distance.get(
                    (
                        node_type,
                        distance,
                    ),
                    0,
                )
            )

            rows.append(
                (
                    node_type,
                    distance,
                    winner_count,
                    loser_count,
                    winner_count - loser_count,
                )
            )

        rows.sort(
            key=lambda row:
                abs(row[4]),
            reverse=True,
        )

        print(
            f"{'Node type':<25}"
            f"{'Distance':<12}"
            f"{'Winner':>10}"
            f"{'Loser':>10}"
            f"{'Diff':>10}"
        )

        print(
            "-" * 70
        )

        for row in rows:

            print(
                f"{row[0]:<25}"
                f"{row[1]:<12}"
                f"{row[2]:>10}"
                f"{row[3]:>10}"
                f"{row[4]:>10}"
            )


# ============================================================
# COMPACT SUMMARY
# ============================================================

def print_winner_context_summary(
    results,
):

    print()
    print(
        "=" * 100
    )

    print(
        "WINNER CONTEXT SUMMARY"
    )

    print(
        "=" * 100
    )

    for outcome in (
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT",
    ):

        records = results.get(
            outcome,
            []
        )

        if not records:
            continue

        winner_position = (
            _aggregate_dimension(
                records,
                "winner_context",
                "position",
            )
        )

        loser_position = (
            _aggregate_dimension(
                records,
                "loser_context",
                "position",
            )
        )

        winner_distance = (
            _aggregate_dimension(
                records,
                "winner_context",
                "distance_bucket",
            )
        )

        loser_distance = (
            _aggregate_dimension(
                records,
                "loser_context",
                "distance_bucket",
            )
        )

        print()
        print(
            outcome
        )

        print(
            "-" * 60
        )

        print(
            "Winner BEFORE:",
            winner_position.get(
                "BEFORE",
                0
            )
        )

        print(
            "Winner AFTER:",
            winner_position.get(
                "AFTER",
                0
            )
        )

        print(
            "Loser BEFORE:",
            loser_position.get(
                "BEFORE",
                0
            )
        )

        print(
            "Loser AFTER:",
            loser_position.get(
                "AFTER",
                0
            )
        )

        print()

        print(
            "Winner local (1-3):",
            (
                winner_distance.get(
                    "1",
                    0
                )
                +
                winner_distance.get(
                    "2-3",
                    0
                )
            )
        )

        print(
            "Winner distant (>5):",
            winner_distance.get(
                ">5",
                0
            )
        )

        print(
            "Loser local (1-3):",
            (
                loser_distance.get(
                    "1",
                    0
                )
                +
                loser_distance.get(
                    "2-3",
                    0
                )
            )
        )

        print(
            "Loser distant (>5):",
            loser_distance.get(
                ">5",
                0
            )
        )