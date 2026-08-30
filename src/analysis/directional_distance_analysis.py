from collections import defaultdict
import statistics

from src.analysis.slice_analysis import get_slice_nodes


def _sample_id(sample):

    return (
        sample.repo,
        sample.parent_commit,
        sample.file_path,
        sample.label
    )


def _build_adjacency(cfg):

    adjacency = defaultdict(list)

    for source, target in cfg["edges"]:

        adjacency[source].append(target)

    return adjacency


def _build_reverse_adjacency(cfg):

    reverse = defaultdict(list)

    for source, target in cfg["edges"]:

        reverse[target].append(source)

    return reverse


def _shortest_distances(
    adjacency,
    seeds
):
    """
    Calculate shortest CFG distance from any seed
    to every reachable node.
    """

    distances = {}

    queue = []

    for seed in seeds:

        if seed in distances:
            continue

        distances[seed] = 0
        queue.append(seed)

    position = 0

    while position < len(queue):

        current = queue[position]

        position += 1

        current_distance = distances[
            current
        ]

        for neighbor in adjacency.get(
            current,
            []
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


def _distance_statistics(
    node_ids,
    distances
):

    values = [

        distances[node_id]

        for node_id in node_ids

        if node_id in distances

    ]

    if not values:

        return {

            "count": 0,
            "average": 0.0,
            "minimum": 0,
            "maximum": 0

        }

    return {

        "count":
            len(values),

        "average":
            statistics.mean(values),

        "minimum":
            min(values),

        "maximum":
            max(values)

    }


def _distance_buckets(
    node_ids,
    distances
):

    buckets = {

        "1": 0,
        "2-3": 0,
        "4-5": 0,
        ">5": 0

    }

    for node_id in node_ids:

        distance = distances.get(
            node_id
        )

        if distance is None:
            continue

        #
        # Seed itself is distance 0.
        # We are interested in nodes
        # surrounding the seed.
        #
        if distance == 0:
            continue

        if distance == 1:

            buckets["1"] += 1

        elif distance <= 3:

            buckets["2-3"] += 1

        elif distance <= 5:

            buckets["4-5"] += 1

        else:

            buckets[">5"] += 1

    return buckets


def _analyze_sample(
    sample
):

    cfg = sample.cfg

    seed_nodes = set(
        sample.seed_nodes
    )

    function_nodes = set(
        sample.function_nodes
    )

    if not seed_nodes:
        return None

    if not function_nodes:
        return None

    forward_nodes = set(
        get_slice_nodes(

            cfg,

            seed_nodes,

            function_nodes,

            forward=True

        )
    )

    backward_nodes = set(
        get_slice_nodes(

            cfg,

            seed_nodes,

            function_nodes,

            forward=False

        )
    )

    overlap = (
        forward_nodes
        &
        backward_nodes
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
    # Forward distance:
    #
    # distance from seed following CFG edges.
    #
    forward_adjacency = (
        _build_adjacency(
            cfg
        )
    )

    forward_distances = (
        _shortest_distances(

            forward_adjacency,

            seed_nodes

        )
    )

    #
    # Backward distance:
    #
    # distance from seed following reversed
    # CFG edges.
    #
    backward_adjacency = (
        _build_reverse_adjacency(
            cfg
        )
    )

    backward_distances = (
        _shortest_distances(

            backward_adjacency,

            seed_nodes

        )
    )

    return {

        "sample_id":
            _sample_id(sample),

        "seed_count":
            len(seed_nodes),

        "function_size":
            len(function_nodes),

        "forward_size":
            len(forward_nodes),

        "backward_size":
            len(backward_nodes),

        "overlap":
            len(overlap),

        "forward_only":
            len(forward_only),

        "backward_only":
            len(backward_only),

        #
        # Forward-only distance.
        #
        "forward_only_distance":
            _distance_statistics(

                forward_only,

                forward_distances

            ),

        #
        # Backward-only distance.
        #
        "backward_only_distance":
            _distance_statistics(

                backward_only,

                backward_distances

            ),

        #
        # Distance buckets.
        #
        "forward_only_buckets":
            _distance_buckets(

                forward_only,

                forward_distances

            ),

        "backward_only_buckets":
            _distance_buckets(

                backward_only,

                backward_distances

            )

    }


def analyze_directional_distances(
    records,
    samples
):

    sample_lookup = {

        _sample_id(sample):
            sample

        for sample in samples

    }

    results = []

    for record in records:

        sample = sample_lookup.get(
            record["sample_id"]
        )

        if sample is None:
            continue

        distance_data = _analyze_sample(
            sample
        )

        if distance_data is None:
            continue

        distance_data[
            "outcome"
        ] = record[
            "outcome"
        ]

        distance_data[
            "label"
        ] = record[
            "label"
        ]

        distance_data[
            "forward_prediction"
        ] = record[
            "forward_prediction"
        ]

        distance_data[
            "backward_prediction"
        ] = record[
            "backward_prediction"
        ]

        results.append(
            distance_data
        )

    return results


def _mean(
    records,
    key
):

    values = []

    for record in records:

        value = record[key]

        if isinstance(
            value,
            dict
        ):

            value = value[
                "average"
            ]

        values.append(
            value
        )

    if not values:

        return 0.0

    return statistics.mean(
        values
    )


def print_directional_distance_analysis(
    results
):

    grouped = defaultdict(list)

    for record in results:

        grouped[
            record["outcome"]
        ].append(
            record
        )

    print()
    print("=" * 80)
    print(
        "CFG DISTANCE BY PREDICTION OUTCOME"
    )
    print("=" * 80)

    outcomes = [

        "BOTH_CORRECT",
        "BOTH_WRONG",
        "FORWARD_CORRECT",
        "BACKWARD_CORRECT"

    ]

    for outcome in outcomes:

        group = grouped.get(
            outcome,
            []
        )

        print()
        print(outcome)
        print("-" * 60)

        print(
            "Samples:",
            len(group)
        )

        if not group:
            continue

        print()

        print(
            f"Forward-only average distance : "
            f"{_mean(group, 'forward_only_distance'):.2f}"
        )

        print(
            f"Backward-only average distance: "
            f"{_mean(group, 'backward_only_distance'):.2f}"
        )

        #
        # Minimum distance.
        #

        forward_min = [

            r[
                "forward_only_distance"
            ]["minimum"]

            for r in group

            if r[
                "forward_only_distance"
            ]["count"] > 0

        ]

        backward_min = [

            r[
                "backward_only_distance"
            ]["minimum"]

            for r in group

            if r[
                "backward_only_distance"
            ]["count"] > 0

        ]

        print(
            f"Forward-only minimum distance : "
            f"{statistics.mean(forward_min):.2f}"
            if forward_min
            else
            "Forward-only minimum distance : N/A"
        )

        print(
            f"Backward-only minimum distance: "
            f"{statistics.mean(backward_min):.2f}"
            if backward_min
            else
            "Backward-only minimum distance: N/A"
        )

        #
        # Maximum distance.
        #

        forward_max = [

            r[
                "forward_only_distance"
            ]["maximum"]

            for r in group

            if r[
                "forward_only_distance"
            ]["count"] > 0

        ]

        backward_max = [

            r[
                "backward_only_distance"
            ]["maximum"]

            for r in group

            if r[
                "backward_only_distance"
            ]["count"] > 0

        ]

        print(
            f"Forward-only maximum distance : "
            f"{statistics.mean(forward_max):.2f}"
            if forward_max
            else
            "Forward-only maximum distance : N/A"
        )

        print(
            f"Backward-only maximum distance: "
            f"{statistics.mean(backward_max):.2f}"
            if backward_max
            else
            "Backward-only maximum distance: N/A"
        )

        #
        # Distance buckets.
        #

        forward_buckets = defaultdict(int)
        backward_buckets = defaultdict(int)

        for record in group:

            for bucket, count in record[
                "forward_only_buckets"
            ].items():

                forward_buckets[
                    bucket
                ] += count

            for bucket, count in record[
                "backward_only_buckets"
            ].items():

                backward_buckets[
                    bucket
                ] += count

        print()
        print(
            "Forward-only distance buckets:"
        )

        for bucket in [
            "1",
            "2-3",
            "4-5",
            ">5"
        ]:

            print(
                f"  {bucket:<5}: "
                f"{forward_buckets[bucket]}"
            )

        print()
        print(
            "Backward-only distance buckets:"
        )

        for bucket in [
            "1",
            "2-3",
            "4-5",
            ">5"
        ]:

            print(
                f"  {bucket:<5}: "
                f"{backward_buckets[bucket]}"
            )

    #
    # --------------------------------------------------
    # Direct comparison.
    # --------------------------------------------------
    #

    forward_correct = grouped.get(
        "FORWARD_CORRECT",
        []
    )

    backward_correct = grouped.get(
        "BACKWARD_CORRECT",
        []
    )

    print()
    print("=" * 80)
    print(
        "FORWARD-CORRECT VS BACKWARD-CORRECT"
    )
    print("=" * 80)

    print()

    print(
        f"{'Metric':<35}"
        f"{'Forward correct':>20}"
        f"{'Backward correct':>20}"
    )

    print("-" * 75)

    metrics = [

        (
            "Forward-only avg distance",
            "forward_only_distance"
        ),

        (
            "Backward-only avg distance",
            "backward_only_distance"
        )

    ]

    for name, key in metrics:

        print(
            f"{name:<35}"
            f"{_mean(forward_correct, key):>20.2f}"
            f"{_mean(backward_correct, key):>20.2f}"
        )