import statistics
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

from tqdm import tqdm


# ============================================================
# Dataset / Graph Statistics
# ============================================================
def analyze_graph_statistics(
    dataset
):

    print("\n" + "=" * 70)
    print("GRAPH STATISTICS DIAGNOSTICS")
    print("=" * 70)

    if not dataset:

        print("Dataset is empty.")

        return

    grouped = defaultdict(list)

    missing_graphs = []

    for sample in dataset:

        graph = sample.graph

        #
        # Graph was not generated.
        #
        if graph is None:

            missing_graphs.append(
                sample
            )

            continue

        num_nodes = graph.num_nodes

        num_edges = (
            graph.edge_index.shape[1]
            if graph.edge_index is not None
            else 0
        )

        total_tokens = sum(
            len(tokens)
            for tokens in graph.node_tokens
        )

        grouped[
            sample.label
        ].append({

            "nodes":
                num_nodes,

            "edges":
                num_edges,

            "tokens":
                total_tokens
        })

    #
    # Report missing graphs.
    #
    print(
        f"Total samples      : "
        f"{len(dataset)}"
    )

    print(
        f"Valid graphs       : "
        f"{len(dataset) - len(missing_graphs)}"
    )

    print(
        f"Missing graphs     : "
        f"{len(missing_graphs)}"
    )

    if missing_graphs:

        print(
            "\n## Samples Without Graph"
        )

        for sample in missing_graphs[:10]:

            print(
                f"Label={sample.label} | "
                f"{sample.repo} | "
                f"{sample.file_path}"
            )

    #
    # Statistics by label.
    #
    for label in sorted(grouped):

        group = grouped[label]

        print(
            "\n" + "-" * 50
        )

        print(
            f"Label {label}"
        )

        print(
            "-" * 50
        )

        print(
            f"Samples        : "
            f"{len(group)}"
        )

        print(
            f"Average nodes  : "
            f"{statistics.mean(x['nodes'] for x in group):.2f}"
        )

        print(
            f"Average edges  : "
            f"{statistics.mean(x['edges'] for x in group):.2f}"
        )

        print(
            f"Average tokens : "
            f"{statistics.mean(x['tokens'] for x in group):.2f}"
        )

        print(
            f"Min nodes      : "
            f"{min(x['nodes'] for x in group)}"
        )

        print(
            f"Max nodes      : "
            f"{max(x['nodes'] for x in group)}"
        )

        print(
            f"Min edges      : "
            f"{min(x['edges'] for x in group)}"
        )

        print(
            f"Max edges      : "
            f"{max(x['edges'] for x in group)}"
        )

# ============================================================
# Node Token Length Statistics
# ============================================================

def analyze_node_token_lengths(
    dataset
):
    """
    Analyze token count per CFG node grouped by label.
    """

    print("\n" + "=" * 70)
    print("NODE TOKEN LENGTH DIAGNOSTICS")
    print("=" * 70)

    grouped = defaultdict(list)

    for sample in dataset:

        if sample.graph is None:
            continue
        for tokens in sample.graph.node_tokens:

            grouped[sample.label].append(
                len(tokens)
            )

    for label in sorted(grouped):

        lengths = grouped[label]

        print("\n" + "-" * 50)
        print(f"Label {label}")
        print("-" * 50)

        print(
            f"Nodes analyzed : {len(lengths)}"
        )

        print(
            f"Average        : "
            f"{statistics.mean(lengths):.2f}"
        )

        print(
            f"Median         : "
            f"{statistics.median(lengths):.2f}"
        )

        print(
            f"Minimum        : "
            f"{min(lengths)}"
        )

        print(
            f"Maximum        : "
            f"{max(lengths)}"
        )

        print(
            f"> 50 tokens    : "
            f"{sum(x > 50 for x in lengths)}"
        )

        print(
            f"> 100 tokens   : "
            f"{sum(x > 100 for x in lengths)}"
        )

        print(
            f"> 200 tokens   : "
            f"{sum(x > 200 for x in lengths)}"
        )


# ============================================================
# CFG Node Type Distribution
# ============================================================

def analyze_node_types(
    dataset
):
    """
    Count CFG node types for each label.
    """

    print("\n" + "=" * 70)
    print("CFG NODE TYPE DIAGNOSTICS")
    print("=" * 70)

    grouped = defaultdict(Counter)

    for sample in dataset:

        for node_type in sample.graph.node_types:

            if torch.is_tensor(node_type):

                node_type = node_type.item()

            grouped[sample.label][
                int(node_type)
            ] += 1

    labels = sorted(grouped)

    all_types = sorted({

        node_type

        for counter in grouped.values()

        for node_type in counter

    })

    print()

    header = (
        f"{'Node Type':>12}"
        +
        "".join(
            f"{'Label ' + str(label):>12}"
            for label in labels
        )
    )

    print(header)
    print("-" * len(header))

    for node_type in all_types:

        print(
            f"{node_type:>12}"
            +
            "".join(
                f"{grouped[label][node_type]:>12}"
                for label in labels
            )
        )


# ============================================================
# Token Distribution
# ============================================================

def analyze_token_distribution(
    dataset,
    padding_id=0,
    top_k=30
):
    """
    Analyze token frequencies separately for each label.
    """

    print("\n" + "=" * 70)
    print("TOKEN DISTRIBUTION DIAGNOSTICS")
    print("=" * 70)

    grouped = {
        0: Counter(),
        1: Counter()
    }

    for sample in dataset:

        label = sample.label

        for node_tokens in sample.graph.node_tokens:

            for token in node_tokens:

                if torch.is_tensor(token):

                    token = token.item()

                token = int(token)

                if token == padding_id:
                    continue

                grouped[label][token] += 1

    for label in sorted(grouped):

        counter = grouped[label]

        print("\n" + "-" * 50)
        print(f"Label {label}")
        print("-" * 50)

        print(
            f"Unique tokens : "
            f"{len(counter)}"
        )

        print(
            f"Total tokens  : "
            f"{sum(counter.values())}"
        )

        print(
            f"Top {top_k} tokens:"
        )

        for token, count in counter.most_common(top_k):

            print(
                f"  {token:>6} : {count}"
            )


# ============================================================
# Token Overlap
# ============================================================

def analyze_token_overlap(
    dataset,
    padding_id=0
):
    """
    Compare token vocabulary overlap between labels.
    """

    print("\n" + "=" * 70)
    print("TOKEN VOCABULARY OVERLAP")
    print("=" * 70)

    vocab = {
        0: set(),
        1: set()
    }

    for sample in dataset:

        label = sample.label

        for node_tokens in sample.graph.node_tokens:

            for token in node_tokens:

                if torch.is_tensor(token):

                    token = token.item()

                token = int(token)

                if token != padding_id:

                    vocab[label].add(token)

    intersection = (
        vocab[0] &
        vocab[1]
    )

    only_0 = (
        vocab[0] -
        vocab[1]
    )

    only_1 = (
        vocab[1] -
        vocab[0]
    )

    union = (
        vocab[0] |
        vocab[1]
    )

    print(
        f"Label 0 unique tokens : "
        f"{len(vocab[0])}"
    )

    print(
        f"Label 1 unique tokens : "
        f"{len(vocab[1])}"
    )

    print(
        f"Shared tokens         : "
        f"{len(intersection)}"
    )

    print(
        f"Only label 0          : "
        f"{len(only_0)}"
    )

    print(
        f"Only label 1          : "
        f"{len(only_1)}"
    )

    if union:

        print(
            f"Vocabulary overlap   : "
            f"{len(intersection) / len(union):.2%}"
        )


# ============================================================
# Sample Graph Inspection
# ============================================================

def inspect_samples(
    dataset,
    count=10
):
    """
    Print detailed information about several graphs.
    """

    print("\n" + "=" * 70)
    print("SAMPLE GRAPH INSPECTION")
    print("=" * 70)

    for index, sample in enumerate(
        dataset[:count]
    ):

        graph = sample.graph

        print("\n" + "-" * 70)

        print(
            f"Sample {index}"
        )

        print(
            f"Label : {sample.label}"
        )

        print(
            f"Repo  : {sample.repo}"
        )

        print(
            f"File  : {sample.file_path}"
        )

        print(
            f"Nodes : {graph.num_nodes}"
        )

        print(
            f"Edges : "
            f"{graph.edge_index.shape[1]}"
        )

        print(
            "Node types:"
        )

        print(
            graph.node_types.tolist()
        )

        print(
            "Token lengths:"
        )

        print([
            len(tokens)
            for tokens in graph.node_tokens
        ])


# ============================================================
# Tiny Dataset Overfit Test
# ============================================================

def overfit_small_dataset(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epochs=100
):
    """
    Attempt to overfit a very small dataset.

    This is a diagnostic, not a production training routine.

    If the model cannot reach very high training accuracy
    on a tiny dataset, the representation/model pipeline
    likely has a fundamental problem.
    """

    print("\n" + "=" * 70)
    print("TINY DATASET OVERFIT TEST")
    print("=" * 70)

    model.to(device)

    model.train()

    for epoch in range(
        1,
        epochs + 1
    ):

        total_loss = 0.0

        total_samples = 0

        correct = 0

        for graph, labels in loader:

            graph = graph.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                graph
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            batch_size = (
                labels.size(0)
            )

            total_loss += (
                loss.item()
                *
                batch_size
            )

            total_samples += (
                batch_size
            )

            correct += (
                predictions == labels
            ).sum().item()

        accuracy = (
            correct /
            total_samples
        )

        average_loss = (
            total_loss /
            total_samples
        )

        if (
            epoch == 1
            or
            epoch % 10 == 0
            or
            accuracy >= 0.95
        ):

            print(
                f"Epoch {epoch:3d} | "
                f"Loss={average_loss:.4f} | "
                f"Accuracy={accuracy:.4f}"
            )

        if accuracy >= 0.99:

            print(
                "\n✓ Model successfully "
                "overfit the tiny dataset."
            )

            return {
                "loss":
                    average_loss,

                "accuracy":
                    accuracy,

                "success":
                    True
            }

    print(
        "\n✗ Model failed to overfit "
        "the tiny dataset."
    )

    return {
        "loss":
            average_loss,

        "accuracy":
            accuracy,

        "success":
            False
    }

import statistics
from collections import Counter


def _get_graph(sample):
    """
    Try to obtain the graph representation used by the model.

    Returns:
        graph object or None
    """

    return getattr(sample, "graph", None)


def analyze_graph_availability(dataset):
    """
    Check whether samples actually contain the graph representation
    expected by the model.
    """

    print("\n" + "=" * 70)
    print("GRAPH AVAILABILITY DIAGNOSTICS")
    print("=" * 70)

    total = len(dataset)

    valid = []
    missing = []

    for sample in dataset:

        graph = _get_graph(sample)

        if graph is None:
            missing.append(sample)

        else:
            valid.append(sample)

    print(f"Total samples : {total}")
    print(
        f"Valid graphs  : "
        f"{len(valid)} "
        f"({len(valid) / total:.2%})"
    )

    print(
        f"Missing graphs: "
        f"{len(missing)} "
        f"({len(missing) / total:.2%})"
    )

    # ---------------------------------------------------------
    # Missing graphs by label
    # ---------------------------------------------------------

    print("\n## Missing Graphs By Label")

    missing_labels = Counter(
        sample.label
        for sample in missing
    )

    for label, count in sorted(
        missing_labels.items()
    ):

        print(
            f"Label {label}: "
            f"{count}"
        )

    # ---------------------------------------------------------
    # Valid graphs by label
    # ---------------------------------------------------------

    print("\n## Valid Graphs By Label")

    valid_labels = Counter(
        sample.label
        for sample in valid
    )

    for label, count in sorted(
        valid_labels.items()
    ):

        print(
            f"Label {label}: "
            f"{count}"
        )

    return valid, missing


def analyze_graph_statistics(dataset):
    """
    Analyze basic statistics of the actual PyG graph
    attached to each sample.
    """

    print("\n" + "=" * 70)
    print("GRAPH STATISTICS")
    print("=" * 70)

    valid = [
        sample
        for sample in dataset
        if _get_graph(sample) is not None
    ]

    if not valid:

        print("No valid graphs found.")

        return

    print(
        f"Samples analyzed : "
        f"{len(valid)}"
    )

    # ---------------------------------------------------------
    # Overall
    # ---------------------------------------------------------

    node_counts = []
    edge_counts = []

    for sample in valid:

        graph = sample.graph

        node_counts.append(
            graph.num_nodes
        )

        edge_counts.append(
            graph.edge_index.shape[1]
            if graph.edge_index is not None
            else 0
        )

    print("\n## Overall")

    print(
        f"Average Nodes : "
        f"{statistics.mean(node_counts):.2f}"
    )

    print(
        f"Min Nodes     : "
        f"{min(node_counts)}"
    )

    print(
        f"Max Nodes     : "
        f"{max(node_counts)}"
    )

    print(
        f"Average Edges : "
        f"{statistics.mean(edge_counts):.2f}"
    )

    print(
        f"Min Edges     : "
        f"{min(edge_counts)}"
    )

    print(
        f"Max Edges     : "
        f"{max(edge_counts)}"
    )

    # ---------------------------------------------------------
    # By label
    # ---------------------------------------------------------

    labels = sorted(
        set(sample.label for sample in valid)
    )

    print("\n## By Label")

    for label in labels:

        group = [
            sample
            for sample in valid
            if sample.label == label
        ]

        nodes = [
            sample.graph.num_nodes
            for sample in group
        ]

        edges = [
            (
                sample.graph.edge_index.shape[1]
                if sample.graph.edge_index is not None
                else 0
            )
            for sample in group
        ]

        print(f"\nLabel {label}")

        print(
            f"Samples       : "
            f"{len(group)}"
        )

        print(
            f"Average Nodes : "
            f"{statistics.mean(nodes):.2f}"
        )

        print(
            f"Average Edges : "
            f"{statistics.mean(edges):.2f}"
        )

        print(
            f"Min Nodes     : "
            f"{min(nodes)}"
        )

        print(
            f"Max Nodes     : "
            f"{max(nodes)}"
        )


def analyze_node_features(dataset):
    """
    Inspect the node feature tensor given to the model.
    """

    print("\n" + "=" * 70)
    print("NODE FEATURE DIAGNOSTICS")
    print("=" * 70)

    valid = [
        sample
        for sample in dataset
        if _get_graph(sample) is not None
        and getattr(sample.graph, "x", None) is not None
    ]

    if not valid:

        print("No graphs with node features found.")

        return

    print(
        f"Graphs analyzed : "
        f"{len(valid)}"
    )

    feature_dimensions = Counter()

    feature_counts = []

    for sample in valid:

        x = sample.graph.x

        feature_dimensions[
            tuple(x.shape)
        ] += 1

        feature_counts.append(
            x.shape[0]
        )

    print("\n## Feature Shapes")

    for shape, count in feature_dimensions.most_common():

        print(
            f"{shape} : "
            f"{count}"
        )

    print("\n## Feature Statistics")

    print(
        f"Average feature rows : "
        f"{statistics.mean(feature_counts):.2f}"
    )

    # ---------------------------------------------------------
    # Check whether features are constant
    # ---------------------------------------------------------

    constant_graphs = 0

    for sample in valid:

        x = sample.graph.x

        if x.numel() == 0:
            constant_graphs += 1
            continue

        if x.ndim >= 2:

            if x.shape[0] > 1:

                if x.float().std().item() == 0:

                    constant_graphs += 1

    print(
        f"Graphs with constant "
        f"features : {constant_graphs}"
    )


def analyze_graph_node_types(dataset):
    """
    Analyze node type distribution if node types
    are available on the graph.
    """

    print("\n" + "=" * 70)
    print("GRAPH NODE TYPE DIAGNOSTICS")
    print("=" * 70)

    valid = [
        sample
        for sample in dataset
        if _get_graph(sample) is not None
        and hasattr(sample.graph, "node_types")
    ]

    if not valid:

        print(
            "No graphs containing "
            "`node_types` found."
        )

        return

    overall = Counter()
    by_label = {}

    for sample in valid:

        node_types = sample.graph.node_types

        label = sample.label

        if label not in by_label:
            by_label[label] = Counter()

        for node_type in node_types:

            if hasattr(node_type, "item"):

                node_type = node_type.item()

            overall[node_type] += 1
            by_label[label][node_type] += 1

    print("\n## Overall")

    for node_type, count in overall.most_common():

        print(
            f"{node_type}: "
            f"{count}"
        )

    print("\n## By Label")

    for label in sorted(by_label):

        print(
            f"\nLabel {label}"
        )

        for node_type, count in (
            by_label[label].most_common()
        ):

            print(
                f"{node_type}: "
                f"{count}"
            )

def analyze_class_graph_statistics(
    dataset
):

    print("\n" + "=" * 70)
    print("CLASS-WISE GRAPH STATISTICS")
    print("=" * 70)

    groups = {}

    for sample in dataset:

        if sample.graph is None:
            continue

        label = sample.label

        groups.setdefault(
            label,
            []
        ).append(sample)

    for label in sorted(groups):

        samples = groups[label]

        node_counts = [
            sample.graph.num_nodes
            for sample in samples
        ]

        edge_counts = [

            sample.graph.edge_index.shape[1]

            for sample in samples

        ]

        token_counts = [

            sample.graph.node_tokens
                .ne(0)
                .sum()
                .item()

            for sample in samples

        ]

        print(
            f"\nLabel {label}"
        )

        print(
            f"Samples          : "
            f"{len(samples)}"
        )

        print(
            f"Avg nodes        : "
            f"{statistics.mean(node_counts):.2f}"
        )

        print(
            f"Avg edges        : "
            f"{statistics.mean(edge_counts):.2f}"
        )

        print(
            f"Avg token IDs    : "
            f"{statistics.mean(token_counts):.2f}"
        )

        print(
            f"Min nodes        : "
            f"{min(node_counts)}"
        )

        print(
            f"Max nodes        : "
            f"{max(node_counts)}"
        )

        print(
            f"Min edges        : "
            f"{min(edge_counts)}"
        )

        print(
            f"Max edges        : "
            f"{max(edge_counts)}"
        )