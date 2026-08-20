from collections import Counter, defaultdict


def analyze_seed_localization(samples):
    """
    Analyze whether seed_nodes correctly correspond
    to changed lines in the CFG.
    """

    print("=" * 80)
    print("SEED LOCALIZATION VERIFICATION")
    print("=" * 80)

    total_samples = 0
    total_seed_nodes = 0

    exact_matches = 0
    near_matches = 0
    no_matches = 0

    node_type_counts = Counter()
    node_type_by_label = defaultdict(Counter)

    match_by_label = defaultdict(Counter)

    worst_samples = []

    for sample in samples:

        if sample.cfg is None:
            continue

        if not sample.seed_lines:
            continue

        total_samples += 1

        #
        # Build lookup:
        #
        # CFG node ID -> node
        #
        node_lookup = {
            node.node_id: node
            for node in sample.cfg["nodes"]
        }

        sample_exact = 0
        sample_near = 0
        sample_none = 0

        for node_id in sample.seed_nodes:

            total_seed_nodes += 1

            node = node_lookup.get(node_id)

            #
            # Seed node does not exist in CFG.
            #
            if node is None:

                no_matches += 1
                sample_none += 1

                match_by_label[sample.label]["none"] += 1

                continue

            node_line = getattr(
                node,
                "lineno",
                None
            )

            node_type = getattr(
                node,
                "node_type",
                "UNKNOWN"
            )

            #
            # Record node type.
            #
            node_type_counts[node_type] += 1

            node_type_by_label[
                sample.label
            ][node_type] += 1

            #
            # Exact line match.
            #
            if node_line in sample.seed_lines:

                exact_matches += 1
                sample_exact += 1

                match_by_label[
                    sample.label
                ]["exact"] += 1

            #
            # Near line match.
            #
            elif (
                node_line is not None
                and any(
                    abs(node_line - seed_line) <= 1
                    for seed_line in sample.seed_lines
                )
            ):

                near_matches += 1
                sample_near += 1

                match_by_label[
                    sample.label
                ]["near"] += 1

            #
            # No line match.
            #
            else:

                no_matches += 1
                sample_none += 1

                match_by_label[
                    sample.label
                ]["none"] += 1

        #
        # Save suspicious samples.
        #
        if sample_none > 0:

            worst_samples.append({

                "repo": sample.repo,
                "file": sample.file_path,
                "label": sample.label,

                "seed_lines": sample.seed_lines,

                "seed_nodes": sample.seed_nodes,

                "exact": sample_exact,
                "near": sample_near,
                "none": sample_none

            })

    #
    # Print summary.
    #
    print()
    print("=" * 80)
    print("SEED LOCALIZATION DIAGNOSTICS")
    print("=" * 80)

    print(f"Samples analyzed : {total_samples}")
    print(f"Total seed nodes : {total_seed_nodes}")

    if total_seed_nodes > 0:

        print()
        print("## Seed Line Matching")

        print(
            f"Exact changed line : "
            f"{exact_matches} "
            f"({exact_matches / total_seed_nodes * 100:.2f}%)"
        )

        print(
            f"Near changed line  : "
            f"{near_matches} "
            f"({near_matches / total_seed_nodes * 100:.2f}%)"
        )

        print(
            f"No line match      : "
            f"{no_matches} "
            f"({no_matches / total_seed_nodes * 100:.2f}%)"
        )

    #
    # Node types.
    #
    print()
    print("## Seed Node Type Distribution")

    for node_type, count in node_type_counts.most_common():

        print(
            f"{str(node_type):<25} "
            f"{count}"
        )

    #
    # Node types by label.
    #
    print()
    print("## Seed Node Types By Label")

    for label in sorted(node_type_by_label):

        print()
        print(f"Label {label}")

        for node_type, count in (
            node_type_by_label[label]
            .most_common()
        ):

            print(
                f"{str(node_type):<25} "
                f"{count}"
            )

    #
    # Matching by label.
    #
    print()
    print("## Seed Line Matching By Label")

    for label in sorted(match_by_label):

        counts = match_by_label[label]

        total = sum(counts.values())

        print()
        print(f"Label {label}")

        for match_type in [

            "exact",
            "near",
            "none"

        ]:

            count = counts[match_type]

            percentage = (
                count / total * 100
                if total > 0
                else 0
            )

            print(
                f"{match_type:<15} "
                f"{count:>6} "
                f"({percentage:6.2f}%)"
            )

    #
    # Worst samples.
    #
    if worst_samples:

        worst_samples.sort(
            key=lambda x: x["none"],
            reverse=True
        )

        print()
        print("=" * 80)
        print("WORST SEED LOCALIZATION")
        print("=" * 80)

        for item in worst_samples[:20]:

            print()

            print(
                f"Label : {item['label']}"
            )

            print(
                f"Repo  : {item['repo']}"
            )

            print(
                f"File  : {item['file']}"
            )

            print(
                f"Seed lines : "
                f"{item['seed_lines']}"
            )

            print(
                f"Seed nodes : "
                f"{item['seed_nodes']}"
            )

            print(
                f"Exact={item['exact']} "
                f"Near={item['near']} "
                f"None={item['none']}"
            )

    return {

        "samples": total_samples,

        "seed_nodes": total_seed_nodes,

        "exact": exact_matches,

        "near": near_matches,

        "none": no_matches,

        "node_types": node_type_counts,

        "node_types_by_label":
            node_type_by_label,

        "matches_by_label":
            match_by_label,

        "worst_samples":
            worst_samples

    }