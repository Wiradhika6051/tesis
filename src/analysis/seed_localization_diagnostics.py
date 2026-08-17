from collections import Counter, defaultdict


class SeedLocalizationDiagnostics:

    def __init__(self):
        self.total_samples = 0

        self.seed_type_counter = Counter()
        self.seed_type_by_label = defaultdict(Counter)

        self.total_seeds = 0

        self.exact_matches = 0
        self.near_matches = 0
        self.no_line_matches = 0

        self.match_by_label = defaultdict(Counter)

        self.samples = []

    def analyze(self, samples):

        self.total_samples = len(samples)

        for sample_index, sample in enumerate(samples):

            seed_lines = getattr(
                sample,
                "seed_lines",
                None
            )

            seed_nodes = getattr(
                sample,
                "seed_nodes",
                None
            )

            if seed_lines is None:
                seed_lines = set()

            if seed_nodes is None:
                seed_nodes = set()

            seed_lines = set(seed_lines)

            #
            # Normalize seed node collection.
            #
            seed_nodes = list(seed_nodes)

            label = getattr(
                sample,
                "label",
                None
            )

            sample_info = {
                "index": sample_index,
                "repo": getattr(sample, "repo", None),
                "file": getattr(sample, "file_path", None),
                "label": label,
                "seed_lines": seed_lines,
                "seed_nodes": seed_nodes,
            }

            self.samples.append(sample_info)

            #
            # Analyze every seed node.
            #
            for node in seed_nodes:

                self.total_seeds += 1

                node_type = getattr(
                    node,
                    "node_type",
                    "UNKNOWN"
                )

                self.seed_type_counter[
                    node_type
                ] += 1

                self.seed_type_by_label[
                    label
                ][node_type] += 1

                #
                # Determine whether the node line
                # corresponds to the changed line.
                #
                node_line = getattr(
                    node,
                    "line",
                    None
                )

                if node_line in seed_lines:

                    self.exact_matches += 1

                    self.match_by_label[
                        label
                    ]["exact"] += 1

                else:

                    #
                    # Check whether the seed is at least
                    # near a changed line.
                    #
                    if self._is_near_line(
                        node_line,
                        seed_lines
                    ):

                        self.near_matches += 1

                        self.match_by_label[
                            label
                        ]["near"] += 1

                    else:

                        self.no_line_matches += 1

                        self.match_by_label[
                            label
                        ]["none"] += 1

        self.print_summary()

    @staticmethod
    def _is_near_line(
        node_line,
        seed_lines,
        distance=3
    ):

        if node_line is None:
            return False

        if node_line < 0:
            return False

        for seed_line in seed_lines:

            if abs(
                node_line - seed_line
            ) <= distance:

                return True

        return False

    def print_summary(self):

        print()
        print("=" * 70)
        print("SEED LOCALIZATION DIAGNOSTICS")
        print("=" * 70)

        print(
            f"Samples analyzed : "
            f"{self.total_samples}"
        )

        print(
            f"Total seed nodes : "
            f"{self.total_seeds}"
        )

        #
        # --------------------------------------------------
        # Match statistics
        # --------------------------------------------------
        #

        print()
        print("## Seed Line Matching")

        if self.total_seeds > 0:

            exact = (
                self.exact_matches
                / self.total_seeds
                * 100
            )

            near = (
                self.near_matches
                / self.total_seeds
                * 100
            )

            none = (
                self.no_line_matches
                / self.total_seeds
                * 100
            )

        else:

            exact = 0
            near = 0
            none = 0

        print(
            f"Exact changed line : "
            f"{self.exact_matches} "
            f"({exact:.2f}%)"
        )

        print(
            f"Near changed line  : "
            f"{self.near_matches} "
            f"({near:.2f}%)"
        )

        print(
            f"No line match      : "
            f"{self.no_line_matches} "
            f"({none:.2f}%)"
        )

        #
        # --------------------------------------------------
        # Node type distribution
        # --------------------------------------------------
        #

        print()
        print("## Seed Node Type Distribution")

        for node_type, count in (
            self.seed_type_counter.most_common()
        ):

            percentage = (
                count
                / self.total_seeds
                * 100
                if self.total_seeds
                else 0
            )

            print(
                f"{node_type:<20} "
                f"{count:>6} "
                f"({percentage:>6.2f}%)"
            )

        #
        # --------------------------------------------------
        # Distribution by label
        # --------------------------------------------------
        #

        print()
        print("## Seed Node Types By Label")

        for label in sorted(
            self.seed_type_by_label.keys(),
            key=lambda x: str(x)
        ):

            print()
            print(f"Label {label}")

            counter = (
                self.seed_type_by_label[
                    label
                ]
            )

            total = sum(
                counter.values()
            )

            for node_type, count in (
                counter.most_common()
            ):

                percentage = (
                    count / total * 100
                    if total
                    else 0
                )

                print(
                    f"{node_type:<20} "
                    f"{count:>6} "
                    f"({percentage:>6.2f}%)"
                )

        #
        # --------------------------------------------------
        # Matching by label
        # --------------------------------------------------
        #

        print()
        print("## Seed Line Matching By Label")

        for label in sorted(
            self.match_by_label.keys(),
            key=lambda x: str(x)
        ):

            counter = (
                self.match_by_label[
                    label
                ]
            )

            total = sum(
                counter.values()
            )

            print()
            print(f"Label {label}")

            for category in (
                "exact",
                "near",
                "none"
            ):

                count = counter[
                    category
                ]

                percentage = (
                    count / total * 100
                    if total
                    else 0
                )

                print(
                    f"{category:<10} "
                    f"{count:>6} "
                    f"({percentage:>6.2f}%)"
                )

        #
        # --------------------------------------------------
        # Suspicious seed types
        # --------------------------------------------------
        #

        self._print_suspicious_seed_types()

    def _print_suspicious_seed_types(self):

        suspicious_types = {
            "Import",
            "ImportFrom",
            "ENTRY",
            "EXIT",
            "MERGE",
            "LOOP_EXIT",
        }

        suspicious = {
            node_type: count
            for node_type, count
            in self.seed_type_counter.items()
            if node_type in suspicious_types
        }

        print()
        print("## Potentially Broad Seed Types")

        if not suspicious:

            print("None found.")

            return

        total = sum(
            suspicious.values()
        )

        for node_type, count in sorted(
            suspicious.items(),
            key=lambda x: x[1],
            reverse=True
        ):

            percentage = (
                count / self.total_seeds * 100
                if self.total_seeds
                else 0
            )

            print(
                f"{node_type:<20} "
                f"{count:>6} "
                f"({percentage:>6.2f}%)"
            )

    def print_worst_samples(
        self,
        limit=20
    ):

        #
        # Samples with the largest number of
        # seed nodes that do NOT correspond to
        # changed lines.
        #
        ranked = []

        for sample in self.samples:

            seed_lines = sample[
                "seed_lines"
            ]

            seed_nodes = sample[
                "seed_nodes"
            ]

            bad_nodes = []

            for node in seed_nodes:

                node_line = getattr(
                    node,
                    "line",
                    None
                )

                if node_line in seed_lines:
                    continue

                if self._is_near_line(
                    node_line,
                    seed_lines
                ):
                    continue

                bad_nodes.append(
                    node
                )

            ranked.append(
                (
                    len(bad_nodes),
                    sample,
                    bad_nodes
                )
            )

        ranked.sort(
            key=lambda x: x[0],
            reverse=True
        )

        print()
        print("=" * 70)
        print("WORST SEED LOCALIZATION")
        print("=" * 70)

        for rank, (
            bad_count,
            sample,
            bad_nodes
        ) in enumerate(
            ranked[:limit],
            start=1
        ):

            if bad_count == 0:
                break

            print()
            print("-" * 70)

            print(
                f"#{rank}"
            )

            print(
                f"Repo  : {sample['repo']}"
            )

            print(
                f"File  : {sample['file']}"
            )

            print(
                f"Label : {sample['label']}"
            )

            print(
                f"Changed lines : "
                f"{sorted(sample['seed_lines'])}"
            )

            print(
                f"Seed nodes : "
                f"{len(sample['seed_nodes'])}"
            )

            print(
                f"Unmatched seeds : "
                f"{bad_count}"
            )

            print()

            for node in bad_nodes:

                node_id = getattr(
                    node,
                    "node_id",
                    "?"
                )

                node_line = getattr(
                    node,
                    "line",
                    "?"
                )

                node_type = getattr(
                    node,
                    "node_type",
                    "UNKNOWN"
                )

                text = getattr(
                    node,
                    "text",
                    ""
                )

                text = str(
                    text
                ).replace(
                    "\n",
                    " "
                )

                if len(text) > 120:
                    text = text[:117] + "..."

                print(
                    f"Node {node_id:<4} "
                    f"| Line {node_line:<4} "
                    f"| {node_type:<15} "
                    f"| {text}"
                )