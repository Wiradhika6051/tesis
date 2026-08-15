import statistics


class PruningDiagnostics:

    def __init__(self):

        self.samples = []

    def add_sample(
        self,
        label,
        function_nodes,
        seed_nodes,
        retained_nodes,
        sample_id=None,
    ):

        #
        # Convert to sets.
        #
        function_nodes = set(
            function_nodes
        )

        seed_nodes = set(
            seed_nodes
        )

        retained_nodes = set(
            retained_nodes
        )

        #
        # Localization diagnostics.
        #
        valid_seeds = (
            seed_nodes &
            function_nodes
        )

        outside_seeds = (
            seed_nodes -
            function_nodes
        )

        #
        # Node counts.
        #
        function_count = len(
            function_nodes
        )

        retained_count = len(
            retained_nodes
        )

        #
        # Retention.
        #
        if function_count > 0:

            retention = (
                retained_count /
                function_count
            )

        else:

            retention = 0.0

        #
        # Check whether retained
        # nodes actually belong to
        # the localized function.
        #
        retained_outside_function = (
            retained_nodes -
            function_nodes
        )

        self.samples.append({

            "sample_id":
                sample_id,

            "label":
                label,

            "function_nodes":
                function_count,

            "seed_nodes":
                len(seed_nodes),

            "valid_seeds":
                len(valid_seeds),

            "outside_seeds":
                len(outside_seeds),

            "retained_nodes":
                retained_count,

            "retention":
                retention,

            "retained_outside_function":
                len(
                    retained_outside_function
                ),

            "has_localization_mismatch":
                len(outside_seeds) > 0,

            "has_valid_seed":
                len(valid_seeds) > 0,
        })

    def summary(
        self
    ):

        if not self.samples:

            print(
                "No samples."
            )

            return

        n = len(
            self.samples
        )

        print(
            "=" * 70
        )

        print(
            "PRUNING DIAGNOSTICS"
        )

        print(
            "=" * 70
        )

        print(
            f"Samples analyzed : {n}"
        )

        #
        # --------------------------------------------------
        # Localization
        # --------------------------------------------------
        #

        mismatch = sum(
            s[
                "has_localization_mismatch"
            ]
            for s in self.samples
        )

        no_valid_seed = sum(
            not s[
                "has_valid_seed"
            ]
            for s in self.samples
        )

        print(
            "\n## Localization"
        )

        print(
            f"Samples with mismatch : "
            f"{mismatch} "
            f"({mismatch / n:.2%})"
        )

        print(
            f"Samples with no valid seed : "
            f"{no_valid_seed} "
            f"({no_valid_seed / n:.2%})"
        )

        outside_counts = [

            s[
                "outside_seeds"
            ]

            for s in self.samples

        ]

        print(
            f"Average outside seeds : "
            f"{statistics.mean(outside_counts):.2f}"
        )

        print(
            f"Maximum outside seeds : "
            f"{max(outside_counts)}"
        )

        #
        # --------------------------------------------------
        # Pruning
        # --------------------------------------------------
        #

        print(
            "\n## Overall"
        )

        self._print_group(
            self.samples
        )

        #
        # --------------------------------------------------
        # By label
        # --------------------------------------------------
        #

        labels = sorted(
            set(
                s["label"]
                for s in self.samples
            )
        )

        print(
            "\n## By Label"
        )

        for label in labels:

            group = [

                s

                for s in self.samples

                if s["label"] == label

            ]

            print(
                f"\nLabel {label}"
            )

            self._print_group(
                group
            )

        #
        # --------------------------------------------------
        # Label retention difference
        # --------------------------------------------------
        #

        if len(labels) == 2:

            grouped = {

                label: [

                    s

                    for s in self.samples

                    if s["label"] == label

                ]

                for label in labels

            }

            retention_a = statistics.mean(

                s["retention"]

                for s in grouped[
                    labels[0]
                ]

            )

            retention_b = statistics.mean(

                s["retention"]

                for s in grouped[
                    labels[1]
                ]

            )

            print(
                "\n## Label Difference"
            )

            print(
                f"Retention difference : "
                f"{abs(
                    retention_a -
                    retention_b
                ):.2%}"
            )

        #
        # --------------------------------------------------
        # Worst localization mismatches
        # --------------------------------------------------
        #

        print(
            "\n## Worst Localization Mismatches"
        )

        mismatches = sorted(

            [

                s

                for s in self.samples

                if s[
                    "has_localization_mismatch"
                ]

            ],

            key=lambda s:
                s["outside_seeds"],

            reverse=True
        )

        for s in mismatches[:10]:

            print(

                f"{s['sample_id']} | "

                f"Label={s['label']} | "

                f"Function="
                f"{s['function_nodes']} | "

                f"Seeds="
                f"{s['seed_nodes']} | "

                f"Valid="
                f"{s['valid_seeds']} | "

                f"Outside="
                f"{s['outside_seeds']}"

            )

        #
        # --------------------------------------------------
        # Lowest retention
        # --------------------------------------------------
        #

        print(
            "\n## Lowest Retention"
        )

        lowest = sorted(

            self.samples,

            key=lambda s:
                s["retention"]

        )

        for s in lowest[:10]:

            print(

                f"{s['sample_id']} | "

                f"Label={s['label']} | "

                f"Function="
                f"{s['function_nodes']} | "

                f"Retained="
                f"{s['retained_nodes']} | "

                f"Retention="
                f"{s['retention']:.2%}"

            )

    @staticmethod
    def _print_group(
        group
    ):

        if not group:

            return

        def avg(
            key
        ):

            return statistics.mean(

                s[key]

                for s in group

            )

        print(
            f"Samples              : "
            f"{len(group)}"
        )

        print(
            f"Average Function     : "
            f"{avg('function_nodes'):.2f}"
        )

        print(
            f"Average Seeds        : "
            f"{avg('seed_nodes'):.2f}"
        )

        print(
            f"Average Valid Seeds  : "
            f"{avg('valid_seeds'):.2f}"
        )

        print(
            f"Average Retained     : "
            f"{avg('retained_nodes'):.2f}"
        )

        print(
            f"Average Retention    : "
            f"{avg('retention'):.2%}"
        )