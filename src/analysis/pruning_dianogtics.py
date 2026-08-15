from collections import Counter, defaultdict
import statistics


class PruningDiagnostics:
    def __init__(self):
        self.samples = []

    def add_sample(
        self,
        label,
        function_nodes,
        seed_nodes,
        backward_nodes,
        forward_nodes,
        sample_id=None,
    ):
        function_nodes = set(function_nodes)
        seed_nodes = set(seed_nodes)
        backward_nodes = set(backward_nodes)
        forward_nodes = set(forward_nodes)

        valid_seeds = seed_nodes & function_nodes
        outside_seeds = seed_nodes - function_nodes

        # Nodes retained by the complete slice
        slice_nodes = backward_nodes | forward_nodes | valid_seeds

        # Retention relative to the localized function
        if function_nodes:
            retention = len(slice_nodes) / len(function_nodes)
        else:
            retention = 0.0

        # Check whether seeds survive into the slice
        missing_seeds = valid_seeds - slice_nodes

        self.samples.append({
            "sample_id": sample_id,
            "label": label,

            "function_nodes": len(function_nodes),
            "seed_nodes": len(seed_nodes),
            "valid_seeds": len(valid_seeds),
            "outside_seeds": len(outside_seeds),

            "backward_nodes": len(backward_nodes),
            "forward_nodes": len(forward_nodes),
            "slice_nodes": len(slice_nodes),

            "retention": retention,

            "missing_seeds": len(missing_seeds),
            "has_localization_mismatch": len(outside_seeds) > 0,
            "has_valid_seed": len(valid_seeds) > 0,
        })

    def summary(self):
        if not self.samples:
            print("No samples.")
            return

        n = len(self.samples)

        print("=" * 70)
        print("PRUNING DIAGNOSTICS")
        print("=" * 70)

        print(f"Samples analyzed : {n}")

        # ---------------------------------------------------------
        # Localization
        # ---------------------------------------------------------

        mismatch = sum(
            s["has_localization_mismatch"]
            for s in self.samples
        )

        no_valid_seed = sum(
            not s["has_valid_seed"]
            for s in self.samples
        )

        print("\n## Localization")

        print(
            f"Samples with mismatch : "
            f"{mismatch} ({mismatch / n:.2%})"
        )

        print(
            f"Samples with no valid seed : "
            f"{no_valid_seed} ({no_valid_seed / n:.2%})"
        )

        outside_counts = [
            s["outside_seeds"]
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

        # ---------------------------------------------------------
        # Overall pruning
        # ---------------------------------------------------------

        print("\n## Overall")

        self._print_group(self.samples)

        # ---------------------------------------------------------
        # By label
        # ---------------------------------------------------------

        labels = sorted(
            set(s["label"] for s in self.samples)
        )

        print("\n## By Label")

        for label in labels:
            group = [
                s for s in self.samples
                if s["label"] == label
            ]

            print(f"\nLabel {label}")
            self._print_group(group)

        # ---------------------------------------------------------
        # Retention difference
        # ---------------------------------------------------------

        print("\n## Label Difference")

        grouped = {
            label: [
                s for s in self.samples
                if s["label"] == label
            ]
            for label in labels
        }

        if len(labels) == 2:
            a = statistics.mean(
                s["retention"] for s in grouped[labels[0]]
            )

            b = statistics.mean(
                s["retention"] for s in grouped[labels[1]]
            )

            print(
                f"Retention difference : "
                f"{abs(a - b):.2%}"
            )

        # ---------------------------------------------------------
        # Worst localization mismatches
        # ---------------------------------------------------------

        print("\n## Worst Localization Mismatches")

        mismatches = sorted(
            [
                s for s in self.samples
                if s["has_localization_mismatch"]
            ],
            key=lambda s: s["outside_seeds"],
            reverse=True
        )

        for s in mismatches[:10]:
            print(
                f"{s['sample_id']} | "
                f"Label={s['label']} | "
                f"Function={s['function_nodes']} | "
                f"Seeds={s['seed_nodes']} | "
                f"Valid={s['valid_seeds']} | "
                f"Outside={s['outside_seeds']}"
            )

        # ---------------------------------------------------------
        # Worst retention
        # ---------------------------------------------------------

        print("\n## Lowest Retention")

        lowest = sorted(
            self.samples,
            key=lambda s: s["retention"]
        )

        for s in lowest[:10]:
            print(
                f"{s['sample_id']} | "
                f"Label={s['label']} | "
                f"Function={s['function_nodes']} | "
                f"Slice={s['slice_nodes']} | "
                f"Retention={s['retention']:.2%}"
            )

    @staticmethod
    def _print_group(group):
        if not group:
            return

        def avg(key):
            return statistics.mean(
                s[key] for s in group
            )

        print(f"Samples              : {len(group)}")
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
            f"Average Backward     : "
            f"{avg('backward_nodes'):.2f}"
        )
        print(
            f"Average Forward      : "
            f"{avg('forward_nodes'):.2f}"
        )
        print(
            f"Average Slice        : "
            f"{avg('slice_nodes'):.2f}"
        )
        print(
            f"Average Retention    : "
            f"{avg('retention'):.2%}"
        )