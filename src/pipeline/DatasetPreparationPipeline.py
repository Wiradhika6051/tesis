from tqdm import tqdm

from src.analysis.pruning_dianogtics import (
    PruningDiagnostics
)

from src.analysis.slice_analysis import (
    analyze_slice
)

from src.analysis.slice_comparison import (
    compare_slices,
    print_slice_comparison_summary,
    print_worst_slice_differences
)

from src.analysis.seed_localization_diagnostics import (
    SeedLocalizationDiagnostics
)

class DatasetPreparationPipeline:

    def __init__(
        self,
        cfg_builder,
        diff_localizer,
        cfg_localizer,
        function_localizer,
        pruner
    ):

        self.cfg_builder = cfg_builder
        self.diff_localizer = diff_localizer
        self.cfg_localizer = cfg_localizer
        self.function_localizer = function_localizer
        self.pruner = pruner

    def prepare(self, samples):

        prepared_samples = []

        diagnostics = PruningDiagnostics()

        seed_diagnostics = (
            SeedLocalizationDiagnostics()
        )

        localized_samples = []

        for sample in tqdm(
            samples,
            desc="Preparing dataset"
        ):

            #
            # 1. Build CFG
            #
            sample.cfg = self.cfg_builder.build(
                sample
            )
            if not sample.cfg:
                continue
            print("\nCFG LINE COVERAGE")

            cfg_lines = sorted(
                node.lineno
                for node in sample.cfg["nodes"]
                if node.lineno >= 0
            )

            print("CFG lines:")
            print(cfg_lines)

            print("\nSeed lines:")
            print(sample.seed_lines)

            print("\nMissing seed lines:")

            for line in sample.seed_lines:
            
                if line not in cfg_lines:
                
                    print(
                        f"Line {line} is NOT represented in CFG"
                    )

            #
            # 2. Localize changed lines
            #
            sample.seed_lines = (
                self.diff_localizer.localize(
                    sample
                )
            )

            #
            # 3. Localize changed lines
            #    to CFG nodes.
            #
            sample = self.cfg_localizer.localize(
                sample
            )
            print("=" * 70)
            print("LOCALIZATION DEBUG")
            print("=" * 70)
            
            print("Repo:", sample.repo)
            print("File:", sample.file_path)
            print("Label:", sample.label)
            
            print()
            print("Seed lines:")
            print(sample.seed_lines[:50])
            
            print()
            print("Seed nodes:")
            print(sample.seed_nodes[:50])
            
            print()
            print("CFG nodes:")
            for node in sample.cfg["nodes"][:30]:
                print(
                    node.node_id,
                    node.lineno,
                    node.node_type,
                    repr(node.text[:100])
                )
            
            print()
            print("Function nodes:")
            print(sample.function_nodes[:50])
            
            print("=" * 70)
            #
            # 4. Localize target function.
            #
            sample = self.function_localizer.localize(
                sample
            )

            #
            # Keep it for seed diagnostics.
            #
            localized_samples.append(
                sample
            )

            #
            # Continue normal pipeline.
            #
            if not sample.seed_nodes:
                continue

            #
            # 5. Analyze slicing BEFORE pruning.
            #
            slice_analysis = analyze_slice(
                sample
            )

            if slice_analysis is not None:
                sample.slice_analysis = (
                    slice_analysis
                )

            #
            # 6. Prune CFG.
            #
            sample = self.pruner.prune(
                sample
            )

            if sample.pruned_cfg is None:
                continue

            if not sample.pruned_cfg["nodes"]:
                continue

            diagnostics.add_sample(
                label=sample.label,

                function_nodes=
                    sample.function_nodes,

                seed_nodes=
                    sample.seed_nodes,

                retained_nodes={
                    node.node_id
                    for node in sample.pruned_cfg["nodes"]
                },

                sample_id=(
                    f"{sample.repo}:"
                    f"{sample.file_path}"
                )
            )

            prepared_samples.append(
                sample
            )

        print("Comparing slices...")

        results = compare_slices(
            localized_samples
        )
        
        print_slice_comparison_summary(
            results
        )
        
        print_worst_slice_differences(
            results,
            limit=20
        )

        #
        # --------------------------------------------------
        # Seed diagnostics.
        # --------------------------------------------------
        #

        seed_diagnostics.analyze(
            localized_samples
        )

        seed_diagnostics.print_worst_samples(
            limit=20
        )

        #
        # Normal pruning diagnostics.
        #
        diagnostics.summary()

        return prepared_samples