from tqdm import tqdm
from src.analysis.pruning_dianogtics import PruningDiagnostics
from src.analysis.slice_analysis import analyze_slice

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

    def prepare(
        self,
        samples
    ):

        prepared_samples = []

        diagnostics = PruningDiagnostics()

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

            #
            # 4. Localize target function.
            #
            sample = self.function_localizer.localize(
                sample
            )

            #
            # No seed nodes means
            # there is nothing to prune around.
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

            print("Comparing slices...")
            from src.analysis.slice_comparison import (
                compare_slices,
                print_slice_comparison_summary,
                print_worst_slice_differences
            )

            results = compare_slices(
                samples
            )

            print_slice_comparison_summary(
                results
            )

            print_worst_slice_differences(
                results,
                limit=20
            )
            #
            # 6. Prune CFG.
            #
            sample = self.pruner.prune(
                sample
            )

            #
            # 7. Make sure pruning produced
            #    a valid CFG.
            #
            if sample.pruned_cfg is None:
                print(
                    "WARNING: Pruner returned None"
                )

                print(
                    "Pruner:",
                    type(self.pruner).__name__
                )

                print(
                    "Repo:",
                    sample.repo
                )

                print(
                    "File:",
                    sample.file_path
                )

                continue

            #
            # 8. Skip empty pruned graphs.
            #
            if not sample.pruned_cfg["nodes"]:
                continue

            #
            # 9. Collect pruning diagnostics.
            #
            #
            # Collect pruning diagnostics.
            #
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

            #
            # 10. Keep prepared sample.
            #
            prepared_samples.append(
                sample
            )

        #
        # 11. Print diagnostics.
        #
        diagnostics.summary()

        return prepared_samples