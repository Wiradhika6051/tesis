from tqdm import tqdm

from src.debug.pruning_debug_exporter import (
    PruningDebugExporter
)


class DatasetPreparationPipeline:

    def __init__(
        self,
        cfg_builder,
        diff_localizer,
        cfg_localizer,
        function_localizer,
        pruner,
        debug_output_dir=None,
        debug_name=""
    ):

        self.cfg_builder = cfg_builder
        self.diff_localizer = diff_localizer
        self.cfg_localizer = cfg_localizer
        self.function_localizer = function_localizer
        self.pruner = pruner
        self.debug_name = debug_name

        self.debug_exporter = None

        if debug_output_dir is not None:
            self.debug_exporter = (
                PruningDebugExporter(
                    debug_output_dir
                )
            )

    def prepare(self, samples):

        prepared_samples = []

        for sample in tqdm(
            samples,
            desc="Preparing dataset"
        ):

            # --------------------------------------------------
            # 1. Build CFG
            # --------------------------------------------------

            sample.cfg = self.cfg_builder.build(
                sample
            )

            if not sample.cfg:
                continue

            # --------------------------------------------------
            # 2. Localize changed source lines
            # --------------------------------------------------

            sample.seed_lines = (
                self.diff_localizer.localize(
                    sample
                )
            )

            # --------------------------------------------------
            # 3. Map changed lines to CFG nodes
            # --------------------------------------------------

            sample = self.cfg_localizer.localize(
                sample
            )

            # --------------------------------------------------
            # 4. Identify target function
            # --------------------------------------------------

            sample = self.function_localizer.localize(
                sample
            )

            # No seed nodes means there is nothing to prune.
            if not sample.seed_nodes:
                continue

            # --------------------------------------------------
            # 5. Prune CFG
            # --------------------------------------------------

            sample = self.pruner.prune(
                sample
            )

            if sample.pruned_cfg is None:
                continue

            if not sample.pruned_cfg["nodes"]:
                continue

            # --------------------------------------------------
            # 6. Keep prepared sample
            # --------------------------------------------------

            prepared_samples.append(
                sample
            )

        # ------------------------------------------------------
        # 7. Export pruning verification data
        # ------------------------------------------------------

        if self.debug_exporter is not None:

            self.debug_exporter.export(
                prepared_samples,
                self.debug_name
            )

        return prepared_samples

    def _get_pruner_name(self):

        name = self.pruner.__class__.__name__

        # Remove the "Pruner" suffix.
        if name.endswith("Pruner"):
            name = name[:-6]

        return name.lower()