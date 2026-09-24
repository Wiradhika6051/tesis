from tqdm import tqdm

from src.debug.pipeline_audit import PipelineAuditExporter
from src.debug.pipeline_statistics import print_statistics


class FunctionDatasetPreparationPipeline:

    def __init__(
        self,
        function_sample_builder,
        cfg_builder,
        diff_localizer,
        cfg_localizer,
        function_scope_localizer,
        pruner,
        audit_exporter=None,
    ):
        self.function_sample_builder = function_sample_builder
        self.cfg_builder = cfg_builder
        self.diff_localizer = diff_localizer
        self.cfg_localizer = cfg_localizer
        self.function_scope_localizer = function_scope_localizer
        self.pruner = pruner
        self.audit_exporter = audit_exporter

    def prepare(self, samples):

        # --------------------------------------------------------
        # Stage 1: raw samples
        # --------------------------------------------------------

        if self.audit_exporter:
            self.audit_exporter.export_raw_samples(samples)

        print_statistics(
            "Raw Samples",
            samples
        )

        prepared_samples = []

        # --------------------------------------------------------
        # Stage 2: diff localization
        # --------------------------------------------------------

        diff_samples = []

        for sample in tqdm(
            samples,
            desc="Diff localization"
        ):

            sample.seed_lines = (
                self.diff_localizer.localize(sample)
            )

            if not sample.seed_lines:
                continue

            diff_samples.append(sample)

        if self.audit_exporter:
            self.audit_exporter.export_diff_localized(
                diff_samples
            )

        print_statistics(
            "After Diff Localization",
            diff_samples
        )

        # --------------------------------------------------------
        # Stage 3: function splitting
        # --------------------------------------------------------

        function_samples = []

        for sample in diff_samples:

            generated = (
                self.function_sample_builder.build(
                    sample
                )
            )

            if not generated:
                continue

            function_samples.extend(generated)

        if self.audit_exporter:
            self.audit_exporter.export_function_samples(
                function_samples
            )

        print_statistics(
            "Function-Level Samples",
            function_samples
        )

        # --------------------------------------------------------
        # Stage 4+: CFG + localization + pruning
        # --------------------------------------------------------

        cfg_samples = []
        localized_samples = []
        scoped_samples = []

        for function_sample in tqdm(
            function_samples,
            desc="CFG preparation"
        ):

            # CFG
            function_sample.cfg = (
                self.cfg_builder.build(
                    function_sample
                )
            )

            if not function_sample.cfg:
                continue

            cfg_samples.append(function_sample)

            # CFG localization
            function_sample = (
                self.cfg_localizer.localize(
                    function_sample
                )
            )

            if not function_sample.seed_nodes:
                continue

            localized_samples.append(
                function_sample
            )

            # Function scope
            function_sample = (
                self.function_scope_localizer.localize(
                    function_sample
                )
            )

            valid_seeds = (
                set(function_sample.seed_nodes)
                &
                set(function_sample.function_nodes)
            )

            if not valid_seeds:
                continue

            function_sample.seed_nodes = valid_seeds

            scoped_samples.append(
                function_sample
            )

        # --------------------------------------------------------
        # Export intermediate CFG stages
        # --------------------------------------------------------

        if self.audit_exporter:
            self.audit_exporter.export_cfg(
                cfg_samples
            )

            self.audit_exporter.export_localized(
                localized_samples
            )

            self.audit_exporter.export_function_scope(
                scoped_samples
            )

        print_statistics(
            "After CFG Construction",
            cfg_samples
        )

        print_statistics(
            "After CFG Localization",
            localized_samples
        )

        print_statistics(
            "After Function Scope",
            scoped_samples
        )

        # --------------------------------------------------------
        # Stage 7: pruning
        # --------------------------------------------------------

        for function_sample in tqdm(
            scoped_samples,
            desc="Pruning"
        ):

            function_sample = (
                self.pruner.prune(
                    function_sample
                )
            )

            if function_sample.pruned_cfg is None:
                continue

            if not function_sample.pruned_cfg["nodes"]:
                continue

            prepared_samples.append(
                function_sample
            )

        # --------------------------------------------------------
        # Export final pruning result
        # --------------------------------------------------------

        if self.audit_exporter:
            self.audit_exporter.export_pruned(
                prepared_samples,
                self.pruner.__class__.__name__
            )

        print_statistics(
            "After Pruning",
            prepared_samples
        )

        return prepared_samples