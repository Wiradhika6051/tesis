from collections import defaultdict
from tqdm import tqdm

from src.debug.pipeline_statistics import print_statistics


class FunctionDatasetPreparationPipeline:

    def __init__(
        self,
        function_sample_builder,
        paired_function_extractor,
        cfg_builder,
        diff_localizer,
        cfg_localizer,
        function_scope_localizer,
        pruner,
        audit_exporter=None,
    ):
        self.function_sample_builder = function_sample_builder
        self.paired_function_extractor = paired_function_extractor
        self.cfg_builder = cfg_builder
        self.diff_localizer = diff_localizer
        self.cfg_localizer = cfg_localizer
        self.function_scope_localizer = function_scope_localizer
        self.pruner = pruner
        self.audit_exporter = audit_exporter

        self.audit = {"counts": {}}

    @staticmethod
    def _complete_pairs(samples):
        """
        Keep only pair IDs with exactly two samples:
        one vulnerable (label=1), one fixed (label=0).
        """
        groups = defaultdict(list)

        for sample in samples:
            if sample.function_pair_id:
                groups[sample.function_pair_id].append(sample)

        result = []

        for pair_samples in groups.values():
            if (
                len(pair_samples) == 2
                and {s.label for s in pair_samples} == {0, 1}
                and {
                    s.revision for s in pair_samples
                } == {"vulnerable", "fixed"}
            ):
                result.extend(pair_samples)

        return result

    def _retain_complete_pairs(self, samples, stage_name):
        retained = self._complete_pairs(samples)

        before_ids = {
            s.function_pair_id
            for s in samples
            if s.function_pair_id
        }
        after_ids = {
            s.function_pair_id
            for s in retained
            if s.function_pair_id
        }

        dropped = len(before_ids - after_ids)
        self.audit["counts"][f"{stage_name}_pairs_dropped"] = dropped

        return retained

    def prepare(self, samples):
        self.audit = {"counts": {}}

        if self.audit_exporter:
            self.audit_exporter.export_raw_samples(samples)

        print_statistics("Raw Samples", samples)

        # --------------------------------------------------------
        # Stage 2: Diff localization
        # Do not discard samples with empty changed-line lists.
        # The counterpart may still be needed for pairing.
        # --------------------------------------------------------

        for sample in tqdm(samples, desc="Diff localization"):
            sample.seed_lines = (
                self.diff_localizer.localize(sample) or []
            )

        diff_samples = samples

        if self.audit_exporter:
            self.audit_exporter.export_diff_localized(diff_samples)

        print_statistics("After Diff Localization", diff_samples)

        # --------------------------------------------------------
        # Stage 3: Extract all function candidates
        # --------------------------------------------------------

        function_samples = []

        for sample in tqdm(diff_samples, desc="Function extraction"):
            function_samples.extend(
                self.function_sample_builder.build(sample)
            )

        if self.audit_exporter:
            self.audit_exporter.export_function_samples(function_samples)

        print_statistics(
            "Function Candidates Before Pairing",
            function_samples,
        )

        # --------------------------------------------------------
        # Stage 4: Pair functions across revisions
        # --------------------------------------------------------

        paired_function_samples = (
            self.paired_function_extractor.extract(function_samples)
        )

        paired_function_samples = self._retain_complete_pairs(
            paired_function_samples,
            "pairing",
        )

        if self.audit_exporter:
            self.audit_exporter.export_paired_function_samples(
                paired_function_samples
            )

        print_statistics(
            "After Paired-Function Extraction",
            paired_function_samples,
        )

        print("Pairing summary:")
        for reason, count in sorted(
            self.paired_function_extractor.audit["counts"].items()
        ):
            print(f"  {reason}: {count}")

        # --------------------------------------------------------
        # Stage 5: CFG construction
        # If one member fails, exclude both members.
        # --------------------------------------------------------

        cfg_samples = []

        for sample in tqdm(paired_function_samples, desc="CFG construction"):
            sample.cfg = self.cfg_builder.build(sample)

            if sample.cfg is not None:
                cfg_samples.append(sample)

        cfg_samples = self._retain_complete_pairs(
            cfg_samples, "cfg_construction"
        )

        if self.audit_exporter:
            self.audit_exporter.export_cfg(cfg_samples)

        print_statistics("After CFG Construction", cfg_samples)

        # --------------------------------------------------------
        # Stage 6: CFG localization
        # --------------------------------------------------------

        localized_samples = []

        for sample in tqdm(cfg_samples, desc="CFG localization"):
            sample = self.cfg_localizer.localize(sample)

            if sample.seed_nodes:
                localized_samples.append(sample)

        localized_samples = self._retain_complete_pairs(
            localized_samples, "cfg_localization"
        )

        if self.audit_exporter:
            self.audit_exporter.export_localized(localized_samples)

        print_statistics("After CFG Localization", localized_samples)

        # --------------------------------------------------------
        # Stage 7: Function-scope localization
        # --------------------------------------------------------

        scoped_samples = []

        for sample in tqdm(localized_samples, desc="Function scope"):
            sample = self.function_scope_localizer.localize(sample)

            valid_seeds = (
                set(sample.seed_nodes)
                & set(sample.function_nodes)
            )

            if not valid_seeds:
                continue

            sample.seed_nodes = sorted(valid_seeds)
            scoped_samples.append(sample)

        scoped_samples = self._retain_complete_pairs(
            scoped_samples, "function_scope"
        )

        if self.audit_exporter:
            self.audit_exporter.export_function_scope(scoped_samples)

        print_statistics("After Function Scope", scoped_samples)

        # --------------------------------------------------------
        # Stage 8: Pruning
        # --------------------------------------------------------

        prepared_samples = []

        for sample in tqdm(scoped_samples, desc="Pruning"):
            sample = self.pruner.prune(sample)

            if sample.pruned_cfg is None:
                continue

            if not sample.pruned_cfg.get("nodes"):
                continue

            prepared_samples.append(sample)

        prepared_samples = self._retain_complete_pairs(
            prepared_samples, "pruning"
        )

        if self.audit_exporter:
            self.audit_exporter.export_pruned(
                prepared_samples,
                self.pruner.__class__.__name__,
            )

        print_statistics("After Pruning", prepared_samples)

        print("Pipeline pair-retention summary:")
        for stage, count in sorted(self.audit["counts"].items()):
            print(f"  {stage}: {count}")

        return prepared_samples
