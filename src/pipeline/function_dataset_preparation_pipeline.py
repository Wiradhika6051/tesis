
from collections import Counter, defaultdict
from tqdm import tqdm

from src.debug.pipeline_statistics import print_statistics


class FunctionDatasetPreparationPipeline:
    """
    Prepare paired vulnerable/fixed function samples and audit every stage.

    Audit structure:
        self.audit = {
            "counts": {...},
            "stages": {...},
            "events": [...]
        }

    Each prepare() call resets the audit. The experiment runner should
    export or copy the audit immediately after each call.
    """

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

        self.audit = self._new_audit()

    # ==========================================================
    # 1. AUDIT INITIALIZATION AND RECORDING
    # ==========================================================

    @staticmethod
    def _new_audit():
        return {
            "counts": {},
            "stages": {},
            "events": [],
        }

    @staticmethod
    def _sample_metadata(sample):
        """Return available sample metadata without assuming every
        optional attribute exists.
        """
        if sample is None:
            return {}

        return {
            "sample_id": getattr(sample, "sample_id", None),
            "pair_id": getattr(sample, "function_pair_id", None),
            "parent_sample_id": getattr(
                sample, "parent_sample_id", None
            ),
            "repo": getattr(sample, "repo", None),
            "fix_commit_hash": getattr(
                sample, "fix_commit_hash", None
            ),
            "commit_hash": getattr(sample, "commit_hash", None),
            "file_path": getattr(sample, "file_path", None),
            "revision": getattr(sample, "revision", None),
            "label": getattr(sample, "label", None),
            "function_name": getattr(
                sample, "function_name", None
            ),
            "qualified_name": getattr(
                sample, "qualified_name", None
            ),
        }

    def _record_event(
        self,
        stage,
        reason,
        sample=None,
        pair_id=None,
        details=None,
    ):
        """Record a sample-level or pair-level audit event."""
        metadata = self._sample_metadata(sample)

        event = {
            "stage": stage,
            "reason": reason,
            **metadata,
            "pair_id": (
                pair_id
                if pair_id is not None
                else metadata.get("pair_id")
            ),
            "details": details or {},
        }

        self.audit["events"].append(event)
        self.audit["counts"][f"{stage}:{reason}"] = (
            self.audit["counts"].get(
                f"{stage}:{reason}", 0
            ) + 1
        )

    # ==========================================================
    # 2. STAGE STATISTICS
    # ==========================================================

    @staticmethod
    def _unique_pair_count(samples):
        return len({
            sample.function_pair_id
            for sample in samples
            if getattr(sample, "function_pair_id", None)
            is not None
        })

    def _record_stage(self, stage_name, samples):
        """Record sample and pair counts for a preparation stage."""
        samples = list(samples)

        labels = Counter(
            getattr(sample, "label", None)
            for sample in samples
        )

        self.audit["stages"][stage_name] = {
            "sample_count": len(samples),
            "pair_count": self._unique_pair_count(samples),
            "positive_count": labels.get(1, 0),
            "negative_count": labels.get(0, 0),
            "other_label_count": sum(
                count
                for label, count in labels.items()
                if label not in (0, 1)
            ),
        }

    # ==========================================================
    # 3. COMPLETE-PAIR VALIDATION
    # ==========================================================

    @staticmethod
    def _complete_pairs(samples):
        """
        Retain pair IDs containing exactly two samples:
        one vulnerable (label=1), one fixed (label=0),
        with revisions 'vulnerable' and 'fixed'.
        """
        groups = defaultdict(list)

        for sample in samples:
            pair_id = getattr(
                sample, "function_pair_id", None
            )

            if pair_id is not None:
                groups[pair_id].append(sample)

        result = []

        for pair_samples in groups.values():
            labels = {
                getattr(sample, "label", None)
                for sample in pair_samples
            }
            revisions = {
                getattr(sample, "revision", None)
                for sample in pair_samples
            }

            if (
                len(pair_samples) == 2
                and labels == {0, 1}
                and revisions == {"vulnerable", "fixed"}
            ):
                result.extend(pair_samples)

        return result

    def _retain_complete_pairs(self, samples, stage_name):
        """
        Retain complete pairs and audit every excluded member.

        Pair-level events explain why the pair was excluded.
        Members already rejected at an earlier stage retain their
        original failure event as well.
        """
        groups = defaultdict(list)
        retained = []

        for sample in samples:
            pair_id = getattr(
                sample, "function_pair_id", None
            )

            if pair_id is None:
                self._record_event(
                    stage=stage_name,
                    reason="missing_pair_id",
                    sample=sample,
                )
                continue

            groups[pair_id].append(sample)

        dropped_pair_count = 0

        for pair_id, members in groups.items():
            labels = [
                getattr(member, "label", None)
                for member in members
            ]
            revisions = [
                getattr(member, "revision", None)
                for member in members
            ]

            is_complete = (
                len(members) == 2
                and set(labels) == {0, 1}
                and set(revisions) == {"vulnerable", "fixed"}
            )

            if is_complete:
                retained.extend(members)
                continue

            dropped_pair_count += 1

            self._record_event(
                stage=stage_name,
                reason="incomplete_pair",
                pair_id=pair_id,
                details={
                    "member_count": len(members),
                    "labels": labels,
                    "revisions": revisions,
                    "sample_ids": [
                        getattr(member, "sample_id", None)
                        for member in members
                    ],
                    "qualified_names": [
                        getattr(member, "qualified_name", None)
                        for member in members
                    ],
                },
            )

        self.audit["counts"][
            f"{stage_name}_pairs_dropped"
        ] = dropped_pair_count

        return retained

    # ==========================================================
    # 4. PIPELINE
    # ==========================================================

    def prepare(self, samples):
        self.audit = self._new_audit()

        samples = list(samples)

        # ------------------------------------------------------
        # Stage 1: Raw samples
        # ------------------------------------------------------

        self._record_stage("raw", samples)

        if self.audit_exporter:
            self.audit_exporter.export_raw_samples(samples)

        print_statistics("Raw Samples", samples)

        # ------------------------------------------------------
        # Stage 2: Diff localization
        # Keep samples with empty changed-line lists for pairing.
        # ------------------------------------------------------

        for sample in tqdm(samples, desc="Diff localization"):
            sample.seed_lines = (
                self.diff_localizer.localize(sample) or []
            )

        diff_samples = samples

        self._record_stage("diff_localization", diff_samples)

        if self.audit_exporter:
            self.audit_exporter.export_diff_localized(
                diff_samples
            )

        print_statistics(
            "After Diff Localization",
            diff_samples,
        )

        # ------------------------------------------------------
        # Stage 3: Function extraction
        # ------------------------------------------------------

        function_samples = []

        for sample in tqdm(
            diff_samples, desc="Function extraction"
        ):
            extracted = self.function_sample_builder.build(
                sample
            )

            function_samples.extend(extracted)

        self._record_stage(
            "function_extraction",
            function_samples,
        )

        if self.audit_exporter:
            self.audit_exporter.export_function_samples(
                function_samples
            )

        print_statistics(
            "Function Candidates Before Pairing",
            function_samples,
        )

        # ------------------------------------------------------
        # Stage 4: Pair functions across revisions
        # ------------------------------------------------------

        paired_function_samples = (
            self.paired_function_extractor.extract(
                function_samples
            )
        )

        paired_function_samples = self._retain_complete_pairs(
            paired_function_samples,
            "pairing",
        )

        self._record_stage(
            "pairing",
            paired_function_samples,
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
        pairing_audit = getattr(
            self.paired_function_extractor,
            "audit",
            {},
        )

        for reason, count in sorted(
            pairing_audit.get("counts", {}).items()
        ):
            print(f"  {reason}: {count}")

        # ------------------------------------------------------
        # Stage 5: CFG construction
        # ------------------------------------------------------

        cfg_samples = []

        for sample in tqdm(
            paired_function_samples,
            desc="CFG construction",
        ):
            sample.cfg = self.cfg_builder.build(sample)

            if sample.cfg is None:
                self._record_event(
                    stage="cfg_construction",
                    reason="cfg_build_failed",
                    sample=sample,
                    details={
                        "source": "sample.source",
                        "function_source_available": (
                            getattr(
                                sample, "function_source", None
                            ) is not None
                        ),
                    },
                )
                continue

            cfg_samples.append(sample)

        cfg_samples = self._retain_complete_pairs(
            cfg_samples,
            "cfg_construction",
        )

        self._record_stage("cfg_construction", cfg_samples)

        if self.audit_exporter:
            self.audit_exporter.export_cfg(cfg_samples)

        print_statistics(
            "After CFG Construction",
            cfg_samples,
        )

        # ------------------------------------------------------
        # Stage 6: CFG localization
        # ------------------------------------------------------

        localized_samples = []

        for sample in tqdm(
            cfg_samples,
            desc="CFG localization",
        ):
            sample = self.cfg_localizer.localize(sample)

            if not getattr(sample, "seed_nodes", None):
                self._record_event(
                    stage="cfg_localization",
                    reason="no_seed_nodes",
                    sample=sample,
                    details={
                        "seed_lines": list(
                            getattr(sample, "seed_lines", None)
                            or []
                        ),
                        "cfg_node_count": len(
                            (getattr(sample, "cfg", None) or {}).get(
                                "nodes", []
                            )
                        ),
                    },
                )
                continue

            localized_samples.append(sample)

        localized_samples = self._retain_complete_pairs(
            localized_samples,
            "cfg_localization",
        )

        self._record_stage(
            "cfg_localization",
            localized_samples,
        )

        if self.audit_exporter:
            self.audit_exporter.export_localized(
                localized_samples
            )

        print_statistics(
            "After CFG Localization",
            localized_samples,
        )

        # ------------------------------------------------------
        # Stage 7: Function-scope localization
        # ------------------------------------------------------

        scoped_samples = []

        for sample in tqdm(
            localized_samples,
            desc="Function scope",
        ):
            sample = self.function_scope_localizer.localize(
                sample
            )

            valid_seeds = (
                set(getattr(sample, "seed_nodes", None) or [])
                & set(
                    getattr(sample, "function_nodes", None) or []
                )
            )

            if not valid_seeds:
                self._record_event(
                    stage="function_scope",
                    reason="no_valid_function_seed_nodes",
                    sample=sample,
                    details={
                        "seed_node_count": len(
                            getattr(sample, "seed_nodes", None)
                            or []
                        ),
                        "function_node_count": len(
                            getattr(sample, "function_nodes", None)
                            or []
                        ),
                    },
                )
                continue

            sample.seed_nodes = sorted(valid_seeds)
            scoped_samples.append(sample)

        scoped_samples = self._retain_complete_pairs(
            scoped_samples,
            "function_scope",
        )

        self._record_stage("function_scope", scoped_samples)

        if self.audit_exporter:
            self.audit_exporter.export_function_scope(
                scoped_samples
            )

        print_statistics(
            "After Function Scope",
            scoped_samples,
        )

        # ------------------------------------------------------
        # Stage 8: Pruning
        # ------------------------------------------------------

        prepared_samples = []

        for sample in tqdm(scoped_samples, desc="Pruning"):
            sample = self.pruner.prune(sample)

            if getattr(sample, "pruned_cfg", None) is None:
                self._record_event(
                    stage="pruning",
                    reason="pruned_cfg_is_none",
                    sample=sample,
                )
                continue

            if not sample.pruned_cfg.get("nodes"):
                self._record_event(
                    stage="pruning",
                    reason="pruned_cfg_has_no_nodes",
                    sample=sample,
                    details={
                        "pruned_edge_count": len(
                            sample.pruned_cfg.get("edges", [])
                        ),
                    },
                )
                continue

            prepared_samples.append(sample)

        prepared_samples = self._retain_complete_pairs(
            prepared_samples,
            "pruning",
        )

        self._record_stage("pruning", prepared_samples)

        if self.audit_exporter:
            self.audit_exporter.export_pruned(
                prepared_samples,
                self.pruner.__class__.__name__,
            )

        print_statistics(
            "After Pruning",
            prepared_samples,
        )

        # ------------------------------------------------------
        # Final audit summary
        # ------------------------------------------------------

        print("Pipeline stage summary:")

        for stage, metrics in self.audit["stages"].items():
            print(
                f"  {stage}: "
                f"samples={metrics['sample_count']}, "
                f"pairs={metrics['pair_count']}"
            )

        print("Pipeline exclusion summary:")

        for reason, count in sorted(
            self.audit["counts"].items()
        ):
            print(f"  {reason}: {count}")

        print(
            f"Detailed audit events: "
            f"{len(self.audit['events'])}"
        )

        return prepared_samples
