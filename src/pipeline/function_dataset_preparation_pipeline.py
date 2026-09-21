from tqdm import tqdm


class FunctionDatasetPreparationPipeline:

    def __init__(
        self,
        function_sample_builder,
        cfg_builder,
        diff_localizer,
        cfg_localizer,
        function_scope_localizer,
        pruner,
    ):

        self.function_sample_builder = (
            function_sample_builder
        )

        self.cfg_builder = cfg_builder

        self.diff_localizer = diff_localizer

        self.cfg_localizer = cfg_localizer

        self.function_scope_localizer = (
            function_scope_localizer
        )

        self.pruner = pruner

    def prepare(self, samples):

        prepared_samples = []

        for sample in tqdm(samples):

            # ------------------------------------------------
            # 1. Extract changed lines from the complete diff
            # ------------------------------------------------

            sample.seed_lines = (
                self.diff_localizer.localize(sample)
            )

            if not sample.seed_lines:
                continue

            # ------------------------------------------------
            # 2. Split changed lines by containing function
            # ------------------------------------------------

            function_samples = (
                self.function_sample_builder.build(sample)
            )

            # No changed line belongs to a function.
            if not function_samples:
                continue

            # ------------------------------------------------
            # 3. Process each function independently
            # ------------------------------------------------

            for function_sample in function_samples:

                # --------------------------------------------
                # Build CFG from the original source
                # --------------------------------------------

                function_sample.cfg = (
                    self.cfg_builder.build(
                        function_sample
                    )
                )

                if not function_sample.cfg:
                    continue

                # --------------------------------------------
                # Map changed lines to CFG seed nodes
                # --------------------------------------------

                function_sample = (
                    self.cfg_localizer.localize(
                        function_sample
                    )
                )

                # Changed lines existed, but none could be
                # mapped to a CFG node.
                if not function_sample.seed_nodes:
                    continue

                # --------------------------------------------
                # Determine function scope
                # --------------------------------------------

                function_sample = (
                    self.function_scope_localizer.localize(
                        function_sample
                    )
                )

                # --------------------------------------------
                # Make sure at least one seed belongs to
                # the selected function.
                # --------------------------------------------

                valid_seeds = (
                    set(function_sample.seed_nodes)
                    &
                    set(function_sample.function_nodes)
                )

                if not valid_seeds:
                    continue

                function_sample.seed_nodes = valid_seeds

                # --------------------------------------------
                # Apply pruning
                # --------------------------------------------

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

        return prepared_samples