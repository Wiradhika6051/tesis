from tqdm import tqdm


class DatasetPreparationPipeline:

    def __init__(
        self,
        cfg_builder,
        diff_localizer,
        cfg_localizer,
        pruner
    ):
        self.cfg_builder = cfg_builder
        self.diff_localizer = diff_localizer
        self.cfg_localizer = cfg_localizer
        self.pruner = pruner

    def prepare(
        self,
        samples
    ):

        prepared_samples = []

        for sample in tqdm(
            samples,
            desc="Preparing dataset"
        ):

            #
            # Build CFG
            #
            sample.cfg = self.cfg_builder.build(
                sample
            )

            #
            # Localize changed lines
            #
            sample.seed_lines = self.diff_localizer.localize(
                sample
            )

            #
            # Map lines to CFG nodes
            #
            sample.seed_nodes = self.cfg_localizer.localize(
                sample.cfg,
                sample.seed_lines
            )

            #
            # Prune CFG
            #
            sample.pruned_cfg = self.pruner.prune(
                sample.cfg,
                sample.seed_nodes
            )

            prepared_samples.append(
                sample
            )

        return prepared_samples