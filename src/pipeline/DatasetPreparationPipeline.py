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
            if not sample.cfg:
                continue

            #
            # Localize changed lines
            #
            sample.seed_lines = self.diff_localizer.localize(
                sample
            )

            #
            # Map lines to CFG nodes
            #
            sample = self.cfg_localizer.localize(
                sample
            )

            #
            # Prune CFG
            #
            sample = self.pruner.prune(
                sample
            )

            prepared_samples.append(
                sample
            )

        return prepared_samples