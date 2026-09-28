from src.Encoder import Encoder


class FeaturePreparationPipeline:

    def __init__(
        self,
        token_vocab_builder,
        cfg_vocab_builder,
        audit_exporter=None
    ):

        self.encoder = None

        self.token_vocab_builder = (
            token_vocab_builder
        )

        self.cfg_vocab_builder = (
            cfg_vocab_builder
        )

        self.audit_exporter = audit_exporter


    def prepare(
        self,
        samples
    ):

        # ==================================================
        # 1. Build vocabularies
        # ==================================================

        token_vocab = self.token_vocab_builder(
            samples
        )

        cfg_vocab = self.cfg_vocab_builder(
            samples
        )

        # ==================================================
        # 2. Export vocabularies
        # ==================================================

        if self.audit_exporter:

            self.audit_exporter.export_vocabularies(
                token_vocab,
                cfg_vocab
            )

        # ==================================================
        # 3. Create encoder
        # ==================================================

        self.encoder = Encoder(
            token_vocab,
            cfg_vocab
        )

        # ==================================================
        # 4. Export tokenization
        # ==================================================

        if self.audit_exporter:

            self.audit_exporter.export_tokenization(
                samples
            )

        # ==================================================
        # 5. Encode samples
        # ==================================================

        encoded_samples = []

        for sample in samples:

            encoded_sample = (
                self.encoder.encode(sample)
            )

            encoded_samples.append(
                encoded_sample
            )

        # ==================================================
        # 6. Export encoded representation
        # ==================================================

        if self.audit_exporter:

            self.audit_exporter.export_encoded(
                encoded_samples
            )

        return (
            encoded_samples,
            token_vocab,
            cfg_vocab
        )