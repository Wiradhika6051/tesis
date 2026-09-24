from src.Encoder import Encoder

class FeaturePreparationPipeline:

    def __init__(
        self,
        token_vocab_builder,
        cfg_vocab_builder
    ):
        self.encoder = None
        self.token_vocab_builder = token_vocab_builder
        self.cfg_vocab_builder = cfg_vocab_builder

    def prepare(
        self,
        samples
    ):

        #
        # Build vocabularies.
        #
        token_vocab = self.token_vocab_builder(
            samples
        )

        cfg_vocab = self.cfg_vocab_builder(
            samples
        )

        #
        # Update encoder with the vocabularies.
        #
        self.encoder = Encoder(
            token_vocab,
            cfg_vocab
        )

        #
        # Encode every sample.
        #
        encoded_samples = []

        for sample in samples:

            encoded_samples.append(
                self.encoder.encode(
                    sample
                )
            )

        return (
            encoded_samples,
            token_vocab,
            cfg_vocab
        )