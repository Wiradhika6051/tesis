from src.encoding.vudenc_vectorizer import (
    VUDENCVectorizer
)

from src.encoding.gcn_encoder import (
    GCNEncoder
)


class FeaturePreparationPipeline:

    def __init__(
        self,
        w2v_path,
        cfg_vocab_builder,
        audit_exporter=None
    ):

        self.encoder = None

        self.w2v_path = w2v_path

        self.cfg_vocab_builder = (
            cfg_vocab_builder
        )

        self.audit_exporter = (
            audit_exporter
        )

        self.vudenc_vectorizer = (
            VUDENCVectorizer(
                w2v_path
            )
        )

    def prepare(
        self,
        samples
    ):

        # ==================================================
        # 1. Build CFG vocabulary
        # ==================================================

        cfg_vocab = self.cfg_vocab_builder(
            samples
        )

        # ==================================================
        # 2. Export CFG vocabulary
        # ==================================================

        if self.audit_exporter:

            self.audit_exporter.export_vocabularies(
                None,
                cfg_vocab
            )

        # ==================================================
        # 3. Create GCN encoder
        # ==================================================

        self.encoder = GCNEncoder(
            self.vudenc_vectorizer,
            cfg_vocab
        )

        # ==================================================
        # 4. Encode samples
        # ==================================================

        encoded_samples = []

        for sample in samples:

            encoded_sample = (
                self.encoder.encode(
                    sample
                )
            )

            encoded_samples.append(
                encoded_sample
            )

        # ==================================================
        # 5. Export encoded representation
        # ==================================================

        if self.audit_exporter:

            self.audit_exporter.export_encoded(
                encoded_samples
            )

        return (
            encoded_samples,
            None,
            cfg_vocab
        )