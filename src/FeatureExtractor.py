class FeatureExtractor:

    def extract(self, sample):

        sample.node_features = sample.pruned_cfg["nodes"]

        sample.edges = sample.pruned_cfg["edges"]

        return sample