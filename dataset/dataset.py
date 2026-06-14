import torch

from torch.utils.data import Dataset


class VulnerabilityDataset(
    Dataset
):

    def __init__(
        self,
        samples,
        vocab,
        max_len=100
    ):

        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def pad(self, ids):

        if len(ids) > self.max_len:
            return ids[:self.max_len]

        return ids + [0] * (
            self.max_len - len(ids)
        )

    def __getitem__(
        self,
        idx
    ):

        from vocab import encode_code
        from graph_builder import build_token_graph

        sample = self.samples[idx]

        token_ids = encode_code(
            sample["code"],
            self.vocab
        )

        token_ids = self.pad(
            token_ids
        )

        graph = build_token_graph(
            token_ids
        )

        return (
            graph,
            torch.tensor(
                token_ids,
                dtype=torch.long
            ),
            torch.tensor(
                sample["label"],
                dtype=torch.long
            )
        )