# dataset.py

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class VulnerabilityDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]

        return (
            torch.tensor(
                item["tokens"],
                dtype=torch.long
            ),
            torch.tensor(
                item["label"],
                dtype=torch.long
            )
        )

from torch.utils.data import Dataset


class PhpNetDataset(
    Dataset
):

    def __init__(
        self,
        samples
    ):

        self.samples = samples

    def __len__(
        self
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index
    ):

        sample = self.samples[
            index
        ]

        graph = sample.graph

        label = torch.tensor(

            sample.label,

            dtype=torch.long

        )

        #
        # Same identifier for vulnerable/fixed pair.
        #
        pair_id = (

            sample.repo,

            sample.parent_commit,

            sample.file_path

        )

        #
        # Unique identifier for this version.
        #
        sample_id = (

            sample.repo,

            sample.parent_commit,

            sample.file_path,

            sample.label

        )

        return (

            graph,

            label,

            pair_id,

            sample_id

        )