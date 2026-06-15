# dataset.py

import torch
from torch.utils.data import Dataset


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