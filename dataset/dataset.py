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

class PhpNetDataset(Dataset):

    def __init__(
        self,
        samples
    ):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
    
        sample = self.samples[idx]
    
        graph = Data(
        
            x=torch.tensor(
                sample.node_types,
                dtype=torch.long
            ),
    
            edge_index=torch.tensor(
                sample.edges,
                dtype=torch.long
            ).t().contiguous()
    
        )
    
        return (
        
            graph,
    
            torch.tensor(
                sample.tokens,
                dtype=torch.long
            ),
    
            torch.tensor(
                sample.label,
                dtype=torch.long
            )
    
        )