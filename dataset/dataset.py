# dataset.py

import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Batch


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


class OldPhpNetDataset(
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
    
import json
from torch.utils.data import Dataset


def get_value(obj, key, default=None):
    """Read a field from either a dict or an object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_jsonable(value):
    """Convert common Python and NumPy/PyTorch values to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]

    return str(value)


def serialize_node(node):
    """Keep the node fields useful for code and pruning analysis."""
    fields = (
        "node_id", "lineno", "end_lineno",
        "node_type", "text"
    )
    return {
        field: to_jsonable(get_value(node, field))
        for field in fields
        if get_value(node, field) is not None
    }


def serialize_cfg(cfg):
    """Serialize a CFG dict containing nodes and edges."""
    nodes = get_value(cfg, "nodes", []) or []
    edges = get_value(cfg, "edges", []) or []

    return {
        "nodes": [serialize_node(node) for node in nodes],
        "edges": to_jsonable(edges),
    }


class PhpNetDataset(Dataset):
    def __init__(
        self,
        samples,
        pruning_strategy="unknown",
        architecture="gcn",
        split="unknown",
    ):
        self.samples = samples
        self.pruning_strategy = pruning_strategy
        self.architecture = architecture
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        graph = sample.graph
        label = int(sample.label)

        repo = sample.repo
        parent_commit = sample.parent_commit
        commit_hash = getattr(sample, "commit_hash", None)
        file_path = sample.file_path
        function_start = getattr(sample, "function_start", None)
        function_end = getattr(sample, "function_end", None)

        # Include function boundaries to distinguish functions in the same file.
        identity = [
            repo,
            parent_commit,
            file_path,
            function_start,
            function_end,
        ]

        sample_uid = json.dumps(
            identity,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        # Preserve the existing pair concept separately from sample identity.
        pair_id = (
            repo,
            parent_commit,
            file_path,
        )

        original_cfg = serialize_cfg(sample.cfg)
        pruned_cfg = serialize_cfg(sample.pruned_cfg)

        metadata = {
            "sample_uid": sample_uid,
            "repo": repo,
            "parent_commit": parent_commit,
            "commit_hash": commit_hash,
            "file_path": file_path,
            "function_start": function_start,
            "function_end": function_end,

            "source": to_jsonable(sample.source),
            "diff": to_jsonable(getattr(sample, "diff", None)),
            "label": label,
            "seed_lines": to_jsonable(
                getattr(sample, "seed_lines", [])
            ),
            "seed_nodes": to_jsonable(
                getattr(sample, "seed_nodes", [])
            ),

            "original_node_count": len(original_cfg["nodes"]),
            "original_edge_count": len(original_cfg["edges"]),
            "original_nodes": original_cfg["nodes"],
            "original_edges": original_cfg["edges"],

            "pruned_node_count": len(pruned_cfg["nodes"]),
            "pruned_edge_count": len(pruned_cfg["edges"]),
            "pruned_nodes": pruned_cfg["nodes"],
            "pruned_edges": pruned_cfg["edges"],

            "original_to_pruned": to_jsonable(
                get_value(
                    sample.pruned_cfg,
                    "original_to_pruned",
                    getattr(sample, "original_to_pruned", {}),
                )
            ),

            "pruning_strategy": self.pruning_strategy,
            "architecture": self.architecture,
            "split": self.split,
        }

        return graph, label, pair_id, sample_uid, metadata