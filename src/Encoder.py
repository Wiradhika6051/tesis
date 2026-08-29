import torch

from torch_geometric.data import Data

from src.vocabulary import tokenize_code


class Encoder:

    def __init__(
        self,
        token_vocab,
        cfg_vocab
    ):

        self.token_vocab = token_vocab
        self.cfg_vocab = cfg_vocab


    def encode(
        self,
        sample
    ):

        node_tokens = []

        node_types = []

        for node in sample.pruned_cfg["nodes"]:

            tokens = tokenize_code(
                node.text
            )

            encoded = [

                self.token_vocab.get(
                    token,
                    self.token_vocab["<UNK>"]
                )

                for token in tokens

            ]

            node_tokens.append(
                encoded
            )

            node_types.append(

                self.cfg_vocab.get(

                    node.node_type,

                    self.cfg_vocab["<UNK>"]

                )

            )

        graph = Data(

            edge_index=torch.tensor(

                sample.pruned_cfg["edges"],

                dtype=torch.long

            ).t().contiguous()

        )

        graph.node_tokens = node_tokens

        graph.node_types = torch.tensor(

            node_types,

            dtype=torch.long

        )

        #
        # Use the same ID helpers.
        #
        graph.sample_id = self.get_sample_id(
            sample
        )

        graph.pair_id = self.get_pair_id(
            sample
        )

        graph.repo = sample.repo

        graph.file_path = sample.file_path

        graph.parent_commit = sample.parent_commit

        graph.label = sample.label

        sample.graph = graph

        return sample

    def get_sample_id(
        self,
        sample
    ):
    
        return (
        
            sample.repo,
            sample.parent_commit,
            sample.file_path,
            sample.label
    
        )
    
    
    def get_pair_id(
        self,
        sample
    ):
    
        return (
        
            sample.repo,
            sample.parent_commit,
            sample.file_path
    
        )