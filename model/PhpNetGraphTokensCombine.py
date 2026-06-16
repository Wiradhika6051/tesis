import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torch_geometric.nn import GCNConv, EdgePooling
from torch_geometric.nn import  global_max_pool
class PhpNetGraphTokensCombineFull(nn.Module):
    def __init__(self):
        super(PhpNetGraphTokensCombineFull, self).__init__()
        self.embed1 = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=100)
        self.cfg_head = nn.Sequential(
            nn.Linear(4000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 4)
        )
        self.tok_head = nn.Sequential(
            nn.Linear(1200, 1000),
            nn.ReLU(),
            nn.Linear(1000, 4)
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(5200, 1000),
            nn.ReLU(),
            nn.Linear(1000, 4)
        )

        self.conv1 = GCNConv(100,2000)
        self.pool1 = EdgePooling(2000)
        self.conv2 = GCNConv(2000, 4000)
        self.pool2 = EdgePooling(4000)
        self.conv3 = GCNConv(4000, 4000)
        self.pool3 = EdgePooling(4000)
        self.conv4 = GCNConv(2000, 2000)
        self.pool4 = EdgePooling(8000)

        #
        self.embed = nn.Embedding(num_embeddings=5000,
                                  embedding_dim=100)
        self.lstm1 = nn.GRU(input_size=100,
                            hidden_size=200,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=True)



        self.lin1 = nn.Linear(5200, 1000)
        self.lin11 = nn.Linear(1000, 500)
        self.lin2 = nn.Linear(500, 4)
        self.lin3 = nn.Linear(200, 100)
        self.lin4 = nn.Linear(100,4)

    def forward(self, cfg_emb, tok_emb):
        x = torch.cat([cfg_emb, tok_emb], dim=1)
        return self.fusion_head(x)

    
    def encode_tokens(self, dataTokens):
        x = self.embed(dataTokens)
        _, hidden = self.lstm1(x)

        token_emb = torch.cat(
            [hidden[i] for i in range(hidden.shape[0])],
            dim=1
        )
        return token_emb   
    
    def encode_graph(self, dataGraph):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
    
        # 👇 FIX: handle single-graph case
        if hasattr(dataGraph, "batch"):
            batch = dataGraph.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    
        # pre_x_len = len(x)
        # x = self.embed1(x).reshape(pre_x_len, -1)

        x = self.embed1(x)      # [N, T, 100]
        x = x.mean(dim=1)       # [N, 100]

    
        x = self.conv1(x, edge_index)
        x, edge_index, batch, _ = self.pool1(x, edge_index, batch=batch)
        x = F.relu(x)
    
        x = self.conv2(x, edge_index)
        x, edge_index, batch, _ = self.pool2(x, edge_index, batch=batch)
        x = F.relu(x)
    
        x = self.conv3(x, edge_index)
        x, edge_index, batch, _ = self.pool3(x, edge_index, batch=batch)
        x = F.relu(x)
    
        x = global_max_pool(x, batch)
        return x


    def _encode_graph(self, dataGraph):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        print(dataGraph)
        batch = dataGraph.batch

        pre_x_len = len(x)
        x = self.embed1(x).reshape(pre_x_len, -1)

        x = self.conv1(x, edge_index)
        x, edge_index, batch, _ = self.pool1(x, edge_index, batch=batch)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x, edge_index, batch, _ = self.pool2(x, edge_index, batch=batch)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x, edge_index, batch, _ = self.pool3(x, edge_index, batch=batch)
        x = F.relu(x)

        x = global_max_pool(x, batch)
        return x    

class PhpNetGraphTokensCombine(nn.Module):
    def __init__(self, vocab_size):
        super(PhpNetGraphTokensCombine, self).__init__()
        self.embed1 = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=100)
        self.cfg_head = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )
        self.tok_head = nn.Sequential(
            nn.Linear(1200, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2)
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(640,512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,2)
        )

        self.conv1 = GCNConv(100,128)
        self.pool1 = EdgePooling(128)
        self.conv2 = GCNConv(128, 256)
        self.pool2 = EdgePooling(256)
        self.conv3 = GCNConv(256, 256)
        self.pool3 = EdgePooling(256)

        #
        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=100)
        self.lstm1 = nn.GRU(input_size=100,
                            hidden_size=64,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, cfg_emb, tok_emb):
        print(cfg_emb.shape)
        print(tok_emb.shape)
        x = torch.cat([cfg_emb, tok_emb], dim=1)
        return self.fusion_head(x)

    
    def encode_tokens(self, dataTokens):
        x = self.embed(dataTokens)
        _, hidden = self.lstm1(x)

        token_emb = torch.cat(
            [hidden[i] for i in range(hidden.shape[0])],
            dim=1
        )
        return token_emb   
    
    def encode_graph(self, dataGraph):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
    
        # handle single-graph case
        if hasattr(dataGraph, "batch"):
            batch = dataGraph.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    
        # pre_x_len = len(x)
        # x = self.embed1(x).reshape(pre_x_len, -1)

        # x = self.embed1(x)      # [N, T, 100]
        # x = x.mean(dim=1)       # [N, 100]
        x = self.embed1(x.squeeze(-1))

    
        x = self.conv1(x, edge_index)
        x, edge_index, batch, _ = self.pool1(x, edge_index, batch=batch)
        x = F.relu(x)
    
        x = self.conv2(x, edge_index)
        x, edge_index, batch, _ = self.pool2(x, edge_index, batch=batch)
        x = F.relu(x)
    
        x = self.conv3(x, edge_index)
        x, edge_index, batch, _ = self.pool3(x, edge_index, batch=batch)
        x = F.relu(x)
    
        x = global_max_pool(x, batch)
        return x


    def _encode_graph(self, dataGraph):
        x, edge_index = dataGraph.x.long(), dataGraph.edge_index
        print(dataGraph)
        batch = dataGraph.batch

        pre_x_len = len(x)
        x = self.embed1(x).reshape(pre_x_len, -1)

        x = self.conv1(x, edge_index)
        x, edge_index, batch, _ = self.pool1(x, edge_index, batch=batch)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x, edge_index, batch, _ = self.pool2(x, edge_index, batch=batch)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x, edge_index, batch, _ = self.pool3(x, edge_index, batch=batch)
        x = F.relu(x)

        x = global_max_pool(x, batch)
        return x    
