import torch_geometric as torch_g
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops

import hamgnn.nn_modules.EncodeProcessDecodeNN as encodeDecode

#python3 terminal_scripts/train_model_variation.py --run-name gat_model --gpu 0 train_request_HamS_gpu_GAT

"""
in models_list.py append:

from hamgnn.nn_modules.GraphAttentionNN import EncodeProcessDecodeAlgorithmGAT

#GAT
train_request_HamS_gpu_GAT = copy.deepcopy(train_request_HamS_gpu_with_rand_node_encoding)
train_request_HamS_gpu_GAT.arguments["model_class"] = EncodeProcessDecodeAlgorithmGAT
train_request_HamS_gpu_GAT.arguments["model_hyperparams"].update({"processor_depth": 4})
train_request_HamS_gpu_GAT.arguments["trainer_hyperparams"].update({"max_epochs": 2000}) 
"""

class GATlayer(torch_g.nn.MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels

        self.heads = heads

        self.w = nn.Linear(2 * in_channels, out_channels * heads)
        nn.init.xavier_normal_(self.w.weight)

        self.att = nn.Parameter(torch.zeros(heads, out_channels, 1))
        nn.init.xavier_uniform_(self.att.data)

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_attr=edge_weight, fill_value=1, num_nodes=num_nodes)

        out = self.propagate(edge_index, x=x, edge_attr=edge_weight)

        out = out.mean(dim=0).mean(dim=0)

        return out

    def message(self, x_j, x_i, edge_attr):
        edgeAttr = torch.cat((edge_attr[:, 0].unsqueeze(1),) * self.in_dim, dim=1)

        xi = x_i + edgeAttr
        xj = x_j + edgeAttr

        e = torch.cat([xi, xj], dim=1)
        x = self.w(e)
        x = F.leaky_relu(x)

        x = x.reshape(self.heads, self.out_dim, x.size(0)).squeeze()
        alpha = x * self.att
        alpha = F.softmax(alpha, dim=1)

        return x_j * alpha.unsqueeze(-1)
    
class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads, nr_layers):
        assert nr_layers >= 1

        super(GAT, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.heads = heads
        self.nr_layers = nr_layers

        layers = []
        if (nr_layers == 1):
            layers += [GATlayer(in_dim, out_dim, heads=heads)]
        else:
            layers += [GATlayer(in_dim, hidden_dim, heads=heads)]
            for layer_index in range(nr_layers - 2):
                layers += [GATlayer(hidden_dim, hidden_dim, heads=heads)]
            layers += [GATlayer(hidden_dim, out_dim, heads=heads)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weights):
        for layer in self.layers:
            if isinstance(layer, GATlayer):
                x = layer(x, edge_index, edge_weights)
            else:
                x = layer(x)
        return x

class EncodeProcessDecodeAlgorithmGAT(encodeDecode.EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return GAT(in_dim=self.hidden_dim, hidden_dim=self.hidden_dim, out_dim=self.hidden_dim, heads=5, nr_layers=4)
