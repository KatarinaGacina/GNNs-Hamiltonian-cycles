import torch
import torch.nn as nn
import torch_geometric as torch_g
from torch_geometric.utils import add_remaining_self_loops, degree, to_torch_coo_tensor, to_torch_csc_tensor

import hamgnn.nn_modules.EncodeProcessDecodeNN as encodeDecode

#python3 terminal_scripts/train_model_variation.py --run-name gcn_model --gpu 0 train_request_HamS_gpu_GCN

"""
in models_list.py append:

from hamgnn.nn_modules.GraphConvolutionalNN import EncodeProcessDecodeAlgorithmGCN

#GCN
train_request_HamS_gpu_GCN = copy.deepcopy(train_request_HamS_gpu_with_rand_node_encoding)
train_request_HamS_gpu_GCN.arguments["model_class"] = EncodeProcessDecodeAlgorithmGCN
train_request_HamS_gpu_GCN.arguments["model_hyperparams"].update({"processor_depth": 2})
train_request_HamS_gpu_GCN.arguments["trainer_hyperparams"].update({"max_epochs": 100}) 
"""

class GCNConvLayer(torch_g.nn.MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim)

    def message(self, x_j, norm):
        return norm * x_j

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        #num_node_features = x.size(1)

        #adj_matrix = to_torch_coo_tensor(edge_index, edge_attr=edge_weight, size=(num_nodes, num_nodes))

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_attr=edge_weight, fill_value=1, num_nodes=num_nodes)

        x = self.lin(x)

        row, col = edge_index
        deg = degree(col, num_nodes, dtype=x.dtype).unsqueeze(1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)

        return out

class GCN(torch.nn.Module):
    def __init__(self, dim, edges_dim=1, nr_layers=2):
        assert nr_layers >= 1

        super(GCN, self).__init__()

        self.dim = dim
        self.edges_dim = edges_dim
        self.nr_layers = nr_layers

        self.layers = nn.ModuleList()
        for layer_index in range(nr_layers):
            self.layers.append(GCNConvLayer(dim, dim))

    def forward(self, x, edge_index, edge_weight):
        n = len(self.layers) - 1
        i = 0

        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)

            if (i < n):
                layer_relu = nn.ReLU()
                x = layer_relu(x)
            elif (i == n):
                layer_act = nn.Sigmoid()
                x = layer_act(x)

            i += 1

        return x
    
class EncodeProcessDecodeAlgorithmGCN(encodeDecode.EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return GCN(dim=self.hidden_dim)
    
