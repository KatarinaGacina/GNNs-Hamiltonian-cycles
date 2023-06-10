import torch
import torch.nn as nn
import torch_geometric as torch_g
from torch_geometric.utils import add_remaining_self_loops, degree, to_torch_coo_tensor, to_torch_csc_tensor

import hamgnn.nn_modules.EncodeProcessDecodeRandFeatures as encodeDecode

#python3 terminal_scripts/train_model_variation.py --run-name gcn_4l --gpu 1 train_request_HamS_gpu_GCN

"""
in models_list.py append:

from hamgnn.nn_modules.GraphConvolutionalNN import EncodeProcessDecodeAlgorithmGCN

#GCN
train_request_HamS_gpu_GCN = copy.deepcopy(train_request_HamS_gpu_with_rand_node_encoding)
train_request_HamS_gpu_GCN.arguments["model_class"] = EncodeProcessDecodeAlgorithmGCN
train_request_HamS_gpu_GCN.arguments["trainer_hyperparams"].update({"max_epochs": 2000})

#for 1 layer:
train_request_HamS_gpu_GCN.arguments["model_hyperparams"].update({"processor_depth": 1})
#for 2 layers:
train_request_HamS_gpu_GCN.arguments["model_hyperparams"].update({"processor_depth": 2})
#for 4 layers:
train_request_HamS_gpu_GCN.arguments["model_hyperparams"].update({"processor_depth": 4})
"""

#Graph Convolutional layer
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

        #adding the remaining self loops
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_attr=edge_weight, fill_value=1, num_nodes=num_nodes)
        #adj_matrix = to_torch_coo_tensor(edge_index, edge_attr=edge_weight, size=(num_nodes, num_nodes))

        #applying linear layer to node features
        x = self.lin(x)

        #calculating the normalization
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=x.dtype).unsqueeze(1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, norm=norm)

        return out

#Graph Convolutional Neural Network with an arbitrary number of Graph Convolutional layers; after each of them the ReLU activation function is used
class GCN(torch.nn.Module):
    def __init__(self, dim, edges_dim=1, nr_layers=1):
        assert nr_layers >= 1

        super(GCN, self).__init__()

        self.dim = dim
        self.edges_dim = edges_dim
        self.nr_layers = nr_layers

        layers = []
        for layer_index in range(nr_layers):
            layers += [GCNConvLayer(dim, dim), nn.ReLU()]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers:
            if isinstance(layer, GCNConvLayer):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x

#overridden processor part with Graph Convolutional Network with 4 GCN layers
class EncodeProcessDecodeAlgorithmGCN(encodeDecode.EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return GCN(dim=self.hidden_dim, nr_layers=4)
