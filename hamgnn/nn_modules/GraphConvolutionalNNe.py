import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as torch_g
from torch_geometric.utils import add_remaining_self_loops, degree

import hamgnn.nn_modules.EncodeProcessDecodeNN as encodeDecode

#python3 terminal_scripts/train_model_variation.py --run-name gcn_edge_4l --gpu 1 train_request_HamS_gpu_GCNe

"""
in models_list.py append:

from hamgnn.nn_modules.GraphConvolutionalNNe import EncodeProcessDecodeAlgorithmGCNedge

#GCN_edge
train_request_HamS_gpu_GCNe = copy.deepcopy(train_request_HamS_gpu_with_rand_node_encoding)
train_request_HamS_gpu_GCNe.arguments["model_class"] = EncodeProcessDecodeAlgorithmGCNedge
train_request_HamS_gpu_GCNe.arguments["trainer_hyperparams"].update({"max_epochs": 2000})

#for 1 layer:
train_request_HamS_gpu_GCNe.arguments["model_hyperparams"].update({"processor_depth": 1})
#for 2 layers:
train_request_HamS_gpu_GCNe.arguments["model_hyperparams"].update({"processor_depth": 2})
#for 4 layers:
train_request_HamS_gpu_GCNe.arguments["model_hyperparams"].update({"processor_depth": 4})
"""

class GCNeConvLayer(torch_g.nn.MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin_f = nn.Linear(in_dim, out_dim)
        self.lin_b = nn.Linear(in_dim, out_dim)

    def directed_gate(self, x, edge_index, edge_weight, forward):
        num_nodes = x.size(0)

        if forward:
            row, col = edge_index
            new_edge_index = edge_index
        else:
            col, row = edge_index
            new_edge_index = torch.vstack((col, row))

        deg = degree(col, num_nodes, dtype=x.dtype).unsqueeze(1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
 
        h = self.propagate(edge_index=new_edge_index, x=x, norm=norm)
 
        return h

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_attr=edge_weight, fill_value=1, num_nodes=num_nodes)

        x_f = self.lin_f(x)
        x_b = self.lin_b(x)
        
        out_f = self.directed_gate(x_f, edge_index, edge_weight, forward=True)
        out_b = self.directed_gate(x_b, edge_index, edge_weight, forward=False)

        out = out_f + out_b

        return out

    def message(self, x_j, norm):
        return norm * x_j
    
class GCNe(torch.nn.Module):
    def __init__(self, dim, edges_dim=1, nr_layers=1):
        assert nr_layers >= 1

        super(GCNe, self).__init__()

        self.dim = dim
        self.edges_dim = edges_dim
        self.nr_layers = nr_layers

        layers = []
        for layer_index in range(nr_layers):
            layers += [GCNeConvLayer(dim, dim), nn.ReLU()]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers:
            if isinstance(layer, GCNeConvLayer):
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x)
        return x
    

class EncodeProcessDecodeAlgorithmGCNedge(encodeDecode.EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return GCNe(dim=self.hidden_dim, nr_layers=4)
