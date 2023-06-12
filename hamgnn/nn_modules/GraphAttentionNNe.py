import torch_geometric as torch_g
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops

import hamgnn.nn_modules.EncodeProcessDecodeNN as encodeDecode

#python3 terminal_scripts/train_model_variation.py --run-name gat_edge_4l --gpu 1 train_request_HamS_gpu_GATe

"""
in models_list.py append:

from hamgnn.nn_modules.GraphAttentionNNe import EncodeProcessDecodeAlgorithmGATedge

#GAT
train_request_HamS_gpu_GATe = copy.deepcopy(train_request_HamS_gpu_with_rand_node_encoding)
train_request_HamS_gpu_GATe.arguments["model_class"] = EncodeProcessDecodeAlgorithmGATedge
train_request_HamS_gpu_GATe.arguments["trainer_hyperparams"].update({"max_epochs": 2000})

#for 1 layer:
train_request_HamS_gpu_GATe.arguments["model_hyperparams"].update({"processor_depth": 1})
#for 2 layers:
train_request_HamS_gpu_GATe.arguments["model_hyperparams"].update({"processor_depth": 2})
#for 4 layers:
train_request_HamS_gpu_GATe.arguments["model_hyperparams"].update({"processor_depth": 4})
"""

#Graph Attention layer with edge distinction
class GATelayer(torch_g.nn.MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels

        self.heads = heads

        #w_f, w_b weights split in two, for calculations for ongoing and outgoing edges
        self.w_f = nn.Linear(2 * in_channels, out_channels * heads)
        nn.init.xavier_normal_(self.w_f.weight)

        self.w_b = nn.Linear(2 * in_channels, out_channels * heads)
        nn.init.xavier_normal_(self.w_b.weight)

        #att_f, att_b attentions split in two, for calculations for ongoing and outgoing edges
        self.att_f = nn.Parameter(torch.zeros(heads, out_channels, 1))
        nn.init.xavier_uniform_(self.att_f.data)

        self.att_b = nn.Parameter(torch.zeros(heads, out_channels, 1))
        nn.init.xavier_uniform_(self.att_b.data)

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)

        #adding the remaining self loops
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_attr=edge_weight, fill_value=1, num_nodes=num_nodes)

        out_f = self.directed_gate(x, edge_index, edge_weight, forward=True)
        out_b = self.directed_gate(x, edge_index, edge_weight, forward=False)

        #aggregation
        out_f = out_f.mean(dim=0).mean(dim=0)
        out_b = out_b.mean(dim=0).mean(dim=0)

        #final result
        out = out_f + out_b

        return out
    
    def directed_gate(self, x, edge_index, edge_weight, forward):
        num_nodes = x.size(0)

        if forward:
            row, col = edge_index
            new_edge_index = edge_index
        else:
            col, row = edge_index
            new_edge_index = torch.vstack((col, row))

        h = self.propagate(edge_index=new_edge_index, x=x, edge_attr=edge_weight, forward=forward)
 
        return h

    def message(self, x_j, x_i, forward, edge_attr):
        #adding edge weights
        edgeAttr = torch.cat((edge_attr[:, 0].unsqueeze(1),) * self.in_dim, dim=1)

        xi = x_i + edgeAttr
        xj = x_j + edgeAttr

        #concat
        e = torch.cat([xi, xj], dim=1)

        #apply weigths based on edge direction
        if forward:
            x = self.w_f(e)
        else:
            x = self.w_b(e)

        #apply LeakyReLU
        x = F.leaky_relu(x)

        x = x.reshape(self.heads, self.out_dim, x.size(0)).squeeze()

        #apply attention based on edge direction
        if forward:
            alpha = x * self.att_f
        else:
            alpha = x * self.att_b

        #apply softmax
        alpha = F.softmax(alpha, dim=1)

        return x_j * alpha.unsqueeze(-1)

#Graph Attention Neural Network with edge distinction with an arbitrary number of Graph Attention layers with edge distinction; after each of them the ReLU activation function is used    
class GATe(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads, nr_layers):
        assert nr_layers >= 1

        super(GATe, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.heads = heads
        self.nr_layers = nr_layers

        layers = []
        if (nr_layers == 1):
            layers += [GATelayer(in_dim, out_dim, heads=heads), nn.ReLU()]
        else:
            layers += [GATelayer(in_dim, hidden_dim, heads=heads), nn.ReLU()]
            for layer_index in range(nr_layers - 2):
                layers += [GATelayer(hidden_dim, hidden_dim, heads=heads), nn.ReLU()]
            layers += [GATelayer(hidden_dim, out_dim, heads=heads), nn.ReLU()]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weights):
        for layer in self.layers:
            if isinstance(layer, GATelayer):
                x = layer(x, edge_index, edge_weights)
            else:
                x = layer(x)
        return x

#overridden processor part with Graph Attention Network with edge distinction with 4 GAT layers with edge distinction
class EncodeProcessDecodeAlgorithmGATedge(encodeDecode.EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return GATe(in_dim=self.hidden_dim, hidden_dim=self.hidden_dim, out_dim=self.hidden_dim, heads=5, nr_layers=4)
    
