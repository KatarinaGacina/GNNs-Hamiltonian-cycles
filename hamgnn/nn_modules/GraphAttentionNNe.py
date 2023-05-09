import torch_geometric as torch_g
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import add_remaining_self_loops

import hamgnn.nn_modules.EncodeProcessDecodeNN as encodeDecode

#python3 terminal_scripts/train_model_variation.py --run-name gat_model_edge --gpu 1 train_request_HamS_gpu_GATe

"""
in models_list.py append:

from hamgnn.nn_modules.GraphAttentionNNe import EncodeProcessDecodeAlgorithmGATedge

#GAT
train_request_HamS_gpu_GATe = copy.deepcopy(train_request_HamS_gpu_with_rand_node_encoding)
train_request_HamS_gpu_GATe.arguments["model_class"] = EncodeProcessDecodeAlgorithmGATedge
train_request_HamS_gpu_GATe.arguments["model_hyperparams"].update({"processor_depth": 2})
train_request_HamS_gpu_GATe.arguments["trainer_hyperparams"].update({"max_epochs": 2000}) 
"""

class GATelayer(torch_g.nn.MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()
        self.in_dim = in_channels
        self.out_dim = out_channels

        self.heads = heads

        self.w_f = nn.Linear(2 * in_channels, out_channels * heads)
        nn.init.xavier_normal_(self.w.weight)

        self.w_b = nn.Linear(2 * in_channels, out_channels * heads)
        nn.init.xavier_normal_(self.w.weight)

        self.att_f = nn.Parameter(torch.zeros(heads, out_channels, 1))
        nn.init.xavier_uniform_(self.att_f.data)

        self.att_b = nn.Parameter(torch.zeros(heads, out_channels, 1))
        nn.init.xavier_uniform_(self.att_b.data)

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_attr=edge_weight, fill_value=1, num_nodes=num_nodes)

        out_f = self.directed_gate(x, edge_index, edge_weight, forward=True)
        out_b = self.directed_gate(x, edge_index, edge_weight, forward=False)

        out = out_f + out_b

        out = out.mean(dim=0).mean(dim=0)
        out = F.relu(out)

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
        edgeAttr = torch.cat((edge_attr[:, 0].unsqueeze(1),) * self.in_dim, dim=1)

        xi = x_i + edgeAttr
        xj = x_j + edgeAttr

        e = torch.cat([xi, xj], dim=1)

        if forward:
            x = self.w_f(e)
        else:
            x = self.w_b(e)

        x = F.leaky_relu(x)

        x = x.reshape(self.heads, self.out_dim, x.size(0)).squeeze()

        if forward:
            alpha = x * self.att_f
        else:
            alpha = x * self.att_b

        alpha = F.softmax(alpha, dim=1)

        return x_j * alpha.unsqueeze(-1)
    
class GATe(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads): #in_dim = number of input features
        super(GATe, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.heads = heads

        self.gat1 = GATelayer(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATelayer(hidden_dim, out_dim, heads=1)
        self.elu = nn.ELU()
        self.sig = nn.Sigmoid()

    def forward(self, x, edge_index, edge_weights):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat1(x, edge_index, edge_weights)
        x = self.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index, edge_weights)
        x = self.sig(x)

        return x

class EncodeProcessDecodeAlgorithmGATedge(encodeDecode.EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return GATe(in_dim=self.hidden_dim, hidden_dim=self.hidden_dim, out_dim=self.hidden_dim, heads=2)