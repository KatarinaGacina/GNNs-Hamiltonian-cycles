import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as torch_g
from torch_geometric.utils import add_remaining_self_loops

import hamgnn.nn_modules.EncodeProcessDecodeNN as encodeDecode

#python3 terminal_scripts/train_model_variation.py --run-name graphsage_edge_4l --gpu 0 train_request_HamS_gpu_gsagee

"""
in models_list.py append:

from hamgnn.nn_modules.GraphSageE import EncodeProcessDecodeAlgorithmGraphSageE

#GraphSageEdge
train_request_HamS_gpu_gsagee = copy.deepcopy(train_request_HamS_gpu_with_rand_node_encoding)
train_request_HamS_gpu_gsagee.arguments["model_class"] = EncodeProcessDecodeAlgorithmGraphSageE
train_request_HamS_gpu_gsagee.arguments["model_hyperparams"].update({"processor_depth": 4})
train_request_HamS_gpu_gsagee.arguments["trainer_hyperparams"].update({"max_epochs": 2000}) 
"""

class GraphSageELayer(torch_g.nn.MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='mean')
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w_f = nn.Linear(2*in_dim, out_dim)
        nn.init.xavier_normal_(self.w_f.weight)

        self.w_b = nn.Linear(2*in_dim, out_dim)
        nn.init.xavier_normal_(self.w_b.weight)

    def forward(self, x, edge_index, edge_weight):
        col, row = edge_index
        new_edge_index = torch.vstack((col, row))
        
        out_f = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_weight, forward=True)
        out_b = self.propagate(edge_index=new_edge_index, x=x, edge_attr=edge_weight, forward=False)

        out = out_f + out_b

        return out

    def message(self, x_j, edge_attr):
        return x_j

    def update(self, aggr_out, x, forward):

        h = torch.cat([x, aggr_out], dim=1)

        if (forward):
            h = self.w_f(h)
        else:
            h = self.w_b(h)

        h = F.normalize(aggr_out)

        return h

class GraphSageE(torch.nn.Module):
    def __init__(self, dim, edges_dim=1, nr_layers=1):
        assert nr_layers >= 1

        super(GraphSageE, self).__init__()

        self.dim = dim
        self.edges_dim = edges_dim
        self.nr_layers = nr_layers

        layers = []
        for layer_index in range(nr_layers):
            layers += [GraphSageELayer(in_dim=self.dim, out_dim=self.dim), nn.ReLU()]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weights):
        for layer in self.layers:
            if isinstance(layer, GraphSageLayer):
                x = layer(x, edge_index, edge_weights)
            else:
                x = layer(x)
        return x
    
    
class EncodeProcessDecodeAlgorithmGraphSageE(encodeDecode.EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return GraphSageE(dim=self.hidden_dim, nr_layers=1)
