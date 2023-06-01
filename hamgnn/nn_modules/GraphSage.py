import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as torch_g
from torch_geometric.utils import add_remaining_self_loops

import hamgnn.nn_modules.EncodeProcessDecodeNN as encodeDecode

#python3 terminal_scripts/train_model_variation.py --run-name graphsage_4l --gpu 0 train_request_HamS_gpu_gsage

"""
in models_list.py append:

from hamgnn.nn_modules.GraphSage import EncodeProcessDecodeAlgorithmGraphSage

#GraphSage
train_request_HamS_gpu_gsage = copy.deepcopy(train_request_HamS_gpu_with_rand_node_encoding)
train_request_HamS_gpu_gsage.arguments["model_class"] = EncodeProcessDecodeAlgorithmGraphSage
train_request_HamS_gpu_gsage.arguments["model_hyperparams"].update({"processor_depth": 4})
train_request_HamS_gpu_gsage.arguments["trainer_hyperparams"].update({"max_epochs": 2000}) 
"""

class GraphSageLayer(torch_g.nn.MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='mean')
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Linear(2*in_dim, out_dim)
        nn.init.xavier_normal_(self.w.weight)

    def forward(self, x, edge_index, edge_weight):
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_weight)

        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):

        h = torch.cat([x, aggr_out], dim=1)
        h = self.w(h)
        h = F.normalize(aggr_out)

        return h

class GraphSage(torch.nn.Module):
    def __init__(self, dim, edges_dim=1, nr_layers=1):
        assert nr_layers >= 1

        super(GraphSage, self).__init__()

        self.dim = dim
        self.edges_dim = edges_dim
        self.nr_layers = nr_layers

        layers = []
        for layer_index in range(nr_layers):
            layers += [GraphSageLayer(in_dim=self.dim, out_dim=self.dim), nn.ReLU()]

        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weights):
        for layer in self.layers:
            if isinstance(layer, GraphSageLayer):
                x = layer(x, edge_index, edge_weights)
            else:
                x = layer(x)
        return x
    
class EncodeProcessDecodeAlgorithmGraphSage(encodeDecode.EncodeProcessDecodeAlgorithm):
    def _construct_processor(self):
        return GraphSage(dim=self.hidden_dim, nr_layers=4)
