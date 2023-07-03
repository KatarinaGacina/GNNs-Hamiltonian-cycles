import networkx as nx
import igraph
import math
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch_geometric as torch_g
import torchinfo

from hamgnn.nn_modules.nn_modules import ResidualMultilayerMPNN
import hamgnn.nn_modules.hamilton_gnn_utils as gnn_utils
from hamgnn.data.GraphDataset import get_shifts_for_graphs_in_batch

class EncodeProcessDecodeAlgorithm(gnn_utils.HamFinderGNN):
    def _construct_processor(self):
        return ResidualMultilayerMPNN(dim=self.hidden_dim, message_dim=self.hidden_dim, edges_dim=1,
                                      nr_layers=self.processor_depth)

    def _construct_encoder_and_decoder(self):
        encoder_nn = torch.nn.Sequential(torch.nn.Linear(3 + self.hidden_dim, self.hidden_dim))
        decoder_nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim + self.hidden_dim, self.out_dim))
        return encoder_nn, decoder_nn

    def __init__(self, processor_depth=3, in_dim=1, out_dim=1, hidden_dim=32, graph_updater_class=gnn_utils.WalkUpdater,
                 loss_type="mse", **kwargs):
        super().__init__(graph_updater_class, **kwargs)
        self.save_hyperparameters()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.processor_depth = processor_depth
        assert loss_type in ["mse", "entropy"]
        self.loss_type = loss_type

        self.initial_h = torch.nn.Parameter(torch.rand(self.hidden_dim))
        self.encoder_nn, self.decoder_nn = self._construct_encoder_and_decoder()
        self.processor_nn = self._construct_processor()

    def description(self):
        return f"encoder: {torchinfo.summary(self.encoder_nn, verbose=0, depth=5)}\n" \
               f"processor (hidden dim={self.hidden_dim}):" \
               f" {torchinfo.summary(self.processor_nn, verbose=0, depth=5)}\n" \
               f"decoder: {torchinfo.summary(self.decoder_nn, verbose=0, depth=5)}"

    def raw_next_step_logits(self, d: torch_g.data.Data):
        d.z = self.encoder_nn(torch.cat([d.x, d.h], dim=-1))
        d.h = self.processor_nn(d.z, d.edge_index, d.edge_attr)
        return torch.squeeze(self.decoder_nn(torch.cat([d.z, d.h], dim=-1)), dim=-1)
    
    def init_graph(self, d):
        d.h = torch.stack([self.initial_h for _ in range(d.num_nodes)], dim=-2)

        #set edge_attr using ForceAtlas2
        if d.edge_attr is None:
            edge_index = d.edge_index
            ptr = d.ptr  
            h = d.h
            
            """
            #alternative
            graph = nx.DiGraph()
    
            graph.add_nodes_from(range(ptr[0], ptr[-1]))

            graph_edges = edge_index
            edges_list = graph_edges.transpose(0, 1).tolist()
            graph.add_edges_from(edges_list)
    
            graph_features = h
            features_dict = {node: features.tolist() for node, features in zip(range(ptr[0], ptr[-1]), graph_features)}
            nx.set_node_attributes(graph, features_dict, name='features')

            forceatlas2 = ForceAtlas2(
                # Behavior alternatives
                outboundAttractionDistribution=True,  # Dissuade hubs
                linLogMode=False,  # NOT IMPLEMENTED
                adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                edgeWeightInfluence=1.0,

                # Performance
                jitterTolerance=1.0,  # Tolerance
                barnesHutOptimize=True,
                barnesHutTheta=1.2,
                multiThreaded=False,  # NOT IMPLEMENTED

                # Tuning
                scalingRatio=2.0,
                strongGravityMode=False,
                gravity=1.0,

                # Log
                verbose=True)
            
            layout = forceatlas2.forceatlas2_networkx_layout(G=graph, pos=None, iterations=200)

            edge_attr = []
            for x, y in edges_list:  
                x1, y1 = layout[x]
                x2, y2 = layout[y]
                
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                edge_attr.append([distance])

            d.edge_attr = torch.tensor(edge_attr).to(self.device)
            """

            sections = torch.bincount(d.batch)
            node_features = torch.split(h, tuple(sections.tolist()))

            row, col = edge_index
            sections = torch.bincount(d.batch[row])
            rows = torch.split(row, tuple(sections.tolist()))
            cols = torch.split(col, tuple(sections.tolist()))

            edge_attr = []
            for i in range(len(ptr) - 1):
                start_id = ptr[i]
                end_id = ptr[i + 1]
    
                graph = nx.DiGraph()
    
                graph.add_nodes_from(range(start_id, end_id))

                list_edges = [rows[i].tolist(), cols[i].tolist()]
                graph_edges = torch.tensor(list_edges)
                edges_list = graph_edges.transpose(0, 1).tolist()
                graph.add_edges_from(edges_list)
    
                graph_features = node_features[i]
                features_dict = {node: features.tolist() for node, features in zip(range(start_id, end_id), graph_features)}
                nx.set_node_attributes(graph, features_dict, name='features')

                forceatlas2 = ForceAtlas2(
                    # Behavior alternatives
                    outboundAttractionDistribution=True,  # Dissuade hubs
                    linLogMode=False,  # NOT IMPLEMENTED
                    adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                    edgeWeightInfluence=0,

                    # Performance
                    jitterTolerance=1.0,  # Tolerance
                    barnesHutOptimize=True,
                    barnesHutTheta=1.2,
                    multiThreaded=False,  # NOT IMPLEMENTED

                    # Tuning
                    scalingRatio=2.0,
                    strongGravityMode=False,
                    gravity=1.0,

                    # Log
                    verbose=False)

                layout = forceatlas2.forceatlas2_networkx_layout(G=graph, pos=None, iterations=200)

                for x, y in edges_list:  
                    x1, y1 = layout[x]
                    x2, y2 = layout[y]
                
                    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    edge_attr.append([distance])
        
            d.edge_attr = torch.tensor(edge_attr).to(self.device)

    def forward(self, d: torch_g.data.Data):
        return self.next_step_logits_masked_over_neighbors(d)

    def _training_step_procedure(self, graph_batch_dict):
        batch_graph, teacher_paths = self._unpack_graph_batch_dict(graph_batch_dict)

        class TrainingRunInstructions(self._RunInstructions):
            def __init__(self, gnn_model: gnn_utils.HamFinderGNN, graph_batch_dict) -> None:
                self.gnn_model: gnn_utils.HamFinderGNN = gnn_model
                self.batch_graph, teacher_paths = self.gnn_model._unpack_graph_batch_dict(graph_batch_dict)
                self.loss = torch.zeros(1, device=self.batch_graph.edge_index.device)
                graph_shifts = get_shifts_for_graphs_in_batch(batch_graph)
                self.teacher_tensor = torch.stack(teacher_paths, 0)
                self._logits_per_step = []

                if self.gnn_model.loss_type == "mse":
                    mse_weights = F.normalize(F.one_hot(batch_graph.batch, batch_graph.num_graphs).float(), 1, 0).sum(-1)
                    def _compute_loss(logits, probabilities, step_number, is_algorithm_stopped_mask):
                        self._logits_per_step.append(logits)
                        mse_loss = torch.nn.MSELoss(reduction="none")
                        teacher_p = torch.zeros_like(probabilities)
                        teacher_p[self.teacher_tensor[:, step_number]] = 1
                        return (mse_loss(probabilities, teacher_p) * mse_weights).sum()
                elif self.gnn_model.loss_type == "entropy":
                    def _compute_loss(logits, probabilites, step_number, is_algorithm_stopped_mask):
                        self._logits_per_step.append(logits)
                        entropy_loss = torch.nn.CrossEntropyLoss()
                        subgraph_losses = []
                        for subgraph_index in range(len(graph_shifts)):
                            graph_start_index = graph_shifts[subgraph_index]
                            graph_end_index = graph_shifts[subgraph_index+1] if subgraph_index + 1 < len(graph_shifts) else self.batch_graph.num_nodes
                            _graph_logits = logits[graph_start_index: graph_end_index].unsqueeze(0)
                            subgraph_losses.append(
                                entropy_loss(_graph_logits, self.teacher_tensor[subgraph_index: subgraph_index + 1, step_number] - graph_start_index))
                        return torch.stack(subgraph_losses).sum()
                self.compute_loss = _compute_loss

            def on_algorithm_start_fn(self, batch_graph):
                self.gnn_model.init_graph(batch_graph)
                self.gnn_model.prepare_for_first_step(batch_graph, None)
                _ = self.gnn_model.raw_next_step_logits(batch_graph)
                self.gnn_model.graph_updater.batch_prepare_for_first_step(batch_graph, self.teacher_tensor[:, 0])
                start_index = 1
                return start_index

            def choose_next_step_fn(self, batch_graph, current_nodes, step_number, is_algorithm_stopped_mask):
                logits = self.gnn_model.raw_next_step_logits(batch_graph)
                probabilites = self.gnn_model.scattered_logits_to_probabilities(batch_graph, logits)
                self.loss += self.compute_loss(logits, probabilites, step_number, is_algorithm_stopped_mask)
                return self.teacher_tensor[:, step_number]

            def update_algorithm_stopped_mask(self, all_previous_choices, current_choice, is_algorithm_stopped_mask, step_number):
                # Quickfix to stop training if batch dim is 1 and teacher path is shorter than graph length.
                if step_number == self.teacher_tensor.shape[-1] - 1:
                    return torch.full_like(is_algorithm_stopped_mask, True)
                else:
                    return is_algorithm_stopped_mask

        run_instructions = TrainingRunInstructions(self, graph_batch_dict)
        self._run_on_graph_batch(batch_graph, run_instructions)

        avg_loss = run_instructions.loss / run_instructions.teacher_tensor.nelement()
        self.logits_per_step = run_instructions._logits_per_step
        return avg_loss

    def training_step(self, graph_batch_dict):
        avg_loss = self._training_step_procedure(graph_batch_dict)
        self.log("train/loss", avg_loss)
        return avg_loss

    def validation_step(self, graph_batch_dict, batch_idx, dataloader_idx=0):
        dataloader_tag = self.get_validation_dataloader_tag(dataloader_idx)
        with torch.no_grad():
            loss = self._training_step_procedure(graph_batch_dict)
            self.log(f"{dataloader_tag}/loss", loss)
            self._compute_and_update_accuracy_metrics(graph_batch_dict, dataloader_tag)
            return loss
