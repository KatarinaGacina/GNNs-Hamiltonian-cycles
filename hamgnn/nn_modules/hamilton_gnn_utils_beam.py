from hamgnn.HamiltonSolver import HamiltonSolver
from hamgnn.nn_modules.hamilton_gnn_utils import HamFinderGNN, WalkUpdater
import torch_geometric as torch_g

#beam search
class BeamSearchWrapper2(HamFinderGNN):
    def __init__(self, graph_updater_class: WalkUpdater,
                inference_batch_size=8, starting_learning_rate=1e-4, optimizer_class=None, optimizer_hyperparams=None, lr_scheduler_class=None, lr_scheduler_hyperparams=None, val_dataloader_tags=None, beam_width=5):
        super().__init__(graph_updater_class,
                inference_batch_size, starting_learning_rate, optimizer_class, optimizer_hyperparams, lr_scheduler_class, lr_scheduler_hyperparams, val_dataloader_tags)
        #beam width initialization
        self.beam_width = beam_width

    def solve_graphs(self, graphs: list[torch_g.data.Data]) -> list[list[int]]:
        results = []
        for g in graphs:
            results.append(self.run_beam_search(g, beam_width=self.beam_width))
        return results
    
    def _compute_and_update_accuracy_metrics(self, graph_batch_dict, accuracy_metrics_tag):
        batch_graph, _ = self._unpack_graph_batch_dict(graph_batch_dict)
        graph_list = batch_graph.to_data_list()
        walks = self.solve_graphs(graph_list)
        self.update_accuracy_metrics(accuracy_metrics_tag, graph_list, walks)
