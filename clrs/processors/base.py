import abc
from typing import Tuple, Optional
from dataclasses import dataclass
import torch
from torch import nn
from torch import Tensor


@dataclass
class GraphFeatures:
    adj_mat: Tensor
    node_fts: Tensor
    edge_fts: Tensor
    graph_fts: Tensor

    def __add__(self, other: "GraphFeatures") -> "GraphFeatures":
        return GraphFeatures(self.adj_mat + other.adj_mat, 
                             self.node_fts + other.node_fts, 
                             self.edge_fts + other.edge_fts, 
                             self.graph_fts + other.graph_fts)
    
    def __iadd__(self, other: "GraphFeatures") -> "GraphFeatures":
        self.adj_mat += other.adj_mat
        self.node_fts += other.node_fts
        self.edge_fts += other.edge_fts
        self.graph_fts += other.graph_fts
        return self
    
    @classmethod
    def empty(cls, batch_size: int, nb_nodes: int, hidden_dim: int, device: torch.device = torch.device("cpu")) -> "GraphFeatures":
        return cls(adj_mat = torch.eye(nb_nodes, device=device).unsqueeze(0).repeat(batch_size, 1, 1), 
                   node_fts = torch.zeros((batch_size, nb_nodes, hidden_dim), device=device), 
                   edge_fts = torch.zeros((batch_size, nb_nodes, nb_nodes, hidden_dim), device=device), 
                   graph_fts = torch.zeros((batch_size, hidden_dim), device=device))
    

class Processor(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, graph_features: GraphFeatures, processor_state: Tensor)-> Tuple[Tensor, Optional[Tensor]]:
        """
        Processes the graph features.

        Args:
            graph_features: Graph features.
            processor_state: Processor state from the previous step.

        Returns:
            A tuple (next_processor_state, next_edge_fts (optional)).
            next_edge_fts can be None if the processor doesn't update them.
        """
        pass

    @property
    def inf_bias(self):
        return False

    @property
    def inf_bias_edge(self):
        return False
    
    @property
    def returns_edge_fts(self):
        return False

