from enum import Enum
from typing import List, Optional
from ..utils import Linear, batch_mask, expand, POS_INF, NEG_INF
from .base import ProcessorBase, GraphFeatures
from torch import nn
import torch
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F

class Reduction(str, Enum):
    MAX = "max"
    MIN = "min"
    MEAN = "mean"

class TripletNN(nn.Module):
    def __init__(self, 
                 node_feat_size: int, 
                 edge_feat_size: int, 
                 graph_feat_size: int, 
                 triplet_feature_size: int, 
                 out_features: int,
                 activation: Optional[nn.Module] = nn.ReLU()):
        super().__init__()
        assert triplet_feature_size is not None, "triplet_feature_size must be provided if use_triplets is True"
        self.triplet_fc_z1 = Linear(node_feat_size, triplet_feature_size)
        self.triplet_fc_z2 = Linear(node_feat_size, triplet_feature_size)
        self.triplet_fc_z3 = Linear(node_feat_size, triplet_feature_size)
        self.triplet_fc_e1 = Linear(edge_feat_size, triplet_feature_size)
        self.triplet_fc_e2 = Linear(edge_feat_size, triplet_feature_size)
        self.triplet_fc_e3 = Linear(edge_feat_size, triplet_feature_size)
        self.triplet_fc_g = Linear(graph_feat_size, triplet_feature_size)
        self.fc_triplet_out = Linear(triplet_feature_size, out_features)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, z: Tensor, edge_feats: Tensor, graph_feats: Tensor, num_nodes: Optional[Tensor] = None) -> Tensor:
        B, N, _ = z.size()
        tri_z1 = self.triplet_fc_z1(z) # [B, N, T]
        tri_z2 = self.triplet_fc_z2(z) # [B, N, T]
        tri_z3 = self.triplet_fc_z3(z) # [B, N, T]
        tri_e1 = self.triplet_fc_e1(edge_feats) # [B, N, N, T]
        tri_e2 = self.triplet_fc_e2(edge_feats) # [B, N, N, T]
        tri_e3 = self.triplet_fc_e3(edge_feats) # [B, N, N, T]
        tri_g = self.triplet_fc_g(graph_feats) # [B, T]

        triplet_tensor =  (
            tri_z1.unsqueeze(2).unsqueeze(3) +              # [B, N, 1, 1, T]
            tri_z2.unsqueeze(1).unsqueeze(3) +              # [B, 1, N, 1, T]
            tri_z3.unsqueeze(1).unsqueeze(2) +              # [B, 1, 1, N, T]
            tri_e1.unsqueeze(3) +                           # [B, N, N, 1, T]
            tri_e2.unsqueeze(2) +                           # [B, N, 1, N, T]
            tri_e3.unsqueeze(1) +                           # [B, 1, N, N, T]
            tri_g.unsqueeze(1).unsqueeze(2).unsqueeze(3)    # [B, 1, 1, 1, T]
        )
        

        if num_nodes is not None:
            # Create mask for valid nodes and apply it consistently
            triplet_mask = expand(batch_mask(num_nodes, N, 3), triplet_tensor) #  [B, N, N, N, T]                      # Apply mask and ensure consistent numerical behavior regardless of padding
            masked_triplet = triplet_tensor.masked_fill(~triplet_mask, NEG_INF) # [B, N, N, N, T]
            tri_max, _ = masked_triplet.max(dim=1) # [B, N, N, T]
            edge_mask = batch_mask(num_nodes, N, 2).unsqueeze(-1) # [B, N, N, 1]
            tri_max = tri_max.masked_fill(~edge_mask, 0.0) # [B, N, N, T]
        else:
            tri_max = triplet_tensor.max(dim=1)[0] # [B, N, N, T]            

        tri_msgs = self.fc_triplet_out(tri_max) # [B, N, N, O]
        tri_msgs = self.activation(tri_msgs) # [B, N, N, O]

        if num_nodes is not None:
            tri_msgs = tri_msgs.masked_fill(~edge_mask, 0.0)

        return tri_msgs

class PGN(ProcessorBase):
    """
    Pointer Graph Network (Veličković et al., NeurIPS 2020) with optional
    triplet messages (Dudzik & Veličković, 2022).

    Args:
        out_size (int): Dimensionality of both hidden and output node features.
        node_feat_size (int): Dimensionality of input node features.
        edge_feat_size (int): Dimensionality of input edge features. Defaults to node_feat_size.
        graph_feat_size (int): Dimensionality of input global graph features. Defaults to node_feat_size.
        mid_size (int, optional): Size of intermediate message features; defaults to out_size.
        mid_activation (callable, optional): Activation applied after message MLP.
        activation (callable, optional): Activation applied after combining h1 & h2; defaults to F.relu.
        reduction (str or callable): 'mean', 'max', or a custom function; reduction over neighbors.
        msgs_mlp_sizes (list[int], optional): Hidden sizes for MLP applied to raw messages.
        use_layer_norm (bool): Apply LayerNorm at the end.
        use_triplets (bool): Enable triplet-based messages.
        triplet_feature_size (int): Feature size for triplet message components.
        gated (bool): Use gated update mechanism.
    """
    def __init__(
        self,
        hidden_size: int,
        node_feat_size: int,
        edge_feat_size: Optional[int] = None,
        graph_feat_size: Optional[int] = None,
        mid_size: Optional[int] = None,
        mid_activation: Optional[nn.Module] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        reduction: Reduction = Reduction.MAX,
        msgs_mlp_sizes: Optional[List[int]] = None,
        use_layer_norm: bool = False,
        use_triplets: bool = False,
        triplet_feature_size: Optional[int] = None,
        gated: bool = False,
        mp_steps: int = 1,
    ):
        super().__init__()
        # Using a single size for hidden and output states simplifies iterative builds
        self.hidden_size = hidden_size
        self.mid_size = mid_size or hidden_size
        self.mid_activation = mid_activation if mid_activation is not None else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        self.reduction = reduction
        self.msgs_mlp_sizes = msgs_mlp_sizes
        self.use_triplets = use_triplets
        self.gated = gated
        self.mp_steps = mp_steps

        if edge_feat_size is None:
            edge_feat_size = node_feat_size
        if graph_feat_size is None:
            graph_feat_size = node_feat_size

        if use_triplets:
            assert triplet_feature_size is not None, "triplet_feature_size must be provided if use_triplets is True"

        # Dimension for concatenated node and hidden features (hidden == out_size)
        z_feat_size = node_feat_size + self.hidden_size

        # Message linear transforms
        self.fc_source = Linear(z_feat_size, self.mid_size)
        self.fc_target = Linear(z_feat_size, self.mid_size)
        self.fc_edge = Linear(edge_feat_size, self.mid_size)
        self.fc_graph = Linear(graph_feat_size, self.mid_size)

        # Output transforms
        self.fc_out_source = Linear(z_feat_size, self.hidden_size)
        self.fc_out_messages = Linear(self.mid_size, self.hidden_size)

        # Optional MLP on raw messages
        if self.msgs_mlp_sizes:
            assert self.msgs_mlp_sizes[-1] == self.mid_size, "Last layer of msgs_mlp must match mid_size"
            layers = []
            layers.append(nn.ReLU())
            in_size = self.mid_size
            for idx, h in enumerate(self.msgs_mlp_sizes):
                layers.append(Linear(in_size, h))
                if idx < len(self.msgs_mlp_sizes) - 1:
                    layers.append(nn.ReLU())
                else:
                    layers.append(self.mid_activation)
                in_size = h
            self.msgs_mlp = nn.Sequential(*layers)
        else:
            self.msgs_mlp = self.mid_activation

        # Layer normalization after update
        self.layer_norm = nn.LayerNorm(self.hidden_size) if use_layer_norm else nn.Identity()

        # Gating mechanism
        if gated:
            self.gate_fc1 = Linear(z_feat_size, self.hidden_size)
            self.gate_fc2 = Linear(self.mid_size, self.hidden_size)
            self.gate_fc3 = Linear(self.hidden_size, self.hidden_size)
            nn.init.constant_(self.gate_fc3.bias, -3.0)

        # Triplet-based message components
        if use_triplets:
            assert triplet_feature_size is not None, "triplet_feature_size must be provided if use_triplets is True"
            self.triplet_nn = TripletNN(node_feat_size=z_feat_size, 
                                        edge_feat_size=edge_feat_size, 
                                        graph_feat_size=graph_feat_size, 
                                        triplet_feature_size=triplet_feature_size, 
                                        out_features=self.hidden_size,
                                        activation=self.activation)

    @property
    def returns_edge_fts(self):
        return self.use_triplets

    def forward(self, graph_features: GraphFeatures, processor_state: Tensor, num_nodes: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        B, N, _ = graph_features.node_fts.size()
        assert processor_state.size(-1) == self.hidden_size, \
            f"Hidden dimension ({processor_state.size(-1)}) must equal out_size ({self.out_size})"
        

        m_edge = self.fc_edge(graph_features.edge_fts) # [B, N, N, M]
        m_graph = self.fc_graph(graph_features.graph_fts) # [B, M]

        if num_nodes is not None:
            edge_mask = batch_mask(num_nodes, N, 2) # [B, N, N]
            node_mask = batch_mask(num_nodes, N, 1) # [B, N]

        for step in range(self.mp_steps):
            # Combine node and hidden features
            z = torch.cat([graph_features.node_fts, processor_state], dim=-1) # [B, N, (O+N)]

            # Standard message components
            m_src = self.fc_source(z) # [B, N, M]
            m_tgt = self.fc_target(z) # [B, N, M]
 
            # Triplet messages calculation (optional)
            tri_msgs = None
            if self.use_triplets and step == self.mp_steps - 1: # compute higher over triplets only on the last step
                tri_msgs = self.triplet_nn(z, graph_features.edge_fts, graph_features.graph_fts, num_nodes) # [B, N, N, O]

            # Combine and transform messages
            msgs = (
                m_src.unsqueeze(1) + # [B, 1, N, M]
                m_tgt.unsqueeze(2) + # [B, N, 1, M]
                m_edge + # [B, N, N, M]
                m_graph.unsqueeze(1).unsqueeze(2) # [B, 1, 1, M]
            )

            msgs = self.msgs_mlp(msgs)  # [B, N, N, M]
            if self.reduction == Reduction.MEAN:
                adj = graph_features.adj_mat.to(msgs.dtype) # [B, N, N]
                if num_nodes is not None:
                    adj = adj * edge_mask
                # weighted sum over the neighbours, then divide by the number of neighbours
                weighted_sum  = (msgs * adj.unsqueeze(-1)).sum(dim=1) # [B, N, M]
                counts = adj.sum(dim=-1, keepdim=True).clamp(min=1) # [B, N, 1]
                msgs = weighted_sum / counts # [B, N, M]
            elif self.reduction == Reduction.MAX or self.reduction == Reduction.MIN:
                mask = graph_features.adj_mat == 0 # [B, N, N, 1]
                if num_nodes is not None:
                    mask = mask | ~edge_mask  
                fill_value = float('-inf') if self.reduction == Reduction.MAX else float('inf')
                msgs = msgs.masked_fill(mask.unsqueeze(-1), fill_value) # [B, N, N, M]
                msgs = msgs.max(dim=1)[0] if self.reduction == Reduction.MAX else msgs.min(dim=1)[0]# [B, N, M]
            else:
                raise ValueError(f"Invalid reduction: {self.reduction}")

            
            if num_nodes is not None:
                msgs = torch.where(node_mask.unsqueeze(-1), msgs, 0.0)

            # Update node features
            h1 = self.fc_out_source(z) # [B, N, O]
            h2 = self.fc_out_messages(msgs) # [B, N, O]
            updated = h1 + h2 # [B, N, O]
            
            updated = self.activation(updated) # [B, N, O]
            updated = self.layer_norm(updated) # [B, N, O]

            # Gated update (optional)
            if self.gated:
                gate_input = self.gate_fc1(z) + self.gate_fc2(msgs) # [B, N, O]
                gate = torch.sigmoid(self.gate_fc3(F.relu(gate_input))) # [B, N, O]
                updated = updated * gate + processor_state * (1 - gate) # [B, N, O]


            if num_nodes is not None:
                updated = updated.masked_fill(~node_mask.unsqueeze(-1), 0.0)

            processor_state = updated

        return processor_state, tri_msgs

        
class DeepSets(PGN):
    """
    Deep Sets (Zaheer et al., NeurIPS 2017)
    """

    def forward(self, graph_features: GraphFeatures, processor_state: Tensor, num_nodes: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        # Turn into a batch of adjacency matrices
        fully_connected = torch.ones_like(graph_features.adj_mat) 

        if num_nodes is not None:
            valid = batch_mask(num_nodes, fully_connected.size(-1), 2).type_as(fully_connected)
            fully_connected = fully_connected * valid

        graph_features.adj_mat = fully_connected * torch.eye(graph_features.adj_mat.size(-1))
        return super().forward(graph_features, processor_state, num_nodes)
    

class MPNN(PGN):
    """
    Message-Passing Neural Network (Gilmer et al., ICML 2017)
    """
    def forward(self, graph_features: GraphFeatures, processor_state: Tensor, num_nodes: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        graph_features.adj_mat = torch.ones_like(graph_features.adj_mat)
        if num_nodes is not None:
            valid = batch_mask(num_nodes, graph_features.adj_mat.size(-1), 2).type_as(graph_features.adj_mat)
            graph_features.adj_mat = graph_features.adj_mat * valid
        return super().forward(graph_features, processor_state, num_nodes) 
    
class PGNMask(PGN):
  """Masked Pointer Graph Networks (Veličković et al., NeurIPS 2020)."""

  @property
  def inf_bias(self):
    return True

  @property
  def inf_bias_edge(self):
    return True