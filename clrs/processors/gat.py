from typing import Optional, Tuple
from torch import nn
from torch import Tensor
import torch
from .base import ProcessorBase, GraphFeatures
import torch.nn.functional as F
from ..utils import Linear, batch_mask, expand, POS_INF, NEG_INF

class GAT(ProcessorBase):
    """Graph Attention Network (Velickovic et al., ICLR 2018)."""

    def __init__(self,
      node_feat_size: int,
      hidden_size: int,
      nb_heads: int,
      activation: Optional[nn.Module] = nn.ReLU(),
      edge_feat_size: Optional[int] = None,
      graph_feat_size: Optional[int] = None,
      mp_steps: int = 1,
      use_residual: bool = True,
      use_layer_norm: bool = False):
        super().__init__()
        self.node_feat_size = node_feat_size

        if edge_feat_size is None:
            edge_feat_size = node_feat_size
        if graph_feat_size is None:
            graph_feat_size = node_feat_size

        self.out_size = hidden_size
        self.nb_heads = nb_heads
        if hidden_size % nb_heads != 0:
            raise ValueError('The number of attention heads must divide the width!')
        self.head_size = hidden_size // nb_heads
        assert isinstance(activation, nn.Module) or activation is None
        self.activation = activation if activation is not None else nn.Identity()
        self.use_ln = use_layer_norm
        self.use_residual = use_residual
        self.mp_steps = mp_steps

        node_features_dim = self.node_feat_size + self.out_size # Because of concatenation

        self.m = Linear(node_features_dim, self.out_size)
        self.a_1n = Linear(node_features_dim, self.nb_heads)
        self.a_2n = Linear(node_features_dim, self.nb_heads)
        self.a_e = Linear(edge_feat_size, self.nb_heads)
        self.a_g = Linear(graph_feat_size, self.nb_heads)
        self.residual = None
        if self.use_residual:
            self.residual = Linear(node_features_dim, self.out_size)
        self.ln = nn.LayerNorm(self.out_size) if use_layer_norm else nn.Identity()

    @property
    def returns_edge_fts(self):
        return False

    def forward(self, graph_features: GraphFeatures, processor_state: Optional[Tensor] = None, num_nodes: Optional[int] = None) -> Tuple[Tensor, Optional[Tensor]]:
        B, N, D = graph_features.node_fts.shape
        assert graph_features.adj_mat.shape == (B, N, N)
        assert graph_features.edge_fts.shape == (B, N, N, D)
        assert graph_features.graph_fts.shape == (B, D)

        org_dtype = graph_features.node_fts.dtype

        if processor_state is None:
            processor_state = torch.zeros(B, N, self.out_size, device=graph_features.node_fts.device)

        assert processor_state.shape == (B, N, self.out_size)
        att_e = self.a_e(graph_features.edge_fts).permute(0, 3, 1, 2) # [B, H, N, N]
        att_g = self.a_g(graph_features.graph_fts).unsqueeze(-1).unsqueeze(-1) # [B, H, 1, 1]

        bias_mat = (graph_features.adj_mat - 1.0) * 1.0e9  # [B, N, N]
        bias_mat = torch.tile(bias_mat[..., None], (1, 1, 1, self.nb_heads)) # [B, N, N, H]
        bias_mat = bias_mat.permute(0, 3, 1, 2) # [B, H, N, N]

        if num_nodes is not None:
            edge_mask = batch_mask(num_nodes, N, 2) # [B, N, N]
        else:
            edge_mask = graph_features.adj_mat.to(torch.bool) # [B, N, N]

        head_mask = edge_mask.unsqueeze(1) # [B, 1, N, N]

  
        for _ in range(self.mp_steps):
            node_state = torch.cat([graph_features.node_fts, processor_state], dim=-1) # [B, N, 2D] == [B, N, H*F]
            att_1n = self.a_1n(node_state).unsqueeze(-1).permute(0, 2, 1, 3) # [B, H, N, 1]
            att_2n = self.a_2n(node_state).unsqueeze(-1).permute(0, 2, 3, 1) # [B, H, 1, N]

            logits = att_1n + att_2n + att_e + att_g # [B, H, N, N]
            logits = logits.masked_fill(~head_mask, NEG_INF)  # Apply the node mask

            # 2) find rows with *zero* valid entries
            # head_mask.any(dim=-1): [B, 1, N], True if node i has ≥1 valid neighbor
            # row_has_valid = head_mask.any(dim=-1, keepdim=True)           # [B, 1, N, 1]
            # row_has_valid = row_has_valid.expand_as(logits) # [B, H, N, N]

            # # 3) for those entirely-invalid rows, fill logits with 0 instead of -inf
            # logits = torch.where(row_has_valid, logits, torch.zeros_like(logits))

            att_coeff = torch.softmax(F.leaky_relu(logits) + bias_mat, dim=-1).to(org_dtype) # [B, H, N, N]

            values = self.m(node_state) # [B, N, H*F]
            values = values.reshape(B, N, self.nb_heads, self.head_size) # [B, N, H, F]
            values = values.permute(0, 2, 1, 3) # [B, H, N, F]

            ret = torch.matmul(att_coeff, values) # [B, H, N, F]
            ret = ret.permute(0, 2, 1, 3) # [B, N, H, F]
            ret = ret.reshape(B, N, self.out_size) # [B, N, H*F] == [B, N, D]

            if self.use_residual:
                ret += self.residual(node_state)

            ret = self.activation(ret)
            ret = self.ln(ret)

            if num_nodes is not None:
                ret = ret.masked_fill(~expand(batch_mask(num_nodes, N, 1), ret), 0.0)

            processor_state = ret
        return processor_state, None


class GATFull(GAT):
    """Graph Attention Network with full adjacency matrix."""

    def forward(self, graph_features: GraphFeatures, processor_state: Optional[Tensor] = None, num_nodes: Optional[int] = None) -> Tuple[Tensor, Optional[Tensor]]:
        graph_features.adj_mat = torch.ones_like(graph_features.adj_mat)
        if num_nodes is not None:
            valid = batch_mask(num_nodes, graph_features.adj_mat.size(-1), 2).type_as(graph_features.adj_mat)
            graph_features.adj_mat = graph_features.adj_mat * valid
        
        return super().forward(graph_features, processor_state, num_nodes)
  

class GATv2(ProcessorBase):
    """Graph Attention Network v2 (Brody et al., ICLR 2022)."""

    @property
    def returns_edge_fts(self):
        return False

    def __init__(
        self,
        node_feat_size: int,
        hidden_size: int,
        nb_heads: int,
        mid_size: Optional[int] = None,
        edge_feat_size: Optional[int] = None,
        graph_feat_size: Optional[int] = None,
        activation: Optional[nn.Module] = nn.ReLU(),
        use_residual: bool = True,
        mp_steps: int = 1,
        use_layer_norm: bool = False):
        super().__init__()

        self.node_feat_size = node_feat_size
        if mid_size is None:
            self.mid_size = hidden_size
        else:
            self.mid_size = mid_size

        if edge_feat_size is None:
            edge_feat_size = node_feat_size
        if graph_feat_size is None:
            graph_feat_size = node_feat_size
        
        self.hidden_size = hidden_size
        self.nb_heads = nb_heads
        if hidden_size % nb_heads != 0:
            raise ValueError('The number of attention heads must divide the width!')
        
        self.head_size = hidden_size // nb_heads

        # This is check of self.mid_size divisibility by nb_heads, not really needed 
        # with the vectorized implementation, but we keep it for clarity
        if self.mid_size % nb_heads != 0:
            raise ValueError('The number of attention heads must divide the message!')
        
        self.activation = activation if activation is not None else nn.Identity()
        self.use_residual = use_residual
        self.use_ln = use_layer_norm
        self.mp_steps = mp_steps

        z_feat_size = self.node_feat_size + self.hidden_size # Because of concatenation
        self.residual = None
        if self.use_residual:
            self.residual = Linear(z_feat_size, self.hidden_size)

        self.m = Linear(z_feat_size, self.hidden_size)
        self.w_1 = Linear(z_feat_size, self.mid_size)
        self.w_2 = Linear(z_feat_size, self.mid_size)
        self.w_e = Linear(edge_feat_size, self.mid_size)
        self.w_g = Linear(graph_feat_size, self.mid_size)

        self.a_heads = Linear(self.mid_size, self.nb_heads)

        self.ln = nn.LayerNorm(self.hidden_size) if use_layer_norm else nn.Identity()

    def forward(self, graph_features: GraphFeatures, processor_state: Optional[Tensor] = None, num_nodes: Optional[int] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """GATv2 inference step."""
        B, N, D = graph_features.node_fts.shape
        assert graph_features.adj_mat.shape == (B, N, N)
        assert graph_features.edge_fts.shape == (B, N, N, D)
        assert graph_features.graph_fts.shape == (B, D)

        org_dtype = graph_features.node_fts.dtype

        if processor_state is None:
            processor_state = torch.zeros(B, N, self.hidden_size, device=graph_features.node_fts.device)

        assert processor_state.shape == (B, N, self.hidden_size)
        bias_mat = (graph_features.adj_mat - 1.0) * 1e9  # [B, N, N]
        bias_mat = torch.tile(bias_mat[..., None], (1, 1, 1, self.nb_heads)) # [B, N, N, H]
        bias_mat = bias_mat.permute(0, 3, 1, 2) # [B, H, N, N]

        # Build a boolean [B,N,N] mask of which (i,j) pairs are valid
        if num_nodes is not None:
            edge_mask = batch_mask(num_nodes, N, 2)             # bool [B,N,N]
        else:
            edge_mask = graph_features.adj_mat.to(torch.bool) # fallback

        # Turn that into a [B,1,N,N] head-level mask
        head_mask = edge_mask[:, None, :, :]                  # [B,1,N,N]

        pre_att_e = self.w_e(graph_features.edge_fts)                            # [B, N, N, M]
        pre_att_g = self.w_g(graph_features.graph_fts).unsqueeze(1).unsqueeze(2) # [B, 1, 1, M]


        for _ in range(self.mp_steps):
            z = torch.cat([graph_features.node_fts, processor_state], dim=-1)

            values = self.m(z)                                           # [B, N, H*F]
            values = values.reshape(B, N, self.nb_heads, self.head_size)  # [B, N, H, F]
            values = values.permute(0, 2, 1, 3)                          # [B, H, N, F]

            pre_att_1 = self.w_1(z).unsqueeze(1)                                     # [B, 1, N, M]
            pre_att_2 = self.w_2(z).unsqueeze(2)                                     # [B, N, 1, M]

            pre_att = pre_att_1 + pre_att_2 + pre_att_e + pre_att_g # [B, N, N, M]

            pre_att = F.leaky_relu(pre_att) # [B, N, N, M]
            logits = self.a_heads(pre_att) # [B, N, N, H]
            logits = logits.permute(0, 3, 1, 2) # [B, H, N, N]

            # 2) force all padded or non-existent edges to -inf
            logits = logits.masked_fill(~head_mask, NEG_INF)

            coefs = torch.softmax(logits + bias_mat, dim=-1).to(org_dtype) * head_mask
            ret = torch.matmul(coefs, values)  # [B, H, N, F]
            ret = ret.permute(0, 2, 1, 3)  # [B, N, H, F]
            ret = ret.reshape(B, N, self.hidden_size) # [B, N, H*F] == [B, N, D]

            if self.use_residual:
                ret += self.residual(z)

            ret = self.activation(ret)
            ret = self.ln(ret)

            if num_nodes is not None:
                ret = ret.masked_fill(~expand(batch_mask(num_nodes, N, 1), ret), 0.0)

            processor_state = ret

        return processor_state, None


class GATv2Full(GATv2):
  """Graph Attention Network v2 with full adjacency matrix."""

  def process(self, graph_features: GraphFeatures, processor_state: Optional[Tensor] = None, num_nodes: Optional[int] = None) -> Tuple[Tensor, Optional[Tensor]]:
        graph_features.adj_mat = torch.ones_like(graph_features.adj_mat)
        if num_nodes is not None:
            valid = batch_mask(num_nodes, graph_features.adj_mat.size(-1), 2).type_as(graph_features.adj_mat)
            graph_features.adj_mat = graph_features.adj_mat * valid
        
        return super().process(graph_features, processor_state, num_nodes)
