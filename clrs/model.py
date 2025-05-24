from enum import Enum
from typing import Dict, List, Optional, Tuple, NamedTuple, Union
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from .specs import Location, Type, Spec, Stage, Trajectory, Hints, Input, Output, OutputClass, Algorithm, Feature, NumNodes, NumSteps
from .utils import log_sinkhorn, Linear, batch_mask, expand, POS_INF, NEG_INF
from .processors import GraphFeatures, ProcessorBase

_USE_NUM_NODES_FOR_LOSS_AND_EVAL = True  # This is just used for testing. Keep it to True for all practical purposes.
def set_use_num_nodes(use_num_nodes: bool):
    global _USE_NUM_NODES_FOR_LOSS_AND_EVAL
    _USE_NUM_NODES_FOR_LOSS_AND_EVAL = use_num_nodes




# Define LSTMState NamedTuple
class LSTMState(NamedTuple):
    h: Tensor
    c: Tensor

    @classmethod
    def empty(cls, shape: Tuple[int, ...], device: torch.device = torch.device("cpu")) -> "LSTMState":
        return cls(h=torch.zeros(shape, device=device), 
                   c=torch.zeros(shape, device=device))
    
    def reshape(self, shape: Tuple[int, ...]) -> "LSTMState":
        return LSTMState(h=self.h.reshape(shape), 
                         c=self.c.reshape(shape))

class LSTMCell(nn.LSTMCell):
    """
    Wrapper around nn.LSTMCell to support LSTMState NamedTuple for input and output.
    """
    def __init__(self,  input_size: int, hidden_size: int, bias: bool = True, device=None, dtype=None,):
        super().__init__(input_size, hidden_size, bias, device, dtype)

    def forward(self, input: Tensor, state: LSTMState) -> Tuple[Tensor, LSTMState]:
        # super().forward(input, hx) returns a tuple (h_next, c_next)
        # where hx is (h_prev, c_prev)
        next_h, next_c = super().forward(input, (state.h, state.c))
        
        # The "output" of the LSTM cell for this timestep is typically its new hidden state.
        # The new LSTMState comprises both the new hidden and cell states.
        return next_h, LSTMState(h=next_h, c=next_c)

class ReconstMode(str, Enum):
    SOFT = "soft"
    HARD = "hard"
    HARD_ON_EVAL = "hard_on_eval"


def get_steps_mask(num_steps: NumSteps, data: Tensor) -> Tensor:
    """
    This already accounts for one less step in predicted hints.
    """
    max_steps = data.shape[0]
    batch_size = num_steps.shape[0]
    steps_mask = num_steps > (torch.arange(max_steps, device=num_steps.device).unsqueeze(1) + 1)
    assert steps_mask.shape == (max_steps, batch_size)
    target_mask_shape = (max_steps, batch_size) + (1,) * (data.dim() - 2)
    return steps_mask.view(target_mask_shape).expand(data.shape)

class Loss(nn.Module):
    def __init__(self, type_: Type, stage: Stage):
        super().__init__()
        self.type_ = type_
        self.stage = stage

    def get_node_mask(self, num_nodes: Tensor, prediction: Tensor) -> Tensor:
        offset = 1 if self.type_ == Type.CATEGORICAL else 0
        prior_dims = 1 if self.stage == Stage.OUTPUT else 2  # 2 for hint
        num_node_dims = prediction.ndim - offset - prior_dims
        if num_node_dims > 0:
            return expand(batch_mask(num_nodes, prediction.shape[prior_dims], num_node_dims), prediction, prior_dims=prior_dims-1)
        else:
            return torch.ones_like(prediction, device=prediction.device).bool()
    
    
    def get_loss(self, pred: Tensor, target: Tensor, num_nodes: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        if num_nodes is not None:
            node_mask = self.get_node_mask(num_nodes, pred).type_as(pred)
        
        if self.type_ == Type.SCALAR:
            loss = (pred - target)**2
            mask = torch.ones_like(loss)
            if num_nodes is not None:
                mask = mask * node_mask
        elif self.type_ == Type.MASK:
            # stable binary cross entropy or logistic loss
            loss = (F.relu(pred) - pred * target + torch.log1p(torch.exp(-torch.abs(pred))))
            mask = (target != OutputClass.MASKED).type_as(pred)
            if num_nodes is not None:
                mask = mask * node_mask
                loss = torch.nan_to_num(loss, nan=0.0)
        elif self.type_ in [Type.MASK_ONE, Type.CATEGORICAL]:
            masked_target = target * (target != OutputClass.MASKED).type_as(pred)
            if num_nodes is not None:
                masked_target = masked_target * node_mask
                target = target * node_mask
            ce = masked_target * F.log_softmax(pred, -1).type_as(pred)
            if num_nodes is not None:
                ce = torch.nan_to_num(ce, nan=0.0)
            loss = -torch.sum(ce, dim=-1, keepdim=True)
            mask = torch.any(target == OutputClass.POSITIVE, dim=-1, keepdim=True).type_as(pred)
        elif self.type_ == Type.POINTER:
            ce = target * F.log_softmax(pred, -1).type_as(pred)
            if num_nodes is not None:
                node_mask = self.get_node_mask(num_nodes, pred)
                # This is necessary to get rid of NaNs
                ce = ce.masked_fill(~(node_mask.bool()), 0.0)
                loss = -torch.sum(ce, dim=-1)
                mask = node_mask.any(-1)
            else:
                loss = -torch.sum(ce, dim=-1)
                mask = torch.ones_like(loss)
        
        elif self.type_ == Type.PERMUTATION_POINTER:
            # Predictions are NxN logits aiming to represent a doubly stochastic matrix.
            # Compute the cross entropy between doubly stochastic pred and truth_data
            if num_nodes is not None:
                pred = pred.masked_fill(~node_mask.bool(), 0.0)
                target = target * node_mask

            prod = target * pred
            loss = -torch.sum(prod, dim=-1)
            mask = torch.ones_like(loss)

            if num_nodes is not None:
                mask = node_mask.any(-1).type_as(loss)
        else:
            raise ValueError(f"Invalid type: {self.type_}")
        return loss, mask
    
    def forward(self, pred: Tensor, target: Tensor, num_steps: Optional[Tensor] = None, num_nodes: Optional[Tensor] = None) -> Tensor:
        loss, mask = self.get_loss(pred, target, num_nodes)
        assert loss.shape == mask.shape, f"loss.shape: {loss.shape}, mask.shape: {mask.shape}"
        
        if self.stage == Stage.HINT:
            assert num_steps is not None
            steps_mask = get_steps_mask(num_steps, loss).type_as(mask)
            mask = steps_mask * mask

        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

class Encoder(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 location: Location, 
                 type_: Type,
                 num_classes: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.location = location
        self.encoders = nn.ModuleList()
        self.type_ = type_
        self.num_classes = num_classes

        if self.type_ == Type.CATEGORICAL:
            assert num_classes is not None
            self.encoders.append(Linear(num_classes, hidden_dim))
        else:
            self.encoders.append(Linear(1, hidden_dim))

        if self.location == Location.EDGE and self.type_ == Type.POINTER:
            # Edge pointers need two-way encoders
            self.encoders.append(Linear(1, hidden_dim))

    def encode_to_adj_mat(self, data: Tensor, adj_mat: Tensor, num_nodes: Optional[Tensor] = None) -> Tensor:
        if self.location == Location.NODE and self.type_ in [Type.POINTER, Type.PERMUTATION_POINTER]:
            adj_mat += ((data + data.permute(0, 2, 1)) > 0.5)
        elif self.location == Location.EDGE and self.type_ == Type.MASK:
            adj_mat += ((data + data.permute(0, 2, 1)) > 0.0)
        return (adj_mat > 0.).type_as(data)
    
    def encode_to_node_fts(self, data: Tensor, node_fts: Tensor, num_nodes: Optional[Tensor] = None) -> Tensor:
        is_pointer = (self.type_ in [Type.POINTER, Type.PERMUTATION_POINTER])
        if self.type_ != Type.CATEGORICAL:
            data = data.unsqueeze(-1)

        if (self.location == Location.NODE and not is_pointer) or (self.location == Location.GRAPH and self.type_ == Type.POINTER):
            encoding = self.encoders[0](data)
            if num_nodes is not None:
                encoding = encoding.masked_fill(~expand(batch_mask(num_nodes, data.size(1), 1), encoding), 0.0)
            node_fts += encoding
        return node_fts

    def encode_to_edge_fts(self, data: Tensor, edge_fts: Tensor, num_nodes: Optional[Tensor] = None) -> Tensor:
        if self.type_ != Type.CATEGORICAL:
            data = data.unsqueeze(-1)

        if self.location == Location.NODE and self.type_ in [Type.POINTER, Type.PERMUTATION_POINTER]:
            encoding = self.encoders[0](data) # [B, N, N, H]
            if num_nodes is not None:
                encoding = encoding.masked_fill(~expand(batch_mask(num_nodes, data.size(1), 2), encoding), 0.0)
            edge_fts += encoding
        elif self.location == Location.EDGE:
            encoding = self.encoders[0](data) 
            if self.type_ == Type.POINTER:
                # Aggregate pointer contributions across sender and receiver nodes.
                encoding_2 = self.encoders[1](data) # [B, N, N, N, H]

                if num_nodes is None:
                    # exactly your old behavior
                    mean_s = encoding.mean(dim=1)    # [B, N, N, T]
                    mean_r = encoding_2.mean(dim=2)  # [B, N, N, T]
                else:
                    B, N, _, _, _ = encoding_2.shape
                    # 1) build masks for senders / receivers
                    edge_mask = batch_mask(num_nodes, N, 3).unsqueeze(-1).type_as(encoding_2)        # [B, N, N, N, 1]
                    sum_s =  torch.sum(encoding * edge_mask, dim=1) # [B, N, N, H]
                    sum_r =  torch.sum(encoding_2 * edge_mask, dim=2) # [B, N, N, H]

                    # # 3) divide by the true neighbor count
                    counts = num_nodes.view(B, 1, 1, 1).to(encoding.dtype)   # [B,1,1,1]
                    mean_s = sum_s / counts                                    # [B,N, N, H]
                    mean_r = sum_r / counts                                    # [B, N, N, H]


                # import ipdb; ipdb.set_trace()
                edge_fts += mean_s + mean_r
        return edge_fts

    def encode_to_graph_fts(self, data: Tensor, graph_fts: Tensor) -> Tensor:
        if self.type_ != Type.CATEGORICAL:
            data = data.unsqueeze(-1)
        if self.location == Location.GRAPH and self.type_ != Type.POINTER:
            encoding = self.encoders[0](data)
            graph_fts += encoding
        return graph_fts

    def encode(self, data: Tensor, graph_features: GraphFeatures = None, num_nodes: Tensor = None) -> GraphFeatures:
        graph_features.adj_mat = self.encode_to_adj_mat(data, graph_features.adj_mat)
        graph_features.node_fts = self.encode_to_node_fts(data, graph_features.node_fts, num_nodes)
        graph_features.edge_fts = self.encode_to_edge_fts(data, graph_features.edge_fts, num_nodes)
        graph_features.graph_fts = self.encode_to_graph_fts(data, graph_features.graph_fts)
        return graph_features

class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, location: Location, type_: Type, 
                 hard_mode: ReconstMode,
                 sinkhorn_temp: Optional[float] = None,
                 sinkhorn_steps: Optional[int] = None,
                 node_dim_multiplier: int = 3,
                 edge_dim_multiplier: int = 2,
                 inf_bias: bool = False,
                 inf_bias_edge: bool = False,
                 num_classes: Optional[int] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.location = location
        self.type_ = type_
        self.num_cats = num_classes
        self.hard_mode = hard_mode
        self.inf_bias = inf_bias
        self.inf_bias_edge = inf_bias_edge
        edge_dim = edge_dim_multiplier * hidden_dim
        node_dim = node_dim_multiplier * hidden_dim

        if self.type_ == Type.CATEGORICAL:
            assert num_classes is not None

        if self.type_ == Type.PERMUTATION_POINTER:
            assert sinkhorn_temp is not None and sinkhorn_steps is not None

        self.sinkhorn_temp = sinkhorn_temp
        self.sinkhorn_steps = sinkhorn_steps
        self.decoders = nn.ModuleList()

        if self.location == Location.NODE:
            # Node decoders.
            if self.type_ in [Type.SCALAR, Type.MASK, Type.MASK_ONE]:
                self.decoders.append(Linear(node_dim, 1))
            elif self.type_ == Type.CATEGORICAL:
                self.decoders.append(Linear(node_dim, self.num_cats))
            elif self.type_ in [Type.POINTER, Type.PERMUTATION_POINTER]:
                self.decoders.append(Linear(node_dim, hidden_dim))
                self.decoders.append(Linear(node_dim, hidden_dim))
                self.decoders.append(Linear(edge_dim, hidden_dim))
                self.decoders.append(Linear(hidden_dim, 1))
        elif self.location == Location.EDGE:
            # Edge decoders.
            if self.type_ in [Type.SCALAR, Type.MASK, Type.MASK_ONE]:
                self.decoders.append(Linear(node_dim, 1))
                self.decoders.append(Linear(node_dim, 1))
                self.decoders.append(Linear(edge_dim, 1))
            elif self.type_ == Type.CATEGORICAL:
                self.decoders.append(Linear(node_dim, self.num_cats))
                self.decoders.append(Linear(node_dim, self.num_cats))
                self.decoders.append(Linear(edge_dim, self.num_cats))
            elif self.type_ == Type.POINTER:
                self.decoders.append(Linear(node_dim, hidden_dim))
                self.decoders.append(Linear(node_dim, hidden_dim))
                self.decoders.append(Linear(edge_dim, hidden_dim))
                self.decoders.append(Linear(node_dim, hidden_dim))
                self.decoders.append(Linear(hidden_dim, 1))
        elif self.location == Location.GRAPH:
            # Graph decoders.
            if self.type_ in [Type.SCALAR, Type.MASK, Type.MASK_ONE]:
                self.decoders.append(Linear(node_dim, 1)) # node_fts
                self.decoders.append(Linear(hidden_dim, 1)) # graph_fts
            elif self.type_ == Type.CATEGORICAL:
                self.decoders.append(Linear(node_dim, self.num_cats)) # node_fts
                self.decoders.append(Linear(hidden_dim, self.num_cats)) # graph_fts
                self.decoders.append(Linear(hidden_dim, self.num_cats)) # output
            elif self.type_ == Type.POINTER:
                self.decoders.append(Linear(node_dim, 1)) # node_fts
                self.decoders.append(Linear(hidden_dim, 1)) # graph_fts
                self.decoders.append(Linear(node_dim, 1)) # node_fts
        
    def decode_node_fts(self, graph_features: GraphFeatures, num_nodes: Optional[Tensor] = None) -> Tensor:
        """
        graph_features.node_fts shape: (batch_size, nb_nodes, hidden_dim)
        graph_features.edge_fts shape: (batch_size, nb_nodes, nb_nodes, hidden_dim)
        graph_features.adj_mat shape: (batch_size, nb_nodes, nb_nodes)
        graph_features.graph_fts shape: (batch_size, hidden_dim)
        """

        B, N, _ = graph_features.node_fts.size()

        # 1) build masks if padding is present
        if num_nodes is not None:
            node_mask = batch_mask(num_nodes, N, 1)        # bool [B, N]
            edge_mask = batch_mask(num_nodes, N, 2)        # bool [B, N, N]
            # for elementwise multiplies, we'll need float versions
            node_mask_float = node_mask.unsqueeze(-1).to(graph_features.node_fts.dtype)  # [B, N, 1]
            edge_mask_float = edge_mask.unsqueeze(-1).to(graph_features.edge_fts.dtype)  # [B, N, N, 1]
        else:
            node_mask = edge_mask = None
            node_mask_float = edge_mask_float = None

        fill_value = 0.0 if self.type_  == Type.SCALAR else NEG_INF

        if self.type_ in [Type.SCALAR, Type.MASK, Type.MASK_ONE]:
            preds = self.decoders[0](graph_features.node_fts).squeeze(-1) # (batch_size, nb_nodes)
            if num_nodes is not None:
                if self.type_ == Type.SCALAR:
                    preds = preds * node_mask_float.squeeze(-1)  # zero out padded nodes
                else:
                    preds = preds.masked_fill(~node_mask, fill_value)
        elif self.type_ == Type.CATEGORICAL:
            preds = self.decoders[0](graph_features.node_fts) # (batch_size, nb_nodes, num_cats)
            if num_nodes is not None:
                preds = preds.masked_fill(~node_mask.unsqueeze(-1), fill_value)
        elif self.type_ in [Type.POINTER, Type.PERMUTATION_POINTER]:
            p_1 = self.decoders[0](graph_features.node_fts) # (batch_size, nb_nodes, hidden_dim)
            p_2 = self.decoders[1](graph_features.node_fts) # (batch_size, nb_nodes, hidden_dim)
            p_3 = self.decoders[2](graph_features.edge_fts) # (batch_size, nb_nodes, nb_nodes, hidden_dim)

            p_e = torch.unsqueeze(p_2, -2) + p_3 # (batch_size, nb_nodes, nb_nodes, hidden_dim)

            if num_nodes is not None:
                p_1 = p_1.masked_fill(~node_mask.unsqueeze(-1), fill_value)
                p_e = p_e.masked_fill(~edge_mask.unsqueeze(-1), fill_value)

            p_m = torch.maximum(torch.unsqueeze(p_1, -2), p_e.permute(0, 2, 1, 3)) # (batch_size, nb_nodes, nb_nodes, hidden_dim)
            preds = self.decoders[3](p_m).squeeze(-1) # (batch_size, nb_nodes, nb_nodes)


            if num_nodes is not None:
                preds = preds.masked_fill(~edge_mask, fill_value)  # For pointer, we use -inf


            if self.inf_bias:
                if num_nodes is not None:
                    # below operation will be hijacked by the -inf bias so we need to take care of it
                    preds = preds.masked_fill(~edge_mask, POS_INF)  # For pointer, we use -inf
                per_batch_min = torch.amin(preds, dim=tuple(range(1, preds.dim())),keepdim=True)  # shape: (batch, 1, 1, …)
                neg_one = torch.tensor(-1.0, device=preds.device, dtype=preds.dtype)
                # mask preds wherever adj_mat <= 0.5
                preds = torch.where(graph_features.adj_mat > 0.5, preds, torch.minimum(neg_one, per_batch_min - 1.0))

                if num_nodes is not None:
                    preds = preds.masked_fill(~edge_mask, fill_value)  # For pointer, we use -inf

            if self.type_ == Type.PERMUTATION_POINTER:
                if not self.training:  # testing or validation, no Gumbel noise
                    preds = log_sinkhorn(x=preds, steps=10, temperature=0.1, zero_diagonal=True, add_noise=False, num_nodes=num_nodes)
                else:  # training, add Gumbel noise
                    preds = log_sinkhorn(x=preds, steps=10, temperature=0.1, zero_diagonal=True, add_noise=True, num_nodes=num_nodes)
        else:
            raise ValueError("Invalid output type")
        return preds
    
    def decode_edge_fts(self, graph_features: GraphFeatures, num_nodes: Optional[Tensor] = None) -> Tensor:
        """
        graph_features.node_fts shape: (batch_size, nb_nodes, node_dim)
        graph_features.edge_fts shape: (batch_size, nb_nodes, nb_nodes, edge_dim)
        graph_features.adj_mat shape: (batch_size, nb_nodes, nb_nodes)
        graph_features.graph_fts shape: (batch_size, hidden_dim)
        """
        """Decodes edge features."""

        B, N, _ = graph_features.node_fts.size()
        dtype = graph_features.node_fts.dtype


        pred_1 = self.decoders[0](graph_features.node_fts) # (batch_size, nb_nodes, 1/num_cats/hidden_dim)
        pred_2 = self.decoders[1](graph_features.node_fts) # (batch_size, nb_nodes, 1/num_cats/hidden_dim)
        pred_e = self.decoders[2](graph_features.edge_fts) # (batch_size, nb_nodes, nb_nodes, 1/num_cats/hidden_dim)
        pred = (torch.unsqueeze(pred_1, -2) + torch.unsqueeze(pred_2, -3) + pred_e) # (batch_size, nb_nodes, nb_nodes, 1/num_cats/hidden_dim)

        # 1) build masks if padding is present
        if num_nodes is not None:
            node_mask = batch_mask(num_nodes, N, 1).unsqueeze(-1)  # [B, N, 1]
            edge_mask = batch_mask(num_nodes, N, 2).unsqueeze(-1) # [B, N, N, 1]
        else:
            node_mask = edge_mask = None

        fill_value = 0.0 if self.type_  == Type.SCALAR else NEG_INF

        if self.type_ in [Type.SCALAR, Type.MASK, Type.MASK_ONE, Type.CATEGORICAL]:
            if num_nodes is not None:
                pred = pred.masked_fill(~edge_mask, fill_value)
            preds = pred.squeeze(-1) if self.type_ != Type.CATEGORICAL else pred
        elif self.type_ == Type.POINTER:
            pred_3 = self.decoders[3](graph_features.node_fts) # (B, N, D)
            if num_nodes is not None:
                pred = pred.masked_fill(~edge_mask, fill_value)  # [B, N, N, D]
                pred_3 = pred_3.masked_fill(~node_mask, fill_value) # [B, N, D]
            p_m = torch.max(pred.unsqueeze(-2),   # [B, N, N, 1, D]
                            pred_3.unsqueeze(-3).unsqueeze(-3) # [B, 1, 1, N, D]
                        ) # [B, N, N, N, D]
            preds = self.decoders[4](p_m).squeeze(-1) # [B, N, N, N]
            if num_nodes is not None:
                mask = expand(batch_mask(num_nodes, N, 3), preds) # [B, N, N, N]
                preds = preds.masked_fill(~mask, fill_value)  # For pointer, we use -inf
        else:
            raise ValueError("Invalid output type")
        if self.inf_bias_edge and self.type_ in [Type.MASK, Type.MASK_ONE]:
            # below operation will be hijacked by the -inf bias so we need to take care of it
            if num_nodes is not None:
                preds_inf = preds.masked_fill(~edge_mask.squeeze(-1), POS_INF) 
                per_batch_min = torch.amin(preds_inf, dim=tuple(range(1, preds.dim())), keepdim=True) # [B, 1, 1]
            else:
                per_batch_min = torch.amin(preds, dim=tuple(range(1, preds.dim())), keepdim=True) # [B, 1, 1]
            neg_one = torch.tensor(-1.0, device=preds.device, dtype=preds.dtype)
            preds = torch.where(graph_features.adj_mat > 0.5,
                            preds,
                            torch.minimum(neg_one, per_batch_min - 1.0))
            
            if num_nodes is not None:
                preds = preds.masked_fill(~edge_mask.squeeze(-1), fill_value) 

        return preds
    
    def decode_graph_fts(self, graph_features: GraphFeatures, num_nodes: Optional[Tensor] = None) -> Tensor:
        """Decodes graph features."""

        if num_nodes is not None:
            node_mask = batch_mask(num_nodes, graph_features.node_fts.size(1), 1).unsqueeze(-1)
            node_fts = graph_features.node_fts.masked_fill(~node_mask, NEG_INF)
        else:
            node_fts = graph_features.node_fts

        gr_emb = torch.max(node_fts, dim=-2, keepdim=False).values # (batch_size, node_dim)
        pred_n = self.decoders[0](gr_emb) # (batch_size, 1/num_cats)
        pred_g = self.decoders[1](graph_features.graph_fts) # (batch_size, 1/num_cats)
        pred = pred_n + pred_g # (batch_size, 1/num_cats)
        if self.type_ in [Type.SCALAR, Type.MASK, Type.MASK_ONE]:
            preds = pred.squeeze(-1) # (batch_size,)
        elif self.type_ == Type.CATEGORICAL:
            preds = pred # (batch_size, num_cats)
        elif self.type_ == Type.POINTER:
            # I raise the error because in the specs, there is no algorithm that uses pointer output type for graph features
            raise ValueError("Pointer output type not supported for graph features")
            
            # I keep the code here for reference, but it is never reached in the dataset
            pred_2 = self.decoders[2](graph_features.node_fts) # (batch_size, nb_nodes, 1)
            ptr_p = pred.unsqueeze(1) + pred_2.permute(0, 2, 1) # (batch_size, 1, nb_nodes)
            preds = ptr_p.squeeze(1)
        else:
            raise ValueError("Invalid output type")

        return preds

    def postprocess(self, data: Tensor, num_nodes: Optional[Tensor] = None) -> Tensor:
        """Postprocesses decoder output """

        if self.hard_mode == ReconstMode.HARD:
            hard = True
        elif self.hard_mode == ReconstMode.HARD_ON_EVAL:
            hard = not self.training
        elif self.hard_mode == ReconstMode.SOFT:
            hard = False
        else:
            raise ValueError("Invalid hard mode")

        if self.type_ == Type.SCALAR:
            if hard:
                data = data.detach() # (batch_size, nb_nodes)
    
        elif self.type_ == Type.MASK:
            if hard:
                data = (data > 0.0) * 1.0 
            else:
                data = F.sigmoid(data)
        elif self.type_ in [Type.MASK_ONE, Type.CATEGORICAL]:
            if hard:
                best = data.argmax(-1)
                data = F.one_hot(best, data.shape[-1]).type_as(data)
            else:
                data = F.softmax(data, dim=-1).type_as(data)
        elif self.type_ == Type.POINTER:
            if hard:
                data = F.one_hot(data.argmax(dim=-1).long(), data.shape[-1]).type_as(data)
            else:
                data = F.softmax(data, dim=-1).type_as(data)
        elif self.type_ == Type.PERMUTATION_POINTER:
            # Convert the matrix of logits to a doubly stochastic matrix.
            data = log_sinkhorn(
                x=data,
                steps=self.sinkhorn_steps,
                temperature=self.sinkhorn_temp,
                zero_diagonal=True,
                num_nodes=num_nodes,
                add_noise=False)
            data = torch.exp(data)
            if hard:
                data = F.one_hot(data.argmax(dim=-1), data.shape[-1]).type_as(data)
        else:
            raise ValueError("Invalid type")
        
        if num_nodes is not None:
            offset = 1 if self.type_ == Type.CATEGORICAL else 0
            num_node_dims = data.ndim - offset - 1
            if num_node_dims > 0:
                mask = expand(batch_mask(num_nodes, data.shape[1], num_node_dims), data)
                data = data.masked_fill(~mask, 0.0)

        return data

    def decode(self, graph_features: GraphFeatures, num_nodes: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.location == Location.NODE:
            raw_result = self.decode_node_fts(graph_features, num_nodes)
        elif self.location == Location.EDGE:
            raw_result = self.decode_edge_fts(graph_features, num_nodes)
        elif self.location == Location.GRAPH:
            raw_result = self.decode_graph_fts(graph_features, num_nodes)
        return raw_result, self.postprocess(raw_result, num_nodes)


class Evaluator(nn.Module):
    def __init__(self, type_: Type, stage: Stage):
        super().__init__()
        self.type_ = type_
        self.stage = stage
        self.evaluations = nn.ModuleDict()

    def eval_mask_one(self, prediction: Tensor, target: Tensor, steps: Optional[Tensor] = None, num_nodes: Optional[Tensor] = None) -> Tensor:

        # turn one-hot vectors into integer class labels
        pred_labels = prediction.argmax(dim=-1)    # shape (...,)
        truth_labels = target.argmax(dim=-1)  # shape (...,)
        # build mask of entries we care about
        valid = (target != OutputClass.MASKED).all(dim=-1)   # boolean tensor (...)

        if num_nodes is not None:
            if self.type_ == Type.CATEGORICAL:
                mask = self.get_node_mask(num_nodes, prediction).all(-1)
            else:
                mask = self.get_node_mask(num_nodes, valid)
            valid = valid * (mask.type_as(valid))

        if self.stage == Stage.HINT:
            assert steps is not None, "Steps must be provided for hint stage"
            steps_mask = get_steps_mask(steps, valid).type_as(valid)
            valid = valid * steps_mask

        # 5) now compute elementwise equality and cast to float
        eq = (pred_labels == truth_labels).type_as(prediction)

        # 6) float‐mask and reductions
        mask_f   = valid.type_as(eq)     # same dtype and device as eq
        correct  = (eq * mask_f).sum()    # sum of correct predictions
        total    = mask_f.sum()           # number of valid entries

        # 7) scalar accuracy
        return correct / total
    
    def eval_mask(self, prediction: Tensor, target: Tensor, steps: Optional[Tensor] = None, num_nodes: Optional[Tensor] = None) -> Tensor:
        # 1) Build float mask of "valid" positions
        valid = (target != OutputClass.MASKED).type_as(prediction)

        if num_nodes is not None:
            mask = self.get_node_mask(num_nodes, prediction)
            valid = valid * (mask.type_as(valid))

        if self.stage == Stage.HINT:
            assert steps is not None, "Steps must be provided for hint stage"
            steps_mask = get_steps_mask(steps, valid).type_as(valid)
            valid = valid * steps_mask

        # 2) Boolean tensors for positive
        pred_pos  = prediction > 0.5
        truth_pos = target > 0.5

        # 3) True positives / false positives / false negatives
        tp = ((pred_pos  & truth_pos).type_as(prediction) * valid).sum()
        fp = ((pred_pos  & ~truth_pos).type_as(prediction) * valid).sum()
        fn = ((~pred_pos & truth_pos).type_as(prediction) * valid).sum()

        # 4) Precision & recall, defaulting to 1 when denominator == 0
        precision = torch.where((tp + fp) > 0,
                                tp / (tp + fp),
                                torch.ones_like(tp))
        recall    = torch.where((tp + fn) > 0,
                                tp / (tp + fn),
                                torch.ones_like(tp))

        # 5) F1, defaulting to 0 when precision+recall == 0
        denom = precision + recall
        f1 = torch.where(denom > 0,
                        2 * precision * recall / denom,
                        torch.zeros_like(denom))

        return f1
    

    def get_node_mask(self, num_nodes: Tensor, prediction: Tensor) -> Tensor:
        offset = 1 if self.type_ == Type.CATEGORICAL else 0
        prior_dims = 1 if self.stage == Stage.OUTPUT else 2  # 2 for hint
        num_node_dims = prediction.ndim - offset - prior_dims
        if num_node_dims > 0:
            return expand(batch_mask(num_nodes, prediction.shape[prior_dims], num_node_dims), prediction, prior_dims=prior_dims-1)
        else:
            return torch.ones_like(prediction, device=prediction.device).bool()
    
    def eval_pointer(self, prediction: Tensor, target: Tensor, steps: Optional[Tensor] = None, num_nodes: Optional[Tensor] = None) -> Tensor:
        if num_nodes is not None:
            mask = self.get_node_mask(num_nodes, prediction)
        else:
            mask = torch.ones_like(prediction, device=prediction.device).bool()
        if self.stage == Stage.HINT:
            steps_mask = get_steps_mask(steps, prediction)
            mask = mask & steps_mask

        # elementwise equality, as float
        eq = (prediction == target).type_as(prediction)

        # convert mask to the same dtype, sum will produce scalars
        mask_f = mask.type_as(prediction)

        # sum over all elements, then divide
        correct = (eq * mask_f).sum()
        total   = mask_f.sum()

        # returns a single scalar Tensor
        return correct / total

    
    def eval_scalar(self, prediction: Tensor, target: Tensor, steps: Optional[Tensor] = None, num_nodes: Optional[Tensor] = None) -> Tensor:
        if num_nodes is not None:
            mask = self.get_node_mask(num_nodes, prediction)
        else:
            mask = torch.ones_like(prediction, device=prediction.device).bool()

        if self.stage == Stage.HINT:
            steps_mask = get_steps_mask(steps, prediction)
            mask = mask & steps_mask

        # 2) Compute per‐element squared error (no reduction)
        sq_err = (prediction - target).pow(2)

        # 3) Cast mask to the same dtype, so we can multiply
        mask_f = mask.type_as(sq_err)

        # 4) Sum only the masked errors, then divide by number of masked elements
        total_err = (sq_err * mask_f).sum()
        count     = mask_f.sum()
        return total_err / count
        
        
    def forward(self, prediction: Tensor, target: Tensor, steps: Optional[Tensor] = None, num_nodes: Optional[Tensor] = None) -> Tensor:
        assert prediction.shape == target.shape, "Prediction and target must have the same shape"
        if self.type_ in [Type.MASK_ONE, Type.CATEGORICAL]:
            return self.eval_mask_one(prediction, target, steps, num_nodes)
        elif self.type_ == Type.MASK:
            return self.eval_mask(prediction, target, steps, num_nodes)
        elif self.type_ == Type.SCALAR:
            return self.eval_scalar(prediction, target, steps, num_nodes)
        elif self.type_ == Type.POINTER:
            return self.eval_pointer(prediction, target, steps, num_nodes)
        else:
            # import ipdb; ipdb.set_trace()
            raise ValueError("Invalid type")
        
class AlgoEvaluator(nn.Module):
    def __init__(self, spec: Spec, decode_hints: bool):
        super().__init__()
        self.spec, self.skip = self.process_spec(spec)
        self.evaluators = nn.ModuleDict()
        self.evaluators.add_module(Stage.OUTPUT, nn.ModuleDict())
        self.decode_hints = decode_hints
        if self.decode_hints:
            self.evaluators.add_module(Stage.HINT, nn.ModuleDict())

        for name, (stage, _, type_, _) in self.spec.items():
            if stage == Stage.OUTPUT:
                self.evaluators[Stage.OUTPUT].add_module(name, Evaluator(type_, stage))
            elif stage == Stage.HINT and self.decode_hints:
                self.evaluators[Stage.HINT].add_module(name, Evaluator(type_, stage))

    @staticmethod
    def process_spec(spec: Spec):
        new_spec = {}
        skips = set()
        for name, (stage, location, type_, metadata) in spec.items():
            if stage == Stage.OUTPUT and type_ == Type.PERMUTATION_POINTER:
                assert location == Location.NODE
                type_ = Type.POINTER # replace permutation pointer with pointer
                skips.add(f"{name}_mask") # remove the mask_one from the spec
            new_spec[name] = (stage, location, type_, metadata)

        if len(skips) > 0:
            for name in skips:
                dp = new_spec.pop(name)
                assert dp[2] == Type.MASK_ONE
                assert dp[1] == Location.NODE
        return new_spec, skips
    
    def replace_permutations_with_pointers(self, outputs: Output) -> Output:
        new_outputs = {}
        dont_copy = set()

        for skipped in self.skip:
            mask_one_name = skipped
            perm_name = skipped.replace("_mask", "")
            
            mask_one = outputs[mask_one_name]
            perm = outputs[perm_name]

            dont_copy.add(mask_one_name)
            dont_copy.add(perm_name)
            
            new_outputs[perm_name] = torch.where(
                mask_one > 0.5,
                torch.arange(perm.size(-1), device=perm.device),
                torch.argmax(perm, dim=-1)
            ).type_as(perm)

        for name in outputs:
            if name not in dont_copy:
                new_outputs[name] = outputs[name]
            
        return new_outputs

    def evaluate_hints(self, prediction: Hints, target: Hints, steps: Tensor, num_nodes: Optional[Tensor] = None) -> Hints:
        evaluations = {}
        if self.decode_hints:
            for name, evaluator in self.evaluators[Stage.HINT].items():
                evaluations[name] = evaluator(prediction[name], target[name], steps, num_nodes)
        return evaluations
    
    def evaluate_output(self, prediction: Output, target: Output, steps: Tensor, num_nodes: Optional[Tensor] = None) -> Output:
        evaluations = {}
        if len(self.skip) > 0:
            prediction = self.replace_permutations_with_pointers(prediction)
            target = self.replace_permutations_with_pointers(target)

        for name, evaluator in self.evaluators[Stage.OUTPUT].items():
            evaluations[name] = evaluator(prediction[name], target[name], steps, num_nodes)
        return evaluations

class AlgoLoss(nn.Module):
    def __init__(self, spec: Spec, decode_hints: bool):
        super().__init__()
        self.spec = spec
        self.decode_hints = decode_hints
        self.loss = nn.ModuleDict()
        self.loss.add_module(Stage.OUTPUT, nn.ModuleDict())

        if self.decode_hints:
            self.loss.add_module(Stage.HINT, nn.ModuleDict())

        for name, (stage, _, type_, _) in spec.items():
            if stage == Stage.OUTPUT:
                self.loss[Stage.OUTPUT].add_module(name, Loss(type_, stage))
            elif stage == Stage.HINT and self.decode_hints:
                self.loss[Stage.HINT].add_module(name, Loss(type_, stage))
    
    def hint_loss(self, prediction: Hints, target: Hints, steps: Tensor, num_nodes: Optional[Tensor] = None) -> Hints:
        losses = {}
        if self.decode_hints:
            for name, loss in self.loss[Stage.HINT].items():
                losses[name] = loss(prediction[name], target[name], steps, num_nodes)
        return losses

    def output_loss(self, prediction: Output, target: Output, steps: Tensor, num_nodes: Optional[Tensor] = None) -> Output:
        losses = {}
        for name, loss in self.loss[Stage.OUTPUT].items():
            losses[name] = loss(prediction[name], target[name], steps, num_nodes)
        return losses

class AlgoEncoder(nn.ModuleDict):
    def __init__(self, spec: Spec, 
                 hidden_dim: int,
                 encode_hints: bool = False):
        super().__init__()
        self.spec = spec
        self.encode_hints = encode_hints
        self.hidden_dim = hidden_dim
        self.encoders = nn.ModuleDict()
        self.encoders.add_module(Stage.INPUT, nn.ModuleDict())
        if encode_hints:
            self.encoders.add_module(Stage.HINT, nn.ModuleDict())

        for name, (stage, location, type_, metadata) in spec.items():
            encoder = Encoder(hidden_dim, 
                            location=location, 
                            type_=type_, 
                            num_classes=metadata.get("num_classes", None))
            if stage == Stage.INPUT:
                self.encoders[Stage.INPUT].add_module(name, encoder)
            elif stage == Stage.HINT and self.encode_hints:
                self.encoders[Stage.HINT].add_module(name, encoder)

    def forward(self, input: Input, step_hints: Hints, num_nodes: Optional[Tensor] = None) -> GraphFeatures:
        batch_size = input['pos'].shape[0]
        node_dim = input['pos'].shape[1]
        device = input['pos'].device
        graph_features = GraphFeatures.empty(batch_size, node_dim, self.hidden_dim, device=device)

        if num_nodes is not None:
            node_mask = batch_mask(num_nodes, node_dim, 2)
            graph_features.adj_mat = graph_features.adj_mat * node_mask.type_as(graph_features.adj_mat)

        for name, data in input.items():
            graph_features = self.encoders[Stage.INPUT][name].encode(data, graph_features, num_nodes)

        if self.encode_hints:
            for name, data in step_hints.items():
                graph_features = self.encoders[Stage.HINT][name].encode(data, graph_features, num_nodes)

        return graph_features
    
class AlgoDecoder(nn.ModuleDict):
    def __init__(self, spec: Spec, hidden_dim: int, decode_hints: bool, hint_reconst_mode: ReconstMode, inf_bias: bool = False, 
                 inf_bias_edge: bool = False, edge_dim_multiplier: int = 2):
        super().__init__()
        self.spec = spec
        self.hidden_dim = hidden_dim
        self.inf_bias = inf_bias
        self.inf_bias_edge = inf_bias_edge
        self.decoders = nn.ModuleDict()
        self.decoders.add_module(Stage.OUTPUT, nn.ModuleDict())
        self.decode_hints = decode_hints
        self.output_decoder = nn.ModuleDict()

        if self.decode_hints:
            self.decoders.add_module(Stage.HINT, nn.ModuleDict())

        
        for name, (stage, location, type_, metadata) in spec.items():
            if stage == Stage.OUTPUT:
                decoder = Decoder(hidden_dim=hidden_dim, 
                                       location=location, 
                                       sinkhorn_steps=50,
                                       sinkhorn_temp=0.1,
                                       hard_mode=ReconstMode.HARD,
                                       num_classes=metadata.get("num_classes", None),
                                       type_=type_,
                                       edge_dim_multiplier=edge_dim_multiplier,
                                       inf_bias=self.inf_bias,
                                       inf_bias_edge=self.inf_bias_edge)
                self.decoders[Stage.OUTPUT].add_module(name, decoder)
            elif stage == Stage.HINT and self.decode_hints:
                decoder = Decoder(hidden_dim=hidden_dim, 
                                       location=location, 
                                       sinkhorn_steps=25,
                                       sinkhorn_temp=0.1,
                                       num_classes=metadata.get("num_classes", None),
                                       hard_mode=hint_reconst_mode,
                                       type_=type_,
                                       edge_dim_multiplier=edge_dim_multiplier,
                                       inf_bias=self.inf_bias,
                                       inf_bias_edge=self.inf_bias_edge)
                self.decoders[Stage.HINT].add_module(name, decoder)
    

    def hints_decode(self, graph_features: GraphFeatures, num_nodes: Optional[Tensor] = None) -> Tuple[Hints, Hints]:
        hints: Hints = {}
        raw_hints: Hints = {}
        if self.decode_hints:
            for name, decoder in self.decoders[Stage.HINT].items():
                raw_hints[name], hints[name] = decoder.decode(graph_features, num_nodes)
        return raw_hints, hints
    
    def output_decode(self, graph_features: GraphFeatures, num_nodes: Optional[Tensor] = None) -> Tuple[Output, Output]:
        raw_output: Output = {}
        output: Output = {}
        for name, decoder in self.decoders[Stage.OUTPUT].items():
            raw_output[name], output[name] = decoder.decode(graph_features, num_nodes)
        return raw_output, output



class ModelState(NamedTuple):
    processor_state: Tensor
    lstm_state: Optional[LSTMState] = None
    last_predicted_hint: Optional[Hints] = None

    @classmethod
    def init(cls, batch_size: int, nb_nodes: int, hidden_dim: int, use_lstm: bool, last_predicted_hint: Optional[Hints] = None, device: torch.device = torch.device("cpu")) -> "ModelState":
        # Important to mark requires_grad=True to prevent recompilation on every step
        return cls(processor_state=torch.zeros((batch_size, nb_nodes, hidden_dim), device=device, requires_grad=True),
                   lstm_state=LSTMState.empty((batch_size * nb_nodes, hidden_dim), device=device, requires_grad=True) if use_lstm else None,
                   last_predicted_hint=last_predicted_hint)
    
    def detach(self) -> "ModelState":
        # Important to mark requires_grad=True after clone.detach() to prevent recompilation
        detached_hints = {k: v.clone().detach() for k, v in self.last_predicted_hint.items()} if self.last_predicted_hint is not None else None
        return ModelState(processor_state=self.processor_state.clone().detach().requires_grad_(True),
                          lstm_state=self.lstm_state.clone().detach().requires_grad_(True) if self.lstm_state is not None else None,
                          last_predicted_hint=detached_hints)


class AlgoModel(torch.nn.Module):
    def __init__(self, 
                 spec: Spec, 
                 processor: ProcessorBase, 
                 hidden_dim: int, 
                 encode_hints: bool, 
                 decode_hints: bool,
                 use_lstm: bool = True,
                 hint_reconst_mode: ReconstMode = ReconstMode.HARD,
                 hint_teacher_forcing: float = 1.0,
                 dropout: float = 0.0): 
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encode_hints = encode_hints
        self.decode_hints = decode_hints
        self.spec = spec
        self.hint_teacher_forcing = hint_teacher_forcing
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.encoder = AlgoEncoder(spec, hidden_dim, encode_hints)
        self.processor = processor
        self.loss = AlgoLoss(spec=spec, decode_hints=decode_hints)
        self.evaluator = AlgoEvaluator(spec, decode_hints=decode_hints)
        self.decoder = AlgoDecoder(spec, 
                                hidden_dim=hidden_dim, 
                                hint_reconst_mode=hint_reconst_mode,
                                decode_hints=decode_hints,
                                inf_bias=self.processor.inf_bias,
                                edge_dim_multiplier=2 if self.processor.returns_edge_fts else 1,
                                inf_bias_edge=self.processor.inf_bias_edge)
        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm_cell = LSTMCell(self.hidden_dim, self.hidden_dim)

    def apply_lstm(self, model_state: ModelState) -> ModelState:
        if self.use_lstm:
            processor_state, lstm_state = model_state
            assert lstm_state is not None
            batch_size, nb_nodes, _ = processor_state.shape            
            # LSTM takes the procesor state as its input_x for the current time step
            lstm_input = processor_state.reshape(batch_size * nb_nodes, self.hidden_dim)
            lstm_out, lstm_state = self.lstm_cell(lstm_input, lstm_state)
            processor_state = lstm_out.reshape(batch_size, nb_nodes, self.hidden_dim)
            return ModelState(processor_state=processor_state, lstm_state=lstm_state, last_predicted_hint=model_state.last_predicted_hint)

        return model_state

    def step(self, 
             input: Input, 
             step_hint: Hints, 
             num_nodes: NumNodes, 
             model_state: ModelState,
             ) -> Tuple[Trajectory, Trajectory, ModelState]:
        
        graph_features: GraphFeatures = self.encoder(input, step_hint, num_nodes)

        nxt_processor_state, nxt_edge = model_state.processor_state, None 
        nxt_processor_state, nxt_edge = self.processor(
            graph_features,
            processor_state=nxt_processor_state, # hidden for GNN processor within the step's mp_steps
            num_nodes=num_nodes
        )

        # Output of GNN processing for this step becomes input to LSTM (if used)
        nxt_model_state = self.apply_lstm(ModelState(processor_state=self.dropout(nxt_processor_state), 
                                                     lstm_state=model_state.lstm_state))

        nodes_fts_decoder = torch.cat([graph_features.node_fts, model_state.processor_state, nxt_model_state.processor_state], dim=-1)
        edge_fts_decoder = graph_features.edge_fts 
        if nxt_edge is not None:
            edge_fts_decoder = torch.cat([graph_features.edge_fts, nxt_edge], dim=-1)

        graph_features_decoder = GraphFeatures(adj_mat=graph_features.adj_mat, 
                                           node_fts=nodes_fts_decoder, 
                                           edge_fts=edge_fts_decoder, 
                                           graph_fts=graph_features.graph_fts)
        
        raw_predicted_hints, predicted_hints = self.decoder.hints_decode(graph_features_decoder, num_nodes=num_nodes)
        raw_predicted_output, predicted_output = self.decoder.output_decode(graph_features_decoder, num_nodes=num_nodes)


        # Update the model state last predicted hint with the predicted hints of the current step
        nxt_model_state = ModelState(processor_state=model_state.processor_state,
                                      lstm_state=model_state.lstm_state,
                                      last_predicted_hint=predicted_hints)

        prediction = {Stage.HINT: predicted_hints, Stage.OUTPUT: predicted_output}
        raw_prediction = {Stage.HINT: raw_predicted_hints, Stage.OUTPUT: raw_predicted_output}
        return raw_prediction, prediction, nxt_model_state
    
    @staticmethod
    def get_hint_at_step(hints: Hints, start_step: int, end_step: Optional[int] = None) -> Hints:
        if len(hints) == 0:
            return {}
        if end_step is None:
            return {k: v[start_step] for k, v in hints.items()}
        return {k: v[start_step:end_step] for k, v in hints.items()}
      
    @staticmethod
    def append_step_hints(hints: Hints, step_hints: Hints) -> Hints:
        for k in step_hints:
            hint_k_data = step_hints[k].unsqueeze(0) # Add the time dimension
            if k in hints:
                hint_k_data = torch.cat([hints[k], hint_k_data], dim=0)
            hints[k] = hint_k_data
        return hints
    
    @staticmethod
    def stack_step_prediction(step_prediction: List[Union[Hints, Output]]) -> Union[Hints, Output]:
        stacked_step_prediction = {}
        for key in step_prediction[0]:
            data = []
            for i in range(len(step_prediction)):
                data.append(step_prediction[i][key])
            stacked_step_prediction[key] = torch.stack(data, dim=0)
        return stacked_step_prediction

    def teacher_force(self, orig_hints: Hints, predicted_hints: Hints) -> Hints:
        if self.hint_teacher_forcing == 1.0:
            return orig_hints
        
        if not self.training or self.hint_teacher_forcing == 0.0:
            # For inference, use the predicted features from the previous step
            return predicted_hints
        
        hints = {}
        for name, data in orig_hints.items():
            batch_size = data.shape[0]
            target_mask_shape = (batch_size,) + (1,) * (data.dim() - 1)
            teacher_forcing_probability = torch.full(target_mask_shape, self.hint_teacher_forcing)
            teacher_forcing_mask = torch.bernoulli(teacher_forcing_probability).bool()
            data = torch.where(teacher_forcing_mask, data, predicted_hints[name])
            hints[name] = data

        return hints

    def init_model_state(self, feature: Feature) -> ModelState:
        trajectory, num_steps = feature[0], feature[1]
        hint_at_step0 = self.get_hint_at_step(trajectory[Stage.HINT], start_step=0)
        device = num_steps.device
        batch_size = num_steps.shape[0]
        nb_nodes = trajectory[Stage.INPUT]['pos'].shape[1]
        return ModelState.init(
            batch_size=batch_size,
            nb_nodes=nb_nodes,
            hidden_dim=self.hidden_dim,
            use_lstm=self.use_lstm,
            device=device,
            last_predicted_hint=hint_at_step0
        )
        
    @staticmethod 
    def extract_last_step(output: Output, num_steps: NumSteps) -> Output:
        B = num_steps.shape[0]
        last_step  = num_steps - 1      # shape (B,)        
        batch_idx = torch.arange(B, device=num_steps.device)
        extracted_output: Output = {}
        for k, data in output.items():
            extracted_output[k] = data[last_step, batch_idx]
        return extracted_output
    
    def _loop(self, input: Input, hints: Hints, num_nodes: Tensor, model_state: ModelState) -> Tuple[Trajectory, Trajectory, ModelState]:
        max_steps = next(iter(hints.values())).shape[0]
         # Disable the loop as it makes the computational graph too large for torch.compile

        # Hints
        predicted_hints: List[Hints] = []
        raw_predicted_hints: List[Hints] = []

        predicted_outputs: List[Output] = []
        raw_predicted_outputs: List[Output] = []
        
        for step_idx in range(max_steps):
            orig_hints_at_step = self.get_hint_at_step(hints, step_idx)
            hints_at_step = self.teacher_force(orig_hints_at_step, model_state.last_predicted_hint)

            last_raw_predictions, last_predictions, model_state = self.step(
                                                                            input=input,
                                                                            step_hint=hints_at_step,
                                                                            num_nodes=num_nodes,
                                                                            model_state=model_state          
                                                                        )

            # Set the last predicted hints to the predicted hints of the current step
            predicted_hints.append(last_predictions[Stage.HINT])
            raw_predicted_hints.append(last_raw_predictions[Stage.HINT])
            predicted_outputs.append(last_predictions[Stage.OUTPUT])
            raw_predicted_outputs.append(last_raw_predictions[Stage.OUTPUT])

        raw_predictions = {
            Stage.HINT: self.stack_step_prediction(raw_predicted_hints),
            Stage.OUTPUT: self.stack_step_prediction(raw_predicted_outputs)
        }
        predictions = {
            Stage.HINT: self.stack_step_prediction(predicted_hints),
            Stage.OUTPUT: self.stack_step_prediction(predicted_outputs)
        }
            
        return raw_predictions, predictions, model_state


    def forward(self, feature: Feature, model_state: Optional[ModelState] = None) -> Tuple[Tuple[Trajectory, Trajectory, Trajectory], ModelState]:

        trajectory, num_steps, num_nodes = feature[0], feature[1], feature[2]
        input, hints, output = trajectory[Stage.INPUT], trajectory[Stage.HINT], trajectory[Stage.OUTPUT]
        max_steps = next(iter(hints.values())).shape[0]

        prev_model_state = self.init_model_state(feature) if model_state is None else model_state
        raw_predictions, predictions, nxt_model_state = self._loop(input=input, 
                                                                    hints=hints, 
                                                                    num_nodes=num_nodes, 
                                                                    model_state=prev_model_state)
        
        raw_predictions[Stage.OUTPUT] = self.extract_last_step(raw_predictions[Stage.OUTPUT], num_steps)
        predictions[Stage.OUTPUT] = self.extract_last_step(predictions[Stage.OUTPUT], num_steps)

        # first target hint is never predicted
        # predicted hints are shifted by one step, and the first predicted hint is actually second target hint
        target_hints = self.get_hint_at_step(hints, start_step=1, end_step=max_steps)
        raw_predictions[Stage.HINT] = self.get_hint_at_step(raw_predictions[Stage.HINT], start_step=0, end_step=max_steps-1)
        predictions[Stage.HINT] = self.get_hint_at_step(predictions[Stage.HINT], start_step=0, end_step=max_steps-1)

        hint_loss = self.loss.hint_loss(prediction=raw_predictions[Stage.HINT], target=target_hints, steps=num_steps, num_nodes=num_nodes)
        hint_evaluations = self.evaluator.evaluate_hints(prediction=predictions[Stage.HINT], target=target_hints, steps=num_steps, num_nodes=num_nodes)
        
        output_loss = self.loss.output_loss(prediction=raw_predictions[Stage.OUTPUT], target=output, steps=num_steps, num_nodes=num_nodes)
        output_evaluations = self.evaluator.evaluate_output(prediction=predictions[Stage.OUTPUT], target=output, steps=num_steps, num_nodes=num_nodes)

        losses = {
            Stage.HINT: hint_loss,
            Stage.OUTPUT: output_loss
        }

        evaluations = {
            Stage.HINT: hint_evaluations,
            Stage.OUTPUT: output_evaluations
        }

        return (predictions, losses, evaluations), nxt_model_state


DictFeature = Dict[Algorithm, Feature]
DictTrajectory = Dict[Algorithm, Trajectory]
DictModelState = Dict[Algorithm, ModelState]

class Model(torch.nn.Module):
    def __init__(self, 
                 specs: Dict[str, Spec], 
                 processor: ProcessorBase, 
                 hidden_dim: int, 
                 encode_hints: bool = True, 
                 decode_hints: bool = True,
                 use_lstm: bool = True,
                 hint_reconst_mode: ReconstMode = ReconstMode.SOFT,
                 hint_teacher_forcing: float = 0.0,
                 dropout: float = 0.0):
        super().__init__()
        self.models = nn.ModuleDict()
        self.specs = specs

        for algo_name, spec in specs.items():
            self.models.add_module(algo_name, AlgoModel(spec=spec, 
                                                            processor=processor, 
                                                            hidden_dim=hidden_dim, 
                                                            encode_hints=encode_hints, 
                                                            decode_hints=decode_hints, 
                                                            use_lstm=use_lstm,
                                                            hint_reconst_mode=hint_reconst_mode, 
                                                            hint_teacher_forcing=hint_teacher_forcing, 
                                                            dropout=dropout))
            
    def compile(self):
        torch._dynamo.config.cache_size_limit = 256
        for name, mod in self.models.items():
            fullgraph = True
            spec = self.models[name].spec
            if 'pred' in spec and spec['pred'][2] == Type.PERMUTATION_POINTER:
                print(f"▸ disabling fullgraph for {name}…")
                fullgraph = False

            print(f"▸ compiling {name}…  with fullgraph={fullgraph}")
            self.models[name] = torch.compile(
                mod,
                fullgraph=fullgraph,
                mode="reduce-overhead",
                backend="inductor",
            )
        return self

    def init_model_state(self, algorithm: Algorithm, feature: Feature) -> ModelState:
        return self.models[algorithm].init_model_state(feature)
        
    def forward(self, features: DictFeature, 
                model_state: Optional[DictModelState] = None) -> Tuple[Tuple[DictTrajectory, DictTrajectory, DictModelState], DictModelState]:
        predictions, losses, evaluations, nxt_model_state = {}, {}, {}, {}
        for algo, feature in features.items():
            if model_state is None or algo not in model_state:
                model_state = {algo: self.init_model_state(algo, feature)}

            (predictions[algo], losses[algo], evaluations[algo]), nxt_model_state[algo] = self.models[algo](feature, model_state[algo])
        return (predictions, losses, evaluations), nxt_model_state

