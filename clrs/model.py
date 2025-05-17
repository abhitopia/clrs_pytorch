from enum import Enum
from typing import Dict, Optional, Tuple, NamedTuple
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from .specs import Location, Type, Spec, Stage, Trajectory, Hints, Input, Output, OutputClass, AlgorithmEnum, Feature, NumNodes, NumSteps
from .utils import log_sinkhorn, Linear, batch_mask, expand
from .processors import GraphFeatures, Processor



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
    
    def get_loss(self, pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        mask = None
        if self.type_ == Type.SCALAR:
            loss = (pred - target)**2
        elif self.type_ == Type.MASK:
            # stable binary cross entropy or logistic loss
            loss = (F.relu(pred) - pred * target + torch.log1p(torch.exp(-torch.abs(pred))))
            mask = (target != OutputClass.MASKED).type_as(pred)
        elif self.type_ == Type.MASK_ONE:
            if self.stage == Stage.OUTPUT:
                masked_target = target * (target != OutputClass.MASKED).type_as(pred)
                loss = -torch.sum(masked_target * F.log_softmax(pred, -1), dim=-1, keepdim=True)
                mask = torch.any(target == OutputClass.POSITIVE, dim=-1).type_as(pred).unsqueeze(-1)
            else:
                loss = -torch.sum(target * F.log_softmax(pred, -1), dim=-1, keepdim=True)
                mask = None
        elif self.type_ == Type.CATEGORICAL:
            if self.stage == Stage.OUTPUT:
                masked_target = target * (target != OutputClass.MASKED).type_as(pred)
                loss = -torch.sum(masked_target * F.log_softmax(pred, -1), dim=-1, keepdim=True)
                mask = torch.any(target == OutputClass.POSITIVE, dim=-1).type_as(pred).unsqueeze(-1)
            else:
                loss = -torch.sum(target * F.log_softmax(pred, -1), dim=-1)
                mask = torch.any(target == OutputClass.POSITIVE, dim=-1).type_as(pred)
        elif self.type_ == Type.POINTER:
            loss = -torch.sum(target * F.log_softmax(pred, -1), dim=-1)
        elif self.type_ == Type.PERMUTATION_POINTER:
            # Predictions are NxN logits aiming to represent a doubly stochastic matrix.
            # Compute the cross entropy between doubly stochastic pred and truth_data
            loss = -torch.sum(target * pred, dim=-1)

        if mask is None:
            mask = torch.ones_like(loss)
        return loss, mask
    
    def forward(self, pred: Tensor, target: Tensor, num_steps: Optional[Tensor] = None) -> Tensor:
        loss, mask = self.get_loss(pred, target)
        assert loss.shape == mask.shape, f"loss.shape: {loss.shape}, mask.shape: {mask.shape}"
        dim = tuple(range(1, loss.ndim))
        
        if self.stage == Stage.HINT:
            assert num_steps is not None
            steps_mask = get_steps_mask(num_steps, loss).type_as(mask)
            mask = steps_mask * mask
            dim = (0,) + tuple(range(2, loss.ndim))

        loss = torch.sum(loss * mask, dim=dim) / (torch.sum(mask, dim=dim) + 1e-8)
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

    def encode_to_adj_mat(self, data: Tensor, adj_mat: Tensor) -> Tensor:
        if self.location == Location.NODE and self.type_ in [Type.POINTER, Type.PERMUTATION_POINTER]:
            adj_mat += ((data + data.permute(0, 2, 1)) > 0.5)
        elif self.location == Location.EDGE and self.type_ == Type.MASK:
            adj_mat += ((data + data.permute(0, 2, 1)) > 0.0)
        return (adj_mat > 0.).type_as(data)
    
    def encode_to_node_fts(self, data: Tensor, node_fts: Tensor) -> Tensor:
        is_pointer = (self.type_ in [Type.POINTER, Type.PERMUTATION_POINTER])
        if self.type_ != Type.CATEGORICAL:
            data = data.unsqueeze(-1)

        if (self.location == Location.NODE and not is_pointer) or (self.location == Location.GRAPH and self.type_ == Type.POINTER):
            encoding = self.encoders[0](data)
            node_fts += encoding
        return node_fts

    def encode_to_edge_fts(self, data: Tensor, edge_fts: Tensor, num_nodes: Optional[Tensor] = None) -> Tensor:
        if self.type_ != Type.CATEGORICAL:
            data = data.unsqueeze(-1)

        if self.location == Location.NODE and self.type_ in [Type.POINTER, Type.PERMUTATION_POINTER]:
            encoding = self.encoders[0](data)
            edge_fts += encoding
        elif self.location == Location.EDGE:
            encoding = self.encoders[0](data)
            if self.type_ == Type.POINTER:
                # Aggregate pointer contributions across sender and receiver nodes.
                encoding_2 = self.encoders[1](data)
                if num_nodes is not None:
                    sum_2dims = torch.sum(encoding, dim=1) + torch.sum(encoding_2, dim=2)
                    edge_fts += sum_2dims/expand(num_nodes, sum_2dims)
                else:
                    edge_fts += torch.mean(encoding, dim=1) + torch.mean(encoding_2, dim=2)
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
        graph_features.node_fts = self.encode_to_node_fts(data, graph_features.node_fts)
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
        
    def decode_node_fts(self, graph_features: GraphFeatures) -> Tensor:
        """
        graph_features.node_fts shape: (batch_size, nb_nodes, hidden_dim)
        graph_features.edge_fts shape: (batch_size, nb_nodes, nb_nodes, hidden_dim)
        graph_features.adj_mat shape: (batch_size, nb_nodes, nb_nodes)
        graph_features.graph_fts shape: (batch_size, hidden_dim)
        """
        if self.type_ in [Type.SCALAR, Type.MASK, Type.MASK_ONE]:
            preds = self.decoders[0](graph_features.node_fts).squeeze(-1) # (batch_size, nb_nodes)
        elif self.type_ == Type.CATEGORICAL:
            preds = self.decoders[0](graph_features.node_fts) # (batch_size, nb_nodes, num_cats)
        elif self.type_ in [Type.POINTER, Type.PERMUTATION_POINTER]:
            p_1 = self.decoders[0](graph_features.node_fts) # (batch_size, nb_nodes, hidden_dim)
            p_2 = self.decoders[1](graph_features.node_fts) # (batch_size, nb_nodes, hidden_dim)
            p_3 = self.decoders[2](graph_features.edge_fts) # (batch_size, nb_nodes, nb_nodes, hidden_dim)

            p_e = torch.unsqueeze(p_2, -2) + p_3 # (batch_size, nb_nodes, nb_nodes, hidden_dim)
            p_m = torch.maximum(torch.unsqueeze(p_1, -2), p_e.permute(0, 2, 1, 3)) # (batch_size, nb_nodes, nb_nodes, hidden_dim)
            preds = self.decoders[3](p_m).squeeze(-1) # (batch_size, nb_nodes, nb_nodes)

            if self.inf_bias:
                per_batch_min = torch.amin(preds, dim=tuple(range(1, preds.dim())),keepdim=True)  # shape: (batch, 1, 1, …)
                neg_one = torch.tensor(-1.0, device=preds.device, dtype=preds.dtype)
                # mask preds wherever adj_mat <= 0.5
                preds = torch.where(graph_features.adj_mat > 0.5, preds, torch.minimum(neg_one, per_batch_min - 1.0))
            if self.type_ == Type.PERMUTATION_POINTER:
                if not self.training:  # testing or validation, no Gumbel noise
                    preds = log_sinkhorn(x=preds, steps=10, temperature=0.1, zero_diagonal=True, add_noise=False)
                else:  # training, add Gumbel noise
                    preds = log_sinkhorn(x=preds, steps=10, temperature=0.1, zero_diagonal=True, add_noise=True)
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

        N = graph_features.edge_fts.size(-2)

        pred_1 = self.decoders[0](graph_features.node_fts) # (batch_size, nb_nodes, 1/num_cats/hidden_dim)
        pred_2 = self.decoders[1](graph_features.node_fts) # (batch_size, nb_nodes, 1/num_cats/hidden_dim)
        pred_e = self.decoders[2](graph_features.edge_fts) # (batch_size, nb_nodes, nb_nodes, 1/num_cats/hidden_dim)
        pred = (torch.unsqueeze(pred_1, -2) + torch.unsqueeze(pred_2, -3) + pred_e) # (batch_size, nb_nodes, nb_nodes, 1/num_cats/hidden_dim)

        if self.type_ in [Type.SCALAR, Type.MASK, Type.MASK_ONE]:
            preds = pred.squeeze(-1)
        elif self.type_ == Type.CATEGORICAL:
            preds = pred
        elif self.type_ == Type.POINTER:
            pred_2 = self.decoders[3](graph_features.node_fts) # (batch_size, nb_nodes, hidden_dim)
            if num_nodes is not None:
                pred_2 = pred_2.masked_fill(~expand(batch_mask(num_nodes, N, 1), pred_2), float("-inf"))
                pred = pred.masked_fill(~expand(batch_mask(num_nodes, N, 2), pred), float("-inf"))
                p_m = torch.max(pred.unsqueeze(-2), pred_2.unsqueeze(-3).unsqueeze(-3)) # (batch_size, nb_nodes, nb_nodes, nb_nodes, hidden_dim)
                p_m = torch.where(expand(batch_mask(num_nodes, N, 3), p_m), p_m, torch.tensor(0.0, device=p_m.device, dtype=p_m.dtype))
            else:
                p_m = torch.max(pred.unsqueeze(-2), pred_2.unsqueeze(-3).unsqueeze(-3)) # (batch_size, nb_nodes, nb_nodes, nb_nodes, hidden_dim)
            preds = self.decoders[4](p_m).squeeze(-1) # (batch_size, nb_nodes, nb_nodes, nb_nodes)
        else:
            raise ValueError("Invalid output type")
        if self.inf_bias_edge and self.type_ in [Type.MASK, Type.MASK_ONE]:
            import ipdb; ipdb.set_trace()
            per_batch_min = torch.amin(preds, dim=tuple(range(1, preds.dim())), keepdim=True)
            neg_one = torch.tensor(-1.0, device=preds.device, dtype=preds.dtype)
            preds = torch.where(graph_features.adj_mat > 0.5,
                            preds,
                            torch.minimum(neg_one, per_batch_min - 1.0))

        return preds
    
    def decode_graph_fts(self, graph_features: GraphFeatures) -> Tensor:
        """Decodes graph features."""

        gr_emb = torch.max(graph_features.node_fts, dim=-2, keepdim=False).values # (batch_size, node_dim)
        pred_n = self.decoders[0](gr_emb) # (batch_size, 1/num_cats)
        pred_g = self.decoders[1](graph_features.graph_fts) # (batch_size, 1/num_cats)
        pred = pred_n + pred_g # (batch_size, 1/num_cats)
        if self.type_ in [Type.SCALAR, Type.MASK, Type.MASK_ONE]:
            preds = pred.squeeze(-1) # (batch_size,)
        elif self.type_ == Type.CATEGORICAL:
            preds = pred # (batch_size, num_cats)
        elif self.type_ == Type.POINTER:
            pred_2 = self.decoders[2](graph_features.node_fts) # (batch_size, nb_nodes, 1)
            ptr_p = pred.unsqueeze(1) + pred_2.permute(0, 2, 1) # (batch_size, 1, nb_nodes)
            preds = ptr_p.squeeze(1)
        else:
            raise ValueError("Invalid output type")

        return preds

    def postprocess(self, data: Tensor) -> Tensor:
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
                data = F.softmax(data, dim=-1)
        elif self.type_ == Type.POINTER:
            if hard:
                data = F.one_hot(data.argmax(dim=-1).long(), data.shape[-1]).type_as(data)
            else:
                data = F.softmax(data, dim=-1)
        elif self.type_ == Type.PERMUTATION_POINTER:
            # Convert the matrix of logits to a doubly stochastic matrix.
            data = log_sinkhorn(
                x=data,
                steps=self.sinkhorn_steps,
                temperature=self.sinkhorn_temp,
                zero_diagonal=True,
                add_noise=False)
            data = torch.exp(data)
            if hard:
                data = F.one_hot(data.argmax(dim=-1), data.shape[-1])
        else:
            raise ValueError("Invalid type")

        return data

    def decode(self, graph_features: GraphFeatures, num_nodes: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.location == Location.NODE:
            raw_result = self.decode_node_fts(graph_features)
        elif self.location == Location.EDGE:
            raw_result = self.decode_edge_fts(graph_features, num_nodes)
        elif self.location == Location.GRAPH:
            raw_result = self.decode_graph_fts(graph_features)
        return self.postprocess(raw_result),raw_result

class Evaluator(nn.Module):
    def __init__(self, type_: Type, stage: Stage):
        super().__init__()
        self.type_ = type_
        self.stage = stage
        self.evaluations = nn.ModuleDict()

    def eval_mask_one(self, prediction: Tensor, target: Tensor, steps: Optional[Tensor] = None) -> Tensor:
        # turn one-hot vectors into integer class labels
        pred_labels = prediction.argmax(dim=-1)    # shape (...,)
        truth_labels = target.argmax(dim=-1)  # shape (...,)
        # build mask of entries we care about
        valid = truth_labels != OutputClass.MASKED  # shape (...,), boolean

        if self.stage == Stage.HINT:
            assert steps is not None, "Steps must be provided for hint stage"
            steps_mask = get_steps_mask(steps, valid).type_as(valid)
            valid = valid * steps_mask

        # compute accuracy only where valid
        return (pred_labels[valid] == truth_labels[valid]).float().mean()
    
    def eval_mask(self, prediction: Tensor, target: Tensor, steps: Optional[Tensor] = None) -> Tensor:
        # 1) Build float mask of "valid" positions
        valid = (target != OutputClass.MASKED).float()

        if self.stage == Stage.HINT:
            assert steps is not None, "Steps must be provided for hint stage"
            steps_mask = get_steps_mask(steps, valid).type_as(valid)
            valid = valid * steps_mask

        # 2) Boolean tensors for positive
        pred_pos  = prediction > 0.5
        truth_pos = target > 0.5

        # 3) True positives / false positives / false negatives
        tp = ((pred_pos  & truth_pos).float() * valid).sum()
        fp = ((pred_pos  & ~truth_pos).float() * valid).sum()
        fn = ((~pred_pos & truth_pos).float() * valid).sum()

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
        
    def forward(self, prediction: Tensor, target: Tensor, steps: Optional[Tensor] = None) -> Tensor:
        assert prediction.shape == target.shape, "Prediction and target must have the same shape"

        if self.type_ in [Type.MASK_ONE, Type.CATEGORICAL]:
            return self.eval_mask_one(prediction, target, steps)
        elif self.type_ == Type.MASK:
            return self.eval_mask(prediction, target, steps)
        elif self.type_ == Type.SCALAR:
            if self.stage == Stage.HINT:
                assert steps is not None, "Steps must be provided for hint stage"
                steps_mask = get_steps_mask(steps, prediction)    
                return F.mse_loss(prediction[steps_mask], target[steps_mask])
            return F.mse_loss(prediction, target)
        elif self.type_ == Type.POINTER:
            if self.stage == Stage.HINT:
                assert steps is not None, "Steps must be provided for hint stage"
                steps_mask = get_steps_mask(steps, prediction)
                return (prediction[steps_mask] == target[steps_mask]).float().mean()
            return (prediction == target).float().mean()
        else:
            import ipdb; ipdb.set_trace()
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
    
    def replace_permutations_with_pointers(self, trajectory: Trajectory) -> Trajectory:
        # Create a new dictionary for the output trajectory.
        # This will hold copies of the inner dictionaries for each stage.
        new_trajectory = {k: v.copy() for k, v in trajectory.items()}
        output_data_to_modify = new_trajectory[Stage.OUTPUT]

        for skipped in self.skip:
            mask_one_name = skipped
            perm_name = skipped.replace("_mask", "")
            
            mask_one = output_data_to_modify.pop(mask_one_name)
            perm = output_data_to_modify.pop(perm_name)
            
            output_data_to_modify[perm_name] = torch.where(
                mask_one > 0.5,
                torch.arange(perm.size(-1), device=perm.device),
                torch.argmax(perm, dim=-1)
            ).type_as(perm)
            
        return new_trajectory

    def forward(self, prediction: Trajectory, target: Trajectory, steps: Tensor) -> Trajectory:
        evaluations = {stage: {} for stage in self.evaluators.keys()}

        if len(self.skip) > 0:
            prediction = self.replace_permutations_with_pointers(prediction)
            target = self.replace_permutations_with_pointers(target)
        
        for name, evaluator in self.evaluators[Stage.OUTPUT].items():
            evaluations[Stage.OUTPUT][name] = evaluator(prediction[Stage.OUTPUT][name], target[Stage.OUTPUT][name], steps)

        if self.decode_hints:
            for name, evaluator in self.evaluators[Stage.HINT].items():
                evaluations[Stage.HINT][name] = evaluator(prediction[Stage.HINT][name], target[Stage.HINT][name], steps)
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

    def forward(self, prediction: Trajectory, target: Trajectory, steps: Tensor) -> Trajectory:
        losses = {Stage.OUTPUT: {}}

        for name, loss in self.loss[Stage.OUTPUT].items():
            losses[Stage.OUTPUT][name] = loss(prediction[Stage.OUTPUT][name], target[Stage.OUTPUT][name], steps)

        if self.decode_hints:
            losses[Stage.HINT] = {}
            for name, loss in self.loss[Stage.HINT].items():
                losses[Stage.HINT][name] = loss(prediction[Stage.HINT][name], target[Stage.HINT][name], steps)
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

    def forward(self, graph_features: GraphFeatures, num_nodes: Optional[Tensor] = None) -> Tuple[Trajectory, Trajectory]:
        trajectory = {Stage.OUTPUT: {}, Stage.HINT: {}}
        raw_trajectory = {Stage.OUTPUT: {}, Stage.HINT: {}}
        
        for name, decoder in self.decoders[Stage.OUTPUT].items():
            trajectory[Stage.OUTPUT][name], raw_trajectory[Stage.OUTPUT][name] = decoder.decode(graph_features, num_nodes)

        if self.decode_hints:
            for name, decoder in self.decoders[Stage.HINT].items():
                trajectory[Stage.HINT][name], raw_trajectory[Stage.HINT][name] = decoder.decode(graph_features, num_nodes)

        return trajectory, raw_trajectory

class AlgoModel(torch.nn.Module):
    def __init__(self, 
                 spec: Spec, 
                 processor: Processor, 
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
        self.hint_teacher_forcing = hint_teacher_forcing
        self.dropout = nn.Dropout(dropout)
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


    def apply_lstm(self, processor_state: Tensor, lstm_state: Optional[LSTMState]) -> Tuple[Tensor, Optional[LSTMState]]:
        if self.use_lstm:
            assert lstm_state is not None
            batch_size, nb_nodes, _ = processor_state.shape            
            # LSTM takes the procesor state as its input_x for the current time step
            lstm_input = processor_state.reshape(batch_size * nb_nodes, self.hidden_dim)
            lstm_out, lstm_state = self.lstm_cell(lstm_input, lstm_state)
            processor_state = lstm_out.reshape(batch_size, nb_nodes, self.hidden_dim)

        return processor_state, lstm_state

    def step(self, 
             input: Input, 
             step_hint: Hints, 
             num_nodes: NumNodes, 
             processor_state: Tensor, # Processor's h_{t-1} for this step
             lstm_state: Optional[LSTMState] = None # LSTM's (h,c) from previous step
             ) -> Tuple[Output, Hints, Tensor, Optional[LSTMState]]:
        
        graph_features: GraphFeatures = self.encoder(input, step_hint, num_nodes)

        nxt_processor_state, nxt_edge = processor_state, None 
        nxt_processor_state, nxt_edge = self.processor(
            graph_features,
            processor_state=nxt_processor_state, # hidden for GNN processor within the step's mp_steps
        )

        # Output of GNN processing for this step becomes input to LSTM (if used)
        nxt_processor_state = self.dropout(nxt_processor_state)
        nxt_processor_state, nxt_lstm_state = self.apply_lstm(nxt_processor_state, lstm_state)

        # Assemble features for the decoder
        # nodes_fts_decoder uses:
        # - graph_features.node_fts (current encoded inputs for this step)
        # - processor_state (Processor's h_{t-1} from the previous overall step)
        # - nxt_processor_state (Processor's h_t from the current step)
        nodes_fts_decoder = torch.cat([graph_features.node_fts, processor_state, nxt_processor_state], dim=-1)
        edge_fts_decoder = graph_features.edge_fts 
        if nxt_edge is not None:
            edge_fts_decoder = torch.cat([graph_features.edge_fts, nxt_edge], dim=-1)

        graph_features_decoder = GraphFeatures(adj_mat=graph_features.adj_mat, 
                                           node_fts=nodes_fts_decoder, 
                                           edge_fts=edge_fts_decoder, 
                                           graph_fts=graph_features.graph_fts)
        
        prediction_step, raw_prediction_step = self.decoder(graph_features_decoder)
        return prediction_step, raw_prediction_step, nxt_processor_state, nxt_lstm_state
    
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

    def teacher_force(self, orig_hints: Hints, predicted_hints: Hints) -> Hints:
        if len(predicted_hints) == 0 or self.hint_teacher_forcing == 1.0:
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

    def merge_predicted_output(self, output: Output, predicted_output: Output, not_done_mask: Tensor) -> Output:
        for k, data in predicted_output.items():
            if k not in output:
                output[k] = data
            else:
                target_mask_shape = (not_done_mask.shape[0],) + (1,) * (data.dim() - 1)
                keep_prediction = not_done_mask.view(target_mask_shape)
                output[k] = torch.where(keep_prediction, predicted_output[k], output[k])
        return output

    def forward(self, feature: Feature) -> Tuple[Trajectory, Trajectory, Trajectory]:
        trajectory, num_steps, num_nodes = feature[0], feature[1], feature[2]
        input, hints, output = trajectory[Stage.INPUT], trajectory[Stage.HINT], trajectory[Stage.OUTPUT]

        prediction: Trajectory = {Stage.OUTPUT: {}, Stage.HINT: {}}
        raw_prediction: Trajectory = {Stage.OUTPUT: {}, Stage.HINT: {}}

        # Determine device for new tensors
        device = num_steps.device
        batch_size = num_steps.shape[0]
        nb_nodes = next(iter(input.values())).shape[1]
        max_steps = next(iter(hints.values())).shape[0]
        
        # Initialize processor_state (h_{t-1} for the overall step recurrence for processor)
        processor_state = torch.zeros((batch_size, nb_nodes, self.hidden_dim), device=device)
        
        # Initialize lstm_state (LSTM's own (h,c) from the previous time step)
        lstm_state: Optional[LSTMState] = LSTMState.empty((batch_size * nb_nodes, self.hidden_dim), device=device) if self.use_lstm else None

        for step_idx in range(max_steps):
            orig_hints_at_step = self.get_hint_at_step(hints, step_idx)
            predicted_hints_at_step = self.get_hint_at_step(prediction[Stage.HINT], step_idx-1) 
            hints_at_step = self.teacher_force(orig_hints_at_step, predicted_hints_at_step)
            
            prediction_step, raw_prediction_step, processor_state, lstm_state = self.step(
                input=input,
                step_hint=hints_at_step,
                num_nodes=num_nodes,
                processor_state=processor_state,          
                lstm_state=lstm_state                     
            )

            not_done_mask = num_steps > (step_idx + 1)
            prediction[Stage.HINT] = self.append_step_hints(prediction[Stage.HINT], prediction_step[Stage.HINT])
            prediction[Stage.OUTPUT] = self.merge_predicted_output(output=prediction[Stage.OUTPUT], 
                                                        predicted_output=prediction_step[Stage.OUTPUT], 
                                                        not_done_mask=not_done_mask)
            raw_prediction[Stage.HINT] = self.append_step_hints(raw_prediction[Stage.HINT], raw_prediction_step[Stage.HINT])
            raw_prediction[Stage.OUTPUT] = self.merge_predicted_output(output=raw_prediction[Stage.OUTPUT], 
                                                    predicted_output=raw_prediction_step[Stage.OUTPUT], 
                                                    not_done_mask=not_done_mask)

        # first target hint is never predicted
        # predicted hints are shifted by one step, and the first predicted hint is actually second target hint
        target = {
            Stage.OUTPUT: output,
            Stage.HINT: self.get_hint_at_step(hints, start_step=1, end_step=max_steps)
        }
        raw_prediction = {
            Stage.OUTPUT: raw_prediction[Stage.OUTPUT],
            Stage.HINT: self.get_hint_at_step(raw_prediction[Stage.HINT], start_step=0, end_step=max_steps-1)
        }
        prediction = {
            Stage.OUTPUT: prediction[Stage.OUTPUT],
            Stage.HINT: self.get_hint_at_step(prediction[Stage.HINT], start_step=1, end_step=max_steps)
        }
        loss = self.loss(prediction=raw_prediction, target=target, steps=num_steps)
        evaluations = self.evaluator(prediction=prediction, target=target, steps=num_steps)
        return prediction, loss, evaluations


class Model(torch.nn.Module):
    def __init__(self, 
                 specs: Dict[str, Spec], 
                 processor: Processor, 
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
        for algo_name, model in self.models.items():
            self.models[algo_name] = torch.compile(model, fullgraph=True, mode="reduce-overhead", backend="inductor")
        
    def forward(self, features: Dict[AlgorithmEnum, Feature]) -> Tuple[Dict[AlgorithmEnum, Trajectory], Dict[AlgorithmEnum, Trajectory]]:
        predictions, losses, evaluations = {}, {}, {}
        for algo, feature in features.items():
            predictions[algo], losses[algo], evaluations[algo] = self.models[algo](feature)
        return predictions, losses, evaluations


if __name__ == "__main__":
    import pytorch_lightning as pl
    pl.seed_everything(42)
    torch.use_deterministic_algorithms(True)
    from .dataset import get_dataset
    from torch.utils.data import DataLoader
    from .processors import ProcessorFactory

    algo = 'matrix_chain_order'
    algos = [AlgorithmEnum.matrix_chain_order]

    ds1 = get_dataset(algos=algos,
                      trajectory_sizes=[4],
                      num_samples=200,
                      stacked=False,
                      generate_on_the_fly=False,
                      static_batch_size=False)
    ds2 = get_dataset(algos=algos,
                      trajectory_sizes=[4, 16],
                      num_samples=200,
                      stacked=False,
                      generate_on_the_fly=False,
                      static_batch_size=True)
    
    
    batch_size = 32
    dl1 = ds1.get_dataloader(
        batch_size=batch_size,
        shuffle=False, 
        drop_last=False,
        num_workers=0
    )
    dl2 = ds2.get_dataloader(
        batch_size=batch_size,
        shuffle=False, 
        drop_last=False,
        num_workers=0
    )

    b1, b2 = next(iter(dl1)), next(iter(dl2))
    # algo = 'bfs'

    spec = ds1.specs[algo]
    b1 = b1[algo]
    b2 = b2[algo]
    t1, s1, n1 = b1[0], b1[1], b1[2]
    t2, s2, n2 = b2[0], b2[1], b2[2]
    i1, h1, o1 = t1[Stage.INPUT], t1[Stage.HINT], t1[Stage.OUTPUT]
    i2, h2, o2 = t2[Stage.INPUT], t2[Stage.HINT], t2[Stage.OUTPUT]

    from rich import print
    print(spec)

    hidden_dim = 128
    encode_hints = True
    decode_hints = True
    use_lstm = True
    hint_reconst_mode = ReconstMode.SOFT
    hint_teacher_forcing = 0.0
    dropout = 0.0


    process_classes = [p for p in list(ProcessorFactory)]

    import ipdb; ipdb.set_trace()
    for processor_cls in process_classes:
        print('Testing', processor_cls.name)
        processor = processor_cls(hidden_dim=hidden_dim, mp_steps=1)
        model = Model(specs=ds1.specs,
                    processor=processor,
                    hidden_dim=hidden_dim).models[algo]
        
        h1_s0 = model.get_hint_at_step(h1, 0)
        h2_s0 = model.get_hint_at_step(h2, 0)

        g1 = GraphFeatures.empty(batch_size, 4, hidden_dim)
        g1.adj_mat = g1.adj_mat * batch_mask(n1, 4, 2).type_as(g1.adj_mat)
        g2 = GraphFeatures.empty(batch_size, 16, hidden_dim)
        g2.adj_mat = g2.adj_mat * batch_mask(n2, 16, 2).type_as(g2.adj_mat)

        # i1A, i1adj, i1pos, i1s = i1['A'], i1['adj'], i1['pos'], i1['s']
        # i2A, i2adj, i2pos, i2s = i2['A'], i2['adj'], i2['pos'], i2['s']
        # h1pi_h, h1reach_h = h1_s0['pi_h'], h1_s0['reach_h']
        # h2pi_h, h2reach_h = h2_s0['pi_h'], h2_s0['reach_h']

        # model.encoder(i1, h1_s0, n1)
        # model.encoder(i2, h2_s0, n2)
        # import ipdb; ipdb.set_trace()
        # model.encoder.encoders[Stage.INPUT]['A'].encode(i1A, g1)
        # model.encoder.encoders[Stage.INPUT]['A'].encode(i2A, g2)
        # model.encoder.encoders[Stage.INPUT]['adj'].encode(i1adj, g1)
        # model.encoder.encoders[Stage.INPUT]['adj'].encode(i2adj, g2)
        # model.encoder.encoders[Stage.INPUT]['pos'].encode(i1pos, g1)
        # model.encoder.encoders[Stage.INPUT]['pos'].encode(i2pos, g2)
        # model.encoder.encoders[Stage.INPUT]['s'].encode(i1s, g1)
        # model.encoder.encoders[Stage.INPUT]['s'].encode(i2s, g2)
        # model.encoder.encoders[Stage.HINT]['pi_h'].encode(h1pi_h, g1)
        # model.encoder.encoders[Stage.HINT]['pi_h'].encode(h2pi_h, g2)
        # model.encoder.encoders[Stage.HINT]['reach_h'].encode(h1reach_h, g1)
        # model.encoder.encoders[Stage.HINT]['reach_h'].encode(h2reach_h, g2)


        g1 = model.encoder(i1, h1_s0, None)
        g2 = model.encoder(i2, h2_s0, n2)

  

        assert (g1.adj_mat == g2.adj_mat[:, :4, :4]).all()
        assert (g2.adj_mat[:, :4, 4:] == 0).all()
        assert (g2.adj_mat[:, 4:, :4] == 0).all()
        assert (g2.adj_mat[:, 4:, 4:] == 0).all()
        assert (g1.node_fts == g2.node_fts[:, :4, :]).all()
        assert (g1.node_fts[:, 4:, :] == 0).all()
        assert (g1.edge_fts == g2.edge_fts[:, :4, :4, :]).all()
        assert (g1.edge_fts[:, 4:, :4, :] == 0).all()
        assert (g1.edge_fts[:, :4, 4:, :] == 0).all()
        assert (g1.edge_fts[:, 4:, 4:, :] == 0).all()
        assert (g1.graph_fts == g2.graph_fts[:, :]).all()

        # ps1 = torch.zeros((batch_size, 4, hidden_dim))
        # ps2 = torch.zeros((batch_size, 16, hidden_dim))

        # # import ipdb; ipdb.set_trace()
        # nps1, nxe1 = model.processor(g1, processor_state=ps1, num_nodes=None)
        # nps2, nxe2 = model.processor(g2, processor_state=ps2, num_nodes=n2)

        # try:
        #     assert (nps1[:, :, :] == nps2[:, :4, :]).all()
        # except Exception as e:
        #     print("Trying allclose for {}".format(processor_cls.name))
        #     assert torch.allclose(nps1[:, :, :], nps2[:, :4, :], atol=1e-6, rtol=1e-6)
            
        # assert (nps2[:, 4:, :] == 0).all()
 

        # if nxe1 is None: 
        #     assert nxe2 is None
        # else:
        #     assert (nxe1 == nxe2[:, :4, :4, :]).all()
        #     # import ipdb; ipdb.set_trace()
        #     assert (nxe2[:, 4:, :4, :] == 0).all()
        #     assert (nxe2[:, :4, 4:, :] == 0).all()
        #     assert (nxe2[:, 4:, 4:, :] == 0).all()


        # nfd1 = torch.cat([g1.node_fts, ps1, nps1], dim=-1)
        # nfd2 = torch.cat([g2.node_fts, ps2, nps2], dim=-1)

        # if nxe1 is not None:
        #     nxe1 = torch.cat([g1.edge_fts, nxe1], dim=-1)
        #     nxe2 = torch.cat([g2.edge_fts, nxe2], dim=-1)
        # else:
        #     nxe1 = g1.edge_fts
        #     nxe2 = g2.edge_fts
        
        # gd1 = GraphFeatures(adj_mat=g1.adj_mat, node_fts=nfd1, edge_fts=nxe1, graph_fts=g1.graph_fts)
        # gd2 = GraphFeatures(adj_mat=g2.adj_mat, node_fts=nfd2, edge_fts=nxe2, graph_fts=g2.graph_fts)

        # preds1, raw_preds1 = model.decoder.decoders[Stage.OUTPUT]['s'].decode(gd1, num_nodes=None)
        # preds2, raw_preds2 = model.decoder.decoders[Stage.OUTPUT]['s'].decode(gd2, num_nodes=n2)

        # # assert (preds2['output']['s'][:, :4, :4, :4] == preds1['output']['s']).all()
        # assert (raw_preds2[:, :4, :4, :4] - raw_preds1).abs().max() < 1e-6
        # assert (raw_preds2[:, 4:, :4, :4] == 0.0).all()
        # assert (raw_preds2[:, :4, 4:, :4] == 0.0).all()
        # assert (raw_preds2[:, :4, :4, 4:] == 0.0).all()
        # assert (raw_preds2[:, 4:, 4:, :4] == 0.0).all()
        # assert (raw_preds2[:, 4:, :4, 4:] == 0.0).all()
        # assert (raw_preds2[:, :4, 4:, 4:] == 0.0).all()
        # assert (raw_preds2[:, 4:, 4:, 4:] == 0.0).all()
        
        # # pd1, rd1 = model.decoder(gd1)
        # # pd2, rd2 = model.decoder(gd2)

        # # assert (pd2['output']['s'][:, :4, :4, :4] == pd1['output']['s']).all()        
        # doSomething = 1


    # import ipdb; ipdb.set_trace()

    
    # p1, l1, e1 = model(b1)
    # print("///"*100)
    # p2, l2, e2 = model(b2)
    # import ipdb; ipdb.set_trace()
    # for batch in dl1:
    #     predictions, losses, evaluations  = model(batch)
    #     # evaluations = model.evaluate(predictions, batch)
    #     import pdb; pdb.set_trace()
    #     # print(batch)
