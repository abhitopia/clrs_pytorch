import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from torch import nn
from torch import Tensor
import torch
import itertools

_BIAS_VALUE = 0.0


POS_INF = 1e9
NEG_INF = -1e9


def set_bias_value(val):
    global _BIAS_VALUE
    _BIAS_VALUE = val

def batch_mask(valid_count: Tensor, max_count: int, trailing_dims: int = 2) -> Tensor:

    assert valid_count.ndim == 1, "num_nodes must be a 1D tensor"

    if trailing_dims <= 0:
        raise ValueError("trailing_dims must be a positive integer.")
    
    # Create a tensor of shape (max_count,) with values 0 to max_count-1
    arange_nodes = torch.arange(max_count, device=valid_count.device)
    
    # Reshape num_nodes to (batch_size, 1, ..., 1) with num_dims ones
    valid_count = valid_count.view(-1, *([1] * trailing_dims))
    
    # Create a list of views for each dimension
    # For num_dims=2, this creates [(1, N, 1), (1, 1, N)]
    # For num_dims=3, this creates [(1, N, 1, 1), (1, 1, N, 1), (1, 1, 1, N)]
    views = [arange_nodes.view(1, *([1] * i), max_count, *([1] * (trailing_dims - i - 1))) 
             for i in range(trailing_dims)]
    
    # Compare each view with num_nodes and combine using AND
    # We don't need to stack since broadcasting will handle the different shapes
    mask = views[0] < valid_count
    for view in views[1:]:
        mask = mask & (view < valid_count)
    
    return mask

def expand(mask: Tensor, target: Tensor, prior_dims: int = 0) -> Tensor:
    unsqueezed_shape = (1,) * prior_dims + mask.shape + (1,) * (target.ndim - mask.ndim - prior_dims)
    return mask.view(unsqueezed_shape).expand(target.shape)

def default_linear_init(weight: nn.Parameter, bias: Optional[nn.Parameter]) -> None:
    """
    Dynamically choose between LeCun and Xavier Initialisation
    """

    assert weight.ndim == 2, "Weight must be 2D"
    assert bias is None or bias.ndim == 1, "Bias must be 1D"

    fan_in, fan_out = weight.shape[1], weight.shape[0]

    # Dynamically Choose Initialisation
    if fan_in == 1:  # Xavier Initialisation
        std = 1.0 / math.sqrt(fan_out)
    else:            # LeCun Initialisation
        std = 1.0 / math.sqrt(fan_in)

    mean = 0.0
    a_val = mean - 2.0 * std
    b_val = mean + 2.0 * std
    nn.init.trunc_normal_(weight, mean=mean, std=std, a=a_val, b=b_val)

    # Bias Initialization (Haiku's default)
    if bias is not None:
        # nn.init.zeros_(bias)
        # print(f"Setting bias to {_BIAS_VALUE}")
        nn.init.constant_(bias, _BIAS_VALUE)

class Linear(nn.Linear):
    """
    Allows for custom weight and bias initializers.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 initializer: Optional[Callable[[nn.Parameter, Optional[nn.Parameter]], None]] = default_linear_init,
                 device=None,
                 dtype=None) -> None:
        """
        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias to the output. Default: True.
            initializer: Optional custom initializer function for the weight and bias matrix.
                    It should take tensors and modify them in-place.
            device: The desired device of the parameters and buffers of this module.
            dtype: The desired floating point type of the parameters and buffers of this module.
        """
        self._initizer = initializer
        # Call super().__init__ last, as it will call reset_parameters()
        super().__init__(in_features, out_features, bias, device, dtype)

    def reset_parameters(self) -> None:
        if self._initizer is not None:
            self._initizer(self.weight, self.bias)
        else:
            super().reset_parameters()

@torch.compiler.disable()
def log_sinkhorn(x: Tensor, steps: int, temperature: float, zero_diagonal: bool, add_noise: bool = True, num_nodes: Optional[Tensor] = None) -> Tensor:
    """Sinkhorn operator in log space, to postprocess permutation pointer logits.

    Args:
        x: input of shape [..., n, n], a batch of square matrices.
        steps: number of iterations.
        temperature: temperature parameter (as temperature approaches zero, the
        output approaches a permutation matrix).
        zero_diagonal: whether to force the diagonal logits towards -inf.
        add_noise: whether to add Gumbel noise.

    Returns:
        Elementwise logarithm of a doubly-stochastic matrix (a matrix with
        non-negative elements whose rows and columns sum to 1).
    """
    assert x.ndim >= 2
    assert x.shape[-1] == x.shape[-2]

    org_dtype = x.dtype

    NEG_INF_LOCAL = -1e9

    if add_noise:
        # Add standard Gumbel noise (see https://arxiv.org/abs/1802.08665)
        noise = -torch.log(-torch.log(torch.rand_like(x) + 1e-12) + 1e-12).to(org_dtype)
        x = x + noise

    x = x / temperature    # Can't do in-place or pytorch gods will scream!

    if zero_diagonal:
        eye = torch.eye(x.shape[-1], device=x.device, dtype=torch.bool)
        x = x.masked_fill(eye.unsqueeze(0), NEG_INF_LOCAL)

    if num_nodes is not None:
        edge_mask = expand(batch_mask(num_nodes, x.size(-1), 2), x)

    for _ in range(steps):
        x = torch.log_softmax(x, dim=-1).to(org_dtype)
        if num_nodes is not None:
            x = x.masked_fill(~edge_mask, NEG_INF_LOCAL)
    
        x = torch.log_softmax(x, dim=-2).to(org_dtype)
        if num_nodes is not None:
            x = x.masked_fill(~edge_mask, NEG_INF_LOCAL)

    return x

def np_one_hot(labels: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """One-hot encode a set of labels.
    
    Args:
        labels: A numpy array of shape (d1, d2, ..., dn) where each element is an integer label.
        num_classes: The number of classes in the dataset. If not provided, the maximum label value will be used.
    """
    if num_classes is None:
        num_classes = np.max(labels) + 1
    one_hot_encoded = np.eye(num_classes)[labels]

    return one_hot_encoded

NestedTensors = Union[torch.Tensor, Dict[Any, Any], List[Any], Tuple[Any, ...]]

def tree_map(fn: Callable[[torch.Tensor], Any], struct: NestedTensors) -> NestedTensors:
    if isinstance(struct, torch.Tensor):
        return fn(struct)
    elif isinstance(struct, dict):
        return {k: tree_map(fn, v) for k, v in struct.items()}
    elif isinstance(struct, list):
        return [tree_map(fn, v) for v in struct]
    elif isinstance(struct, tuple):
        return tuple(tree_map(fn, v) for v in struct)
    else:
        raise ValueError(f"Unsupported type: {type(struct)}")

def tree_flatten(struct: NestedTensors) -> List[torch.Tensor]:
    flat_list = []
    tree_map(lambda x: flat_list.append(x), struct)
    return flat_list

def tree_map_list(fn: Callable[[List[torch.Tensor]], torch.Tensor], structs: Sequence[NestedTensors]) -> NestedTensors:
    """
    Given a list of identical nested-structures `structs`,
    apply `fn` to the list of corresponding leaf-tensors and
    rebuild the same nesting with fn‘s outputs.

    fn: a function that takes a List[Tensor] and returns a Tensor
    structs: a sequence of pytrees all having the same structure
    """
    # grab the first structure to inspect its type
    first = structs[0]

    if isinstance(first, torch.Tensor):
        # collect that leaf from all structs
        leafs = [s for s in structs]           # List[Tensor]
        return fn(leafs)

    elif isinstance(first, dict):
        # must have same keys in each dict
        return {
            k: tree_map_list(fn, [d[k] for d in structs])
            for k in first.keys()
        }
    elif isinstance(first, list):
        # must all have same length
        return [
            tree_map_list(fn, [lst[i] for lst in structs])
            for i in range(len(first))
        ]
    elif isinstance(first, tuple):
        return tuple(
            tree_map_list(fn, [tup[i] for tup in structs])
            for i in range(len(first))
        )

    else:
        raise ValueError(f"Unsupported type: {type(first)}")

def tree_sort(struct: NestedTensors) -> NestedTensors:
    """
    Return a copy of `struct` where all dicts have their keys in sorted order.
    Recurses into lists and tuples unchanged.
    """
    if isinstance(struct, dict):
        # sort the keys, recurse on each value
        return {k: tree_sort(struct[k]) for k in sorted(struct)}
    elif isinstance(struct, list):
        return [tree_sort(v) for v in struct]
    elif isinstance(struct, tuple):
        return tuple(tree_sort(v) for v in struct)
    elif isinstance(struct, torch.Tensor):
        return struct
    else:
        raise ValueError(f"Unsupported type: {type(struct)}")

def tree_binary_op(fn: Callable[[Any, Any], Any], a: NestedTensors, b: NestedTensors) -> NestedTensors:
    """
    Recurse through a and b (which must have the same structure),
    applying `fn(a_leaf, b_leaf)` at each Tensor leaf and
    preserving the outer dict/list/tuple structure.
    """
    # 1) Tensors
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return fn(a, b)
    # 2) Dicts
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            raise ValueError(f"Dict keys differ: {set(a)} vs {set(b)}")
        # keep insertion order / sort if you like
        return {k: tree_binary_op(fn, a[k], b[k]) for k in a.keys()}

    # 3) Lists
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError(f"List lengths differ: {len(a)} vs {len(b)}")
        return [tree_binary_op(fn, x, y) for x, y in zip(a, b)]

    # 4) Tuples
    elif isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            raise ValueError(f"Tuple lengths differ: {len(a)} vs {len(b)}")
        return tuple(tree_binary_op(fn, x, y) for x, y in zip(a, b))
    else:
        raise ValueError(f"Unsupported type: {type(a)}")

def tree_equal(a: NestedTensors, b: NestedTensors, *, atol=0., rtol=0.) -> bool:
    """
    Returns the nested-structure of booleans or small tensors
    indicating leafwise comparison results.
    """
    def cmp(x: torch.Tensor, y: torch.Tensor):
        if atol == 0 and rtol == 0:
            # returns a single bool
            return torch.tensor(torch.equal(x, y))
        else:
            # returns a single bool
            return torch.tensor(torch.allclose(x, y, atol=atol, rtol=rtol))

    bool_tree = tree_binary_op(cmp, a, b)
    flat_bool_tree = tree_flatten(bool_tree)
    return torch.stack(flat_bool_tree).all()


if __name__ == "__main__":
    labels = np.array([[1, 0, 3, 2, 4], [1, 0, 0, 1, 2]])
    one_hot_encoded = np_one_hot(labels)
    print(one_hot_encoded, one_hot_encoded.shape)

