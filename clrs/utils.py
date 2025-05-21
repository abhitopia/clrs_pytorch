import math
from typing import Any, Callable, List, Optional
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




def log_sinkhorn(x: Tensor, steps: int, temperature: float, zero_diagonal: bool, add_noise: bool = True, num_nodes: Optional[int] = None) -> Tensor:
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

    if add_noise:
        # Add standard Gumbel noise (see https://arxiv.org/abs/1802.08665)
        noise = -torch.log(-torch.log(torch.rand_like(x) + 1e-12) + 1e-12)
        x = x + noise

    x = x / temperature    # Can't do in-place or pytorch gods will scream!


    if zero_diagonal:
        eye = torch.eye(x.shape[-1], device=x.device, dtype=torch.bool)
        x = x.masked_fill(eye.unsqueeze(0), NEG_INF)
    if num_nodes is not None:
        edge_mask = expand(batch_mask(num_nodes, x.size(-1), 2), x)

    for _ in range(steps):
        x = torch.log_softmax(x, dim=-1)
        if num_nodes is not None:
            x = x.masked_fill(~edge_mask, NEG_INF)
    
        x = torch.log_softmax(x, dim=-2)
        if num_nodes is not None:
            x = x.masked_fill(~edge_mask, NEG_INF)

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


def all_lists_equal(list_of_lists: List[List[Any]]) -> bool:
    """
    Checks if all lists within a list of lists are equal.

    Args:
        list_of_lists: A list where each element is a list.

    Returns:
        True if all inner lists are equal or if the input list is empty 
        or contains only one list, False otherwise.
    """
    if not list_of_lists or len(list_of_lists) == 1:
        return True
    
    first_list = list_of_lists[0]
    for sub_list in list_of_lists[1:]:
        if sub_list != first_list:
            return False
    return True


def cycle_flatten_list_of_lists(list_of_lists: list[list]) -> list:
    """
    Flattens a list of lists by cycling through their elements.

    Given a list of lists where all sublists are of equal length, this function
    creates a single list by taking the first element of the first sublist,
    then the first element of the second sublist, and so on, followed by
    the second elements of each sublist, etc.

    Args:
        list_of_lists: A list of lists. All sublists must have the same length.

    Returns:
        A new list containing the elements from the sublists, cycled and flattened.
        The total number of elements will be len(list_of_lists) * len(sublist).

    Raises:
        TypeError: If any element in list_of_lists is not a list.
        ValueError: If sublists do not all have the same length.

    Example:
        >>> cycle_flatten_list_of_lists([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        [1, 4, 7, 2, 5, 8, 3, 6, 9]
        >>> cycle_flatten_list_of_lists([['a', 'b'], ['c', 'd']])
        ['a', 'c', 'b', 'd']
    """
    if not list_of_lists:
        return []

    if not all(isinstance(sublist, list) for sublist in list_of_lists):
        raise TypeError("All elements of the input list must be lists.")

    # Get length of the first sublist to use as a reference
    # This assumes list_of_lists is not empty, which is checked above.
    first_len = len(list_of_lists[0])

    if not all(len(sublist) == first_len for sublist in list_of_lists):
        raise ValueError("All sublists must have the same length.")

    if first_len == 0:  # All sublists are empty
        return []

    # Transpose using zip and then flatten
    return list(itertools.chain.from_iterable(zip(*list_of_lists)))




def tree_map(fn: Callable[[Tensor], Any], tree: Any) -> None:
    """
    Recursively applies a function `fn` to all Tensors found within a nested
    data structure `tree` which can consist of dictionaries, lists, and tuples.

    This function does NOT return a new data structure. It traverses the
    input `tree` and applies `fn` to each Tensor encountered. If `fn`
    modifies tensors in-place, the original `tree` will be modified.
    The return value of `fn` is ignored by `tree_map`.

    Args:
        fn: A function that takes a `torch.Tensor` as input. Its return
            value is disregarded.
        tree: The nested data structure (e.g., dict, list, tuple) to traverse.
              It can contain Tensors or other nested structures.
    """
    if isinstance(tree, dict):
        for key in tree:
            tree_map(fn, tree[key])  # Recurse on values
    elif isinstance(tree, list) or isinstance(tree, tuple):
        # For lists, if fn modifies tensors in-place, the list's elements are changed.
        # For tuples, elements can be modified if they are mutable (like tensors that fn changes in-place).
        for item in tree:
            tree_map(fn, item)
    elif isinstance(tree, Tensor):
        fn(tree)  # Apply fn. Its return value is ignored.
    # If tree is of any other type, it's ignored, and the function implicitly returns None.


if __name__ == "__main__":
    labels = np.array([[1, 0, 3, 2, 4], [1, 0, 0, 1, 2]])
    one_hot_encoded = np_one_hot(labels)
    print(one_hot_encoded, one_hot_encoded.shape)

