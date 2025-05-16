from typing import Any, Dict, ItemsView, Iterator, KeysView, List, NamedTuple, Optional, Tuple, Union, ValuesView
import numpy as np
import torch
from torch import Tensor
from .specs import Location, Stage, Type, OutputClass, Trajectory, Spec, Feature
from .utils import np_one_hot

Array = np.ndarray 
Data = Union[Array, List[Array]]
DataOrType = Union[Data, Type]
ProbeKey = str
NextProbe = Dict[ProbeKey, Data]
ProbeValue = Dict[str, DataOrType]
Probe = Dict[ProbeKey, ProbeValue]
StageProbes = Dict[Location, Probe]
ProbesDictType = Dict[Stage, StageProbes]

# Tell attrs not to auto-generate repr
class DataPoint(NamedTuple):
  """Describes a data point."""
  name: ProbeKey
  location: Location
  type_: Type
  data: Data

  def __repr__(self) -> str:
    # Add a marker here
    return f"DataPoint(name={self.name}, location={self.location}, type_={self.type_}, data={self.data.shape})"

  def is_compatible_with(self, other: 'DataPoint') -> bool:
    """Checks if this DataPoint is compatible with another for batching.

    Compatibility means having the same name, location, and type.
    The actual data content is not compared.
    """
    if not isinstance(other, DataPoint):
        return False
    return (self.name == other.name and
            self.location == other.location and
            self.type_ == other.type_)
  
  def __ne__(self, other: 'DataPoint') -> bool:
    return not self.__eq__(other)
  
  def __eq__(self, other: 'DataPoint') -> bool:
    if not isinstance(other, DataPoint):
        return False
    
    data_close = False
    if isinstance(self.data, Tensor) and isinstance(other.data, Tensor):
      data_close = torch.allclose(self.data, other.data)
    elif isinstance(self.data, Array) and isinstance(other.data, Array):
      if self.data.shape != other.data.shape:
         data_close = False
      else:
         data_close = np.allclose(self.data, other.data)

    return (self.name == other.name and
            self.location == other.location and
            self.type_ == other.type_ and 
            data_close)
            

class ProbeError(Exception):
    """Custom exception for errors during probe operations."""
    pass


class ProbesDict:
    """
    Manages a collection of probes, organized by stage and location,
    corresponding to a given specification. Provides dictionary-like access.
    """

    def __init__(self, spec: Spec):
        """Initializes the ProbesDict structure based on the provided spec."""
        self._probes: ProbesDictType = {}
        self._finalized: bool = False
        self.spec: Spec = spec

        for stage in Stage:
            self._probes[stage] = {}
            for loc in Location:
                self._probes[stage][loc] = {}

        for name, (stage, loc, type_) in spec.items():
            self._probes[stage][loc][name] = {'data': [], 'type_': type_}

    def push(self, stage: Stage, next_probe: NextProbe):
        """Pushes a probe into an existing `ProbesDict`."""
        for loc in Location:
            for name in self._probes[stage][loc]:
                if name not in next_probe:
                    raise ProbeError(f'Missing probe for {name}.')
                if isinstance(self._probes[stage][loc][name]['data'], Array):
                    raise ProbeError('Attemping to push to finalized `ProbesDict`.')
                self._probes[stage][loc][name]['data'].append(next_probe[name])

    def finalize(self):
        """Finalizes the ProbesDict by converting list data into NumPy arrays."""
        if self._finalized:
            raise ProbeError('Attempting to re-finalize a finalized ProbesDict.')

        for stage in Stage:
            for loc in Location:
                for name, probe_info in self._probes[stage][loc].items():
                    current_data = probe_info['data']
                    if isinstance(current_data, Array):
                        raise ProbeError('Attemping to re-finalize a finalized `ProbesDict`.')
                    if stage == Stage.HINT:
                        probe_info['data'] = np.stack(current_data)
                    else:
                        probe_info['data'] = np.squeeze(np.array(current_data), axis=0)
        self._finalized = True

    def __getitem__(self, stage: Stage) -> StageProbes:
        """
        Allows dictionary-style access by stage (e.g., probes[Stage.INPUT]).
        Returns the inner dictionary for that stage, allowing further nested access.
        """
        return self._probes[stage]

    def __contains__(self, stage: Stage) -> bool:
        """Allows checking for stage existence using 'in' (e.g., Stage.INPUT in probes)."""
        return stage in self._probes

    def __len__(self) -> int:
        """Returns the number of top-level stages."""
        return len(self._probes)

    def keys(self) -> KeysView[Stage]:
        """Returns a view of the top-level stages (keys)."""
        return self._probes.keys()

    def values(self) -> ValuesView[StageProbes]:
        """Returns a view of the inner stage dictionaries (values)."""
        return self._probes.values()

    def items(self) -> ItemsView[Stage, StageProbes]:
        """Returns a view of the top-level stage-dictionary pairs (items)."""
        return self._probes.items()

    def __iter__(self) -> Iterator[Stage]:
        """Allows iteration over the top-level stages (keys)."""
        return iter(self._probes)
    
    @property
    def finalized(self) -> bool:
        """Returns True if the ProbesDict has been finalized."""
        return self._finalized

    def __repr__(self) -> str:
        state = "finalized" if self._finalized else "collecting"
        structure_summary = {
            s: {l: list(p.keys()) for l, p in locs.items()}
            for s, locs in self._probes.items()
        }
        return f"<ProbesDict ({state}) Spec Keys: {list(self.spec.keys())} Structure: {structure_summary}>"


    def split_stages(self) -> Tuple[List[DataPoint], List[DataPoint], List[DataPoint]]:
        """Splits the ProbesDict into input, output, and hint probes."""
        assert self.finalized, "ProbesDict must be finalized before splitting into stages."
        inputs = []
        outputs = []
        hints = []

        for name in self.spec:
            stage, loc, t = self.spec[name]

            if stage not in self._probes:
                raise ProbeError(f'Missing stage {stage}.')
            if loc not in self._probes[stage]:
                raise ProbeError(f'Missing location {loc}.')
            if name not in self._probes[stage][loc]:
                raise ProbeError(f'Missing probe {name}.')
            if 'type_' not in self._probes[stage][loc][name]:
                raise ProbeError(f'Probe {name} missing attribute `type_`.')
            if 'data' not in self._probes[stage][loc][name]:
                raise ProbeError(f'Probe {name} missing attribute `data`.')
            if t != self._probes[stage][loc][name]['type_']:
                raise ProbeError(f'Probe {name} of incorrect type {t}.')

            data = self._probes[stage][loc][name]['data']
            if not isinstance(data, Array):
              raise ProbeError((f'Invalid `data` for probe "{name}". ' +
                          'Did you forget to call `probing.finalize`?'))

            if t in [Type.MASK, Type.MASK_ONE, Type.CATEGORICAL]:
                # pytype: disable=attribute-error
                if not ((data == 0) | (data == 1) | (data == -1)).all():
                    raise ProbeError(f'0|1|-1 `data` for probe "{name}"')
                # pytype: enable=attribute-error
            if t in [Type.MASK_ONE, Type.CATEGORICAL] and not np.all(np.sum(np.abs(data), -1) == 1):
                raise ProbeError(f'Expected one-hot `data` for probe "{name}"')

            if t == Type.POINTER:
              # Added this so all pointers are one-hot encoded
              data = np_one_hot(data.astype(int), data.shape[-1])

            dim_to_expand = 1 if stage == Stage.HINT else 0
            data_point = DataPoint(name=name, location=loc, type_=t,
                           data=np.expand_dims(data, dim_to_expand))

            if stage == Stage.INPUT:
              inputs.append(data_point)
            elif stage == Stage.OUTPUT:
              outputs.append(data_point)
            else:
              hints.append(data_point)
        

        inputs = sorted(inputs, key=lambda x: x.name)
        outputs = sorted(outputs, key=lambda x: x.name)
        hints = sorted(hints, key=lambda x: x.name)
        return inputs, outputs, hints
    
    def to_feature(self, 
                      randomize_pos: bool = False,
                      move_predh_to_input: bool = False, 
                      enforce_permutations: bool = False,
                      rng: Optional[np.random.RandomState] = None) -> Tuple[Feature, Spec]:
        inputs, outputs, hints = self.split_stages()
        num_steps_hint = hints[0].data.shape[0]
        trajectory: Trajectory = {}
        spec: Spec = {}

        if move_predh_to_input:
            pred_h = hints.pop(hints.index(next(x for x in hints if x.name == 'pred_h')))
            assert pred_h.data.shape[1] == 1, "batch size of pred_h must be 1"
            assert np.sum(np.abs(pred_h.data - pred_h.data[0])) == 0.0, "pred_h must be same across time steps"
            pred_h_data = pred_h.data[0]
            inputs.append(DataPoint(name='pred_h', location=pred_h.location, type_=pred_h.type_, data=pred_h_data))

        for features, stage in zip((inputs, outputs, hints), (Stage.INPUT, Stage.OUTPUT, Stage.HINT)):
            trajectory[stage] = {}       # stage -> (name -> array) 
            features = preprocess_pointer_or_permutations(features, enforce_permutations)
            for dp in features:                   
                trajectory[stage][dp.name] = dp.data
                if dp.name == 'pos' and randomize_pos:
                    trajectory[stage][dp.name] = randomize_pos_data(dp.data, rng)
                meta = {}
                if dp.type_ == Type.CATEGORICAL:
                   meta = {'num_classes': dp.data.shape[-1]}
                elif stage == Stage.INPUT and dp.name == 'pos':
                   num_nodes = dp.data.shape[1]
                spec[dp.name] = (stage, dp.location, dp.type_, meta)

        feature = (trajectory, num_steps_hint, num_nodes)
        
        return feature, spec


def randomize_pos_data(data: Array, rng: Optional[np.random.RandomState] = None) -> Array:
    if rng is None:
       rng = np.random.RandomState()

    batch_size, num_nodes = data.shape
    unsorted = rng.uniform(size=(batch_size, num_nodes))
    new_pos_data_list = []
    for i in range(batch_size):  # we check one example at a time.
        # Find splits in the original pos data
        split, = np.where(data[i] == 0)
        split = np.concatenate([split, [num_nodes]])

        # Construct the new randomized pos data for this example
        new_pos_data_list.append(
            np.concatenate([np.sort(unsorted[i, split[j]:split[j+1]])
                            for j in range(len(split) - 1)])
        )
    
    return np.array(new_pos_data_list) # Combine batch results
    
# Helper function adapted from clrs._src.samplers
def preprocess_pointer_or_permutations(probes: List[DataPoint], enforce_permutations: bool) -> List[DataPoint]:
    """Replace should-be permutations with proper permutation pointer + mask."""
    output = []
    for x in probes:
        if x.type_ != Type.POINTER_OR_PERMUTATION_WITH_MASK:
            output.append(x)
            continue
        # Node location is assumed for SHOULD_BE_PERMUTATION based on original code
        # assert x.location == specs.Location.NODE 
        if enforce_permutations:
            new_x, mask = predecessor_to_cyclic_predecessor_and_first(x.data)
            output.append(
                DataPoint(
                    name=x.name,
                    location=x.location, # Assumes location is present on DataPoint
                    type_=Type.PERMUTATION_POINTER,
                    data=new_x))
            output.append(
                DataPoint(
                    name=x.name + '_mask',
                    location=x.location, # Assumes location is present on DataPoint
                    type_=Type.MASK_ONE,
                    data=mask))
        else:
            data = np_one_hot(x.data, x.data.shape[-1])
            output.append(DataPoint(name=x.name, location=x.location, # Assumes location is present on DataPoint
                                          type_=Type.POINTER, data=data))
    return output

def predecessor_to_cyclic_predecessor_and_first(
    pointers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converts predecessor pointers to cyclic predecessor + first node mask (NumPy).

    This function assumes that the pointers represent potentially multiple batches
    of linear orders of nodes (akin to linked lists), where each node points to
    its predecessor and the first node points to itself. It returns the same
    pointers in one-hot format, except that the first node points to the last,
    and a mask_one marking the first node for each item in the batch.

    Example (batch size 1):
    ```
    pointers = [[2, 1, 1]] # B=1, N=3
    P, M = predecessor_to_cyclic_predecessor_and_first(np.array(pointers))
    P is [[[0., 0., 1.],  # Node 0 still points to 2
          [1., 0., 0.],  # Node 1 now points to 0 (last)
          [0., 1., 0.]]] # Node 2 still points to 1
    M is [[0., 1., 0.]]   # Node 1 is the first node
    ```

    Args:
      pointers: array of shape [B, N] containing predecessor indices.
                B is batch dim, N is number of nodes.
                Assumes pointers[b, i] is the predecessor of node i in batch b.

    Returns:
      Tuple of (permutation_pointers, first_node_mask):
        - permutation_pointers: np.ndarray of shape [B, N, N] (one-hot representation).
        - first_node_mask: np.ndarray of shape [B, N] (one-hot representation).
    """
    # Ensure input is at least 2D (add batch dim if needed)
    if pointers.ndim == 1:
        pointers = pointers[np.newaxis, :]

    batch_size, nb_nodes = pointers.shape

    # Create one-hot representation: (B, N, N)
    # pointers_one_hot[b, i, j] = 1 iff pointers[b, i] == j
    pointers_one_hot = np.eye(nb_nodes, dtype=pointers.dtype)[pointers]

    # Find the index of the last node for each batch item: node that no other node points to.
    # Sum incoming pointers for each node: sum over axis 1 (the source node index i)
    incoming_counts = np.sum(pointers_one_hot, axis=1) # Shape (B, N)
    last = np.argmin(incoming_counts, axis=1) # Shape (B,)

    # Find the first node for each batch item: the one pointing to itself.
    # Get diagonal elements for each batch item: pointers_one_hot[b, i, i]
    self_pointers = np.diagonal(pointers_one_hot, axis1=1, axis2=2) # Shape (B, N)
    first = np.argmax(self_pointers, axis=1) # Shape (B,)

    # Create one-hot mask for the first node: (B, N)
    mask = np.eye(nb_nodes, dtype=pointers.dtype)[first]

    # Create one-hot representation for the last node: (B, N)
    last_one_hot = np.eye(nb_nodes, dtype=pointers.dtype)[last]

    # Modify pointers_one_hot: make the first node point to the last node
    # Add connection: first -> last using broadcasting
    # mask[..., None] -> (B, N, 1); last_one_hot[:, None, :] -> (B, 1, N)
    pointers_one_hot += mask[..., None] * last_one_hot[:, None, :]

    # Remove self-loop: first -> first using broadcasting
    # mask[..., None] -> (B, N, 1); mask[:, None, :] -> (B, 1, N)
    pointers_one_hot -= mask[..., None] * mask[:, None, :]

    # Return shapes are (B, N, N) and (B, N)
    return pointers_one_hot, mask


def array(A_pos: np.ndarray) -> np.ndarray:
  """Constructs an `array` probe."""
  probe = np.arange(A_pos.shape[0])
  for i in range(1, A_pos.shape[0]):
    probe[A_pos[i]] = A_pos[i - 1]
  return probe


def array_cat(A: np.ndarray, n: int) -> np.ndarray:
  """Constructs an `array_cat` probe."""
  assert n > 0
  probe = np.zeros((A.shape[0], n))
  for i in range(A.shape[0]):
    probe[i, A[i]] = 1
  return probe


def heap(A_pos: np.ndarray, heap_size: int) -> np.ndarray:
  """Constructs a `heap` probe."""
  assert heap_size > 0
  probe = np.arange(A_pos.shape[0])
  for i in range(1, heap_size):
    probe[A_pos[i]] = A_pos[(i - 1) // 2]
  return probe


def graph(A: np.ndarray) -> np.ndarray:
  """Constructs a `graph` probe."""
  probe = (A != 0) * 1.0
  probe = ((A + np.eye(A.shape[0])) != 0) * 1.0
  return probe


def mask_one(i: int, n: int) -> np.ndarray:
  """Constructs a `mask_one` probe."""
  assert n > i
  probe = np.zeros(n)
  probe[i] = 1
  return probe


def strings_id(T_pos: np.ndarray, P_pos: np.ndarray) -> np.ndarray:
  """Constructs a `strings_id` probe."""
  probe_T = np.zeros(T_pos.shape[0])
  probe_P = np.ones(P_pos.shape[0])
  return np.concatenate([probe_T, probe_P])


def strings_pair(pair_probe: np.ndarray) -> np.ndarray:
  """Constructs a `strings_pair` probe."""
  n = pair_probe.shape[0]
  m = pair_probe.shape[1]
  probe_ret = np.zeros((n + m, n + m))
  for i in range(0, n):
    for j in range(0, m):
      probe_ret[i, j + n] = pair_probe[i, j]
  return probe_ret


def strings_pair_cat(pair_probe: np.ndarray, nb_classes: int) -> np.ndarray:
  """Constructs a `strings_pair_cat` probe."""
  assert nb_classes > 0
  n = pair_probe.shape[0]
  m = pair_probe.shape[1]

  # Add an extra class for 'this cell left blank.'
  probe_ret = np.zeros((n + m, n + m, nb_classes + 1))
  for i in range(0, n):
    for j in range(0, m):
      probe_ret[i, j + n, int(pair_probe[i, j])] = OutputClass.POSITIVE

  # Fill the blank cells.
  for i_1 in range(0, n):
    for i_2 in range(0, n):
      probe_ret[i_1, i_2, nb_classes] = OutputClass.MASKED
  for j_1 in range(0, m):
    for x in range(0, n + m):
      probe_ret[j_1 + n, x, nb_classes] = OutputClass.MASKED
  return probe_ret


def strings_pi(T_pos: np.ndarray, P_pos: np.ndarray,
               pi: np.ndarray) -> np.ndarray:
  """Constructs a `strings_pi` probe."""
  probe = np.arange(T_pos.shape[0] + P_pos.shape[0])
  for j in range(P_pos.shape[0]):
    probe[T_pos.shape[0] + P_pos[j]] = T_pos.shape[0] + pi[P_pos[j]]
  return probe


def strings_pos(T_pos: np.ndarray, P_pos: np.ndarray) -> np.ndarray:
  """Constructs a `strings_pos` probe."""
  probe_T = np.copy(T_pos) * 1.0 / T_pos.shape[0]
  probe_P = np.copy(P_pos) * 1.0 / P_pos.shape[0]
  return np.concatenate([probe_T, probe_P])


def strings_pred(T_pos: np.ndarray, P_pos: np.ndarray) -> np.ndarray:
  """Constructs a `strings_pred` probe."""
  probe = np.arange(T_pos.shape[0] + P_pos.shape[0])
  for i in range(1, T_pos.shape[0]):
    probe[T_pos[i]] = T_pos[i - 1]
  for j in range(1, P_pos.shape[0]):
    probe[T_pos.shape[0] + P_pos[j]] = T_pos.shape[0] + P_pos[j - 1]
  return probe



if __name__ == "__main__":
    from .specs import RAW_SPECS
    from rich import print
    spec = RAW_SPECS['binary_search']
    print(spec)
    probes = ProbesDict(spec)
    print(probes)
    # next_probe = {'pos': np.array([1, 2, 3]), 'key': np.array([4, 5, 6]), 'target': np.array([7, 8, 9])}
    # probes.push(Stage.INPUT, next_probe)
    # probes.finalize()
    # print(probes[Stage.INPUT])