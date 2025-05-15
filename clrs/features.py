from typing import List, NamedTuple, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from .probing import DataPoint, ProbesDict
from .specs import Type, Stage, Spec
from .utils import np_one_hot

def extract_spec(dps: List[DataPoint], stage: Stage) -> Spec:
    return {dp.name: (stage, dp.location, dp.type_) for dp in dps}

def to_tensor(dp: DataPoint, device: torch.device=torch.device('cpu')) -> DataPoint:
    data = torch.from_numpy(dp.data) if isinstance(dp.data, np.ndarray) else dp.data
    data = data.float().to(device)
    return dp._replace(data=data)

class Features(NamedTuple):
    inputs: List[DataPoint]
    hints: List[DataPoint]
    lengths: List[int]

    @property
    def spec(self):
        spec = extract_spec(self.inputs, Stage.INPUT)
        spec.update(extract_spec(self.hints, Stage.HINT))
        return spec

    def randomize_pos(self, rng: np.random.RandomState):
        # Find the original pos DataPoint
        try:
            pos_dp = next(x for x in self.inputs if x.name == 'pos')
            original_inputs = self.inputs
        except StopIteration:
            # Handle case where 'pos' input doesn't exist, maybe return self?
            print("Warning: 'pos' input not found. Returning original Features.")
            return self

        batch_size, num_nodes = pos_dp.data.shape
        unsorted = rng.uniform(size=(batch_size, num_nodes))
        new_pos_data_list = []
        for i in range(batch_size):  # we check one example at a time.
            # Find splits in the original pos data
            split, = np.where(pos_dp.data[i] == 0)
            split = np.concatenate([split, [num_nodes]])

            # Construct the new randomized pos data for this example
            new_pos_data_list.append(
                np.concatenate([np.sort(unsorted[i, split[j]:split[j+1]])
                                for j in range(len(split) - 1)])
            )
        
        # Create the new pos DataPoint with the randomized data
        new_pos_dp = DataPoint(
            name=pos_dp.name,
            location=pos_dp.location,
            type_=pos_dp.type_,
            data=np.array(new_pos_data_list) # Combine batch results
        )

        # Create the new list of input DataPoints
        new_inputs = [new_pos_dp if x.name == 'pos' else x for x in original_inputs]

        # Use _replace to create a new Features instance with the updated inputs
        return self._replace(inputs=new_inputs)

    def move_hint_to_input(self, hint_name: str = 'pred_h'):
        # Find the hint DataPoint
        try:
            hint = next(h for h in self.hints if h.name == hint_name)
        except StopIteration:
            print(f"Warning: Hint '{hint_name}' not found. Returning original Features.")
            return self

        # Create the new list of hints excluding the moved one
        new_hints = [h for h in self.hints if h.name != hint_name]
        assert isinstance(self.lengths, list), "Lengths must be a list"
        batch_size = hint.data.shape[1] # Assuming batch dim is 1
        assert len(self.lengths) == batch_size, "Length mismatch between features.lengths and hint batch size"
        for i in range(batch_size):
            # Check consistency only up to the actual length for this batch item
            length_i = int(self.lengths[i])
            if length_i > 1: # Only check if there's more than one step
                delta = hint.data[1:length_i, i] - hint.data[0, i]
                assert np.sum(np.abs(delta)) < 1e-6, \
                      f"Hint '{hint_name}' is not constant over time for batch item {i}"

        # Create the new 'pred' DataPoint from the first time step
        new_pred_dp = DataPoint(name='pred',
                                location=hint.location,
                                type_=hint.type_,
                                data=hint.data[0]) # Use data from the first time step

        # Create the new list of input DataPoints
        new_inputs = self.inputs + [new_pred_dp] # Append the new DataPoint

        # Use _replace to create a new Features instance with updated inputs and hints
        # Need to provide all fields for _replace when modifying multiple
        return self._replace(inputs=new_inputs, hints=new_hints, lengths=self.lengths)

    def process_permutations(self, enforce_permutations: bool):
        """Processes SHOULD_BE_PERMUTATION probes in inputs, hints.
        """
        processed_inputs = _preprocess_permutations_internal(
            list(self.inputs), enforce_permutations
        )
        processed_hints = _preprocess_permutations_internal(
            list(self.hints), enforce_permutations
        )
        return self._replace(inputs=processed_inputs, hints=processed_hints, lengths=self.lengths)

    @property
    def batch_size(self) -> int:
        return self.inputs[0].data.shape[0]
    
    @property
    def num_nodes(self) -> int:
        return self.inputs[0].data.shape[1]

    def to_tensor(self, device: torch.device=torch.device('cpu')) -> "Features":
        return Features(inputs=[to_tensor(dp, device) for dp in self.inputs],
                        hints=[to_tensor(dp, device) for dp in self.hints],
                        lengths=torch.tensor(self.lengths, device=device))
    
    def at_step(self, step: int) -> "Features":
        assert step >= 0
        if step == 0:
            return self._replace(inputs=self.inputs, hints=[], lengths=self.lengths)
        hints = [dp._replace(data=dp.data[step-1]) for dp in self.hints]
        return self._replace(inputs=self.inputs, hints=hints, lengths=self.lengths)

    @property
    def max_steps(self):
        return self.hints[0].data.shape[0]

class Feedback(NamedTuple):
    features: Features
    outputs: List[DataPoint]
    algo: Optional[str] = None

    def __len__(self):
        return len(self.outputs)
    
    @property
    def spec(self):
        spec = self.features.spec
        spec.update(extract_spec(self.outputs, Stage.OUTPUT))
        return spec
    
    @property
    def batch_size(self):
        return self.features.batch_size
    
    @property
    def num_nodes(self):
        return self.features.num_nodes

    @classmethod
    def from_probes(cls, probes: Union[List[ProbesDict], ProbesDict], algo_name: Optional[str] = None, min_hint_steps: int = 0):
        if isinstance(probes, ProbesDict):
            probes = [probes]
        inputs, outputs, hints = [], [], []
        for probe in probes:
            inputs_, outputs_, hints_ = probe.split_stages()
            inputs.append(inputs_)
            outputs.append(outputs_)
            hints.append(hints_)

        inputs, _ = batch_datapoints(inputs, is_hints=False)
        outputs, _ = batch_datapoints(outputs, is_hints=False)
        hints, lengths = batch_datapoints(hints, is_hints=True, min_hint_steps=min_hint_steps)

        features = Features(inputs=inputs, hints=hints, lengths=lengths)
        return cls(features=features, outputs=outputs, algo=algo_name)
    
    @classmethod
    def batch_feedbacks(cls, feedbacks: List["Feedback"], min_hint_steps: int = 0):
        algo_name = feedbacks[0].algo
        inputs, outputs, hints = [], [], []
        for feedback in feedbacks:
            inputs.append(feedback.features.inputs)
            outputs.append(feedback.outputs)
            hints.append(feedback.features.hints)
        inputs, _ = batch_datapoints(inputs, is_hints=False)
        outputs, _ = batch_datapoints(outputs, is_hints=False)
        hints, lengths = batch_datapoints(hints, is_hints=True, min_hint_steps=min_hint_steps)
        features = Features(inputs=inputs, hints=hints, lengths=lengths)
        return cls(features=features, outputs=outputs, algo=algo_name)
    
    @property
    def max_steps(self):
        return self.features.max_steps
        
    def randomize_pos(self, rng: np.random.RandomState):
        """Randomize the pos data for each example in the batch."""
        features = self.features.randomize_pos(rng)
        return Feedback(features=features, outputs=self.outputs, algo=self.algo)
    
    def move_hint_to_input(self, hint_name: str = 'pred_h'):
        features = self.features.move_hint_to_input(hint_name)
        return Feedback(features=features, outputs=self.outputs, algo=self.algo)

    def process_permutations(self, enforce_permutations: bool):
        """Processes SHOULD_BE_PERMUTATION probes in inputs, hints, and outputs.

        Replaces probes of type SHOULD_BE_PERMUTATION based on the enforce_permutations flag.
        If True, replaces with PERMUTATION_POINTER and MASK_ONE.
        If False, replaces with POINTER.

        Args:
            enforce_permutations: Boolean flag to determine the replacement strategy.

        Returns:
            A new Feedback object with processed probes.
        """
        processed_outputs = _preprocess_permutations_internal(
            list(self.outputs), enforce_permutations
        )
        new_features = self.features.process_permutations(enforce_permutations)

        return Feedback(features=new_features, outputs=processed_outputs, algo=self.algo)

    def to_tensor(self, device: torch.device=torch.device('cpu')) -> "Feedback":
        outputs = [to_tensor(dp, device) for dp in self.outputs]
        return Feedback(features=self.features.to_tensor(device), outputs=outputs, algo=self.algo)
    
    def at_step(self, step: int) -> "Feedback":
        features = self.features.at_step(step)
        return Feedback(features=features, outputs=self.outputs, algo=self.algo)


def batch_datapoints(list_dps: List[List[DataPoint]], is_hints: bool = False, min_hint_steps: int = 0) -> List[DataPoint]:
  """Batches list of DataPoints along the batch axis using NumPy. When is_hints is True, the 
     time axis is also padded to the maximum steps across all DataPoints.
  """

  batching_axis = 1 if is_hints else 0
  time_axis = 0 if is_hints else None

  assert len(list_dps) > 0
  ref_io = list_dps[0]
  num_probes = len(ref_io)
  batch = []
  steps = None
  for i in range(num_probes):
    data_to_batch = []
    ref_dp = ref_io[i]
    for sample_io in list_dps:
      assert len(sample_io) == num_probes
      dp = sample_io[i]
      assert ref_dp.is_compatible_with(dp)
      data_to_batch.append(dp.data)
      assert dp.data.shape[batching_axis] == 1, "DataPoint must have batch dimension 1"
      
    if is_hints:
       # The time axis can be different for each data
       steps_probe_i = [d.shape[time_axis] for d in data_to_batch]
       if i == 0:
         steps = steps_probe_i
       else:
         assert steps == steps_probe_i, "All hints must have the same number of steps"
       max_steps = max(min_hint_steps, max(steps_probe_i))
       # ((before_axis_0, after_axis_0), (before_axis_1, after_axis_1), ...)
       # We only pad *after* axis 0. All other axes have (0, 0) padding.
       apply_pad = lambda d: np.pad(d, [(0, max_steps - d.shape[time_axis])] + [(0, 0)]*(d.ndim-1), mode='constant', constant_values=0)
       data_to_batch = [apply_pad(d) for d in data_to_batch]

    batched_data = np.concatenate(data_to_batch, axis=batching_axis)
    new_dp = DataPoint(
        name=ref_dp.name,
        location=ref_dp.location,
        type_=ref_dp.type_,
        data=batched_data
    )
    batch.append(new_dp)
  return batch, steps

# Helper function adapted from clrs._src.samplers
def _preprocess_permutations_internal(probes: List[DataPoint], enforce_permutations: bool) -> List[DataPoint]:
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
