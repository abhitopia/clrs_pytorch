import math
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from .specs import AlgorithmEnum, Feature, Spec, Stage, NumNodes, NumSteps, Type
from .algorithm import Algorithm
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm


Batch = Tuple[List[int], int, bool]
IsLast = bool
IsFirst = bool
FeatureBatch = Tuple[Feature, IsFirst, IsLast]


def load_algo_features(algo: Algorithm, num_samples: int, cache_dir: Union[str, Path] = None) -> List[Feature]:
    if cache_dir is not None:
        cache_dir = Path(cache_dir) / algo.name
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        cache_file = cache_dir / f"{algo.unique_hash()}-{num_samples}.pkl"
        if cache_file.exists():
            return pickle.load(cache_file.open("rb"))
        
    algo.reset() # Reset the algorithm to ensure reproducibility
    features = []
    for _ in tqdm(range(num_samples), position=1, leave=False):
        features.append(algo.sample_feature())

    if cache_dir is not None:
        pickle.dump(features, cache_file.open("wb"))
    return features

def max_hint_steps(features: List[Feature]) -> int:
    steps = [feature[1] for feature in features]
    assert all(isinstance(s, int) for s in steps), "All steps must be an integer"
    return max(steps)

def max_num_nodes(features: List[Feature]) -> int:
    num_nodes = [feature[2] for feature in features]
    assert all(isinstance(n, int) for n in num_nodes), "All num_nodes must be an integer"
    return max(num_nodes)

def collate_features(spec: Spec, features: List[Feature], min_hint_steps: int = 0, min_num_nodes: int = 0, device: Optional[torch.device]=torch.device('cpu')) -> Feature:
    """
    Batches features into a single feature. If device is None, numpy arrays are returned.
    Otherwise, torch tensors are returned on the specified device.
    If min_hint_steps if less than the maximum hint steps in the batch, the hint steps are padded to the maximum hint steps in the batch.
    Otherwise, the hint steps are padded to the min_hint_steps.
    """
    batch_trajectory = {Stage.INPUT: {}, Stage.OUTPUT: {}, Stage.HINT: {}}
    padded_hint_steps = max(min_hint_steps, max_hint_steps(features))
    padded_num_nodes = max(min_num_nodes, max_num_nodes(features))

    def apply_padding(name, data, padded_hint_steps, padded_num_nodes):
        stage, _, type_, metadata = spec[name]
        dim_paddings = []
        if stage == Stage.HINT:
            dim_paddings.append((0, padded_hint_steps - data.shape[0]))  # For the time dimension
        dim_paddings.append((0, 0))                                      # For the batch dimension

        # The remaining dimensions for hint remove time and batch dimensions, while for others there is no batch dimension
        offset = len(dim_paddings)
        remaining_dims = data.ndim - offset

        for dim in range(remaining_dims):
            if dim == remaining_dims - 1 and type_ == Type.CATEGORICAL:
                assert metadata['num_classes'] == data.shape[-1], "Number of classes does not match the number of classes in the metadata"
                dim_paddings.append((0, 0))  # For categorical feature, last dimension is the number of classes
            else:
                dim_paddings.append((0, padded_num_nodes - data.shape[dim + offset]))

        data = np.pad(data, dim_paddings, mode='constant', constant_values=0)
        return data

    for stage in Stage:
        for trajectory, _, _ in features:
            for key, value in trajectory[stage].items():
                assert isinstance(value, np.ndarray), "Value must be a numpy array" # At this point, the value is a numpy array
                if key not in batch_trajectory[stage]:
                    batch_trajectory[stage][key] = []
                value = apply_padding(key, value, padded_hint_steps, padded_num_nodes)
                batch_trajectory[stage][key].append(value)

        for key, value in batch_trajectory[stage].items():
            concat_dim = 1 if stage == Stage.HINT else 0
            batch_trajectory[stage][key] = np.concatenate(value, axis=concat_dim)

            if device is not None:
                batch_trajectory[stage][key] = torch.from_numpy(batch_trajectory[stage][key]).float().to(device)

    steps = [num_steps for _, num_steps, _ in features]
    num_nodes = [num_nodes for _, _, num_nodes in features]
    assert all(isinstance(s, int) for s in steps), "All steps must be an integer" # At this point, the steps are integers
    assert all(isinstance(n, int) for n in num_nodes), "All num_nodes must be an integer"
    if device is not None:
        steps = torch.from_numpy(np.array(steps)).long().to(device)
        num_nodes = torch.from_numpy(np.array(num_nodes)).long().to(device)
    else:
        steps = steps[0] if len(steps) == 1 else steps   # In the case of a single trajectory and not converted to tensor
        num_nodes = num_nodes[0] if len(num_nodes) == 1 else num_nodes
    batch_feature = (batch_trajectory, steps, num_nodes)
    return batch_feature

def construct_batches(chunk_size: int, batch_size: int, features: List[Feature], seed: int) -> List[Batch]:
    num_chunks = [math.ceil(num_steps / chunk_size) for (_, num_steps, _) in features]

    buckets = {}
    for fid, c in enumerate(num_chunks):
        buckets.setdefault(c, []).append(fid)

    rng = np.random.RandomState(seed)

    blocks = []
    for num_chunks, fids in buckets.items():
        rng.shuffle(fids)
        # split fids into sub‐batches of at most batch_size
        for i in range(0, len(fids), batch_size):
            batch_fids = fids[i:i+batch_size]
            if len(batch_fids) < batch_size:
                break
            batch = []
            for chunk_idx in range(num_chunks):
                indices = batch_fids
                is_last = chunk_idx == num_chunks - 1
                batch.append((indices, chunk_idx, is_last))
            blocks.append(batch)

    rng.shuffle(blocks)
    batches = []
    for block in blocks:
        for sub_block in block:
            batches.append(sub_block)

    return batches

def load_features(algo: AlgorithmEnum, 
                  sizes: List[int], 
                  seed: int, 
                  num_samples: int, 
                  cache_dir: Optional[Union[str, Path]] = None, 
                  algo_kwargs: Dict = {}) -> Tuple[Spec, List[Feature], NumNodes, NumSteps]:
    features = [] 
    max_num_nodes = 0
    max_num_steps = 0
    spec = None

    progress_bar = tqdm(sizes, position=0, leave=True)
    for size in progress_bar:
        progress_bar.set_description(f"Loading {algo.name} features for size {size}")
        algorithm = Algorithm(algo, seed=seed, length=size, **algo_kwargs)
        spec = algorithm.spec if spec is None else spec
        size_features = load_algo_features(algo=algorithm, 
                                    num_samples=num_samples,
                                    cache_dir=cache_dir)
            
        for feature in size_features:
            max_num_nodes = max(max_num_nodes, feature[2])
            max_num_steps = max(max_num_steps, feature[1])
            features.append(feature)

    return spec, features, max_num_nodes, max_num_steps

def batch_to_features(batch: Batch, features: List[Feature], chunk_size: int) -> Tuple[List[Feature], IsFirst, IsLast]:
    chunk_features = []
    feature_indices, chunk_idx, is_last = batch

    for fid in feature_indices:
        trajectory, num_steps, num_nodes = features[fid]
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_steps)

        inputs = trajectory[Stage.INPUT]
        hints = {}
        outputs = {}
        num_steps = end_idx - start_idx
        for key, value in trajectory[Stage.HINT].items():
            hints[key] = value[start_idx:end_idx]

        if is_last:
            for key, value in trajectory[Stage.OUTPUT].items():
                outputs[key] = value

        new_trajectory = {
            Stage.INPUT: inputs,
            Stage.HINT: hints,
            Stage.OUTPUT: outputs,
        }
        chunk_features.append((new_trajectory, num_steps, num_nodes))

    return chunk_features, chunk_idx == 0, is_last
    
class AlgoFeatureDataset(Dataset):
    def __init__(self, algo: AlgorithmEnum, 
                sizes: List[int], 
                chunk_size: Optional[int] = 16,
                seed: int = 42,
                batch_size: int = 32,
                num_batches: int = 1000,
                static_batch_size: bool = True,
                cache_dir: Union[str, Path] = None,
                algo_kwargs: Dict = {}):
        super().__init__()
        self.static_batch_size = static_batch_size
        self.algo_kwargs = algo_kwargs
        self.num_batches = num_batches
        self.batch_size = batch_size

        assert "length" not in self.algo_kwargs, "length must be specified in sizes"

        self.spec, self.features, self.max_num_nodes, self.max_num_steps = load_features(algo=algo, 
                                                                                        sizes=sizes, 
                                                                                        seed=seed, 
                                                                                        num_samples=self.batch_size * self.num_batches, 
                                                                                        cache_dir=cache_dir, 
                                                                                        algo_kwargs=self.algo_kwargs)
            
        # If chunk_size is not specified, use the maximum number of steps in the features
        # Essentially, this means no chunking as every trajectory fits into a single chunk
        self.chunk_size = chunk_size if chunk_size is not None else self.max_num_steps
        self.batches = construct_batches(chunk_size=self.chunk_size, 
                                        batch_size=batch_size, 
                                        features=self.features, 
                                        seed=seed)


    def __getitem__(self, idx: int) -> FeatureBatch:
        batch = self.batches[idx]
        features, is_first, is_last = batch_to_features(self.spec, 
                                                        batch=batch, 
                                                        features=self.features, 
                                                        chunk_size=self.chunk_size)
        
        batch_feature = collate_features(self.spec, 
                                        features=features, 
                                        min_hint_steps=self.chunk_size if self.static_batch_size else 0, 
                                        min_num_nodes=self.max_num_nodes if self.static_batch_size else 0)
        return batch_feature, is_first, is_last
      

    def __len__(self):
        return self.num_batches
    
    

if __name__ == "__main__":
    dataset = AlgoFeatureDataset(algo=AlgorithmEnum.articulation_points, 
                                        sizes=[4, 8, 16], 
                                        seed=42,
                                        chunk_size=16, 
                                        batch_size=32, 
                                        num_batches=1000,
                                        static_batch_size=True,
                                        algo_kwargs={},
                                        cache_dir=".cache")

    for i in range(len(dataset)):
        batch = dataset[i]
        import ipdb; ipdb.set_trace()