import math
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .specs import Algorithm, Feature, Spec, Stage, NumNodes, NumSteps, Type
from .algorithm import AlgorithmSampler
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path


Batch = Tuple[List[int], int, bool]
IsLast = bool
IsFirst = bool
FeatureBatch = Tuple[Feature, IsFirst, IsLast]
DictFeatureBatch = Tuple[Dict[Algorithm, FeatureBatch], Dict[Algorithm, IsFirst], Dict[Algorithm, IsLast]]


def sample_features(algo: AlgorithmSampler, num_samples: int, cache_dir: Union[str, Path] = None, verbose: bool = True) -> List[Feature]:
    if cache_dir is not None:
        cache_dir = Path(cache_dir) / algo.name
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        cache_file = cache_dir / f"{algo.unique_hash()}-{num_samples}.pkl"
        if cache_file.exists():
            return pickle.load(cache_file.open("rb"))
        
    algo.reset() # Reset the algorithm to ensure reproducibility
    features = []

    if verbose:
        progress_bar = tqdm(range(num_samples), position=1, leave=False)
    else:
        progress_bar = range(num_samples)

    for _ in progress_bar:
        features.append(algo.sample_feature())

    if cache_dir is not None:
        pickle.dump(features, cache_file.open("wb"))
    return features

def max_hint_steps(features: List[Feature]) -> NumSteps:
    steps = [feature[1] for feature in features]
    assert all(isinstance(s, int) for s in steps), "All steps must be an integer"
    return max(steps)

def max_num_nodes(features: List[Feature]) -> NumNodes:
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

def load_features(algo: Algorithm, 
                  sizes: List[int], 
                  seed: int, 
                  num_samples: int, 
                  cache_dir: Optional[Union[str, Path]] = None, 
                  algo_kwargs: Dict = {},
                  verbose: bool = True) -> Tuple[Spec, List[Feature], NumNodes, NumSteps]:
    features = [] 
    max_num_nodes = 0
    max_num_steps = 0
    spec = None

    if verbose:
        progress_bar = tqdm(sizes, position=0, leave=True)
    else:
        progress_bar = sizes

    for size in progress_bar:
        if verbose:
            progress_bar.set_description(f"Loading features for {algo} with size {size}")
        algorithm = AlgorithmSampler(algo, seed=seed, length=size, **algo_kwargs)
        spec = algorithm.spec if spec is None else spec
        size_features = sample_features(algo=algorithm, 
                                    num_samples=num_samples,
                                    cache_dir=cache_dir,
                                    verbose=verbose)
            
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
    def __init__(self, algorithm: Algorithm, 
                sizes: Union[List[int], int], 
                chunk_size: Optional[int] = 16,
                seed: int = 42,
                batch_size: int = 32,
                num_batches: int = 1000,
                static_batch_size: bool = True,
                cache_dir: Union[str, Path] = None,
                algo_kwargs: Dict = {}):
        super().__init__()
        assert isinstance(algorithm, Algorithm), "algo must be an AlgorithmEnum"
        if isinstance(sizes, int):
            sizes = [sizes]

        assert all(isinstance(size, int) for size in sizes), "All sizes must be an integer"
        assert all(size > 0 for size in sizes), "All sizes must be positive"

        self.algorithm = algorithm
        self.static_batch_size = static_batch_size
        self.algo_kwargs = algo_kwargs
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.sizes = sizes
        self.seed = seed
        self.chunk_size = chunk_size
        self.cache_dir = cache_dir
        self.batches, self.features, self._spec, self.max_num_nodes, self.max_num_steps = None, None, None, None, None
        assert "length" not in self.algo_kwargs, "length must be specified in sizes"
        self._maybe_load_data(verbose=True) # Just to save the data cache on the disk, and load the spec
        self.reset()

    def reset(self):
        self.batches, self.features, self.max_num_nodes, self.max_num_steps = None, None, None, None

    @property
    def spec(self):
        if self._spec is None:
            self._maybe_load_data()
        return self._spec

    def _maybe_load_data(self, verbose: bool = False):
        if self.batches is not None:
            return
        num_samples_per_size = math.ceil(self.batch_size * self.num_batches / len(self.sizes))
        self._spec, self.features, self.max_num_nodes, self.max_num_steps = load_features(algo=self.algorithm, 
                                                                                        sizes=self.sizes, 
                                                                                        seed=self.seed, 
                                                                                        num_samples=num_samples_per_size, 
                                                                                        cache_dir=self.cache_dir, 
                                                                                        algo_kwargs=self.algo_kwargs,
                                                                                        verbose=verbose)
            
        # If chunk_size is not specified, use the maximum number of steps in the features
        # Essentially, this means no chunking as every trajectory fits into a single chunk
        self.chunk_size = self.chunk_size if self.chunk_size is not None else self.max_num_steps
        self.batches = construct_batches(chunk_size=self.chunk_size, 
                                        batch_size=self.batch_size, 
                                        features=self.features, 
                                        seed=self.seed)


    def __getitem__(self, idx: int) -> FeatureBatch:
        self._maybe_load_data()
        batch = self.batches[idx]
        features, is_first, is_last = batch_to_features(batch=batch, 
                                                        features=self.features, 
                                                        chunk_size=self.chunk_size)
        
        batch_feature = collate_features(self.spec, 
                                        features=features, 
                                        min_hint_steps=self.chunk_size if self.static_batch_size else 0, 
                                        min_num_nodes=self.max_num_nodes if self.static_batch_size else 0)
        return batch_feature, is_first, is_last
      

    def __len__(self):
        return self.num_batches
    
    def get_dataloader(self, num_workers: int = 0):
        return DataLoader(self, 
                          shuffle=False,
                          batch_size=1,
                          collate_fn=lambda x: x[0],
                          persistent_workers=num_workers > 0,
                          num_workers=num_workers)
    

    
class CyclicAlgoFeatureDataset(Dataset):
    def __init__(self, datasets: List[AlgoFeatureDataset]):
        super().__init__()
        assert all(isinstance(dataset, AlgoFeatureDataset) for dataset in datasets), "All datasets must be AlgoFeatureDataset"
        assert len(set([ds.algorithm for ds in datasets])) == len(datasets), "All datasets must have unique algorithms"
        assert all(len(dataset) == len(datasets[0]) for dataset in datasets), "All datasets must have the same number of batches"
        self.datasets = datasets

    @property
    def specs(self):
        return {ds.algorithm: ds.spec for ds in self.datasets}

    def __getitem__(self, idx: int) -> DictFeatureBatch:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for size {len(self)}")
        
        # cycles through the datasets 
        dataset_idx = idx % len(self.datasets)
        offset_idx = idx // len(self.datasets)

        ds = self.datasets[dataset_idx]
        feature_batch, is_first, is_last = ds[offset_idx]
        return {ds.algorithm: feature_batch}, {ds.algorithm: is_first}, {ds.algorithm: is_last}
    
    def __len__(self):
        return len(self.datasets[0]) * len(self.datasets)
    
    def get_dataloader(self, num_workers: int = 0):
        return DataLoader(self, 
                          shuffle=False,
                          batch_size=1,
                          collate_fn=lambda x: x[0],
                          persistent_workers=num_workers > 0,
                          num_workers=num_workers)
    

class StackedAlgoFeatureDataset(Dataset):
    def __init__(self, datasets: List[AlgoFeatureDataset]):
        super().__init__()
        assert all(isinstance(dataset, AlgoFeatureDataset) for dataset in datasets), "All datasets must be AlgoFeatureDataset"
        assert len(set([ds.algorithm for ds in datasets])) == len(datasets), "All datasets must have unique algorithms"
        assert all(len(dataset) == len(datasets[0]) for dataset in datasets), "All datasets must have the same number of batches"
        self.datasets = {ds.algorithm: ds for ds in datasets}
        self._num_batches = len(datasets[0])

    @property
    def specs(self):
        return {ds.algorithm: ds.spec for ds in self.datasets.values()}

    def __getitem__(self, idx: int) -> DictFeatureBatch:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for size {len(self)}")
        feature_batch, is_first, is_last = {}, {}, {}
        for algo, dataset in self.datasets.items():
            feature_batch[algo], is_first[algo], is_last[algo] = dataset[idx]
        return feature_batch, is_first, is_last
    
    def __len__(self):
        return self._num_batches
    
    def get_dataloader(self, num_workers: int = 0):
        return DataLoader(self, 
                          shuffle=False,
                          collate_fn=lambda x: x[0],
                          batch_size=1,
                          persistent_workers=num_workers > 0,
                          num_workers=num_workers)



if __name__ == "__main__":
    from .utils import tree_map
    dataset = AlgoFeatureDataset(algorithm=Algorithm.articulation_points, 
                                        sizes=[4, 8, 12], 
                                        seed=42,
                                        chunk_size=8, 
                                        batch_size=4, 
                                        num_batches=1000,
                                        static_batch_size=True,
                                        algo_kwargs={},
                                        cache_dir=".cache")


    # dataset._maybe_load_data()

    # for idx, batch in enumerate(dataset.batches):
    #     print(idx, batch)
    #     if idx > 10:
    #         break
    dl_1 = dataset.get_dataloader(num_workers=0)
    batches_1 = []
    for idx, batch in enumerate(dl_1):
        import ipdb; ipdb.set_trace()
        batches_1.append(batch)
        if idx >= 4:
            break

    dl_2 = dataset.get_dataloader(num_workers=4)
    batches_2 = []
    for idx, batch in enumerate(dl_2):
        batches_2.append(batch)
        if idx >= 4:
            break

    print(len(batches_1), len(batches_2))

    for i in range(len(batches_1)):
        b1, b2 = batches_1[i], batches_2[i]
        b1_tensors, b2_tensors = [], []
        tree_map(lambda x: b1_tensors.append(x), b1)
        tree_map(lambda x: b2_tensors.append(x), b2)

        for t1, t2 in zip(b1_tensors, b2_tensors):
            assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} != {t2.shape}"
            assert (t1 == t2).all(), f"Value mismatch: {t1} != {t2}"

