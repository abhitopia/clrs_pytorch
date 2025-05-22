from collections import defaultdict
import hashlib
import math
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .specs import Algorithm, Feature, Spec, Stage, NumNodes, NumSteps, Type
from .algorithm import AlgorithmSampler
from typing import Iterator, List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
from multiprocessing import Pool, cpu_count


Batch = Tuple[List[int], int, bool]
IsLast = bool
IsFirst = bool
FeatureBatch = Tuple[Feature, IsFirst, IsLast]
DictFeatureBatch = Tuple[Dict[Algorithm, FeatureBatch], Dict[Algorithm, IsFirst], Dict[Algorithm, IsLast]]


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

def combined_algorithm_sampler(samplers: List[AlgorithmSampler]):
    def _fn():
        while True:
            for sampler in samplers:
                yield sampler.sample_feature()
    return _fn()

def construct_batches(
    chunk_size: Optional[int],
    batch_size: int,
    sampler: Iterator[Feature],
    num_batches: int, seed: int,
    progress_bar: Optional[tqdm] = None) -> Tuple[List[Batch], List[Feature], int, int]:
    """
    Pull features one by one from `sampler` (an iterator over Feature).
    Group them into buckets by how many chunks they'll produce.  As soon
    as any bucket has ≥ batch_size features, pop a batch, shuffle it, and
    yield all its chunk-level sub-batches.  Stop after yielding num_batches.
    """
    rng = np.random.RandomState(seed)
    buckets = defaultdict(list)

    if isinstance(chunk_size, int) and chunk_size <= 0:
        chunk_size = None

    batches = []
    features = []
    max_num_nodes = 0
    max_num_steps = 0

    for fid, feature in enumerate(sampler):
        # how many chunks this feature will produce
        features.append(feature)

        _, num_steps, num_nodes = feature
        max_num_nodes = max(max_num_nodes, num_nodes)
        max_num_steps = max(max_num_steps, num_steps)

        if chunk_size is None:
            n_chunks = math.ceil(num_steps / max_num_steps)
        else:
            n_chunks = math.ceil(num_steps / chunk_size)

        # add to bucket
        bucket = buckets[n_chunks]
        bucket.append(fid)

        # if we have enough to form one batch, do it
        if len(bucket) >= batch_size:
            # take the first batch_size features
            block_feat_ids = bucket[:batch_size]
            del bucket[:batch_size]

            # shuffle for randomness
            rng.shuffle(block_feat_ids)

            # emit one Batch per chunk index
            for chunk_idx in range(n_chunks):
                is_last = (chunk_idx == n_chunks - 1)
                batches.append((block_feat_ids, chunk_idx, is_last))
                if progress_bar is not None:
                    progress_bar.update(1)

        if len(batches) >= num_batches:
            break


    # Cannot shuffle the batches here or they will lose the order of the batches
    batches = batches[:num_batches] # Keep only the required number of batches

    assert len(batches) == num_batches

    if chunk_size is None:
        assert len(features) == num_batches * batch_size, "With Chunk Size None, the number of features should be equal to the number of batches times the batch size"

    return batches, features, max_num_nodes, max_num_steps

def _load_data_wrapper(kwargs_dict: Dict[str, Any]):
    """Helper function to unpack keyword arguments for load_data in multiprocessing."""
    return load_data(**kwargs_dict)

def load_data_parallel(
    algo_configs: Dict[Algorithm, Dict[str, any]],
    num_processes: Optional[int] = None,
    verbose: bool = True
) -> Dict[Algorithm, Tuple[Spec, List[Batch], List[Feature], int, int]]:
    """
    Loads data for multiple algorithms in parallel using multiprocessing.

    Args:
        algo_configs: A dictionary where keys are Algorithm enums and values are
                      dictionaries of parameters for the load_data function.
                      Each inner dictionary can specify:
                        - num_batches: int (default: 1000)
                        - sizes: Union[List[int], int] (default: [4, 7, 11, 13, 16])
                        - seed: int (default: 42)
                        - batch_size: int (default: 32)
                        - chunk_size: Optional[int] (default: None)
                        - algo_kwargs: Dict (default: {})
                        - cache_dir: Optional[str] (default: None)
        num_processes: Number of parallel processes to use. Defaults to cpu_count() - 1.

    Returns:
        A dictionary mapping each algorithm to its loaded data tuple:
        (Spec, List[Batch], List[Feature], max_num_nodes, max_num_steps).
    """
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1) # Ensure at least 1 process

    tasks = []
    algorithms_in_order = [] # To maintain order for results reconstruction
    for alg, config in algo_configs.items():
        algorithms_in_order.append(alg)
        # Prepare kwargs for load_data, ensuring all necessary keys are present with defaults
        task_kwargs = {
            'algorithm': alg,
            'num_batches': config.get('num_batches', 1000),
            'sizes': config.get('sizes', [4, 7, 11, 13, 16]),
            'seed': config.get('seed', 42),
            'batch_size': config.get('batch_size', 32),
            'chunk_size': config.get('chunk_size', None),
            'algo_kwargs': config.get('algo_kwargs', {}),
            'cache_dir': config.get('cache_dir', None),
            'verbose': verbose  
        }
        tasks.append(task_kwargs)

    results_list = []
    if tasks: # Proceed only if there are tasks to run
        with Pool(processes=min(num_processes, len(tasks))) as pool: # Don't create more processes than tasks
            results_list = list(pool.imap(_load_data_wrapper, tasks))

    results_dict = {}
    for i, alg in enumerate(algorithms_in_order):
        if i < len(results_list): # Check if result exists for the algorithm
             results_dict[alg] = results_list[i]
        
    return results_dict

def load_data(algorithm: Algorithm, 
              num_batches: int = 1000,
              sizes: Union[List[int], int] = [4, 7, 11, 13, 16], 
              seed: int = 42, 
              batch_size: int = 32,
              chunk_size: Optional[int] = None,
              algo_kwargs: Dict = {},
              cache_dir: Optional[str] = None,
              verbose: bool = True) -> Tuple[Spec, List[Batch], List[Feature], int, int]:
    
    if isinstance(sizes, int):
        sizes = [sizes]

    sizes = sorted(sizes) # Sort to make the hash key deterministics
    samplers = []
    hash_keys = [num_batches, batch_size, str(chunk_size), sizes, seed]
    for idx, size in enumerate(sizes):
        sampler = AlgorithmSampler(algorithm, length=size, seed=seed+idx, **algo_kwargs)
        samplers.append(sampler)
        hash_keys.append(sampler.unique_hash())

    spec = samplers[0].spec
    hash_key = hashlib.md5(str(hash_keys).encode()).hexdigest()

    progress_bar = tqdm(total=num_batches, desc=f"Loading batches for {algorithm}") if verbose else None

    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / algorithm / f"{hash_key}.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            if verbose:
                progress_bar.update(num_batches)
            return pickle.load(cache_file.open("rb"))
        
    combined_sampler = combined_algorithm_sampler(samplers)
    batches, features, max_num_nodes, max_num_steps = construct_batches(
                                                                    chunk_size=chunk_size, 
                                                                    batch_size=batch_size, 
                                                                    sampler=combined_sampler, 
                                                                    num_batches=num_batches, 
                                                                    seed=seed, 
                                                                    progress_bar=progress_bar)
    result = spec, batches, features, max_num_nodes, max_num_steps

    if cache_file is not None:
        pickle.dump(result, cache_file.open("wb"))

    return result

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

    @property
    def spec(self):
        if self._spec is None:
            self._maybe_load_data()
        return self._spec

    def _maybe_load_data(self, verbose: bool = False):
        if self.batches is not None:
            return
        spec, batches, features, max_num_nodes, max_num_steps = load_data(algorithm=self.algorithm, 
                                                                          num_batches=self.num_batches, 
                                                                          sizes=self.sizes, 
                                                                          seed=self.seed, 
                                                                          batch_size=self.batch_size, 
                                                                          chunk_size=self.chunk_size, 
                                                                          cache_dir=self.cache_dir, 
                                                                          verbose=verbose)
        self._spec = spec
        self.batches = batches
        self.features = features
        self.max_num_nodes = max_num_nodes
        self.max_num_steps = max_num_steps

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

