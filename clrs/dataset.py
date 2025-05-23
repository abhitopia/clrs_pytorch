from collections import defaultdict
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .specs import Algorithm, Feature, Spec, Stage, NumNodes, NumSteps, Type
from .algorithm import AlgorithmSampler
from typing import Iterator, List, Dict, Optional, Tuple, Union


IsLast = bool
IsFirst = bool
FeatureBatch = Tuple[Feature, IsFirst, IsLast]
DictFeatureBatch = Tuple[Dict[Algorithm, FeatureBatch], Dict[Algorithm, IsFirst], Dict[Algorithm, IsLast]]


def _collate_fn(batch):
    return batch

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

def combined_algorithm_sampler(algorithm: Algorithm, sizes: List[int], seed: int, algo_kwargs: Dict):
    samplers = []
    for idx, size in enumerate(sizes):
        sampler = AlgorithmSampler(algorithm, length=size, seed=seed+idx, **algo_kwargs)
        samplers.append(sampler)

    def _fn():
        while True:
            for sampler in samplers:
                yield sampler.sample_feature()
    return _fn()

def batch_iterator(
    spec: Spec,
    seed: int,
    chunk_size: Optional[int],
    batch_size: int,
    static_batch_size: bool,
    sampler: Iterator[Feature],
    warm_up: int = 1000) -> Iterator[FeatureBatch]:

    rng = np.random.RandomState(seed)
    buckets = defaultdict(list)
    if isinstance(chunk_size, int) and chunk_size <= 0:
        chunk_size = None

    max_num_nodes = 0
    max_num_steps = 0

    for idx, feature in enumerate(sampler):
        # how many chunks this feature will produce
        _, num_steps, num_nodes = feature
        max_num_nodes = max(max_num_nodes, num_nodes)
        max_num_steps = max(max_num_steps, num_steps)

        if idx < warm_up:  # Warm up to compute the max_num_nodes and max_num_steps
            continue

        chunk_size_now = chunk_size if chunk_size is not None else max_num_steps
        n_chunks = math.ceil(num_steps / chunk_size_now)

        # add to bucket
        bucket = buckets[n_chunks]
        bucket.append(feature)

        # if we have enough to form one batch, do it
        if len(bucket) >= batch_size:
            # take the first batch_size features
            block_feats = bucket[:batch_size]

            # remove the first batch_size features from the bucket, but keep the rest for the next batch
            del bucket[:batch_size]

            # shuffle for randomness
            rng.shuffle(block_feats)

            # emit one Batch per chunk index
            for chunk_idx in range(n_chunks):
                is_first = (chunk_idx == 0)
                is_last = (chunk_idx == n_chunks - 1)

                # yield block_indices, is_first, is_last

                chunk_features = []

                for feature in block_feats:
                    trajectory, num_steps, num_nodes = feature
                    start_idx = chunk_idx * chunk_size_now
                    end_idx = min(start_idx + chunk_size_now, num_steps)

                    inputs = trajectory[Stage.INPUT]
                    outputs = trajectory[Stage.OUTPUT]
                    hints = {}
                    num_steps = end_idx - start_idx
                    for key, value in trajectory[Stage.HINT].items():
                        hints[key] = value[start_idx:end_idx]

                    # if is_last:
                    #     for key, value in trajectory[Stage.OUTPUT].items():
                    #         outputs[key] = value

                    new_trajectory = {
                        Stage.INPUT: inputs,
                        Stage.HINT: hints,
                        Stage.OUTPUT: outputs,
                    }
                    chunk_features.append((new_trajectory, num_steps, num_nodes))

                batch_feature = collate_features(spec, 
                                        features=chunk_features, 
                                        min_hint_steps=chunk_size_now if static_batch_size else 0, 
                                        min_num_nodes=max_num_nodes if static_batch_size else 0)

                yield batch_feature, is_first, is_last

    
class AlgoFeatureDataset(Dataset):
    def __init__(self, algorithm: Algorithm, 
                sizes: Union[List[int], int], 
                chunk_size: Optional[int] = 16,
                seed: int = 42,
                batch_size: int = 32,
                num_batches: int = 1000,
                static_batch_size: bool = True,
                algo_kwargs: Dict = {}):
        super().__init__()
        assert isinstance(algorithm, Algorithm), "algo must be an AlgorithmEnum"
        if isinstance(sizes, int):
            sizes = [sizes]

        self.algorithm = algorithm
        self.static_batch_size = static_batch_size
        self.algo_kwargs = algo_kwargs
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.sizes = sorted(sizes)
        self.seed = seed
        self.chunk_size = chunk_size

        # This is a hack to get the spec. We should not need to do this.
        self.spec = AlgorithmSampler(self.algorithm, length=sizes[0], **self.algo_kwargs).spec

        # internal state
        self._iter     = None     # will hold the active iterator
        self._last_idx = -1       # last index we successfully returned


    def _get_batch_iterator(self):
        combined_sampler = combined_algorithm_sampler(
            algorithm=self.algorithm, 
            sizes=self.sizes, 
            seed=self.seed, 
            algo_kwargs=self.algo_kwargs
        )
        return batch_iterator(
            spec=self.spec,
            seed=self.seed,
            chunk_size=self.chunk_size, 
            batch_size=self.batch_size, 
            static_batch_size=self.static_batch_size,
            sampler=combined_sampler,
            warm_up=1000)


    def __getitem__(self, idx: int) -> FeatureBatch:
        if idx >= len(self):
            self._iter = None
            raise IndexError(f"Index {idx} out of range for size {len(self)}")

        # if we're starting, or idx has gone backwards, re‐start
        if self._iter is None or idx <= self._last_idx:
            # Create the batch iterator lazily to avoid pickling issues
            self._iter     = iter(self._get_batch_iterator())
            self._last_idx = -1

        # advance from (last_idx+1) up to idx, inclusive
        for i in range(self._last_idx + 1, idx + 1):
            try:
                batch = next(self._iter)
            except StopIteration:
                raise IndexError(f"Batch iterator exhausted at position {i}") 
        self._last_idx = idx
        return batch

    def __len__(self):
        return self.num_batches
    
    def get_dataloader(self, num_workers: int = 0):
        return DataLoader(self, 
                          shuffle=False,
                          batch_size=None,
                          collate_fn=_collate_fn,
                          persistent_workers=num_workers > 0,
                          prefetch_factor=30 if num_workers > 0 else None,
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
                          batch_size=None,
                          collate_fn=_collate_fn,
                          persistent_workers=num_workers > 0,
                          prefetch_factor=30 if num_workers > 0 else None,
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
                          batch_size=None,
                          collate_fn=_collate_fn,
                          persistent_workers=num_workers > 0,
                          prefetch_factor=30 if num_workers > 0 else None,
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

