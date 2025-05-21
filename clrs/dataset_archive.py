from collections import defaultdict
import copy
import itertools
from pathlib import Path
import pickle
import random
from typing import Dict, List, Optional, Union
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
from .specs import Stage, AlgorithmEnum, Spec, Type, Feature, Trajectory
from .algorithm import Algorithm
from .utils import all_lists_equal



SIZES_MAX_NUM_STEPS = {
    AlgorithmEnum.articulation_points: ([4, 7, 9, 10, 11], 50),
    AlgorithmEnum.bridges: ([4, 7, 9, 10, 12], 50),
    AlgorithmEnum.bubble_sort: ([4, 7, 8, 9, 10], 50),
    AlgorithmEnum.dag_shortest_paths: ([4, 7, 11, 13, 16], 40), 
    AlgorithmEnum.heapsort: ([4, 5, 7, 9, 10], 50), 
    AlgorithmEnum.jarvis_march: ([3, 4, 5, 6, 7], 50),
    AlgorithmEnum.mst_kruskal: ([4, 7, 11, 13, 16], 50),
    AlgorithmEnum.quickselect: ([4, 7, 11, 13, 16], 50),
    AlgorithmEnum.quicksort: ([4, 7, 11, 12, 13], 50),
    AlgorithmEnum.strongly_connected_components: ([4, 5, 6, 7, 8], 50),
}

DEFAULT_MAX_NUM_STEPS = 50
DEFAULT_SIZES = [4, 7, 11, 13, 16]



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
    for _ in range(num_samples):
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


class AlgoFeatureDataset(Dataset):
    def __init__(self, algo: AlgorithmEnum, 
                size: int, 
                num_samples: int, 
                cache_dir: Union[str, Path] = None,
                generate_on_the_fly: bool = False,
                static_batch_size: bool = True,
                algo_kwargs: Optional[Dict] = None):
        super().__init__()
        assert algo_kwargs is None or "length" not in algo_kwargs, "length must be specified in trajectory_sizes"

        self.algorithm = Algorithm(algo, length=size, **algo_kwargs if algo_kwargs is not None else {})
        self.cache_dir = cache_dir
        self.features: List[Feature] = None
        self._max_hint_steps = None
        self._num_nodes = None
        self.num_samples = num_samples
        self.generate_on_the_fly = generate_on_the_fly
        self.static_batch_size = static_batch_size
    @property
    def spec(self) -> Spec:
        return self.algorithm.spec
    
    @property
    def name(self) -> str:
        return self.algorithm.name
    
    @property
    def max_hint_steps(self) -> int:
        if self._max_hint_steps is None:
            self._maybe_load_data()
        return self._max_hint_steps
    
    @property
    def num_nodes(self) -> int:
        if self._num_nodes is None:
            self._maybe_load_data()
        return self._num_nodes

    def _maybe_load_data(self):
        if self.features is None:
            self.features = load_algo_features(self.algorithm, self.num_samples, self.cache_dir)
            self._max_hint_steps = max_hint_steps(self.features)
            self._num_nodes = max_num_nodes(self.features)

    def __len__(self):
        return self.num_samples
    
    def collate_fn(self, batch: List[Feature], device: Optional[torch.device]=torch.device('cpu')) -> Feature:
        self._maybe_load_data()
        batched_feature = collate_features(self.spec, 
                                        batch, 
                                        min_hint_steps=self.max_hint_steps,
                                        min_num_nodes=0,  # For same algorith, num_nodes is the same
                                        device=device)
        return batched_feature

    def __getitem__(self, idx) -> Feature:
        self._maybe_load_data()
        feature = self.algorithm.sample_feature() if self.generate_on_the_fly else self.features[idx]
        if self.static_batch_size:
            return self.collate_fn([feature], device=None)
        return feature
    

class MultiSizeAlgoFeatureDataset(Dataset):
    def __init__(self, algo: AlgorithmEnum, 
                sizes: Union[int, List[int]],
                num_samples: int, 
                cache_dir: Union[str, Path] = None,
                generate_on_the_fly: bool = False,
                static_batch_size: bool = True,
                algo_kwargs: Optional[Dict] = None):
        super().__init__()
        
        if isinstance(sizes, int):
            sizes = [sizes]

        self.datasets = []
        self._num_samples_per_algo = num_samples // len(sizes)
        assert self._num_samples_per_algo > 0, "Number of samples per algorithm must be greater than 0"
        
        for idx, size in enumerate(sizes):
            assert size > 0, "Trajectory size must be greater than 0"

            if idx == len(sizes) - 1:
                # Last dataset gets the remaining samples
                num_samples_per_algo = num_samples - self._num_samples_per_algo * idx
            else:
                num_samples_per_algo = self._num_samples_per_algo

            ds = AlgoFeatureDataset(algo=algo,
                                    size=size,
                                    num_samples=num_samples_per_algo,
                                    cache_dir=cache_dir,
                                    generate_on_the_fly=generate_on_the_fly,
                                    static_batch_size=False,
                                    algo_kwargs=algo_kwargs)
            self.datasets.append(ds)

        self._num_samples = num_samples
        self._name = algo
        self._spec = self.datasets[0].spec
        self._max_hint_steps = None
        self._max_num_nodes = None
        self.static_batch_size = static_batch_size

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def spec(self) -> Spec:
        return self._spec

    @property
    def max_hint_steps(self) -> int:
        if self._max_hint_steps is None:
            self._maybe_load_data()
        return self._max_hint_steps
    
    @property
    def max_num_nodes(self) -> int:
        if self._max_num_nodes is None:
            self._maybe_load_data()
        return self._max_num_nodes

    def _maybe_load_data(self):
        for dataset in self.datasets:
            dataset._maybe_load_data()
        self._max_hint_steps = max(dataset.max_hint_steps for dataset in self.datasets)
        self._max_num_nodes = max(dataset.num_nodes for dataset in self.datasets)

    def __len__(self):
        return self._num_samples
    
    def __getitem__(self, idx: int) -> Feature:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for size {len(self)}")
        self._maybe_load_data()
        dataset_idx = idx // self._num_samples_per_algo
        offset = idx % self._num_samples_per_algo
        # print(f"MultiSizeAlgoFeatureDataset[{self.name}] dataset_idx: {dataset_idx}, offset: {offset}")
        dataset = self.datasets[dataset_idx]
        return dataset[offset]
    
    
    def collate_fn(self, batch: List[Feature], device: Optional[torch.device]=torch.device('cpu')) -> Feature:
        self._maybe_load_data()
        min_hint_steps = self.max_hint_steps if self.static_batch_size else 0
        min_num_nodes = self.max_num_nodes if self.static_batch_size else 0
        batched_feature = collate_features(self.spec,
                                        batch, 
                                        min_hint_steps=min_hint_steps,
                                        min_num_nodes=min_num_nodes,
                                        device=device)
        return batched_feature
    

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        return DataLoader(self, 
                        collate_fn=self.collate_fn,
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        drop_last=drop_last, 
                        num_workers=num_workers)
    

class RoundRobinBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset_lengths: list[int],
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """
        Args:
            dataset_lengths: lengths of each sub‐dataset, in order.
            batch_size:      how many samples per batch (within each sub‐dataset).
            shuffle:         whether to shuffle indices within each sub‐dataset every epoch.
            drop_last:       if True, drop the final batch of a sub‐dataset when it's smaller than batch_size.
        """
        self.dataset_lengths = list(dataset_lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # cumulative offsets to map local → global indices
        self.cum_offsets = [0] + list(itertools.accumulate(self.dataset_lengths))  # len = n+1

    def __iter__(self):
        # 1) build per‐dataset list of batches of *local* indices
        per_ds_batches: list[list[list[int]]] = []
        for ds_len in self.dataset_lengths:
            idxs = list(range(ds_len))
            if self.shuffle:
                random.shuffle(idxs)
            # chunk into batches
            batches = [
                idxs[i : i + self.batch_size]
                for i in range(0, ds_len, self.batch_size)
            ]
            if self.drop_last and batches and len(batches[-1]) < self.batch_size:
                batches.pop()
            per_ds_batches.append(batches)

        # 2) round-robin yield one batch from each dataset until all are exhausted
        pointers = [0] * len(self.dataset_lengths)
        while True:
            any_yielded = False
            for ds_idx, batches in enumerate(per_ds_batches):
                ptr = pointers[ds_idx]
                if ptr < len(batches):
                    any_yielded = True
                    local_batch = batches[ptr]
                    # map to global indices
                    offset = self.cum_offsets[ds_idx]
                    global_batch = [offset + i for i in local_batch]
                    yield global_batch
                    pointers[ds_idx] += 1
            if not any_yielded:
                break

    def __len__(self):
        # total number of batches across all sub‐datasets
        total = 0
        for ds_len in self.dataset_lengths:
            if self.drop_last:
                total += ds_len // self.batch_size
            else:
                total += (ds_len + self.batch_size - 1) // self.batch_size
        return total


class ConcatAlgoTrajectoryDataset(Dataset):
    def __init__(self, algo_datasets: Union[MultiSizeAlgoFeatureDataset, List[MultiSizeAlgoFeatureDataset]]):
        super().__init__()
        if isinstance(algo_datasets, MultiSizeAlgoFeatureDataset):
            algo_datasets = [algo_datasets]
        assert all(isinstance(ds, MultiSizeAlgoFeatureDataset) for ds in algo_datasets), "All datasets must be MultiSizeAlgoTrajectoryDataset"
        assert all_lists_equal([len(ds) for ds in algo_datasets]), "All MultiSizeAlgoTrajectoryDataset have equal same length"
        self.datasets = algo_datasets
        self.algo_to_ds_idx = {ds.name: i for i, ds in enumerate(self.datasets)}
        self._individual_ds_len = len(self.datasets[0])
        self._spec = {ds.name: ds.spec for ds in algo_datasets}

    @property
    def specs(self) -> Dict[AlgorithmEnum, Spec]:
        return self._spec

    def __len__(self):
        return self._individual_ds_len * len(self.datasets)
    
    def __getitem__(self, idx) -> Dict[AlgorithmEnum, Feature]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for size {len(self)}")
        dataset_idx = idx // self._individual_ds_len
        offset = idx % self._individual_ds_len
        dataset = self.datasets[dataset_idx]
        # print(f"ConcatAlgoFeatureDataset dataset_idx: {dataset_idx}[{dataset.name}], offset: {offset}")
        return {dataset.name: dataset[offset]}
    
    def collate_fn(self, batch: List[Dict[AlgorithmEnum, Feature]], device: Optional[torch.device]=torch.device('cpu')) -> Dict[AlgorithmEnum, Feature]:
        dict_batch = defaultdict(list)
        for sample in batch:
            for name in sample.keys():
                dict_batch[name].append(sample[name])
        
        assert len(dict_batch) == 1, "Batch must contain only one algorithm"
        for name, features in dict_batch.items():
            dict_batch[name] = self.datasets[self.algo_to_ds_idx[name]].collate_fn(features, device)
        return dict_batch
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        return DataLoader(self, 
                        batch_sampler=RoundRobinBatchSampler(dataset_lengths=[len(ds) for ds in self.datasets],
                                                      batch_size=batch_size,
                                                      shuffle=shuffle,
                                                      drop_last=drop_last),
                        collate_fn=self.collate_fn,
                        num_workers=num_workers)

    

class StackedAlgoFeatureDataset(Dataset):
    """
    Concatenates multiple StackedAlgoFeatureDataset, where the Algorithms are different
    """
    def __init__(self, algo_datasets: Union[MultiSizeAlgoFeatureDataset, List[MultiSizeAlgoFeatureDataset]]):
        super().__init__()
        if isinstance(algo_datasets, MultiSizeAlgoFeatureDataset):
            algo_datasets = [algo_datasets]
        assert all(isinstance(ds, MultiSizeAlgoFeatureDataset) for ds in algo_datasets), "All datasets must be MultiSizeAlgoFeatureDataset"
        assert all_lists_equal([len(ds) for ds in algo_datasets]), "All MultiSizeAlgoFeatureDataset have equal same length"
        self.datasets = {ds.name: ds for ds in algo_datasets}
        self.algo_names = list(self.datasets.keys())
        self._len = len(self.datasets[self.algo_names[0]])
        assert len(self.algo_names) == len(algo_datasets), "All MultiSizeAlgoFeatureDatasets must have different algorithms"


    def __len__(self):
        return self._len
    
    @property
    def specs(self) -> Dict[str, Spec]:
        return {k: v.spec for k, v in self.datasets.items()}
    
    def __getitem__(self, idx) -> Dict[str, Feature]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for size {len(self)}")
        
        # print(f"StackedAlgoFeatureDataset idx: {idx}")
        result = {}
        for name, dataset in self.datasets.items():
            result[name] = dataset[idx]
        return result

    def collate_fn(self, batch: List[Dict[str, Feature]], device: Optional[torch.device]=torch.device('cpu')) -> Dict[str, Feature]:
        dict_batch = defaultdict(list)
        algo_names = set(self.datasets.keys())
        for sample in batch:
            for name in algo_names:
                dict_batch[name].append(sample[name])
        
        for name, features in dict_batch.items():
            dict_batch[name] = self.datasets[name].collate_fn(features, device)
        return dict_batch
    
    def get_dataloader(self, batch_size: int = 32, shuffle: bool = False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        return DataLoader(self, 
                        collate_fn=self.collate_fn,
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        drop_last=drop_last, 
                        num_workers=num_workers)


def get_dataset(algos: Union[AlgorithmEnum, List[AlgorithmEnum]],
                split: str = "train",
                sizes: Optional[Union[int, List[int]]] = None,
                cache_dir: Optional[Union[str, Path]] = None,
                generate_on_the_fly: bool = False,
                static_batch_size: bool = True,
                max_num_steps: Optional[int] = None,
                stacked: bool = False,
                string_matcher_override: bool = True, 
                **algo_kwargs) -> StackedAlgoFeatureDataset:
    
    assert split in ["train", "val", "test"], "Split must be one of train, val, test"

    if split == "train":
        num_samples = 1000
    elif split == "val":
        num_samples = 32
    elif split == "test":
        num_samples = 32
    
    if isinstance(algos, AlgorithmEnum):
        algos = [algos]

    datasets = []
    for algo in algos:
        algo_sizes = sizes
        algo_max_num_steps = max_num_steps

        if algo_sizes is None:
            algo_sizes = SIZES_MAX_NUM_STEPS[algo][0] if algo in SIZES_MAX_NUM_STEPS else DEFAULT_SIZES

        if algo_max_num_steps is None:
            algo_max_num_steps = SIZES_MAX_NUM_STEPS[algo][1] if algo in SIZES_MAX_NUM_STEPS else DEFAULT_MAX_NUM_STEPS

        if split == "val":
            algo_sizes = [max(algo_sizes)]

        # As per the generalise algorithmic learner paper
        if algo in [AlgorithmEnum.naive_string_matcher, AlgorithmEnum.kmp_matcher] and string_matcher_override:
            max_length = max(algo_sizes)
            max_length = (max_length * 5) // 4
            algo_sizes = [max_length]*len(algo_sizes)

        algo_kwargs = copy.deepcopy(algo_kwargs)
        algo_kwargs['max_steps'] = algo_max_num_steps

        datasets.append(MultiSizeAlgoFeatureDataset(algo=algo,
                                            sizes=algo_sizes,
                                            num_samples=num_samples,
                                            cache_dir=cache_dir,
                                            generate_on_the_fly=generate_on_the_fly,
                                            static_batch_size=static_batch_size,
                                            algo_kwargs=algo_kwargs))    
    if stacked:
        return StackedAlgoFeatureDataset(datasets)
    else:
        return ConcatAlgoTrajectoryDataset(datasets)
    


if __name__ == "__main__":

    from clrs.specs import CLRS30Algorithms

    dataset = get_dataset(CLRS30Algorithms,
                          stacked=False,
                          static_batch_size=True,
                          split="train")
    
    for multi in dataset.datasets:
        for ds in multi.datasets:
            print(ds.name, ds.algorithm._max_steps, ds.algorithm._sampler_kwargs)
            import ipdb; ipdb.set_trace()

