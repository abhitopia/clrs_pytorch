from collections import defaultdict
from functools import partial
import itertools
from pathlib import Path
import pickle
import random
from typing import Dict, List, Optional, Union
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
import torch

from .specs import Stage, Algorithms, Spec
from .algorithm import Algorithm, Trajectory
from .utils import all_lists_equal, cycle_flatten_list_of_lists

def load_algo_trajectories(algo: Algorithm, num_samples: int, cache_dir: Union[str, Path] = None) -> List[Trajectory]:
    if cache_dir is not None:
        cache_dir = Path(cache_dir) / algo.name
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)
        cache_file = cache_dir / f"{algo.unique_hash()}-{num_samples}.pkl"
        if cache_file.exists():
            return pickle.load(cache_file.open("rb"))
        
    algo.reset() # Reset the algorithm to ensure reproducibility
    trajectories = []
    for _ in range(num_samples):
        trajectories.append(algo.sample_trajectory())

    if cache_dir is not None:
        pickle.dump(trajectories, cache_file.open("wb"))
    return trajectories


def max_hint_steps(trajectories: List[Trajectory]) -> int:
    max_steps = 0
    for trajectory in trajectories:
        for value in trajectory[Stage.HINT].values():
            assert value.shape[1] == 1, "Batch size of hint must be 1"
            if value.shape[0] > max_steps:
                max_steps = value.shape[0]
    return max_steps


def batch_trajectories(trajectories: List[Trajectory], min_hint_steps: int = 0, device: Optional[torch.device]=torch.device('cpu')) -> Trajectory:
    """
    Batches trajectories into a single trajectory. If device is None, numpy arrays are returned.
    Otherwise, torch tensors are returned on the specified device.
    If min_hint_steps if less than the maximum hint steps in the batch, the hint steps are padded to the maximum hint steps in the batch.
    Otherwise, the hint steps are padded to the min_hint_steps.
    """
    batch_trajectory = {Stage.INPUT: {}, Stage.OUTPUT: {}, Stage.HINT: {}}
    padded_hint_steps = max(min_hint_steps, max_hint_steps(trajectories))
    apply_pad = lambda d: np.pad(d, [(0, padded_hint_steps - d.shape[0])] + [(0, 0)]*(d.ndim-1), mode='constant', constant_values=0)
    steps = np.array([max_hint_steps([trajectory]) for trajectory in trajectories])
    if device is not None:
        steps = torch.from_numpy(steps).long().to(device)

    for stage in Stage:
        for trajectory in trajectories:
            for key, value in trajectory[stage].items():
                if key not in batch_trajectory[stage]:
                    batch_trajectory[stage][key] = []
                
                if stage == Stage.HINT:
                    value = apply_pad(value)

                batch_trajectory[stage][key].append(value)

        for key, value in batch_trajectory[stage].items():
            concat_dim = 1 if stage == Stage.HINT else 0
            batch_trajectory[stage][key] = np.concatenate(value, axis=concat_dim)

            if device is not None:
                batch_trajectory[stage][key] = torch.from_numpy(batch_trajectory[stage][key]).float().to(device)

    return batch_trajectory, steps


class AlgoTrajectoryDataset(Dataset):
    def __init__(self, algo: Algorithms, 
                trajectory_size: int, 
                num_samples: int, 
                cache_dir: Union[str, Path] = None,
                generate_on_the_fly: bool = False,
                uniform_hint_steps: bool = True,
                **algo_kwargs):
        super().__init__()
        assert "length" not in algo_kwargs, "length must be specified in trajectory_sizes"
        self.algorithm = Algorithm(algo, length=trajectory_size, **algo_kwargs)
        self.cache_dir = cache_dir
        self.trajectories: List[Trajectory] = None
        self._min_hint_steps = None
        self.num_samples = num_samples
        self.generate_on_the_fly = generate_on_the_fly
        self.uniform_hint_steps = uniform_hint_steps
    @property
    def spec(self) -> Spec:
        return self.algorithm.spec
    
    @property
    def name(self) -> str:
        return self.algorithm.name
    
    @property
    def min_hint_steps(self) -> int:
        if self._min_hint_steps is None:
            self._maybe_load_data()
        return self._min_hint_steps

    def _maybe_load_data(self):
        if self.trajectories is None:
            self.trajectories = load_algo_trajectories(self.algorithm, self.num_samples, self.cache_dir)
            self._min_hint_steps = max_hint_steps(self.trajectories)

    def __len__(self):
        return self.num_samples
    
    def collate_fn(self, batch: List[Trajectory], device: Optional[torch.device]=torch.device('cpu')) -> Trajectory:
        self._maybe_load_data()
        batched_trajectory, steps = batch_trajectories(batch, 
                                                       min_hint_steps=self.min_hint_steps,
                                                       device=device)
        return batched_trajectory, steps

    def __getitem__(self, idx) -> Trajectory:
        self._maybe_load_data()
        trajectory = self.algorithm.sample_trajectory() if self.generate_on_the_fly else self.trajectories[idx]
        if self.uniform_hint_steps:
            return self.collate_fn([trajectory])[0]
        return trajectory
    

class MultiSizeAlgoTrajectoryDataset(ConcatDataset):
    def __init__(self, algo: Algorithms, 
                trajectory_sizes: Union[int, List[int]], 
                num_samples: int, 
                cache_dir: Union[str, Path] = None,
                generate_on_the_fly: bool = False,
                uniform_hint_steps: bool = True,
                **algo_kwargs):
        
        if isinstance(trajectory_sizes, int):
            trajectory_sizes = [trajectory_sizes]

        datasets = []
        num_samples_per_algo = num_samples // len(trajectory_sizes)
        assert num_samples_per_algo > 0, "Number of samples per algorithm must be greater than 0"
        
        for idx, size in enumerate(trajectory_sizes):
            assert size > 0, "Trajectory size must be greater than 0"

            if idx == len(trajectory_sizes) - 1:
                # Last dataset gets the remaining samples
                num_samples_per_algo = num_samples - num_samples_per_algo * idx

            ds = AlgoTrajectoryDataset(algo=algo,
                                    trajectory_size=size,
                                    num_samples=num_samples_per_algo,
                                    cache_dir=cache_dir,
                                    generate_on_the_fly=generate_on_the_fly,
                                    uniform_hint_steps=True,  # Each subset must return a batch with the same number of hint steps
                                    **algo_kwargs)
            datasets.append(ds)
        super().__init__(datasets)
        self._name = algo
        self._spec = datasets[0].spec
        self._min_hint_steps = None
        self.uniform_hint_steps = uniform_hint_steps

    @property
    def min_hint_steps(self) -> int:
        if self._min_hint_steps is None:
            self._maybe_load_data()
        return self._min_hint_steps

    def _maybe_load_data(self):
        for dataset in self.datasets:
            dataset._maybe_load_data()
        self._min_hint_steps = max(dataset.min_hint_steps for dataset in self.datasets)

    def __getitem__(self, idx) -> Trajectory:
        self._maybe_load_data()
        return super().__getitem__(idx)
    
    def collate_fn(self, batch: List[Trajectory], device: Optional[torch.device]=torch.device('cpu')) -> Trajectory:
        self._maybe_load_data()
        min_hint_steps = self.min_hint_steps if self.uniform_hint_steps else 0

        # Even with zero, all the subsets have their own unique min_hint_steps
        batched_trajectory, steps = batch_trajectories(batch, 
                                                       min_hint_steps=min_hint_steps,
                                                       device=device)
        return batched_trajectory, steps

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def spec(self) -> Spec:
        return self._spec
    
    @property
    def subdataset_lens(self) -> List[int]:
        return [len(ds) for ds in self.datasets]
    

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


class CyclicAlgoTrajectoryDataset(Dataset):
    def __init__(self, algo_datasets: Union[MultiSizeAlgoTrajectoryDataset, List[MultiSizeAlgoTrajectoryDataset]]):
        super().__init__()
        if isinstance(algo_datasets, MultiSizeAlgoTrajectoryDataset):
            algo_datasets = [algo_datasets]

        assert all(isinstance(ds, MultiSizeAlgoTrajectoryDataset) for ds in algo_datasets), "All datasets must be MultiSizeAlgoTrajectoryDataset"
        dataset_lens = [ds.subdataset_lens for ds in algo_datasets]
        assert all_lists_equal(dataset_lens), "All MultiSizeAlgoTrajectoryDataset subdatasets must have the same length"

        datasets = cycle_flatten_list_of_lists([ds.datasets for ds in algo_datasets])
        self.datasets = datasets
        self.algo_to_ds_idx = {ds.name: i for i, ds in enumerate(self.datasets)}
        self._individual_ds_len = len(self.datasets[0])
        assert all(len(ds) == self._individual_ds_len for ds in self.datasets), "All subdatasets must have the same length"
        self._spec = {ds.name: ds.spec for ds in algo_datasets}

    @property
    def specs(self) -> Dict[str, Spec]:
        return self._spec

    def __len__(self):
        return self._individual_ds_len * len(self.datasets)
    
    def __getitem__(self, idx) -> Dict[str, Trajectory]:
        dataset_idx = idx // self._individual_ds_len
        offset = idx % self._individual_ds_len
        dataset = self.datasets[dataset_idx]
        dataset._maybe_load_data()
        return {dataset.name: dataset[offset]}
    
    def collate_fn(self, batch: List[Dict[str, Trajectory]], device: Optional[torch.device]=torch.device('cpu')) -> Dict[str, Trajectory]:
        dict_batch = defaultdict(list)
        for sample in batch:
            for name in sample.keys():
                dict_batch[name].append(sample[name])
        
        assert len(dict_batch) == 1, "Batch must contain only one algorithm"

        for name, trajectories in dict_batch.items():
            dict_batch[name] = self.datasets[self.algo_to_ds_idx[name]].collate_fn(trajectories, device)
        return dict_batch

    @property
    def subdataset_lens(self) -> List[int]:
        return [len(ds) for ds in self.datasets]
    

class StackedAlgoTrajectoryDataset(Dataset):
    """
    Concatenates multiple StackedAlgoTrajectoryDataset, where the Algorithms are different
    """
    def __init__(self, algo_datasets: Union[MultiSizeAlgoTrajectoryDataset, List[MultiSizeAlgoTrajectoryDataset]]):
        super().__init__()
        if isinstance(algo_datasets, MultiSizeAlgoTrajectoryDataset):
            algo_datasets = [algo_datasets]

        assert all(isinstance(ds, MultiSizeAlgoTrajectoryDataset) for ds in algo_datasets), "All datasets must be MultiSizeAlgoTrajectoryDataset"
        dataset_lens = [ds.subdataset_lens for ds in algo_datasets]
        assert all_lists_equal(dataset_lens), "All MultiSizeAlgoTrajectoryDataset subdatasets must have the same length"
        self.datasets = {ds.name: ds for ds in algo_datasets}
        self.algo_names = list(self.datasets.keys())
        assert len(self.algo_names) == len(algo_datasets), "All MultiSizeAlgoTrajectoryDatasets must have different algorithms"
        self._subset_lens = dataset_lens[0]
        

    @property
    def specs(self) -> Dict[str, Spec]:
        return {k: v.spec for k, v in self.datasets.items()}
    
    def __getitem__(self, idx) -> Dict[str, Trajectory]:
        self._maybe_load_data()
        result = {}
        for name, dataset in self.datasets.items():
            result[name] = dataset[idx]
        return result

    def _maybe_load_data(self):
        for dataset in self.datasets.values():
            dataset._maybe_load_data()

    def collate_fn(self, batch: List[Dict[str, Trajectory]], device: Optional[torch.device]=torch.device('cpu')) -> Dict[str, Trajectory]:
        dict_batch = defaultdict(list)
        algo_names = set(self.datasets.keys())
        for sample in batch:
            for name in algo_names:
                dict_batch[name].append(sample[name])
        
        for name, trajectories in dict_batch.items():
            dict_batch[name] = self.datasets[name].collate_fn(trajectories, device)
        return dict_batch
    
    @property
    def subdataset_lens(self) -> List[int]:
        return self._subset_lens



def get_dataset(algos: Union[Algorithms, List[Algorithms]],
                num_samples: int = 1000,
                trajectory_sizes: Union[int, List[int]] = [4, 7, 11, 13, 16],
                cache_dir: Optional[Union[str, Path]] = None,
                generate_on_the_fly: bool = True,
                uniform_hint_steps_per_algo: bool = True,
                stacked: bool = False,
                seed: int = 42,
                **algo_kwargs) -> StackedAlgoTrajectoryDataset:
    
    if isinstance(algos, Algorithms):
        algos = [algos]
    if isinstance(trajectory_sizes, int):
        trajectory_sizes = [trajectory_sizes]

    trajectory_sizes = sorted(trajectory_sizes, reverse=True) # Ensure that the largest trajectories are sampled first
    datasets = []
    for algo in algos:
        trajectory_sizes_algo = trajectory_sizes
        # As per the generalise algorithmic learner paper
        if algo in [Algorithms.naive_string_matcher, Algorithms.kmp_matcher]:
            max_length = max(trajectory_sizes)
            max_length = (max_length * 5) // 4
            trajectory_sizes_algo = [max_length]*len(trajectory_sizes)

        datasets.append(MultiSizeAlgoTrajectoryDataset(algo=algo,
                                            trajectory_sizes=trajectory_sizes_algo,
                                            num_samples=num_samples,
                                            cache_dir=cache_dir,
                                            generate_on_the_fly=generate_on_the_fly,
                                            uniform_hint_steps=uniform_hint_steps_per_algo,
                                            seed=seed,
                                            **algo_kwargs))    
    if stacked:
        return StackedAlgoTrajectoryDataset(datasets)
    else:
        return CyclicAlgoTrajectoryDataset(datasets)
    

def get_dataloader(dataset: Union[StackedAlgoTrajectoryDataset, CyclicAlgoTrajectoryDataset],
                   batch_size: int = 32,
                   shuffle: bool = False,
                   drop_last: bool = False,
                   device: torch.device=torch.device('cpu'),
                   num_workers: int = 0):
    assert isinstance(dataset, StackedAlgoTrajectoryDataset) or isinstance(dataset, CyclicAlgoTrajectoryDataset), "Dataset must be a StackedAlgoTrajectoryDataset or CyclicAlgoTrajectoryDataset"
    collate_fn = partial(dataset.collate_fn, device=device)
    return DataLoader(dataset, 
                      batch_sampler=RoundRobinBatchSampler(dataset_lengths=dataset.subdataset_lens,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle,
                                                          drop_last=drop_last),
                      collate_fn=collate_fn,
                      persistent_workers=num_workers > 0,
                      num_workers=num_workers)


if __name__ == "__main__":
    # Original stacked dataset and dataloader (for comparison or other tests)
    print("Testing Original Stacked Dataloader:")
    ds = get_dataset(algos=["bfs", "dfs"],
                    trajectory_sizes=[4, 16],
                    num_samples=100, # Reduced for faster testing
                    generate_on_the_fly=True,
                    stacked=False,
                    cache_dir=None)
    
    dataloader = get_dataloader(ds, 
                            batch_size=32,
                            shuffle=True, 
                            drop_last=False, 
                            device=torch.device('cpu'), 
                            num_workers=0)
    
    for batch in dataloader:
        print(batch)
        doSomething  = 1
 