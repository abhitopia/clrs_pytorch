import abc
import copy
import hashlib
import inspect
import numpy as np
from typing import Callable, Tuple, Union
from .specs import RAW_SPECS, HAS_STATIC_HINT, Feature, Spec, CLRS30Algorithms, AlgorithmEnum
from .probing import ProbesDict
from .sampler import SAMPLER_REGISTRY
from . import algorithms


class Algorithm(abc.ABC):
    def __init__(self, name: Union[str, AlgorithmEnum], seed: int=42, 
                 randomize_pos: bool=True, 
                 move_predh_to_input: bool=True, 
                 enforce_permutations: bool=True, 
                 **sampler_kwargs):
        assert name in CLRS30Algorithms, f"Algorithm {name} not in {CLRS30Algorithms}"
        self._name = name
        self._runner: Callable = None
        self._seed = seed
        self._randomize_pos = randomize_pos
        self._move_predh_to_input = move_predh_to_input
        self._enforce_permutations = enforce_permutations
        self._spec = None

        self._rng = np.random.RandomState(seed)
        self._sampler = SAMPLER_REGISTRY[self.name]
        self._sampler_kwargs = self.clean_sampler_kwargs(sampler_kwargs)

    @property
    def name(self) -> str:
        return self._name

    @property
    def can_move_predh_to_input(self) -> bool:
        return self.name in HAS_STATIC_HINT

    @property
    def rng(self):
        return self._rng

    def reset(self):
        self._rng = np.random.RandomState(self._seed)

    def clean_sampler_kwargs(self, sampler_overrides: dict) -> dict:
        sampler_args = inspect.signature(self._sampler).parameters
        default_kwargs = copy.deepcopy(self.default_sampler_kwargs)
        default_kwargs.update(sampler_overrides)
        return {k: default_kwargs[k] for k in default_kwargs if k in sampler_args}

    @property
    def default_sampler_kwargs(self) -> dict:
        p = [0.1 + 0.1 * i for i in range(9)]
        if self.name in ['articulation_points',
                        'bridges',
                        'mst_kruskal', 
                        'bipartite_matching']:
            p = [prob / 2 for prob in p]
        return {
            'length': 16,
            'p': p, 
            'length_needle': None, # 0 < needle < floor(length / 2)
        }

    @property
    def runner(self) -> Callable:
        if self._runner is None:
            self._runner = getattr(algorithms, self.name)
        return self._runner

    @property
    def spec(self) -> Spec:
        if self._spec is None:
            _, spec = self.sample_feature(return_spec=True)
            self._spec = spec
        return self._spec

    def sample_feature(self, return_spec: bool = False) -> Union[Feature, Tuple[Feature, Spec]]:
        sample = self._sampler(self._rng, **self._sampler_kwargs)
        probes: ProbesDict = ProbesDict(RAW_SPECS[self.name])
        _, probes = self.runner(probes, *sample)        
        feature, spec = probes.to_feature(
            randomize_pos=self._randomize_pos,
            move_predh_to_input=self._move_predh_to_input and self.can_move_predh_to_input,
            enforce_permutations=self._enforce_permutations,
            rng=self._rng
        )

        if return_spec:
            return feature, spec
        return feature
    
    def unique_hash(self):
        """
        Returns a unique hash for the algorithm based on its name, seed, and sampler kwargs.
        """
        name = self.name
        seed = self._seed
        randomize_pos = self._randomize_pos
        move_predh_to_input = self._move_predh_to_input
        enforce_permutations = self._enforce_permutations
        hash_dict = {
            'name': name,
            'seed': seed,
            'randomize_pos': randomize_pos,
            'move_predh_to_input': move_predh_to_input,
            'enforce_permutations': enforce_permutations,
        }
        hash_dict.update(self._sampler_kwargs)

        sorted_hash_dict = sorted(hash_dict.items())
        return hashlib.md5(str(sorted_hash_dict).encode()).hexdigest()

