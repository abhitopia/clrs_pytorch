from enum import Enum
import types
from typing import Any, Dict, Tuple, Union
from torch import Tensor
import numpy as np


class Stage(str, Enum):
  INPUT = 'input'
  OUTPUT = 'output'
  HINT = 'hint'


class Location(str, Enum):
  NODE = 'node'
  EDGE = 'edge'
  GRAPH = 'graph'


class Type(str, Enum):
  SCALAR = 'scalar'
  CATEGORICAL = 'categorical'
  MASK = 'mask'
  POINTER = 'pointer'

  # default to pointer, but can be converted to permutation pointer with mask depending on the enfoce_permutations flag
  POINTER_OR_PERMUTATION_WITH_MASK = 'pointer_or_permutation_with_mask'

  # Following is used only if POINTER_OR_PERMUTATION_WITH_MASK is converted to permutation pointer with mask
  PERMUTATION_POINTER = 'permutation_pointer'
  MASK_ONE = 'mask_one'

#   SOFT_POINTER = 'soft_pointer'                         # Not used in the code

Array = Union[Tensor, np.ndarray]
NumSteps = Union[Tensor, int]                # Int for a single trajectory, Tensor for batched trajectories
NumNodes = Union[Tensor, int]                # Int for a single trajectory, Tensor for batched trajectories
DataPoint = Dict[str, Array] # Could be input or output
# Features = List[Feature] # hints
Input = DataPoint
Hints = DataPoint
Output = DataPoint
Spec = Dict[str, Tuple[Stage, Location, Type, Dict[str, Any]]] # name -> (stage, location, type, metadata)
Trajectory = Dict[Stage, Union[Input, Hints, Output, DataPoint]] # stage -> (name -> array, num_steps)
Feature = Tuple[Trajectory, NumSteps, NumNodes]


class OutputClass(int, Enum):
  POSITIVE = 1
  NEGATIVE = 0
  MASKED = -1

class Algorithm(str, Enum):
    activity_selector = 'activity_selector'
    articulation_points = 'articulation_points'
    bellman_ford = 'bellman_ford'
    bfs = 'bfs'
    binary_search = 'binary_search'
    bipartite_matching = 'bipartite_matching'
    bridges = 'bridges'
    bubble_sort = 'bubble_sort'
    dag_shortest_paths = 'dag_shortest_paths'
    dfs = 'dfs'
    dijkstra = 'dijkstra'
    find_maximum_subarray = 'find_maximum_subarray'
    find_maximum_subarray_kadane = 'find_maximum_subarray_kadane'
    floyd_warshall = 'floyd_warshall'
    graham_scan = 'graham_scan'
    heapsort = 'heapsort'
    insertion_sort = 'insertion_sort'
    jarvis_march = 'jarvis_march'
    kmp_matcher = 'kmp_matcher'
    lcs_length = 'lcs_length'
    matrix_chain_order = 'matrix_chain_order'
    minimum = 'minimum'
    mst_kruskal = 'mst_kruskal'
    mst_prim = 'mst_prim'
    naive_string_matcher = 'naive_string_matcher'
    optimal_bst = 'optimal_bst'
    quickselect = 'quickselect'
    quicksort = 'quicksort'
    segments_intersect = 'segments_intersect'
    strongly_connected_components = 'strongly_connected_components'
    task_scheduling = 'task_scheduling'
    topological_sort = 'topological_sort'

CLRS30Algorithms = [algo for algo in Algorithm if algo not in [Algorithm.find_maximum_subarray, Algorithm.bipartite_matching]]


# These algorithms have a static hint called pred_h which 
# can be moved to the input
HAS_STATIC_HINT = set([
    Algorithm.binary_search,
    Algorithm.minimum,
    Algorithm.find_maximum_subarray,
    Algorithm.find_maximum_subarray_kadane,
    Algorithm.matrix_chain_order,
    Algorithm.lcs_length,
    Algorithm.optimal_bst,
    Algorithm.activity_selector,
    Algorithm.task_scheduling,
    Algorithm.naive_string_matcher,
    Algorithm.kmp_matcher,
    Algorithm.jarvis_march
])

RAW_SPECS = types.MappingProxyType({
    Algorithm.insertion_sort: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.POINTER_OR_PERMUTATION_WITH_MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.bubble_sort: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.POINTER_OR_PERMUTATION_WITH_MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.heapsort: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.POINTER_OR_PERMUTATION_WITH_MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'parent': (Stage.HINT, Location.NODE, Type.POINTER),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'largest': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'heap_size': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL),
    },
    Algorithm.quicksort: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'pred': (Stage.OUTPUT, Location.NODE, Type.POINTER_OR_PERMUTATION_WITH_MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'p': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'r': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.quickselect: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'median': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'p': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'r': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i_rank': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'target': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'pivot': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.minimum: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'min': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'min_h': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.binary_search: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'target': (Stage.INPUT, Location.GRAPH, Type.SCALAR),
        'return': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'mid': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.find_maximum_subarray: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'start': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'end': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'mid': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'left_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'left_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'left_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'right_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'right_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'right_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'cross_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'cross_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'cross_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'ret_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'ret_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'ret_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'left_x_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'right_x_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL),
    },
    Algorithm.find_maximum_subarray_kadane: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'start': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'end': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'best_low': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'best_high': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'best_sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'sum': (Stage.HINT, Location.GRAPH, Type.SCALAR),
    },
    Algorithm.matrix_chain_order: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'p': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.OUTPUT, Location.EDGE, Type.POINTER),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'm': (Stage.HINT, Location.EDGE, Type.SCALAR),
        's_h': (Stage.HINT, Location.EDGE, Type.POINTER),
        'msk': (Stage.HINT, Location.EDGE, Type.MASK),
    },
    Algorithm.lcs_length: {
        'string': (Stage.INPUT, Location.NODE, Type.MASK),
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.CATEGORICAL),
        'b': (Stage.OUTPUT, Location.EDGE, Type.CATEGORICAL),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'b_h': (Stage.HINT, Location.EDGE, Type.CATEGORICAL),
        'c': (Stage.HINT, Location.EDGE, Type.SCALAR),
    },
    Algorithm.optimal_bst: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'p': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'q': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'root': (Stage.OUTPUT, Location.EDGE, Type.POINTER),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'root_h': (Stage.HINT, Location.EDGE, Type.POINTER),
        'e': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'w': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'msk': (Stage.HINT, Location.EDGE, Type.MASK),
    },
    Algorithm.activity_selector: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'f': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'selected': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'selected_h': (Stage.HINT, Location.NODE, Type.MASK),
        'm': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.task_scheduling: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'd': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'w': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'selected': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'selected_h': (Stage.HINT, Location.NODE, Type.MASK),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        't': (Stage.HINT, Location.GRAPH, Type.SCALAR),
    },
    Algorithm.dfs: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR),
    },
    Algorithm.topological_sort: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'topo': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'topo_head': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'topo_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'topo_head_h': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.strongly_connected_components: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'scc_id': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'scc_id_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'A_t': (Stage.HINT, Location.EDGE, Type.MASK),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK),
    },
    Algorithm.articulation_points: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'is_cut': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'is_cut_h': (Stage.HINT, Location.NODE, Type.MASK),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        'low': (Stage.HINT, Location.NODE, Type.SCALAR),
        'child_cnt': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR),
    },
    Algorithm.bridges: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'is_bridge': (Stage.OUTPUT, Location.EDGE, Type.MASK),
        'is_bridge_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'f': (Stage.HINT, Location.NODE, Type.SCALAR),
        'low': (Stage.HINT, Location.NODE, Type.SCALAR),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'time': (Stage.HINT, Location.GRAPH, Type.SCALAR),
    },
    Algorithm.bfs: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'reach_h': (Stage.HINT, Location.NODE, Type.MASK),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
    },
    Algorithm.mst_kruskal: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'in_mst': (Stage.OUTPUT, Location.EDGE, Type.MASK),
        'in_mst_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'pi': (Stage.HINT, Location.NODE, Type.POINTER),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'root_u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'root_v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'mask_u': (Stage.HINT, Location.NODE, Type.MASK),
        'mask_v': (Stage.HINT, Location.NODE, Type.MASK),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL),
    },
    Algorithm.mst_prim: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'key': (Stage.HINT, Location.NODE, Type.SCALAR),
        'mark': (Stage.HINT, Location.NODE, Type.MASK),
        'in_queue': (Stage.HINT, Location.NODE, Type.MASK),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.bellman_ford: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'msk': (Stage.HINT, Location.NODE, Type.MASK),
    },
    Algorithm.dag_shortest_paths: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'mark': (Stage.HINT, Location.NODE, Type.MASK),
        'topo_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'topo_head_h': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'color': (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        's_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'v': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        's_last': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK),
    },
    Algorithm.dijkstra: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'mark': (Stage.HINT, Location.NODE, Type.MASK),
        'in_queue': (Stage.HINT, Location.NODE, Type.MASK),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.floyd_warshall: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'Pi': (Stage.OUTPUT, Location.EDGE, Type.POINTER),
        'Pi_h': (Stage.HINT, Location.EDGE, Type.POINTER),
        'D': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'msk': (Stage.HINT, Location.EDGE, Type.MASK),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.bipartite_matching: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        't': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'in_matching': (Stage.OUTPUT, Location.EDGE, Type.MASK),
        'in_matching_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'A_h': (Stage.HINT, Location.EDGE, Type.SCALAR),
        'adj_h': (Stage.HINT, Location.EDGE, Type.MASK),
        'd': (Stage.HINT, Location.NODE, Type.SCALAR),
        'msk': (Stage.HINT, Location.NODE, Type.MASK),
        'pi': (Stage.HINT, Location.NODE, Type.POINTER),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK),
    },
    Algorithm.naive_string_matcher: {
        'string': (Stage.INPUT, Location.NODE, Type.MASK),
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.CATEGORICAL),
        'match': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
    },
    Algorithm.kmp_matcher: {
        'string': (Stage.INPUT, Location.NODE, Type.MASK),
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'key': (Stage.INPUT, Location.NODE, Type.CATEGORICAL),
        'match': (Stage.OUTPUT, Location.NODE, Type.MASK_ONE),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'pi': (Stage.HINT, Location.NODE, Type.POINTER),
        'is_reset': (Stage.HINT, Location.NODE, Type.MASK),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'k_reset': (Stage.HINT, Location.GRAPH, Type.MASK),
        'q': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'q_reset': (Stage.HINT, Location.GRAPH, Type.MASK),
        's': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.MASK),
    },
    Algorithm.segments_intersect: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'x': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'y': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'intersect': (Stage.OUTPUT, Location.GRAPH, Type.MASK),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'j': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'k': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'dir': (Stage.HINT, Location.NODE, Type.SCALAR),
        'on_seg': (Stage.HINT, Location.NODE, Type.MASK),
    },
    Algorithm.graham_scan: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'x': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'y': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'in_hull': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'best': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'atans': (Stage.HINT, Location.NODE, Type.SCALAR),
        'in_hull_h': (Stage.HINT, Location.NODE, Type.MASK),
        'stack_prev': (Stage.HINT, Location.NODE, Type.POINTER),
        'last_stack': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL),
    },
    Algorithm.jarvis_march: {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'x': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'y': (Stage.INPUT, Location.NODE, Type.SCALAR),
        'in_hull': (Stage.OUTPUT, Location.NODE, Type.MASK),
        'pred_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'in_hull_h': (Stage.HINT, Location.NODE, Type.MASK),
        'best': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'last_point': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'endpoint': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'i': (Stage.HINT, Location.NODE, Type.MASK_ONE),
        'phase': (Stage.HINT, Location.GRAPH, Type.CATEGORICAL),
    },
})
