import copy
import itertools
from typing import Any, List, Optional, Tuple
import numpy as np
from .probing import Array


def random_sequence(rng: np.random.RandomState, length, low=0.0, high=1.0):
    """Random sequence."""
    return rng.uniform(low=low, high=high, size=(length,))

def random_string(rng: np.random.RandomState, length, chars=4):
    """Random string."""
    return rng.randint(0, high=chars, size=(length,))

def random_er_graph(rng: np.random.RandomState, nb_nodes, p=0.5, directed=False, acyclic=False,
                    weighted=False, low=0.0, high=1.0):
    """Random Erdos-Renyi graph."""

    mat = rng.binomial(1, p, size=(nb_nodes, nb_nodes))
    if not directed:
        mat *= np.transpose(mat)
    elif acyclic:
        mat = np.triu(mat, k=1)
        p = rng.permutation(nb_nodes)  # To allow nontrivial solutions
        mat = mat[p, :][:, p]
    if weighted:
        weights = rng.uniform(low=low, high=high, size=(nb_nodes, nb_nodes))
        if not directed:
            weights *= np.transpose(weights)
            weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
        mat = mat.astype(float) * weights
    return mat

def random_bipartite_graph(rng: np.random.RandomState, n, m, p=0.25):
    """Random bipartite graph-based flow network."""
    nb_nodes = n + m + 2
    s = 0
    t = n + m + 1
    mat = np.zeros((nb_nodes, nb_nodes))
    mat[s, 1:n+1] = 1.0  # supersource
    mat[n+1:n+m+1, t] = 1.0  # supersink
    mat[1:n+1, n+1:n+m+1] = rng.binomial(1, p, size=(n, m))
    return mat

def random_community_graph(rng: np.random.RandomState, nb_nodes, k=4, p=0.5, eps=0.01,
                              directed=False, acyclic=False, weighted=False,
                              low=0.0, high=1.0):
    """Random perturbed k-community graph."""
    mat = np.zeros((nb_nodes, nb_nodes))
    if k > nb_nodes:
      raise ValueError(f'Cannot generate graph of too many ({k}) communities.')
    los, his = [], []
    lo = 0
    for i in range(k):
      if i == k - 1:
        hi = nb_nodes
      else:
        hi = lo + nb_nodes // k
      mat[lo:hi, lo:hi] = random_er_graph(rng=rng,
          nb_nodes=hi - lo, p=p, directed=directed,
          acyclic=acyclic, weighted=weighted,
          low=low, high=high)
      los.append(lo)
      his.append(hi)
      lo = hi
    toggle = random_er_graph(rng=rng,
                             nb_nodes=nb_nodes, p=eps, directed=directed,
                             acyclic=acyclic, weighted=weighted,
                             low=low, high=high)

    # Prohibit closing new cycles
    for i in range(k):
      for j in range(i):
        toggle[los[i]:his[i], los[j]:his[j]] *= 0

    mat = np.where(toggle > 0.0, (1.0 - (mat > 0.0)) * toggle, mat)
    p = rng.permutation(nb_nodes)  # To allow nontrivial solutions
    mat = mat[p, :][:, p]
    return mat

def _is_float_array(data: Any) -> bool:
  """Checks if the given data is a float numpy array."""
  if isinstance(data, np.ndarray):
    return issubclass(data.dtype.type, np.floating)
  return False

def _trunc_array(data: Any, truncate_decimals: Optional[int] = None) -> List[Array]:
    """Truncates the data if needed."""
    data = copy.deepcopy(data)

    if truncate_decimals is None:
      return data

    for index in range(len(data)):
      input_data = data[index]
      if not (_is_float_array(input_data) or isinstance(input_data, float)):
        continue

      data[index] = np.trunc(input_data * 10**truncate_decimals) / (
          10**truncate_decimals
      )

      if isinstance(input_data, float):
        data[index] = float(data[index])

    return data

def bfs_sampler(
    rng: np.random.RandomState,
    length: int,
    p: Tuple[float, ...] = (0.5,),
) -> Tuple[np.ndarray, int]:
    graph = random_er_graph(rng=rng,
                            nb_nodes=length, p=rng.choice(p),
                            directed=False, acyclic=False, weighted=False)
    source_node = rng.choice(length)
    return graph, source_node

def dfs_sampler(
    rng: np.random.RandomState,
    length: int,
    p: Tuple[float, ...] = (0.5,),
):
    graph = random_er_graph(rng=rng,
                            nb_nodes=length, p=rng.choice(p),
                            directed=True, acyclic=False, weighted=False)
    return [graph]

def topo_sampler(
    rng: np.random.RandomState,
    length: int,
    p: Tuple[float, ...] = (0.5,),
):
    graph = random_er_graph(rng=rng,
                            nb_nodes=length, p=rng.choice(p),
                            directed=True, acyclic=True, weighted=False)
    return [graph]

def articulation_sampler(
    rng: np.random.RandomState,
    length: int,
    p: Tuple[float, ...] = (0.2,),
):
    graph = random_er_graph(rng=rng,
                            nb_nodes=length, p=rng.choice(p),
                            directed=False, acyclic=False, weighted=False)
    return [graph]

def mst_sampler(
    rng: np.random.RandomState,
    length: int,
    p: Tuple[float, ...] = (0.2,),  # lower p to account for class imbalance
    low: float = 0.,
    high: float = 1.,
):
    graph = random_er_graph(rng=rng,
                            nb_nodes=length,
                            p=rng.choice(p),
                            directed=False,
                            acyclic=False,
                            weighted=True,
                            low=low,
        high=high)
    return [graph]

def bellman_ford_sampler(
      rng: np.random.RandomState,
      length: int,
      p: Tuple[float, ...] = (0.5,),
      low: float = 0.,
      high: float = 1.,
  ):
    graph = random_er_graph(rng=rng,
                            nb_nodes=length,
                            p=rng.choice(p),
                            directed=False,
                            acyclic=False,
                            weighted=True,
                            low=low,
        high=high)
    source_node = rng.choice(length)
    return [graph, source_node]

def dag_path_sampler(
    rng: np.random.RandomState,
    length: int,
    p: Tuple[float, ...] = (0.5,),
    low: float = 0.,
    high: float = 1.,
):
    graph = random_er_graph(rng=rng,
                            nb_nodes=length,
                            p=rng.choice(p),
                            directed=True,
                            acyclic=True,
                            weighted=True,
                            low=low,
                            high=high)
    source_node = rng.choice(length)
    return [graph, source_node]

def floyd_warshall_sampler(
    rng: np.random.RandomState,
    length: int,
    p: Tuple[float, ...] = (0.5,),
    low: float = 0.,
    high: float = 1.,
):
    graph = random_er_graph(rng=rng,
                            nb_nodes=length,
                            p=rng.choice(p),
                            directed=False,
                            acyclic=False,
                            weighted=True,
                            low=low,
        high=high)
    return [graph]

def scc_sampler(
    rng: np.random.RandomState,
    length: int,
    k: int = 4,
    p: Tuple[float, ...] = (0.5,),
    eps: float = 0.01,
):
    graph = random_community_graph(rng=rng,
        nb_nodes=length, k=k, p=rng.choice(p), eps=eps,
        directed=True, acyclic=False, weighted=False)
    return [graph]

def bipartite_sampler(
    rng: np.random.RandomState,
    length: int,
    length_2: Optional[int] = None,
    p: Tuple[float, ...] = (0.3,),
):
    if length_2 is None:
        # Assume provided length is total length.
        length_2 = length // 2
        length -= length_2
    graph = random_bipartite_graph(rng=rng, n=length, m=length_2,
                                            p=rng.choice(p))
    return [graph, length, length_2, 0, length + length_2 + 1]

# Divide and Conquer
def max_subarray_sampler(
    rng: np.random.RandomState,
    length: int,
    low: float = -1.,
    high: float = 1.,
):
    arr = rng.uniform(low=low, high=high, size=length)
    return [arr]

def max_subarray_sampler(
    rng: np.random.RandomState,
    length: int,
    low: float = -1.,
    high: float = 1.,
):
    arr = random_sequence(rng=rng, length=length, low=low, high=high)
    return [arr]


# Greedy
def activity_selector_sampler(
    rng: np.random.RandomState,
    length: int,
    low: float = 0.,
    high: float = 1.,
):
    arr_1 = random_sequence(rng=rng, length=length, low=low, high=high)
    arr_2 = random_sequence(rng=rng, length=length, low=low, high=high)
    return [np.minimum(arr_1, arr_2), np.maximum(arr_1, arr_2)]

def task_scheduling_sampler(
    rng: np.random.RandomState,
    length: int,
    max_deadline: Optional[int] = None,
    low: float = 0.,
    high: float = 1.,
):
    if max_deadline is None:
      max_deadline = length
    d = random_string(rng=rng, length=length, chars=max_deadline) + 1
    w = random_sequence(rng=rng, length=length, low=low, high=high)
    return [d, w]

# Sorting and Search
def sorting_sampler(
    rng: np.random.RandomState,
    length: int,
    low: float = 0.,
    high: float = 1.,
):
    arr = random_sequence(rng=rng, length=length, low=low, high=high)
    return [arr]

def search_sampler(
    rng: np.random.RandomState,
    length: int,
    low: float = 0.,
    high: float = 1.,
):
    arr = random_sequence(rng=rng, length=length, low=low, high=high)
    arr.sort()
    x = rng.uniform(low=low, high=high)
    return [x, arr]


# Dynamic Programming
def lcs_sampler(
    rng: np.random.RandomState,
    length: int,
    length_2: Optional[int] = None,
    chars: int = 4,
):
    if length_2 is None:
        # Assume provided length is total length.
        length_2 = length // 2
        length -= length_2
    a = random_string(rng=rng, length=length, chars=chars)
    b = random_string(rng=rng, length=length_2, chars=chars)
    return [a, b]

def optimal_bst_sampler(
    rng: np.random.RandomState,
    length: int
):

    tot_length = length + (length + 1)
    arr = random_sequence(rng=rng, length=tot_length, low=0.0, high=1.0)
    arr /= np.sum(arr)
    p = arr[:length]
    q = arr[length:]
    return [p, q]

# Geometry
def segments_sampler(rng: np.random.RandomState, length: int, low: float = 0., high: float = 1.):
    del length  # There are exactly four endpoints.

    # Quick CCW check (ignoring collinearity) for rejection sampling
    def ccw(x_a, y_a, x_b, y_b, x_c, y_c):
      return (y_c - y_a) * (x_b - x_a) > (y_b - y_a) * (x_c - x_a)
    def intersect(xs, ys):
      return ccw(xs[0], ys[0], xs[2], ys[2], xs[3], ys[3]) != ccw(
          xs[1], ys[1], xs[2], ys[2], xs[3], ys[3]) and ccw(
              xs[0], ys[0], xs[1], ys[1], xs[2], ys[2]) != ccw(
                  xs[0], ys[0], xs[1], ys[1], xs[3], ys[3])

    # Decide (with uniform probability) should this sample intersect
    coin_flip = rng.binomial(1, 0.5)

    xs = random_sequence(rng=rng, length=4, low=low, high=high)
    ys = random_sequence(rng=rng, length=4, low=low, high=high)

    while intersect(xs, ys) != coin_flip:
      xs = random_sequence(rng=rng, length=4, low=low, high=high)
      ys = random_sequence(rng=rng, length=4, low=low, high=high)

    return [xs, ys]

def _cross2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """Computes the cross product of two 2D vectors.

  Args:
    x: The first 2D vector.
    y: The second 2D vector.

  Returns:
    The cross product of the two vectors.
  """
  return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

def _is_collinear(
    point_1: np.ndarray,
    point_2: np.ndarray,
    point_3: np.ndarray,
    eps: float,
) -> bool:
  """Checks if three points are collinear.

  Args:
    point_1: The first point.
    point_2: The second point.
    point_3: The third point.
    eps: The tolerance for collinearity.

  Returns:
    True if the three points are collinear, False otherwise.

  Raises:
    ValueError: If any of the points is not a 2D vector.
  """
  for point in [point_1, point_2, point_3]:
    if point.shape != (2,):
      raise ValueError(f'Point {point} is not a 2D vector.')

  # Vectors from p1
  v_1 = point_2 - point_1
  v_2 = point_3 - point_1

  cross_val = _cross2d(v_1, v_2)

  return bool(abs(cross_val) < eps)

def convex_hull_sampler(
    rng: np.random.RandomState,
    length: int,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    radius: float = 2.0,
    collinearity_resampling_attempts: int = 1000,
    collineararity_eps: float = 1e-12,
    truncate_decimals: Optional[int] = None,
):
    """Samples a convex hull of points over a disk of radius r.

    Args:
      length: The number of points to sample.
      origin_x: The x-coordinate of the origin of the disk.
      origin_y: The y-coordinate of the origin of the disk.
      radius: The radius of the disk.
      collinearity_resampling_attempts: The number of times to resample if
        collinear points are found.
      collineararity_eps: The tolerance for collinearity.

    Returns:
      A list of the sampled points.

    Raises:
      RuntimeError: If it could not sample stable points within the specified
        number of attempts.
    """
    for _ in range(collinearity_resampling_attempts):
      thetas = random_sequence(
          rng=rng,
          length=length,
          low=0.0,
          high=2.0 * np.pi,
      )
      rs = radius * np.sqrt(
          random_sequence(
              rng=rng,
              length=length,
              low=0.0,
              high=1.0
          )
      )

      xs = rs * np.cos(thetas) + origin_x
      ys = rs * np.sin(thetas) + origin_y

      # Sampler._make_batch may do truncation of the input data after
      # calling _sample_data.
      # Truncation can lead to collinearity of points, which in turn leads to
      # numerous correct traces in the Graham scan algorithm. To prevent this,
      # we check for collinearity and resample if collinear points are found.

      # NOTE: This seems to be only needed when CLRS text is generated.
      xs = _trunc_array(xs, truncate_decimals=truncate_decimals)
      ys = _trunc_array(ys, truncate_decimals=truncate_decimals)

      collinear_found = False
      points = np.stack([xs, ys], axis=1)
      for point_1, point_2, point_3 in itertools.combinations(points, 3):
        if _is_collinear(point_1, point_2, point_3, collineararity_eps):
          collinear_found = True
          break

      if collinear_found:
        continue

      return [xs, ys]

    raise RuntimeError(
        f'Could not sample {length} stable points within {10000} tries.'
    )

# Strings
def matcher_sampler(
    rng: np.random.RandomState,
    length: int,  # length of haystack + needle, i.e., total number of nodes
    length_needle: Optional[int] = None,
    chars: int = 4,
):
    """
    length_needle:
        If None, the needle length is set to "negative" half the length of the haystack.
        If 0, the needle length is 1 if the haystack length is less than 5,
        otherwise it is one fifth of the haystack length.
        If negative, the needle length is sampled uniformly from [1, 1 - length_needle).
    """
    
    if length_needle is None:
      length_needle = (length // 2) - 1 # floor of half the length
      length_needle = -length_needle

    if length_needle == 0:
      if length < 5:
        length_needle = 1
      else:
        length_needle = length // 5
    elif length_needle < 0:  # randomize needle length
      length_needle = rng.randint(1, high=1 - length_needle)

    length_haystack = length - length_needle
    needle = random_string(rng=rng, length=length_needle, chars=chars)
    haystack = random_string(rng=rng, length=length_haystack, chars=chars)
    embed_pos = rng.choice(length_haystack - length_needle)
    haystack[embed_pos:embed_pos + length_needle] = needle
    return [haystack, needle]



SAMPLER_REGISTRY = {
    # Graphs
    'dfs': dfs_sampler,
    'bfs': bfs_sampler,
    'topological_sort': topo_sampler,
    'strongly_connected_components': scc_sampler,
    'articulation_points': articulation_sampler,
    'bridges': articulation_sampler,
    'mst_kruskal': mst_sampler,
    'mst_prim': bellman_ford_sampler,
    'bellman_ford': bellman_ford_sampler,
    'dag_shortest_paths': dag_path_sampler,
    'dijkstra': bellman_ford_sampler,
    'floyd_warshall': floyd_warshall_sampler,
    'bipartite_matching': bipartite_sampler,

    # Divide and Conquer
    'find_maximum_subarray': max_subarray_sampler,
    'find_maximum_subarray_kadane': max_subarray_sampler,

    # Greedy
    'activity_selector': activity_selector_sampler,
    'task_scheduling': task_scheduling_sampler,

    # Searching
    'minimum': sorting_sampler,
    'quickselect': sorting_sampler,
    'binary_search': search_sampler,

    # Sorting
    'insertion_sort': sorting_sampler,
    'bubble_sort': sorting_sampler,
    'heapsort': sorting_sampler,
    'quicksort': sorting_sampler,
    'quickselect': sorting_sampler,

    # Dynamic Programming
    'matrix_chain_order': sorting_sampler,
    'lcs_length': lcs_sampler,
    'optimal_bst': optimal_bst_sampler,

    # Geometry
    'segments_intersect': segments_sampler,
    'graham_scan': convex_hull_sampler,
    'jarvis_march': convex_hull_sampler,

    # Strings
    'naive_string_matcher': matcher_sampler,
    'kmp_matcher': matcher_sampler,
}