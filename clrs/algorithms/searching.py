"""Searching algorithm generators.

Currently implements the following:
- Minimum
- Binary search
- Quickselect (Hoare, 1961)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""

from typing import Tuple, Union
import numpy as np

from ..probing import ProbesDict, array, mask_one, Array
from ..specs import Stage
Numeric = Union[int, float]
Out = Tuple[int, ProbesDict]


def minimum(probes: ProbesDict, A: Array) -> Out:
  """Minimum."""

  A_pos = np.arange(A.shape[0])

  probes.push(
      Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A.shape[0],
          'key': np.copy(A)
      })

  probes.push(
      Stage.HINT,
      next_probe={
          'pred_h': array(np.copy(A_pos)),
          'min_h': mask_one(0, A.shape[0]),
          'i': mask_one(0, A.shape[0])
      })

  min_ = 0
  for i in range(1, A.shape[0]):
    if A[min_] > A[i]:
      min_ = i

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'min_h': mask_one(min_, A.shape[0]),
            'i': mask_one(i, A.shape[0])
        })

  probes.push(
      Stage.OUTPUT,
      next_probe={'min': mask_one(min_, A.shape[0])})

  probes.finalize()
  return min_, probes


def binary_search(probes: ProbesDict, x: Numeric, A: Array) -> Out:
  """Binary search."""

  T_pos = np.arange(A.shape[0])

  probes.push(
      Stage.INPUT,
      next_probe={
          'pos': np.copy(T_pos) * 1.0 / A.shape[0],
          'key': np.copy(A),
          'target': x
      })

  probes.push(
      Stage.HINT,
      next_probe={
          'pred_h': array(np.copy(T_pos)),
          'low': mask_one(0, A.shape[0]),
          'high': mask_one(A.shape[0] - 1, A.shape[0]),
          'mid': mask_one((A.shape[0] - 1) // 2, A.shape[0]),
      })

  low = 0
  high = A.shape[0] - 1  # make sure return is always in array
  while low < high:
    mid = (low + high) // 2
    if x <= A[mid]:
      high = mid
    else:
      low = mid + 1

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(T_pos)),
            'low': mask_one(low, A.shape[0]),
            'high': mask_one(high, A.shape[0]),
            'mid': mask_one((low + high) // 2, A.shape[0]),
        })

  probes.push(
      Stage.OUTPUT,
      next_probe={'return': mask_one(high, A.shape[0])})

  probes.finalize()

  return high, probes


def quickselect(
    probes: ProbesDict,
    A: Array,
    A_pos=None,
    p=None,
    r=None,
    i=None,
    *,  # Make subsequent arguments keyword-only
    is_initial_call: bool = True,
) -> Out:
  """Quickselect (Hoare, 1961)."""


  def partition(A, A_pos, p, r, target, probes):
    x = A[r]
    i = p - 1
    for j in range(p, r):
      if A[j] <= x:
        i += 1
        tmp = A[i]
        A[i] = A[j]
        A[j] = tmp
        tmp = A_pos[i]
        A_pos[i] = A_pos[j]
        A_pos[j] = tmp

      probes.push(
          Stage.HINT,
          next_probe={
              'pred_h': array(np.copy(A_pos)),
              'p': mask_one(A_pos[p], A.shape[0]),
              'r': mask_one(A_pos[r], A.shape[0]),
              'i': mask_one(A_pos[i + 1], A.shape[0]),
              'j': mask_one(A_pos[j], A.shape[0]),
              'i_rank': (i + 1) * 1.0 / A.shape[0],
              'target': target * 1.0 / A.shape[0],
              'pivot': mask_one(A_pos[r], A.shape[0]),
          },
      )

    tmp = A[i + 1]
    A[i + 1] = A[r]
    A[r] = tmp
    tmp = A_pos[i + 1]
    A_pos[i + 1] = A_pos[r]
    A_pos[r] = tmp

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'p': mask_one(A_pos[p], A.shape[0]),
            'r': mask_one(A_pos[r], A.shape[0]),
            'i': mask_one(A_pos[i + 1], A.shape[0]),
            'j': mask_one(A_pos[r], A.shape[0]),
            'i_rank': (i + 1 - p) * 1.0 / A.shape[0],
            'target': target * 1.0 / A.shape[0],
            'pivot': mask_one(A_pos[i + 1], A.shape[0]),
        },
    )

    return i + 1

  if A_pos is None:
    A_pos = np.arange(A.shape[0])
  if p is None:
    p = 0
  if r is None:
    r = len(A) - 1
  if i is None:
    i = len(A) // 2

  if is_initial_call:
    probes.push(
        Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'key': np.copy(A)
        })

  q = partition(A, A_pos, p, r, i, probes)
  k = q - p
  if i == k:
    probes.push(
        Stage.OUTPUT,
        next_probe={'median': mask_one(A_pos[q], A.shape[0])})
    probes.finalize()
    return A[q], probes
  elif i < k:
    return quickselect(probes, A, A_pos, p, q - 1, i, is_initial_call=False)
  else:
    return quickselect(probes, A, A_pos, q + 1, r, i - k - 1, is_initial_call=False)
