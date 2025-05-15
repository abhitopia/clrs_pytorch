"""Sorting algorithm generators.

Currently implements the following:
- Insertion sort
- Bubble sort
- Heapsort (Williams, 1964)
- Quicksort (Hoare, 1962)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""


from typing import Tuple
import numpy as np

from ..specs import Stage
from ..probing import ProbesDict, Array, array, heap, mask_one


Out = Tuple[Array, ProbesDict]


def insertion_sort(probes: ProbesDict, A: Array) -> Out:
  """Insertion sort."""

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
          'i': mask_one(0, A.shape[0]),
          'j': mask_one(0, A.shape[0])
      })

  for j in range(1, A.shape[0]):
    key = A[j]
    # Insert A[j] into the sorted sequence A[1 .. j - 1]
    i = j - 1
    while i >= 0 and A[i] > key:
      A[i + 1] = A[i]
      A_pos[i + 1] = A_pos[i]
      i -= 1
    A[i + 1] = key
    stor_pos = A_pos[i + 1]
    A_pos[i + 1] = j

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'i': mask_one(stor_pos, np.copy(A.shape[0])),
            'j': mask_one(j, np.copy(A.shape[0]))
        })

  probes.push(
      Stage.OUTPUT,
      next_probe={'pred': array(np.copy(A_pos))})

  probes.finalize()
  return A, probes


def bubble_sort(probes: ProbesDict, A: Array) -> Out:
  """Bubble sort."""

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
          'i': mask_one(0, A.shape[0]),
          'j': mask_one(0, A.shape[0])
      })

  for i in range(A.shape[0] - 1):
    for j in reversed(range(i + 1, A.shape[0])):
      if A[j] < A[j - 1]:
        A[j], A[j - 1] = A[j - 1], A[j]
        A_pos[j], A_pos[j - 1] = A_pos[j - 1], A_pos[j]

      probes.push(
          Stage.HINT,
          next_probe={
              'pred_h': array(np.copy(A_pos)),
              'i': mask_one(A_pos[i], np.copy(A.shape[0])),
              'j': mask_one(A_pos[j], np.copy(A.shape[0]))
          })

  probes.push(
      Stage.OUTPUT,
      next_probe={'pred': array(np.copy(A_pos))},
  )

  probes.finalize()

  return A, probes


def heapsort(probes: ProbesDict, A: Array) -> Out:
  """Heapsort (Williams, 1964)."""

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
          'parent': heap(np.copy(A_pos), A.shape[0]),
          'i': mask_one(A.shape[0] - 1, A.shape[0]),
          'j': mask_one(A.shape[0] - 1, A.shape[0]),
          'largest': mask_one(A.shape[0] - 1, A.shape[0]),
          'heap_size': mask_one(A.shape[0] - 1, A.shape[0]),
          'phase': mask_one(0, 3)
      })

  def max_heapify(A, i, heap_size, ind, phase):
    l = 2 * i + 1
    r = 2 * i + 2
    if l < heap_size and A[l] > A[i]:
      largest = l
    else:
      largest = i
    if r < heap_size and A[r] > A[largest]:
      largest = r
    if largest != i:
      A[i], A[largest] = A[largest], A[i]
      A_pos[i], A_pos[largest] = A_pos[largest], A_pos[i]

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'parent': heap(np.copy(A_pos), heap_size),
            'i': mask_one(A_pos[ind], A.shape[0]),
            'j': mask_one(A_pos[i], A.shape[0]),
            'largest': mask_one(A_pos[largest], A.shape[0]),
            'heap_size': mask_one(A_pos[heap_size - 1], A.shape[0]),
            'phase': mask_one(phase, 3)
        })

    if largest != i:
      max_heapify(A, largest, heap_size, ind, phase)

  def build_max_heap(A):
    for i in reversed(range(A.shape[0])):
      max_heapify(A, i, A.shape[0], i, 0)

  build_max_heap(A)
  heap_size = A.shape[0]
  for i in reversed(range(1, A.shape[0])):
    A[0], A[i] = A[i], A[0]
    A_pos[0], A_pos[i] = A_pos[i], A_pos[0]

    heap_size -= 1

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'parent': heap(np.copy(A_pos), heap_size),
            'i': mask_one(A_pos[0], A.shape[0]),
            'j': mask_one(A_pos[i], A.shape[0]),
            'largest': mask_one(0, A.shape[0]),  # Consider masking
            'heap_size': mask_one(A_pos[heap_size - 1], A.shape[0]),
            'phase': mask_one(1, 3)
        })

    max_heapify(A, 0, heap_size, i, 2)  # reduce heap_size!

  probes.push(
      Stage.OUTPUT,
      next_probe={'pred': array(np.copy(A_pos))},
  )

  probes.finalize()

  return A, probes


def quicksort(probes: ProbesDict, A: Array, A_pos=None, p=None, r=None, *, is_initial_call: bool = True) -> Out:
  """Quicksort (Hoare, 1962)."""

  def partition(A, A_pos, p, r, probes):
    x = A[r]
    i = p - 1
    for j in range(p, r):
      if A[j] <= x:
        i += 1
        A[i], A[j] = A[j], A[i]
        A_pos[i], A_pos[j] = A_pos[j], A_pos[i]

      probes.push(
          Stage.HINT,
          next_probe={
              'pred_h': array(np.copy(A_pos)),
              'p': mask_one(A_pos[p], A.shape[0]),
              'r': mask_one(A_pos[r], A.shape[0]),
              'i': mask_one(A_pos[i + 1], A.shape[0]),
              'j': mask_one(A_pos[j], A.shape[0])
          })

    A[i + 1], A[r] = A[r], A[i + 1]
    A_pos[i + 1], A_pos[r] = A_pos[r], A_pos[i + 1]

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'p': mask_one(A_pos[p], A.shape[0]),
            'r': mask_one(A_pos[r], A.shape[0]),
            'i': mask_one(A_pos[i + 1], A.shape[0]),
            'j': mask_one(A_pos[r], A.shape[0])
        })

    return i + 1

  if A_pos is None:
    A_pos = np.arange(A.shape[0])
  if p is None:
    p = 0
  if r is None:
    r = len(A) - 1
  if is_initial_call:
    probes.push(
        Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'key': np.copy(A)
        })

  if p < r:
    q = partition(A, A_pos, p, r, probes)
    quicksort(probes, A, A_pos, p, q - 1, is_initial_call=False)
    quicksort(probes, A, A_pos, q + 1, r, is_initial_call=False)

  if p == 0 and r == len(A) - 1:
    probes.push(
        Stage.OUTPUT,
        next_probe={'pred': array(np.copy(A_pos))},
    )
    probes.finalize()

  return A, probes
