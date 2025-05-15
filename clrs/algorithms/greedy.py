"""Greedy algorithm generators.

Currently implements the following:
- Activity selection (Gavril, 1972)
- Task scheduling (Lawler, 1985)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""

from typing import Tuple
from ..specs import Stage
from ..probing import ProbesDict, Array, array, mask_one
import numpy as np


Out = Tuple[Array, ProbesDict]


def activity_selector(probes: ProbesDict, s: Array, f: Array) -> Out:
  """Activity selection (Gavril, 1972)."""

  A_pos = np.arange(s.shape[0])

  probes.push(
      Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A_pos.shape[0],
          's': np.copy(s),
          'f': np.copy(f)
      })

  A = np.zeros(s.shape[0])

  probes.push(
      Stage.HINT,
      next_probe={
          'pred_h': array(np.copy(A_pos)),
          'selected_h': np.copy(A),
          'm': mask_one(0, A_pos.shape[0]),
          'k': mask_one(0, A_pos.shape[0])
      })

  ind = np.argsort(f)
  A[ind[0]] = 1
  k = ind[0]

  probes.push(
      Stage.HINT,
      next_probe={
          'pred_h': array(np.copy(A_pos)),
          'selected_h': np.copy(A),
          'm': mask_one(ind[0], A_pos.shape[0]),
          'k': mask_one(k, A_pos.shape[0])
      })

  for m in range(1, s.shape[0]):
    if s[ind[m]] >= f[k]:
      A[ind[m]] = 1
      k = ind[m]
    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'selected_h': np.copy(A),
            'm': mask_one(ind[m], A_pos.shape[0]),
            'k': mask_one(k, A_pos.shape[0])
        })

  probes.push(Stage.OUTPUT, next_probe={'selected': np.copy(A)})
  probes.finalize()

  return A, probes


def task_scheduling(probes: ProbesDict, d: Array, w: Array) -> Out:
  """Task scheduling (Lawler, 1985)."""

  A_pos = np.arange(d.shape[0])

  probes.push(
      Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A_pos.shape[0],
          'd': np.copy(d),
          'w': np.copy(w)
      })

  A = np.zeros(d.shape[0])

  probes.push(
      Stage.HINT,
      next_probe={
          'pred_h': array(np.copy(A_pos)),
          'selected_h': np.copy(A),
          'i': mask_one(0, A_pos.shape[0]),
          't': 0
      })

  ind = np.argsort(-w)
  A[ind[0]] = 1
  t = 1

  probes.push(
      Stage.HINT,
      next_probe={
          'pred_h': array(np.copy(A_pos)),
          'selected_h': np.copy(A),
          'i': mask_one(ind[0], A_pos.shape[0]),
          't': t
      })

  for i in range(1, d.shape[0]):
    if t < d[ind[i]]:
      A[ind[i]] = 1
      t += 1
    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'selected_h': np.copy(A),
            'i': mask_one(ind[i], A_pos.shape[0]),
            't': t
        })

  probes.push(Stage.OUTPUT, next_probe={'selected': np.copy(A)})
  probes.finalize()

  return A, probes
