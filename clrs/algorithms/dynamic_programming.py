"""Dynamic programming algorithm generators.

Currently implements the following:
- Matrix-chain multiplication
- Longest common subsequence
- Optimal binary search tree (Aho et al., 1974)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""

from typing import Tuple
import numpy as np

from ..specs import Stage
from ..probing import ProbesDict, Array, array, strings_id, strings_pair, strings_pos, array_cat, strings_pred, strings_pair_cat

Out = Tuple[Array, ProbesDict]


def matrix_chain_order(probes: ProbesDict, p: Array) -> Out:
  """Matrix-chain multiplication."""

  A_pos = np.arange(p.shape[0])

  probes.push(
      Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A_pos.shape[0],
          'p': np.copy(p)
      })

  m = np.zeros((p.shape[0], p.shape[0]))
  s = np.zeros((p.shape[0], p.shape[0]))
  msk = np.zeros((p.shape[0], p.shape[0]))
  for i in range(1, p.shape[0]):
    m[i, i] = 0
    msk[i, i] = 1
  while True:
    prev_m = np.copy(m)
    prev_msk = np.copy(msk)
    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'm': np.copy(prev_m),
            's_h': np.copy(s),
            'msk': np.copy(msk)
        })
    for i in range(1, p.shape[0]):
      for j in range(i + 1, p.shape[0]):
        flag = prev_msk[i, j]
        for k in range(i, j):
          if prev_msk[i, k] == 1 and prev_msk[k + 1, j] == 1:
            msk[i, j] = 1
            q = prev_m[i, k] + prev_m[k + 1, j] + p[i - 1] * p[k] * p[j]
            if flag == 0 or q < m[i, j]:
              m[i, j] = q
              s[i, j] = k
              flag = 1
    if np.all(prev_m == m):
      break

  probes.push(
      Stage.OUTPUT,
      next_probe={'s': np.copy(s)})
  probes.finalize()

  return s[1:, 1:], probes


def lcs_length(probes: ProbesDict, x: Array, y: Array) -> Out:
  """Longest common subsequence."""

  x_pos = np.arange(x.shape[0])
  y_pos = np.arange(y.shape[0])
  b = np.zeros((x.shape[0], y.shape[0]))
  c = np.zeros((x.shape[0], y.shape[0]))

  probes.push(
      Stage.INPUT,
      next_probe={
          'string': strings_id(x_pos, y_pos),
          'pos': strings_pos(x_pos, y_pos),
          'key': array_cat(np.concatenate([np.copy(x), np.copy(y)]), 4)
      })

  for i in range(x.shape[0]):
    if x[i] == y[0]:
      c[i, 0] = 1
      b[i, 0] = 0
    elif i > 0 and c[i - 1, 0] == 1:
      c[i, 0] = 1
      b[i, 0] = 1
    else:
      c[i, 0] = 0
      b[i, 0] = 1
  for j in range(y.shape[0]):
    if x[0] == y[j]:
      c[0, j] = 1
      b[0, j] = 0
    elif j > 0 and c[0, j - 1] == 1:
      c[0, j] = 1
      b[0, j] = 2
    else:
      c[0, j] = 0
      b[0, j] = 1

  while True:
    prev_c = np.copy(c)

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': strings_pred(x_pos, y_pos),
            'b_h': strings_pair_cat(np.copy(b), 3),
            'c': strings_pair(prev_c)
        })

    for i in range(1, x.shape[0]):
      for j in range(1, y.shape[0]):
        if x[i] == y[j]:
          c[i, j] = prev_c[i - 1, j - 1] + 1
          b[i, j] = 0
        elif prev_c[i - 1, j] >= prev_c[i, j - 1]:
          c[i, j] = prev_c[i - 1, j]
          b[i, j] = 1
        else:
          c[i, j] = prev_c[i, j - 1]
          b[i, j] = 2
    if np.all(prev_c == c):
      break

  probes.push(
      Stage.OUTPUT,
      next_probe={'b': strings_pair_cat(np.copy(b), 3)})
  probes.finalize()

  return b, probes


def optimal_bst(probes: ProbesDict, p: Array, q: Array) -> Out:
  """Optimal binary search tree (Aho et al., 1974)."""

  A_pos = np.arange(q.shape[0])
  A_pos = np.arange(q.shape[0])
  p_cpy = np.zeros(q.shape[0])
  p_cpy[:-1] = np.copy(p)

  probes.push(
      Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / q.shape[0],
          'p': np.copy(p_cpy),
          'q': np.copy(q)
      })

  e = np.zeros((q.shape[0], q.shape[0]))
  w = np.zeros((q.shape[0], q.shape[0]))
  root = np.zeros((q.shape[0], q.shape[0]))
  msks = np.zeros((q.shape[0], q.shape[0]))

  for i in range(q.shape[0]):
    e[i, i] = q[i]
    w[i, i] = q[i]
    msks[i, i] = 1

  probes.push(
      Stage.HINT,
      next_probe={
          'pred_h': array(np.copy(A_pos)),
          'root_h': np.copy(root),
          'e': np.copy(e),
          'w': np.copy(w),
          'msk': np.copy(msks)
      })

  for l in range(1, p.shape[0] + 1):
    for i in range(p.shape[0] - l + 1):
      j = i + l
      e[i, j] = 1e9
      w[i, j] = w[i, j - 1] + p[j - 1] + q[j]
      for r in range(i, j):
        t = e[i, r] + e[r + 1, j] + w[i, j]
        if t < e[i, j]:
          e[i, j] = t
          root[i, j] = r
      msks[i, j] = 1
    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'root_h': np.copy(root),
            'e': np.copy(e),
            'w': np.copy(w),
            'msk': np.copy(msks)
        })

  probes.push(Stage.OUTPUT, next_probe={'root': np.copy(root)})
  probes.finalize()

  return root, probes
