"""Strings algorithm generators.

Currently implements the following:
- Naive string matching
- Knuth-Morris-Pratt string matching (Knuth et al., 1977)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""

from typing import Tuple
import numpy as np

from ..probing import Array, ProbesDict, Stage, strings_id, strings_pos, array_cat, mask_one, strings_pred, strings_pi

Out = Tuple[int, ProbesDict]

ALPHABET_SIZE = 4


def naive_string_matcher(probes: ProbesDict, T: Array, P: Array) -> Out:
  """Naive string matching."""

  assert T.ndim == 1
  assert P.ndim == 1

  T_pos = np.arange(T.shape[0])
  P_pos = np.arange(P.shape[0])

  probes.push(
      Stage.INPUT,
      next_probe={
          'string':
              strings_id(T_pos, P_pos),
          'pos':
              strings_pos(T_pos, P_pos),
          'key':
              array_cat(
                  np.concatenate([np.copy(T), np.copy(P)]), ALPHABET_SIZE),
      })

  s = 0
  while s <= T.shape[0] - P.shape[0]:
    i = s
    j = 0

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': strings_pred(T_pos, P_pos),
            's': mask_one(s, T.shape[0] + P.shape[0]),
            'i': mask_one(i, T.shape[0] + P.shape[0]),
            'j': mask_one(T.shape[0] + j, T.shape[0] + P.shape[0])
        })

    while True:
      if T[i] != P[j]:
        break
      elif j == P.shape[0] - 1:
        probes.push(
            Stage.OUTPUT,
            next_probe={'match': mask_one(s, T.shape[0] + P.shape[0])})
        probes.finalize()
        return s, probes
      else:
        i += 1
        j += 1
        probes.push(
            Stage.HINT,
            next_probe={
                'pred_h': strings_pred(T_pos, P_pos),
                's': mask_one(s, T.shape[0] + P.shape[0]),
                'i': mask_one(i, T.shape[0] + P.shape[0]),
                'j': mask_one(T.shape[0] + j, T.shape[0] + P.shape[0])
            })

    s += 1

  # By convention, set probe to head of needle if no match is found
  probes.push(
      Stage.OUTPUT,
      next_probe={
          'match': mask_one(T.shape[0], T.shape[0] + P.shape[0])
      })
  return T.shape[0], probes


def kmp_matcher(probes: ProbesDict, T: Array, P: Array) -> Out:
  """Knuth-Morris-Pratt string matching (Knuth et al., 1977)."""

  assert T.ndim == 1
  assert P.ndim == 1

  T_pos = np.arange(T.shape[0])
  P_pos = np.arange(P.shape[0])

  probes.push(
      Stage.INPUT,
      next_probe={
          'string':
              strings_id(T_pos, P_pos),
          'pos':
              strings_pos(T_pos, P_pos),
          'key':
              array_cat(
                  np.concatenate([np.copy(T), np.copy(P)]), ALPHABET_SIZE),
      })

  pi = np.arange(P.shape[0])
  is_reset = np.zeros(P.shape[0])

  k = 0
  k_reset = 1
  is_reset[0] = 1

  # Cover the edge case where |P| = 1, and the first half is not executed.
  delta = 1 if P.shape[0] > 1 else 0

  probes.push(
      Stage.HINT,
      next_probe={
          'pred_h': strings_pred(T_pos, P_pos),
          'pi': strings_pi(T_pos, P_pos, pi),
          'is_reset': np.concatenate(
              [np.zeros(T.shape[0]), np.copy(is_reset)]),
          'k': mask_one(T.shape[0], T.shape[0] + P.shape[0]),
          'k_reset': k_reset,
          'q': mask_one(T.shape[0] + delta, T.shape[0] + P.shape[0]),
          'q_reset': 1,
          's': mask_one(0, T.shape[0] + P.shape[0]),
          'i': mask_one(0, T.shape[0] + P.shape[0]),
          'phase': 0
      })

  for q in range(1, P.shape[0]):
    while k_reset == 0 and P[k + 1] != P[q]:
      if is_reset[k] == 1:
        k_reset = 1
        k = 0
      else:
        k = pi[k]
      probes.push(
          Stage.HINT,
          next_probe={
              'pred_h': strings_pred(T_pos, P_pos),
              'pi': strings_pi(T_pos, P_pos, pi),
              'is_reset': np.concatenate(
                  [np.zeros(T.shape[0]), np.copy(is_reset)]),
              'k': mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
              'k_reset': k_reset,
              'q': mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
              'q_reset': 1,
              's': mask_one(0, T.shape[0] + P.shape[0]),
              'i': mask_one(0, T.shape[0] + P.shape[0]),
              'phase': 0
          })
    if k_reset == 1:
      k_reset = 0
      k = -1
    if P[k + 1] == P[q]:
      k += 1
    if k == -1:
      k = 0
      k_reset = 1
      is_reset[q] = 1
    pi[q] = k
    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': strings_pred(T_pos, P_pos),
            'pi': strings_pi(T_pos, P_pos, pi),
            'is_reset': np.concatenate(
                [np.zeros(T.shape[0]), np.copy(is_reset)]),
            'k': mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
            'k_reset': k_reset,
            'q': mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
            'q_reset': 1,
            's': mask_one(0, T.shape[0] + P.shape[0]),
            'i': mask_one(0, T.shape[0] + P.shape[0]),
            'phase': 0
        })
  q = 0
  q_reset = 1
  s = 0
  for i in range(T.shape[0]):
    if i >= P.shape[0]:
      s += 1
    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': strings_pred(T_pos, P_pos),
            'pi': strings_pi(T_pos, P_pos, pi),
            'is_reset': np.concatenate(
                [np.zeros(T.shape[0]), np.copy(is_reset)]),
            'k': mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
            'k_reset': k_reset,
            'q': mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
            'q_reset': q_reset,
            's': mask_one(s, T.shape[0] + P.shape[0]),
            'i': mask_one(i, T.shape[0] + P.shape[0]),
            'phase': 1
        })
    while q_reset == 0 and P[q + 1] != T[i]:
      if is_reset[q] == 1:
        q = 0
        q_reset = 1
      else:
        q = pi[q]
      probes.push(
          Stage.HINT,
          next_probe={
              'pred_h': strings_pred(T_pos, P_pos),
              'pi': strings_pi(T_pos, P_pos, pi),
              'is_reset': np.concatenate(
                  [np.zeros(T.shape[0]), np.copy(is_reset)]),
              'k': mask_one(T.shape[0] + k, T.shape[0] + P.shape[0]),
              'k_reset': k_reset,
              'q': mask_one(T.shape[0] + q, T.shape[0] + P.shape[0]),
              'q_reset': q_reset,
              's': mask_one(s, T.shape[0] + P.shape[0]),
              'i': mask_one(i, T.shape[0] + P.shape[0]),
              'phase': 1
          })
    if q_reset == 1:
      q = -1
      q_reset = 0
    if P[q + 1] == T[i]:
      if q == P.shape[0] - 2:
        probes.push(
            Stage.OUTPUT,
            next_probe={'match': mask_one(s, T.shape[0] + P.shape[0])})
        probes.finalize()
        return s, probes
      q += 1
    if q == -1:
      q_reset = 1
      q = 0

  # By convention, set probe to head of needle if no match is found
  probes.push(
      Stage.OUTPUT,
      next_probe={
          'match': mask_one(T.shape[0], T.shape[0] + P.shape[0])
      })
  probes.finalize()

  return T.shape[0], probes
