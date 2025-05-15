"""Divide and conquer algorithm generators.

Currently implements the following:
- Maximum subarray
- Kadane's variant of Maximum subarray (Bentley, 1984)

See "Introduction to Algorithms" 3ed (CLRS3) for more information.

"""

from typing import Any, Union
import numpy as np

from ..specs import Stage
from ..probing import ProbesDict, Array, mask_one, array


Numeric = Union[int, float]
Out = Any


def find_maximum_subarray(
    probes: ProbesDict,
    A: Array,
    A_pos=None,
    low=None,
    high=None,
    *,  # Make subsequent arguments keyword-only
    is_initial_call: bool = True,
) -> Out:
  """Maximum subarray."""

  def find_max_crossing_subarray(A, A_pos, low, mid, high, left_ctx, right_ctx,
                                 probes):
    (left_low, left_high, l_ctx_sum) = left_ctx
    (right_low, right_high, r_ctx_sum) = right_ctx
    left_sum = A[mid] - 0.1
    sum_ = 0

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'low': mask_one(low, A.shape[0]),
            'high': mask_one(high, A.shape[0]),
            'mid': mask_one(mid, A.shape[0]),
            'left_low': mask_one(left_low, A.shape[0]),
            'left_high': mask_one(left_high, A.shape[0]),
            'left_sum': l_ctx_sum,
            'right_low': mask_one(right_low, A.shape[0]),
            'right_high': mask_one(right_high, A.shape[0]),
            'right_sum': r_ctx_sum,
            'cross_low': mask_one(mid, A.shape[0]),
            'cross_high': mask_one(mid + 1, A.shape[0]),
            'cross_sum': A[mid] + A[mid + 1] - 0.2,
            'ret_low': mask_one(low, A.shape[0]),
            'ret_high': mask_one(high, A.shape[0]),
            'ret_sum': 0.0,
            'i': mask_one(mid, A.shape[0]),
            'j': mask_one(mid + 1, A.shape[0]),
            'sum': 0.0,
            'left_x_sum': A[mid] - 0.1,
            'right_x_sum': A[mid + 1] - 0.1,
            'phase': mask_one(1, 3)
        })

    for i in range(mid, low - 1, -1):
      sum_ += A[i]
      if sum_ > left_sum:
        left_sum = sum_
        max_left = i

      probes.push(
          Stage.HINT,
          next_probe={
              'pred_h': array(np.copy(A_pos)),
              'low': mask_one(low, A.shape[0]),
              'high': mask_one(high, A.shape[0]),
              'mid': mask_one(mid, A.shape[0]),
              'left_low': mask_one(left_low, A.shape[0]),
              'left_high': mask_one(left_high, A.shape[0]),
              'left_sum': l_ctx_sum,
              'right_low': mask_one(right_low, A.shape[0]),
              'right_high': mask_one(right_high, A.shape[0]),
              'right_sum': r_ctx_sum,
              'cross_low': mask_one(max_left, A.shape[0]),
              'cross_high': mask_one(mid + 1, A.shape[0]),
              'cross_sum': left_sum + A[mid + 1] - 0.1,
              'ret_low': mask_one(low, A.shape[0]),
              'ret_high': mask_one(high, A.shape[0]),
              'ret_sum': 0.0,
              'i': mask_one(i, A.shape[0]),
              'j': mask_one(mid + 1, A.shape[0]),
              'sum': sum_,
              'left_x_sum': left_sum,
              'right_x_sum': A[mid + 1] - 0.1,
              'phase': mask_one(1, 3)
          })

    right_sum = A[mid + 1] - 0.1
    sum_ = 0

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'low': mask_one(low, A.shape[0]),
            'high': mask_one(high, A.shape[0]),
            'mid': mask_one(mid, A.shape[0]),
            'left_low': mask_one(left_low, A.shape[0]),
            'left_high': mask_one(left_high, A.shape[0]),
            'left_sum': left_sum,
            'right_low': mask_one(right_low, A.shape[0]),
            'right_high': mask_one(right_high, A.shape[0]),
            'right_sum': right_sum,
            'cross_low': mask_one(max_left, A.shape[0]),
            'cross_high': mask_one(mid + 1, A.shape[0]),
            'cross_sum': left_sum + right_sum,
            'ret_low': mask_one(low, A.shape[0]),
            'ret_high': mask_one(high, A.shape[0]),
            'ret_sum': 0.0,
            'i': mask_one(i, A.shape[0]),
            'j': mask_one(mid + 1, A.shape[0]),
            'sum': 0.0,
            'left_x_sum': left_sum,
            'right_x_sum': A[mid + 1] - 0.1,
            'phase': mask_one(2, 3)
        })

    for j in range(mid + 1, high + 1):
      sum_ += A[j]
      if sum_ > right_sum:
        right_sum = sum_
        max_right = j

      probes.push(
          Stage.HINT,
          next_probe={
              'pred_h': array(np.copy(A_pos)),
              'low': mask_one(low, A.shape[0]),
              'high': mask_one(high, A.shape[0]),
              'mid': mask_one(mid, A.shape[0]),
              'left_low': mask_one(left_low, A.shape[0]),
              'left_high': mask_one(left_high, A.shape[0]),
              'left_sum': left_sum,
              'right_low': mask_one(right_low, A.shape[0]),
              'right_high': mask_one(right_high, A.shape[0]),
              'right_sum': right_sum,
              'cross_low': mask_one(max_left, A.shape[0]),
              'cross_high': mask_one(max_right, A.shape[0]),
              'cross_sum': left_sum + right_sum,
              'ret_low': mask_one(low, A.shape[0]),
              'ret_high': mask_one(high, A.shape[0]),
              'ret_sum': 0.0,
              'i': mask_one(i, A.shape[0]),
              'j': mask_one(j, A.shape[0]),
              'sum': sum_,
              'left_x_sum': left_sum,
              'right_x_sum': right_sum,
              'phase': mask_one(2, 3)
          })

    return (max_left, max_right, left_sum + right_sum), (sum_, left_sum,
                                                         right_sum)

  if A_pos is None:
    A_pos = np.arange(A.shape[0])
  if low is None:
    low = 0
  if high is None:
    high = A.shape[0] - 1

  # Only push input probes on the initial call
  if is_initial_call:
    probes.push(
        Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A_pos.shape[0],
            'key': np.copy(A)
        })

  mid = (low + high) // 2

  if high == low:
    if A.shape[0] == 1:
      probes.push(
          Stage.OUTPUT,
          next_probe={
              'start': mask_one(low, A.shape[0]),
              'end': mask_one(high, A.shape[0])
          })
      probes.finalize()
      return (low, high, A[low]), probes
    else:
      probes.push(
          Stage.HINT,
          next_probe={
              'pred_h': array(np.copy(A_pos)),
              'low': mask_one(low, A.shape[0]),
              'high': mask_one(high, A.shape[0]),
              'mid': mask_one(mid, A.shape[0]),
              'left_low': mask_one(low, A.shape[0]),
              'left_high': mask_one(high, A.shape[0]),
              'left_sum': 0.0,
              'right_low': mask_one(low, A.shape[0]),
              'right_high': mask_one(high, A.shape[0]),
              'right_sum': 0.0,
              'cross_low': mask_one(low, A.shape[0]),
              'cross_high': mask_one(high, A.shape[0]),
              'cross_sum': 0.0,
              'ret_low': mask_one(low, A.shape[0]),
              'ret_high': mask_one(high, A.shape[0]),
              'ret_sum': A[low],
              'i': mask_one(low, A.shape[0]),
              'j': mask_one(high, A.shape[0]),
              'sum': 0.0,
              'left_x_sum': A[low] - 0.1,
              'right_x_sum': A[high] - 0.1,
              'phase': mask_one(0, 3)
          })
      return (low, high, A[low])
  else:
    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'low': mask_one(low, A.shape[0]),
            'high': mask_one(high, A.shape[0]),
            'mid': mask_one(mid, A.shape[0]),
            'left_low': mask_one(low, A.shape[0]),
            'left_high': mask_one(mid, A.shape[0]),
            'left_sum': 0.0,
            'right_low': mask_one(mid + 1, A.shape[0]),
            'right_high': mask_one(high, A.shape[0]),
            'right_sum': 0.0,
            'cross_low': mask_one(mid, A.shape[0]),
            'cross_high': mask_one(mid + 1, A.shape[0]),
            'cross_sum': A[mid] + A[mid + 1] - 0.2,
            'ret_low': mask_one(low, A.shape[0]),
            'ret_high': mask_one(high, A.shape[0]),
            'ret_sum': 0.0,
            'i': mask_one(mid, A.shape[0]),
            'j': mask_one(mid + 1, A.shape[0]),
            'sum': 0.0,
            'left_x_sum': A[mid] - 0.1,
            'right_x_sum': A[mid + 1] - 0.1,
            'phase': mask_one(0, 3)
        })

    (left_low, left_high,  # pylint: disable=unbalanced-tuple-unpacking
     left_sum) = find_maximum_subarray(probes, A, A_pos, low, mid,
                                     is_initial_call=False)

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'low': mask_one(low, A.shape[0]),
            'high': mask_one(high, A.shape[0]),
            'mid': mask_one(mid, A.shape[0]),
            'left_low': mask_one(left_low, A.shape[0]),
            'left_high': mask_one(left_high, A.shape[0]),
            'left_sum': left_sum,
            'right_low': mask_one(mid + 1, A.shape[0]),
            'right_high': mask_one(high, A.shape[0]),
            'right_sum': 0.0,
            'cross_low': mask_one(mid, A.shape[0]),
            'cross_high': mask_one(mid + 1, A.shape[0]),
            'cross_sum': A[mid] + A[mid + 1] - 0.2,
            'ret_low': mask_one(low, A.shape[0]),
            'ret_high': mask_one(high, A.shape[0]),
            'ret_sum': 0.0,
            'i': mask_one(mid, A.shape[0]),
            'j': mask_one(mid + 1, A.shape[0]),
            'sum': 0.0,
            'left_x_sum': A[mid] - 0.1,
            'right_x_sum': A[mid + 1] - 0.1,
            'phase': mask_one(0, 3)
        })

    (right_low, right_high,  # pylint: disable=unbalanced-tuple-unpacking
     right_sum) = find_maximum_subarray(probes, A, A_pos, mid + 1, high,
                                      is_initial_call=False)

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'low': mask_one(low, A.shape[0]),
            'high': mask_one(high, A.shape[0]),
            'mid': mask_one(mid, A.shape[0]),
            'left_low': mask_one(left_low, A.shape[0]),
            'left_high': mask_one(left_high, A.shape[0]),
            'left_sum': left_sum,
            'right_low': mask_one(right_low, A.shape[0]),
            'right_high': mask_one(right_high, A.shape[0]),
            'right_sum': right_sum,
            'cross_low': mask_one(mid, A.shape[0]),
            'cross_high': mask_one(mid + 1, A.shape[0]),
            'cross_sum': A[mid] + A[mid + 1] - 0.2,
            'ret_low': mask_one(low, A.shape[0]),
            'ret_high': mask_one(high, A.shape[0]),
            'ret_sum': 0.0,
            'i': mask_one(mid, A.shape[0]),
            'j': mask_one(mid + 1, A.shape[0]),
            'sum': 0.0,
            'left_x_sum': A[mid] - 0.1,
            'right_x_sum': A[mid + 1] - 0.1,
            'phase': mask_one(0, 3)
        })

    (cross_low, cross_high,
     cross_sum), (x_sum, x_left, x_right) = find_max_crossing_subarray(
         A, A_pos, low, mid, high, (left_low, left_high, left_sum),
         (right_low, right_high, right_sum), probes)
    if left_sum >= right_sum and left_sum >= cross_sum:
      best = (left_low, left_high, left_sum)
    elif right_sum >= left_sum and right_sum >= cross_sum:
      best = (right_low, right_high, right_sum)
    else:
      best = (cross_low, cross_high, cross_sum)

    if low == 0 and high == A.shape[0] - 1:
      probes.push(
          Stage.OUTPUT,
          next_probe={
              'start': mask_one(best[0], A.shape[0]),
              'end': mask_one(best[1], A.shape[0])
          })
      probes.finalize()
      return best, probes
    else:
      probes.push(
          Stage.HINT,
          next_probe={
              'pred_h': array(np.copy(A_pos)),
              'low': mask_one(low, A.shape[0]),
              'high': mask_one(high, A.shape[0]),
              'mid': mask_one(mid, A.shape[0]),
              'left_low': mask_one(left_low, A.shape[0]),
              'left_high': mask_one(left_high, A.shape[0]),
              'left_sum': left_sum,
              'right_low': mask_one(right_low, A.shape[0]),
              'right_high': mask_one(right_high, A.shape[0]),
              'right_sum': right_sum,
              'cross_low': mask_one(cross_low, A.shape[0]),
              'cross_high': mask_one(cross_high, A.shape[0]),
              'cross_sum': cross_sum,
              'ret_low': mask_one(best[0], A.shape[0]),
              'ret_high': mask_one(best[1], A.shape[0]),
              'ret_sum': best[2],
              'i': mask_one(low, A.shape[0]),
              'j': mask_one(high, A.shape[0]),
              'sum': x_sum,
              'left_x_sum': x_left,
              'right_x_sum': x_right,
              'phase': mask_one(0, 3)
          })

      return best


def find_maximum_subarray_kadane(probes: ProbesDict, A: Array) -> Out:
  """Kadane's variant of Maximum subarray (Bentley, 1984)."""

  A_pos = np.arange(A.shape[0])

  probes.push(
      Stage.INPUT,
      next_probe={
          'pos': np.copy(A_pos) * 1.0 / A_pos.shape[0],
          'key': np.copy(A)
      })
  probes.push(
      Stage.HINT,
      next_probe={
          'pred_h': array(np.copy(A_pos)),
          'best_low': mask_one(0, A.shape[0]),
          'best_high': mask_one(0, A.shape[0]),
          'best_sum': A[0],
          'i': mask_one(0, A.shape[0]),
          'j': mask_one(0, A.shape[0]),
          'sum': A[0]
      })

  best_low = 0
  best_high = 0
  best_sum = A[0]
  i = 0
  sum_ = A[0]

  for j in range(1, A.shape[0]):
    x = A[j]
    if sum_ + x >= x:
      sum_ += x
    else:
      i = j
      sum_ = x
    if sum_ > best_sum:
      best_low = i
      best_high = j
      best_sum = sum_

    probes.push(
        Stage.HINT,
        next_probe={
            'pred_h': array(np.copy(A_pos)),
            'best_low': mask_one(best_low, A.shape[0]),
            'best_high': mask_one(best_high, A.shape[0]),
            'best_sum': best_sum,
            'i': mask_one(i, A.shape[0]),
            'j': mask_one(j, A.shape[0]),
            'sum': sum_
        })

  probes.push(
      Stage.OUTPUT,
      next_probe={
          'start': mask_one(best_low, A.shape[0]),
          'end': mask_one(best_high, A.shape[0])
      })

  probes.finalize()

  return (best_low, best_high, best_sum), probes
