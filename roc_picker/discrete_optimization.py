"""
Various math algorithms for minimization or root finding for piecewise constant functions
that are only evaluated at discrete values.
"""

import collections.abc

import numpy as np

def binary_search_sign_change(
  objective_function: collections.abc.Callable[[float], float],
  probs: np.ndarray,
  lo: int,
  hi: int,
  verbose: bool = False,
) -> float:
  """Binary search for first sign change across adjacent values."""
  if objective_function(probs[lo]) * objective_function(probs[hi]) > 0:
    raise ValueError(f"No sign change found between indices {lo} and {hi}")
  v_hi = objective_function(probs[hi])
  v_lo = objective_function(probs[lo])
  if verbose:
    print("=================")
    print(lo, probs[lo], v_lo)
    print(hi, probs[hi], v_hi)
  while hi - lo > 1:
    mid = (lo + hi) // 2
    v_mid = objective_function(probs[mid])
    if verbose:
      print(mid, probs[mid], v_mid)
    if v_mid * v_hi <= 0:
      lo = mid
      v_lo = v_mid
    elif v_mid * v_lo <= 0:
      hi = mid
      v_hi = v_mid
    else:
      raise ValueError(f"No sign change found between indices {lo} and {hi}")
  assert (v_lo <= 0) + (v_hi <= 0) == 1, (
    f"Expected one of v_lo or v_hi to be <= 0, got "
    f"v_lo={v_lo}, v_hi={v_hi} for indices {lo} and {hi}"
  )
  if v_hi <= 0:
    if verbose:
      print(f"Returning {probs[hi]} at index {hi} with v_hi={v_hi}")
    return probs[hi]
  if v_lo <= 0:
    if verbose:
      print(f"Returning {probs[lo]} at index {lo} with v_lo={v_lo}")
    return probs[lo]
  raise ValueError(f"No sign change found between indices {lo} and {hi}")

def minimize_discrete_single_minimum( #pylint: disable=too-many-locals, too-many-branches, too-many-statements
  objective_function: collections.abc.Callable[[float], float],
  possible_values: np.ndarray,
  verbose: bool = False,
):
  """
  Minimize a function that is only evaluated at discrete values
  The function should be piecewise constant, and should have
  a single minimum range (several consecutive inputs can have
  the same output, but there shouldn't be any other local minima)
  """
  left = 0
  right = len(possible_values) - 1
  p_left = possible_values[left]
  p_right = possible_values[right]
  v_left = objective_function(p_left)
  v_right = objective_function(p_right)
  while right - left > 3:
    third = (right - left) // 3
    mid1 = left + third
    mid2 = right - third
    p_mid1 = possible_values[mid1]
    p_mid2 = possible_values[mid2]
    v_mid1 = objective_function(p_mid1)
    v_mid2 = objective_function(p_mid2)
    while np.isclose(v_mid1, v_mid2) and (mid1 > left + 1 or mid2 < right - 1):
      if (mid1 - left) > (right - mid2):
        #mid1 is further from the end, so move it closer
        mid1 = (mid1 + left) // 2
      else:
        #mid2 is further from the end, so move it closer
        mid2 = (mid2 + right) // 2
      p_mid1 = possible_values[mid1]
      p_mid2 = possible_values[mid2]
      v_mid1 = objective_function(p_mid1)
      v_mid2 = objective_function(p_mid2)
    if verbose:
      print("--------------------")
      print(f"{left:3d} {p_left:6.3f} {v_left:15.9g}")
      print(f"{mid1:3d} {p_mid1:6.3f} {v_mid1:15.9g}")
      print(f"{mid2:3d} {p_mid2:6.3f} {v_mid2:15.9g}")
      print(f"{right:3d} {p_right:6.3f} {v_right:15.9g}")
    if not max(v_mid1, v_mid2) <= max(v_left, v_right):
      raise ValueError(
        "The probability doesn't have a single minimum:\n"
        f"left  ={left:12d}, mid1  ={mid1:12d}, "
        f"mid2  ={mid2:12d}, right  ={right:12d}\n"
        f"p_left={p_left:12.3f}, p_mid1={p_mid1:12.3f}, "
        f"p_mid2={p_mid2:12.3f}, p_right={p_right:12.3f}\n"
        f"v_left={v_left:12.6g}, v_mid1={v_mid1:12.6g}, "
        f"v_mid2={v_mid2:12.6g}, v_right={v_right:12.6g}\n"
      )
    if v_mid1 < v_mid2:
      right = mid2
      p_right = p_mid2
      v_right = v_mid2
    elif v_mid2 < v_mid1:
      left = mid1
      p_left = p_mid1
      v_left = v_mid1
    else:
      if v_left > v_mid2 or v_mid2 > v_right:
        left = mid1
        p_left = p_mid1
        v_left = v_mid1
      elif v_mid1 < v_right or v_left < v_mid1:
        right = mid2
        p_right = p_mid2
        v_right = v_mid2
      elif v_left == v_right:
        assert v_mid1 == v_mid2 == v_left == v_right
        assert mid1 == left + 1 and mid2 == right - 1
        left = mid1
        p_left = p_mid1
        v_left = v_mid1
        right = mid2
        p_right = p_mid2
        v_right = v_mid2
      else:
        # This should not happen, as we already checked that v_left != v_right
        raise AssertionError(
          "Unexpected case where v_mid1 == v_mid2 and neither is less than the endpoints.\n"
          f"p_left={p_left:6.3f}, p_mid1={p_mid1:6.3f}, "
          f"p_mid2={p_mid2:6.3f}, p_right={p_right:6.3f}\n"
          f"v_left={v_left:9.3g}, v_mid1={v_mid1:9.3g}, "
          f"v_mid2={v_mid2:9.3g}, v_right={v_right:9.3g}\n"
        )

  # Evaluate final narrowed range to find the best
  candidates = possible_values[left:right+1]
  values = [objective_function(p) for p in candidates]
  i_min = int(np.argmin(values))
  if verbose:
    print("Final candidates:")
    for i, (p, v) in enumerate(zip(candidates, values, strict=True)):
      print(f"{i + left:3d} {p:6.3f} {v:9.5g}")
    print("Winner:")
    print(f"{i_min + left:3d} {candidates[i_min]:6.3f} {values[i_min]:9.5g}")
  return candidates[i_min], values[i_min]
