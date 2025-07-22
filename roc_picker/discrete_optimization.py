"""
Various math algorithms for minimization or root finding for piecewise constant functions
that are only evaluated at discrete values.
"""

import collections.abc
import typing

import numpy as np

from .utilities import InspectableCache

def _is_strictly_less(a: float, b: float, atol: float, rtol: float) -> bool:
  """Returns True if a is strictly less than b, considering tolerances."""
  return b - a > atol + rtol * abs(b)

def _is_strictly_greater(a: float, b: float, atol: float, rtol: float) -> bool:
  """Returns True if a is strictly greater than b, considering tolerances."""
  return a - b > atol + rtol * abs(a)

def extract_inspectable_cache_values(
  func: typing.Callable,
  possible_values: np.ndarray
) -> dict[int, float]:
  """Return a dict mapping index → cached value from an InspectableCache-decorated function."""
  if not isinstance(func, InspectableCache):
    return {}
  cache = func.cache  # Safe if you followed earlier protocol + cast

  output = {}
  for args, value in cache.items():
    if not isinstance(args, tuple) or len(args) != 1:
      continue
    (x,) = args
    matches = np.nonzero(possible_values == x)[0]
    if len(matches) == 1:
      output[int(matches[0])] = value
  return output


def smart_bisect(start, end, evaluated):
  """
  Bisect the range [start, end] to find the closest value to target
  that has been evaluated in the `evaluated` list.
  """
  if start >= end - 1:
    raise ValueError(f"Invalid range: start={start}, end={end}")
  candidates = [i for i in evaluated if start < i < end]
  target = (start + end) // 2
  if not candidates:
    return target
  return min(candidates, key=lambda i: abs(i - target))

def smart_double_bisect(left, mid1, mid2, right, evaluated):
  """
  Bisect either [left, mid1] or [mid2, right] to find the closest value
  to the midpoints that has been evaluated in the `evaluated` list.
  If no such value exists, bisect the smaller of the two ranges.
  Returns the new mid1 and mid2 (one of which will be unchanged).
  """
  if not (left < mid1 < mid2 < right) or (left == mid1-1 and mid2 == right-1):
    raise ValueError(f"Invalid range: left={left}, mid1={mid1}, mid2={mid2}, right={right}")
  cand1 = (left + mid1) // 2
  cand2 = (mid2 + right) // 2

  choices = []
  for i in evaluated:
    if left < i < mid1:
      choices.append((abs(i - cand1), 'mid1', i))
    elif mid2 < i < right:
      choices.append((abs(i - cand2), 'mid2', i))

  if choices:
    _, which, idx = min(choices)
    return (idx, mid2) if which == 'mid1' else (mid1, idx)

  if (mid1 - left) > (right - mid2):
    return ((left + mid1) // 2, mid2)
  return (mid1, (mid2 + right) // 2)


def smart_trisect(left, right, evaluated):
  """
  Trisect the range [left, right] to find two points that are closest
  to the thirds of the range, using the evaluated points.
  """
  if left >= right - 2:
    raise ValueError(f"Invalid range: left={left}, right={right}")
  span = right - left
  default_mid1 = left + span // 3
  default_mid2 = right - span // 3

  known = [i for i in evaluated if left < i < right]
  if not known:
    return default_mid1, default_mid2

  def dist(i):
    return min(abs(i - default_mid1), abs(i - default_mid2))
  best = min(known, key=dist)
  if abs(best - default_mid1) <= abs(best - default_mid2):
    mid1 = best
    mid2 = smart_bisect(mid1, right, evaluated)
  else:
    mid2 = best
    mid1 = smart_bisect(left, mid2, evaluated)
  return mid1, mid2

def binary_search_sign_change(
  objective_function: collections.abc.Callable[[float], float],
  probs: np.ndarray,
  lo: int,
  hi: int,
  verbose: bool = False,
) -> float:
  """Binary search for first sign change across adjacent values."""
  evaluated = extract_inspectable_cache_values(objective_function, probs)

  def eval_or_get(i: int) -> float:
    if i not in evaluated:
      evaluated[i] = objective_function(probs[i])
    return evaluated[i]

  v_lo = eval_or_get(lo)
  v_hi = eval_or_get(hi)

  if v_lo * v_hi > 0:
    raise ValueError(f"No sign change found between indices {lo} and {hi}")

  if verbose:
    print("=================")
    print(lo, probs[lo], v_lo)
    print(hi, probs[hi], v_hi)

  while hi - lo > 1:
    mid = smart_bisect(lo, hi, evaluated)
    v_mid = eval_or_get(mid)

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
  atol: float = 1e-8,
  rtol: float = 0,
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

  evaluated = extract_inspectable_cache_values(objective_function, possible_values)
  evaluated.update({left: v_left, right: v_right})

  while right - left > 3:
    mid1, mid2 = smart_trisect(left, right, evaluated)

    for mid in (mid1, mid2):
      if mid not in evaluated:
        evaluated[mid] = objective_function(possible_values[mid])

    p_mid1 = possible_values[mid1]
    p_mid2 = possible_values[mid2]
    v_mid1 = evaluated[mid1]
    v_mid2 = evaluated[mid2]

    if verbose:
      print("--------------------")
      print(f"{left:3d} {p_left:6.3f} {v_left:15.9g}")
      print(f"{mid1:3d} {p_mid1:6.3f} {v_mid1:15.9g}")
      print(f"{mid2:3d} {p_mid2:6.3f} {v_mid2:15.9g}")
      print(f"{right:3d} {p_right:6.3f} {v_right:15.9g}")

    if _is_strictly_less(v_left, v_mid1, atol, rtol):
      right = mid1
      p_right = p_mid1
      v_right = v_mid1
      continue
    if _is_strictly_less(v_right, v_mid2, atol, rtol):
      left = mid2
      p_left = p_mid2
      v_left = v_mid2
      continue

    while (
      np.isclose(v_mid1, v_mid2, atol=atol, rtol=rtol)
        and (mid1 > left + 1 or mid2 < right - 1)
    ):
      if verbose:
        print("  --------")
        print("  Adjusting mid1 and mid2 due to equal values")
        print(f"  {left:3d} {p_left:6.3f} {v_left:15.9g}")
        print(f"  {mid1:3d} {p_mid1:6.3f} {v_mid1:15.9g}")
        print(f"  {mid2:3d} {p_mid2:6.3f} {v_mid2:15.9g}")
        print(f"  {right:3d} {p_right:6.3f} {v_right:15.9g}")

      new_mid1, new_mid2 = smart_double_bisect(left, mid1, mid2, right, evaluated)
      if new_mid1 == mid1 and new_mid2 == mid2:
        break
      mid1, mid2 = new_mid1, new_mid2

      for mid in (mid1, mid2):
        if mid not in evaluated:
          evaluated[mid] = objective_function(possible_values[mid])

      p_mid1 = possible_values[mid1]
      p_mid2 = possible_values[mid2]
      v_mid1 = evaluated[mid1]
      v_mid2 = evaluated[mid2]

    # Use robust comparison
    if _is_strictly_greater(max(v_mid1, v_mid2), max(v_left, v_right), atol, rtol):
      raise ValueError(
        "The probability doesn't have a single minimum:\n"
        f"left  ={left:12d}, mid1  ={mid1:12d}, "
        f"mid2  ={mid2:12d}, right  ={right:12d}\n"
        f"p_left={p_left:12.3f}, p_mid1={p_mid1:12.3f}, "
        f"p_mid2={p_mid2:12.3f}, p_right={p_right:12.3f}\n"
        f"v_left={v_left:12.6g}, v_mid1={v_mid1:12.6g}, "
        f"v_mid2={v_mid2:12.6g}, v_right={v_right:12.6g}\n"
      )

    if _is_strictly_less(v_mid1, v_mid2, atol, rtol):
      right = mid2
      p_right = p_mid2
      v_right = v_mid2
    elif _is_strictly_less(v_mid2, v_mid1, atol, rtol):
      left = mid1
      p_left = p_mid1
      v_left = v_mid1
    else: # v_mid1 isclose v_mid2
      if _is_strictly_greater(v_left, v_mid2, atol, rtol) or _is_strictly_greater(v_mid2, v_right, atol, rtol):
        left = mid1
        p_left = p_mid1
        v_left = v_mid1
      elif _is_strictly_less(v_mid1, v_right, atol, rtol) or _is_strictly_less(v_left, v_mid1, atol, rtol):
        right = mid2
        p_right = p_mid2
        v_right = v_mid2
      elif np.isclose(v_left, v_right, atol=atol, rtol=rtol): # Use np.isclose for final check of equality
        assert np.isclose(v_mid1, v_mid2, atol=atol, rtol=rtol) and \
               np.isclose(v_mid2, v_left, atol=atol, rtol=rtol) and \
               np.isclose(v_left, v_right, atol=atol, rtol=rtol)
        assert mid1 == left + 1 and mid2 == right - 1
        left = mid1
        p_left = p_mid1
        v_left = v_mid1
        right = mid2
        p_right = p_mid2
        v_right = v_mid2
      else:
        raise AssertionError(
          "Unexpected case where v_mid1 == v_mid2 and neither is less than the endpoints.\n"
          f"p_left={p_left:6.3f}, p_mid1={p_mid1:6.3f}, "
          f"p_mid2={p_mid2:6.3f}, p_right={p_right:6.3f}\n"
          f"v_left={v_left:9.3g}, v_mid1={v_mid1:9.3g}, "
          f"v_mid2={v_mid2:9.3g}, v_right={v_right:9.3g}\n"
        )

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
