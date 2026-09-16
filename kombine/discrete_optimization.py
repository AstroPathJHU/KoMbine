"""
Various math algorithms for minimization or root finding for piecewise constant functions
that are only evaluated at discrete values.
"""

import collections.abc
import dataclasses
import time
import typing

import numpy as np
import scipy.optimize

from .utilities import InspectableCache

def _is_close(a: float, b: float, atol: float, rtol: float) -> bool:
  """Returns True if a and b are close, considering symmetric tolerances."""
  return abs(a - b) <= atol + rtol * max(abs(a), abs(b))

def _is_strictly_less(a: float, b: float, atol: float, rtol: float) -> bool:
  """Returns True if a is strictly less than b, beyond tolerance."""
  return (a < b) and not _is_close(a, b, atol, rtol)

def _is_strictly_greater(a: float, b: float, atol: float, rtol: float) -> bool:
  """Returns True if a is strictly greater than b, beyond tolerance."""
  return (a > b) and not _is_close(a, b, atol, rtol)

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

def binary_search_sign_change( #pylint: disable=too-many-arguments, too-many-branches
  objective_function: collections.abc.Callable[[float], float],
  probs: np.ndarray,
  lo: int,
  hi: int,
  *,
  verbose: bool = False,
  MIPGap: float | None = None,
  MIPGapAbs: float | None = None,
) -> float:
  """Binary search for first sign change across adjacent values.

  Parameters
  ----------
  objective_function : callable
      Function to evaluate at each probability value
  probs : np.ndarray
      Array of probability values to search over
  lo : int
      Starting index for the search
  hi : int
      Ending index for the search
  verbose : bool, default False
      If True, print detailed search progress
  MIPGap : float, optional
      Relative tolerance for probability convergence
  MIPGapAbs : float, optional
      Absolute tolerance for probability convergence

  Returns
  -------
  float
      Probability value where the sign change occurs

  Notes
  -----
  The search stops when either:
  1. Adjacent indices are reached (hi - lo <= 1), or
  2. The difference between probs[lo] and probs[hi] is within tolerance
  """
  # Set default tolerance values if not provided
  if MIPGapAbs is None:
    MIPGapAbs = 1e-7  # Default absolute tolerance
  if MIPGap is None:
    MIPGap = 1e-4  # Default relative tolerance

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

  while hi - lo > 1 and not _is_close(probs[lo], probs[hi], MIPGapAbs, MIPGap):
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
      _is_close(v_mid1, v_mid2, atol=atol, rtol=rtol) # Use _is_close here
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
    else: # v_mid1 is considered close to v_mid2
      assert _is_close(v_mid1, v_mid2, atol=atol, rtol=rtol) # Keep this assertion
      if (
        _is_strictly_greater(v_left, v_mid2, atol, rtol)
        or _is_strictly_greater(v_mid2, v_right, atol, rtol)
      ):
        left = mid1
        p_left = p_mid1
        v_left = v_mid1
      elif (
        _is_strictly_less(v_mid1, v_right, atol, rtol)
        or _is_strictly_less(v_left, v_mid1, atol, rtol)
      ):
        right = mid2
        p_right = p_mid2
        v_right = v_mid2
      elif (
        _is_close(v_left, v_right, atol=atol, rtol=rtol) # Use _is_close here
      ):
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


def _innermost_bracket(
  cache: dict[float, float],
  level: float,
  x_outer: float,
  x_inner: float,
) -> tuple[float, float]:
  """Return ``(last_inside, first_outside)`` for the MLE-connected crossing.

  ``first_outside`` is the cached outside point closest to ``x_inner``.
  ``last_inside`` is ``x_inner`` (or an inside point still closer to the MLE
  than that outside point).  Do not walk through other inside points: a
  MINLP profile can dip back below the cut in a disconnected valley.
  Endpoints must be in ``cache`` with ``f(x_outer) >= level >= f(x_inner)``.
  """
  lo = min(x_outer, x_inner)
  hi = max(x_outer, x_inner)
  outside = [x for x, fx in cache.items() if lo <= x <= hi and fx >= level]
  inside = [x for x, fx in cache.items() if lo <= x <= hi and fx < level]
  if not outside or not inside:
    raise ValueError(
      f"No cached sign change for level {level} on [{x_outer}, {x_inner}]"
    )
  if x_outer < x_inner:
    first_outside = max(outside)
    inner_side = [x for x in inside if x > first_outside]
    last_inside = max(inner_side) if inner_side else x_inner
  else:
    first_outside = min(outside)
    inner_side = [x for x in inside if x < first_outside]
    last_inside = min(inner_side) if inner_side else x_inner
  if last_inside == first_outside:
    raise ValueError(
      f"Cached bracket collapsed for level {level}: {last_inside}"
    )
  return last_inside, first_outside


def _adjacent_outside_bracket(  # pylint: disable=too-many-arguments, too-many-positional-arguments
  f_cached: collections.abc.Callable[[float], float],
  level: float,
  last_inside: float,
  first_outside: float,
  xtol: float,
  rtol: float,
) -> tuple[float, float]:
  """Double from ``last_inside`` toward ``first_outside`` until ``f >= level``.

  Returns a left-to-right pair for ``brentq``.
  """
  if _is_close(last_inside, first_outside, xtol, rtol):
    return min(last_inside, first_outside), max(last_inside, first_outside)
  direction = 1.0 if first_outside > last_inside else -1.0
  remaining = abs(first_outside - last_inside)
  step = min(
    max(
      xtol,
      rtol * max(abs(last_inside), abs(first_outside), 1e-12),
      0.05 * remaining,
    ),
    remaining,
  )
  x = last_inside
  while True:
    x_next = x + direction * step
    if (x_next - first_outside) * direction >= 0.0:
      x_next = first_outside
    if f_cached(x_next) >= level:
      return (x, x_next) if x < x_next else (x_next, x)
    if x_next == first_outside:
      return (x, x_next) if x < x_next else (x_next, x)
    x = x_next
    step = min(step * 2.0, abs(first_outside - x))
    if step <= 0.0:
      return (x, first_outside) if x < first_outside else (first_outside, x)


def cached_level_crossings(  # pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments
  func: collections.abc.Callable[[float], float],
  x_outer: float,
  x_inner: float,
  levels: collections.abc.Sequence[float],
  *,
  xtol: float = 1e-4,
  rtol: float = 1e-4,
) -> list[float]:
  """Find where a 1D profile crosses several levels, sharing evaluations.

  ``func`` should be nonnegative and smaller toward ``x_inner`` (e.g.
  ``2NLL(p) - 2NLL_min`` on one side of the MLE).  ``levels`` are the
  target heights (e.g. χ² cuts).  After every new evaluation, each
  remaining cut walks cached points from ``x_inner`` outward and takes
  the first exit (closest outside point, last inside point), then doubles
  from that inside point until the cut is exceeded so a thin near-MLE
  crossing is not skipped.

  If ``func(x_outer) <= level``, the whole interval is inside that cut and
  the crossing is ``x_outer``.  Remaining cuts are solved widest-bracket
  first (typically the outermost level).
  """
  if not levels:
    return []
  if x_outer == x_inner:
    return [x_inner] * len(levels)

  cache: dict[float, float] = {}

  def f_cached(x: float) -> float:
    if x not in cache:
      cache[x] = float(func(x))
    return cache[x]

  f_cached(x_outer)
  f_cached(x_inner)

  results: list[float | None] = [None] * len(levels)
  pending = list(range(len(levels)))

  def remaining_width(index: int) -> float:
    level = levels[index]
    if f_cached(x_outer) <= level:
      return -1.0
    if f_cached(x_inner) >= level:
      return -1.0
    left, right = _innermost_bracket(cache, level, x_outer, x_inner)
    return abs(right - left)

  while pending:
    pending.sort(key=lambda i: (remaining_width(i), levels[i]), reverse=True)
    index = pending.pop(0)
    level = levels[index]
    f_outer = f_cached(x_outer)
    f_inner = f_cached(x_inner)
    if f_outer <= level:
      results[index] = x_outer
      continue
    if f_inner >= level:
      results[index] = x_inner
      continue
    last_inside, first_outside = _innermost_bracket(cache, level, x_outer, x_inner)
    left, right = _adjacent_outside_bracket(
      f_cached, level, last_inside, first_outside, xtol, rtol,
    )
    if left == right or _is_close(left, right, xtol, rtol):
      closer = left if abs(f_cached(left) - level) <= abs(f_cached(right) - level) else right
      results[index] = closer
      continue
    g_left = f_cached(left) - level
    g_right = f_cached(right) - level
    if g_left == 0.0:
      results[index] = left
      continue
    if g_right == 0.0:
      results[index] = right
      continue
    results[index] = typing.cast(
      float,
      scipy.optimize.brentq(
        lambda x, _level=level: f_cached(x) - _level,
        left,
        right,
        xtol=xtol,
        rtol=np.float64(rtol),
        full_output=False,
      ),
    )

  crossings: list[float] = []
  for x in results:
    if x is None:
      raise RuntimeError("cached_level_crossings left a level unsolved")
    crossings.append(x)
  return crossings


SignOracleStatus = typing.Literal["inside", "outside", "unknown"]
SignOracleResult = SignOracleStatus | tuple[SignOracleStatus, float]


@dataclasses.dataclass
class OracleCostTracker:  # pylint: disable=too-many-instance-attributes
  """Online estimate of cost-biased bisection probe fraction ``r``.

  Prior ``r₀=0.5`` (midpoint). Updates from outcome counts and unitless
  wall-time ratios; clamps with uncertainty so outliers cannot poison ``r``.
  """

  r_prior: float = 0.5
  kappa: float = 4.0
  z: float = 1.0
  epsilon_floor: float = 0.05
  winsor_cap: float = 4.0
  ema_alpha: float = 0.3
  w_threshold: float = 2.0
  alpha: float = dataclasses.field(init=False)
  beta: float = dataclasses.field(init=False)
  n_in: int = 0
  n_out: int = 0
  u_in: float = 1.0
  u_out: float = 1.0
  _times: list[float] = dataclasses.field(default_factory=list)

  def __post_init__(self) -> None:
    if not 0.0 < self.r_prior < 1.0:
      raise ValueError(f"r_prior must be in (0, 1), got {self.r_prior}")
    self.alpha = self.kappa * self.r_prior
    self.beta = self.kappa * (1.0 - self.r_prior)

  def record(self, outcome: typing.Literal["inside", "outside"], wall_time: float) -> None:
    """Record one oracle outcome and its wall time (seconds)."""
    t = float(wall_time)
    if self._times:
      median_all = float(np.median(self._times))
      if median_all > 0.0:
        t = min(t, self.winsor_cap * median_all)
    self._times.append(t)
    t_ref = float(np.median(self._times))
    if t_ref <= 0.0:
      t_ref = 1.0
    norm_t = t / t_ref

    if outcome == "inside":
      self.n_in += 1
      self.alpha += 1.0
      self.u_in = (1.0 - self.ema_alpha) * self.u_in + self.ema_alpha * norm_t
    else:
      self.n_out += 1
      self.beta += 1.0
      self.u_out = (1.0 - self.ema_alpha) * self.u_out + self.ema_alpha * norm_t

  def probe_fraction(self) -> float:
    """Fraction from ``inside`` toward ``outside`` for the next probe."""
    r_count = self.alpha / (self.alpha + self.beta)
    if self.n_in >= 1 and self.n_out >= 1:
      r_time = self.u_in / (self.u_in + self.u_out)
      w = min(1.0, min(self.n_in, self.n_out) / self.w_threshold)
      r_raw = w * r_time + (1.0 - w) * r_count
    else:
      r_raw = r_count

    n_eff = self.n_in + self.n_out
    denom = self.alpha + self.beta + n_eff
    se_r = float(np.sqrt(max(r_raw * (1.0 - r_raw) / denom, 0.0)))
    epsilon = min(0.5, self.z * se_r + self.epsilon_floor)
    return float(np.clip(r_raw, epsilon, 1.0 - epsilon))


def feasibility_assisted_level_crossings(  # pylint: disable=too-many-locals, too-many-arguments, too-many-positional-arguments, too-many-branches, too-many-statements
  value_func: collections.abc.Callable[[float], float],
  sign_oracle: collections.abc.Callable[[float, float], SignOracleResult],
  x_outer: float,
  x_inner: float,
  levels: collections.abc.Sequence[float],
  *,
  xtol: float = 1e-4,
  rtol: float = 1e-4,
  cost_biased_bisection: bool = True,
  r_prior: float = 0.5,
  kappa: float = 4.0,
  outer_oracle: list[SignOracleStatus | None] | None = None,
) -> list[float]:
  """Locate profile level crossings using a cheap inside/outside oracle.

  ``value_func(p)`` returns nonnegative excess (e.g. ``2NLL(p)-2NLL_min``).
  ``sign_oracle(p, level)`` returns whether excess can be ``<= level``
  (``inside``), is proven ``> level`` (``outside``), or ``unknown``.
  It may also return ``(status, cost)`` where ``cost`` is a sleep-immune
  effort measure (prefer Gurobi ``Work`` over wall-clock).

  Coarse bracketing uses the oracle (falling back to ``value_func`` on
  ``unknown``); the final root is polished with ``brentq`` on ``value_func``.

  When ``cost_biased_bisection`` is True, probe points use ``OracleCostTracker``
  (prior ``r_prior=0.5`` = midpoint until data arrives).
  """
  if not levels:
    return []
  if x_outer == x_inner:
    return [x_inner] * len(levels)

  value_cache: dict[float, float] = {}

  def f_cached(x: float) -> float:
    if x not in value_cache:
      value_cache[x] = float(value_func(x))
    return value_cache[x]

  def _call_oracle(
    x: float, level: float,
  ) -> tuple[SignOracleStatus, float | None]:
    raw = sign_oracle(x, level)
    if isinstance(raw, tuple):
      status, cost = raw
      return status, float(cost)
    return raw, None

  def classify(x: float, level: float) -> typing.Literal["inside", "outside"]:
    status, _cost = _call_oracle(x, level)
    if status == "unknown":
      return "inside" if f_cached(x) <= level else "outside"
    return status

  def classify_timed(
    x: float,
    level: float,
    tracker: OracleCostTracker | None,
  ) -> typing.Literal["inside", "outside"]:
    # Prefer oracle-reported cost (Gurobi Work). Fall back to process CPU
    # time — not wall clock — so OS sleep cannot poison the tracker.
    t0 = time.process_time()
    status, cost = _call_oracle(x, level)
    if cost is None:
      cost = time.process_time() - t0
    if status == "unknown":
      return "inside" if f_cached(x) <= level else "outside"
    if tracker is not None:
      tracker.record(status, cost)
    return status

  crossings: list[float] = []
  for level in levels:
    tracker = (
      OracleCostTracker(r_prior=r_prior, kappa=kappa)
      if cost_biased_bisection
      else None
    )
    raw_outer, outer_cost = _call_oracle(x_outer, level)
    if outer_oracle is not None:
      outer_oracle.append(None if raw_outer == "unknown" else raw_outer)
    if raw_outer == "unknown":
      outer_side = "inside" if f_cached(x_outer) <= level else "outside"
    else:
      outer_side = raw_outer
      if tracker is not None:
        cost = outer_cost if outer_cost is not None else 0.0
        tracker.record(outer_side, cost)
    if outer_side == "inside":
      crossings.append(x_outer)
      continue
    inner_side = (
      classify_timed(x_inner, level, tracker)
      if tracker is not None
      else classify(x_inner, level)
    )
    if inner_side == "outside":
      crossings.append(x_inner)
      continue

    inside = x_inner
    outside = x_outer
    while not _is_close(inside, outside, xtol, rtol):
      if tracker is not None:
        r = tracker.probe_fraction()
        mid = inside + r * (outside - inside)
      else:
        mid = 0.5 * (inside + outside)
      if _is_close(mid, inside, xtol, rtol) or _is_close(mid, outside, xtol, rtol):
        break
      side = (
        classify_timed(mid, level, tracker)
        if tracker is not None
        else classify(mid, level)
      )
      if side == "inside":
        inside = mid
      else:
        outside = mid

    left, right = (inside, outside) if inside < outside else (outside, inside)
    if left == right or _is_close(left, right, xtol, rtol):
      closer = left if abs(f_cached(left) - level) <= abs(f_cached(right) - level) else right
      crossings.append(closer)
      continue
    g_left = f_cached(left) - level
    g_right = f_cached(right) - level
    if g_left == 0.0:
      crossings.append(left)
      continue
    if g_right == 0.0:
      crossings.append(right)
      continue
    if g_left * g_right > 0.0:
      # Oracle bracket disagreed with numeric signs; pick closer endpoint.
      closer = left if abs(g_left) <= abs(g_right) else right
      crossings.append(closer)
      continue
    crossings.append(
      typing.cast(
        float,
        scipy.optimize.brentq(
          lambda x, _level=level: f_cached(x) - _level,
          left,
          right,
          xtol=xtol,
          rtol=np.float64(rtol),
          full_output=False,
        ),
      )
    )
  return crossings
