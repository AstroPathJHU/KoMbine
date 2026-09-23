"""
Toy Neyman inversion of the profile LRT for HR intervals and KM bands.

Toys are compared to the observed Δ2NLL (inside / outside), not converted
to a χ² percentile. Sequential stopping uses a pre-set maximum B.
"""

from __future__ import annotations

import dataclasses
import math
import typing

import numpy as np
import numpy.typing as npt


DEFAULT_CLS: tuple[float, ...] = (0.68, 0.95)


def mc_p_value(n_extreme: int, n_max: int) -> float:
  """Monte Carlo p-value with the +1 continuity correction."""
  if n_max < 0:
    raise ValueError(f"n_max must be >= 0, got {n_max}")
  if n_extreme < 0:
    raise ValueError(f"n_extreme must be >= 0, got {n_extreme}")
  return (1.0 + n_extreme) / (1.0 + n_max)


def sequential_status(
  n_extreme: int,
  n_run: int,
  n_max: int,
  alpha: float,
) -> typing.Literal["accept", "reject", "continue"]:
  """
  Decision-safe early stop for a planned maximum of ``n_max`` toys.

  Accept if even the remaining toys being non-extreme still gives p > α.
  Reject if even the remaining toys being extreme still gives p ≤ α.
  """
  if n_max < 0:
    raise ValueError(f"n_max must be >= 0, got {n_max}")
  if not 0 <= n_run <= n_max:
    raise ValueError(f"n_run must be in [0, n_max], got {n_run} / {n_max}")
  if not 0 <= n_extreme <= n_run:
    raise ValueError(
      f"n_extreme must be in [0, n_run], got {n_extreme} / {n_run}"
    )
  if not 0.0 < alpha < 1.0:
    raise ValueError(f"alpha must be in (0, 1), got {alpha}")
  remaining = n_max - n_run
  denom = 1.0 + n_max
  p_if_no_more = (1.0 + n_extreme) / denom
  p_if_all_more = (1.0 + n_extreme + remaining) / denom
  if p_if_no_more > alpha:
    return "accept"
  if p_if_all_more <= alpha:
    return "reject"
  return "continue"


@dataclasses.dataclass
class SequentialToyCounter:
  """Running extreme-count for one or more confidence levels."""

  n_max: int
  confidence_levels: tuple[float, ...] = DEFAULT_CLS
  n_extreme: int = 0
  n_run: int = 0

  def __post_init__(self) -> None:
    if self.n_max < 0:
      raise ValueError(f"n_max must be >= 0, got {self.n_max}")
    if not self.confidence_levels:
      raise ValueError("confidence_levels must be non-empty")
    for cl in self.confidence_levels:
      if not 0.0 < cl < 1.0:
        raise ValueError(f"confidence level must be in (0, 1), got {cl}")

  def observe(self, extreme: bool) -> dict[float, str]:
    """Record one toy and return the current decision per CL."""
    if self.n_run >= self.n_max:
      raise RuntimeError(
        f"Already ran n_max={self.n_max} toys; cannot observe another"
      )
    self.n_run += 1
    if extreme:
      self.n_extreme += 1
    return self.status()

  def status(self) -> dict[float, str]:
    """Map each CL to accept / reject / continue."""
    return {
      cl: sequential_status(
        self.n_extreme, self.n_run, self.n_max, 1.0 - cl,
      )
      for cl in self.confidence_levels
    }

  def all_decided(self) -> bool:
    """True when no CL is still ``continue``."""
    return all(decision != "continue" for decision in self.status().values())

  @property
  def p_value(self) -> float:
    """Monte Carlo p-value using the planned ``n_max``."""
    return mc_p_value(self.n_extreme, self.n_max)

  def as_result(self, t_obs: float) -> "ToyTestResult":
    """Pack this counter into a ``ToyTestResult``."""
    return ToyTestResult(
      n_extreme=self.n_extreme,
      n_run=self.n_run,
      n_max=self.n_max,
      t_obs=float(t_obs),
      p_value=self.p_value,
      decisions=self.status(),
    )


@dataclasses.dataclass
class ToyTestResult:
  """Result of a Neyman test of one hypothesized parameter value."""

  n_extreme: int
  n_run: int
  n_max: int
  t_obs: float
  p_value: float
  decisions: dict[float, str]


def as_generator(rng: int | np.random.Generator | None) -> np.random.Generator:
  """Normalize a seed or Generator."""
  return np.random.default_rng(rng)


def hard_group_high(
  observed_parameters: npt.ArrayLike,
  parameter_threshold: float,
  parameter_max: float = np.inf,
) -> npt.NDArray[np.bool_]:
  """Hard high-group label: observed parameter in [threshold, max)."""
  observed = np.asarray(observed_parameters, dtype=float)
  return (observed >= parameter_threshold) & (observed < parameter_max)


def weighted_cox_permutation(  # pylint: disable=too-many-locals
  times: npt.ArrayLike,
  censored: npt.ArrayLike,
  group_high: npt.ArrayLike,
  hazard_ratio: float,
  rng: int | np.random.Generator | None = None,
) -> tuple[list[float], list[bool]]:
  """
  Re-pair the observed ``(time, censored)`` multiset using Cox weights.

  Death times are visited in order. At each time, decedents are drawn
  without replacement from the remaining patients with weights
  ``H^{1[high]}``. Censorings at that time are assigned uniformly.
  At H = 1 every weight is 1, so this is a uniform shuffle of the
  outcome multiset.
  """
  if hazard_ratio <= 0 or not math.isfinite(hazard_ratio):
    raise ValueError(f"hazard_ratio must be finite and > 0, got {hazard_ratio}")
  times_arr = np.asarray(times, dtype=float)
  censored_arr = np.asarray(censored, dtype=bool)
  high = np.asarray(group_high, dtype=bool)
  n_patients = times_arr.size
  if censored_arr.size != n_patients or high.size != n_patients:
    raise ValueError(
      "times, censored, and group_high must have the same length, "
      f"got {n_patients}, {censored_arr.size}, {high.size}"
    )
  generator = as_generator(rng)
  by_time: dict[float, list[bool]] = {}
  for time_value, is_censored in zip(times_arr.tolist(), censored_arr.tolist()):
    by_time.setdefault(time_value, []).append(bool(is_censored))

  remaining = list(range(n_patients))
  new_times = [0.0] * n_patients
  new_censored = [False] * n_patients
  weights = np.where(high, float(hazard_ratio), 1.0)

  def _assign(  # pylint: disable=too-many-arguments
    candidates: npt.NDArray[np.intp],
    n_draw: int,
    probs: npt.NDArray[np.float64] | None,
    is_censored: bool,
    event_time: float,
  ) -> None:
    if n_draw <= 0:
      return
    if n_draw > candidates.size:
      raise RuntimeError(
        f"Need {n_draw} patients at t={event_time} but {candidates.size} remain"
      )
    chosen = generator.choice(candidates, size=n_draw, replace=False, p=probs)
    for index in np.atleast_1d(chosen).astype(int, copy=False):
      new_times[index] = event_time
      new_censored[index] = is_censored
      remaining.remove(index)

  for time_value in sorted(by_time):
    flags = by_time[time_value]
    n_deaths = sum(not flag for flag in flags)
    n_censors = sum(flag for flag in flags)
    remaining_arr = np.asarray(remaining, dtype=int)
    if n_deaths:
      raw = np.maximum(weights[remaining_arr], 0.0)
      if float(raw.sum()) <= 0.0:
        probs = None
      else:
        probs = raw / raw.sum()
      _assign(remaining_arr, n_deaths, probs, False, time_value)
      remaining_arr = np.asarray(remaining, dtype=int)
    if n_censors:
      _assign(remaining_arr, n_censors, None, True, time_value)

  if remaining:
    raise RuntimeError(f"Unassigned patients after permutation: {remaining}")
  return new_times, new_censored


def binomial_km_outcomes(  # pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
  times: npt.ArrayLike,
  censored: npt.ArrayLike,
  selected: npt.ArrayLike,
  death_times: npt.ArrayLike,
  p_survived: npt.ArrayLike,
  rng: int | np.random.Generator | None = None,
) -> tuple[list[float], list[bool]]:
  """
  Redraw deaths on the observed time grid from interval survival probabilities.

  Unselected patients keep their original outcomes. Selected patients who are
  at risk at each ``death_times[i]`` contribute ``Binomial(r, 1 - p_survived[i])``
  deaths at that time. Patients who reach their original follow-up time without
  dying are censored there.
  """
  times_arr = np.asarray(times, dtype=float)
  censored_arr = np.asarray(censored, dtype=bool)
  selected_arr = np.asarray(selected, dtype=bool)
  death_times_arr = np.asarray(death_times, dtype=float)
  p_survived_arr = np.asarray(p_survived, dtype=float)
  n_patients = times_arr.size
  if censored_arr.size != n_patients or selected_arr.size != n_patients:
    raise ValueError("times, censored, and selected must have the same length")
  if death_times_arr.size != p_survived_arr.size:
    raise ValueError("death_times and p_survived must have the same length")
  generator = as_generator(rng)
  new_times = times_arr.tolist()
  new_censored = censored_arr.tolist()
  still_alive = selected_arr.copy()
  for time_value, p_s in zip(death_times_arr.tolist(), p_survived_arr.tolist()):
    at_risk = still_alive & (times_arr >= time_value)
    risk_idx = np.flatnonzero(at_risk)
    n_risk = int(risk_idx.size)
    if n_risk == 0:
      continue
    p_clip = min(max(float(p_s), 0.0), 1.0)
    n_deaths = int(generator.binomial(n_risk, 1.0 - p_clip))
    n_deaths = min(n_deaths, n_risk)
    if n_deaths:
      died = generator.choice(risk_idx, size=n_deaths, replace=False)
      for index in np.atleast_1d(died).astype(int, copy=False):
        new_times[index] = float(time_value)
        new_censored[index] = False
        still_alive[index] = False
    for index in risk_idx:
      if still_alive[index] and times_arr[index] == time_value:
        new_times[int(index)] = float(time_value)
        new_censored[int(index)] = True
        still_alive[index] = False
  return new_times, new_censored


def unique_positive_hrs(*groups: typing.Iterable[float]) -> list[float]:
  """Deduplicate positive finite H values, preserving first-seen order."""
  seen: set[float] = set()
  ordered: list[float] = []
  for group in groups:
    for value in group:
      hazard_ratio = float(value)
      if not math.isfinite(hazard_ratio) or hazard_ratio <= 0:
        continue
      key = round(hazard_ratio, 10)
      if key in seen:
        continue
      seen.add(key)
      ordered.append(hazard_ratio)
  return ordered


def logspaced_hr_probes(x_min: float, x_max: float, n_grid: int = 9) -> list[float]:
  """Log-spaced H probes on ``[x_min, x_max]``."""
  if x_min <= 0 or x_max <= 0 or x_max < x_min:
    raise ValueError(
      f"Need 0 < x_min <= x_max, got x_min={x_min}, x_max={x_max}"
    )
  if n_grid < 2:
    raise ValueError(f"n_grid must be >= 2, got {n_grid}")
  return np.geomspace(x_min, x_max, num=n_grid).tolist()


def unique_unit_interval(*groups: typing.Iterable[float]) -> list[float]:
  """Deduplicate finite values in (0, 1), preserving first-seen order."""
  seen: set[float] = set()
  ordered: list[float] = []
  for group in groups:
    for value in group:
      probability = float(value)
      if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        continue
      key = round(probability, 10)
      if key in seen:
        continue
      seen.add(key)
      ordered.append(probability)
  return ordered


def linspaced_s_probes(
  s_min: float, s_max: float, n_grid: int = 9,
) -> list[float]:
  """Linearly spaced survival-probability probes on ``[s_min, s_max]``."""
  if not 0.0 < s_min <= s_max < 1.0:
    raise ValueError(
      f"Need 0 < s_min <= s_max < 1, got s_min={s_min}, s_max={s_max}"
    )
  if n_grid < 2:
    raise ValueError(f"n_grid must be >= 2, got {n_grid}")
  return np.linspace(s_min, s_max, num=n_grid).tolist()


def first_interior(
  is_inside: typing.Callable[[float], bool],
  candidates: typing.Sequence[float],
) -> float | None:
  """Return the first candidate that ``is_inside`` accepts, else None."""
  for value in candidates:
    if is_inside(float(value)):
      return float(value)
  return None


def bisection_endpoint(  # pylint: disable=too-many-arguments
  is_inside: typing.Callable[[float], bool],
  x_inner: float,
  x_outer: float,
  *,
  xtol: float,
  log_scale: bool = False,
  max_eval: int = 24,
) -> float:
  """
  Bisect the accept/reject crossing between ``x_inner`` (inside) and ``x_outer``.

  If ``x_outer`` is still inside, it is returned. ``x_inner`` is assumed inside.
  """
  if log_scale:
    if x_inner <= 0 or x_outer <= 0:
      raise ValueError("log_scale requires positive inner and outer values")
    inner = math.log(x_inner)
    outer = math.log(x_outer)
    def to_x(value: float) -> float:
      return math.exp(value)
  else:
    inner = x_inner
    outer = x_outer
    def to_x(value: float) -> float:
      return value

  if is_inside(x_outer):
    return x_outer
  n_eval = 1
  while abs(outer - inner) > xtol and n_eval < max_eval:
    mid = 0.5 * (inner + outer)
    n_eval += 1
    if is_inside(to_x(mid)):
      inner = mid
    else:
      outer = mid
  return to_x(outer)


@dataclasses.dataclass(frozen=True)
class MonotoneBandFit:
  """Binomial-MLE decreasing survival band edges."""

  lo: npt.NDArray[np.float64]
  hi: npt.NDArray[np.float64]
  pi_in: float
  pi_out: float
  loglik: float
  s_grid: npt.NDArray[np.float64]


def _binomial_loglik(n_extreme: int, n_max: int, pi: float) -> float:
  """Binomial log-likelihood for one cell (up to an additive constant)."""
  if n_max < 0:
    raise ValueError(f"n_max must be >= 0, got {n_max}")
  if not 0 <= n_extreme <= n_max:
    raise ValueError(
      f"n_extreme must be in [0, n_max], got {n_extreme} / {n_max}"
    )
  pi_clip = min(max(float(pi), 1e-12), 1.0 - 1e-12)
  return (
    float(n_extreme) * math.log(pi_clip)
    + float(n_max - n_extreme) * math.log(1.0 - pi_clip)
  )


def _pair_scores_one_time(
  cell_ll: npt.NDArray[np.float64],
  best_i: int | None,
) -> npt.NDArray[np.float64]:
  """Scores for every (lo_idx, hi_idx) pair at one time."""
  n_s = cell_ll.shape[0]
  pair_score = np.full((n_s, n_s), -np.inf)
  for lo_i in range(n_s):
    for hi_i in range(lo_i, n_s):
      if best_i is not None and not lo_i <= best_i <= hi_i:
        continue
      inside = (np.arange(n_s) >= lo_i) & (np.arange(n_s) <= hi_i)
      pair_score[lo_i, hi_i] = float(
        cell_ll[inside, 1].sum() + cell_ll[~inside, 0].sum()
      )
  return pair_score


def _best_previous_pair(
  dp_prev: npt.NDArray[np.float64],
  lo_i: int,
  hi_i: int,
) -> tuple[float, int, int]:
  """Best feasible previous (lo, hi) for nonincreasing edges."""
  n_s = dp_prev.shape[0]
  best_prev = -np.inf
  arg_lo = -1
  arg_hi = -1
  for lo_prev in range(lo_i, n_s):
    for hi_prev in range(max(lo_prev, hi_i), n_s):
      candidate = float(dp_prev[lo_prev, hi_prev])
      if candidate > best_prev:
        best_prev = candidate
        arg_lo = lo_prev
        arg_hi = hi_prev
  return best_prev, arg_lo, arg_hi


def _best_monotone_indices_for_pis(  # pylint: disable=too-many-locals
  n_extreme: npt.NDArray[np.integer],
  n_max: int,
  pi_in: float,
  pi_out: float,
  *,
  best_idx: npt.NDArray[np.integer] | None = None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], float]:
  """
  DP: best decreasing lo/hi index paths for fixed ``pi_in`` / ``pi_out``.

  Survival decreases with time, so index paths on an ascending S-grid are
  componentwise nonincreasing. Optional ``best_idx[t]`` must lie in
  ``[lo_idx[t], hi_idx[t]]``.
  """
  n_times, n_s = n_extreme.shape
  cell_ll = np.empty((n_times, n_s, 2), dtype=float)
  for i_time in range(n_times):
    for i_s in range(n_s):
      n_ext = int(n_extreme[i_time, i_s])
      cell_ll[i_time, i_s, 0] = _binomial_loglik(n_ext, n_max, pi_out)
      cell_ll[i_time, i_s, 1] = _binomial_loglik(n_ext, n_max, pi_in)

  pair_score = np.empty((n_times, n_s, n_s), dtype=float)
  for i_time in range(n_times):
    best_i = int(best_idx[i_time]) if best_idx is not None else None
    pair_score[i_time] = _pair_scores_one_time(cell_ll[i_time], best_i)

  dp = np.full((n_times, n_s, n_s), -np.inf)
  prev_lo = np.full((n_times, n_s, n_s), -1, dtype=np.int64)
  prev_hi = np.full((n_times, n_s, n_s), -1, dtype=np.int64)
  dp[0] = pair_score[0]

  for i_time in range(1, n_times):
    for lo_i in range(n_s):
      for hi_i in range(lo_i, n_s):
        base = float(pair_score[i_time, lo_i, hi_i])
        if not math.isfinite(base):
          continue
        best_prev, arg_lo, arg_hi = _best_previous_pair(dp[i_time - 1], lo_i, hi_i)
        if math.isfinite(best_prev):
          dp[i_time, lo_i, hi_i] = base + best_prev
          prev_lo[i_time, lo_i, hi_i] = arg_lo
          prev_hi[i_time, lo_i, hi_i] = arg_hi

  end = dp[n_times - 1]
  if not np.isfinite(end).any():
    raise RuntimeError("No feasible monotone band on the S-grid")
  end_idx = np.unravel_index(int(np.argmax(end)), end.shape)
  lo_end = int(end_idx[0])
  hi_end = int(end_idx[1])

  lo_path = np.empty(n_times, dtype=np.int64)
  hi_path = np.empty(n_times, dtype=np.int64)
  lo_path[-1] = lo_end
  hi_path[-1] = hi_end
  for i_time in range(n_times - 1, 0, -1):
    lo_cur = int(lo_path[i_time])
    hi_cur = int(hi_path[i_time])
    lo_path[i_time - 1] = int(prev_lo[i_time, lo_cur, hi_cur])
    hi_path[i_time - 1] = int(prev_hi[i_time, lo_cur, hi_cur])
  return lo_path, hi_path, float(dp[n_times - 1, lo_end, hi_end])


def fit_monotone_band_edges_binomial(  # pylint: disable=too-many-locals,too-many-branches,too-many-arguments
  s_grid: npt.ArrayLike,
  n_extreme: npt.ArrayLike,
  n_max: int,
  *,
  best: npt.ArrayLike | None = None,
  pi_in_values: typing.Sequence[float] | None = None,
  pi_out_values: typing.Sequence[float] | None = None,
) -> MonotoneBandFit:
  """
  Fit decreasing ``lo(t)``, ``hi(t)`` by binomial MLE on an S-grid.

  Model: at time ``t`` and grid survival ``s``,
  ``n_extreme ~ Binomial(n_max, π_in)`` if ``lo(t) <= s <= hi(t)``, else
  ``Binomial(n_max, π_out)``, with ``π_in > π_out`` and both edges
  nonincreasing in ``t``.

  High ``n_extreme`` means toys were often more extreme than the data, so the
  hypothesized ``S`` is compatible (standard MC p-value). Inside the band we
  therefore expect a *higher* extreme rate than outside.
  """
  s_arr = np.asarray(s_grid, dtype=float)
  counts = np.asarray(n_extreme, dtype=int)
  if s_arr.ndim != 1 or s_arr.size < 2:
    raise ValueError("s_grid must be a 1-d array with at least 2 points")
  if counts.ndim != 2 or counts.shape[1] != s_arr.size:
    raise ValueError(
      f"n_extreme shape {counts.shape} incompatible with s_grid length {s_arr.size}"
    )
  if np.any(np.diff(s_arr) <= 0):
    raise ValueError("s_grid must be strictly increasing")
  if n_max < 1:
    raise ValueError(f"n_max must be >= 1, got {n_max}")

  best_idx = None
  if best is not None:
    best_arr = np.asarray(best, dtype=float)
    if best_arr.shape != (counts.shape[0],):
      raise ValueError(
        f"best shape {best_arr.shape} != (n_times,)={(counts.shape[0],)}"
      )
    best_idx = np.array(
      [int(np.argmin(np.abs(s_arr - float(value)))) for value in best_arr],
      dtype=np.int64,
    )

  # Inside: high extreme rate (accept); outside: low (reject).
  if pi_in_values is None:
    pi_in_values = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
  if pi_out_values is None:
    pi_out_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

  best_fit: MonotoneBandFit | None = None
  for pi_in in pi_in_values:
    for pi_out in pi_out_values:
      if not 0.0 < pi_out < pi_in < 1.0:
        continue
      lo_idx, hi_idx, loglik = _best_monotone_indices_for_pis(
        counts, n_max, float(pi_in), float(pi_out), best_idx=best_idx,
      )
      candidate = MonotoneBandFit(
        lo=s_arr[lo_idx].copy(),
        hi=s_arr[hi_idx].copy(),
        pi_in=float(pi_in),
        pi_out=float(pi_out),
        loglik=float(loglik),
        s_grid=s_arr.copy(),
      )
      if best_fit is None or candidate.loglik > best_fit.loglik:
        best_fit = candidate
  if best_fit is None:
    raise RuntimeError("No valid (pi_in, pi_out) pair for monotone band fit")
  return best_fit


def naive_pointwise_band_edges_from_grid(  # pylint: disable=too-many-locals
  s_grid: npt.ArrayLike,
  n_extreme: npt.ArrayLike,
  n_max: int,
  *,
  confidence_level: float = 0.95,
  best: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
  """
  Per-time toy accept intervals on an S-grid (no monotone constraint).

  At each time, accept grid values with Monte Carlo p-value
  ``(1 + n_extreme) / (1 + n_max) > 1 - confidence_level``. The naive
  interval is the min/max accepted ``S`` (so edges need not decrease in
  ``t``). If nothing is accepted, falls back to the nearest grid point to
  ``best`` (or the mid grid point).

  Returns an ``(n_times, 2)`` array of ``[lo, hi]``.
  """
  s_arr = np.asarray(s_grid, dtype=float)
  counts = np.asarray(n_extreme, dtype=int)
  if s_arr.ndim != 1 or s_arr.size < 1:
    raise ValueError("s_grid must be a non-empty 1-d array")
  if counts.ndim != 2 or counts.shape[1] != s_arr.size:
    raise ValueError(
      f"n_extreme shape {counts.shape} incompatible with s_grid length {s_arr.size}"
    )
  if not 0.0 < confidence_level < 1.0:
    raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
  alpha = 1.0 - float(confidence_level)
  n_times = counts.shape[0]
  best_arr = None if best is None else np.asarray(best, dtype=float)
  if best_arr is not None and best_arr.shape != (n_times,):
    raise ValueError(
      f"best shape {best_arr.shape} != (n_times,)={(n_times,)}"
    )

  edges = np.empty((n_times, 2), dtype=float)
  for i_time in range(n_times):
    accepted = [
      float(s_arr[i_s])
      for i_s in range(s_arr.size)
      if mc_p_value(int(counts[i_time, i_s]), n_max) > alpha
    ]
    if accepted:
      edges[i_time, 0] = min(accepted)
      edges[i_time, 1] = max(accepted)
      continue
    if best_arr is None:
      mid = float(s_arr[s_arr.size // 2])
      edges[i_time, 0] = mid
      edges[i_time, 1] = mid
    else:
      nearest = float(s_arr[int(np.argmin(np.abs(s_arr - best_arr[i_time])))])
      edges[i_time, 0] = nearest
      edges[i_time, 1] = nearest
  return edges
