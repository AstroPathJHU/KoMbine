"""
Tests for toy-calibrated HR intervals and KM bands.
"""

import pathlib
import tempfile
import warnings

import numpy as np

from kombine.datacard import Datacard
from kombine.toy_calibration import (
  SequentialToyCounter,
  binomial_km_outcomes,
  bisection_endpoint,
  first_interior,
  fit_monotone_band_edges_binomial,
  linspaced_s_probes,
  logspaced_hr_probes,
  mc_p_value,
  naive_pointwise_band_edges_from_grid,
  resolve_hazard_ratio_interval_method,
  sequential_status,
  unique_positive_hrs,
  unique_unit_interval,
  weighted_cox_permutation,
)


def test_resolve_hazard_ratio_interval_method():
  """Toys are the default only when assignments are free."""
  assert resolve_hazard_ratio_interval_method(None, cox_only=False) == "toy"
  assert resolve_hazard_ratio_interval_method(None, cox_only=True) == "chi2"
  assert resolve_hazard_ratio_interval_method("chi2", cox_only=False) == "chi2"
  assert resolve_hazard_ratio_interval_method("chi2", cox_only=True) == "chi2"
  assert resolve_hazard_ratio_interval_method("toy", cox_only=False) == "toy"
  assert resolve_hazard_ratio_interval_method("toy", cox_only=True) == "toy"
  try:
    resolve_hazard_ratio_interval_method("wilks", cox_only=False)
  except ValueError:
    pass
  else:
    raise AssertionError("expected ValueError for an unknown method")


TINY_DATACARD = """
observable_type discrete_classes
------------
survival_time	1	2	3	4	5	6
censored	0	0	0	1	0	1
prob0	0.9	0.9	0.9	0.1	0.1	0.1
prob1	0.1	0.1	0.1	0.9	0.9	0.9
""".strip()


def _parse_tiny():
  with tempfile.NamedTemporaryFile(
    mode="w", delete=False, suffix=".txt", encoding="utf-8",
  ) as handle:
    handle.write(TINY_DATACARD)
    path = pathlib.Path(handle.name)
  try:
    return Datacard.parse_datacard(path)
  finally:
    path.unlink()


def test_sequential_status_accept_reject_continue():
  """Decision-safe early stop uses planned B, not the running fraction."""
  assert sequential_status(0, 0, 99, 0.05) == "continue"
  assert sequential_status(5, 5, 99, 0.05) == "accept"
  assert sequential_status(0, 99, 99, 0.05) == "reject"
  assert sequential_status(4, 99, 99, 0.05) == "reject"
  assert sequential_status(5, 99, 99, 0.05) == "accept"
  assert sequential_status(0, 1, 99, 0.05) == "continue"
  np.testing.assert_allclose(mc_p_value(5, 99), 6.0 / 100.0)


def test_sequential_counter_stops_when_all_cls_decided():
  """68% and 95% share the toy stream. B=19 is the smallest 95% rejectable n_max."""
  counter = SequentialToyCounter(n_max=19, confidence_levels=(0.68, 0.95))
  status = None
  for _ in range(14):
    status = counter.observe(False)
  assert status is not None
  assert status[0.68] == "reject"
  assert status[0.95] == "continue"
  assert not counter.all_decided()
  while not counter.all_decided():
    counter.observe(False)
  assert counter.n_run == 19
  assert counter.status()[0.68] == "reject"
  assert counter.status()[0.95] == "reject"


def test_weighted_cox_h1_preserves_outcome_multiset():
  """H = 1 is a uniform shuffle of the observed (time, censored) pairs."""
  times = [1.0, 2.0, 2.0, 4.0]
  censored = [False, True, False, True]
  high = [False, True, False, True]
  new_times, new_censored = weighted_cox_permutation(
    times, censored, high, 1.0, rng=0,
  )
  assert sorted(zip(new_times, new_censored)) == sorted(zip(times, censored))


def test_weighted_cox_h2_first_death_counts_are_seeded():
  """H = 2, one high-group patient: P(high dies first) = 2/4 with equal low weights."""
  times = [1.0, 2.0, 3.0]
  censored = [False, False, False]
  high = [False, False, True]
  rng = np.random.default_rng(0)
  n_trials = 200
  n_high_first = 0
  for _ in range(n_trials):
    new_times, _censored = weighted_cox_permutation(
      times, censored, high, 2.0, rng=rng,
    )
    who_first = int(np.argmin(new_times))
    if who_first == 2:
      n_high_first += 1
  assert n_high_first == 111


def test_binomial_km_outcomes_respects_risk_and_p():
  """d ~ Binomial(r, 1-p) among selected at-risk patients; unselected unchanged."""
  times = [1.0, 2.0, 3.0, 4.0]
  censored = [False, False, True, False]
  selected = [True, True, True, False]
  death_times = [1.0]
  p_survived = [1.0]
  new_times, new_censored = binomial_km_outcomes(
    times, censored, selected, death_times, p_survived, rng=0,
  )
  assert new_times[3] == 4.0
  assert new_censored[3] is False
  assert new_times[0] != 1.0 or new_censored[0] is True

  p_survived_zero = [0.0]
  rng = np.random.default_rng(1)
  n_deaths_at_1 = 0
  n_rep = 50
  for _ in range(n_rep):
    toy_times, toy_censored = binomial_km_outcomes(
      times, censored, selected, death_times, p_survived_zero, rng=rng,
    )
    n_deaths_at_1 += sum(
      (t == 1.0) and (not c) for t, c in zip(toy_times, toy_censored)
    )
  assert n_deaths_at_1 == 150


def test_bisection_endpoint_from_chi2_edges():
  """Endpoint search starts at a chi2-like outer edge and only bisects the crossing."""
  def inside(value: float) -> bool:
    return 0.5 <= value <= 2.0

  assert bisection_endpoint(inside, 1.0, 2.0, xtol=1e-3, log_scale=True) == 2.0
  upper = bisection_endpoint(inside, 1.0, 4.0, xtol=1e-3, log_scale=True)
  assert 2.0 < upper < 4.0
  lower = bisection_endpoint(inside, 1.0, 0.1, xtol=1e-3, log_scale=True)
  assert 0.1 < lower < 0.5


def test_first_interior_uses_h1_when_mle_rejected():
  """When the MLE sits outside, H=1 is tried before the log-spaced grid."""
  def inside(value: float) -> bool:
    return 0.5 <= value <= 2.0

  probes = unique_positive_hrs(
    [20.0, 1.0],
    logspaced_hr_probes(0.01, 100.0),
  )
  assert probes[0] == 20.0
  assert probes[1] == 1.0
  assert first_interior(inside, probes) == 1.0
  assert first_interior(inside, [20.0, 0.05, 100.0]) is None


def test_first_interior_uses_half_when_km_mle_rejected():
  """When the KM MLE sits outside, S=0.5 is tried before the linspace grid."""
  def inside(value: float) -> bool:
    return 0.4 <= value <= 0.6

  probes = unique_unit_interval(
    [0.95, 0.5],
    linspaced_s_probes(1e-6, 1.0 - 1e-6),
  )
  assert probes[0] == 0.95
  assert probes[1] == 0.5
  assert first_interior(inside, probes) == 0.5
  assert first_interior(inside, [0.95, 0.05, 0.99]) is None


def test_fit_monotone_band_edges_binomial_recovers_step_band():
  """Synthetic binomial counts recover a known decreasing band on a grid."""
  s_grid = np.linspace(0.1, 0.9, 9)
  true_lo = np.array([0.4, 0.4, 0.3, 0.3, 0.2])
  true_hi = np.array([0.8, 0.7, 0.7, 0.6, 0.5])
  n_max = 40
  rng = np.random.default_rng(0)
  n_extreme = np.empty((len(true_lo), len(s_grid)), dtype=int)
  for i_time, (lo, hi) in enumerate(zip(true_lo, true_hi)):
    for i_s, survival in enumerate(s_grid):
      # High n_extreme inside = accept (matches real MC toys).
      pi = 0.7 if lo <= survival <= hi else 0.1
      n_extreme[i_time, i_s] = rng.binomial(n_max, pi)
  best = 0.5 * (true_lo + true_hi)
  fit = fit_monotone_band_edges_binomial(
    s_grid, n_extreme, n_max, best=best,
  )
  assert np.all(np.diff(fit.lo) <= 1e-12)
  assert np.all(np.diff(fit.hi) <= 1e-12)
  assert np.all(fit.lo <= fit.hi)
  assert fit.pi_in > fit.pi_out
  np.testing.assert_allclose(fit.lo, true_lo, atol=0.15)
  np.testing.assert_allclose(fit.hi, true_hi, atol=0.15)


def test_naive_pointwise_band_edges_need_not_be_monotone():
  """Per-time accept min/max can rise with t even when a monotone fit would not."""
  s_grid = np.array([0.2, 0.4, 0.6, 0.8])
  # Extreme = toy more extreme than data. High n_ext => accept H0.
  # B=19: accept 95% when (1+n_ext)/20 > 0.05 i.e. n_ext >= 1.
  n_extreme = np.array([
    [0, 5, 5, 0],   # accept 0.4 and 0.6
    [5, 0, 0, 5],   # accept 0.2 and 0.8 -> non-monotone vs previous
  ], dtype=int)
  edges = naive_pointwise_band_edges_from_grid(
    s_grid, n_extreme, 19, confidence_level=0.95,
  )
  np.testing.assert_allclose(edges[0], [0.4, 0.6])
  np.testing.assert_allclose(edges[1], [0.2, 0.8])
  assert edges[1, 0] < edges[0, 0] or edges[1, 1] > edges[0, 1]


def test_hr_toy_test_golden():
  """Seeded tiny-cohort HR Neyman test; numbers are regression goldens."""
  datacard = _parse_tiny()
  calc = datacard.km_hazard_ratio(
    parameter_threshold=1.0,
    parameter_min=-np.inf,
    parameter_max=np.inf,
  )
  result = calc.hypothesized_hr_toy_test(
    1.0,
    n_max=19,
    rng=0,
    Threads=1,
    confidence_levels=(0.68, 0.95),
  )
  assert result.n_max == 19
  assert result.n_run == 19
  assert result.n_extreme == 0
  np.testing.assert_allclose(result.t_obs, 6.005813431284308, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(result.p_value, 0.05)
  assert result.decisions[0.68] == "reject"
  assert result.decisions[0.95] == "reject"


def test_km_toy_test_golden():
  """Seeded tiny-cohort KM Neyman test at one time and S."""
  datacard = _parse_tiny()
  km = datacard.km_likelihood(
    parameter_min=-np.inf,
    parameter_max=1.0,
  )
  death_times = sorted(p.time for p in km.all_patients if not p.censored)
  time_point = death_times[0]
  result = km.hypothesized_s_toy_test(
    time_point,
    0.5,
    n_max=19,
    rng=0,
    Threads=1,
    confidence_levels=(0.68, 0.95),
  )
  assert result.n_max == 19
  assert result.n_run == 19
  assert result.n_extreme == 5
  np.testing.assert_allclose(result.t_obs, 0.33979725754943546, rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(result.p_value, 0.3)
  assert result.decisions[0.68] == "reject"
  assert result.decisions[0.95] == "accept"


def test_hr_mle_at_bound_toy_region_golden():
  """Tiny-card free-H fit sits against the lower log-HR bound.

  The profile is flat there: locked 2NLL changes by about 0.01 between the
  incumbents Gurobi reports on Windows (HR 4.63e-5) and Linux (HR 9.47e-5).
  The reported MLE and the likelihood-ratio statistic move with that choice.
  The Neyman decision does not. Every toy is less extreme, and both
  confidence levels reject.
  """
  datacard = _parse_tiny()
  calc = datacard.km_hazard_ratio(
    parameter_threshold=1.0,
    parameter_min=-np.inf,
    parameter_max=np.inf,
  )
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    _, result_alt = calc._fit_null_and_alternative(  # pylint: disable=protected-access
      cox_only=False,
      patient_wise_only=False,
      verbose=False,
      print_progress=False,
      solve_null=False,
      MIPGap=1e-6,
      MIPGapAbs=1e-8,
      TimeLimit=None,
      Threads=1,
      MIPFocus=None,
      LogFile=None,
    )
    best = float(result_alt.hazard_ratio)
    result = calc.hypothesized_hr_toy_test(
      best,
      n_max=19,
      rng=0,
      Threads=1,
      confidence_levels=(0.68, 0.95),
    )
  lower = float(np.exp(calc.log_hazard_ratio_bounds[0]))
  assert lower * 0.99 <= best < 10.0 * lower
  assert result.n_run == 19
  assert result.n_extreme == 0
  np.testing.assert_allclose(result.p_value, 0.05)
  assert result.decisions[0.68] == "reject"
  assert result.decisions[0.95] == "reject"


if __name__ == "__main__":
  test_resolve_hazard_ratio_interval_method()
  test_sequential_status_accept_reject_continue()
  test_sequential_counter_stops_when_all_cls_decided()
  test_weighted_cox_h1_preserves_outcome_multiset()
  test_weighted_cox_h2_first_death_counts_are_seeded()
  test_binomial_km_outcomes_respects_risk_and_p()
  test_bisection_endpoint_from_chi2_edges()
  test_first_interior_uses_h1_when_mle_rejected()
  test_first_interior_uses_half_when_km_mle_rejected()
  test_fit_monotone_band_edges_binomial_recovers_step_band()
  test_naive_pointwise_band_edges_need_not_be_monotone()
  print("[PASS] generator / sequential tests")
  test_hr_toy_test_golden()
  print("[PASS] HR toy golden")
  test_km_toy_test_golden()
  print("[PASS] KM toy golden")
  test_hr_mle_at_bound_toy_region_golden()
  print("[PASS] HR MLE at bound")
  print("[SUCCESS] toy calibration tests passed")
