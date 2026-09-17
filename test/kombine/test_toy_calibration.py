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
  logspaced_hr_probes,
  mc_p_value,
  sequential_status,
  unique_positive_hrs,
  weighted_cox_permutation,
)


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
  """Tiny-card free-H MLE sits on the log-HR bound; toys reject both CLs (seeded)."""
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
  np.testing.assert_allclose(best, 4.6274862563194215e-05, rtol=1e-3)
  assert result.n_run == 19
  assert result.n_extreme == 0
  np.testing.assert_allclose(result.t_obs, 0.03635340527999009, rtol=1e-3, atol=1e-4)
  np.testing.assert_allclose(result.p_value, 0.05)
  assert result.decisions[0.68] == "reject"
  assert result.decisions[0.95] == "reject"


if __name__ == "__main__":
  test_sequential_status_accept_reject_continue()
  test_sequential_counter_stops_when_all_cls_decided()
  test_weighted_cox_h1_preserves_outcome_multiset()
  test_weighted_cox_h2_first_death_counts_are_seeded()
  test_binomial_km_outcomes_respects_risk_and_p()
  test_bisection_endpoint_from_chi2_edges()
  test_first_interior_uses_h1_when_mle_rejected()
  print("[PASS] generator / sequential tests")
  test_hr_toy_test_golden()
  print("[PASS] HR toy golden")
  test_km_toy_test_golden()
  print("[PASS] KM toy golden")
  test_hr_mle_at_bound_toy_region_golden()
  print("[PASS] HR MLE at bound")
  print("[SUCCESS] toy calibration tests passed")
