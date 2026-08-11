# pylint: disable=duplicate-code
"""
Test Küchenhoff MC-SIMEX comparison estimators.
"""

import pathlib
import warnings

import numpy as np

import kombine.datacard
from kombine.datacard import (
  DiscreteClassObservable,
  FixedObservable,
  Patient,
  PoissonRatioObservable,
)
from kombine.comparisons import McSimexBase
from .test_yi_correction import generate_synthetic_datacard_with_perfect_classification

warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here / "datacards" / "simple_examples"


def test_mc_simex_poisson_ratio_not_implemented():
  """
  MC-SIMEX should raise for Poisson ratio observables, same as Yi.
  """
  obs = PoissonRatioObservable(
    numerator=10,
    denominator=5,
    unique_id_numerator=1,
    unique_id_denominator=2,
  )
  patient = Patient(
    survival_time=1.0,
    censored=False,
    observable=obs,
  )
  simex = McSimexBase([patient])

  try:
    simex.compute_patient_prob_in_range(patient, 0.0, 1.0)
  except NotImplementedError:
    return
  raise AssertionError("Expected NotImplementedError for PoissonRatioObservable")


def test_mc_simex_km_matches_nominal_fixed_observable():
  """
  With e_i = 0, MC-SIMEX KM matches nominal KM and Yi.
  """
  datacard = kombine.datacard.Datacard.parse_datacard(
    datacards / "fixed_hr_example.txt"
  )
  threshold = 0.5

  simex_low = datacard.km_survival_mc_simex(
    parameter_min=-np.inf,
    parameter_max=threshold,
    B=1,
    rng=0,
  )
  simex_high = datacard.km_survival_mc_simex(
    parameter_min=threshold,
    parameter_max=np.inf,
    B=1,
    rng=0,
  )
  yi_low = datacard.km_survival_yi(
    parameter_min=-np.inf,
    parameter_max=threshold,
    times_for_plot=simex_low['times_for_plot'],
  )
  yi_high = datacard.km_survival_yi(
    parameter_min=threshold,
    parameter_max=np.inf,
    times_for_plot=simex_high['times_for_plot'],
  )
  km_low = datacard.km_likelihood(
    parameter_min=-np.inf,
    parameter_max=threshold,
  )
  km_high = datacard.km_likelihood(
    parameter_min=threshold,
    parameter_max=np.inf,
  )
  low_nominal = km_low.nominalkm.survival_probabilities(
    times_for_plot=simex_low['times_for_plot']
  )
  high_nominal = km_high.nominalkm.survival_probabilities(
    times_for_plot=simex_high['times_for_plot']
  )

  np.testing.assert_allclose(
    simex_low['survival_probabilities'],
    low_nominal,
    atol=1e-10,
    rtol=1e-10,
  )
  np.testing.assert_allclose(
    simex_high['survival_probabilities'],
    high_nominal,
    atol=1e-10,
    rtol=1e-10,
  )
  np.testing.assert_allclose(
    simex_low['survival_probabilities'],
    yi_low['survival_probabilities'],
    atol=1e-10,
    rtol=1e-10,
  )
  np.testing.assert_allclose(
    simex_high['survival_probabilities'],
    yi_high['survival_probabilities'],
    atol=1e-10,
    rtol=1e-10,
  )
  assert simex_low['method'] == 'mc_simex'


def test_mc_simex_logrank_and_hr_match_hard_labels():
  """
  With e_i = 0, logrank and HR match cox_only / hard-label estimators.
  """
  datacard = generate_synthetic_datacard_with_perfect_classification(
    target_hr=2.5,
    n_patients_per_group=40,
    random_seed=123,
  )
  threshold = 100.0

  p_standard = datacard.km_p_value_logrank(
    parameter_threshold=threshold,
    cox_only=True,
  )
  p_simex = datacard.km_p_value_logrank_mc_simex(
    parameter_threshold=threshold,
    B=1,
    rng=0,
  )
  rel_diff = abs(p_standard - p_simex['p_value']) / max(p_standard, 1e-300)
  assert rel_diff < 0.1, (
    f"MC-SIMEX logrank differs from cox_only under perfect classification: "
    f"standard={p_standard:.4f}, SIMEX={p_simex['p_value']:.4f}"
  )

  simex = datacard.km_hazard_ratio_mc_simex(
    parameter_threshold=threshold,
    B=1,
    rng=0,
  )
  estimate = simex.estimate_hazard_ratio()
  yi_hrs = np.logspace(-1, 1, 41)
  yi_2nlls = [
    datacard.km_hazard_ratio_yi(
      parameter_threshold=threshold,
      hazard_ratio=float(hr),
    ).x
    for hr in yi_hrs
  ]
  yi_best = float(yi_hrs[int(np.argmin(yi_2nlls))])
  assert np.isclose(estimate['hazard_ratio'], yi_best, rtol=0.15), (
    f"MC-SIMEX HR {estimate['hazard_ratio']:.3f} should match Yi/naive "
    f"{yi_best:.3f} when e_i = 0"
  )


def test_mc_simex_discrete_small_error_moves_away_from_one():
  """
  At e = 0.05, SIMEX HR moves away from 1 relative to naive hard labels.
  """
  datacard = kombine.datacard.Datacard.parse_datacard(
    datacards / "discrete_classes_hr_example_small.txt"
  )
  threshold = 0.5001
  hard_label_card = kombine.datacard.Datacard([
    Patient(
      survival_time=patient.time,
      censored=patient.censored,
      observable=FixedObservable(patient.observed_parameter),
    )
    for patient in datacard.patients
  ])
  naive_hr = hard_label_card.km_hazard_ratio_mc_simex(
    parameter_threshold=threshold,
    B=1,
    rng=0,
  ).estimate_hazard_ratio()['hazard_ratio']

  simex = datacard.km_hazard_ratio_mc_simex(
    parameter_threshold=threshold,
    B=200,
    rng=0,
  )
  simex_estimate = simex.estimate_hazard_ratio()

  yi_hrs = np.logspace(-1, 1, 41)
  yi_2nlls = [
    datacard.km_hazard_ratio_yi(
      parameter_threshold=threshold,
      hazard_ratio=float(hr),
    ).x
    for hr in yi_hrs
  ]
  yi_best = float(yi_hrs[int(np.argmin(yi_2nlls))])

  naive_log = abs(np.log(naive_hr))
  simex_log = abs(np.log(simex_estimate['hazard_ratio']))
  assert simex_log > naive_log, (
    f"SIMEX should move away from 1 relative to naive: "
    f"naive={naive_hr:.3f}, "
    f"SIMEX={simex_estimate['hazard_ratio']:.3f}"
  )
  assert not np.isclose(simex_estimate['hazard_ratio'], yi_best, rtol=1e-3), (
    "MC-SIMEX point HR should not be identical to Yi weights"
  )


def _high_noise_discrete_datacard() -> kombine.datacard.Datacard:
  baseline = kombine.datacard.Datacard.parse_datacard(
    datacards / "discrete_classes_hr_example_small.txt"
  )
  # Draw observed labels from the e ~ 0.5 misclassification process so the
  # naive (and therefore SIMEX) hard-label HR is already near 1.
  label_rng = np.random.default_rng(0)
  patients = []
  for patient in baseline.patients:
    observed_high = bool(label_rng.random() < 0.5)
    if observed_high:
      probs = [0.49, 0.51]
    else:
      probs = [0.51, 0.49]
    patients.append(Patient(
      survival_time=patient.time,
      censored=patient.censored,
      observable=DiscreteClassObservable(class_probs=probs),
    ))
  return kombine.datacard.Datacard(patients)


def test_mc_simex_high_noise_hr_near_one_finite_wald():
  """
  When e_i -> 0.5, the point HR is near 1 and the Wald CI stays finite.
  """
  datacard = _high_noise_discrete_datacard()
  simex = datacard.km_hazard_ratio_mc_simex(
    parameter_threshold=0.5001,
    B=80,
    rng=0,
  )
  estimate = simex.estimate_hazard_ratio()
  assert 0.25 < estimate['hazard_ratio'] < 4.0, (
    f"Expected HR near 1 under e~0.5, got {estimate['hazard_ratio']:.3f}"
  )
  assert np.isfinite(estimate['ci_lower'])
  assert np.isfinite(estimate['ci_upper'])
  assert estimate['ci_lower'] > 0.0
  assert estimate['ci_upper'] < np.inf
  assert estimate['ci_lower'] < estimate['ci_upper']


def test_mc_simex_rng_reproducible():
  """
  The same rng seed should reproduce logrank and HR results.
  """
  datacard = kombine.datacard.Datacard.parse_datacard(
    datacards / "discrete_classes_hr_example_small.txt"
  )
  threshold = 0.5001
  first = datacard.km_p_value_logrank_mc_simex(
    parameter_threshold=threshold,
    B=40,
    rng=7,
  )
  second = datacard.km_p_value_logrank_mc_simex(
    parameter_threshold=threshold,
    B=40,
    rng=7,
  )
  third = datacard.km_p_value_logrank_mc_simex(
    parameter_threshold=threshold,
    B=40,
    rng=8,
  )
  assert first['p_value'] == second['p_value']
  assert first['logrank_statistic'] == second['logrank_statistic']
  assert first['p_value'] != third['p_value']

  hr_a = datacard.km_hazard_ratio_mc_simex(
    parameter_threshold=threshold,
    B=40,
    rng=7,
  ).estimate_hazard_ratio()
  hr_b = datacard.km_hazard_ratio_mc_simex(
    parameter_threshold=threshold,
    B=40,
    rng=7,
  ).estimate_hazard_ratio()
  assert hr_a['hazard_ratio'] == hr_b['hazard_ratio']
  assert hr_a['se_log_hazard_ratio'] == hr_b['se_log_hazard_ratio']


def test_mc_simex_return_structure():
  """
  Return dicts / Wald 2NLL objects should have the documented keys.
  """
  datacard = kombine.datacard.Datacard.parse_datacard(
    datacards / "fixed_hr_example.txt"
  )
  threshold = 0.5

  km = datacard.km_survival_mc_simex(
    parameter_min=-np.inf,
    parameter_max=threshold,
    B=1,
    rng=0,
  )
  for key in (
    'survival_probabilities',
    'times_for_plot',
    'death_times',
    'method',
    'parameter_min',
    'parameter_max',
    'lambda_grid',
    'B',
  ):
    assert key in km, f"Missing KM key {key}"
  assert km['method'] == 'mc_simex'
  assert km['survival_probabilities'][0] == 1.0

  logrank = datacard.km_p_value_logrank_mc_simex(
    parameter_threshold=threshold,
    B=1,
    rng=0,
  )
  for key in (
    'p_value',
    'logrank_statistic',
    'n_low_observed',
    'n_high_observed',
    'method',
    'lambda_grid',
    'B',
  ):
    assert key in logrank, f"Missing logrank key {key}"
  assert 0.0 <= logrank['p_value'] <= 1.0
  assert logrank['logrank_statistic'] >= 0.0

  simex = datacard.km_hazard_ratio_mc_simex(
    parameter_threshold=threshold,
    B=1,
    rng=0,
  )
  estimate = simex.estimate_hazard_ratio()
  for key in (
    'hazard_ratio',
    'log_hazard_ratio',
    'se_log_hazard_ratio',
    'ci_lower',
    'ci_upper',
    'method',
  ):
    assert key in estimate, f"Missing HR key {key}"
  assert estimate['method'] == 'mc_simex'
  assert estimate['ci_lower'] < estimate['hazard_ratio'] < estimate['ci_upper']

  wald = simex.compute_2nll_at_hazard_ratio(estimate['hazard_ratio'])
  assert wald.method == 'mc_simex_wald'
  assert wald.x == 0.0 or abs(wald.x) < 1e-12
  wald_edge = estimate['hazard_ratio'] * np.exp(1.96 * estimate['se_log_hazard_ratio'])
  off_mle = simex.compute_2nll_at_hazard_ratio(wald_edge)
  assert np.isclose(off_mle.x, 1.96 ** 2, rtol=1e-6)


if __name__ == "__main__":
  print("Running MC-SIMEX tests...")

  print("Test 1: Poisson ratio not implemented...")
  test_mc_simex_poisson_ratio_not_implemented()
  print("[PASS] Poisson ratio not implemented")

  print("Test 2: KM matches nominal / Yi when e_i = 0...")
  test_mc_simex_km_matches_nominal_fixed_observable()
  print("[PASS] KM matches nominal / Yi when e_i = 0")

  print("Test 3: logrank and HR match hard labels...")
  test_mc_simex_logrank_and_hr_match_hard_labels()
  print("[PASS] logrank and HR match hard labels")

  print("Test 4: discrete e=0.05 moves away from 1...")
  test_mc_simex_discrete_small_error_moves_away_from_one()
  print("[PASS] discrete e=0.05 moves away from 1")

  print("Test 5: high-noise HR near 1 with finite Wald CI...")
  test_mc_simex_high_noise_hr_near_one_finite_wald()
  print("[PASS] high-noise HR near 1 with finite Wald CI")

  print("Test 6: rng reproducibility...")
  test_mc_simex_rng_reproducible()
  print("[PASS] rng reproducibility")

  print("Test 7: return structure...")
  test_mc_simex_return_structure()
  print("[PASS] return structure")

  print("\n[SUCCESS] All tests passed!")
