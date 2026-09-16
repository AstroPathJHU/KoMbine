# pylint: disable=duplicate-code
"""
Test Yi's misclassification correction methods.

This module tests the implementation of Yi's correction method (Section 3.7.1)
for discrete covariate misclassification in Kaplan-Meier survival analysis.
"""

import pathlib
import tempfile
import warnings

import numpy as np
import scipy.stats

import kombine.datacard
from kombine.datacard import (
  FixedObservable,
  PoissonDensityObservable,
  PoissonObservable,
  PoissonRatioObservable,
  Patient,
)
from kombine.comparisons import YiCorrectionBase, YiCorrectionForCoxPH
from kombine.utilities import prob_poisson_density_in_range
from ..utility_testing_functions import generate_two_group_datacard_from_hr

# Treat all warnings as errors
warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here / "datacards" / "simple_examples"


def test_prob_poisson_density_in_range_bayesian():
  """
  Test the Bayesian probability estimation for Poisson density range membership.
  """
  # Test 1: High count, clearly above threshold
  # count=100, area=1.0, threshold=50.0 => density ~ 100, should be > 0.99
  prob = prob_poisson_density_in_range(
    observed_count=100,
    observed_area=1.0,
    range_min=50.0,
    range_max=np.inf,
  )
  assert prob > 0.99, f"Expected prob > 0.99 for high count, got {prob}"

  # Test 2: Low count, clearly below threshold
  # count=10, area=1.0, threshold=50.0 => density ~ 10, should be ~0.0
  prob = prob_poisson_density_in_range(
    observed_count=10,
    observed_area=1.0,
    range_min=50.0,
    range_max=np.inf,
  )
  assert prob < 0.001, f"Expected prob < 0.001 for low count, got {prob}"

  # Test 3: Count near threshold (ambiguous)
  # count=50, area=1.0, threshold=48.0 => density ~ 50, threshold slightly below
  # Probability should be moderately high
  prob = prob_poisson_density_in_range(
    observed_count=50,
    observed_area=1.0,
    range_min=48.0,
    range_max=np.inf,
  )
  if not 0.5 < prob < 0.9:
    raise AssertionError(
      f"Expected prob in (0.5, 0.9) for count near threshold, got {prob}"
    )

  # Test 4: Zero count (edge case)
  prob = prob_poisson_density_in_range(
    observed_count=0,
    observed_area=1.0,
    range_min=10.0,
    range_max=np.inf,
  )
  assert prob < 0.01, f"Expected prob < 0.01 for zero count, got {prob}"

  # Test 5: Finite range captures middle probability mass
  prob = prob_poisson_density_in_range(
    observed_count=50,
    observed_area=1.0,
    range_min=40.0,
    range_max=60.0,
  )
  assert 0.6 < prob < 0.95, (
    f"Expected prob in (0.6, 0.95) for middle range, got {prob}"
  )


def test_observable_probability_in_range_fixed():
  """
  Fixed observables should use deterministic range membership.
  """
  obs = FixedObservable(5.0)
  assert obs.probability_in_range(0.0, 10.0) == 1.0
  assert obs.probability_in_range(5.0, 6.0) == 1.0
  assert obs.probability_in_range(6.0, 10.0) == 0.0


def test_observable_probability_in_range_poisson():
  """
  Poisson observables should return valid probabilities from posteriors.
  """
  obs = PoissonObservable(count=50, unique_id=1)
  prob_narrow = obs.probability_in_range(40.0, 60.0)
  prob_wide = obs.probability_in_range(0.0, 100.0)
  assert 0.0 < prob_narrow < 1.0
  assert 0.0 < prob_wide <= 1.0
  assert prob_wide >= prob_narrow


def test_observable_probability_in_range_poisson_density_matches_utility():
  """
  Poisson density observable should delegate to the utility function.
  """
  obs = PoissonDensityObservable(
    numerator=50,
    denominator=1.0,
    unique_id_numerator=1,
  )
  prob_obs = obs.probability_in_range(48.0, np.inf)
  prob_util = prob_poisson_density_in_range(
    observed_count=50,
    observed_area=1.0,
    range_min=48.0,
    range_max=np.inf,
  )
  assert np.isclose(prob_obs, prob_util)


def test_yi_correction_poisson_ratio_not_implemented():
  """
  Yi correction should raise for Poisson ratio observables.
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
  yi = YiCorrectionBase([patient])

  try:
    yi.compute_patient_prob_in_range(patient, 0.0, 1.0)
  except NotImplementedError:
    return
  raise AssertionError("Expected NotImplementedError for PoissonRatioObservable")


def generate_synthetic_datacard_with_perfect_classification( # pylint: disable=too-many-locals, too-many-arguments
  target_hr: float = 2.0,
  n_patients_per_group: int = 30,
  threshold: float = 100.0,
  random_seed: int = 42
) -> kombine.datacard.Datacard:
  """
  Generate a synthetic datacard with perfect classification (no measurement error).

  Uses 'fixed' observable type to ensure no measurement uncertainty.
  This allows testing Yi's method under the condition where it should
  reduce to the standard logrank test.
  """
  np.random.seed(random_seed)

  # Generate synthetic two-group datacard from target hazard ratio
  datacard_content = generate_two_group_datacard_from_hr(
    n_patients_per_group=n_patients_per_group,
    target_hr=target_hr,
    threshold=threshold,
    observable_range_0=(0.0, threshold - 50.0),
    observable_range_1=(threshold + 50.0, threshold + 200.0),
    censor_rate=0.3,
    seed=42,
  )

  # Write to temporary file and parse
  with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
    f.write(datacard_content)
    temp_path = pathlib.Path(f.name)

  datacard = kombine.datacard.Datacard.parse_datacard(temp_path)
  pathlib.Path(temp_path).unlink()  # Clean up temp file

  return datacard


def test_yi_logrank_perfect_classification():
  """
  Test Yi's logrank method with perfect classification.

  Under perfect classification (no measurement error), Yi's method should
  give essentially the same p-value as the standard logrank test.
  """
  datacard = generate_synthetic_datacard_with_perfect_classification(
    target_hr=2.5,
    n_patients_per_group=40,
    random_seed=123
  )

  threshold = 100.0

  # Standard logrank test
  p_value_standard = datacard.km_p_value_logrank(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
    cox_only=True
  )

  # Yi's corrected logrank test
  result_yi = datacard.km_p_value_logrank_yi(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
  )
  p_value_yi = result_yi['p_value']

  # With perfect classification, the p-values should be very similar
  # Allow 10% relative tolerance
  rel_diff = abs(p_value_standard - p_value_yi) / p_value_standard
  assert rel_diff < 0.1, (
    f"Yi's method differs from standard under perfect classification: "
    f"standard={p_value_standard:.4f}, Yi={p_value_yi:.4f}, "
    f"rel_diff={rel_diff:.2%}"
  )

  # Both should detect the hazard ratio difference (p < 0.05)
  assert p_value_standard < 0.05, f"Standard logrank should be significant, got {p_value_standard}"
  assert p_value_yi < 0.05, f"Yi's logrank should be significant, got {p_value_yi}"


def test_yi_hazard_ratio_perfect_classification():
  """
  Test Yi's hazard ratio method with perfect classification.

  Under perfect classification, Yi's method should give essentially
  the same 2NLL values as the standard MINLP method.
  """
  datacard = generate_synthetic_datacard_with_perfect_classification(
    target_hr=2.0,
    n_patients_per_group=50,
    random_seed=456
  )

  threshold = 100.0

  # Test at the true hazard ratio
  hr_test = 2.0

  # Standard MINLP method
  hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
  )
  result_standard = hr_calc.compute_2nll_at_hazard_ratio(hr_test, cox_only=True)
  twonll_standard = result_standard.x

  # Yi's corrected method
  result_yi = datacard.km_hazard_ratio_yi(
    parameter_threshold=threshold,
    hazard_ratio=hr_test,
    parameter_min=-np.inf,
    parameter_max=np.inf,
  )
  twonll_yi = result_yi.x

  # With perfect classification, 2NLL values should be similar
  # Allow 20% relative tolerance (Yi uses fractional weighting, MINLP uses integer optimization)
  rel_diff = abs(twonll_standard - twonll_yi) / twonll_standard
  assert rel_diff < 0.2, (
    f"Yi's 2NLL differs from standard under perfect classification: "
    f"standard={twonll_standard:.2f}, Yi={twonll_yi:.2f}, "
    f"rel_diff={rel_diff:.2%}"
  )


def test_yi_methods_return_valid_structure():
  """
  Test that Yi's methods return properly structured results.
  """
  datacard = generate_synthetic_datacard_with_perfect_classification(
    target_hr=1.8,
    n_patients_per_group=25,
    random_seed=999
  )

  threshold = 100.0

  # Test logrank result structure
  result_logrank = datacard.km_p_value_logrank_yi(
    parameter_threshold=threshold
  )

  # Check required keys (new implementation uses per-patient probabilities)
  required_keys = [
    'p_value', 'logrank_statistic', 'U', 'V',
    'n_low_observed', 'n_high_observed'
  ]
  for key in required_keys:
    assert key in result_logrank, f"Missing key '{key}' in logrank result"

  # Check types
  assert isinstance(result_logrank['p_value'], float)
  assert 0 <= result_logrank['p_value'] <= 1
  assert isinstance(result_logrank['logrank_statistic'], (int, float))
  assert result_logrank['logrank_statistic'] >= 0

  # Test hazard ratio result structure
  result_hr = datacard.km_hazard_ratio_yi(
    parameter_threshold=threshold,
    hazard_ratio=2.0
  )

  # Check required attributes
  assert hasattr(result_hr, 'x')  # 2NLL value
  assert hasattr(result_hr, 'success')
  assert hasattr(result_hr, 'hazard_ratio')
  assert hasattr(result_hr, 'log_hazard_ratio')
  assert hasattr(result_hr, 'cox_2NLL')

  # Check types
  assert isinstance(result_hr.x, (int, float))
  assert result_hr.x >= 0
  assert result_hr.success is True
  assert np.isclose(result_hr.hazard_ratio, 2.0)
  assert np.isclose(result_hr.log_hazard_ratio, np.log(2.0))


def test_yi_km_matches_nominal_fixed_observable():
  """
  Yi KM curves should match nominal KM for fixed observables.

  With deterministic group assignment, Yi's weighting reduces to a
  standard KM curve for each group when using all patients.
  """
  datacard = kombine.datacard.Datacard.parse_datacard(
    datacards / "fixed_hr_example.txt"
  )

  threshold = 0.5

  yi_low = datacard.km_survival_yi(
    parameter_min=-np.inf,
    parameter_max=threshold,
  )
  yi_high = datacard.km_survival_yi(
    parameter_min=threshold,
    parameter_max=np.inf,
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
    times_for_plot=yi_low['times_for_plot']
  )
  high_nominal = km_high.nominalkm.survival_probabilities(
    times_for_plot=yi_high['times_for_plot']
  )

  np.testing.assert_allclose(
    yi_low['survival_probabilities'],
    low_nominal,
    atol=1e-10,
    rtol=1e-10,
  )
  np.testing.assert_allclose(
    yi_high['survival_probabilities'],
    high_nominal,
    atol=1e-10,
    rtol=1e-10,
  )


def test_yi_bayesian_method_only():
  """
  Test that the Bayesian method works correctly.
  """
  datacard = generate_synthetic_datacard_with_perfect_classification(
    target_hr=1.5,
    n_patients_per_group=30,
    random_seed=111
  )

  threshold = 100.0

  # Test Bayesian method (now the only method)
  result_bayes = datacard.km_p_value_logrank_yi(
    parameter_threshold=threshold,
  )
  assert 'p_value' in result_bayes
  assert 0 <= result_bayes['p_value'] <= 1


def test_yi_kaplan_meier_survival():
  """
  Test Yi's weighted Kaplan-Meier survival probability estimation.

  This tests the new km_survival_yi() method which applies Yi's inverse
  probability weighting to Kaplan-Meier curve estimation.
  """
  # Use a dataset with fixed observables for perfect classification test
  datacardfile = datacards / "fixed_km_censoring.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(datacardfile)

  threshold = 0.5

  # Calculate Yi's weighted KM survival for the low group
  result_yi_low = datacard.km_survival_yi(
    parameter_min=-np.inf,
    parameter_max=threshold,  # Only low group
  )

  # Validate return structure
  assert 'survival_probabilities' in result_yi_low
  assert 'times_for_plot' in result_yi_low
  assert 'n_at_risk_weighted' in result_yi_low
  assert 'n_deaths_weighted' in result_yi_low
  assert 'n_at_risk' in result_yi_low
  assert 'n_deaths' in result_yi_low
  assert 'death_times' in result_yi_low
  assert 'method' in result_yi_low
  assert result_yi_low['method'] == 'yi_correction'

  # Validate survival probabilities are valid (between 0 and 1, monotonically decreasing)
  surv_probs = result_yi_low['survival_probabilities']
  assert len(surv_probs) == len(result_yi_low['times_for_plot'])
  assert all(0 <= p <= 1 for p in surv_probs), "Survival probabilities must be in [0, 1]"
  assert surv_probs[0] == 1.0, "Initial survival probability should be 1.0"

  # Check monotonically non-increasing
  for i in range(len(surv_probs) - 1):
    assert surv_probs[i] >= surv_probs[i + 1], (
      f"Survival probabilities must be non-increasing: "
      f"S(t={result_yi_low['times_for_plot'][i]})={surv_probs[i]:.4f} > "
      f"S(t={result_yi_low['times_for_plot'][i+1]})={surv_probs[i+1]:.4f}"
    )

  # Test high group as well
  result_yi_high = datacard.km_survival_yi(
    parameter_min=threshold,
    parameter_max=np.inf,  # Only high group
  )

  assert 'survival_probabilities' in result_yi_high
  assert result_yi_high['survival_probabilities'][0] == 1.0

  # For fixed observables with perfect classification, weighted counts should
  # approximately match unweighted counts (within rounding)
  # Note: Some patients may have weight 0 if they don't belong to the filtered group,
  # so weighted counts can be 0 even if unweighted is >0. We check that the weighted
  # curve is monotonically decreasing and starts at 1.0 instead.

  # Verify that weighted death counts are non-negative
  assert all(n >= 0 for n in result_yi_low['n_deaths_weighted']), (
    f"Weighted death counts must be non-negative: {result_yi_low['n_deaths_weighted']}"
  )
  assert all(n >= 0 for n in result_yi_low['n_at_risk_weighted']), (
    f"Weighted at-risk counts must be non-negative: {result_yi_low['n_at_risk_weighted']}"
  )

  print(f"  Yi's KM low group: {len(result_yi_low['death_times'])} death times")
  print(f"  Yi's KM high group: {len(result_yi_high['death_times'])} death times")
  print(f"  Final survival (low): {result_yi_low['survival_probabilities'][-1]:.4f}")
  print(f"  Final survival (high): {result_yi_high['survival_probabilities'][-1]:.4f}")


def test_yi_continuous_hr_matches_naive_breslow_fixed_card():
  """
  On a fixed observable, Yi's continuous Breslow MLE matches MC-SIMEX.
  """
  datacard = kombine.datacard.Datacard.parse_datacard(
    datacards / "fixed_hr_example.txt"
  )
  threshold = 0.5001
  yi_calc = YiCorrectionForCoxPH(
    patients=datacard.patients,
    parameter_min=-np.inf,
    parameter_max=np.inf,
    parameter_threshold=threshold,
  )
  best_hr_yi, lower_ci, upper_ci, best_fit = yi_calc.hazard_ratio_confidence_interval(
    confidence_level=0.95,
    hazard_ratio_min=0.01,
    hazard_ratio_max=100.0,
  )
  simex = datacard.km_hazard_ratio_mc_simex(
    parameter_threshold=threshold,
    parameter_min=-np.inf,
    parameter_max=np.inf,
    B=1,
    rng=0,
  )
  estimate = simex.estimate_hazard_ratio()
  np.testing.assert_allclose(
    best_hr_yi,
    estimate['hazard_ratio'],
    rtol=1e-4,
    atol=0.0,
  )
  chi2_95 = float(scipy.stats.chi2.ppf(0.95, df=1))
  delta_lower = yi_calc.compute_2nll_at_hazard_ratio(lower_ci).x - best_fit.x
  delta_upper = yi_calc.compute_2nll_at_hazard_ratio(upper_ci).x - best_fit.x
  np.testing.assert_allclose(delta_lower, chi2_95, rtol=1e-3, atol=1e-2)
  np.testing.assert_allclose(delta_upper, chi2_95, rtol=1e-3, atol=1e-2)
  assert lower_ci < best_hr_yi < upper_ci


if __name__ == "__main__":
  # Run tests
  print("Running Yi correction tests...")

  print("Test 1: Bayesian probability estimation...")
  test_prob_poisson_density_in_range_bayesian()
  print("[PASS] Bayesian probability estimation")

  print("Test 2: Yi logrank with perfect classification...")
  test_yi_logrank_perfect_classification()
  print("[PASS] Yi logrank with perfect classification")

  print("Test 3: Yi hazard ratio with perfect classification...")
  test_yi_hazard_ratio_perfect_classification()
  print("[PASS] Yi hazard ratio with perfect classification")

  print("Test 4: Result structure validation...")
  test_yi_methods_return_valid_structure()
  print("[PASS] Result structure validation")

  print("Test 5: Yi KM matches nominal with fixed observable...")
  test_yi_km_matches_nominal_fixed_observable()
  print("[PASS] Yi KM matches nominal with fixed observable")

  print("Test 6: Bayesian method works...")
  test_yi_bayesian_method_only()
  print("[PASS] Bayesian method works")

  print("Test 7: Yi's Kaplan-Meier survival probabilities...")
  test_yi_kaplan_meier_survival()
  print("[PASS] Yi's Kaplan-Meier survival probabilities")

  print("Test 8: Continuous Yi HR matches naive Breslow on a fixed card...")
  test_yi_continuous_hr_matches_naive_breslow_fixed_card()
  print("[PASS] Continuous Yi HR matches naive Breslow on a fixed card")

  print("\n[SUCCESS] All tests passed!")
