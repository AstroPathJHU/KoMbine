"""
Test Yi's misclassification correction methods.

This module tests the implementation of Yi's correction method (Section 3.7.1)
for discrete covariate misclassification in Kaplan-Meier survival analysis.
"""

import pathlib
import tempfile
import warnings

import numpy as np

import kombine.datacard
from kombine.utilities import (
  prob_poisson_density_exceeds_threshold,
  estimate_misclassification_matrix,
  invert_misclassification_matrix,
)

# Treat all warnings as errors
warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here / "datacards" / "simple_examples"


def test_prob_poisson_density_exceeds_threshold_bayesian():
  """
  Test the Bayesian probability estimation for Poisson density threshold crossing.
  """
  # Test 1: High count, clearly above threshold
  # count=100, area=1.0, threshold=50.0 => density ~ 100, should be > 0.99
  prob = prob_poisson_density_exceeds_threshold(
    observed_count=100,
    observed_area=1.0,
    threshold=50.0,
    method='bayesian',
  )
  assert prob > 0.99, f"Expected prob > 0.99 for high count, got {prob}"

  # Test 2: Low count, clearly below threshold
  # count=10, area=1.0, threshold=50.0 => density ~ 10, should be ~0.0
  prob = prob_poisson_density_exceeds_threshold(
    observed_count=10,
    observed_area=1.0,
    threshold=50.0,
    method='bayesian',
  )
  assert prob < 0.001, f"Expected prob < 0.001 for low count, got {prob}"

  # Test 3: Count near threshold (ambiguous)
  # count=50, area=1.0, threshold=48.0 => density ~ 50, threshold slightly below
  # Probability should be moderately high
  prob = prob_poisson_density_exceeds_threshold(
    observed_count=50,
    observed_area=1.0,
    threshold=48.0,
    method='bayesian',
  )
  if not (0.5 < prob < 0.9):
    raise AssertionError(
      f"Expected prob in (0.5, 0.9) for count near threshold, got {prob}"
    )

  # Test 4: Zero count (edge case)
  prob = prob_poisson_density_exceeds_threshold(
    observed_count=0,
    observed_area=1.0,
    threshold=10.0,
    method='bayesian',
  )
  assert prob < 0.01, f"Expected prob < 0.01 for zero count, got {prob}"


def test_prob_poisson_density_exceeds_threshold_normal_approx():
  """
  Test the normal approximation for Poisson density threshold crossing.
  """
  # Test with moderately large counts where normal approximation should work
  # count=100, area=1.0, threshold=50.0
  prob_bayes = prob_poisson_density_exceeds_threshold(
    observed_count=100,
    observed_area=1.0,
    threshold=50.0,
    method='bayesian',
  )
  prob_normal = prob_poisson_density_exceeds_threshold(
    observed_count=100,
    observed_area=1.0,
    threshold=50.0,
    method='normal_approx',
  )

  # Normal approximation should be close to Bayesian for large counts
  assert abs(prob_bayes - prob_normal) < 0.05, (
    f"Bayesian and normal approx differ: {prob_bayes} vs {prob_normal}"
  )


def generate_synthetic_datacard_with_perfect_classification(
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

  # Base hazard rate for Group 0
  lambda_0 = 0.1

  # Hazard rate for Group 1 to achieve target HR
  lambda_1 = target_hr * lambda_0

  # Generate survival times from exponential distribution
  # Group 0: well below threshold
  survival_times_0 = np.random.exponential(1.0 / lambda_0, size=n_patients_per_group)
  observables_0 = np.random.uniform(0.0, threshold - 50.0, size=n_patients_per_group)

  # Group 1: well above threshold
  survival_times_1 = np.random.exponential(1.0 / lambda_1, size=n_patients_per_group)
  observables_1 = np.random.uniform(threshold + 50.0, threshold + 200.0, size=n_patients_per_group)

  # Randomly censor ~30% of patients
  censored_0 = np.random.random(n_patients_per_group) < 0.3
  censored_1 = np.random.random(n_patients_per_group) < 0.3

  # Combine all data
  all_survival_times = np.concatenate([survival_times_0, survival_times_1])
  all_censored = np.concatenate([censored_0, censored_1])
  all_observables = np.concatenate([observables_0, observables_1])

  # Format data
  survival_times_str = '\t'.join(f'{t:.4f}' for t in all_survival_times)
  censored_str = '\t'.join('1' if c else '0' for c in all_censored)
  observables_str = '\t'.join(f'{o:.4f}' for o in all_observables)

  datacard_content = f"""observable_type fixed
------------
# Synthetic datacard with perfect classification
# Group 0 (well below {threshold}): lambda_0 = {lambda_0}
# Group 1 (well above {threshold}): lambda_1 = {lambda_1}
# Expected HR = {target_hr}
------------
survival_time\t{survival_times_str}
censored\t{censored_str}
observable\t{observables_str}
"""

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
    method='bayesian'
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
    method='bayesian'
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


def test_misclassification_matrix_identity_for_perfect_data():
  """
  Test that the misclassification matrix is close to identity for data
  with perfect separation.
  """
  datacard = generate_synthetic_datacard_with_perfect_classification(
    target_hr=1.5,
    n_patients_per_group=30,
    random_seed=789
  )

  threshold = 100.0

  # Get patient NLL objects
  patients = [p.get_nll() for p in datacard.patients]

  # Estimate misclassification matrix
  Pi = estimate_misclassification_matrix(
    patients,
    threshold,
    method='bayesian'
  )

  # With perfect separation, Pi should be close to identity
  # Pi[0, 0] ~ 1 (true low, observed low)
  # Pi[1, 1] ~ 1 (true high, observed high)
  # Pi[0, 1] ~ 0 (true low, observed high)
  # Pi[1, 0] ~ 0 (true high, observed low)

  assert Pi[0, 0] > 0.95, f"Expected Pi[0,0] > 0.95, got {Pi[0, 0]}"
  assert Pi[1, 1] > 0.95, f"Expected Pi[1,1] > 0.95, got {Pi[1, 1]}"
  assert Pi[0, 1] < 0.05, f"Expected Pi[0,1] < 0.05, got {Pi[0, 1]}"
  assert Pi[1, 0] < 0.05, f"Expected Pi[1,0] < 0.05, got {Pi[1, 0]}"

  # Test that inverse exists and is close to identity
  Pi_inv = invert_misclassification_matrix(Pi)

  # Pi @ Pi_inv should be close to identity
  product = Pi @ Pi_inv
  identity = np.eye(2)

  assert np.allclose(product, identity, atol=0.01), (
    f"Pi @ Pi_inv should be identity, got:\n{product}"
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

  # Check required keys
  required_keys = [
    'p_value', 'logrank_statistic', 'U', 'V',
    'misclassification_matrix', 'inverse_misclassification_matrix',
    'n_low_observed', 'n_high_observed'
  ]
  for key in required_keys:
    assert key in result_logrank, f"Missing key '{key}' in logrank result"

  # Check types
  assert isinstance(result_logrank['p_value'], float)
  assert 0 <= result_logrank['p_value'] <= 1
  assert isinstance(result_logrank['logrank_statistic'], (int, float))
  assert result_logrank['logrank_statistic'] >= 0
  assert isinstance(result_logrank['misclassification_matrix'], np.ndarray)
  assert result_logrank['misclassification_matrix'].shape == (2, 2)

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
  assert hasattr(result_hr, 'misclassification_matrix')

  # Check types
  assert isinstance(result_hr.x, (int, float))
  assert result_hr.x >= 0
  assert result_hr.success is True
  assert np.isclose(result_hr.hazard_ratio, 2.0)
  assert np.isclose(result_hr.log_hazard_ratio, np.log(2.0))


def test_yi_both_methods_available():
  """
  Test that both Bayesian and normal approximation methods work.
  """
  datacard = generate_synthetic_datacard_with_perfect_classification(
    target_hr=1.5,
    n_patients_per_group=30,
    random_seed=111
  )

  threshold = 100.0

  # Test Bayesian method
  result_bayes = datacard.km_p_value_logrank_yi(
    parameter_threshold=threshold,
    method='bayesian'
  )
  assert 'p_value' in result_bayes
  assert 0 <= result_bayes['p_value'] <= 1

  # Test normal approximation method
  result_normal = datacard.km_p_value_logrank_yi(
    parameter_threshold=threshold,
    method='normal_approx'
  )
  assert 'p_value' in result_normal
  assert 0 <= result_normal['p_value'] <= 1

  # For perfect classification with large sample, methods should agree reasonably well
  p_diff = abs(result_bayes['p_value'] - result_normal['p_value'])
  assert p_diff < 0.15, (
    f"Bayesian and normal approx differ too much: "
    f"bayes={result_bayes['p_value']:.4f}, normal={result_normal['p_value']:.4f}"
  )


if __name__ == "__main__":
  # Run tests
  print("Running Yi correction tests...")

  print("Test 1: Bayesian probability estimation...")
  test_prob_poisson_density_exceeds_threshold_bayesian()
  print("✓ Passed")

  print("Test 2: Normal approximation...")
  test_prob_poisson_density_exceeds_threshold_normal_approx()
  print("✓ Passed")

  print("Test 3: Yi logrank with perfect classification...")
  test_yi_logrank_perfect_classification()
  print("✓ Passed")

  print("Test 4: Yi hazard ratio with perfect classification...")
  test_yi_hazard_ratio_perfect_classification()
  print("✓ Passed")

  print("Test 5: Misclassification matrix identity check...")
  test_misclassification_matrix_identity_for_perfect_data()
  print("✓ Passed")

  print("Test 6: Result structure validation...")
  test_yi_methods_return_valid_structure()
  print("✓ Passed")

  print("Test 7: Both estimation methods...")
  test_yi_both_methods_available()
  print("✓ Passed")

  print("\n✅ All tests passed!")
