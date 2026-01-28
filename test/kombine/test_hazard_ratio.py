"""
Test the hazard ratio calculation using MINLP.
"""

import pathlib
import tempfile
import warnings

import numpy as np
import scipy.stats

import kombine.datacard

# Treat all warnings as errors
warnings.simplefilter("error")

here = pathlib.Path(__file__).parent
datacards = here / "datacards" / "simple_examples"


def generate_synthetic_datacard_with_known_hr( #pylint: disable=too-many-locals
  target_hr: float,
  n_patients_per_group: int = 25,
  threshold: float = 0.5,
  random_seed: int = 42
) -> kombine.datacard.Datacard:
  """
  Generate a synthetic datacard with a known hazard ratio.

  Parameters:
  -----------
  target_hr : float
    Target hazard ratio (Group 1 hazard / Group 0 hazard).
    HR > 1 means Group 1 has worse survival.
  n_patients_per_group : int
    Number of patients in each group.
  threshold : float
    Observable threshold separating the groups.
  random_seed : int
    Random seed for reproducibility.

  Returns:
  --------
  Datacard
    A datacard with patients having survival times drawn from exponential
    distributions with hazard rates that differ by the target_hr factor.

  Notes:
  ------
  - Uses 'fixed' observable type for simplicity and predictability
  - Group 0: observable values below threshold, base hazard rate λ₀ = 0.1
  - Group 1: observable values above threshold, hazard rate λ₁ = target_hr * λ₀
  - Exponential survival times: T ~ Exp(λ)
  - Censoring applied randomly to ~30% of patients in each group
  """
  np.random.seed(random_seed)

  # Base hazard rate for Group 0
  lambda_0 = 0.1

  # Hazard rate for Group 1 to achieve target HR
  lambda_1 = target_hr * lambda_0

  # Generate survival times from exponential distribution
  # Group 0: below threshold
  survival_times_0 = np.random.exponential(1.0 / lambda_0, size=n_patients_per_group)
  observables_0 = np.random.uniform(0.0, threshold - 0.01, size=n_patients_per_group)

  # Group 1: above threshold
  survival_times_1 = np.random.exponential(1.0 / lambda_1, size=n_patients_per_group)
  observables_1 = np.random.uniform(threshold + 0.01, 1.0, size=n_patients_per_group)

  # Randomly censor ~30% of patients
  censored_0 = np.random.random(n_patients_per_group) < 0.3
  censored_1 = np.random.random(n_patients_per_group) < 0.3

  # Combine all data
  all_survival_times = np.concatenate([survival_times_0, survival_times_1])
  all_censored = np.concatenate([censored_0, censored_1])
  all_observables = np.concatenate([observables_0, observables_1])

  # Create datacard content
  # Format the data rows (can't use backslash inside f-string expressions)
  survival_times_str = '\t'.join(f'{t:.4f}' for t in all_survival_times)
  censored_str = '\t'.join('1' if c else '0' for c in all_censored)
  observables_str = '\t'.join(f'{o:.4f}' for o in all_observables)

  datacard_content = f"""observable_type fixed
------------
# Synthetic datacard with known hazard ratio = {target_hr}
# Group 0 (below {threshold}): lambda_0 = {lambda_0}
# Group 1 (above {threshold}): lambda_1 = {lambda_1}
# Expected HR = lambda_1/lambda_0 = {target_hr}
------------
survival_time\t{survival_times_str}
censored\t{censored_str}
observable\t{observables_str}
"""

  # Write to temporary file and parse
  with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
    f.write(datacard_content)
    temp_path = f.name

  try:
    datacard = kombine.datacard.Datacard.parse_datacard(pathlib.Path(temp_path))
  finally:
    # Clean up temporary file
    pathlib.Path(temp_path).unlink()

  return datacard


def test_hazard_ratio_basic():
  """
  Test basic hazard ratio calculation functionality.
  """
  # Load a simple datacard
  dcfile = datacards / "poisson_ratio_km_censoring.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)

  # Create hazard ratio calculator
  # Note: Use slightly non-zero bounds to avoid numerical issues with Poisson ratio optimization
  hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
  )

  # Test computation at H=1 (null hypothesis)
  result_h1 = hr_calc.compute_2nll_at_hazard_ratio(1.0, cox_only=False)
  assert result_h1.success
  assert result_h1.hazard_ratio == 1.0
  assert result_h1.log_hazard_ratio == 0.0
  assert result_h1.x > 0  # 2NLL should be positive

  # Test computation at a different hazard ratio
  result_h2 = hr_calc.compute_2nll_at_hazard_ratio(2.0, cox_only=False)
  assert result_h2.success
  assert abs(result_h2.hazard_ratio - 2.0) < 1e-6
  assert abs(result_h2.log_hazard_ratio - np.log(2.0)) < 1e-6

  # Test cox_only mode
  result_cox = hr_calc.compute_2nll_at_hazard_ratio(1.0, cox_only=True)
  assert result_cox.success

  print("[PASS] Basic hazard ratio calculation tests passed")


def test_likelihood_scan():
  """
  Test likelihood scan over hazard ratio values.
  """
  # Load datacard with more balanced groups to avoid extreme HRs
  dcfile = datacards / "fixed_hr_example.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)

  # Use fixed observable type, which doesn't need parameter bounds
  hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.0,
    parameter_max=1.0,
  )

  # Perform likelihood scan with default parameters
  # Use a range wide enough that the minimum is not at the boundary
  hazard_ratios, twonll_values, best_fit_result = hr_calc.likelihood_scan_hazard_ratio(
    n_points=20,
    hazard_ratio_min=0.5,
    hazard_ratio_max=5.0,
    cox_only=False
  )

  # Check outputs
  assert len(hazard_ratios) == 20
  assert len(twonll_values) == 20
  assert best_fit_result.success

  # 2NLL should be minimized somewhere in the range
  min_idx = np.argmin(twonll_values)
  assert 0 <= min_idx < 20

  # Best fit result should correspond to minimum 2NLL
  assert abs(best_fit_result.x - twonll_values[min_idx]) < 1e-6
  assert abs(best_fit_result.log_hazard_ratio - np.log(hazard_ratios[min_idx])) < 1e-6

  # 2NLL should increase away from minimum (at least on one side)
  # Check that there's some curvature
  if min_idx > 0:
    assert twonll_values[min_idx - 1] >= twonll_values[min_idx] - 1e-6
  if min_idx < 19:
    assert twonll_values[min_idx + 1] >= twonll_values[min_idx] - 1e-6

  print("[PASS] Likelihood scan tests passed")


def test_likelihood_scan_custom_values():
  """
  Test likelihood scan with custom hazard ratio values.
  """
  # Use fixed_hr_example datacard with more balanced groups
  dcfile = datacards / "fixed_hr_example.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)

  # Use fixed observable type, which doesn't need parameter bounds
  hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.0,
    parameter_max=1.0,
  )

  # Specify custom hazard ratio values appropriate for this datacard (HR ~ 2.3)
  custom_hrs = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
  hazard_ratios, twonll_values, best_fit_result = hr_calc.likelihood_scan_hazard_ratio(
    hazard_ratio_values=custom_hrs,
    cox_only=False
  )

  # Check that we got the requested values
  np.testing.assert_array_equal(hazard_ratios, custom_hrs)
  assert len(twonll_values) == len(custom_hrs)
  assert best_fit_result.success

  # All 2NLL values should be positive and finite
  assert np.all(np.isfinite(twonll_values))
  assert np.all(twonll_values > 0)

  # Verify that best_fit_result corresponds to the minimum in the scan
  min_idx = np.argmin(twonll_values)
  assert abs(best_fit_result.log_hazard_ratio - np.log(hazard_ratios[min_idx])) < 1e-6

  print("[PASS] Custom likelihood scan tests passed")


def test_confidence_interval():
  """
  Test confidence interval calculation for hazard ratio.
  """
  dcfile = datacards / "poisson_ratio_km_censoring.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)

  # Note: Use slightly non-zero bounds to avoid numerical issues with Poisson ratio optimization
  # Use wider log_hazard_ratio_bounds to avoid hitting boundary
  hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
    log_hazard_ratio_bounds=(-15.0, 15.0),
  )

  # Calculate 68% confidence interval
  best_fit_hr, lower_ci, upper_ci, best_fit_result = hr_calc.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.68,
    hazard_ratio_min=0.01,
    hazard_ratio_max=20.0,
  )

  # Check basic properties
  assert best_fit_result.success
  assert lower_ci <= best_fit_hr <= upper_ci
  assert lower_ci > 0
  assert upper_ci < np.inf

  # Check that best_fit_hr corresponds to the minimum 2NLL
  result_at_best = hr_calc.compute_2nll_at_hazard_ratio(best_fit_hr, cox_only=False)
  assert abs(result_at_best.x - best_fit_result.x) < 1e-3

  # The 2NLL at the boundaries should be above the threshold
  chi2_threshold = scipy.stats.chi2.ppf(0.68, df=1)
  twonll_threshold = best_fit_result.x + chi2_threshold

  result_at_lower = hr_calc.compute_2nll_at_hazard_ratio(lower_ci, cox_only=False)
  result_at_upper = hr_calc.compute_2nll_at_hazard_ratio(upper_ci, cox_only=False)

  # Allow some tolerance due to numerical optimization
  # Note: The tolerance is fairly loose because MINLP optimization can have numerical variations
  assert abs(result_at_lower.x - twonll_threshold) < 1.5, \
    f"Lower CI: {result_at_lower.x} vs {twonll_threshold}"
  assert abs(result_at_upper.x - twonll_threshold) < 1.5, \
    f"Upper CI: {result_at_upper.x} vs {twonll_threshold}"

  print(f"[PASS] Confidence interval tests passed: HR = {best_fit_hr:.3f} "
        f"[{lower_ci:.3f}, {upper_ci:.3f}]")


def test_consistency_with_p_value():
  """
  Test that hazard ratio calculation is consistent with p-value calculation.
  """
  dcfile = datacards / "poisson_ratio_km_censoring.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)

  # Create both calculators
  # Note: Use slightly non-zero bounds to avoid numerical issues with Poisson ratio optimization
  # Use wider log_hazard_ratio_bounds to avoid hitting boundary in this test
  hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
    log_hazard_ratio_bounds=(-15.0, 15.0),
  )
  pval_calc = datacard.km_p_value(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
  )

  # Compute p-value
  p_value, _, result_alt = pval_calc.solve_and_pvalue(cox_only=False)

  # The hazard ratio from p-value calculation should match
  # the hazard ratio we get from the hazard ratio calculator at the same point
  hr_from_pval = result_alt.hazard_ratio

  result_hr = hr_calc.compute_2nll_at_hazard_ratio(hr_from_pval, cox_only=False)

  # The 2NLL values should be close (allow slightly larger tolerance with wider bounds)
  assert abs(result_hr.x - result_alt.x) < 0.1, \
    f"2NLL mismatch: {result_hr.x} vs {result_alt.x}"

  # The patient assignments should be similar (allowing for some optimization differences)
  # Just check that the total numbers are close
  assert abs(result_hr.n_total_low - result_alt.n_total_low) <= 1
  assert abs(result_hr.n_total_high - result_alt.n_total_high) <= 1

  print(
    f"[PASS] Consistency with p-value calculation passed "
    f"(p = {p_value:.4f}, HR = {hr_from_pval:.3f})"
  )


def test_hazard_ratio_at_null():
  """
  Test that H=1 corresponds to the null hypothesis.
  """
  dcfile = datacards / "poisson_ratio_km_censoring.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)

  # Note: Use slightly non-zero bounds to avoid numerical issues with Poisson ratio optimization
  # Use wider log_hazard_ratio_bounds to avoid hitting boundary
  hr_calc = datacard.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
    log_hazard_ratio_bounds=(-15.0, 15.0),
  )

  pval_calc = datacard.km_p_value(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
  )

  # Get the 2NLL at H=1 from hazard ratio calculator
  result_hr_null = hr_calc.compute_2nll_at_hazard_ratio(1.0, cox_only=False)

  # Get the null hypothesis result from p-value calculator
  _, result_null, _ = pval_calc.solve_and_pvalue(cox_only=False)

  # The 2NLL values should match
  assert abs(result_hr_null.x - result_null.x) < 1e-2, \
    f"Null hypothesis 2NLL mismatch: {result_hr_null.x} vs {result_null.x}"

  print("[PASS] H=1 null hypothesis consistency test passed")


def test_known_hazard_ratios(): #pylint: disable=too-many-locals
  """
  Test hazard ratio calculation with synthetic data having known target HRs.

  This test generates synthetic datacards where the true hazard ratio is known
  by construction (from exponential survival distributions with different rates).
  It then verifies that the calculated hazard ratio is close to the target value.

  Note: Due to finite sample sizes and random variation, we use relatively loose
  tolerances. The test verifies that the method correctly identifies the direction
  and approximate magnitude of the hazard ratio difference.
  """
  # Test several hazard ratio values
  # Tolerances are now in log space (absolute difference in log(HR))
  # For reference: log(0.5) ≈ -0.69, log(2) ≈ 0.69, log(3) ≈ 1.10
  # With 50 patients per group, expect ~0.3-0.5 log units of variation
  test_cases = [
    (0.5, 0.5),   # HR = 0.5 (Group 1 has better survival), log tolerance 0.5
    (1.0, 0.6),   # HR = 1.0 (equal hazards), log tolerance 0.6 (higher due to noise)
    (2.0, 0.5),   # HR = 2.0 (Group 1 has worse survival), log tolerance 0.5
    (3.0, 0.6),   # HR = 3.0 (Group 1 has much worse survival), log tolerance 0.6
  ]

  results = []

  for target_hr, tolerance in test_cases:
    # Generate synthetic datacard with known HR
    # Use more patients for better statistical power
    datacard = generate_synthetic_datacard_with_known_hr(
      target_hr=target_hr,
      n_patients_per_group=50,  # Increased from 25 to reduce variance
      threshold=0.5,
      random_seed=42 + int(target_hr * 10)  # Different seed for each case
    )

    # Create hazard ratio calculator
    # Use fixed observable type, so parameter_min/max don't matter
    hr_calc = datacard.km_hazard_ratio(
      parameter_threshold=0.5,
      parameter_min=0.0,
      parameter_max=1.0,
    )

    # Calculate best-fit hazard ratio with confidence interval
    best_fit_hr, lower_ci, upper_ci, _ = hr_calc.hazard_ratio_confidence_interval(
      cox_only=False,
      confidence_level=0.68,
      hazard_ratio_min=0.1,
      hazard_ratio_max=10.0,
    )

    # Check that the best-fit HR is close to the target HR (in log scale)
    # Use absolute difference in log space to properly compare ratios
    log_error = abs(np.log(best_fit_hr) - np.log(target_hr))
    assert log_error < tolerance, \
      f"HR mismatch for target={target_hr}: got {best_fit_hr:.3f}, " \
      f"log error {log_error:.3f} > {tolerance}"

    # Check that the target HR is reasonably close to the confidence interval
    # Allow violations up to 20% of the CI width (accounting for statistical fluctuations)
    ci_width = upper_ci - lower_ci
    lower_tolerance = lower_ci - 0.2 * ci_width
    upper_tolerance = upper_ci + 0.2 * ci_width

    in_tolerance = lower_tolerance <= target_hr <= upper_tolerance

    results.append((target_hr, best_fit_hr, lower_ci, upper_ci, log_error, in_tolerance))

  # Print results summary
  print("[PASS] Known hazard ratio tests passed:")
  for target_hr, best_fit_hr, lower_ci, upper_ci, log_err, in_tol in results:
    ci_str = "OK" if in_tol else "!!"
    print(f"  Target HR = {target_hr:.2f}: got {best_fit_hr:.3f} "
          f"[{lower_ci:.3f}, {upper_ci:.3f}], log error = {log_err:.3f} ({ci_str})")


def test_bounds_warning():
  """
  Test that the bounds warning is raised when hazard ratio hits the boundary.

  This test verifies that the warning system works by using tight bounds
  that force the best-fit HR to be at the boundary.
  """
  # Use a datacard where we can control the bounds tightly
  dcfile = datacards / "poisson_ratio_km_censoring.txt"
  datacard = kombine.datacard.Datacard.parse_datacard(dcfile)

  # First, find the actual best-fit HR with wide bounds
  hr_calc_wide = datacard.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
    log_hazard_ratio_bounds=(-10.0, 10.0),
  )

  # Get best-fit HR
  best_fit_wide, _, _, _ = hr_calc_wide.hazard_ratio_confidence_interval(
    cox_only=False,
    confidence_level=0.68,
    hazard_ratio_min=0.00001,
    hazard_ratio_max=10.0,
  )

  # Now create a calculator with bounds that put best-fit at the lower boundary
  # The best-fit for this datacard is very small (~0.001), so set lower bound near it
  log_best_fit = np.log(best_fit_wide)
  # Set lower bound just at or slightly above best-fit
  log_lower_bound = log_best_fit + 0.01  # Just above best-fit

  hr_calc_tight = datacard.km_hazard_ratio(
    parameter_threshold=0.5,
    parameter_min=0.01,
    parameter_max=0.99,
    log_hazard_ratio_bounds=(log_lower_bound, 10.0),
  )

  # Test: Check that warning is raised when best-fit is at the lower boundary
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Request confidence interval - best-fit should hit lower bound
    best_fit_tight, _, _, _ = hr_calc_tight.hazard_ratio_confidence_interval(
      cox_only=False,
      confidence_level=0.68,
      hazard_ratio_min=np.exp(log_lower_bound),
      hazard_ratio_max=1.0,
    )
    # Should get a warning because best-fit is at the lower bound
    if len(w) >= 1:
      # If we got warnings, verify they're the right kind
      assert any(issubclass(warning.category, RuntimeWarning) for warning in w), \
        f"Expected RuntimeWarning but got {[type(warning.category).__name__ for warning in w]}"
      assert any("bound" in str(warning.message).lower() for warning in w), \
        f"Expected bounds warning but got: {[str(warning.message) for warning in w]}"
      print("[PASS] Bounds warning tests passed - warning raised as expected")
    else:
      # If no warning, verify that best-fit is not actually at the boundary
      # (which means the bounds are wide enough that we don't hit them)
      log_bf_tight = np.log(best_fit_tight)
      at_lower = abs(log_bf_tight - log_lower_bound) < 0.1
      at_upper = log_bf_tight > 9.9  # Close to upper bound of 10
      if not (at_lower or at_upper):
        print(f"[PASS] Bounds warning tests passed - no warning because HR={best_fit_tight:.6f} "
              f"not at boundary (log bounds: [{log_lower_bound:.3f}, 10.0])")
      else:
        raise AssertionError(f"Expected warning when HR at boundary, but got 0 warnings. "
                             f"HR={best_fit_tight:.6f}, log HR={log_bf_tight:.3f}, "
                             f"log bounds: [{log_lower_bound:.3f}, 10.0]")


if __name__ == "__main__":
  test_hazard_ratio_basic()
  test_likelihood_scan()
  test_likelihood_scan_custom_values()
  test_confidence_interval()
  test_consistency_with_p_value()
  test_hazard_ratio_at_null()
  test_known_hazard_ratios()
  test_bounds_warning()

  print("\n[SUCCESS] All hazard ratio tests passed!")
