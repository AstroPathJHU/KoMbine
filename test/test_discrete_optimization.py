"""
Tests for the discrete optimization module
"""

import typing
import unittest

import numpy as np

from roc_picker.discrete_optimization import (
  minimize_discrete_single_minimum,
  binary_search_sign_change,
)

class TestMinimizeDiscreteSingleMinimum(unittest.TestCase):
  """
    Tests for minimize_discrete_single_minimum function.
  """
  def setUp(self):
    # Define a set of tolerance pairs to test
    self.tolerances = [
        (1e-8, 0),          # Default-like (strict absolute, no relative)
        (1e-6, 1e-5),       # Moderate absolute and relative
        (1e-3, 1e-2),       # Loose absolute and relative
        (1e-10, 1e-10),     # Very strict absolute, some relative
        (0.1, 0.1),         # Very loose absolute and relative
    ]

  def _run_test(  # pylint: disable=too-many-arguments
    self,
    values: np.ndarray,
    func: typing.Callable[[float], float],
    expected_min_range: tuple[float, float],
    *,
    atol: float = 1e-8,
    rtol: float = 0,
  ):
    """
    Run a test for minimize_discrete_single_minimum with given values and function.
    Checks if the minimum is within the expected range
    and if the function value at the minimum is correct.
    """
    x_min, y_min = minimize_discrete_single_minimum(func, values, atol=atol, rtol=rtol)
    y_expected = func(x_min)

    # Assert that the found minimum value is close to the expected value at x_min
    self.assertTrue(np.isclose(y_min, y_expected, atol=atol, rtol=rtol),
                    f"Found y_min ({y_min}) not close to func(x_min) ({y_expected}) "
                    f"for atol={atol}, rtol={rtol}")

    # Assert that x_min is within the expected range, considering atol for boundary fuzziness
    self.assertGreaterEqual(x_min, expected_min_range[0] - atol,
                            f"x_min ({x_min}) too low for expected range {expected_min_range} "
                            f"and atol={atol}, rtol={rtol}")
    self.assertLessEqual(x_min, expected_min_range[1] + atol,
                         f"x_min ({x_min}) too high for expected range {expected_min_range} "
                         f"and atol={atol}, rtol={rtol}")

    # Additionally, verify that the found y_min is indeed the global minimum
    # within the given tolerances across all possible_values.
    # This is a stronger check for robustness.
    min_val_in_possible_values = min(func(val) for val in values)

    self.assertTrue(
      np.isclose(y_min, min_val_in_possible_values, atol=atol, rtol=rtol),
      f"Found y_min ({y_min}) is not globally minimal within tolerances "
      f"(true min: {min_val_in_possible_values}) for atol={atol}, rtol={rtol}"
    )


  def _run_test_with_tolerances(
    self,
    test_case_name: str, # Added for better error messages
    func_factory: typing.Callable[[float, float], typing.Callable[[float], float]],
    expected_min_range_factory: typing.Callable[[float, float], tuple[float, float]],
  ):
    """
    Helper to run tests across various tolerance settings.
    """
    for atol, rtol in self.tolerances:
      with self.subTest(msg=f"{test_case_name} with atol={atol}, rtol={rtol}"):
        values = np.linspace(0, 10, 10001)
        func = func_factory(atol, rtol)
        expected_min_range = expected_min_range_factory(atol, rtol)
        self._run_test(values, func, expected_min_range, atol=atol, rtol=rtol)

  # Existing tests adapted to use _run_test with default tolerances
  def test_long_plateau_middle(self):
    """
    Tests a case where the minimum is a long plateau in the middle of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if 4 <= x <= 6 else 1
    self._run_test(values, f, (4.0, 6.0))

  def test_long_plateau_at_start(self):
    """
    Tests a case where the minimum is a long plateau at the start of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if x <= 1 else 1
    self._run_test(values, f, (0.0, 1.0))

  def test_long_plateau_at_end(self):
    """
    Tests a case where the minimum is a long plateau at the end of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if x >= 9 else 1
    self._run_test(values, f, (9.0, 10.0))

  def test_single_point_minimum_middle(self):
    """
    Tests a case where the minimum is a single point in the middle of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if np.isclose(x, 5.0) else 1
    self._run_test(values, f, (5.0, 5.0))

  def test_single_point_minimum_start(self):
    """
    Tests a case where the minimum is a single point at the start of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if np.isclose(x, 0.0) else 1
    self._run_test(values, f, (0.0, 0.0))

  def test_single_point_minimum_end(self):
    """
    Tests a case where the minimum is a single point at the end of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if np.isclose(x, 10.0) else 1
    self._run_test(values, f, (10.0, 10.0))

  def test_all_values_equal(self):
    """
    Tests a case where all function values are the same.
    """
    values = np.linspace(0, 10, 10001)
    def f(x): #pylint: disable=unused-argument
      return 42
    # For this specific case, the range check might be less meaningful,
    # but the value check is key.
    x_min, y_min = minimize_discrete_single_minimum(f, values)
    self.assertIn(x_min, values)
    self.assertEqual(y_min, 42) # Exact equality is fine here

  def test_noisy_plateau_tiny_variation(self):
    """
    Tests a plateau with very small variations around the minimum.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      if 4 <= x <= 6:
        return 0.001 * abs(x - 5)
      return 1
    # Minimum is at x=5.0 (center of plateau), but any near-zero point is acceptable
    self._run_test(values, f, (5, 5)) # Using default atol/rtol for this existing test


  # --- New Robust Test Cases with Varying Tolerances ---

  def test_robust_plateau_with_slight_slope_left(self):
    """
    Tests a plateau where the true minimum is at the left end,
    and values slightly to the right are within tolerance.
    """
    def func_factory(atol, rtol): # pylint: disable=unused-argument
      def f(x):
        if 0 <= x <= 2:
          return 0.0
        if 2 < x <= 4:
          # Value slightly above 0, but within atol.
          # This should be considered equal to 0.0 within tolerance.
          return 0.5 * atol
        return 1.0
      return f
    def expected_range_factory(atol, rtol): # pylint: disable=unused-argument
      # The algorithm should identify the region [0.0, 4.0] as the effective minimum plateau
      # because values in (2, 4] are within tolerance of 0.0.
      return (0.0, 4.0)
    self._run_test_with_tolerances(
        "test_robust_plateau_with_slight_slope_left",
        func_factory,
        expected_range_factory
    )

  def test_robust_plateau_with_slight_slope_right(self):
    """
    Tests a plateau where the true minimum is at the right end,
    and values slightly to the left are within tolerance.
    """
    def func_factory(atol, rtol): # pylint: disable=unused-argument
      def f(x):
        if 8 <= x <= 10:
          return 0.0
        if 6 < x < 8:
          # Value slightly above 0, but within atol.
          return 0.5 * atol
        return 1.0
      return f
    def expected_range_factory(atol, rtol): # pylint: disable=unused-argument
      # The algorithm should identify the region [6.0, 10.0] as the effective minimum plateau.
      return (6.0, 10.0)
    self._run_test_with_tolerances(
        "test_robust_plateau_with_slight_slope_right",
        func_factory,
        expected_range_factory
    )

  def test_robust_multi_plateau_wavy_function(self):
    """
    Tests a function with a central, lowest plateau and higher side regions,
    all with sine wave noise, to ensure it finds the correct minimum region.
    The central plateau is off-center.
    """
    def func_factory(atol, rtol): # pylint: disable=unused-argument
      def f(x):
        noise = np.sin(x * 10) * 0.1 * atol # Wavy noise, scaled by atol

        if 4 <= x <= 6: # Central plateau: true minimum
          return 0.0 + noise
        if x < 4:
          # Linearly increasing from 0.0 at x=4, plus noise
          # Ensure it's always clearly above the central plateau.
          return (4 - x) * (2.0 * atol) + noise
        if x > 6:
          # Linearly increasing from 0.0 at x=6, plus noise
          return (x - 6) * (2.0 * atol) + noise
        assert False, "Unexpected x value in function"
      return f

    def expected_range_factory(atol, rtol): # pylint: disable=unused-argument
      # The algorithm should find any point within the central plateau [4.0, 6.0]
      # because its values are the lowest (0.0 + noise within atol).
      # The side regions are explicitly set to be higher.
      return (4.0, 6.0)

    self._run_test_with_tolerances(
        "test_robust_multi_plateau_wavy_function",
        func_factory,
        expected_range_factory
    )

  def test_robust_wide_plateau_with_internal_variations(self):
    """
    Tests a wide plateau where some points are the exact minimum
    and others are within tolerance of that minimum.
    Ensures it finds a point within the entire effective minimum plateau.
    """
    def func_factory(atol, rtol): # pylint: disable=unused-argument
      def f(x):
        if 3 <= x <= 7:
          # Within the plateau, some points are exactly 0, others are slightly above
          if 4.5 <= x <= 5.5: # The "true" exact minimum points
            return 0.0
          return 0.5 * atol # Within tolerance of 0.0
        return 1.0
      return f
    def expected_range_factory(atol, rtol): # pylint: disable=unused-argument
      # The algorithm should find any point within the effective minimum plateau.
      # The effective plateau is [3.0, 7.0] because all values in this range
      # are considered equal to 0.0 within the given atol.
      return (3.0, 7.0)
    self._run_test_with_tolerances(
        "test_robust_wide_plateau_with_internal_variations",
        func_factory,
        expected_range_factory
    )


class TestBinarySearchSignChange(unittest.TestCase):
  """
  Tests for binary_search_sign_change function.
  """

  def test_simple_sign_change_positive_to_negative(self):
    """
    Test a simple function that goes from positive to negative.
    """
    probs = np.linspace(0, 10, 11)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def func(x):
      return 5 - x  # Positive for x < 5, zero at x=5, negative for x > 5

    # Search the entire range where sign change occurs
    result = binary_search_sign_change(func, probs, 0, 10)

    # Should find the first x where func(x) <= 0, which is x=5 or higher
    self.assertGreaterEqual(result, 5.0)
    self.assertLessEqual(func(result), 0.0)

  def test_simple_sign_change_negative_to_positive(self):
    """
    Test a simple function that goes from negative to positive.
    """
    probs = np.linspace(0, 10, 21)  # More resolution

    def func(x):
      return x - 3  # Negative for x < 3, zero at x=3, positive for x > 3

    # Search around the sign change
    lo_idx = 0  # x = 0
    hi_idx = 20  # x = 10
    result = binary_search_sign_change(func, probs, lo_idx, hi_idx)

    # Should find the first x where func(x) <= 0, which is x=3
    self.assertGreaterEqual(result, 3.0)
    self.assertLessEqual(func(result), 0.0)

  def test_exact_zero_at_endpoint(self):
    """
    Test case where the function is exactly zero at one of the endpoints.
    """
    probs = np.array([0, 1, 2, 3, 4, 5])

    def func(x):
      if x <= 3:
        return 1  # Positive
      return -1  # Negative

    result = binary_search_sign_change(func, probs, 0, 5)

    # Should return a value where func <= 0 (which is x >= 4)
    self.assertGreaterEqual(result, 4.0)
    self.assertLessEqual(func(result), 0.0)

  def test_with_mip_tolerances(self):
    """
    Test that MIPGap and MIPGapAbs parameters work correctly.
    """
    # Create a more refined array
    probs = np.linspace(0, 1, 1001)  # Very fine resolution

    def func(x):
      return x - 0.5  # Sign change at x=0.5

    # Test with tight tolerances
    result_tight = binary_search_sign_change(
      func, probs, 0, 1000,
      MIPGap=1e-8, MIPGapAbs=1e-8
    )

    # Test with loose tolerances
    result_loose = binary_search_sign_change(
      func, probs, 0, 1000,
      MIPGap=1e-2, MIPGapAbs=1e-2
    )

    # Both should return a value where func <= 0
    self.assertLessEqual(func(result_tight), 0.0)
    self.assertLessEqual(func(result_loose), 0.0)

    # Both should be close to 0.5, but loose tolerance might terminate earlier
    self.assertAlmostEqual(result_tight, 0.5, places=2)
    self.assertAlmostEqual(result_loose, 0.5, places=1)

  def test_no_sign_change_raises_error(self):
    """
    Test that the function raises an error when there's no sign change.
    """
    probs = np.array([0, 1, 2, 3, 4, 5])

    def func_all_positive(x):
      return x + 1  # Always positive

    def func_all_negative(x):
      return -(x + 1)  # Always negative

    # Both should raise ValueError
    with self.assertRaises(ValueError):
      binary_search_sign_change(func_all_positive, probs, 0, 5)

    with self.assertRaises(ValueError):
      binary_search_sign_change(func_all_negative, probs, 0, 5)

  def test_single_interval_sign_change(self):
    """
    Test when the sign change is in a single interval (adjacent indices).
    """
    probs = np.array([0, 1, 2, 3, 4])

    def func(x):
      return 1.5 - x  # Sign change between x=1 and x=2

    result = binary_search_sign_change(func, probs, 0, 4)

    # Should return a value where func <= 0 (x >= 2)
    self.assertGreaterEqual(result, 2.0)
    self.assertLessEqual(func(result), 0.0)

  def test_function_with_cached_calls(self):
    """
    Test that the function works correctly with cached function calls.
    """
    call_count = [0]  # Use list to allow modification in nested function
    probs = np.linspace(0, 10, 21)

    def expensive_func(x):
      call_count[0] += 1
      return x - 5  # Sign change at x=5

    result = binary_search_sign_change(expensive_func, probs, 0, 20)

    # Should find a value where func <= 0
    self.assertLessEqual(expensive_func(result), 0.0)
    self.assertAlmostEqual(result, 5.0, places=2)

    # Should have made fewer calls than the total number of points
    # (exact number depends on the bisection algorithm)
    self.assertLess(call_count[0], len(probs))

  def test_early_termination_with_tolerance(self):
    """
    Test that the function terminates early when tolerance is reached.
    """
    # Create a case where probability differences become small
    probs = np.array([0.0, 0.1, 0.101, 0.102, 0.103, 0.2, 1.0])

    def func(x):
      return x - 0.1015  # Sign change around x=0.1015

    # With loose absolute tolerance, should terminate when prob differences are small
    result = binary_search_sign_change(
      func, probs, 0, 6,
      MIPGapAbs=0.01,  # Large tolerance
      MIPGap=0.1
    )

    # Should terminate early and return a reasonable result near the sign change
    self.assertGreaterEqual(result, 0.1)
    self.assertLessEqual(result, 0.2)


if __name__ == "__main__":
  unittest.main()
