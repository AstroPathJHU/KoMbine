"""
Tests for the discrete optimization module
"""

import typing
import unittest

import numpy as np

from roc_picker.discrete_optimization import minimize_discrete_single_minimum

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

  def test_robust_wavy_function_around_minimum(self):
    """
    Tests a function with small wiggles around the global minimum,
    ensuring it finds the true minimum despite fluctuations within tolerance.
    """
    def func_factory(atol, rtol): # pylint: disable=unused-argument
      def f(x):
        # Global minimum at x=5, value 0.
        # Other points are slightly above 0, but within tolerance.
        if np.isclose(x, 5, atol=1e-9): # Ensure exact global min at 5.0
          return 0.0
        # Create small wiggles that are within the tolerance of the global minimum
        return 0.5 * atol + np.sin(x * 10) * 0.1 * atol
      return f
    def expected_range_factory(atol, rtol): # pylint: disable=unused-argument
      # The expected range should effectively be the entire domain if all values
      # are within tolerance of the lowest value.
      # However, since we're looking for the *single* minimum, we expect it to
      # converge to the region containing the exact 0.0.
      # Let's define the expected range as where the function is exactly 0.
      return (5.0, 5.0) # The true exact minimum point
    self._run_test_with_tolerances(
        "test_robust_wavy_function_around_minimum",
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

if __name__ == "__main__":
  unittest.main()
