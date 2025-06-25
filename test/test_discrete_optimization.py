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
  def _run_test(
    self,
    values: np.ndarray,
    func: typing.Callable[[float], float],
    expected_min_range: tuple[float, float],
    tol: float = 1e-8,
  ):
    """
    Run a test for minimize_discrete_single_minimum with given values and function.
    Checks if the minimum is within the expected range
    and if the function value at the minimum is correct.
    """
    x_min, y_min = minimize_discrete_single_minimum(func, values)
    y_expected = func(x_min)
    self.assertTrue(np.allclose(y_min, y_expected, atol=tol))
    self.assertGreaterEqual(x_min, expected_min_range[0] - tol)
    self.assertLessEqual(x_min, expected_min_range[1] + tol)

  def test_long_plateau_middle(self):
    """
    Test a long plateau minimum in the middle of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if 4 <= x <= 6 else 1
    self._run_test(values, f, (4.0, 6.0))

  def test_long_plateau_at_start(self):
    """
    Test a long plateau minimum at the start of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if x <= 1 else 1
    self._run_test(values, f, (0.0, 1.0))

  def test_long_plateau_at_end(self):
    """
    Test a long plateau minimum at the end of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if x >= 9 else 1
    self._run_test(values, f, (9.0, 10.0))

  def test_single_point_minimum_middle(self):
    """
    Test a single point minimum in the middle of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if np.isclose(x, 5.0) else 1
    self._run_test(values, f, (5.0, 5.0))

  def test_single_point_minimum_start(self):
    """
    Test a single point minimum at the start of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if np.isclose(x, 0.0) else 1
    self._run_test(values, f, (0.0, 0.0))

  def test_single_point_minimum_end(self):
    """
    Test a single point minimum at the end of the range.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      return 0 if np.isclose(x, 10.0) else 1
    self._run_test(values, f, (10.0, 10.0))

  def test_all_values_equal(self):
    """
    Test when all function values are equal.
    """
    values = np.linspace(0, 10, 10001)
    def f(x): #pylint: disable=unused-argument
      return 42
    x_min, y_min = minimize_discrete_single_minimum(f, values)
    self.assertIn(x_min, values)
    self.assertEqual(y_min, 42)

  def test_noisy_plateau_tiny_variation(self):
    """
    Test a plateau with tiny variations around a point.
    """
    values = np.linspace(0, 10, 10001)
    def f(x):
      if 4 <= x <= 6:
        return 0.001 * abs(x - 5)
      return 1
    # Minimum is at x=5.0 (center of plateau), but any near-zero point is acceptable
    self._run_test(values, f, (5, 5))

if __name__ == "__main__":
  unittest.main()
